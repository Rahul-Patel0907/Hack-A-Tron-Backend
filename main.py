import os
import shutil
import json
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from moviepy import VideoFileClip
import assemblyai as aai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.post("/api/process-video")
async def process_video(file: UploadFile = File(...)):
    # 1. Temporarily save the uploaded video to disk
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. Audio Extraction using MoviePy
        audio_path = f"audio_{file.filename}.mp3"
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()

        # 3. Transcription & Diarization using AssemblyAI
        transcriber = aai.Transcriber()
        
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            auto_chapters=True,
            speech_models=[aai.SpeechModel.universal]
        )
        transcript = transcriber.transcribe(audio_path, config=config)

        if transcript.status == aai.TranscriptStatus.error:
            return {"error": transcript.error}

        speaker_wise_transcript = []
        full_text = ""
        for utterance in transcript.utterances:
            speaker_wise_transcript.append({
                "speaker": f"Speaker {utterance.speaker}",
                "text": utterance.text,
                "start": utterance.start,
                "end": utterance.end
            })
            full_text += f"Speaker {utterance.speaker}: {utterance.text}\n"

        # 3.5 Identify speaker names using Groq
        prompt_names = f"Analyze the following transcript and identify the real names of the speakers (e.g., 'Speaker A', 'Speaker B') if they mention them. Return ONLY a valid JSON object mapping the speaker labels to their inferred real names. If a name cannot be inferred, map it to the original label. Do not output any other text or markdown.\n\nTranscript:\n{full_text}"
        
        try:
            name_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a JSON generator. You exclusively output valid JSON mappings of speaker labels to real names."
                    },
                    {
                        "role": "user",
                        "content": prompt_names
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1
            )
            name_mapping_str = name_completion.choices[0].message.content
            # Safely extract json if markdown is present
            json_match = re.search(r'\{.*\}', name_mapping_str, re.DOTALL)
            if json_match:
                name_mapping = json.loads(json_match.group())
            else:
                name_mapping = {}
        except Exception as e:
            print(f"Failed to extract names: {e}")
            name_mapping = {}

        # Update full_text and speaker_wise_transcript with real names
        if name_mapping:
            for item in speaker_wise_transcript:
                original_speaker = item["speaker"]
                if original_speaker in name_mapping:
                    item["speaker"] = name_mapping[original_speaker]
            
            # Rebuild full_text to reflect new names for the summary AI chunks
            full_text = ""
        
            for item in speaker_wise_transcript:
                full_text += f"{item['speaker']}: {item['text']}\n"

        # 4. Summarization using Groq
        prompt = f"Please summarize the following meeting transcript:\n\n{full_text}"
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes meeting transcripts clearly and concisely. Do NOT use markdown. Output plain text only without asterisks or hashes.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        
        summary = chat_completion.choices[0].message.content.replace("*", "").replace("#", "")

        # 4b. Summarization using Groq (Hinglish)
        prompt_hinglish = f"Please summarize the following meeting transcript in Hinglish (a natural mix of Hindi and English, written in the English alphabet):\n\n{full_text}"
        
        chat_completion_hinglish = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes meeting transcripts clearly and concisely in Hinglish (Hindi + English). Do NOT use markdown. Output plain text only without asterisks or hashes.",
                },
                {
                    "role": "user",
                    "content": prompt_hinglish,
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        
        summary_hi = chat_completion_hinglish.choices[0].message.content.replace("*", "").replace("#", "")

        # 4c. Summarization using Groq (Person-wise)
        prompt_speakers = f"Please provide a summary of the meeting, grouped by each speaker. Highlight their key points, action items, and general contributions. Format the output with clear headings for each speaker.\n\nTranscript:\n{full_text}"
        
        chat_completion_speakers = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes meeting transcripts by grouping the summary by each individual speaker. Do NOT use markdown. Output plain text only without asterisks or hashes.",
                },
                {
                    "role": "user",
                    "content": prompt_speakers,
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        
        summary_speakers = chat_completion_speakers.choices[0].message.content.replace("*", "").replace("#", "")

        # 4d. Summarization using Groq (Person-wise, Hinglish)
        prompt_speakers_hi = f"Please provide a summary of the meeting, grouped by each speaker, in Hinglish (a natural mix of Hindi and English, written in the English alphabet). Highlight their key points, action items, and general contributions. Format the output with clear headings for each speaker.\n\nTranscript:\n{full_text}"
        
        chat_completion_speakers_hi = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes meeting transcripts by grouping the summary by each individual speaker in Hinglish. Do NOT use markdown. Output plain text only without asterisks or hashes.",
                },
                {
                    "role": "user",
                    "content": prompt_speakers_hi,
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        
        summary_speakers_hi = chat_completion_speakers_hi.choices[0].message.content.replace("*", "").replace("#", "")

        chapters_list = []
        if getattr(transcript, 'chapters', None):
            for chapter in transcript.chapters:
                chapters_list.append({
                    "headline": chapter.headline,
                    "summary": chapter.summary,
                    "start": chapter.start,
                    "end": chapter.end
                })

        # Cleanup temporary files
        os.remove(video_path)
        os.remove(audio_path)

        # 5. Return Response
        return {
            "transcript": speaker_wise_transcript,
            "chapters": chapters_list,
            "summary": summary,
            "summary_hi": summary_hi,
            "summary_speakers": summary_speakers,
            "summary_speakers_hi": summary_speakers_hi
        }

    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "Welcome to Hack-a-tron Backend API!"}
