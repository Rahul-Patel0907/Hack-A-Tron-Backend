import os
import shutil
import json
import re
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import assemblyai as aai
from dotenv import load_dotenv

load_dotenv()

from services.ai_service import get_speaker_names, get_summaries, get_meeting_intelligence, generate_chat_response

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

@app.post("/api/process-video")
async def process_video(file: UploadFile = File(...)):
    # 1. Temporarily save the uploaded video to disk
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. Transcription & Diarization using AssemblyAI (Direct Video Processing)
        transcriber = aai.Transcriber()
        
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            auto_chapters=True,
            language_detection=True,
            speech_models=["universal-3-pro", "universal-2"],
            word_boost=["Vrize", "VRIZE"]
        )
        transcript = transcriber.transcribe(video_path, config=config)

        if transcript.status == aai.TranscriptStatus.error:
            return {"error": transcript.error}

        speaker_wise_transcript = []
        full_text = ""
        for utterance in transcript.utterances:
            # Hardcoded fix to ensure Vrize is correct
            corrected_text = utterance.text.replace("Vrise", "Vrize").replace("vrise", "vrize").replace("VRISE", "VRIZE")
            
            speaker_wise_transcript.append({
                "speaker": f"Speaker {utterance.speaker}",
                "text": corrected_text,
                "start": utterance.start,
                "end": utterance.end
            })
            full_text += f"Speaker {utterance.speaker}: {corrected_text}\n"

        # 3.5 Identify speaker names using Groq
        name_mapping = get_speaker_names(full_text)

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
        summary, summary_hi, summary_speakers, summary_speakers_hi = get_summaries(full_text)

        # 4e. Advanced Meeting Intelligence (Hackathon Special)
        meeting_intelligence = get_meeting_intelligence(full_text)

        if meeting_intelligence is not None:
            # Compute Speaker Analytics
            speaker_stats = {}
            total_spoken_time = 0
            interruptions = []
            previous_end = 0
            previous_speaker = None
            
            for item in speaker_wise_transcript:
                spk = item["speaker"]
                start_ms = item["start"]
                end_ms = item["end"]
                duration = end_ms - start_ms
                
                if spk not in speaker_stats:
                    speaker_stats[spk] = {"total_time": 0, "longest_monologue": 0}
                
                speaker_stats[spk]["total_time"] += duration
                total_spoken_time += duration
                
                if duration > speaker_stats[spk]["longest_monologue"]:
                    speaker_stats[spk]["longest_monologue"] = duration
                    
                if previous_speaker and previous_speaker != spk and start_ms < previous_end:
                    interruptions.append({
                        "interrupter": spk,
                        "interrupted": previous_speaker,
                        "time": start_ms
                    })
                    
                previous_end = max(previous_end, end_ms)
                previous_speaker = spk
                
            speaker_metrics = {
                "total_spoken_time": total_spoken_time,
                "speakers": [],
                "interruptions": len(interruptions)
            }
            for spk, stats in speaker_stats.items():
                pct = round((stats["total_time"] / total_spoken_time) * 100) if total_spoken_time > 0 else 0
                speaker_metrics["speakers"].append({
                    "name": spk,
                    "total_time": stats["total_time"],
                    "percentage": pct,
                    "longest_monologue": stats["longest_monologue"]
                })
                
            # Sort speakers by percentage descending
            speaker_metrics["speakers"] = sorted(speaker_metrics["speakers"], key=lambda x: x["percentage"], reverse=True)
            meeting_intelligence["speaker_metrics"] = speaker_metrics

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
        if os.path.exists(video_path):
            os.remove(video_path)

        # 5. Return Response
        return {
            "transcript": speaker_wise_transcript,
            "chapters": chapters_list,
            "summary": summary,
            "summary_hi": summary_hi,
            "summary_speakers": summary_speakers,
            "summary_speakers_hi": summary_speakers_hi,
            "meeting_intelligence": meeting_intelligence
        }

    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "Welcome to Hack-a-tron Backend API!"}

class ChatRequest(BaseModel):
    question: str
    context: str

@app.post("/api/chat")
async def chat_with_meeting(request: ChatRequest):
    try:
        answer = generate_chat_response(request.question, request.context)
        if answer is None:
             return {"error": "Failed to generate response."}
        return {"answer": answer}
    except Exception as e:
        print(f"Chat failed: {e}")
        return {"error": "Failed to generate response."}
