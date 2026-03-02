import os
import shutil
import json
import re
import uuid
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import assemblyai as aai
from dotenv import load_dotenv

load_dotenv()

from services.ai_service import get_speaker_names, get_summaries, get_meeting_intelligence, generate_chat_response

app = FastAPI()

# Ensure uploads directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount the uploads directory to serve static video files
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

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
async def process_video(file: UploadFile = File(...), request: Request = None):
    # 1. Save uploaded video to persistent disk instead of temp
    file_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Generate the public URL
    base_url = str(request.base_url).rstrip("/")
    video_url = f"{base_url}/uploads/{file_id}_{file.filename}"

    try:
        # 2. Transcription & Diarization using AssemblyAI (Direct Video Processing)
        transcriber = aai.Transcriber()
        
        # 🚀 Added Custom Vocabulary Injection (Word Boost)
        custom_vocab = [
            "Vrize", "VRIZE", "Mac binding", "MAC binding", "Raipur"
        ]
        
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            auto_chapters=True,
            language_detection=True,
            speech_models=["universal-3-pro", "universal-2"],
            word_boost=custom_vocab
        )
        transcript = transcriber.transcribe(video_path, config=config)

        if transcript.status == aai.TranscriptStatus.error:
            return {"error": transcript.error}

        # 🚀 Added Post-Processing Spell Correction Map
        corrections = {
            r"\bVrise\b": "Vrize", r"\bvrise\b": "Vrize", r"\bVRISE\b": "VRIZE",
            r"\biso 2000 001\b": "ISO 2000 001", r"\bmac binding\b": "MAC binding",
            r"\brypur\b": "Raipur"
        }

        # First pass: Post-processing corrections
        raw_transcript = []
        for utterance in transcript.utterances:
            text = utterance.text
            for pattern, correction in corrections.items():
                text = re.sub(pattern, correction, text, flags=re.IGNORECASE)
                
            raw_transcript.append({
                "speaker": f"Speaker {utterance.speaker}",
                "text": text,
                "start": utterance.start,
                "end": utterance.end,
                "duration": utterance.end - utterance.start
            })

        # 🚀 Added Speaker Diarization Smoothing (Confidence/Heuristic Reassignment)
        # If a very short speech burst (<1500ms) occurs sandwhiched between the same speaker, assume diarization hallucination and correct it.
        if len(raw_transcript) >= 3:
            for i in range(1, len(raw_transcript) - 1):
                prev_spk = raw_transcript[i-1]["speaker"]
                next_spk = raw_transcript[i+1]["speaker"]
                if prev_spk == next_spk and raw_transcript[i]["duration"] < 1500:
                    raw_transcript[i]["speaker"] = prev_spk

        # 🚀 Merge consecutive same-speaker fragments into cohesive blocks
        speaker_wise_transcript = []
        full_text = ""
        for item in raw_transcript:
            if speaker_wise_transcript and speaker_wise_transcript[-1]["speaker"] == item["speaker"]:
                # Merge into previous
                speaker_wise_transcript[-1]["text"] += " " + item["text"]
                speaker_wise_transcript[-1]["end"] = max(speaker_wise_transcript[-1]["end"], item["end"])
            else:
                # Add new segment
                del item["duration"] # Cleanup keys
                speaker_wise_transcript.append(item)

        # Build full text for subsequent AI steps
        for item in speaker_wise_transcript:
            full_text += f"{item['speaker']}: {item['text']}\n"

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

        # 5. Return Response
        return {
            "transcript": speaker_wise_transcript,
            "chapters": chapters_list,
            "summary": summary,
            "summary_hi": summary_hi,
            "summary_speakers": summary_speakers,
            "summary_speakers_hi": summary_speakers_hi,
            "meeting_intelligence": meeting_intelligence,
            "video_url": video_url
        }

    except Exception as e:
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
