import os
import re
import json
from groq import Groq

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_speaker_names(full_text: str):
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
        json_match = re.search(r'\{.*\}', name_mapping_str, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"Failed to extract names: {e}")
    return {}

def get_summaries(full_text: str):
    # Base summary
    prompt = f"Please summarize the following meeting transcript:\n\n{full_text}"
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes meeting transcripts clearly and concisely. Do NOT use markdown. Output plain text only without asterisks or hashes."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
    )
    summary = chat_completion.choices[0].message.content.replace("*", "").replace("#", "")

    # Hinglish summary
    prompt_hi = f"Please summarize the following meeting transcript in Hinglish (a natural mix of Hindi and English, written in the English alphabet):\n\n{full_text}"
    chat_completion_hi = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes meeting transcripts clearly and concisely in Hinglish (Hindi + English). Do NOT use markdown. Output plain text only without asterisks or hashes."},
            {"role": "user", "content": prompt_hi}
        ],
        model="llama-3.3-70b-versatile",
    )
    summary_hi = chat_completion_hi.choices[0].message.content.replace("*", "").replace("#", "")

    # Speaker summary
    prompt_spk = f"Please provide a summary of the meeting, grouped by each speaker. Highlight their key points, action items, and general contributions. Format the output with clear headings for each speaker.\n\nTranscript:\n{full_text}"
    chat_completion_spk = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes meeting transcripts by grouping the summary by each individual speaker. Do NOT use markdown. Output plain text only without asterisks or hashes."},
            {"role": "user", "content": prompt_spk}
        ],
        model="llama-3.3-70b-versatile",
    )
    summary_speakers = chat_completion_spk.choices[0].message.content.replace("*", "").replace("#", "")

    # Speaker summary (Hinglish)
    prompt_spk_hi = f"Please provide a summary of the meeting, grouped by each speaker, in Hinglish (a natural mix of Hindi and English, written in the English alphabet). Highlight their key points, action items, and general contributions. Format the output with clear headings for each speaker.\n\nTranscript:\n{full_text}"
    chat_completion_spk_hi = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes meeting transcripts by grouping the summary by each individual speaker in Hinglish. Do NOT use markdown. Output plain text only without asterisks or hashes."},
            {"role": "user", "content": prompt_spk_hi}
        ],
        model="llama-3.3-70b-versatile",
    )
    summary_speakers_hi = chat_completion_spk_hi.choices[0].message.content.replace("*", "").replace("#", "")

    return summary, summary_hi, summary_speakers, summary_speakers_hi

def get_meeting_intelligence(full_text: str):
    prompt_intelligence = f"""Analyze this meeting transcript and return a structured JSON object with exactly these three keys:
1. "missed_signals": A list of strings. Identify: Unanswered questions, Repeated concerns not resolved, Vague commitments (e.g., "we'll see", "soon"), Conflicts or disagreements without resolution, Decisions without clear ownership.
2. "health": An object with "score" (number from 1.0 to 10.0), "strengths" (list of strings), and "weaknesses" (list of strings).
3. "action_items": A list of objects. Extract: "task" (Task description), "owner" (if mentioned, else null), "deadline" (if mentioned, else null), "risk_level" (Low/Medium/High), and "risk_reason" (Why risk was assigned).

Return ONLY valid JSON. Do not use markdown blocks like `json`.

Transcript:
{full_text}
"""
    try:
        chat_completion_intel = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a meeting intelligence JSON generator. Exclusively output valid JSON without formatting."},
                {"role": "user", "content": prompt_intelligence}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2
        )
        intel_raw = chat_completion_intel.choices[0].message.content
        json_match = re.search(r'\{.*\}', intel_raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return json.loads(intel_raw)
    except Exception as e:
        print(f"Failed to extract intelligence: {e}")
        return None

def generate_chat_response(question: str, context: str):
    prompt_chat = f"""You are a highly capable AI assistant that answers questions based on the provided meeting context. 
If the answer is in the context, provide a clear, helpful response. You are also encouraged to provide your own AI opinions, insights, and objective analysis based on the context when asked.
If the question is completely unrelated to the meeting, politely decline to answer or state that it wasn't discussed.

Meeting Context (Transcript/Summary):
{context}

User Question: {question}
"""
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful, intelligent AI assistant. Answer based on the provided meeting context, and feel free to offer your own opinions and analysis. Keep answers concise but friendly."},
                {"role": "user", "content": prompt_chat}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.5
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Chat failed: {e}")
        return None
