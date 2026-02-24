# Hack-A-Tron Backend

This is the Python-based backend service for the Hack-a-tron project. It safely processes video uploads from your Next.js frontend, extracts the audio efficiently using **MoviePy**, generates speaker-diarized transcripts using the **AssemblyAI** SDK, and generates ultra-fast meeting summaries using the **Groq** API.

## Features
- ✅ Accepts `.mp4` video files from the frontend
- ✅ Extracts audio from video files on the fly
- ✅ Generates fully structured speaker-by-speaker transcripts
- ✅ Produces a fast meeting summary using Groq's high-intelligence `llama-3.3-70b-versatile` model
- ✅ Entirely built on FastAPI for lightning-fast concurrent operations

## Prerequisites
- **Python 3.10+** (You can download it from [python.org](https://www.python.org/downloads/))
- Groq API Key
- AssemblyAI API Key

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Rahul-Patel0907/Hack-A-Tron-Backend.git
cd Hack-A-Tron-Backend
```

### 2. Create and Activate a Virtual Environment
We strongly recommend creating a virtual Python environment to ensure dependencies do not conflict.

*On Windows:*
```bash
python -m venv venv
.\venv\Scripts\activate
```

*On Mac/Linux:*
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the Dependencies
Once your terminal shows you are inside the `(venv)` environment, install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create a file exactly named `.env` in the root of the `Hack-A-Tron-Backend` folder. Inside that secure file, paste the following keys and replace the dummy strings with your active SDK keys:

```env
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Start the Server
Run the FastAPI development server utilizing Uvicorn:

```bash
uvicorn main:app --reload
```

Your server will successfully boot up! It will now be actively listening for incoming HTTP requests from the frontend at: `http://localhost:8000`.

## API Documentation
Because this API is built on FastAPI, the server automatically generates a swagger interactive sandbox.

Once the uvicorn server is running locally, you can view the fully documented endpoints at:
`http://localhost:8000/docs`
