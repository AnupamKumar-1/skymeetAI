# Meeting Transcription and Emotion Analysis Service

## Introduction

This is a Flask-based web service designed to process audio files from meetings, perform automatic speech recognition (ASR) using the Whisper model, classify emotions in transcribed text segments using a pre-trained Hugging Face transformer model, and generate consolidated transcripts in both text and JSON formats. The service supports multi-speaker scenarios by merging overlapping or consecutive segments from the same speaker.

Key features:
- Audio format conversion (to WAV) using FFmpeg.
- Transcription with Whisper's "small" model (English language assumed).
- Emotion classification (e.g., anger, joy, neutral) per text segment.
- Grouping of segments by speaker with emotion summaries.
- Automatic cleanup of temporary files after a delay.
- CORS-enabled for cross-origin requests.

The service is intended for development and production use (e.g., via Gunicorn), handling audio uploads securely and efficiently.

## Architecture

The application follows a microservice architecture built on Flask, with a focus on modular processing pipelines for audio handling, transcription, and analysis. Below is a high-level overview of the components and data flow.

### High-Level Components
1. **Web Server (Flask)**:
   - Handles HTTP requests, file uploads, and responses.
   - Endpoints: `/process_meeting` (core processing) and `/outputs/<filename>` (file downloads).
   - Uses CORS for frontend integration.

2. **Audio Processing Layer**:
   - Validates and saves uploaded audio files (supports formats: webm, wav, mp3, m4a, ogg, aac, mp4).
   - Converts non-WAV files to mono 16kHz WAV using FFmpeg for compatibility with Whisper.

3. **ASR (Automatic Speech Recognition) Module**:
   - Utilizes OpenAI's Whisper model ("small" variant) for transcription.
   - Processes audio into segments with timestamps, text, and filters out short/noisy segments based on thresholds (e.g., min duration 0.3s, min text length 3 chars).

4. **Emotion Analysis Module**:
   - Employs a Hugging Face pipeline (`j-hartmann/emotion-english-distilroberta-base`) for text-classification.
   - Classifies each transcribed segment into emotions (top-1 label, fallback to "neutral").
   - Integrated per-segment during transcription.

5. **Post-Processing Layer**:
   - Merges segments by speaker if they are consecutive (within 1s gap).
   - Aggregates texts and emotions per group.
   - Generates human-readable text transcripts and structured JSON outputs.
   - Computes emotion summaries (top emotion and breakdown) per speaker using `collections.Counter`.

6. **Cleanup Mechanism**:
   - Schedules file deletion (uploads and outputs) after a configurable delay (default: 120s) using `threading.Timer`.
   - Ensures temporary storage management.

7. **External Dependencies**:
   - FFmpeg (system-level, for audio conversion and optional volume checks).
   - Models loaded at startup for efficiency: Whisper and emotion pipeline.

### Data Flow
1. **Request Intake**: Client sends POST to `/process_meeting` with audio files, optional meeting code, and speaker map (JSON).
2. **File Handling**: Save uploads, convert to WAV if needed.
3. **Transcription**: Run Whisper on each WAV file to get segments.
4. **Emotion Enrichment**: For each segment, classify emotion and filter invalid ones.
5. **Merging & Grouping**: Combine results across files, sort by timestamp, group by speaker.
6. **Output Generation**: Create text transcript (with timestamps and summaries) and JSON (structured groups + summaries).
7. **Response**: Return JSON with transcript preview, filenames for download, and cleanup notice.
8. **Download**: Client can GET `/outputs/<filename>` for TXT/JSON files.
9. **Cleanup**: Timer deletes files after delay.

### Diagram (Text-Based Representation)
```
Client (Browser/App) --> [HTTP POST /process_meeting] --> Flask Server
                          |
                          v
                    Upload & Validate Files
                          |
                          v
                    Convert to WAV (FFmpeg)
                          |
                          v
                    Transcribe (Whisper Model)
                          |
                          v
                    Classify Emotions (HF Pipeline)
                          |
                          v
                    Merge & Group Segments
                          |
                          v
                    Generate TXT/JSON Outputs
                          |
                          v
                    Schedule Cleanup (Threading Timer)
                          |
                          v
                    Return JSON Response --> Client
                          |
                          v
Client --> [HTTP GET /outputs/<file>] --> Serve File --> Client
```

### Scalability Notes
- Models are loaded once at startup to reduce latency.
- Stateful processing is per-request; no persistent state across requests.
- For production, use Gunicorn for multi-worker handling.
- Potential bottlenecks: Whisper transcription (GPU acceleration recommended via Torch) and FFmpeg conversions.

## Installation

### Prerequisites
- Python 3.8+.
- FFmpeg installed on the system (e.g., `apt install ffmpeg` on Ubuntu).
- GPU (optional but recommended for faster Whisper inference via CUDA).

### Steps
1. Clone the repository or copy the code into a directory.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   Note: Whisper is installed from GitHub for the latest version.
3. Ensure folders exist: `uploads/` and `outputs/` (created automatically on startup).

### Running the Server
- Development: `python app.py` (runs on http://0.0.0.0:5001 with debug mode).
- Production: Use Gunicorn, e.g., `gunicorn -w 4 app:app -b 0.0.0.0:5001`.

## Usage

### Configuration Variables
These are hardcoded globals; modify in `app.py` as needed:
- `MIN_DURATION_SEC`: Minimum segment duration (0.30s).
- `MIN_TEXT_CHARS`: Minimum text length (3 chars).
- `NO_SPEECH_PROB_THRESH`: Whisper no-speech probability threshold (0.6, not actively used in code).
- `AVG_LOGPROB_THRESH`: Whisper average logprob threshold (-1.5, not actively used).
- `ENABLE_FFMPEG_ENERGY_CHECK`: Toggle mean volume check (False by default).
- `MIN_MEAN_VOLUME_DB`: Minimum mean volume (-45 dB).
- `UPLOAD_FOLDER`: "uploads".
- `OUTPUT_FOLDER`: "outputs".
- `ALLOWED_EXT`: Supported audio extensions.
- `CLEANUP_DELAY_SEC`: File cleanup delay (120s).

### Example Client Request (Using curl)
Upload two audio files with a speaker map:
```
curl -X POST http://localhost:5001/process_meeting \
  -F "audio_files=@speaker1.webm" \
  -F "audio_files=@speaker2.mp3" \
  -F "meeting_code=MEET123" \
  -F "speaker_map={\"speaker1\": \"Alice\", \"speaker2\": \"Bob\"}"
```

Response (JSON):
```
{
  "success": true,
  "transcript_text": "[0.00–5.00] Alice:\n    Hello, how are you?\n\n[5.10–10.00] Bob:\n    I'm good, thanks.\n\n=== Emotion Summary (by speaker) ===\nAlice: top=joy — joy (1)\nBob: top=neutral — neutral (1)",
  "txt_filename": "MEET123_abc123.txt",
  "json_filename": "MEET123_def456.json",
  "files_will_be_deleted_in_sec": 120
}
```

Download files:
```
curl http://localhost:5001/outputs/MEET123_abc123.txt -o transcript.txt
```

## API Endpoints

### POST /process_meeting
- **Description**: Upload audio files, process them, and generate transcripts.
- **Form Parameters**:
  - `audio_files` (multiple files): Audio files to process.
  - `meeting_code` (string, optional): Prefix for output filenames (default: "UNKNOWN").
  - `speaker_map` (JSON string, optional): Mapping of base filenames to speaker names (e.g., `{"file1": "Alice"}`).
- **Response**: JSON with success flag, transcript preview, filenames, and cleanup info.
- **Errors**: Returns 200 even on partial failures (logs errors); check console for issues.

### GET /outputs/<filename>
- **Description**: Download a generated TXT or JSON file.
- **Path Parameters**:
  - `filename`: Name of the file (from response).
- **Response**: File attachment.

## Dependencies
From `requirements.txt`:
- flask: Web framework.
- flask-cors: CORS support.
- werkzeug: Utilities (e.g., secure_filename).
- pydub: Audio manipulation (though not directly used in code; possibly for future).
- openai-whisper: ASR model (GitHub install).
- transformers: Hugging Face pipelines.
- torch: Backend for models.
- gunicorn: Production server.
