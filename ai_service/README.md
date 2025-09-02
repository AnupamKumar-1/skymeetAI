# ai_service — API Documentation

## Overview

`ai_service/app.py` is a small Flask-based microservice that accepts one or more audio files, runs automatic speech recognition (OpenAI Whisper) to produce a transcript, runs a Hugging Face text-classification pipeline to classify emotion for each transcript segment, groups nearby segments by speaker, and returns downloadable transcript and JSON output files.

This document explains how to run the service, its API, configuration options, and deployment notes.

---

# Quickstart

### Requirements
- Python 3.8+
- `ffmpeg` available on PATH (required for format conversions and energy checks)
- Enough memory/CPU (Whisper `small` model is loaded on startup; GPU recommended for speed but not required)

### Example `requirements.txt`
```
flask
flask-cors
werkzeug
pydub
transformers
torch
git+https://github.com/openai/whisper.git
gunicorn
```

Install with:
```bash
pip install -r requirements.txt
```

### Run locally (development)
```bash
python ai_service/app.py
```
This starts the server on `http://0.0.0.0:5001` (debug mode enabled in the file). For production, see the **Deployment** section.

---

# What the service does (high-level)
1. Accepts uploaded audio files via multipart/form-data.
2. For each file: converts to mono WAV @ 16kHz (via `ffmpeg`) unless the original file is usable.
3. Runs Whisper ASR (`whisper.load_model("small")`) to produce time-stamped segments.
4. Runs a Hugging Face `text-classification` pipeline (`j-hartmann/emotion-english-distilroberta-base`) on each segment's text to extract an emotion label.
5. Merges segments from all uploaded files, sorts by timestamp, and groups nearby segments by speaker (using `speaker_map` provided by the client, or falling back to filenames).
6. Produces two outputs placed in `outputs/`: a plain-text transcript (`.txt`) and a JSON summary (`.json`).
7. Schedules the created files (and temporary uploads) for deletion after a configurable cleanup delay (default 120 sec).

---

# Configuration & constants (in `app.py`)
- `MIN_DURATION_SEC = 0.30` — minimum segment duration considered.
- `MIN_TEXT_CHARS = 3` — minimum characters in a text segment to keep.
- `NO_SPEECH_PROB_THRESH = 0.6` — (unused in current code path but present for possible future checks).
- `AVG_LOGPROB_THRESH = -1.5` — (unused in current code path but kept for scoring thresholds).
- `ENABLE_FFMPEG_ENERGY_check = False` — when `True`, the app will attempt to measure mean volume of segments using `ffmpeg -af volumedetect` and can drop segments below `MIN_MEAN_VOLUME_DB`.
- `MIN_MEAN_VOLUME_DB = -45.0` — threshold used by the optional energy check.
- `UPLOAD_FOLDER = "uploads"` and `OUTPUT_FOLDER = "outputs"` — directories where files are saved.
- `ALLOWED_EXT` — allowed upload extensions: `{"webm", "wav", "mp3", "m4a", "ogg", "aac", "mp4"}`
- `CLEANUP_DELAY_SEC = 120` — seconds after which created files are removed by a background timer.

You can change these constants directly in `app.py` or expose them via environment variables and read them at startup if you prefer runtime configurability.

---

# API Endpoints

## `POST /process_meeting`
Process one or more audio files and produce a transcript + emotion summary.

### Form fields
- `audio_files` — one or more files. Use `multipart/form-data` with repeated `audio_files` fields.
- `meeting_code` (optional) — string used to prefix output filenames. Defaults to `UNKNOWN`.
- `speaker_map` (optional) — JSON string mapping uploaded file base names to speaker names. Example: `{"meeting1": "Alice", "meeting2": "Bob"}`

### Behavior
- Each uploaded file is saved to `uploads/` and then converted to WAV (mono, 16kHz) using `ffmpeg` into the same folder unless conversion fails (in which case the original file is used as fallback).
- Whisper transcribes the WAV file. For each speech segment a call to the HF emotion pipeline is made with the segment text.
- The service aggregates segments across files, sorts them by timestamp, groups adjacent segments from the same speaker (gap ≤ 1.0s), and produces:
  - a human-readable `.txt` transcript (placed in `outputs/`)
  - a structured `.json` file with `groups` and `speaker_summary`
- Both generated files are scheduled for deletion after `CLEANUP_DELAY_SEC`.

### Success response (HTTP 200)
```json
{
  "success": true,
  "transcript_text": "<human-readable transcript>",
  "txt_filename": "<meeting_code>_<uuid>.txt",
  "json_filename": "<meeting_code>_<uuid>.json",
  "files_will_be_deleted_in_sec": 120
}
```

### Error handling
- If `ffmpeg` conversion fails for a file, the service logs the failure and uses the original file as `wav_path` fallback.
- If Whisper fails to transcribe, an empty segment list is used (the app does not crash).
- If the emotion pipeline fails for a particular segment, the code uses `"neutral"` as a safe fallback.

---

## `GET /outputs/<filename>`
Download a previously generated output file from the `outputs/` directory. Example:
```
GET /outputs/TEAMSYNC_... .txt
```
Files will be removed automatically when the cleanup timer triggers, so download quickly.

---

# Examples

### cURL — single file
```bash
curl -X POST http://localhost:5001/process_meeting \
  -F "audio_files=@/path/to/meeting1.mp3" \
  -F "meeting_code=TEAM_SYNC_2025-09-03" \
  -F 'speaker_map={"meeting1":"Alice"}'
```

### cURL — multiple files and speaker map
```bash
curl -X POST http://localhost:5001/process_meeting \
  -F "audio_files=@/path/to/meeting1.mp3" \
  -F "audio_files=@/path/to/meeting2.wav" \
  -F "meeting_code=DAILY_STANDUP" \
  -F 'speaker_map={"meeting1":"Alice","meeting2":"Bob"}'
```

### Interpreting results
- `transcript_text` is a readable transcript with timestamps and per-speaker grouped blocks.
- The JSON file contains `groups` (with `start`, `end`, `speaker`, `texts`, `emotions`) and `speaker_summary` (top emotion and counts per speaker).

---

# Files & Cleanup
- Temporary uploads are saved under `uploads/` and converted WAVs are also placed there.
- Final outputs are written to `outputs/` with UUID-based filenames.
- All created files are scheduled for deletion after `CLEANUP_DELAY_SEC` using a `threading.Timer` background job. If you want permanent storage, copy or move the files after receiving the response and before the cleanup delay expires.

---

# Deployment Notes
- **Production server**: do not use `app.run(debug=True)`. Use `gunicorn` with multiple workers, e.g.:

```bash
gunicorn -w 2 -b 0.0.0.0:5001 ai_service.app:app
```

- Whisper model is loaded at import time and kept in memory. Running multiple Gunicorn workers will multiply memory usage (each worker loads its own copy of the model). To avoid this, run a single worker and use a multi-threaded server, or host the model in a separate dedicated service.

- Ensure `ffmpeg` is installed and accessible on PATH on the host.

- For heavy usage, use a GPU and install the appropriate `torch`/CUDA packages.

---