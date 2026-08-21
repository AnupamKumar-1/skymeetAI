from dotenv import load_dotenv

load_dotenv()

import os
import json
import asyncio
import requests

from fastapi import FastAPI, File, Form, Header, UploadFile, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from werkzeug.utils import secure_filename

from config import *
from utils.audio import convert_to_wav, ensure_ffmpeg_available

ensure_ffmpeg_available()

from utils.helpers import allowed_file, clean_speaker, schedule_file_cleanup
from services.asr_service import transcribe_and_emotion, build_intelligent_summary
from services.processing_service import merge_segments, build_transcript_text

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = FastAPI()

allowed_origins = os.getenv("ALLOWED_ORIGINS", "")
origins = [o.strip() for o in allowed_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str, request: Request):
    """Return HTTP 200 for all CORS preflight OPTIONS requests."""
    return JSONResponse(content={"ok": True})


async def run_processing(
    audio_files_data: list[dict],
    meeting_code: str,
    speaker_map_dict: dict,
    host_secret: str,
    user_token: str,
):
    """Orchestrate per-file transcription and deliver results to NODE_API.

    For each audio file: validates extension, writes to disk, converts to
    mono 16 kHz WAV, transcribes with Whisper, and classifies per-segment
    emotion. Merges all speaker segments, builds a formatted transcript and
    aggregate analysis, then POSTs to NODE_API. Skips the callback if the
    merged segment list is empty. Schedules uploaded files for deletion
    regardless of outcome.

    Args:
        audio_files_data: List of dicts with keys ``filename``, ``contents``
            (raw bytes), and ``mimetype``.
        meeting_code: Uppercased meeting identifier forwarded to NODE_API.
        speaker_map_dict: Maps filename base (no extension) to display name.
        host_secret: Forwarded verbatim as ``x-host-secret`` to NODE_API.
        user_token: If non-empty, forwarded as ``Authorization: Bearer`` to
            NODE_API.
    """
    results = {}
    created_files = []

    for file_data in audio_files_data:
        filename = file_data["filename"]
        contents = file_data["contents"]

        if not allowed_file(filename, ALLOWED_EXT):
            continue

        filename = secure_filename(filename)
        base = os.path.splitext(filename)[0]
        real_name = clean_speaker(speaker_map_dict.get(base, base))

        save_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(save_path, "wb") as out:
            out.write(contents)

        created_files.append(save_path)

        wav_path = os.path.join(UPLOAD_FOLDER, f"{base}.wav")

        try:
            convert_to_wav(save_path, wav_path)
            created_files.append(wav_path)
        except Exception:
            wav_path = save_path

        result = await asyncio.to_thread(transcribe_and_emotion, wav_path, real_name)

        results[base] = {
            "speaker": real_name,
            "segments": result["segments"],
            "analysis": result["analysis"],
        }

    merged = merge_segments(
        {k: {"segments": v["segments"]} for k, v in results.items()}
    )

    transcript_text = build_transcript_text(merged)
    analysis = build_intelligent_summary(
        [seg for r in results.values() for seg in r["segments"]]
    )

    schedule_file_cleanup(created_files, CLEANUP_DELAY_SEC)

    if not merged:
        return

    node_headers = {
        "Content-Type": "application/json",
        "x-host-secret": host_secret,
    }

    if user_token:
        node_headers["Authorization"] = f"Bearer {user_token}"

    payload = {
        "meetingCode": meeting_code,
        "transcriptText": transcript_text,
        "metadata": {
            "segments": merged,
            "analysis": analysis,
        },
    }

    delays = [5, 15, 30]
    for attempt in range(len(delays) + 1):
        try:
            res = requests.post(
                NODE_API,
                json=payload,
                headers=node_headers,
                timeout=None,
            )
            
            if 500 <= res.status_code < 600:
                raise requests.RequestException(f"HTTP {res.status_code}")
            if 400 <= res.status_code < 500:
                print(
                    f"[transcript] Node API rejected with "
                    f"{res.status_code}: {res.text[:200]} - not retrying"
                    )
                break
            if res.status_code < 400:
                print(f"[transcript] Node API callback success: {res.status_code}")
                break
            
        except requests.RequestException as e:
            if attempt < len(delays):
                delay = delays[attempt]
                print(f"Node API callback failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
            else:
                print(f"Node API callback permanent failure after {len(delays)} retries: {e}")


@app.post("/process_meeting")
async def process_meeting(
    request: Request,
    background_tasks: BackgroundTasks,
    audio_files: list[UploadFile] = File(...),
    meeting_code: str = Form(default="UNKNOWN"),
    speaker_map: str = Form(default="{}"),
    host_secret: str = Header(default="", alias="x-host-secret"),
    user_token: str = Header(default="", alias="x-user-token"),
    internal_token: str = Header(default="", alias="x-internal-token"),
):
    """Accept a multi-speaker audio upload and dispatch processing asynchronously.

    Reads all uploaded files into memory, parses the speaker map, and enqueues
    ``run_processing`` as a background task. Returns HTTP 202 immediately; the
    NODE_API callback is made after background processing completes.

    Args:
        request: Raw FastAPI request (unused directly; required by FastAPI).
        background_tasks: FastAPI background task queue.
        audio_files: One or more audio files to transcribe.
        meeting_code: Meeting identifier; uppercased before use. Defaults to
            ``"UNKNOWN"``.
        speaker_map: JSON string mapping filename base to display name.
            Defaults to ``"{}"``.
        host_secret: Forwarded to NODE_API as ``x-host-secret``.
        user_token: Forwarded to NODE_API as ``Authorization: Bearer`` if
            non-empty.

    Returns:
        JSONResponse with HTTP 202 and ``{"success": true, "message":
        "Processing started"}``.
    """
    expected_internal_token = os.getenv("INTERNAL_SERVICE_TOKEN")
    if expected_internal_token and internal_token != expected_internal_token:
        return JSONResponse(status_code=401, content={"success": False, "error": "Unauthorized"})

    meeting_code = meeting_code.upper()

    try:
        speaker_map_dict = json.loads(speaker_map)
    except Exception:
        speaker_map_dict = {}

    audio_files_data = []
    for f in audio_files:
        contents = await f.read()
        audio_files_data.append(
            {
                "filename": f.filename,
                "contents": contents,
                "mimetype": f.content_type,
            }
        )

    background_tasks.add_task(
        run_processing,
        audio_files_data,
        meeting_code,
        speaker_map_dict,
        host_secret,
        user_token,
    )

    return JSONResponse(
        status_code=202,
        content={"success": True, "message": "Processing started"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=5001)
