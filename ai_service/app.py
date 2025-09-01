import os
import uuid
import json
import subprocess
import threading
import traceback
from collections import Counter
from flask import Flask, request, jsonify, send_from_directory, make_response, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException

# Lazy-loaded HF pipelines
asr_pipeline = None
emotion_pipeline = None

MIN_TEXT_CHARS = 3
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXT = {"webm", "wav", "mp3", "m4a", "ogg", "aac", "mp4"}
CLEANUP_DELAY_SEC = int(os.environ.get("CLEANUP_DELAY_SEC", 120))
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "https://skymeetai.onrender.com")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

ASR_MODEL_NAME = os.environ.get("ASR_MODEL", "facebook/wav2vec2-base-960h")
EMOTION_MODEL_NAME = os.environ.get("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")


def allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_EXT


def convert_to_wav(src_path, dst_path):
    try:
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-ac", "1", "-ar", "16000", "-vn", dst_path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg conversion failed: {stderr}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg and ffprobe.")


def get_audio_duration(path):
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
               "default=noprint_wrappers=1:nokey=1", path]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        out = p.stdout.decode().strip()
        return float(out)
    except FileNotFoundError:
        app.logger.warning("ffprobe not found on PATH; cannot get duration")
        return None
    except Exception:
        return None


def schedule_file_cleanup(file_paths, delay=CLEANUP_DELAY_SEC):
    def cleanup_job(paths):
        for p in paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
                    print(f"[cleanup] Deleted {p}")
            except Exception as e:
                print(f"[cleanup] Failed to delete {p}: {e}")

    timer = threading.Timer(delay, cleanup_job, [file_paths])
    timer.daemon = True
    timer.start()


def _extract_label_from_pipeline_output(raw_output):
    try:
        if isinstance(raw_output, dict):
            return raw_output.get("label", "neutral")
        if isinstance(raw_output, (list, tuple)) and len(raw_output) > 0:
            first = raw_output[0]
            if isinstance(first, dict) and "label" in first:
                return first["label"]
            if isinstance(first, str):
                return first
    except Exception:
        pass
    return "neutral"


def load_models_if_needed():
    global asr_pipeline, emotion_pipeline
    if asr_pipeline is None:
        try:
            print(f"[models] Loading ASR pipeline: {ASR_MODEL_NAME} ...")
            from transformers import pipeline as _pipeline
            asr_pipeline = _pipeline("automatic-speech-recognition", model=ASR_MODEL_NAME, chunk_length_s=30)
            print("[models] ASR pipeline loaded.")
        except Exception as e:
            print(f"[models] Failed to load ASR pipeline ({ASR_MODEL_NAME}): {e}")
            asr_pipeline = None

    if emotion_pipeline is None and EMOTION_MODEL_NAME:
        try:
            print(f"[models] Loading emotion classifier: {EMOTION_MODEL_NAME} ...")
            from transformers import pipeline as _pipeline
            emotion_pipeline = _pipeline("text-classification", model=EMOTION_MODEL_NAME, top_k=1, device=-1)
            print("[models] Emotion pipeline loaded.")
        except Exception as e:
            print(f"[models] Failed to load emotion pipeline ({EMOTION_MODEL_NAME}): {e}")
            emotion_pipeline = None


def _parse_asr_output(raw, wav_path=None):
    segments = []
    try:
        if isinstance(raw, dict) and 'text' in raw and not raw.get('chunks') and not raw.get('segments'):
            dur = get_audio_duration(wav_path) or 0.0
            segments.append({'start': 0.0, 'end': dur, 'text': raw['text']})
            return segments

        if isinstance(raw, dict) and 'chunks' in raw:
            for c in raw['chunks']:
                text = c.get('text') or c.get('chunk') or ''
                start = c.get('timestamp', [0, 0])[0] if isinstance(c.get('timestamp'), (list, tuple)) else c.get('start', 0.0)
                end = c.get('timestamp', [0, 0])[1] if isinstance(c.get('timestamp'), (list, tuple)) else c.get('end', start)
                segments.append({'start': float(start), 'end': float(end), 'text': text.strip()})
            return segments

        if isinstance(raw, dict) and 'segments' in raw:
            for s in raw['segments']:
                segments.append({'start': float(s.get('start', 0.0)), 'end': float(s.get('end', 0.0)), 'text': (s.get('text') or '').strip()})
            return segments

        if isinstance(raw, (list, tuple)):
            for part in raw:
                if isinstance(part, dict) and 'text' in part:
                    segments.append({'start': float(part.get('start', 0.0)), 'end': float(part.get('end', 0.0)), 'text': (part.get('text') or '').strip()})
            if segments:
                return segments

    except Exception as e:
        print(f"[asr-parse] failed: {e}")

    try:
        text = raw.get('text') if isinstance(raw, dict) else str(raw)
        dur = get_audio_duration(wav_path) or 0.0
        segments.append({'start': 0.0, 'end': dur, 'text': (text or '').strip()})
    except Exception:
        pass

    return segments


@app.after_request
def add_cors_headers(response):
    origin = ALLOWED_ORIGIN or "*"
    response.headers.setdefault("Access-Control-Allow-Origin", origin)
    response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.setdefault("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response


@app.route("/", methods=["GET", "HEAD"])
@app.route("/health", methods=["GET", "HEAD"])
def health_check():
    return jsonify({"status": "ok"}), 200


@app.errorhandler(Exception)
def handle_all_exceptions(e):
    if isinstance(e, HTTPException):
        return jsonify({"success": False, "error": e.description}), e.code

    app.logger.exception("Unhandled exception during request")
    tb = traceback.format_exc()
    body = {"success": False, "error": str(e), "traceback": tb}
    resp = make_response(jsonify(body), 500)
    origin = ALLOWED_ORIGIN or "*"
    resp.headers["Access-Control-Allow-Origin"] = origin
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


@app.route("/process_meeting", methods=["OPTIONS"])
def process_meeting_options():
    resp = Response()
    origin = ALLOWED_ORIGIN or "*"
    resp.headers["Access-Control-Allow-Origin"] = origin
    resp.headers["Access-Control-Allow-Methods"] = "POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    return resp


@app.route("/debug_upload", methods=["POST"])
def debug_upload():
    f = request.files.get("audio_files")
    if not f:
        return jsonify({"success": False, "error": "no audio_files uploaded"}), 400

    filename = secure_filename(f.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        f.save(save_path)
    except Exception as e:
        app.logger.exception("Failed to save uploaded file")
        return jsonify({"success": False, "error": "save failed", "detail": str(e)}), 500

    # Handle conversion only if not already WAV
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "wav":
        wav_path = save_path
    else:
        base_name = os.path.splitext(filename)[0]
        wav_path = os.path.join(UPLOAD_FOLDER, f"{base_name}.wav")
        try:
            convert_to_wav(save_path, wav_path)
        except Exception as e:
            app.logger.exception("ffmpeg conversion failed")
            return jsonify({"success": False, "error": "ffmpeg failed", "detail": str(e)}), 500

    dur = get_audio_duration(wav_path)
    return jsonify({
        "success": True,
        "saved": save_path,
        "wav": wav_path,
        "duration": dur
    }), 200


# (process_meeting and other routes remain unchanged)


@app.route("/outputs/<path:filename>", methods=["GET"])
def download_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
