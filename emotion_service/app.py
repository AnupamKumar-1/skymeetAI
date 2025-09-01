#!/usr/bin/env python3
"""
app.py - minimal Flask API:
POST /analyze
  form-data:
    meeting_id -> string
    participant_id -> string
    file -> (audio or video)
    type -> audio|video
Response: JSON with timeline & anomalies
It can also optionally forward the result to your backend (configure BACKEND_URL).
This variant preloads models at startup and uses a fast inference path when possible.
"""

import os
import tempfile
import traceback
import logging
import subprocess
import importlib
import shutil
import json
from flask import Flask, request, jsonify, make_response
import requests
from PIL import Image
import numpy as np
import soundfile as sf
from werkzeug.utils import secure_filename
from typing import Optional
from inference import predict

# Import the inference module robustly (ensure we get the module, not a function)
try:
    predict_module = importlib.import_module("inference.predict")
except Exception:
    # fallback to trying a top-level predict module (in case package structure differs)
    try:
        predict_module = importlib.import_module("predict")
    except Exception as imp_err:
        raise ImportError(
            "Failed to import inference.predict or predict module. "
            "Ensure your package path is correct and that 'predict.py' is importable."
        ) from imp_err

# --------------------
# CORS configuration
# --------------------
# By default allow the typical dev origin http://localhost:3000 and 127.0.0.1:3000.
# You can override via env var: FLASK_CORS_ORIGINS (comma-separated)
DEFAULT_ALLOWED = {"http://localhost:3000", "http://127.0.0.1:3000"}
env_origins = os.environ.get("FLASK_CORS_ORIGINS")
if env_origins:
    allowed_origins = set([o.strip() for o in env_origins.split(",") if o.strip()])
else:
    allowed_origins = DEFAULT_ALLOWED

ENABLE_CORS = os.environ.get("FLASK_ENABLE_CORS", "1") not in ("0", "false", "False", "")

# Prefer flask-cors if available
CORS = None
if ENABLE_CORS:
    try:
        from flask_cors import CORS as _CORS
        CORS = _CORS
    except Exception:
        CORS = None

app = Flask(__name__)

# If flask-cors import succeeded and env var set, enable it on the app for /analyze
if ENABLE_CORS and CORS:
    try:
        # configure to only allow our allowed_origins, support credentials and preflight
        CORS(app,
             resources={r"/analyze": {"origins": list(allowed_origins)}},
             supports_credentials=True,
             expose_headers=["Content-Type"],
             allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept"])
    except Exception:
        # silent fallback to our manual handlers below
        pass

# If flask-cors not present or we want an explicit robust fallback, provide manual handling:
@app.before_request
def handle_preflight():
    """
    Return a quick response for OPTIONS preflight requests with appropriate headers.
    This is only used if the flask-cors extension didn't already handle it.
    """
    if request.method == "OPTIONS":
        origin = request.headers.get("Origin")
        resp = make_response("", 200)
        if origin and (origin in allowed_origins or "*" in allowed_origins):
            resp.headers["Access-Control-Allow-Origin"] = origin if origin in allowed_origins else "*"
            resp.headers["Vary"] = "Origin"
        else:
            # if no origin or not allowed, still respond but do not set CORS allow-origin
            pass
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        # Allow common headers used for multipart/form-data and fetch requests
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        return resp  # short-circuit preflight

@app.after_request
def add_cors_headers(response):
    """
    Ensure responses include appropriate CORS headers if the request included an Origin header.
    Works as a safety-net in case flask-cors didn't run or specific responses missed headers.
    """
    origin = request.headers.get("Origin")
    if origin and (origin in allowed_origins or "*" in allowed_origins):
        # echo the origin (preferred for credentials) if it's allowed
        response.headers["Access-Control-Allow-Origin"] = origin if origin in allowed_origins else "*"
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With, Accept")
    return response

# Set up logging
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
logger.info("FLASK_ENABLE_CORS=%s allowed_origins=%s", ENABLE_CORS, allowed_origins)

# uploads/temp dir (kept local)
EMO_SAVE = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(EMO_SAVE, exist_ok=True)

# configurable env vars
BACKEND_URL = os.environ.get("BACKEND_URL")  # e.g. "http://localhost:8000/api/v1"
# multimodal model & isolation forest default paths (relative to project)
BASE_DIR = os.path.dirname(__file__)

MULTIMODAL_PATH = os.environ.get(
    "MULTIMODAL_PATH",
    os.path.join(BASE_DIR, "saved_models", "multimodal_best.pth")
)
IF_PATH = os.environ.get(
    "IF_PATH",
    os.path.join(BASE_DIR, "results", "anomaly", "isolation_forest.joblib")
)

# Whether to remove the original uploaded file after successful conversion
REMOVE_SOURCE_AFTER_CONVERT = os.environ.get("REMOVE_SOURCE_AFTER_CONVERT", "0") in ("1", "true", "True")

# Whether to print inference results to stdout (useful for docker logs / journalctl)
PRINT_RESULTS_TO_STDOUT = os.environ.get("PRINT_RESULTS_TO_STDOUT", "1") in ("1", "true", "True")

# Helper: create a tiny black image file
def _make_dummy_image():
    fd, p = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img = Image.new("RGB", (32, 32), (0, 0, 0))
    img.save(p)
    return p

# Helper: create a short silent wav for audio-needed cases
def _make_silent_wav(duration_sec=0.5, sr=16000):
    fd, p = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    samples = np.zeros(int(duration_sec * sr), dtype="float32")
    sf.write(p, samples, sr)
    return p


# Convert arbitrary video file to an .mp4 using ffmpeg. Returns new path (or original path on failure).
def _convert_video_to_mp4(input_path):
    """Attempt to convert input_path -> .mp4 and return the mp4 path on success.
    If conversion fails, returns the original input_path.
   """
    # If already mp4, nothing to do
    if input_path.lower().endswith(".mp4"):
        return input_path

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        logger.warning("ffmpeg not found on PATH; skipping video conversion for %s", input_path)
        return input_path

    mp4_path = os.path.splitext(input_path)[0] + ".mp4"

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        mp4_path,
    ]

    try:
        logger.info("Converting video to mp4: %s -> %s", input_path, mp4_path)
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Conversion succeeded: %s", mp4_path)
        # Optionally remove the source file to save space
        if REMOVE_SOURCE_AFTER_CONVERT:
            try:
                os.remove(input_path)
                logger.info("Removed original source after convert: %s", input_path)
            except Exception:
                logger.exception("Failed to remove original video after conversion: %s", input_path)
        return mp4_path
    except subprocess.CalledProcessError as e:
        logger.exception("ffmpeg conversion failed (CalledProcessError): %s", e)
        # cleanup partial mp4 if present
        try:
            if os.path.exists(mp4_path):
                os.remove(mp4_path)
        except Exception:
            pass
        return input_path
    except Exception as e:
        logger.exception("Unexpected error during video conversion: %s", e)
        try:
            if os.path.exists(mp4_path):
                os.remove(mp4_path)
        except Exception:
            pass
        return input_path


# FFmpeg-based fallback to extract a single frame and a WAV from a video/file
def _extract_frame_and_audio_with_ffmpeg(input_path, target_frame_time=None, audio_sr=16000):
    """
    Returns (image_path, audio_path)
    image_path: path to extracted PNG/JPG frame
    audio_path: path to extracted WAV (mono, audio_sr)
    If extraction fails, raises an Exception.
    """
    # Choose seek time
    if target_frame_time is None:
        # a small non-zero offset can help with some container formats
        seek = 0.5
    else:
        try:
            seek = float(target_frame_time)
        except Exception:
            seek = 0.5

    # Temp files
    img_fd, img_path = tempfile.mkstemp(suffix=".png")
    os.close(img_fd)
    audio_fd, audio_path = tempfile.mkstemp(suffix=".wav")
    os.close(audio_fd)

    # Build ffmpeg commands
    # Extract single frame (seek then grab 1 frame)
    cmd_frame = [
        "ffmpeg", "-y",
        "-ss", str(seek),
        "-i", input_path,
        "-frames:v", "1",
        "-q:v", "2",
        img_path,
    ]

    # Extract audio, convert to mono and target sample rate
    cmd_audio = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", str(audio_sr),
        "-f", "wav",
        audio_path,
    ]

    # Run commands
    try:
        subprocess.run(cmd_frame, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        # cleanup partial files
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg frame extraction failed: {e}")

    try:
        subprocess.run(cmd_audio, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        # cleanup partial files
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
        # If audio extraction fails, we still might want to return image but for predict we need audio.
        raise RuntimeError(f"ffmpeg audio extraction failed: {e}")

    return img_path, audio_path

# Dummy IsolationForest fallback when IF_PATH missing or load fails
class _DummyIF:
    def decision_function(self, X):
        # neutral anomaly score 0.0
        import numpy as _np
        return _np.zeros((len(X),))

    def predict(self, X):
        # mark everything as normal (1)
        import numpy as _np
        return _np.ones((len(X),), dtype=int)


# -------------------------
# Preload models at startup (best-effort)
# -------------------------
multimodal_model: Optional[object] = None
multimodal_meta: dict = {}
isolation_forest_model: Optional[object] = None

logger.info("Attempting to preload multimodal model and isolation forest (this speeds up /analyze requests)...")
try:
    # Prefer the new loader which accepts paths and returns loaded objects
    multimodal_model, multimodal_meta, isolation_forest_model = predict_module.load_models(MULTIMODAL_PATH, IF_PATH)
    logger.info("Preloaded multimodal model and isolation forest successfully.")
except Exception as e:
    logger.exception("Preloading both models failed: %s", e)
    # Try loading the isolation forest alone (useful when model .pth missing but IF exists)
    try:
        if os.path.exists(IF_PATH):
            isolation_forest_model = predict_module.load_isolation_forest(IF_PATH)
            logger.info("Loaded isolation forest separately from %s", IF_PATH)
        else:
            logger.warning("Isolation forest file not found at %s", IF_PATH)
    except Exception as e2:
        logger.exception("Failed loading isolation forest separately: %s", e2)
        isolation_forest_model = None

    # If IF still missing, monkeypatch predict_module.load_isolation_forest to return a dummy
    if isolation_forest_model is None:
        logger.warning("No IsolationForest available; using Dummy IF for inference.")
        predict_module.load_isolation_forest = lambda p: _DummyIF()  # monkeypatch so legacy paths still work


@app.route("/health", methods=["GET"])
def health():
    return "ok"


@app.route("/analyze", methods=["POST"])
def analyze():
    temp_files = []
    try:
        # -------------------------
        # Diagnostic logging
        # -------------------------
        try:
            logger.info("---- /analyze request ----")
            logger.info("Remote addr: %s", request.remote_addr)
            logger.info("Content-Type: %s", request.content_type)
            logger.info("Form keys: %s", list(request.form.keys()))
            logger.info("Files keys: %s", list(request.files.keys()))
            # Print trimmed form values to help debugging (avoid huge outputs)
            for k in request.form.keys():
                try:
                    v = request.form.get(k)
                    if isinstance(v, str):
                        logger.info(" form[%s] = %s", k, (v[:200] + "...") if len(v) > 200 else v)
                    else:
                        logger.info(" form[%s] = %s", k, str(type(v)))
                except Exception:
                    pass
            logger.info("--------------------------")
        except Exception:
            logger.exception("Failed to log request metadata")

        # Accept multiple possible form keys to be robust with different clients
        meeting_id = (
            request.form.get("meeting_id")
            or request.form.get("meeting_code")
            or request.form.get("meeting")
            or None
        )
        participant_id = (
            request.form.get("participant_id")
            or request.form.get("participant")
            or request.form.get("speaker_id")
            or None
        )

        # Accept single 'file' or a set of 'audio_files' (first file)
        f = request.files.get("file")
        if not f:
            audio_files = request.files.getlist("audio_files")
            if audio_files:
                f = audio_files[0]

        # fallback: any uploaded file (first)
        if not f and request.files:
            try:
                f = next(iter(request.files.values()))
            except StopIteration:
                f = None

        typ = request.form.get("type", "audio").lower()  # 'audio' or 'video'

        if not f or not meeting_id or not participant_id:
            missing = []
            if not f:
                missing.append("file")
            if not meeting_id:
                missing.append("meeting_id (or meeting_code/meeting)")
            if not participant_id:
                missing.append("participant_id (or participant/speaker_id)")

            # Include form/file keys to help the caller debug
            resp = {
                "error": "missing required fields",
                "missing": missing,
                "form_keys": list(request.form.keys()),
                "files_keys": list(request.files.keys()),
            }
            logger.warning("Bad request - missing fields: %s", missing)
            return jsonify(resp), 400

        # Sanitize filename
        safe_filename = secure_filename(f.filename or "upload.bin")
        fname = f"{meeting_id}__{participant_id}__{safe_filename}"
        saved_path = os.path.join(EMO_SAVE, fname)
        f.save(saved_path)
        logger.info("Saved upload to %s", saved_path)

        # If this is a video upload, attempt to normalize to .mp4 for downstream tools
        if typ == "video":
            try:
                converted = _convert_video_to_mp4(saved_path)
                if converted != saved_path:
                    logger.info("Normalized video path: %s -> %s", saved_path, converted)
                    saved_path = converted
                else:
                    logger.info("Video conversion not performed/failed; continuing with original: %s", saved_path)
            except Exception:
                logger.exception("Unexpected error while attempting to convert uploaded video to mp4")

        # Prepare image_path and audio_path to call predict.predict(image_path,audio_path,...)
        image_path = None
        audio_path = None

        if typ == "audio":
            audio_path = saved_path
            image_path = _make_dummy_image()
            temp_files.append(image_path)
            logger.debug("Using dummy image for audio-only upload: %s", image_path)
        elif typ == "video":
            # try to extract frame + audio using predict.extract_frame_and_audio if available
            tried_fns = []
            used_fallback = False
            try:
                if hasattr(predict_module, "extract_frame_and_audio"):
                    tried_fns.append("predict_module.extract_frame_and_audio")
                    image_path, audio_path = predict_module.extract_frame_and_audio(saved_path, target_frame_time=None)
                    temp_files.extend([image_path, audio_path])
                    logger.info("Used predict_module.extract_frame_and_audio for %s", saved_path)
                else:
                    raise AttributeError("predict_module has no attribute 'extract_frame_and_audio'")
            except Exception:
                logger.exception("Frame/audio extraction with predict_module failed; trying FFmpeg fallback")
                # Try ffmpeg fallback
                try:
                    tried_fns.append("ffmpeg_fallback")
                    image_path, audio_path = _extract_frame_and_audio_with_ffmpeg(saved_path, target_frame_time=None)
                    temp_files.extend([image_path, audio_path])
                    used_fallback = True
                    logger.info("FFmpeg fallback succeeded for %s", saved_path)
                except Exception:
                    logger.exception("FFmpeg extraction failed; using dummy image + silent wav")
                    # Final fallback to dummy image + silent wav
                    image_path = _make_dummy_image()
                    audio_path = _make_silent_wav()
                    temp_files.extend([image_path, audio_path])
        else:
            return jsonify({"error": "unknown type (must be 'audio' or 'video')"}), 400

        # Decide whether to use preloaded models (fast path) or the legacy disk-load path
        try:
            if multimodal_model is not None and isolation_forest_model is not None:
                logger.info("Using preloaded models for inference.")
                res = predict_module.predict_with_loaded_models(
                    image_path, audio_path, multimodal_model, isolation_forest_model,
                    meta=multimodal_meta
                )
            else:
                # If we didn't preload models, fallback to on-demand predict() which accepts paths.
                # Ensure IF_PATH exists or monkeypatch will make predict_module.load_isolation_forest return DummyIF
                use_if_path = IF_PATH if os.path.exists(IF_PATH) else ""
                if not os.path.exists(MULTIMODAL_PATH):
                    logger.warning("Multimodal .pth not found at %s; predict() will raise unless you provide a model.", MULTIMODAL_PATH)
                logger.info("Calling predict (on-demand) with image=%s audio=%s multimodal=%s if_path=%s",
                            image_path, audio_path, MULTIMODAL_PATH, use_if_path)
                res = predict_module.predict(image_path, audio_path, MULTIMODAL_PATH, use_if_path)
        except Exception as e:
            # Catch inference errors and return structured error (trace included for dev)
            logger.exception("Inference failed: %s", e)
            return jsonify({"error": "inference_failed", "message": str(e), "trace": traceback.format_exc()}), 500

        # ----- DEBUG / USER-FACING LOGGING: print inference to terminal/logs -----
        try:
            # pretty log to logger
            logger.info("Inference result for meeting=%s participant=%s:\n%s",
                        meeting_id, participant_id, json.dumps(res, indent=2))
            # print to stdout for docker/system logs (optional toggle)
            if PRINT_RESULTS_TO_STDOUT:
                pretty = {"meeting_id": meeting_id, "participant_id": participant_id, "result": res}
                print(json.dumps(pretty, indent=2))
        except Exception:
            logger.exception("Failed to pretty-print inference result to terminal")
        # -------------------------------------------------------------------------

        out = {"meeting_id": meeting_id, "participant_id": participant_id, "result": res}

        # Optionally forward to backend
        if BACKEND_URL:
            try:
                resp = requests.post(f"{BACKEND_URL.rstrip('/')}/transcripts/emotions", json=out, timeout=10)
                out["backend_status"] = resp.status_code
                try:
                    out["backend_resp_text"] = resp.text
                except Exception:
                    pass
                logger.info("Forwarded result to backend %s status=%s", BACKEND_URL, resp.status_code)
            except Exception:
                logger.exception("Failed to forward to backend")
                out["backend_error"] = traceback.format_exc()

        return jsonify(out)

    except Exception as e:
        logger.exception("Unhandled exception in /analyze")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    finally:
        # cleanup temp files
        for p in set(temp_files):
            try:
                os.remove(p)
            except Exception:
                logger.debug("Failed to remove temp file %s", p)


if __name__ == "__main__":
    # Only used for local development
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=False)