import os
import uuid
import json
import subprocess
import shlex
import re
import threading
from collections import Counter
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Note: heavy ML libs are imported lazily inside load_models_if_needed()

MIN_DURATION_SEC = 0.30
MIN_TEXT_CHARS = 3
NO_SPEECH_PROB_THRESH = 0.6
AVG_LOGPROB_THRESH = -1.5
ENABLE_FFMPEG_ENERGY_check = False
MIN_MEAN_VOLUME_DB = -45.0

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXT = {"webm", "wav", "mp3", "m4a", "ogg", "aac", "mp4"}

CLEANUP_DELAY_SEC = int(os.environ.get("CLEANUP_DELAY_SEC", 120))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure model names via environment so you can switch in Render UI
# Prefer tiny.en (English-only tiny) for lower memory footprint; change via env ASR_MODEL
ASR_MODEL_NAME = os.environ.get("ASR_MODEL", "tiny.en")  # tiny.en|tiny|base|small|medium
EMOTION_MODEL_NAME = os.environ.get(
    "EMOTION_MODEL",
    "j-hartmann/emotion-english-distilroberta-base",
)

# Keep these as module globals but don't load heavy models at import time
asr_model = None
emotion_pipeline = None


def allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_EXT


def convert_to_wav(src_path, dst_path):
    cmd = f'ffmpeg -y -i {shlex.quote(src_path)} -ac 1 -ar 16000 -vn {shlex.quote(dst_path)}'
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg conversion failed: {stderr}")


def segment_mean_volume(source_file, start, duration):
    try:
        cmd = (
            f'ffmpeg -v error -ss {float(start)} -t {float(duration)} -i {shlex.quote(source_file)} '
            '-af volumedetect -f null -'
        )
        p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        stderr = p.stderr.decode("utf-8", errors="ignore")
        m = re.search(r"mean_volume:\s*([-0-9\.]+)\s*dB", stderr)
        if m:
            return float(m.group(1))
    except Exception as e:
        print(f"[energy-check] failed: {e}")
    return None


def schedule_file_cleanup(file_paths, delay=CLEANUP_DELAY_SEC):
    """
    Schedules deletion of the given files after `delay` seconds.
    """

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
    """
    Safely extract a single label string from a HuggingFace pipeline output.
    Fallback: 'neutral'
    """
    try:
        if isinstance(raw_output, dict):
            return raw_output.get("label", "neutral")

        if isinstance(raw_output, (list, tuple)) and len(raw_output) > 0:
            first = raw_output[0]
            if isinstance(first, dict) and "label" in first:
                return first["label"]
            if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], dict) and "label" in first[0]:
                return first[0]["label"]
            if isinstance(first, str):
                return first
    except Exception:
        pass
    return "neutral"


def load_models_if_needed():
    """Lazy-load ASR and emotion models on first request with safe fallbacks.

    This avoids importing heavy libraries at module import time and ensures the
    server can still respond (with degraded capability) if model loading fails.
    """
    global asr_model, emotion_pipeline

    # --- ASR (Whisper) ---
    if asr_model is None:
        model_name = os.environ.get("ASR_MODEL", ASR_MODEL_NAME) or ASR_MODEL_NAME
        # prefer tiny.en by default (lower RAM than multilingual tiny)
        if model_name == "tiny":
            model_name = "tiny.en"
        try:
            print(f"[models] Loading Whisper ASR model: {model_name} ...")
            # import lazily
            import whisper as _whisper
            asr_model = _whisper.load_model(model_name)
            print("[models] Whisper loaded.")
        except MemoryError as me:
            print(f"[models] MemoryError loading Whisper ({model_name}): {me}")
            asr_model = None
        except Exception as e:
            print(f"[models] Failed to load Whisper model ({model_name}): {e}")
            asr_model = None

    # --- Emotion classifier (optional) ---
    # If EMOTION_MODEL env var is empty string or not set to a model name, skip loading.
    env_emotion_model = os.environ.get("EMOTION_MODEL", EMOTION_MODEL_NAME)
    if (emotion_pipeline is None) and env_emotion_model:
        try:
            print(f"[models] Loading emotion classifier: {env_emotion_model} ...")
            # lazy import the HF pipeline
            from transformers import pipeline as _pipeline
            # device=-1 forces CPU
            emotion_pipeline = _pipeline("text-classification", model=env_emotion_model, top_k=1, device=-1)
            print("[models] Emotion classifier loaded.")
        except MemoryError as me:
            print(f"[models] MemoryError loading emotion model ({env_emotion_model}): {me}")
            emotion_pipeline = None
        except Exception as e:
            print(f"[models] Failed to load emotion classifier ({env_emotion_model}): {e}")
            emotion_pipeline = None
    else:
        if not env_emotion_model:
            print("[models] EMOTION_MODEL empty — skipping emotion pipeline.")


@app.route("/process_meeting", methods=["POST"])
def process_meeting():
    # Ensure models are available (load lazily)
    load_models_if_needed()

    files = request.files.getlist("audio_files")
    if not files:
        return jsonify({"success": False, "error": "no audio_files uploaded"}), 400

    meeting_code = request.form.get("meeting_code", "UNKNOWN")
    speaker_map_raw = request.form.get("speaker_map", "{}")
    try:
        speaker_map = json.loads(speaker_map_raw)
    except Exception:
        speaker_map = {}

    results = {}
    created_file_paths = []

    for f in files:
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            base_name = os.path.splitext(filename)[0]
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(save_path)
            created_file_paths.append(save_path)

            wav_path = os.path.join(UPLOAD_FOLDER, f"{base_name}.wav")
            try:
                convert_to_wav(save_path, wav_path)
                created_file_paths.append(wav_path)
            except Exception as e:
                print(f"ffmpeg convert failed for {save_path}: {e}. Using original file.")
                wav_path = save_path

            # If ASR model failed to load, skip transcription but keep process safe
            if asr_model is None:
                print("[process] ASR model not available; skipping transcription for file:", save_path)
                asr_result = {"segments": []}
            else:
                try:
                    asr_result = asr_model.transcribe(wav_path, language="en")
                except Exception as e:
                    print(f"Whisper failed for {wav_path}: {e}")
                    asr_result = {"segments": []}

            segs_with_emo = []
            for seg in asr_result.get("segments", []):
                text = (seg.get("text") or "").strip()
                if not text or len(text) < MIN_TEXT_CHARS:
                    continue

                emo_label = "neutral"
                if emotion_pipeline is not None:
                    try:
                        raw_emo = emotion_pipeline(text, top_k=1)
                        emo_label = _extract_label_from_pipeline_output(raw_emo)
                    except Exception as e:
                        print(f"[emotion] pipeline failed for text segment: {e}")
                        emo_label = "neutral"
                else:
                    # quiet log to avoid spamming logs for every segment in production
                    # print("[process] Emotion pipeline not available; using fallback 'neutral'")
                    pass

                segs_with_emo.append({
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "text": text,
                    "emotion": emo_label,
                })

            results[base_name] = {
                "speaker": speaker_map.get(base_name, base_name),
                "segments": segs_with_emo,
            }

    # Merge and group logic unchanged
    merged_entries = []
    for file_id, info in results.items():
        for seg in info["segments"]:
            merged_entries.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": info["speaker"],
                "text": seg["text"],
                "emotion": seg["emotion"],
            })
    merged_entries.sort(key=lambda x: x["start"])

    groups = []
    current = None
    for e in merged_entries:
        if current and e["speaker"] == current["speaker"] and e["start"] - current["end"] <= 1.0:
            current["end"] = e["end"]
            current["texts"].append(e["text"])
            current["emotions"].append(e["emotion"])
        else:
            if current:
                groups.append(current)
            current = {
                "start": e["start"],
                "end": e["end"],
                "speaker": e["speaker"],
                "texts": [e["text"]],
                "emotions": [e["emotion"]],
            }
    if current:
        groups.append(current)

    lines = []
    for g in groups:
        lines.append(f"[{g['start']:.2f}\u2013{g['end']:.2f}] {g['speaker']}:")
        for t in g["texts"]:
            lines.append(f"    {t}")
        lines.append("")

    speaker_summary = {}
    for g in groups:
        cnt = Counter(g["emotions"])
        if len(cnt) > 0:
            top, _ = cnt.most_common(1)[0]
        else:
            top = "neutral"
        breakdown = ", ".join([f"{k} ({v})" for k, v in cnt.items()])
        speaker_summary[g["speaker"]] = f"top={top} — {breakdown}"

    lines.append("=== Emotion Summary (by speaker) ===")
    for sp, summary in speaker_summary.items():
        lines.append(f"{sp}: {summary}")

    transcript_text = "\n".join(lines)

    txt_filename = f"{meeting_code}_{uuid.uuid4().hex}.txt"
    txt_path = os.path.join(OUTPUT_FOLDER, txt_filename)
    with open(txt_path, "w", encoding="utf-8") as wf:
        wf.write(transcript_text)
    created_file_paths.append(txt_path)

    json_filename = f"{meeting_code}_{uuid.uuid4().hex}.json"
    json_path = os.path.join(OUTPUT_FOLDER, json_filename)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({"groups": groups, "speaker_summary": speaker_summary}, jf, indent=2)
    created_file_paths.append(json_path)

    schedule_file_cleanup(created_file_paths, delay=CLEANUP_DELAY_SEC)

    return (
        jsonify({
            "success": True,
            "transcript_text": transcript_text,
            "txt_filename": txt_filename,
            "json_filename": json_filename,
            "files_will_be_deleted_in_sec": CLEANUP_DELAY_SEC,
        }),
        200,
    )


@app.route("/outputs/<path:filename>", methods=["GET"])
def download_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    # For local dev: use Flask built-in server
    app.run(host="0.0.0.0", port=port, debug=False)
