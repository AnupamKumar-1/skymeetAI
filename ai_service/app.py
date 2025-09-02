import os
import uuid
import json
import subprocess
import shlex
import re
import threading
import time
from collections import Counter
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import whisper
from transformers import pipeline

MIN_DURATION_SEC = 0.30
MIN_TEXT_CHARS = 3
NO_SPEECH_PROB_THRESH = 0.6
AVG_LOGPROB_THRESH = -1.5
ENABLE_FFMPEG_ENERGY_check = False
MIN_MEAN_VOLUME_DB = -45.0

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXT = {"webm", "wav", "mp3", "m4a", "ogg", "aac", "mp4"}

CLEANUP_DELAY_SEC = 120

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

# Load models at startup
print("Loading Whisper model (this can take time)...")
asr_model = whisper.load_model("small")

print("Loading emotion classifier (HuggingFace pipeline)...")

# keep top_k=1 for efficiency
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)


def allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_EXT

def convert_to_wav(src_path, dst_path):
    cmd = f'ffmpeg -y -i {shlex.quote(src_path)} -ac 1 -ar 16000 -vn {shlex.quote(dst_path)}'
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        stderr = proc.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(f"ffmpeg conversion failed: {stderr}")

def segment_mean_volume(source_file, start, duration):
    try:
        cmd = (
            f'ffmpeg -v error -ss {float(start)} -t {float(duration)} -i {shlex.quote(source_file)} '
            '-af volumedetect -f null -'
        )
        p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        stderr = p.stderr.decode('utf-8', errors='ignore')
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
    Given the raw output from a HuggingFace text-classification pipeline (which can be
    a dict, list-of-dicts, nested lists, etc.), return a single label string.
    If extraction fails, return "neutral" as a safe fallback.
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


@app.route("/process_meeting", methods=["POST"])
def process_meeting():
    files = request.files.getlist("audio_files")
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

                try:
                    raw_emo = emotion_pipeline(text, top_k=1)
                    emo_label = _extract_label_from_pipeline_output(raw_emo)
                except Exception as e:
                    print(f"[emotion] pipeline failed for text segment: {e}")
                    emo_label = "neutral"

                segs_with_emo.append({
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "text": text,
                    "emotion": emo_label,
                })

            results[base_name] = {
                "speaker": speaker_map.get(base_name, base_name),
                "segments": segs_with_emo
            }

    merged_entries = []
    for file_id, info in results.items():
        for seg in info["segments"]:
            merged_entries.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": info["speaker"],
                "text": seg["text"],
                "emotion": seg["emotion"]
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
        lines.append(f"[{g['start']:.2f}–{g['end']:.2f}] {g['speaker']}:")
        for t in g["texts"]:
            lines.append(f"    {t}")
        lines.append("")

    speaker_summary = {}
    for g in groups:
        cnt = Counter(g["emotions"])
        top, _ = cnt.most_common(1)[0]
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

    return jsonify({
        "success": True,
        "transcript_text": transcript_text,
        "txt_filename": txt_filename,
        "json_filename": json_filename,
        "files_will_be_deleted_in_sec": CLEANUP_DELAY_SEC
    }), 200

@app.route("/outputs/<path:filename>", methods=["GET"])
def download_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
