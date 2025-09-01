# # import os
# # import uuid
# # import json
# # import subprocess
# # import shlex
# # import re
# # import threading
# # from collections import Counter
# # from flask import Flask, request, jsonify, send_from_directory
# # from flask_cors import CORS
# # from werkzeug.utils import secure_filename

# # # Note: heavy ML libs are imported lazily inside load_models_if_needed()

# # MIN_DURATION_SEC = 0.30
# # MIN_TEXT_CHARS = 3
# # NO_SPEECH_PROB_THRESH = 0.6
# # AVG_LOGPROB_THRESH = -1.5
# # ENABLE_FFMPEG_ENERGY_check = False
# # MIN_MEAN_VOLUME_DB = -45.0

# # UPLOAD_FOLDER = "uploads"
# # OUTPUT_FOLDER = "outputs"
# # ALLOWED_EXT = {"webm", "wav", "mp3", "m4a", "ogg", "aac", "mp4"}

# # CLEANUP_DELAY_SEC = int(os.environ.get("CLEANUP_DELAY_SEC", 120))

# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # app = Flask(__name__)
# # CORS(app, resources={r"/*": {"origins": "*"}})

# # # Configure model names via environment so you can switch in Render UI
# # # Prefer tiny.en (English-only tiny) for lower memory footprint; change via env ASR_MODEL
# # ASR_MODEL_NAME = os.environ.get("ASR_MODEL", "tiny.en")  # tiny.en|tiny|base|small|medium
# # EMOTION_MODEL_NAME = os.environ.get(
# #     "EMOTION_MODEL",
# #     "j-hartmann/emotion-english-distilroberta-base",
# # )

# # # Keep these as module globals but don't load heavy models at import time
# # asr_model = None
# # emotion_pipeline = None


# # def allowed_file(filename):
# #     ext = filename.rsplit(".", 1)[-1].lower()
# #     return ext in ALLOWED_EXT


# # def convert_to_wav(src_path, dst_path):
# #     cmd = f'ffmpeg -y -i {shlex.quote(src_path)} -ac 1 -ar 16000 -vn {shlex.quote(dst_path)}'
# #     proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# #     if proc.returncode != 0:
# #         stderr = proc.stderr.decode("utf-8", errors="ignore")
# #         raise RuntimeError(f"ffmpeg conversion failed: {stderr}")


# # def segment_mean_volume(source_file, start, duration):
# #     try:
# #         cmd = (
# #             f'ffmpeg -v error -ss {float(start)} -t {float(duration)} -i {shlex.quote(source_file)} '
# #             '-af volumedetect -f null -'
# #         )
# #         p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
# #         stderr = p.stderr.decode("utf-8", errors="ignore")
# #         m = re.search(r"mean_volume:\s*([-0-9\.]+)\s*dB", stderr)
# #         if m:
# #             return float(m.group(1))
# #     except Exception as e:
# #         print(f"[energy-check] failed: {e}")
# #     return None


# # def schedule_file_cleanup(file_paths, delay=CLEANUP_DELAY_SEC):
# #     """
# #     Schedules deletion of the given files after `delay` seconds.
# #     """

# #     def cleanup_job(paths):
# #         for p in paths:
# #             try:
# #                 if os.path.exists(p):
# #                     os.remove(p)
# #                     print(f"[cleanup] Deleted {p}")
# #             except Exception as e:
# #                 print(f"[cleanup] Failed to delete {p}: {e}")

# #     timer = threading.Timer(delay, cleanup_job, [file_paths])
# #     timer.daemon = True
# #     timer.start()


# # def _extract_label_from_pipeline_output(raw_output):
# #     """
# #     Safely extract a single label string from a HuggingFace pipeline output.
# #     Fallback: 'neutral'
# #     """
# #     try:
# #         if isinstance(raw_output, dict):
# #             return raw_output.get("label", "neutral")

# #         if isinstance(raw_output, (list, tuple)) and len(raw_output) > 0:
# #             first = raw_output[0]
# #             if isinstance(first, dict) and "label" in first:
# #                 return first["label"]
# #             if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], dict) and "label" in first[0]:
# #                 return first[0]["label"]
# #             if isinstance(first, str):
# #                 return first
# #     except Exception:
# #         pass
# #     return "neutral"


# # def load_models_if_needed():
# #     """Lazy-load ASR and emotion models on first request with safe fallbacks.

# #     This avoids importing heavy libraries at module import time and ensures the
# #     server can still respond (with degraded capability) if model loading fails.
# #     """
# #     global asr_model, emotion_pipeline

# #     # --- ASR (Whisper) ---
# #     if asr_model is None:
# #         model_name = os.environ.get("ASR_MODEL", ASR_MODEL_NAME) or ASR_MODEL_NAME
# #         # prefer tiny.en by default (lower RAM than multilingual tiny)
# #         if model_name == "tiny":
# #             model_name = "tiny.en"
# #         try:
# #             print(f"[models] Loading Whisper ASR model: {model_name} ...")
# #             # import lazily
# #             import whisper as _whisper
# #             asr_model = _whisper.load_model(model_name)
# #             print("[models] Whisper loaded.")
# #         except MemoryError as me:
# #             print(f"[models] MemoryError loading Whisper ({model_name}): {me}")
# #             asr_model = None
# #         except Exception as e:
# #             print(f"[models] Failed to load Whisper model ({model_name}): {e}")
# #             asr_model = None

# #     # --- Emotion classifier (optional) ---
# #     # If EMOTION_MODEL env var is empty string or not set to a model name, skip loading.
# #     env_emotion_model = os.environ.get("EMOTION_MODEL", EMOTION_MODEL_NAME)
# #     if (emotion_pipeline is None) and env_emotion_model:
# #         try:
# #             print(f"[models] Loading emotion classifier: {env_emotion_model} ...")
# #             # lazy import the HF pipeline
# #             from transformers import pipeline as _pipeline
# #             # device=-1 forces CPU
# #             emotion_pipeline = _pipeline("text-classification", model=env_emotion_model, top_k=1, device=-1)
# #             print("[models] Emotion classifier loaded.")
# #         except MemoryError as me:
# #             print(f"[models] MemoryError loading emotion model ({env_emotion_model}): {me}")
# #             emotion_pipeline = None
# #         except Exception as e:
# #             print(f"[models] Failed to load emotion classifier ({env_emotion_model}): {e}")
# #             emotion_pipeline = None
# #     else:
# #         if not env_emotion_model:
# #             print("[models] EMOTION_MODEL empty — skipping emotion pipeline.")


# # @app.route("/process_meeting", methods=["POST"])
# # def process_meeting():
# #     # Ensure models are available (load lazily)
# #     load_models_if_needed()

# #     files = request.files.getlist("audio_files")
# #     if not files:
# #         return jsonify({"success": False, "error": "no audio_files uploaded"}), 400

# #     meeting_code = request.form.get("meeting_code", "UNKNOWN")
# #     speaker_map_raw = request.form.get("speaker_map", "{}")
# #     try:
# #         speaker_map = json.loads(speaker_map_raw)
# #     except Exception:
# #         speaker_map = {}

# #     results = {}
# #     created_file_paths = []

# #     for f in files:
# #         if f and allowed_file(f.filename):
# #             filename = secure_filename(f.filename)
# #             base_name = os.path.splitext(filename)[0]
# #             save_path = os.path.join(UPLOAD_FOLDER, filename)
# #             f.save(save_path)
# #             created_file_paths.append(save_path)

# #             wav_path = os.path.join(UPLOAD_FOLDER, f"{base_name}.wav")
# #             try:
# #                 convert_to_wav(save_path, wav_path)
# #                 created_file_paths.append(wav_path)
# #             except Exception as e:
# #                 print(f"ffmpeg convert failed for {save_path}: {e}. Using original file.")
# #                 wav_path = save_path

# #             # If ASR model failed to load, skip transcription but keep process safe
# #             if asr_model is None:
# #                 print("[process] ASR model not available; skipping transcription for file:", save_path)
# #                 asr_result = {"segments": []}
# #             else:
# #                 try:
# #                     asr_result = asr_model.transcribe(wav_path, language="en")
# #                 except Exception as e:
# #                     print(f"Whisper failed for {wav_path}: {e}")
# #                     asr_result = {"segments": []}

# #             segs_with_emo = []
# #             for seg in asr_result.get("segments", []):
# #                 text = (seg.get("text") or "").strip()
# #                 if not text or len(text) < MIN_TEXT_CHARS:
# #                     continue

# #                 emo_label = "neutral"
# #                 if emotion_pipeline is not None:
# #                     try:
# #                         raw_emo = emotion_pipeline(text, top_k=1)
# #                         emo_label = _extract_label_from_pipeline_output(raw_emo)
# #                     except Exception as e:
# #                         print(f"[emotion] pipeline failed for text segment: {e}")
# #                         emo_label = "neutral"
# #                 else:
# #                     # quiet log to avoid spamming logs for every segment in production
# #                     # print("[process] Emotion pipeline not available; using fallback 'neutral'")
# #                     pass

# #                 segs_with_emo.append({
# #                     "start": seg.get("start", 0.0),
# #                     "end": seg.get("end", 0.0),
# #                     "text": text,
# #                     "emotion": emo_label,
# #                 })

# #             results[base_name] = {
# #                 "speaker": speaker_map.get(base_name, base_name),
# #                 "segments": segs_with_emo,
# #             }

# #     # Merge and group logic unchanged
# #     merged_entries = []
# #     for file_id, info in results.items():
# #         for seg in info["segments"]:
# #             merged_entries.append({
# #                 "start": seg["start"],
# #                 "end": seg["end"],
# #                 "speaker": info["speaker"],
# #                 "text": seg["text"],
# #                 "emotion": seg["emotion"],
# #             })
# #     merged_entries.sort(key=lambda x: x["start"])

# #     groups = []
# #     current = None
# #     for e in merged_entries:
# #         if current and e["speaker"] == current["speaker"] and e["start"] - current["end"] <= 1.0:
# #             current["end"] = e["end"]
# #             current["texts"].append(e["text"])
# #             current["emotions"].append(e["emotion"])
# #         else:
# #             if current:
# #                 groups.append(current)
# #             current = {
# #                 "start": e["start"],
# #                 "end": e["end"],
# #                 "speaker": e["speaker"],
# #                 "texts": [e["text"]],
# #                 "emotions": [e["emotion"]],
# #             }
# #     if current:
# #         groups.append(current)

# #     lines = []
# #     for g in groups:
# #         lines.append(f"[{g['start']:.2f}\u2013{g['end']:.2f}] {g['speaker']}:")
# #         for t in g["texts"]:
# #             lines.append(f"    {t}")
# #         lines.append("")

# #     speaker_summary = {}
# #     for g in groups:
# #         cnt = Counter(g["emotions"])
# #         if len(cnt) > 0:
# #             top, _ = cnt.most_common(1)[0]
# #         else:
# #             top = "neutral"
# #         breakdown = ", ".join([f"{k} ({v})" for k, v in cnt.items()])
# #         speaker_summary[g["speaker"]] = f"top={top} — {breakdown}"

# #     lines.append("=== Emotion Summary (by speaker) ===")
# #     for sp, summary in speaker_summary.items():
# #         lines.append(f"{sp}: {summary}")

# #     transcript_text = "\n".join(lines)

# #     txt_filename = f"{meeting_code}_{uuid.uuid4().hex}.txt"
# #     txt_path = os.path.join(OUTPUT_FOLDER, txt_filename)
# #     with open(txt_path, "w", encoding="utf-8") as wf:
# #         wf.write(transcript_text)
# #     created_file_paths.append(txt_path)

# #     json_filename = f"{meeting_code}_{uuid.uuid4().hex}.json"
# #     json_path = os.path.join(OUTPUT_FOLDER, json_filename)
# #     with open(json_path, "w", encoding="utf-8") as jf:
# #         json.dump({"groups": groups, "speaker_summary": speaker_summary}, jf, indent=2)
# #     created_file_paths.append(json_path)

# #     schedule_file_cleanup(created_file_paths, delay=CLEANUP_DELAY_SEC)

# #     return (
# #         jsonify({
# #             "success": True,
# #             "transcript_text": transcript_text,
# #             "txt_filename": txt_filename,
# #             "json_filename": json_filename,
# #             "files_will_be_deleted_in_sec": CLEANUP_DELAY_SEC,
# #         }),
# #         200,
# #     )


# # @app.route("/outputs/<path:filename>", methods=["GET"])
# # def download_output(filename):
# #     return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


# # if __name__ == "__main__":
# #     port = int(os.environ.get("PORT", 5001))
# #     # For local dev: use Flask built-in server
# #     app.run(host="0.0.0.0", port=port, debug=False)

# import os
# import uuid
# import json
# import subprocess
# import shlex
# import re
# import threading
# from collections import Counter
# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# from werkzeug.utils import secure_filename

# # Note: heavy ML libs are imported lazily inside load_models_if_needed()

# MIN_DURATION_SEC = 0.30
# MIN_TEXT_CHARS = 3
# NO_SPEECH_PROB_THRESH = 0.6
# AVG_LOGPROB_THRESH = -1.5
# ENABLE_FFMPEG_ENERGY_check = False
# MIN_MEAN_VOLUME_DB = -45.0

# UPLOAD_FOLDER = "uploads"
# OUTPUT_FOLDER = "outputs"
# ALLOWED_EXT = {"webm", "wav", "mp3", "m4a", "ogg", "aac", "mp4"}

# CLEANUP_DELAY_SEC = int(os.environ.get("CLEANUP_DELAY_SEC", 120))

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

# # Configure model names via environment so you can switch in Render UI
# # Prefer tiny (multilingual tiny) by default; change via env ASR_MODEL
# ASR_MODEL_NAME = os.environ.get("ASR_MODEL", "tiny")  # tiny|tiny.en|base|small|medium
# EMOTION_MODEL_NAME = os.environ.get(
#     "EMOTION_MODEL",
#     "j-hartmann/emotion-english-distilroberta-base",
# )

# # Keep these as module globals but don't load heavy models at import time
# asr_model = None
# emotion_pipeline = None


# def allowed_file(filename):
#     ext = filename.rsplit(".", 1)[-1].lower()
#     return ext in ALLOWED_EXT


# def convert_to_wav(src_path, dst_path):
#     cmd = f'ffmpeg -y -i {shlex.quote(src_path)} -ac 1 -ar 16000 -vn {shlex.quote(dst_path)}'
#     proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     if proc.returncode != 0:
#         stderr = proc.stderr.decode("utf-8", errors="ignore")
#         raise RuntimeError(f"ffmpeg conversion failed: {stderr}")


# def segment_mean_volume(source_file, start, duration):
#     try:
#         cmd = (
#             f'ffmpeg -v error -ss {float(start)} -t {float(duration)} -i {shlex.quote(source_file)} '
#             '-af volumedetect -f null -'
#         )
#         p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
#         stderr = p.stderr.decode("utf-8", errors="ignore")
#         m = re.search(r"mean_volume:\s*([-0-9\.]+)\s*dB", stderr)
#         if m:
#             return float(m.group(1))
#     except Exception as e:
#         print(f"[energy-check] failed: {e}")
#     return None


# def schedule_file_cleanup(file_paths, delay=CLEANUP_DELAY_SEC):
#     """
#     Schedules deletion of the given files after `delay` seconds.
#     """

#     def cleanup_job(paths):
#         for p in paths:
#             try:
#                 if os.path.exists(p):
#                     os.remove(p)
#                     print(f"[cleanup] Deleted {p}")
#             except Exception as e:
#                 print(f"[cleanup] Failed to delete {p}: {e}")

#     timer = threading.Timer(delay, cleanup_job, [file_paths])
#     timer.daemon = True
#     timer.start()


# def _extract_label_from_pipeline_output(raw_output):
#     """
#     Safely extract a single label string from a HuggingFace pipeline output.
#     Fallback: 'neutral'
#     """
#     try:
#         if isinstance(raw_output, dict):
#             return raw_output.get("label", "neutral")

#         if isinstance(raw_output, (list, tuple)) and len(raw_output) > 0:
#             first = raw_output[0]
#             if isinstance(first, dict) and "label" in first:
#                 return first["label"]
#             if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], dict) and "label" in first[0]:
#                 return first[0]["label"]
#             if isinstance(first, str):
#                 return first
#     except Exception:
#         pass
#     return "neutral"


# def load_models_if_needed():
#     """Lazy-load ASR and emotion models on first request with safe fallbacks.

#     This avoids importing heavy libraries at import time and ensures the
#     server can still respond (with degraded capability) if model loading fails.
#     """
#     global asr_model, emotion_pipeline

#     # --- ASR (Whisper) ---
#     if asr_model is None:
#         model_name = os.environ.get("ASR_MODEL", ASR_MODEL_NAME) or ASR_MODEL_NAME
#         # NOTE: no forced remap from "tiny" -> "tiny.en"; we allow "tiny" or "tiny.en" explicitly.
#         try:
#             print(f"[models] Loading Whisper ASR model: {model_name} ...")
#             # import lazily
#             import whisper as _whisper
#             asr_model = _whisper.load_model(model_name)
#             print("[models] Whisper loaded.")
#         except MemoryError as me:
#             print(f"[models] MemoryError loading Whisper ({model_name}): {me}")
#             asr_model = None
#         except Exception as e:
#             print(f"[models] Failed to load Whisper model ({model_name}): {e}")
#             asr_model = None

#     # --- Emotion classifier (optional) ---
#     # If EMOTION_MODEL env var is empty string or not set to a model name, skip loading.
#     env_emotion_model = os.environ.get("EMOTION_MODEL", EMOTION_MODEL_NAME)
#     if (emotion_pipeline is None) and env_emotion_model:
#         try:
#             print(f"[models] Loading emotion classifier: {env_emotion_model} ...")
#             # lazy import the HF pipeline
#             from transformers import pipeline as _pipeline
#             # device=-1 forces CPU
#             emotion_pipeline = _pipeline("text-classification", model=env_emotion_model, top_k=1, device=-1)
#             print("[models] Emotion classifier loaded.")
#         except MemoryError as me:
#             print(f"[models] MemoryError loading emotion model ({env_emotion_model}): {me}")
#             emotion_pipeline = None
#         except Exception as e:
#             print(f"[models] Failed to load emotion classifier ({env_emotion_model}): {e}")
#             emotion_pipeline = None
#     else:
#         if not env_emotion_model:
#             print("[models] EMOTION_MODEL empty — skipping emotion pipeline.")


# @app.route("/process_meeting", methods=["POST"])
# def process_meeting():
#     # Ensure models are available (load lazily)
#     load_models_if_needed()

#     files = request.files.getlist("audio_files")
#     if not files:
#         return jsonify({"success": False, "error": "no audio_files uploaded"}), 400

#     meeting_code = request.form.get("meeting_code", "UNKNOWN")
#     speaker_map_raw = request.form.get("speaker_map", "{}")
#     try:
#         speaker_map = json.loads(speaker_map_raw)
#     except Exception:
#         speaker_map = {}

#     results = {}
#     created_file_paths = []

#     for f in files:
#         if f and allowed_file(f.filename):
#             filename = secure_filename(f.filename)
#             base_name = os.path.splitext(filename)[0]
#             save_path = os.path.join(UPLOAD_FOLDER, filename)
#             f.save(save_path)
#             created_file_paths.append(save_path)

#             wav_path = os.path.join(UPLOAD_FOLDER, f"{base_name}.wav")
#             try:
#                 convert_to_wav(save_path, wav_path)
#                 created_file_paths.append(wav_path)
#             except Exception as e:
#                 print(f"ffmpeg convert failed for {save_path}: {e}. Using original file.")
#                 wav_path = save_path

#             # If ASR model failed to load, skip transcription but keep process safe
#             if asr_model is None:
#                 print("[process] ASR model not available; skipping transcription for file:", save_path)
#                 asr_result = {"segments": []}
#             else:
#                 try:
#                     asr_result = asr_model.transcribe(wav_path, language="en")
#                 except Exception as e:
#                     print(f"Whisper failed for {wav_path}: {e}")
#                     asr_result = {"segments": []}

#             segs_with_emo = []
#             for seg in asr_result.get("segments", []):
#                 text = (seg.get("text") or "").strip()
#                 if not text or len(text) < MIN_TEXT_CHARS:
#                     continue

#                 emo_label = "neutral"
#                 if emotion_pipeline is not None:
#                     try:
#                         raw_emo = emotion_pipeline(text, top_k=1)
#                         emo_label = _extract_label_from_pipeline_output(raw_emo)
#                     except Exception as e:
#                         print(f"[emotion] pipeline failed for text segment: {e}")
#                         emo_label = "neutral"
#                 else:
#                     # quiet log to avoid spamming logs for every segment in production
#                     # print("[process] Emotion pipeline not available; using fallback 'neutral'")
#                     pass

#                 segs_with_emo.append({
#                     "start": seg.get("start", 0.0),
#                     "end": seg.get("end", 0.0),
#                     "text": text,
#                     "emotion": emo_label,
#                 })

#             results[base_name] = {
#                 "speaker": speaker_map.get(base_name, base_name),
#                 "segments": segs_with_emo,
#             }

#     # Merge and group logic unchanged
#     merged_entries = []
#     for file_id, info in results.items():
#         for seg in info["segments"]:
#             merged_entries.append({
#                 "start": seg["start"],
#                 "end": seg["end"],
#                 "speaker": info["speaker"],
#                 "text": seg["text"],
#                 "emotion": seg["emotion"],
#             })
#     merged_entries.sort(key=lambda x: x["start"])

#     groups = []
#     current = None
#     for e in merged_entries:
#         if current and e["speaker"] == current["speaker"] and e["start"] - current["end"] <= 1.0:
#             current["end"] = e["end"]
#             current["texts"].append(e["text"])
#             current["emotions"].append(e["emotion"])
#         else:
#             if current:
#                 groups.append(current)
#             current = {
#                 "start": e["start"],
#                 "end": e["end"],
#                 "speaker": e["speaker"],
#                 "texts": [e["text"]],
#                 "emotions": [e["emotion"]],
#             }
#     if current:
#         groups.append(current)

#     lines = []
#     for g in groups:
#         lines.append(f"[{g['start']:.2f}\u2013{g['end']:.2f}] {g['speaker']}:")
#         for t in g["texts"]:
#             lines.append(f"    {t}")
#         lines.append("")

#     speaker_summary = {}
#     for g in groups:
#         cnt = Counter(g["emotions"])
#         if len(cnt) > 0:
#             top, _ = cnt.most_common(1)[0]
#         else:
#             top = "neutral"
#         breakdown = ", ".join([f"{k} ({v})" for k, v in cnt.items()])
#         speaker_summary[g["speaker"]] = f"top={top} — {breakdown}"

#     lines.append("=== Emotion Summary (by speaker) ===")
#     for sp, summary in speaker_summary.items():
#         lines.append(f"{sp}: {summary}")

#     transcript_text = "\n".join(lines)

#     txt_filename = f"{meeting_code}_{uuid.uuid4().hex}.txt"
#     txt_path = os.path.join(OUTPUT_FOLDER, txt_filename)
#     with open(txt_path, "w", encoding="utf-8") as wf:
#         wf.write(transcript_text)
#     created_file_paths.append(txt_path)

#     json_filename = f"{meeting_code}_{uuid.uuid4().hex}.json"
#     json_path = os.path.join(OUTPUT_FOLDER, json_filename)
#     with open(json_path, "w", encoding="utf-8") as jf:
#         json.dump({"groups": groups, "speaker_summary": speaker_summary}, jf, indent=2)
#     created_file_paths.append(json_path)

#     schedule_file_cleanup(created_file_paths, delay=CLEANUP_DELAY_SEC)

#     return (
#         jsonify({
#             "success": True,
#             "transcript_text": transcript_text,
#             "txt_filename": txt_filename,
#             "json_filename": json_filename,
#             "files_will_be_deleted_in_sec": CLEANUP_DELAY_SEC,
#         }),
#         200,
#     )


# @app.route("/outputs/<path:filename>", methods=["GET"])
# def download_output(filename):
#     return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5001))
#     # For local dev: use Flask built-in server
#     app.run(host="0.0.0.0", port=port, debug=False)


import os
import uuid
import json
import subprocess
import shlex
import threading
import traceback
from collections import Counter
from flask import Flask, request, jsonify, send_from_directory, make_response, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

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
# Keep this for broad compatibility; we also add an after_request to guarantee headers on errors
CORS(app, resources={r"/*": {"origins": "*"}})

# Choose a default ASR model. Pick a small one for limited RAM (examples below).
# - "facebook/wav2vec2-base-960h" (English)
# - "openai/whisper-tiny" (multilingual, small)
ASR_MODEL_NAME = os.environ.get("ASR_MODEL", "facebook/wav2vec2-base-960h")
EMOTION_MODEL_NAME = os.environ.get("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")


def allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_EXT


def convert_to_wav(src_path, dst_path):
    """Convert audio to mono 16k WAV using ffmpeg. Raises RuntimeError with stderr if conversion fails."""
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            src_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            dst_path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg conversion failed: {stderr}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg and ffprobe.")


def get_audio_duration(path):
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
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
    """Lazy-load Hugging Face pipelines (ASR + emotion)."""
    global asr_pipeline, emotion_pipeline
    if asr_pipeline is None:
        try:
            print(f"[models] Loading ASR pipeline: {ASR_MODEL_NAME} ...")
            from transformers import pipeline as _pipeline
            # CPU usage (device=-1). chunk_length_s helps processing long audio without OOM.
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
    """Return list of segments: {start, end, text}. Handles multiple possible pipeline outputs."""
    segments = []
    try:
        # Case 1: pipeline returns simple dict with 'text'
        if isinstance(raw, dict) and 'text' in raw and not raw.get('chunks') and not raw.get('segments'):
            dur = get_audio_duration(wav_path) or 0.0
            segments.append({'start': 0.0, 'end': dur, 'text': raw['text']})
            return segments

        # Case 2: pipeline returns 'chunks' (some models)
        if isinstance(raw, dict) and 'chunks' in raw:
            for c in raw['chunks']:
                text = c.get('text') or c.get('chunk') or ''
                start = c.get('timestamp', [0, 0])[0] if isinstance(c.get('timestamp'), (list, tuple)) else c.get('start', 0.0)
                end = c.get('timestamp', [0, 0])[1] if isinstance(c.get('timestamp'), (list, tuple)) else c.get('end', start)
                segments.append({'start': float(start), 'end': float(end), 'text': text.strip()})
            return segments

        # Case 3: pipeline returns 'segments' or list of segments
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

    # Fallback: single segment with text if available
    try:
        text = raw.get('text') if isinstance(raw, dict) else str(raw)
        dur = get_audio_duration(wav_path) or 0.0
        segments.append({'start': 0.0, 'end': dur, 'text': (text or '').strip()})
    except Exception:
        pass

    return segments


# --- Robust CORS + error handling so errors return JSON with traceback ---
@app.after_request
def add_cors_headers(response):
    origin = ALLOWED_ORIGIN or "*"
    response.headers.setdefault("Access-Control-Allow-Origin", origin)
    response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.setdefault("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response


@app.errorhandler(Exception)
def handle_all_exceptions(e):
    app.logger.exception("Unhandled exception during request")
    tb = traceback.format_exc()
    body = {"success": False, "error": str(e), "traceback": tb}
    resp = make_response(jsonify(body), 500)
    origin = ALLOWED_ORIGIN or "*"
    resp.headers["Access-Control-Allow-Origin"] = origin
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


# Explicit OPTIONS handler (safe to keep)
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
    """Lightweight endpoint to verify uploads + ffmpeg conversion without loading HF models."""
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

    base_name = os.path.splitext(filename)[0]
    wav_path = os.path.join(UPLOAD_FOLDER, f"{base_name}.wav")
    try:
        convert_to_wav(save_path, wav_path)
    except Exception as e:
        app.logger.exception("ffmpeg conversion failed")
        return jsonify({"success": False, "error": "ffmpeg failed", "detail": str(e)}), 500

    dur = get_audio_duration(wav_path)
    return jsonify({"success": True, "saved": save_path, "wav": wav_path, "duration": dur}), 200


@app.route("/process_meeting", methods=["POST"])
def process_meeting():
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
                # conversion failed: keep original file and continue, but log the error
                app.logger.exception(f"ffmpeg convert failed for {save_path}: {e}. Using original file.")
                wav_path = save_path

            asr_result = {"segments": []}
            if asr_pipeline is None:
                app.logger.info("[process] ASR pipeline not available; skipping transcription for file: %s", save_path)
            else:
                try:
                    raw = asr_pipeline(wav_path)
                    segs = _parse_asr_output(raw, wav_path=wav_path)
                    asr_result = {"segments": segs}
                except Exception as e:
                    app.logger.exception(f"ASR pipeline failed for {wav_path}: {e}")
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
                        app.logger.exception(f"[emotion] pipeline failed for text segment: {e}")
                        emo_label = "neutral"

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

    # Merge and group logic (unchanged)
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
    app.run(host="0.0.0.0", port=port, debug=False)
