#!/usr/bin/env python3
"""
predict.py

Location: emotion_service/inference/predict.py

Supports inference from either:
  - an image + audio pair (as before), or
  - a single video file (extracts a representative frame and the audio track)

New additions:
 - load_models(...) to preload model(s) into memory
 - predict_with_loaded_models(...) to run inference repeatedly without reloading
"""
import os
import argparse
import json
import logging
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Any

import joblib
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

# optional cv2 import for robust frame reading
try:
    import cv2
except Exception:
    cv2 = None

BASE_DIR = os.path.dirname(__file__)
# -------------------------
# Paths
# -------------------------
DEFAULT_MM_PATH = os.path.join(BASE_DIR, "saved_models", "multimodal_best.pth")
DEFAULT_IF_PATH = os.path.join(BASE_DIR, "results", "anomaly", "isolation_forest.joblib")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("predict")


# -------------------------
# Preprocessing (match training!)
# -------------------------
IMG_SIZE = (224, 224)
AUDIO_SR = 16000
N_MELS = 128

image_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# lazy-built ResNet18 backbone to produce 512-d image features
_IMAGE_BACKBONE = None
_resnet_weights_enum = getattr(models, "ResNet18_Weights", None)


def _build_image_backbone(device: torch.device):
    """
    Returns a model (conv+avgpool) that produces (B,512,1,1) outputs for 224x224 inputs.
    We strip the final fc layer of ResNet18.
    """
    global _IMAGE_BACKBONE
    if _IMAGE_BACKBONE is not None:
        return _IMAGE_BACKBONE

    try:
        if _resnet_weights_enum is not None:
            backbone_full = models.resnet18(weights=_resnet_weights_enum.DEFAULT)
        else:
            backbone_full = models.resnet18(pretrained=True)
    except Exception as e:
        # fallback to uninitialized resnet if pretrained weights not available
        try:
            backbone_full = models.resnet18(pretrained=False)
            logger.warning("Could not load pretrained ResNet18 weights; using uninitialized ResNet18 as fallback.")
        except Exception as e2:
            raise RuntimeError("Failed to build ResNet18 backbone. Install torchvision with model weights or provide 512-d image features.") from e2

    # remove final fc; keep up to avgpool (last layer before fc)
    backbone = torch.nn.Sequential(*list(backbone_full.children())[:-1])
    backbone.to(device)
    backbone.eval()
    _IMAGE_BACKBONE = backbone
    return _IMAGE_BACKBONE


def preprocess_image(image_path: str, device: torch.device = None) -> torch.Tensor:
    """
    Convert image -> (1, 512) feature vector using a ResNet18 backbone.
    Device should match the multimodal model device for efficient extraction.
    Returns a CPU tensor (1,512) to keep downstream code consistent.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    img = Image.open(image_path).convert("RGB")
    x = image_transform(img).unsqueeze(0).to(device)  # (1,3,224,224)

    backbone = _build_image_backbone(device)
    with torch.no_grad():
        feat = backbone(x)   # (1, 512, 1, 1)
        feat = feat.view(feat.shape[0], -1)  # (1, 512)
    return feat.cpu()


def preprocess_audio(audio_path: str, sr: int = AUDIO_SR) -> torch.Tensor:
    """
    Convert an audio file into a (1, 128) vector by computing mel spectrogram
    and averaging over the time axis. This matches audio_encoder.in_features=128.
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)  # (n_mels, time)
    vec = np.mean(mel_db, axis=1)  # (n_mels,)
    t = torch.tensor(vec).unsqueeze(0).float()  # (1, 128)
    return t


# -------------------------
# Video utilities
# -------------------------

def _ensure_ffmpeg_available():
    """Return path to ffmpeg or raise a helpful error."""
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        return ffmpeg_exe
    raise RuntimeError(
        "ffmpeg not found on PATH. Install ffmpeg or provide image+audio separately.\n"
        "On Ubuntu: sudo apt-get install ffmpeg\n"
        "On macOS (homebrew): brew install ffmpeg"
    )


def extract_frame_and_audio(video_path: str, target_frame_time: float = None):
    """
    Robust extraction of one representative frame and 16k mono PCM audio.
    Tries:
      - cv2 (fast)
      - ffmpeg (-ss before -i, then -ss after -i)
      - remux to temporary MP4 and extract frame from that (fixes some EBML/header issues)
    Always uses ffmpeg for audio extraction and raises on audio failure.
    Returns tuple (frame_img_path, audio_wav_path). Caller should remove temp files.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    frame_img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
    audio_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    tmp_remux_mp4 = None

    used_cv2 = False
    # 1) Try OpenCV if available
    if cv2 is not None:
        try:
            cap = cv2.VideoCapture(video_path)
            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                if total_frames > 0 and fps > 0:
                    mid_frame_idx = total_frames // 2
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        Image.fromarray(frame).save(frame_img_path)
                        used_cv2 = True
                    else:
                        # try first frame as fallback
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            Image.fromarray(frame).save(frame_img_path)
                            used_cv2 = True
                        else:
                            logger.warning("cv2 failed to read frame (no frames/fps) from %s", video_path)
                else:
                    # unknown fps/frames: try first frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        Image.fromarray(frame).save(frame_img_path)
                        used_cv2 = True
                    else:
                        logger.warning("cv2 could not read a usable frame from %s", video_path)
            finally:
                cap.release()
        except Exception as e:
            logger.warning("OpenCV attempt raised exception for %s: %s", video_path, e)

    # 2) If cv2 didn't produce a frame, attempt a few ffmpeg strategies
    def _run_ffmpeg_cmd(cmd):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode, proc.stdout, proc.stderr

    if not used_cv2:
        ffmpeg = None
        try:
            ffmpeg = _ensure_ffmpeg_available()
        except Exception as e:
            logger.warning("ffmpeg not on PATH: %s", e)
            ffmpeg = None

        if ffmpeg:
            # choose time t
            duration = None
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                    capture_output=True, text=True, check=True
                )
                duration = float(probe.stdout.strip()) if probe.stdout.strip() else None
            except Exception:
                duration = None

            t = (duration / 2.0) if (duration and target_frame_time is None) else (target_frame_time or 0)

            # Try -ss before -i (fast seek) then -ss after -i (accurate seek)
            attempts = [
                [ffmpeg, "-y", "-ss", str(t), "-i", video_path, "-frames:v", "1", "-q:v", "2", frame_img_path],
                [ffmpeg, "-y", "-i", video_path, "-ss", str(t), "-frames:v", "1", "-q:v", "2", frame_img_path],
            ]

            success = False
            for cmd in attempts:
                try:
                    rc, out, err = _run_ffmpeg_cmd(cmd)
                    if rc == 0 and os.path.exists(frame_img_path) and os.path.getsize(frame_img_path) > 0:
                        success = True
                        logger.info("ffmpeg frame extraction succeeded (cmd: %s)", " ".join(cmd))
                        break
                    else:
                        logger.debug("ffmpeg frame attempt cmd failed rc=%s stderr=%s", rc, err.strip())
                except Exception as e:
                    logger.debug("ffmpeg frame attempt exception: %s", e)

            # If still not success, try remuxing to mp4 and extracting from that (often fixes EBML header problems)
            if not success:
                try:
                    tmp_remux_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                    remux_cmd = [ffmpeg, "-y", "-i", video_path, "-c", "copy", tmp_remux_mp4]
                    rc, out, err = _run_ffmpeg_cmd(remux_cmd)
                    if rc == 0 and os.path.exists(tmp_remux_mp4) and os.path.getsize(tmp_remux_mp4) > 0:
                        # try extracting from remuxed file
                        for cmd in [
                            [ffmpeg, "-y", "-ss", str(t), "-i", tmp_remux_mp4, "-frames:v", "1", "-q:v", "2", frame_img_path],
                            [ffmpeg, "-y", "-i", tmp_remux_mp4, "-ss", str(t), "-frames:v", "1", "-q:v", "2", frame_img_path],
                        ]:
                            rc2, out2, err2 = _run_ffmpeg_cmd(cmd)
                            if rc2 == 0 and os.path.exists(frame_img_path) and os.path.getsize(frame_img_path) > 0:
                                success = True
                                logger.info("ffmpeg frame extraction from remuxed MP4 succeeded.")
                                break
                            else:
                                logger.debug("ffmpeg-from-mp4 attempt failed rc=%s stderr=%s", rc2, err2.strip())
                    else:
                        logger.debug("ffmpeg remux failed rc=%s stderr=%s", rc, err.strip())
                except Exception as e:
                    logger.debug("remux attempt exception: %s", e)

            # If still not success, create a tiny black image as fallback
            if not success:
                try:
                    Image.new("RGB", (32, 32), (0, 0, 0)).save(frame_img_path)
                    logger.warning("Could not extract a frame; created a tiny black fallback image.")
                except Exception:
                    logger.exception("Failed to write fallback image for %s", video_path)

    # 3) Extract audio via ffmpeg (required)
    try:
        ffmpeg = _ensure_ffmpeg_available()
        audio_cmd = [
            ffmpeg, "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", str(AUDIO_SR), "-ac", "1", audio_wav_path
        ]
        proc = subprocess.run(audio_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error("ffmpeg audio extraction failed rc=%s stderr=%s", proc.returncode, proc.stderr.strip())
            raise RuntimeError(f"ffmpeg audio extraction failed: {proc.stderr.strip()}")
    except Exception:
        # audio is critical for your pipeline — surface the real error
        logger.exception("Audio extraction failed for %s", video_path)
        # cleanup tmp_remux_mp4 if present
        if tmp_remux_mp4 and os.path.exists(tmp_remux_mp4):
            try:
                os.remove(tmp_remux_mp4)
            except Exception:
                pass
        raise

    # cleanup remux file (we already used it)
    if tmp_remux_mp4 and os.path.exists(tmp_remux_mp4):
        try:
            os.remove(tmp_remux_mp4)
        except Exception:
            pass

    return frame_img_path, audio_wav_path



# -------------------------
# USER ACTION REQUIRED: Build your model architecture here if you only have a checkpoint (state_dict).
# -------------------------

def build_model(checkpoint_meta: dict = None, inferred: dict = None):
    """
    Construct MultimodalNet by trying (in order):
      1) import emotion_service.training.train_multimodal
      2) import train_multimodal
      3) add repo root to sys.path and retry imports
      4) load training/train_multimodal.py by file path using importlib.util

    If none succeed, raises ImportError with guidance.
    """
    inferred = inferred or {}
    image_input_shape = inferred.get("image_input_shape", (3, 224, 224))
    audio_feat_dim = inferred.get("audio_feat_dim", 128)
    embedding_dim = inferred.get("embedding_dim", 256)
    num_classes = None
    if checkpoint_meta and isinstance(checkpoint_meta, dict) and "class_labels" in checkpoint_meta:
        num_classes = len(checkpoint_meta.get("class_labels"))

    mm_cls = None
    tried = []

    # helper to try import by module name
    def try_import(mod_name):
        try:
            m = __import__(mod_name, fromlist=["*"])
            return m
        except Exception as e:
            tried.append((mod_name, str(e)))
            return None

    # 1) try package import
    mod = try_import("emotion_service.training.train_multimodal")
    if mod and hasattr(mod, "MultimodalNet"):
        mm_cls = getattr(mod, "MultimodalNet")

    # 2) try plain module name
    if mm_cls is None:
        mod = try_import("train_multimodal")
        if mod and hasattr(mod, "MultimodalNet"):
            mm_cls = getattr(mod, "MultimodalNet")

    # 3) if still not found, add repo root (parent of emotion_service) to sys.path and retry
    if mm_cls is None:
        try:
            import sys
            from pathlib import Path
            this_file = Path(__file__).resolve()
            # repo_root is two levels up: .../emotion_service/inference/predict.py -> repo_root
            repo_root = this_file.parents[1]
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            # now try imports again
            mod = try_import("emotion_service.training.train_multimodal")
            if mod and hasattr(mod, "MultimodalNet"):
                mm_cls = getattr(mod, "MultimodalNet")
            else:
                mod = try_import("train_multimodal")
                if mod and hasattr(mod, "MultimodalNet"):
                    mm_cls = getattr(mod, "MultimodalNet")
        except Exception as e:
            tried.append(("sys_path_insert", str(e)))

    # 4) final fallback: load the training file by absolute path using importlib.util
    if mm_cls is None:
        try:
            import importlib.util
            from pathlib import Path
            this_file = Path(__file__).resolve()
            training_path = this_file.parents[1] / "training" / "train_multimodal.py"
            if training_path.exists():
                spec = importlib.util.spec_from_file_location("train_multimodal_local", str(training_path))
                tm = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(tm)  # type: ignore
                if hasattr(tm, "MultimodalNet"):
                    mm_cls = getattr(tm, "MultimodalNet")
                    tried.append(("file_load", f"loaded from {training_path}"))
                else:
                    tried.append(("file_load", f"file present but MultimodalNet not found in {training_path}"))
            else:
                tried.append(("file_load", f"{training_path} not found"))
        except Exception as e:
            tried.append(("file_load_exc", str(e)))

    if mm_cls is None:
        raise ImportError(
            "Cannot import MultimodalNet from training script.\n"
            "Make sure train_multimodal.py exists in emotion_service/training and that it defines MultimodalNet.\n"
            "Attempts/tried: " + str(tried) + "\n"
            "Quick fixes:\n"
            "  - Run from project root directory (the parent of 'emotion_service') so imports work normally.\n"
            "  - Or re-save the full model object during training with torch.save(model, 'multimodal_best.pth').\n"
            "  - Or paste the MultimodalNet class into this file (I can do that for you if you prefer).\n"
        )

    # Build the model instance with inferred args
    kwargs = dict(
        image_input_shape=tuple(image_input_shape),
        audio_feat_dim=int(audio_feat_dim),
        embedding_dim=int(embedding_dim),
        num_classes=int(num_classes) if num_classes is not None else None,
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    model = mm_cls(**kwargs)
    return model



# -------------------------
# Loading utilities
# -------------------------

def load_multimodal_model(path: str, device: torch.device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Multimodal model not found at: {path}")

    logger.info(f"Loading multimodal model from {path} ...")

    loaded = torch.load(path, map_location=device)

    # Case A: saved entire Module object
    if not isinstance(loaded, dict):
        logger.info("Loaded object is a PyTorch Module (full model). Using it directly.")
        model = loaded
        model.to(device)
        model.eval()
        return model, {}

    # Case B: checkpoint dict
    ck = loaded
    logger.info("Loaded a checkpoint dict. Attempting to extract model_state_dict and metadata.")

    if "model_state_dict" not in ck:
        # The dict may be a state_dict itself (no wrapper). Try to detect.
        if all(isinstance(k, str) for k in ck.keys()):
            state_dict = ck
            meta = {}
            logger.info("Checkpoint looks like a plain state_dict. build_model() must know the architecture.")
        else:
            raise RuntimeError("Checkpoint format not recognized. Provide either a saved Module or a checkpoint dict with 'model_state_dict'.")
    else:
        state_dict = ck["model_state_dict"]
        meta = {k: v for k, v in ck.items() if k != "model_state_dict"}

    # Try to infer some dimensions from the state_dict to help build_model
    inferred = {}
    for k, v in state_dict.items():
        if k.startswith("audio_encoder") and k.endswith(".weight") and v.ndim == 2:
            inferred_audio_in = v.shape[1]
            inferred["audio_feat_dim"] = int(inferred_audio_in)
            break
    for k, v in state_dict.items():
        if k.startswith("image_encoder") and k.endswith(".weight") and v.ndim == 2:
            inferred_image_in = v.shape[1]
            if inferred_image_in <= 4:
                inferred["image_input_shape"] = (inferred_image_in, 224, 224)
            else:
                inferred["image_input_shape"] = (int(inferred_image_in),)
            break
    for k, v in state_dict.items():
        if k.startswith("fusion") and k.endswith(".weight") and v.ndim == 2:
            inferred_embed = v.shape[0]
            inferred["embedding_dim"] = int(inferred_embed)
            break

    logger.info(f"Inferred dims for model construction: {inferred}")

    logger.info("Reconstructing model architecture using build_model() - please ensure implementation matches training code.")
    model = build_model(meta, inferred)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        logger.warning("Direct load_state_dict failed. Attempting to strip 'module.' prefixes (DataParallel compatibility). Error: %s", e)
        new_state = {}
        for k, v in state_dict.items():
            nk = k.replace("module.", "") if k.startswith("module.") else k
            new_state[nk] = v
        model.load_state_dict(new_state)

    model.to(device)
    model.eval()
    return model, meta


def load_isolation_forest(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"IsolationForest model not found at: {path}")

    logger.info(f"Loading IsolationForest from {path} ...")

    # Try joblib first
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        logger.warning("joblib.load failed for IsolationForest: %s", e)

    # Try pickle as a fallback
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        # Helpful, actionable error
        raise RuntimeError(
            f"""Failed to load IsolationForest from the provided file. This often happens when the model was serialized
with a different scikit-learn / numpy version than the one currently running.

Workarounds:
  1) Re-save the IsolationForest in the original environment using
     joblib.dump(if_model, 'isolation_forest.joblib') and move that file here.
  2) If you cannot access the original environment, re-train the IsolationForest in this environment (fast).
  3) Provide the saved numpy array of known-good embeddings and re-fit a fresh IsolationForest here.

Underlying loading error: {e}
"""
        )


# -------------------------
# Helper: extract embeddings + logits
# -------------------------


def _try_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.debug(f"call failed for {fn}: {e}")
        return None


def get_embeddings_and_logits(model, img_tensor: torch.Tensor, audio_tensor: torch.Tensor, device: torch.device):
    """
    Deterministic extraction for your inspected MultimodalNet.

    Returns:
      embeddings: the PRE-FUSION concatenated raw vector [image_raw (512) || audio_raw (128)] shape (B,640)
                  (this keeps compatibility with your IsolationForest training)
      logits:     the output of the fusion head (model.fusion) computed from encoder outputs
    """
    img = img_tensor.to(device)
    aud = audio_tensor.to(device)

    def _flatten_if_needed(x: torch.Tensor) -> torch.Tensor:
        # flatten spatial/time dims so we get (B, feat)
        if x.ndim == 1:
            return x.unsqueeze(0)
        if x.ndim == 2:
            return x
        # collapse all dims except batch
        return x.view(x.shape[0], -1)

    # Build raw precomputed feature vectors (the "raw" features used at training)
    raw_img_feat = _flatten_if_needed(img)  # expected (B,512)
    raw_aud_feat = _flatten_if_needed(aud)  # expected (B,128)

    # Safe guards: if dims are transposed (e.g., (1,512,1,1)), flatten still works above.
    # Now we will try to compute encoder outputs (img_enc, aud_enc) that the fusion head expects.
    img_enc = None
    aud_enc = None
    logits = None

    # Helper to try calls and swallow exceptions
    def safe(fn, *a, **k):
        try:
            with torch.no_grad():
                return fn(*a, **k)
        except Exception as e:
            logger.debug("safe call failed for %s: %s", getattr(fn, "__name__", str(fn)), e)
            return None

    # 1) Compute encoder outputs if modules exist
    if hasattr(model, "image_encoder"):
        # image_encoder expects a (B,512) input for your saved model; prefer raw_img_feat if dims match
        try:
            # try raw input first
            if raw_img_feat.ndim == 2:
                trial = safe(model.image_encoder, raw_img_feat)
                if trial is not None:
                    img_enc = trial
            # if that didn't work, try passing img (in case preprocess_image changed)
            if img_enc is None:
                trial = safe(model.image_encoder, img)
                if trial is not None:
                    img_enc = trial
        except Exception:
            img_enc = None

    if hasattr(model, "audio_encoder"):
        try:
            if raw_aud_feat.ndim == 2:
                trial = safe(model.audio_encoder, raw_aud_feat)
                if trial is not None:
                    aud_enc = trial
            if aud_enc is None:
                trial = safe(model.audio_encoder, aud)
                if trial is not None:
                    aud_enc = trial
        except Exception:
            aud_enc = None

    # 2) Compute logits from fusion head using encoder outputs (preferred)
    # The fusion head expects concatenation of encoder outputs (e.g., img_enc (256) + aud_enc (256) -> 512)
    if hasattr(model, "fusion"):
        # If encoder outputs exist, try to use them to compute logits
        if img_enc is not None and aud_enc is not None:
            try:
                enc_concat = torch.cat([img_enc, aud_enc], dim=1)
                if hasattr(model, "fusion_norm"):
                    enc_concat = model.fusion_norm(enc_concat)
                logits = safe(model.fusion, enc_concat)
            except Exception as e:
                logger.debug("fusion(enc_concat) failed: %s", e)

        # If encoder outputs aren't available, try building a placeholder fusion input:
        # some training pipelines used the raw concatenation then internal layers reduced it; try fusion on transformed raw if possible
        if logits is None:
            try:
                raw_concat = torch.cat([raw_img_feat, raw_aud_feat], dim=1)
                # fusion likely expects a smaller vector (512), so we don't expect this to usually succeed,
                # but try in case training used raw concat directly.
                fusion_in = raw_concat
                if hasattr(model, "fusion_norm"):
                    fusion_in = model.fusion_norm(fusion_in)
                logits = safe(model.fusion, fusion_in)
            except Exception as e:
                logger.debug("fusion(raw_concat) attempt failed: %s", e)

    # 3) If logits still None, try direct model(img,aud) forward (some models return logits only)
    if logits is None:
        out = safe(model, raw_img_feat, raw_aud_feat) or safe(model, img, aud) or safe(model, img)
        if out is not None:
            if isinstance(out, tuple) and len(out) >= 2:
                # model returned (embeddings, logits) — use logits directly, and preserve embeddings if provided
                emb_out, logits_out = out[0], out[1]
                # Determine embedding to return to IsolationForest:
                # prefer raw pre-fusion concat (raw_img_feat || raw_aud_feat) if dims match training
                emb_for_if = torch.cat([raw_img_feat, raw_aud_feat], dim=1)
                return emb_for_if, logits_out
            elif isinstance(out, torch.Tensor):
                # out looks like logits
                logits = out

    # 4) Build the embeddings expected by the IsolationForest: the PRE-FUSION raw concatenation
    try:
        embeddings_for_if = torch.cat([raw_img_feat, raw_aud_feat], dim=1)  # (B, 640 expected)
    except Exception as e:
        # final fallback: concatenate flattened versions
        embeddings_for_if = torch.cat([_flatten_if_needed(raw_img_feat), _flatten_if_needed(raw_aud_feat)], dim=1)

    # 5) If we still don't have logits computed, try a best-effort: if model has classifier or fusion, try feeding encoder outputs or embeddings
    if logits is None:
        # try classifier attribute
        if hasattr(model, "classifier") and (img_enc is not None or aud_enc is not None):
            try:
                emb_in = img_enc if img_enc is not None else _flatten_if_needed(raw_img_feat)
                logits = safe(model.classifier, emb_in)
            except Exception:
                logits = None

        # try fusion with encoder outputs again
        if logits is None and img_enc is not None and aud_enc is not None and hasattr(model, "fusion"):
            try:
                enc_concat = torch.cat([img_enc, aud_enc], dim=1)
                if hasattr(model, "fusion_norm"):
                    enc_concat = model.fusion_norm(enc_concat)
                logits = safe(model.fusion, enc_concat)
            except Exception:
                logits = None

    # 6) Final check — if logits is available return (pre-fusion concat, logits)
    if logits is not None:
        # ensure embeddings_for_if is on same device as logits for downstream conversion
        return embeddings_for_if.to(logits.device), logits

    # 7) Last resort: try to interpret model(img,aud) output as dict with keys
    out2 = safe(model, raw_img_feat, raw_aud_feat) or safe(model, img, aud)
    if isinstance(out2, dict):
        emb = out2.get("embeddings", None) or out2.get("features", None)
        logits = out2.get("logits", None)
        if logits is not None:
            # choose embeddings_for_if for IF compatibility
            return embeddings_for_if.to(logits.device), logits

    # If we reach here, we couldn't construct logits — raise informative error
    raise RuntimeError(
        "Could not extract embeddings/logits from the loaded multimodal model.\n"
        "Tried: encoder outputs (image_encoder/audio_encoder), fusion head, direct forward.\n"
        "Check that your build_model() matches the training architecture and that preprocess_image/audio produce the expected raw features.\n"
    )


# -------------------------
# New: load models + predict with loaded models
# -------------------------

def load_models(multimodal_source: Any, if_source: Any, device: Optional[torch.device] = None) -> Tuple[Any, dict, Any]:
    """
    Load and return (model, meta, isolation_forest_model).

    multimodal_source:
      - path string -> load from disk via load_multimodal_model(path, device)
      - (model, meta) tuple -> returned as-is
      - torch.nn.Module instance -> returned as (model, {})

    if_source:
      - path string -> load_isolation_forest(path)
      - object -> returned as-is (assumed to implement predict/decision_function)
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # multimodal model
    if isinstance(multimodal_source, (list, tuple)) and len(multimodal_source) == 2:
        model, meta = multimodal_source
    elif isinstance(multimodal_source, torch.nn.Module):
        model = multimodal_source
        meta = {}
    elif isinstance(multimodal_source, str):
        model, meta = load_multimodal_model(multimodal_source, device)
    else:
        raise ValueError("Unsupported multimodal_source type for load_models()")

    # isolation forest
    if isinstance(if_source, str) and if_source:
        if_model = load_isolation_forest(if_source)
    elif if_source is None or if_source == "":
        raise ValueError("if_source is empty; provide a path or an IsolationForest instance")
    else:
        if_model = if_source

    return model, meta, if_model


def predict_with_loaded_models(image_path: str, audio_path: str, model: Any, if_model: Any,
                                meta: Optional[dict] = None, labels: Optional[list] = None,
                                device: Optional[torch.device] = None) -> dict:
    """
    Run inference using already-loaded model and isolation-forest objects.
    Returns same result dict as predict().
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # preprocess
    img_tensor = preprocess_image(image_path, device=device)  # (1,512) CPU tensor
    audio_tensor = preprocess_audio(audio_path)  # (1,128) CPU tensor

    embeddings, logits = get_embeddings_and_logits(model, img_tensor, audio_tensor, device)

    if logits is None:
        raise RuntimeError("Model did not provide logits. Emotion probabilities cannot be computed.")

    probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    emb_np = embeddings.cpu().numpy().reshape(1, -1)

    score = float(if_model.decision_function(emb_np)[0])
    pred = int(if_model.predict(emb_np)[0])
    flag = "anomaly" if pred == -1 else "normal"

    # labels: try meta first
    if labels is None:
        if isinstance(meta, dict) and "class_labels" in meta:
            labels = meta["class_labels"]
        else:
            labels = [f"label_{i}" for i in range(len(probs))]

    emotion_probs = dict(zip(labels, [float(p) for p in probs]))

    result = {
        "emotion_probs": emotion_probs,
        "anomaly_score": score,
        "anomaly_flag": flag,
    }

    return result


# -------------------------
# Backwards-compatible predict()
# -------------------------


def predict(image_path: str, audio_path: str, multimodal_path: Any, if_path: Any, labels=None, device=None):
    """
    Backwards-compatible wrapper. multimodal_path/if_path may be:
      - file paths (strings) -> they will be loaded
      - loaded objects (model, or (model,meta), or if_model)
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # multimodal argument handling
    if isinstance(multimodal_path, (list, tuple)) and len(multimodal_path) == 2 and isinstance(multimodal_path[0], torch.nn.Module):
        model, meta = multimodal_path
    elif isinstance(multimodal_path, torch.nn.Module):
        model, meta = multimodal_path, {}
    elif isinstance(multimodal_path, str):
        model, meta = load_multimodal_model(multimodal_path, device)
    else:
        raise ValueError("multimodal_path must be a path, a torch Module, or (model, meta) tuple")

    # if_path handling
    if isinstance(if_path, str) and if_path:
        if_model = load_isolation_forest(if_path)
    elif if_path is None or if_path == "":
        raise ValueError("if_path cannot be empty for predict() - pass a loaded IF or a path")
    else:
        if_model = if_path

    return predict_with_loaded_models(image_path, audio_path, model, if_model, meta=meta, labels=labels, device=device)


# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Run multimodal emotion + anomaly prediction")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to input video (will extract frame+audio)")
    group.add_argument("--image", help="Path to input image")
    parser.add_argument("--audio", help="Path to input audio (wav). Required when using --image")

    parser.add_argument("--multimodal", default=DEFAULT_MM_PATH, help="Path to multimodal .pth file")
    parser.add_argument("--isolation", default=DEFAULT_IF_PATH, help="Path to isolation_forest.joblib")
    parser.add_argument("--labels", default=None, help="JSON file with list of emotion labels in order")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Device to run on")

    args = parser.parse_args()

    labels = None
    if args.labels:
        with open(args.labels, "r") as f:
            labels = json.load(f)

    device = torch.device("cuda" if args.device == "cuda" else "cpu") if args.device else None

    temp_files = []
    try:
        if args.video:
            # extract frame + audio
            logger.info(f"Extracting frame+audio from video: {args.video}")
            img_path, audio_path = extract_frame_and_audio(args.video)
            temp_files.extend([img_path, audio_path])
        else:
            # image mode
            if not args.audio:
                parser.error("--audio is required when using --image")
            img_path = args.image
            audio_path = args.audio

        res = predict(img_path, audio_path, args.multimodal, args.isolation, labels=labels, device=device)

        print(json.dumps(res, indent=2))

    finally:
        # cleanup temp files if any
        for tf in temp_files:
            try:
                os.remove(tf)
            except Exception:
                pass


if __name__ == "__main__":
    main()
