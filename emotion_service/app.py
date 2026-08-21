"""
Real-time emotion inference server for meetings (Socket.IO / WebSocket).

Pipeline
--------
audio_chunk  → PCM → Wav2Vec2 embedding → _AUDIO_BUFFER[pid]  (inference clock)
frame        → JPEG/PNG → MediaPipe embedding → _FACE_BUFFER[pid]  (enrichment)
av_chunk     → demuxes to the two paths above

Modality freshness
------------------
A modality is active only if its last successful embedding arrived within
MODALITY_STALE_SEC and its buffer is non-empty.  When a participant
mutes/disables a modality via "participant.media_state", the handler
zeroes the timestamp and clears the buffer immediately — no stale-window lag.

Frame-rate control
------------------
Frames are gated by _FRAME_ACCEPT_AFTER[pid] (monotonic clock), dropped
before any decode work.  targetFps is advertised to the client on connect.
A "backpressure" event is emitted when the face executor queue is saturated.

Threading
---------
_state_lock protects all shared dicts across the event loop and APScheduler GC.
_EMOTION_HISTORY is accessed only from per-pid pump coroutines in the event
loop — no additional lock required; preserve this invariant on refactoring.
_face_pending is protected by _face_pending_lock for cross-thread safety.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import sys
import threading
import time
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype.*",
    category=UserWarning,
    module="google.protobuf.*",
)

import cv2
import numpy as np
import soundfile as sf
import socketio
import torch
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException
from observability.stats import (
    stats_router,
    set_active_participant_provider,
    set_tracker,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import embeddings.extract_embeddings_data as _emo_mod
from embeddings.extract_embeddings_data import (
    extract_face_embedding,
    load_models as load_extractor_models,
)
import anomaly.train_anomaly as _anomaly_mod  # noqa: F401
from inference.predict import EmotionPredictor

if not hasattr(_anomaly_mod, "ModalityAnomalyModel"):
    raise ImportError(
        "ModalityAnomalyModel not found in anomaly.train_anomaly — "
        "ensure the class is defined at module level in that file."
    )


_CONFIG_PATH = ROOT / "config" / "config.json"
with open(_CONFIG_PATH) as _f:
    _CFG = json.load(_f)

SEQ_LEN: int = _CFG["processing"]["seq_len"]
FACE_DIM: int = _CFG["processing"]["face_dim"]
AUDIO_DIM: int = _CFG["processing"]["audio_dim"]
SAMPLE_RATE: int = _CFG["processing"]["sample_rate"]
AUDIO_WINDOW_SEC: float = _CFG["processing"]["audio_window_sec"]

AUDIO_WINDOW_SAMPLES: int = int(SAMPLE_RATE * AUDIO_WINDOW_SEC)

_CLASS_NAMES: list[str] = _CFG["misc"]["class_names"]
_NEUTRAL_LABEL: str = _CLASS_NAMES[-1]


SMOOTHING_ALPHA: float = _CFG["inference"]["smoothing_alpha"]
CONFIDENCE_THRESHOLD: float = _CFG["inference"]["confidence_threshold"]
EMOTION_HISTORY_TTL: float = _CFG["inference"]["emotion_history_ttl"]

MODALITY_STALE_SEC: float = _CFG["inference"]["modality_stale_sec"]

MIN_INFERENCE_INTERVAL: float = _CFG["inference"]["min_inference_interval"]

TARGET_CLIENT_FPS: int = _CFG["inference"]["target_client_fps"]
FRAME_MIN_INTERVAL_SEC: float = 1.0 / TARGET_CLIENT_FPS

BACKPRESSURE_QUEUE_DEPTH: int = 3

MAX_FRAME_SIZE: int = 4 * 1024 * 1024
MAX_AUDIO_SIZE: int = 2 * 1024 * 1024

BUFFER_TTL: int = 90

MODALITY_AUDIO_ONLY: int = 0
MODALITY_VIDEO_ONLY: int = 1
MODALITY_BOTH: int = 2

MODALITY_NAMES: dict[int, str] = {
    MODALITY_AUDIO_ONLY: "audio_only",
    MODALITY_VIDEO_ONLY: "video_only",
    MODALITY_BOTH: "both",
}

DEVICE: str = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("emotion_ws")


class _LatencyTracker:
    """Rolling per-modality latency tracker.

    Accumulates inference latency samples in a fixed-size deque and logs
    P50 / P90 / P95 / min / max every ``report_interval`` seconds.
    Thread-safe via an internal lock; all public methods may be called from
    any thread or coroutine.

    Args:
        window: Maximum number of samples retained (oldest are evicted).
        report_interval: Seconds between periodic log lines.
    """

    _MODALITIES = ("audio_only", "video_only", "both", "overall")

    def __init__(self, window: int = 500, report_interval: float = 60.0) -> None:
        self._window = window
        self._report_interval = report_interval
        self._lock = threading.Lock()
        self._samples: dict[str, deque] = {
            m: deque(maxlen=window) for m in self._MODALITIES
        }
        self._last_report = time.monotonic()

    def record(self, latency_ms: float, modality: str) -> None:
        """Record a single inference latency sample.

        Args:
            latency_ms: End-to-end inference latency in milliseconds.
            modality: One of ``audio_only``, ``video_only``, or ``both``.
        """
        with self._lock:
            if modality in self._samples:
                self._samples[modality].append(latency_ms)
            self._samples["overall"].append(latency_ms)
            if time.monotonic() - self._last_report >= self._report_interval:
                self._report()
                self._last_report = time.monotonic()

    def _percentile(self, data: list[float], p: float) -> float:
        if not data:
            return float("nan")
        k = (len(data) - 1) * p / 100
        lo, hi = int(k), min(int(k) + 1, len(data) - 1)
        return data[lo] + (data[hi] - data[lo]) * (k - lo)

    def _report(self) -> None:
        for mod, dq in self._samples.items():
            if not dq:
                continue
            vals = sorted(dq)
            logger.info(
                "latency_stats mod=%-10s n=%-4d min=%.1f p50=%.1f p90=%.1f "
                "p95=%.1f max=%.1f  (ms)",
                mod,
                len(vals),
                vals[0],
                self._percentile(vals, 50),
                self._percentile(vals, 90),
                self._percentile(vals, 95),
                vals[-1],
            )

    def report_now(self) -> None:
        """Force an immediate stats log regardless of the report interval."""
        with self._lock:
            self._report()
            self._last_report = time.monotonic()


_latency_tracker = _LatencyTracker(window=500, report_interval=60.0)


_face_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="face")
_audio_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="audio")
_inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")

_face_pending: int = 0
_face_pending_lock = threading.Lock()


predictor: Optional[EmotionPredictor] = None

service_ready: bool = False
_norm_stats: Optional[dict] = None


def _load_norm_stats() -> Optional[dict]:
    """Load per-feature mean/std arrays from extracted_dataset/norm_stats.npz.

    Returns None if the file is absent; inference proceeds without normalisation.
    """
    p = ROOT / "extracted_dataset" / "norm_stats.npz"
    if not p.exists():
        logger.warning("norm_stats.npz not found — running WITHOUT normalisation")
        return None
    stats = dict(np.load(p))
    logger.info("Norm stats loaded from %s", p)
    return stats


def _norm_face(xf: np.ndarray, fm: np.ndarray) -> np.ndarray:
    """Z-score normalise face embeddings and zero-out padding time steps.

    Args:
        xf: Raw face embedding array of shape (SEQ_LEN, FACE_DIM).
        fm: Binary mask of shape (SEQ_LEN,); 1 = real frame, 0 = padding.

    Returns:
        Normalised array of shape (SEQ_LEN, FACE_DIM), float32.
        Returns xf unchanged if norm stats are unavailable.
    """
    if _norm_stats is None:
        return xf
    mean = _norm_stats["Xf_mean"].reshape(1, -1)
    std = _norm_stats["Xf_std"].reshape(1, -1)
    out = (xf - mean) / std
    out = out * fm.astype(np.float32)[:, None]
    return out.astype(np.float32)


def _norm_audio(xa: np.ndarray, am: np.ndarray) -> np.ndarray:
    """Z-score normalise audio embeddings and zero-out padding time steps.

    Args:
        xa: Raw audio embedding array of shape (SEQ_LEN, AUDIO_DIM).
        am: Binary mask of shape (SEQ_LEN,); 1 = real frame, 0 = padding.

    Returns:
        Normalised array of shape (SEQ_LEN, AUDIO_DIM), float32.
        Returns xa unchanged if norm stats are unavailable.
    """
    if _norm_stats is None:
        return xa
    mean = _norm_stats["Xa_mean"].reshape(1, -1)
    std = _norm_stats["Xa_std"].reshape(1, -1)
    out = (xa - mean) / std
    out = out * am.astype(np.float32)[:, None]
    return out.astype(np.float32)


_state_lock = threading.Lock()

_FACE_BUFFER: dict[str, deque] = defaultdict(lambda: deque(maxlen=SEQ_LEN))
_AUDIO_BUFFER: dict[str, deque] = defaultdict(lambda: deque(maxlen=SEQ_LEN))

_LATEST_FRAME: dict[str, bytes] = {}
_LATEST_AUDIO: dict[str, np.ndarray] = {}

_MODALITY_TIMESTAMPS: dict[str, dict[str, float]] = defaultdict(
    lambda: {"face": 0.0, "audio": 0.0}
)

_PARTICIPANT_MEDIA_STATE: dict[str, dict[str, bool]] = defaultdict(
    lambda: {"mic": True, "camera": True}
)

_FRAME_ACCEPT_AFTER: dict[str, float] = {}

_EMOTION_HISTORY: dict[str, dict] = {}
_EMOTION_HISTORY_LAST_RESET: dict[str, float] = {}

_LAST_SEEN: dict[str, float] = {}
_LAST_INFERENCE_TIME: dict[str, float] = {}

_PUMP_RUNNING: set[str] = set()

_SID_TO_PID: dict[str, str] = {}
_PID_TO_SIDS: dict[str, set[str]] = defaultdict(set)
_CONNECTED: set[str] = set()

_scheduler_started = False
_scheduler_lock = asyncio.Lock()


def _pid_connected(pid: str) -> bool:
    """Return True if at least one socket for this participant is still connected."""
    return bool(_PID_TO_SIDS.get(pid))


def _to_bytes(data) -> bytes:
    """Coerce bytes, bytearray, base64 string, or int list to bytes."""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        try:
            return base64.b64decode(data)
        except Exception:
            return data.encode("latin-1")
    if isinstance(data, list):
        return bytes(data)
    raise TypeError(f"Cannot coerce {type(data).__name__} to bytes")


def _decode_audio_bytes(raw: bytes) -> Optional[np.ndarray]:
    """Decode audio bytes to float32 mono PCM at SAMPLE_RATE. Zero disk I/O.

    Decode order:
        1. soundfile.read(BytesIO) — WAV, FLAC, OGG, AIFF via libsndfile.
           Resamples with linear interpolation if the source rate differs.
        2. np.frombuffer(float32) — fallback for raw float32 PCM
           (Web Audio API Float32Array sent as ArrayBuffer).

    MP3/AAC are unsupported; clients should send WAV or raw float32 PCM.

    Returns:
        float32 numpy array of mono PCM samples, or None on decode failure.
    """
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            new_len = int(len(audio) * SAMPLE_RATE / sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
        return audio
    except Exception as exc:
        logger.debug(
            "_decode_audio_bytes soundfile decode failed len=%d error=%s",
            len(raw),
            exc,
        )

    if len(raw) % 4 == 0:
        try:
            arr = np.frombuffer(raw, dtype=np.float32)
            if len(arr) > 0:
                return arr
        except Exception:
            pass

    logger.debug("_decode_audio_bytes: unrecognised format len=%d", len(raw))
    return None


def _extract_audio_embedding(pcm: np.ndarray) -> Optional[np.ndarray]:
    """Extract a Wav2Vec2 embedding from a float32 PCM array.

    Uses the trailing AUDIO_WINDOW_SAMPLES of the input (most recent audio).
    Chunks shorter than 1/8 of the window (~75 ms at 16 kHz) are discarded
    as too brief to produce a meaningful embedding.

    Returns:
        float32 array of shape (AUDIO_DIM,), or None on failure.
    """
    proc = _emo_mod.processor
    model = _emo_mod.wav2vec
    if proc is None or model is None:
        return None
    try:
        if len(pcm) > AUDIO_WINDOW_SAMPLES:
            pcm = pcm[-AUDIO_WINDOW_SAMPLES:]
        elif len(pcm) < AUDIO_WINDOW_SAMPLES // 8:
            return None

        inputs = proc(
            [pcm], sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state.mean(dim=1)[0].cpu().numpy()
        emb = emb.astype(np.float32)
        if emb.shape[0] != AUDIO_DIM:
            logger.warning("audio emb dim=%d expected=%d", emb.shape[0], AUDIO_DIM)
            return None
        return emb
    except Exception:
        logger.exception("_extract_audio_embedding failed")
        return None


def _decode_and_embed_face(raw: bytes) -> Optional[np.ndarray]:
    """Decode a JPEG/PNG frame and extract a MediaPipe face embedding.

    Combines decode and embed into one executor job to halve scheduling
    overhead and reduce the memory lifetime of the raw frame bytes.

    Returns:
        float32 embedding array, or None if the frame cannot be decoded
        or no face is detected.
    """
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return extract_face_embedding(rgb)


def _resolve_modality(pid: str) -> tuple[Optional[int], bool, bool]:
    """Determine active modality from authoritative media state and embedding freshness.

    A modality is active only when all three conditions hold:
        1. Not explicitly disabled via participant.media_state.
        2. Last successful embedding arrived within MODALITY_STALE_SEC.
        3. Rolling embedding buffer is non-empty.

    Returns:
        Tuple of (mod_flag, use_face, use_audio).
        Returns (None, False, False) when no modality is active; the pump
        skips the inference cycle entirely on this path.
    """
    now = time.monotonic()
    with _state_lock:
        media = _PARTICIPANT_MEDIA_STATE[pid]
        mic_enabled = media["mic"]
        cam_enabled = media["camera"]
        ts = _MODALITY_TIMESTAMPS[pid]
        ts_audio = ts["audio"]
        ts_face = ts["face"]
        audio_buf_len = len(_AUDIO_BUFFER.get(pid, []))
        face_buf_len = len(_FACE_BUFFER.get(pid, []))

    if not mic_enabled:
        use_audio = False
    else:
        use_audio = (now - ts_audio) <= MODALITY_STALE_SEC and audio_buf_len > 0
    if not cam_enabled:
        use_face = False
    else:
        use_face = (now - ts_face) <= MODALITY_STALE_SEC and face_buf_len > 0

    if use_face and use_audio:
        return MODALITY_BOTH, True, True
    if use_audio:
        return MODALITY_AUDIO_ONLY, False, True
    if use_face:
        return MODALITY_VIDEO_ONLY, True, False

    return None, False, False


def _build_inference_arrays(pid: str) -> Optional[tuple]:
    """Assemble (xf, xa, fm, am, mod_flag) tensors for the predictor.

    Stale modalities are zeroed out even when their buffer contains data,
    ensuring the ensemble selects the correct per-modality anomaly model.
    Padding is handled by front-repeating the oldest frame to fill SEQ_LEN.

    Returns:
        Tuple (xf, xa, fm, am, mod_flag), or None when no modality is active.
    """
    with _state_lock:
        face_frames = list(_FACE_BUFFER[pid])
        audio_frames = list(_AUDIO_BUFFER[pid])

    mod_flag, use_face, use_audio = _resolve_modality(pid)

    if mod_flag is None or (not use_face and not use_audio):
        return None

    if use_face and face_frames:
        n = len(face_frames)
        padded = list(face_frames)
        while len(padded) < SEQ_LEN:
            padded.insert(0, padded[0])
        xf = np.stack(padded[:SEQ_LEN], axis=0).astype(np.float32)
        fm = np.zeros(SEQ_LEN, dtype=np.float32)
        fm[-min(n, SEQ_LEN) :] = 1.0
    else:
        xf = np.zeros((SEQ_LEN, FACE_DIM), dtype=np.float32)
        fm = np.zeros(SEQ_LEN, dtype=np.float32)

    if use_audio and audio_frames:
        n = len(audio_frames)
        padded = list(audio_frames)
        while len(padded) < SEQ_LEN:
            padded.insert(0, padded[0])
        xa = np.stack(padded[:SEQ_LEN], axis=0).astype(np.float32)
        am = np.zeros(SEQ_LEN, dtype=np.float32)
        am[-min(n, SEQ_LEN) :] = 1.0
    else:
        xa = np.zeros((SEQ_LEN, AUDIO_DIM), dtype=np.float32)
        am = np.zeros(SEQ_LEN, dtype=np.float32)

    xf = _norm_face(xf, fm)
    xa = _norm_audio(xa, am)

    return xf, xa, fm, am, mod_flag


def _run_inference(pid: str) -> Optional[dict]:
    """Run the emotion predictor synchronously. Submit via _inference_executor."""
    arrays = _build_inference_arrays(pid)
    if arrays is None:
        return None
    xf, xa, fm, am, mod_flag = arrays
    result = predictor.predict(xf, xa, fm, am)
    result["_mod_flag"] = mod_flag
    return result


def _smooth(pid: str, probs: dict) -> dict:
    """Apply EMA smoothing to raw emotion probabilities.

    Resets the history if EMOTION_HISTORY_TTL has elapsed since the last
    inference, preventing stale state from bleeding into new sessions.

    Note:
        _EMOTION_HISTORY is accessed only from per-pid pump coroutines in
        the event loop. No lock is needed; preserve this invariant on refactor.

    Returns:
        Normalised smoothed probability dict.
    """
    now = time.monotonic()
    if now - _EMOTION_HISTORY_LAST_RESET.get(pid, 0) > EMOTION_HISTORY_TTL:
        _EMOTION_HISTORY.pop(pid, None)
        _EMOTION_HISTORY_LAST_RESET[pid] = now

    prev = _EMOTION_HISTORY.get(pid, probs)
    sm = {
        k: SMOOTHING_ALPHA * probs.get(k, 0.0)
        + (1.0 - SMOOTHING_ALPHA) * prev.get(k, 0.0)
        for k in probs
    }
    total = sum(sm.values()) + 1e-9
    sm = {k: v / total for k, v in sm.items()}
    _EMOTION_HISTORY[pid] = sm
    return sm


def _parse_label(pred: dict, pid: str) -> tuple[str, float]:
    """Resolve the predicted emotion label and confidence from a predictor result.

    Applies EMA smoothing when raw probabilities are available.
    Falls back to _NEUTRAL_LABEL when confidence is below CONFIDENCE_THRESHOLD.

    Returns:
        Tuple of (label, confidence_score).
    """
    raw_probs = pred.get("probs")
    if raw_probs:
        smoothed = _smooth(pid, raw_probs)
        label = max(smoothed, key=smoothed.get)
        score = float(smoothed[label])
    else:
        label = pred.get("emotion") or _NEUTRAL_LABEL
        score = float(pred.get("confidence") or 0.0)

    if score < CONFIDENCE_THRESHOLD:
        label = _NEUTRAL_LABEL
    return label, score


def _parse_payload(sid: str, data) -> tuple[Optional[bytes], str]:
    """Extract (raw_bytes, participant_id) from any incoming payload shape.

    The participant ID is read from the authoritative _SID_TO_PID mapping set
    at connect time and is never overwritten. Dict payloads that include a
    participantId are validated against the established mapping; mismatches
    are logged and ignored.

    Accepted payload shapes:
        bytes / bytearray / int list — coerced directly.
        str                          — decoded as base64.
        dict                         — bytes extracted from buffer/data/payload/chunk keys.
    """
    pid = _SID_TO_PID.get(sid, sid)
    raw: Optional[bytes] = None

    if isinstance(data, (bytes, bytearray, list)):
        try:
            raw = _to_bytes(data)
        except Exception:
            logger.warning("bytes coerce failed sid=%s", sid)

    elif isinstance(data, dict):
        for key in ("participantId", "participant_id", "pid"):
            val = data.get(key)
            if val:
                claimed_pid = str(val)
                if claimed_pid != pid:
                    logger.warning(
                        "pid mismatch sid=%s established=%s claimed=%s — ignoring claim",
                        sid,
                        pid,
                        claimed_pid,
                    )
                break

        for key in ("buffer", "data", "payload", "chunk"):
            candidate = data.get(key)
            if candidate is not None:
                try:
                    raw = _to_bytes(candidate)
                except Exception:
                    pass
                break

    elif isinstance(data, str):
        try:
            raw = base64.b64decode(data)
        except Exception:
            logger.warning("str not valid base64 sid=%s", sid)

    else:
        logger.warning("unexpected payload type=%s sid=%s", type(data).__name__, sid)

    return raw, pid


async def _maybe_signal_backpressure(sid: str) -> None:
    """If face executor is saturated, tell the client to reduce its frame rate."""
    with _face_pending_lock:
        depth = _face_pending
    if depth >= BACKPRESSURE_QUEUE_DEPTH:
        await sio.emit(
            "backpressure",
            {
                "queueDepth": depth,
                "suggestedFps": max(1, TARGET_CLIENT_FPS // 2),
                "ts": int(time.time() * 1000),
            },
            to=sid,
        )


async def _pump(pid: str) -> None:
    """Per-participant inference coroutine (one per pid, keyed by pid not sid).

    Clock selection (re-evaluated each iteration from _PARTICIPANT_MEDIA_STATE):
        normal / audio_only  — audio chunks drive inference.
        video_only           — frame arrivals drive inference when mic is muted.
        both disabled        — pump exits immediately.

    Modality is resolved at inference time via _resolve_modality(), which
    consults authoritative media state, freshness timestamps, and buffer lengths.

    Concurrency: _ensure_pump is called only from the event loop, making the
    _PUMP_RUNNING check+add atomic from the event loop's perspective — no two
    pumps can run for the same pid simultaneously.
    """
    loop = asyncio.get_running_loop()

    try:
        while True:
            if not _pid_connected(pid):
                return

            with _state_lock:
                audio_pcm = _LATEST_AUDIO.pop(pid, None)
                frame_bytes = _LATEST_FRAME.pop(pid, None)
                _media = _PARTICIPANT_MEDIA_STATE[pid]
                mic_on = _media["mic"]
                cam_on = _media["camera"]

            if not mic_on and not cam_on:
                logger.info("pump halted pid=%s reason=all_modalities_disabled", pid)
                return

            video_only_mode = cam_on and not mic_on

            if not video_only_mode and audio_pcm is None:
                await asyncio.sleep(0.005)
                continue

            if video_only_mode and frame_bytes is None:
                await asyncio.sleep(0.005)
                continue

            _LAST_SEEN[pid] = time.time()

            elapsed = time.time() - _LAST_INFERENCE_TIME.get(pid, 0)
            if elapsed < MIN_INFERENCE_INTERVAL:
                await asyncio.sleep(MIN_INFERENCE_INTERVAL - elapsed)
                if not _pid_connected(pid):
                    return
                with _state_lock:
                    audio_pcm = _LATEST_AUDIO.pop(pid, audio_pcm)
                    frame_bytes = _LATEST_FRAME.pop(pid, frame_bytes)

            _LAST_INFERENCE_TIME[pid] = time.time()
            t_start = time.perf_counter()

            if audio_pcm is not None:
                try:
                    audio_emb = await loop.run_in_executor(
                        _audio_executor, _extract_audio_embedding, audio_pcm
                    )
                except Exception:
                    logger.exception("audio embedding pid=%s", pid)
                    audio_emb = None

                if audio_emb is not None:
                    with _state_lock:
                        if not _PARTICIPANT_MEDIA_STATE[pid]["mic"]:
                            audio_emb = None
                        else:
                            _AUDIO_BUFFER[pid].append(audio_emb)
                            _MODALITY_TIMESTAMPS[pid]["audio"] = time.monotonic()

            if frame_bytes is not None:
                global _face_pending
                with _face_pending_lock:
                    _face_pending += 1
                try:
                    face_emb = await loop.run_in_executor(
                        _face_executor, _decode_and_embed_face, frame_bytes
                    )
                except Exception:
                    logger.exception("face extraction pid=%s", pid)
                    face_emb = None
                finally:
                    with _face_pending_lock:
                        _face_pending -= 1

                if face_emb is not None:
                    with _state_lock:
                        if not _PARTICIPANT_MEDIA_STATE[pid]["camera"]:
                            face_emb = None
                        else:
                            _FACE_BUFFER[pid].append(face_emb)
                            _MODALITY_TIMESTAMPS[pid]["face"] = time.monotonic()
                else:
                    logger.debug("no face detected pid=%s", pid)

            if not _pid_connected(pid):
                return

            if video_only_mode:
                if not _FACE_BUFFER.get(pid):
                    await asyncio.sleep(0.005)
                    continue
            else:
                if not _AUDIO_BUFFER.get(pid):
                    await asyncio.sleep(0.005)
                    continue

            try:
                pred = await asyncio.wait_for(
                    loop.run_in_executor(_inference_executor, _run_inference, pid),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                logger.error("inference timeout pid=%s — skipping cycle", pid)
                await asyncio.sleep(0.05)
                continue
            except Exception:
                logger.exception("inference pid=%s", pid)
                await asyncio.sleep(0.01)
                continue

            if not pred:
                await asyncio.sleep(0.005)
                continue

            label, score = _parse_label(pred, pid)
            mod_flag = pred.get("_mod_flag", MODALITY_AUDIO_ONLY)
            modality = MODALITY_NAMES.get(mod_flag, "audio_only")
            is_anomaly = pred.get("anomaly", False)
            anom_score = pred.get("anomaly_score")
            latency_ms = round((time.perf_counter() - t_start) * 1_000, 2)

            _latency_tracker.record(latency_ms, modality)

            logger.info(
                "infer pid=%-20s emotion=%-14s conf=%.3f mod=%-10s "
                "anomaly=%s lat=%.1fms",
                pid,
                label,
                score,
                modality,
                is_anomaly,
                latency_ms,
            )

            if not _pid_connected(pid):
                return

            payload = {
                "participantId": pid,
                "result": {
                    "emotion": label,
                    "confidence": round(score, 4),
                    "probs": pred.get("probs", {}),
                    "modality": modality,
                    "anomaly": is_anomaly,
                    "anomalyScore": (
                        round(anom_score, 6) if anom_score is not None else None
                    ),
                },
                "latencyMs": latency_ms,
                "ts": int(time.time() * 1000),
            }

            for sid in list(_PID_TO_SIDS.get(pid, set())):
                if sid in _CONNECTED:
                    await sio.emit("emotion.result", payload, to=sid)

            await asyncio.sleep(0.002)

    finally:
        with _state_lock:
            _PUMP_RUNNING.discard(pid)


def _ensure_pump(pid: str) -> None:
    """Spawn the inference pump for pid if one is not already running.

    Called by:
        _push_audio  — normal and audio_only modes (primary clock).
        _push_frame  — video_only mode when mic is disabled.

    Must only be called from the event loop; the _PUMP_RUNNING check+add
    is atomic from the event loop's perspective, preventing duplicate pumps.
    """
    if pid not in _PUMP_RUNNING:
        _PUMP_RUNNING.add(pid)
        asyncio.create_task(_pump(pid))


async def _push_frame(sid: str, raw: bytes, pid: str) -> None:
    """Slot a video frame after applying the per-pid monotonic frame-rate gate.

    Frames are dropped before any decode work if they arrive earlier than
    _FRAME_ACCEPT_AFTER[pid], or if the camera is explicitly disabled.
    In video_only mode (mic muted), each accepted frame also wakes the pump
    so inference continues without waiting for audio that will never arrive.
    Back-pressure is signalled to the client when the face executor is saturated.
    """
    if not raw or len(raw) > MAX_FRAME_SIZE:
        await sio.emit(
            "emotion.error",
            {"code": "FRAME_TOO_LARGE", "maxBytes": MAX_FRAME_SIZE},
            to=sid,
        )
        return

    now = time.monotonic()
    with _state_lock:
        if not _PARTICIPANT_MEDIA_STATE[pid]["camera"]:
            return
        accept_after = _FRAME_ACCEPT_AFTER.get(pid, 0.0)
        if now < accept_after:
            return
        _FRAME_ACCEPT_AFTER[pid] = now + FRAME_MIN_INTERVAL_SEC
        _LATEST_FRAME[pid] = raw

    await _maybe_signal_backpressure(sid)

    with _state_lock:
        _pms = _PARTICIPANT_MEDIA_STATE[pid]
        _video_only = _pms["camera"] and not _pms["mic"]
    if _video_only:
        _ensure_pump(pid)


async def _push_audio(sid: str, raw: bytes, pid: str) -> None:
    """Decode audio bytes to PCM, slot into _LATEST_AUDIO, and wake the pump.

    Audio is the inference clock in normal and audio_only modes.
    Hard-rejects chunks arriving after the mic is explicitly disabled to prevent
    stale AudioWorklet flushes from re-entering the pipeline.
    """
    if not raw or len(raw) > MAX_AUDIO_SIZE:
        logger.warning("audio size invalid len=%d pid=%s", len(raw) if raw else 0, pid)
        await sio.emit(
            "emotion.error",
            {"code": "AUDIO_TOO_LARGE", "maxBytes": MAX_AUDIO_SIZE},
            to=sid,
        )
        return

    loop = asyncio.get_running_loop()
    try:
        pcm = await loop.run_in_executor(_audio_executor, _decode_audio_bytes, raw)
    except Exception:
        logger.exception("audio decode pid=%s", pid)
        return

    if pcm is None or len(pcm) == 0:
        return

    if sid not in _CONNECTED:
        return

    with _state_lock:
        if not _PARTICIPANT_MEDIA_STATE[pid]["mic"]:
            return
        _LATEST_AUDIO[pid] = pcm

    _ensure_pump(pid)


sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()] or "*",
    ping_timeout=30,
    ping_interval=15,
    max_http_buffer_size=8 * 1024 * 1024,
    logger=False,
    engineio_logger=False,
)


@sio.event
async def connect(sid: str, environ: dict, auth: Optional[dict]) -> None:
    """Register a new socket connection and emit server capabilities.

    The participant ID is resolved once from auth and stored immutably in
    _SID_TO_PID for the session lifetime. Emits server.status so the client
    can configure its capture loop (targetFps, seqLen, modalityStaleSec).
    Starts the GC scheduler on the first connection.
    """
    global _scheduler_started

    shared_secret = os.getenv("EMOTION_SHARED_SECRET")
    if shared_secret:
        provided = (auth or {}).get("token")
        if provided != shared_secret:
            raise ConnectionRefusedError("unauthorized")

    pid = sid
    if auth:
        for key in ("participantId", "participant_id", "pid"):
            val = auth.get(key)
            if val:
                pid = str(val)
                break

    with _state_lock:
        _CONNECTED.add(sid)
        _SID_TO_PID[sid] = pid
        _PID_TO_SIDS[pid].add(sid)

    logger.info("connect sid=%s pid=%s total=%d", sid, pid, len(_CONNECTED))

    await sio.emit(
        "server.status",
        {
            "status": "connected",
            "participantId": pid,
            "device": DEVICE,
            "seqLen": SEQ_LEN,
            "targetFps": TARGET_CLIENT_FPS,
            "modalityStaleSec": MODALITY_STALE_SEC,
            "modalities": ["video_only", "audio_only", "both"],
            "ts": int(time.time() * 1000),
        },
        to=sid,
    )

    async with _scheduler_lock:
        if not _scheduler_started:
            scheduler.add_job(_gc_stale_participants, "interval", seconds=30)
            scheduler.start()
            _scheduler_started = True


@sio.event
async def disconnect(sid: str) -> None:
    """Deregister a socket and clean up participant state if no sockets remain."""
    with _state_lock:
        _CONNECTED.discard(sid)
        pid = _SID_TO_PID.pop(sid, sid)
        _PID_TO_SIDS[pid].discard(sid)
        still_active = bool(_PID_TO_SIDS[pid])

    logger.info("disconnect sid=%s pid=%s remaining_sids=%s", sid, pid, still_active)

    if not still_active:
        _cleanup_participant(pid)


@sio.on("frame")
@sio.on("emotion.frame")
async def on_frame(sid: str, data) -> None:
    """Video frame — raw bytes, base64, or {participantId, buffer}.
    'emotion.frame' is accepted as a backwards-compatible alias.
    """
    if sid not in _CONNECTED:
        return
    raw, pid = _parse_payload(sid, data)
    if raw:
        await _push_frame(sid, raw, pid)


@sio.on("audio_chunk")
async def on_audio_chunk(sid: str, data) -> None:
    """Audio chunk — WAV / FLAC / OGG bytes, base64, or raw float32 PCM.

    Accepted payload shapes:
      • Raw bytes (WAV / FLAC header present)
      • Base64-encoded string of the above
      • { participantId, buffer|data|payload: <bytes|base64> }
      • Raw float32 PCM bytes (Web Audio API Float32Array as ArrayBuffer)
    """
    if sid not in _CONNECTED:
        return
    raw, pid = _parse_payload(sid, data)
    if raw:
        await _push_audio(sid, raw, pid)


@sio.on("av_chunk")
async def on_av_chunk(sid: str, data) -> None:
    """Combined audio + video for bimodal inference.

    Expected shape:
        {
            participantId: str,
            video?: <bytes|base64>,   // JPEG / PNG frame
            audio?: <bytes|base64>,   // WAV / PCM chunk
        }
    At least one of video / audio must be present.
    Audio presence drives inference; video enriches it.
    """
    if sid not in _CONNECTED:
        return
    if not isinstance(data, dict):
        logger.warning("av_chunk not a dict type=%s sid=%s", type(data).__name__, sid)
        return

    pid = _SID_TO_PID.get(sid, sid)

    if v := (data.get("video") or data.get("frame")):
        try:
            await _push_frame(sid, _to_bytes(v), pid)
        except Exception:
            logger.warning("av_chunk video coerce failed sid=%s", sid)

    if a := (data.get("audio") or data.get("audio_chunk")):
        try:
            await _push_audio(sid, _to_bytes(a), pid)
        except Exception:
            logger.warning("av_chunk audio coerce failed sid=%s", sid)


@sio.on("participant.media_state")
async def on_media_state(sid: str, data) -> None:
    """Handle mic/camera toggle events and update modality state immediately.

    Payload: { participantId?: str, micEnabled: bool, cameraEnabled: bool }

    Zeros the disabled modality's timestamp and clears its buffer so
    _resolve_modality() switches away from that modality on the next cycle —
    no stale-window lag. Cross-promotes the still-live modality's timestamp
    to avoid a no-inference gap while waiting for the next natural embedding.

    Supports relay path: a host socket may send this event with an explicit
    participantId to update a remote participant's state.
    """
    if sid not in _CONNECTED:
        return
    if not isinstance(data, dict):
        logger.warning("media_state not a dict sid=%s", sid)
        return

    pid = _SID_TO_PID.get(sid, sid)

    claimed_pid = data.get("participantId") or data.get("participant_id")
    if claimed_pid and str(claimed_pid) != pid:
        pid = str(claimed_pid)

    mic_enabled = bool(data.get("micEnabled", True))
    cam_enabled = bool(data.get("cameraEnabled", True))

    with _state_lock:
        _PARTICIPANT_MEDIA_STATE[pid]["mic"] = mic_enabled
        _PARTICIPANT_MEDIA_STATE[pid]["camera"] = cam_enabled

        ts = _MODALITY_TIMESTAMPS[pid]

        if not mic_enabled:
            ts["audio"] = 0.0
            _LATEST_AUDIO.pop(pid, None)
            try:
                _AUDIO_BUFFER[pid].clear()
            except Exception:
                pass
            if cam_enabled and len(_FACE_BUFFER.get(pid, [])) > 0:
                ts["face"] = time.monotonic()

        if not cam_enabled:
            ts["face"] = 0.0
            _LATEST_FRAME.pop(pid, None)
            try:
                _FACE_BUFFER[pid].clear()
            except Exception:
                pass
            if mic_enabled and len(_AUDIO_BUFFER.get(pid, [])) > 0:
                ts["audio"] = time.monotonic()

    logger.info(
        "media_state pid=%s mic=%s cam=%s",
        pid,
        mic_enabled,
        cam_enabled,
    )


@sio.on("ping")
async def on_ping(sid: str, _data=None) -> None:
    """Respond to client keepalive pings with a server timestamp."""
    await sio.emit("pong", {"ts": int(time.time() * 1000)}, to=sid)


def _cleanup_participant(pid: str) -> None:
    """Remove all server-side state for a participant.

    Called on disconnect (when no sockets remain) and by the GC for stale pids.
    Acquiring _state_lock prevents a race with _ensure_pump during reconnect.
    """
    with _state_lock:
        for d in (
            _FACE_BUFFER,
            _AUDIO_BUFFER,
            _LATEST_FRAME,
            _LATEST_AUDIO,
            _EMOTION_HISTORY,
            _EMOTION_HISTORY_LAST_RESET,
            _LAST_SEEN,
            _LAST_INFERENCE_TIME,
            _PID_TO_SIDS,
            _MODALITY_TIMESTAMPS,
            _FRAME_ACCEPT_AFTER,
            _PARTICIPANT_MEDIA_STATE,
        ):
            d.pop(pid, None)
        _PUMP_RUNNING.discard(pid)
    logger.info("state cleaned up pid=%s", pid)


def _gc_stale_participants() -> None:
    """Evict participants that have been inactive beyond BUFFER_TTL with no open sockets."""
    now = time.time()
    with _state_lock:
        stale = [
            pid
            for pid, t in _LAST_SEEN.items()
            if now - t > BUFFER_TTL and not _PID_TO_SIDS.get(pid)
        ]
    for pid in stale:
        logger.info("GC evicting pid=%s", pid)
        _cleanup_participant(pid)


scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """FastAPI lifespan: load models on startup, shut down executors on exit.

    Uses try/finally so executors are always released even if model loading
    raises after partial initialisation.
    """
    global _norm_stats, predictor , service_ready

    logger.info("═══ Emotion WS server starting (device=%s) ═══", DEVICE)
    loop = asyncio.get_running_loop()

    try:
        logger.info("Loading extractor models (Wav2Vec2 + MediaPipe) …")
        await loop.run_in_executor(None, load_extractor_models)
        logger.info("Extractor models ready")

        logger.info("Loading norm stats …")
        _norm_stats = await loop.run_in_executor(None, _load_norm_stats)

        logger.info("Loading inference predictor (ensemble + anomaly) …")
        predictor = await loop.run_in_executor(None, EmotionPredictor)
        logger.info("Inference predictor ready")
        
        service_ready = True
        logger.info("═══ Server ready ═══")
        yield

    finally:
        service_ready = False
        logger.info("Shutting down …")
        _latency_tracker.report_now()
        if _scheduler_started and scheduler.running:
            scheduler.shutdown(wait=False)
        _face_executor.shutdown(wait=False)
        _audio_executor.shutdown(wait=False)
        _inference_executor.shutdown(wait=False)
        logger.info("Shutdown complete")


app = FastAPI(lifespan=lifespan, title="Emotion WS Server")

def _count_active_participants() -> int:
    """Participants with at least one connected socket (excludes stale-only state)."""
    with _state_lock:
        return sum(1 for sids in _PID_TO_SIDS.values() if sids)


set_tracker(_latency_tracker)
set_active_participant_provider(_count_active_participants)
app.include_router(stats_router)

@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — no model or I/O checks."""
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    """Readiness probe — true only after extractor + predictor init completed."""
    if not service_ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready"},
        )
    return {"status": "ready"}

app.mount("/", socketio.ASGIApp(sio, socketio_path="socket.io"))
