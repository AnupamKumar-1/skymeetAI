import whisper
from transformers import pipeline
from utils.emotion import normalize_emotion, get_emoji
from collections import Counter
import re
from functools import lru_cache


print("Loading Whisper model...")
asr_model = whisper.load_model("small")

print("Loading emotion model...")
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1,
)

MERGE_GAP_SEC = 2.0
MERGE_MAX_WORDS = 60

STOP_WORDS = {
    "this",
    "that",
    "with",
    "have",
    "from",
    "they",
    "were",
    "been",
    "will",
    "would",
    "could",
    "should",
    "about",
    "there",
    "their",
    "what",
    "when",
    "where",
    "which",
    "while",
    "your",
    "just",
    "also",
    "then",
    "than",
    "into",
    "some",
    "very",
    "more",
    "like",
    "really",
    "okay",
    "yeah",
    "well",
    "know",
    "think",
    "said",
    "going",
    "thing",
}


def _extract_label(raw) -> str:
    """Extract a flat emotion label string from a HuggingFace pipeline response.

    Handles the dict, list-of-dict, and list-of-list-of-dict shapes that
    ``text-classification`` pipelines may return depending on version and
    ``top_k`` setting.

    Args:
        raw: Raw output from the HuggingFace pipeline call.

    Returns:
        Emotion label string, or ``"neutral"`` if extraction fails.
    """
    try:
        if isinstance(raw, dict):
            return raw.get("label", "neutral")
        if isinstance(raw, list) and len(raw) > 0:
            if isinstance(raw[0], dict):
                return raw[0].get("label", "neutral")
            if isinstance(raw[0], list) and len(raw[0]) > 0:
                return raw[0][0].get("label", "neutral")
    except Exception:
        pass
    return "neutral"


def _clean_text(text: str) -> str:
    """Normalise a raw Whisper segment string for downstream processing.

    Strips leading/trailing whitespace, capitalises the first character,
    collapses internal whitespace runs to single spaces, and appends a
    period if the text does not already end with ``.``, ``!``, or ``?``.

    Args:
        text: Raw segment text from Whisper.

    Returns:
        Cleaned text string, or empty string if input is falsy.
    """
    if not text:
        return ""
    text = text.strip()
    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    text = " ".join(text.split())
    if text[-1] not in ".!?":
        text += "."
    return text


def _merge_raw_segments(raw_segs: list[dict]) -> list[dict]:
    """Merge consecutive same-speaker Whisper segments into longer utterances.

    Two segments are merged when they share the same speaker, the silence
    gap between them is ≤ ``MERGE_GAP_SEC``, the combined word count is ≤
    ``MERGE_MAX_WORDS``, and the current buffer does not end with sentence-
    terminal punctuation (``.``, ``!``, ``?``). A speaker change always
    flushes the buffer.

    Args:
        raw_segs: List of Whisper segment dicts, each containing at minimum
            ``start``, ``end``, ``text``, and ``speaker`` keys.

    Returns:
        List of merged segment dicts with keys ``start``, ``end``, ``text``,
        and ``speaker``.
    """
    if not raw_segs:
        return []

    merged = []
    buf = {
        "start": raw_segs[0].get("start", 0),
        "end": raw_segs[0].get("end", 0),
        "text": (raw_segs[0].get("text") or "").strip(),
        "speaker": raw_segs[0].get("speaker", "Unknown"),
    }

    for seg in raw_segs[1:]:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        gap = seg.get("start", 0) - buf["end"]
        combined_words = len((buf["text"] + " " + text).split())
        ends_clean = buf["text"].strip().endswith((".", "!", "?"))
        same_speaker = seg.get("speaker", "Unknown") == buf["speaker"]

        if not same_speaker:
            if buf["text"]:
                merged.append(buf)
            buf = {
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": text,
                "speaker": seg.get("speaker", "Unknown"),
            }
            continue

        if (
            gap <= MERGE_GAP_SEC
            and combined_words <= MERGE_MAX_WORDS
            and not ends_clean
        ):
            buf["text"] += " " + text
            buf["end"] = seg.get("end", buf["end"])
        else:
            if buf["text"]:
                merged.append(buf)
            buf = {
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": text,
                "speaker": seg.get("speaker", "Unknown"),
            }

    if buf["text"]:
        merged.append(buf)

    return merged


@lru_cache(maxsize=1024)
def _get_emotion(text: str) -> str:
    """Classify the emotion of a text segment, with in-process caching.

Segments shorter than 4 words are assigned ``"neutral"`` without
model inference. Results are cached using ``functools.lru_cache``.

Args:
    text: Cleaned segment text to classify.

Returns:
    Normalised emotion label string (e.g. ``"joy"``, ``"anger"``,
    ``"neutral"``).
"""
    
    words = text.split()
    if len(words) < 4:
        emo = "neutral"
    else:
        try:
            raw = emotion_pipeline(text, top_k=1)
            emo = normalize_emotion(_extract_label(raw))
        except Exception as e:
            print(f"asr_service: emotion detection failed — {e}")
            emo = "neutral"

    return emo


def _score_segment(seg: dict) -> int:
    """Compute a heuristic relevance score for a transcript segment.

    Points are awarded for:

    - Word count in the 10–35 range (+3) or 5–9 / 36–50 range (+1).
    - Non-neutral emotion (+3).
    - Presence of decision/summary keywords such as ``"important"``,
      ``"action"``, or ``"plan"`` (+2).
    - Ends with a question mark (+1).
    - Duration longer than 5 seconds (+1).

    Used by ``_build_narrative_summary`` and ``build_intelligent_summary``
    to rank segments for key-point and summary selection.

    Args:
        seg: Segment dict with at minimum ``text``, ``emotion``, ``start``,
            and ``end`` keys.

    Returns:
        Integer relevance score (higher is more relevant).
    """
    score = 0
    words = seg["text"].split()
    word_count = len(words)

    if 10 <= word_count <= 35:
        score += 3
    elif 5 <= word_count < 10 or 35 < word_count <= 50:
        score += 1

    if seg["emotion"] not in ("neutral",):
        score += 3

    if any(
        w in seg["text"].lower()
        for w in [
            "important",
            "key",
            "main",
            "critical",
            "problem",
            "solution",
            "decide",
            "agree",
            "conclusion",
            "summary",
            "action",
            "next",
            "plan",
        ]
    ):
        score += 2

    if seg["text"].strip().endswith("?"):
        score += 1

    duration = seg.get("end", 0) - seg.get("start", 0)
    if duration > 5:
        score += 1

    return score


def _build_narrative_summary(segments: list[dict], full_text: str) -> str:
    """Construct a short narrative summary by selecting the best segment from each third.

    The conversation is divided into opening, middle, and closing thirds.
    The highest-scoring segment (via ``_score_segment``) from each third is
    selected and joined into a sentence. The result is truncated to 80 words
    if necessary.

    Short inputs are handled as special cases: empty text returns ``""``,
    text ≤ 40 words is returned as-is, and transcripts with ≤ 3 segments
    return the full text (capped at 80 words).

    Args:
        segments: List of scored segment dicts (must contain ``text``,
            ``emotion``, ``start``, ``end``).
        full_text: Pre-joined text of all segments; used for word-count
            short-circuit checks.

    Returns:
        Summary string of up to 80 words.
    """
    words = full_text.split()
    total_words = len(words)

    if total_words == 0:
        return ""
    if total_words <= 40:
        return full_text

    num_segs = len(segments)
    if num_segs <= 3:
        return full_text if total_words < 80 else " ".join(words[:80]) + "..."

    third = num_segs // 3
    opening = segments[: max(1, third)]
    middle = segments[max(1, third) : max(2, 2 * third)]
    closing = segments[max(2, 2 * third) :]

    def best_from(group):
        if not group:
            return ""
        return sorted(group, key=_score_segment, reverse=True)[0]["text"]

    parts = []
    o = best_from(opening)
    m = best_from(middle)
    c = best_from(closing)

    if o:
        parts.append(o.rstrip(".") + ".")
    if m and m != o:
        parts.append(m.rstrip(".") + ".")
    if c and c not in (o, m):
        parts.append(c.rstrip(".") + ".")

    summary = " ".join(parts)
    summary_words = summary.split()
    if len(summary_words) > 80:
        summary = " ".join(summary_words[:80]) + "..."

    return summary


def build_intelligent_summary(segments: list[dict]) -> dict:
    """Generate a structured analysis dict from a list of transcript segments.

    Computes a narrative summary, up to 5 key points (ranked by
    ``_score_segment``), emotion distribution, dominant emotion, up to 3
    notable emotional moments, up to 8 top topics (word frequency after
    stop-word filtering on words ≥ 5 characters), per-speaker turn and
    word-count stats, total word count, speaking pace in WPM, and total
    duration.

    Args:
        segments: List of segment dicts, each with ``text``, ``emotion``,
            ``emoji``, ``start``, ``end``, and ``speaker`` keys.

    Returns:
        Dict with keys ``summary`` (str), ``key_points`` (list[str]), and
        ``insights`` (dict). Returns empty values for all fields when
        ``segments`` is empty.
    """
    if not segments:
        return {"summary": "", "key_points": [], "insights": {}}

    full_text = " ".join([s["text"] for s in segments])
    summary = _build_narrative_summary(segments, full_text)

    scored_segs = sorted(segments, key=_score_segment, reverse=True)
    seen_texts = set()
    key_points = []
    for seg in scored_segs:
        txt = seg["text"].strip()
        if txt not in seen_texts and len(txt.split()) >= 5:
            key_points.append(txt)
            seen_texts.add(txt)
        if len(key_points) == 5:
            break

    emotions = [s["emotion"] for s in segments]
    counts = Counter(emotions)
    total = len(emotions)
    emotion_distribution = {k: round(v / total * 100) for k, v in counts.items()}
    dominant_emotion = counts.most_common(1)[0][0]

    emotional_moments = []
    for seg in segments:
        if seg["emotion"] not in ("neutral",):
            emotional_moments.append(
                {
                    "text": seg["text"][:80],
                    "emotion": seg["emotion"],
                    "start": seg.get("start", 0),
                }
            )
    emotional_moments = emotional_moments[:3]

    raw_words = re.findall(r"\b[a-zA-Z]{5,}\b", full_text.lower())
    filtered_words = [w for w in raw_words if w not in STOP_WORDS]
    word_freq = Counter(filtered_words)
    top_topics = [w for w, _ in word_freq.most_common(8)]

    speakers = list({s.get("speaker", "Unknown") for s in segments})
    speaker_stats = {}
    for spk in speakers:
        spk_segs = [s for s in segments if s.get("speaker") == spk]
        spk_emotions = Counter(s["emotion"] for s in spk_segs)
        speaker_stats[spk] = {
            "turns": len(spk_segs),
            "dominant_emotion": (
                spk_emotions.most_common(1)[0][0] if spk_emotions else "neutral"
            ),
            "word_count": sum(len(s["text"].split()) for s in spk_segs),
        }

    total_duration = segments[-1].get("end", 0) if segments else 0
    total_words_spoken = sum(len(s["text"].split()) for s in segments)
    speaking_pace = (
        round(total_words_spoken / (total_duration / 60)) if total_duration > 0 else 0
    )

    return {
        "summary": summary,
        "key_points": key_points,
        "insights": {
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": emotion_distribution,
            "emotional_moments": emotional_moments,
            "top_topics": top_topics,
            "speaker_stats": speaker_stats,
            "total_words": total_words_spoken,
            "speaking_pace_wpm": speaking_pace,
            "total_duration_sec": round(total_duration),
        },
    }


def transcribe_and_emotion(wav_path: str, speaker_name: str = "Unknown") -> dict:
    """Transcribe a WAV file and classify per-segment emotion.

    Runs Whisper ASR (``small``, English, ``fp16=False``), assigns
    ``speaker_name`` to every raw segment, merges consecutive segments via
    ``_merge_raw_segments``, cleans text, truncates segments exceeding 80
    words, classifies emotion, and builds a per-file ``build_intelligent_summary``.

    Note: The per-file summary returned here is stored by the caller but is
    not forwarded to NODE_API; only the aggregate summary across all speakers
    is sent in the callback payload.

    Args:
        wav_path: Absolute or relative path to a mono 16 kHz WAV file.
        speaker_name: Display name assigned to all segments from this file.
            Defaults to ``"Unknown"``.

    Returns:
        Dict with keys ``segments`` (list of dicts with ``start``, ``end``,
        ``text``, ``emotion``, ``emoji``, ``speaker``) and ``analysis``
        (output of ``build_intelligent_summary``). Returns empty segments
        and analysis on transcription failure.
    """
    try:
        asr = asr_model.transcribe(wav_path, language="en", fp16=False)
    except Exception as e:
        print(f"asr_service: transcription failed for {wav_path} — {e}")
        return {"segments": [], "analysis": build_intelligent_summary([])}

    raw_segs = asr.get("segments", [])

    for seg in raw_segs:
        seg["speaker"] = speaker_name

    merged_segs = _merge_raw_segments(raw_segs)
    segments = []

    for seg in merged_segs:
        text = _clean_text(seg["text"])
        if len(text) < 3:
            continue
        words = text.split()
        if len(words) > 80:
            text = " ".join(words[:80]) + "..."
        emo = _get_emotion(text)
        segments.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
                "emotion": emo,
                "emoji": get_emoji(emo),
                "speaker": speaker_name,
            }
        )

    analysis = build_intelligent_summary(segments)
    return {"segments": segments, "analysis": analysis}
