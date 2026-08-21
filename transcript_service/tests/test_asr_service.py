import importlib
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _install_stub_modules():
    whisper_stub = types.ModuleType("whisper")
    whisper_stub.load_model = lambda name: types.SimpleNamespace(
        transcribe=lambda *a, **k: {"segments": []}
    )

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.pipeline = lambda *a, **k: (lambda text, **kw: [{"label": "neutral", "score": 1.0}])

    sys.modules["whisper"] = whisper_stub
    sys.modules["transformers"] = transformers_stub


_install_stub_modules()

asr_service = importlib.import_module("services.asr_service")


class ExtractLabelTests(unittest.TestCase):
    def test_dict_shape(self):
        self.assertEqual(asr_service._extract_label({"label": "joy"}), "joy")

    def test_list_of_dict_shape(self):
        self.assertEqual(asr_service._extract_label([{"label": "anger"}]), "anger")

    def test_list_of_list_of_dict_shape(self):
        self.assertEqual(asr_service._extract_label([[{"label": "fear"}]]), "fear")

    def test_unrecognised_shape_falls_back_to_neutral(self):
        self.assertEqual(asr_service._extract_label(None), "neutral")
        self.assertEqual(asr_service._extract_label([]), "neutral")
        self.assertEqual(asr_service._extract_label("garbage"), "neutral")


class CleanTextTests(unittest.TestCase):
    def test_empty_returns_empty(self):
        self.assertEqual(asr_service._clean_text(""), "")
        self.assertEqual(asr_service._clean_text(None), "")

    def test_capitalises_first_letter(self):
        self.assertEqual(asr_service._clean_text("hello there"), "Hello there.")

    def test_collapses_internal_whitespace(self):
        self.assertEqual(asr_service._clean_text("hello    world  "), "Hello world.")

    def test_preserves_existing_terminal_punctuation(self):
        self.assertEqual(asr_service._clean_text("really?"), "Really?")
        self.assertEqual(asr_service._clean_text("wow!"), "Wow!")

    def test_single_character(self):
        self.assertEqual(asr_service._clean_text("a"), "A.")


class MergeRawSegmentsTests(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(asr_service._merge_raw_segments([]), [])

    def test_merges_same_speaker_within_gap_and_word_limit(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "A"},
            {"start": 1.5, "end": 2.5, "text": "world", "speaker": "A"},
        ]
        merged = asr_service._merge_raw_segments(segs)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "hello world")
        self.assertEqual(merged[0]["end"], 2.5)

    def test_speaker_change_flushes_buffer(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "hi", "speaker": "A"},
            {"start": 1.1, "end": 2.0, "text": "hello", "speaker": "B"},
        ]
        merged = asr_service._merge_raw_segments(segs)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["speaker"], "A")
        self.assertEqual(merged[1]["speaker"], "B")

    def test_large_gap_splits_segments(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "hi", "speaker": "A"},
            {"start": 10.0, "end": 11.0, "text": "later", "speaker": "A"},
        ]
        merged = asr_service._merge_raw_segments(segs)
        self.assertEqual(len(merged), 2)

    def test_sentence_terminal_punctuation_forces_split(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "Done.", "speaker": "A"},
            {"start": 1.2, "end": 2.0, "text": "next thought", "speaker": "A"},
        ]
        merged = asr_service._merge_raw_segments(segs)
        self.assertEqual(len(merged), 2)

    def test_empty_text_segments_are_skipped(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "hi", "speaker": "A"},
            {"start": 1.1, "end": 1.2, "text": "   ", "speaker": "A"},
            {"start": 1.3, "end": 2.0, "text": "there", "speaker": "A"},
        ]
        merged = asr_service._merge_raw_segments(segs)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "hi there")


class GetEmotionTests(unittest.TestCase):
    def setUp(self):
        asr_service._get_emotion.cache_clear()

    def test_short_text_is_neutral_without_model_call(self):
        with mock.patch.object(asr_service, "emotion_pipeline") as mocked:
            result = asr_service._get_emotion("hi there")
            mocked.assert_not_called()
            self.assertEqual(result, "neutral")

    def test_long_text_invokes_model_and_normalises(self):
        with mock.patch.object(
            asr_service, "emotion_pipeline", return_value=[{"label": "happy"}]
        ):
            result = asr_service._get_emotion("this is a much longer sentence here")
            self.assertEqual(result, "joy")

    def test_model_exception_falls_back_to_neutral(self):
        def boom(*a, **k):
            raise RuntimeError("boom")

        with mock.patch.object(asr_service, "emotion_pipeline", side_effect=boom):
            result = asr_service._get_emotion("this text is long enough to classify")
            self.assertEqual(result, "neutral")


class ScoreSegmentTests(unittest.TestCase):
    def test_word_count_scoring_bands(self):
        seg_mid = {"text": " ".join(["word"] * 20), "emotion": "neutral", "start": 0, "end": 0}
        seg_low = {"text": " ".join(["word"] * 6), "emotion": "neutral", "start": 0, "end": 0}
        seg_tiny = {"text": "one two", "emotion": "neutral", "start": 0, "end": 0}
        self.assertGreater(asr_service._score_segment(seg_mid), asr_service._score_segment(seg_low))
        self.assertGreater(asr_service._score_segment(seg_low), asr_service._score_segment(seg_tiny))

    def test_non_neutral_emotion_adds_points(self):
        base = {"text": "word " * 12, "emotion": "neutral", "start": 0, "end": 0}
        emo = {"text": "word " * 12, "emotion": "anger", "start": 0, "end": 0}
        self.assertGreater(asr_service._score_segment(emo), asr_service._score_segment(base))

    def test_keyword_and_question_and_duration_bonuses(self):
        seg = {
            "text": "This is an important decision, what do you think?",
            "emotion": "neutral",
            "start": 0,
            "end": 6,
        }
        plain = {"text": "This is a decision", "emotion": "neutral", "start": 0, "end": 0}
        self.assertGreater(asr_service._score_segment(seg), asr_service._score_segment(plain))


class BuildNarrativeSummaryTests(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(asr_service._build_narrative_summary([], ""), "")

    def test_short_text_returned_as_is(self):
        text = "hello world this is short"
        self.assertEqual(asr_service._build_narrative_summary([{"text": text}], text), text)

    def test_long_text_capped_at_80_words_for_few_segments(self):
        words = ["word"] * 100
        text = " ".join(words)
        segs = [{"text": text}, {"text": text}]
        summary = asr_service._build_narrative_summary(segs, text)
        self.assertTrue(summary.endswith("..."))
        self.assertLessEqual(len(summary.replace("...", "").split()), 80)


class BuildIntelligentSummaryTests(unittest.TestCase):
    def test_empty_segments(self):
        result = asr_service.build_intelligent_summary([])
        self.assertEqual(result, {"summary": "", "key_points": [], "insights": {}})

    def test_populated_summary_structure(self):
        segments = [
            {
                "text": "This is an important decision about the project plan.",
                "emotion": "joy",
                "emoji": "😄",
                "start": 0.0,
                "end": 4.0,
                "speaker": "Alice",
            },
            {
                "text": "I am not sure that is the right conclusion honestly.",
                "emotion": "sadness",
                "emoji": "😢",
                "start": 4.0,
                "end": 9.0,
                "speaker": "Bob",
            },
            {
                "text": "Let us agree on the next action items together.",
                "emotion": "neutral",
                "emoji": "😐",
                "start": 9.0,
                "end": 14.0,
                "speaker": "Alice",
            },
        ]
        result = asr_service.build_intelligent_summary(segments)
        self.assertIn("summary", result)
        self.assertIn("key_points", result)
        insights = result["insights"]
        self.assertIn("dominant_emotion", insights)
        self.assertIn("emotion_distribution", insights)
        self.assertAlmostEqual(sum(insights["emotion_distribution"].values()), 100, delta=2)
        self.assertIn("speaker_stats", insights)
        self.assertEqual(set(insights["speaker_stats"].keys()), {"Alice", "Bob"})
        self.assertEqual(insights["speaker_stats"]["Alice"]["turns"], 2)
        self.assertEqual(insights["total_duration_sec"], 14)
        self.assertGreaterEqual(insights["speaking_pace_wpm"], 0)

    def test_emotional_moments_capped_at_three_and_exclude_neutral(self):
        segments = [
            {"text": f"non neutral sentence number {i} here", "emotion": "anger", "emoji": "😡", "start": i, "end": i + 1, "speaker": "A"}
            for i in range(5)
        ]
        result = asr_service.build_intelligent_summary(segments)
        self.assertEqual(len(result["insights"]["emotional_moments"]), 3)


if __name__ == "__main__":
    unittest.main()
