import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.processing_service import merge_segments, build_transcript_text, _format_time


class MergeSegmentsTests(unittest.TestCase):
    def test_empty_results(self):
        self.assertEqual(merge_segments({}), [])

    def test_flattens_and_sorts_by_start_end(self):
        results = {
            "a": {
                "segments": [
                    {"start": 5, "end": 6, "speaker": "A", "text": " hi ", "emotion": "joy", "emoji": "😄"},
                    {"start": 1, "end": 2, "speaker": "A", "text": "first", "emotion": "neutral", "emoji": "😐"},
                ]
            },
            "b": {
                "segments": [
                    {"start": 1, "end": 3, "speaker": "B", "text": "second", "emotion": "sadness", "emoji": "😢"},
                ]
            },
        }
        merged = merge_segments(results)
        self.assertEqual(len(merged), 3)
        self.assertEqual([m["start"] for m in merged], [1.0, 1.0, 5.0])
        self.assertEqual(merged[0]["end"], 2.0)
        self.assertEqual(merged[1]["end"], 3.0)
        self.assertEqual(merged[0]["text"], "first")
        self.assertIsInstance(merged[0]["start"], float)

    def test_strips_text_and_defaults_missing_fields(self):
        results = {
            "a": {
                "segments": [
                    {"start": "0", "text": "  padded  ", "emotion": "neutral", "emoji": "😐"},
                ]
            }
        }
        merged = merge_segments(results)
        self.assertEqual(merged[0]["text"], "padded")
        self.assertEqual(merged[0]["speaker"], "Unknown")
        self.assertEqual(merged[0]["end"], 0.0)


class FormatTimeTests(unittest.TestCase):
    def test_zero_and_none(self):
        self.assertEqual(_format_time(0), "00:00")
        self.assertEqual(_format_time(None), "00:00")

    def test_minutes_and_seconds(self):
        self.assertEqual(_format_time(65), "01:05")
        self.assertEqual(_format_time(3599), "59:59")

    def test_truncates_fractional_seconds(self):
        self.assertEqual(_format_time(61.9), "01:01")


class BuildTranscriptTextTests(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(build_transcript_text([]), "")

    def test_single_speaker_block(self):
        merged = [
            {"speaker": "Alice", "start": 0.0, "text": "Hello.", "emoji": "😄"},
            {"speaker": "Alice", "start": 2.0, "text": "How are you?", "emoji": "😄"},
        ]
        text = build_transcript_text(merged)
        self.assertTrue(text.startswith("[Alice] (00:00) 😄 Hello. How are you?"))

    def test_speaker_change_creates_new_block(self):
        merged = [
            {"speaker": "Alice", "start": 0.0, "text": "Hi.", "emoji": "😄"},
            {"speaker": "Bob", "start": 3.0, "text": "Hey there.", "emoji": "😢"},
        ]
        text = build_transcript_text(merged)
        blocks = text.split("\n\n")
        self.assertEqual(len(blocks), 2)
        self.assertIn("[Alice]", blocks[0])
        self.assertIn("[Bob]", blocks[1])

    def test_dominant_emoji_is_most_frequent(self):
        merged = [
            {"speaker": "Alice", "start": 0.0, "text": "One.", "emoji": "😄"},
            {"speaker": "Alice", "start": 1.0, "text": "Two.", "emoji": "😄"},
            {"speaker": "Alice", "start": 2.0, "text": "Three.", "emoji": "😢"},
        ]
        text = build_transcript_text(merged)
        self.assertIn("😄", text.splitlines()[0])

    def test_missing_emoji_renders_as_none_string(self):
        merged = [{"speaker": "Alice", "start": 0.0, "text": "Hi.", "emoji": None}]
        text = build_transcript_text(merged)
        self.assertEqual(text, "[Alice] (00:00) None Hi.")


if __name__ == "__main__":
    unittest.main()
