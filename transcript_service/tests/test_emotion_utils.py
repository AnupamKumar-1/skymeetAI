import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.emotion import normalize_emotion, get_emoji, EMOJI_MAP


class NormalizeEmotionTests(unittest.TestCase):
    def test_happy_maps_to_joy(self):
        self.assertEqual(normalize_emotion("happy"), "joy")
        self.assertEqual(normalize_emotion("happiness"), "joy")

    def test_sad_maps_to_sadness(self):
        self.assertEqual(normalize_emotion("sad"), "sadness")

    def test_disgust_passthrough(self):
        self.assertEqual(normalize_emotion("disgust"), "disgust")

    def test_unknown_label_lowercased_unchanged(self):
        self.assertEqual(normalize_emotion("Anger"), "anger")
        self.assertEqual(normalize_emotion("SURPRISE"), "surprise")

    def test_case_insensitive_input(self):
        self.assertEqual(normalize_emotion("HAPPY"), "joy")
        self.assertEqual(normalize_emotion("Sad"), "sadness")


class GetEmojiTests(unittest.TestCase):
    def test_known_labels(self):
        for label, emoji in EMOJI_MAP.items():
            self.assertEqual(get_emoji(label), emoji)

    def test_case_insensitive_lookup(self):
        self.assertEqual(get_emoji("JOY"), EMOJI_MAP["joy"])

    def test_unknown_label_falls_back_to_neutral_emoji(self):
        self.assertEqual(get_emoji("confused"), "😐")


if __name__ == "__main__":
    unittest.main()
