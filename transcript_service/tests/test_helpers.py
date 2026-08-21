import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.helpers import allowed_file, clean_speaker, schedule_file_cleanup


class AllowedFileTests(unittest.TestCase):
    def test_allowed_extension(self):
        self.assertTrue(allowed_file("audio.wav", {"wav", "mp3"}))

    def test_disallowed_extension(self):
        self.assertFalse(allowed_file("audio.exe", {"wav", "mp3"}))

    def test_case_insensitive(self):
        self.assertTrue(allowed_file("audio.WAV", {"wav"}))

    def test_multiple_dots_uses_last_segment(self):
        self.assertTrue(allowed_file("my.file.name.mp3", {"mp3"}))


class CleanSpeakerTests(unittest.TestCase):
    def test_empty_name_returns_guest(self):
        self.assertEqual(clean_speaker(""), "Guest")
        self.assertEqual(clean_speaker(None), "Guest")

    def test_unknown_and_undefined_case_insensitive(self):
        self.assertEqual(clean_speaker("unknown"), "Guest")
        self.assertEqual(clean_speaker("Unknown"), "Guest")
        self.assertEqual(clean_speaker("UNDEFINED"), "Guest")

    def test_valid_name_passthrough(self):
        self.assertEqual(clean_speaker("Alice"), "Alice")


class ScheduleFileCleanupTests(unittest.TestCase):
    def test_deletes_existing_files_after_delay(self):
        path = os.path.join(os.path.dirname(__file__), "_cleanup_target.tmp")
        with open(path, "w") as f:
            f.write("x")
        self.assertTrue(os.path.exists(path))

        schedule_file_cleanup([path], 0.05)
        time.sleep(0.3)

        self.assertFalse(os.path.exists(path))

    def test_missing_file_does_not_raise(self):
        path = os.path.join(os.path.dirname(__file__), "_does_not_exist.tmp")
        schedule_file_cleanup([path], 0.02)
        time.sleep(0.2)


if __name__ == "__main__":
    unittest.main()
