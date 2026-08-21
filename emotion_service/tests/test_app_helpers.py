import io
import os
import sys
import time
import unittest

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as appmod


class ToBytesTests(unittest.TestCase):
    def test_bytes_passthrough(self):
        self.assertEqual(appmod._to_bytes(b"hello"), b"hello")

    def test_bytearray_coerced(self):
        self.assertEqual(appmod._to_bytes(bytearray(b"hi")), b"hi")

    def test_base64_string_decoded(self):
        import base64

        encoded = base64.b64encode(b"payload").decode()
        self.assertEqual(appmod._to_bytes(encoded), b"payload")

    def test_non_base64_string_falls_back_to_latin1(self):
        result = appmod._to_bytes("abc")
        self.assertEqual(result, "abc".encode("latin-1"))

    def test_int_list_coerced(self):
        self.assertEqual(appmod._to_bytes([104, 105]), b"hi")

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            appmod._to_bytes(123)


class DecodeAudioBytesTests(unittest.TestCase):
    def test_valid_wav_decodes_to_float32(self):
        sr = appmod.SAMPLE_RATE
        samples = np.sin(np.linspace(0, 6.28, sr // 10)).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV")
        raw = buf.getvalue()
        decoded = appmod._decode_audio_bytes(raw)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.dtype, np.float32)

    def test_resamples_when_source_rate_differs(self):
        other_sr = 8000
        samples = np.sin(np.linspace(0, 6.28, other_sr // 10)).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, samples, other_sr, format="WAV")
        raw = buf.getvalue()
        decoded = appmod._decode_audio_bytes(raw)
        self.assertIsNotNone(decoded)

    def test_raw_float32_pcm_fallback(self):
        raw = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()
        decoded = appmod._decode_audio_bytes(raw)
        self.assertIsNotNone(decoded)
        self.assertEqual(len(decoded), 4)

    def test_garbage_bytes_return_none(self):
        decoded = appmod._decode_audio_bytes(b"\x01\x02\x03")
        self.assertIsNone(decoded)

    def test_empty_bytes_return_none(self):
        decoded = appmod._decode_audio_bytes(b"")
        self.assertIsNone(decoded)


class ResolveModalityTests(unittest.TestCase):
    def setUp(self):
        self.pid = "test-pid-resolve"
        appmod._PARTICIPANT_MEDIA_STATE.pop(self.pid, None)
        appmod._MODALITY_TIMESTAMPS.pop(self.pid, None)
        appmod._AUDIO_BUFFER.pop(self.pid, None)
        appmod._FACE_BUFFER.pop(self.pid, None)

    def tearDown(self):
        self.setUp()

    def test_no_data_returns_none(self):
        mod_flag, use_face, use_audio = appmod._resolve_modality(self.pid)
        self.assertIsNone(mod_flag)
        self.assertFalse(use_face)
        self.assertFalse(use_audio)

    def test_fresh_audio_only(self):
        appmod._AUDIO_BUFFER[self.pid].append(np.zeros(4))
        appmod._MODALITY_TIMESTAMPS[self.pid]["audio"] = time.monotonic()
        mod_flag, use_face, use_audio = appmod._resolve_modality(self.pid)
        self.assertEqual(mod_flag, appmod.MODALITY_AUDIO_ONLY)
        self.assertTrue(use_audio)
        self.assertFalse(use_face)

    def test_fresh_face_only(self):
        appmod._FACE_BUFFER[self.pid].append(np.zeros(4))
        appmod._MODALITY_TIMESTAMPS[self.pid]["face"] = time.monotonic()
        mod_flag, use_face, use_audio = appmod._resolve_modality(self.pid)
        self.assertEqual(mod_flag, appmod.MODALITY_VIDEO_ONLY)
        self.assertTrue(use_face)
        self.assertFalse(use_audio)

    def test_both_fresh_returns_both(self):
        appmod._AUDIO_BUFFER[self.pid].append(np.zeros(4))
        appmod._FACE_BUFFER[self.pid].append(np.zeros(4))
        appmod._MODALITY_TIMESTAMPS[self.pid]["audio"] = time.monotonic()
        appmod._MODALITY_TIMESTAMPS[self.pid]["face"] = time.monotonic()
        mod_flag, use_face, use_audio = appmod._resolve_modality(self.pid)
        self.assertEqual(mod_flag, appmod.MODALITY_BOTH)
        self.assertTrue(use_face)
        self.assertTrue(use_audio)

    def test_stale_audio_is_ignored(self):
        appmod._AUDIO_BUFFER[self.pid].append(np.zeros(4))
        appmod._MODALITY_TIMESTAMPS[self.pid]["audio"] = (
            time.monotonic() - appmod.MODALITY_STALE_SEC - 10
        )
        mod_flag, use_face, use_audio = appmod._resolve_modality(self.pid)
        self.assertIsNone(mod_flag)
        self.assertFalse(use_audio)

    def test_disabled_mic_ignores_audio_even_if_fresh(self):
        appmod._AUDIO_BUFFER[self.pid].append(np.zeros(4))
        appmod._MODALITY_TIMESTAMPS[self.pid]["audio"] = time.monotonic()
        appmod._PARTICIPANT_MEDIA_STATE[self.pid]["mic"] = False
        mod_flag, use_face, use_audio = appmod._resolve_modality(self.pid)
        self.assertFalse(use_audio)


class SmoothTests(unittest.TestCase):
    def setUp(self):
        self.pid = "test-pid-smooth"
        appmod._EMOTION_HISTORY.pop(self.pid, None)
        appmod._EMOTION_HISTORY_LAST_RESET.pop(self.pid, None)

    def tearDown(self):
        self.setUp()

    def test_first_call_returns_normalised_input(self):
        probs = {"joy": 0.5, "sadness": 0.5}
        smoothed = appmod._smooth(self.pid, probs)
        self.assertAlmostEqual(sum(smoothed.values()), 1.0, places=5)

    def test_repeated_calls_blend_with_history(self):
        appmod._smooth(self.pid, {"joy": 1.0, "sadness": 0.0})
        second = appmod._smooth(self.pid, {"joy": 0.0, "sadness": 1.0})
        self.assertGreater(second["joy"], 0.0)
        self.assertLess(second["joy"], 1.0)

    def test_ttl_expiry_resets_history(self):
        appmod._smooth(self.pid, {"joy": 1.0, "sadness": 0.0})
        appmod._EMOTION_HISTORY_LAST_RESET[self.pid] = (
            time.monotonic() - appmod.EMOTION_HISTORY_TTL - 10
        )
        appmod._smooth(self.pid, {"joy": 0.0, "sadness": 1.0})
        self.assertGreater(
            appmod._EMOTION_HISTORY_LAST_RESET[self.pid],
            time.monotonic() - appmod.EMOTION_HISTORY_TTL,
        )


class ParseLabelTests(unittest.TestCase):
    def setUp(self):
        self.pid = "test-pid-parse-label"
        appmod._EMOTION_HISTORY.pop(self.pid, None)
        appmod._EMOTION_HISTORY_LAST_RESET.pop(self.pid, None)

    def test_uses_probs_when_present(self):
        pred = {"probs": {"joy": 0.9, "sadness": 0.1}}
        label, score = appmod._parse_label(pred, self.pid)
        self.assertEqual(label, "joy")
        self.assertAlmostEqual(score, 0.9, places=3)

    def test_falls_back_to_emotion_field_without_probs(self):
        pred = {"emotion": "sadness", "confidence": 0.8}
        label, score = appmod._parse_label(pred, self.pid + "-b")
        self.assertEqual(label, "sadness")
        self.assertEqual(score, 0.8)

    def test_low_confidence_forces_neutral(self):
        pred = {"probs": {"joy": 0.05, appmod._NEUTRAL_LABEL: 0.95}}
        label, score = appmod._parse_label(pred, self.pid + "-c")
        self.assertEqual(label, appmod._NEUTRAL_LABEL)


class NormFaceAudioTests(unittest.TestCase):
    def test_returns_unchanged_when_no_norm_stats(self):
        original = appmod._norm_stats
        appmod._norm_stats = None
        try:
            xf = np.ones((2, 3), dtype=np.float32)
            fm = np.ones(2, dtype=np.float32)
            out = appmod._norm_face(xf, fm)
            np.testing.assert_array_equal(out, xf)
        finally:
            appmod._norm_stats = original

    def test_zscore_normalisation_applied(self):
        original = appmod._norm_stats
        appmod._norm_stats = {
            "Xf_mean": np.array([1.0, 1.0], dtype=np.float32),
            "Xf_std": np.array([2.0, 2.0], dtype=np.float32),
        }
        try:
            xf = np.array([[3.0, 3.0], [1.0, 1.0]], dtype=np.float32)
            fm = np.array([1.0, 0.0], dtype=np.float32)
            out = appmod._norm_face(xf, fm)
            np.testing.assert_allclose(out[0], [1.0, 1.0], atol=1e-5)
            np.testing.assert_allclose(out[1], [0.0, 0.0], atol=1e-5)
        finally:
            appmod._norm_stats = original


class ParsePayloadTests(unittest.TestCase):
    def setUp(self):
        self.sid = "sid-parse-payload"
        appmod._SID_TO_PID[self.sid] = "pid-parse-payload"

    def tearDown(self):
        appmod._SID_TO_PID.pop(self.sid, None)

    def test_bytes_payload(self):
        raw, pid = appmod._parse_payload(self.sid, b"data")
        self.assertEqual(raw, b"data")
        self.assertEqual(pid, "pid-parse-payload")

    def test_dict_with_buffer_key(self):
        raw, pid = appmod._parse_payload(self.sid, {"buffer": b"xyz"})
        self.assertEqual(raw, b"xyz")

    def test_dict_participant_id_mismatch_is_ignored(self):
        raw, pid = appmod._parse_payload(
            self.sid, {"participantId": "someone-else", "buffer": b"xyz"}
        )
        self.assertEqual(pid, "pid-parse-payload")

    def test_str_payload_treated_as_base64(self):
        import base64

        encoded = base64.b64encode(b"abc").decode()
        raw, pid = appmod._parse_payload(self.sid, encoded)
        self.assertEqual(raw, b"abc")

    def test_unexpected_type_returns_none_raw(self):
        raw, pid = appmod._parse_payload(self.sid, 12345)
        self.assertIsNone(raw)


class ParticipantStateTests(unittest.TestCase):
    def setUp(self):
        self.pid = "pid-state-test"
        appmod._PID_TO_SIDS.pop(self.pid, None)

    def tearDown(self):
        appmod._PID_TO_SIDS.pop(self.pid, None)

    def test_pid_not_connected_by_default(self):
        self.assertFalse(appmod._pid_connected(self.pid))

    def test_pid_connected_after_adding_sid(self):
        appmod._PID_TO_SIDS[self.pid].add("sid-1")
        self.assertTrue(appmod._pid_connected(self.pid))

    def test_count_active_participants(self):
        appmod._PID_TO_SIDS.clear()
        appmod._PID_TO_SIDS["p1"].add("s1")
        appmod._PID_TO_SIDS["p2"].add("s2")
        appmod._PID_TO_SIDS["p3"]
        self.assertEqual(appmod._count_active_participants(), 2)

    def test_cleanup_participant_removes_all_state(self):
        pid = "pid-cleanup-test"
        appmod._FACE_BUFFER[pid].append(1)
        appmod._AUDIO_BUFFER[pid].append(1)
        appmod._LAST_SEEN[pid] = time.time()
        appmod._cleanup_participant(pid)
        self.assertNotIn(pid, appmod._FACE_BUFFER)
        self.assertNotIn(pid, appmod._AUDIO_BUFFER)
        self.assertNotIn(pid, appmod._LAST_SEEN)

    def test_gc_stale_participants_evicts_only_stale_disconnected(self):
        appmod._LAST_SEEN.clear()
        appmod._PID_TO_SIDS.clear()
        appmod._LAST_SEEN["stale-pid"] = time.time() - appmod.BUFFER_TTL - 10
        appmod._LAST_SEEN["fresh-pid"] = time.time()
        appmod._LAST_SEEN["connected-stale-pid"] = time.time() - appmod.BUFFER_TTL - 10
        appmod._PID_TO_SIDS["connected-stale-pid"].add("some-sid")
        appmod._gc_stale_participants()
        self.assertNotIn("stale-pid", appmod._LAST_SEEN)
        self.assertIn("fresh-pid", appmod._LAST_SEEN)
        self.assertIn("connected-stale-pid", appmod._LAST_SEEN)


class LatencyTrackerTests(unittest.TestCase):
    def test_percentiles_computed_on_recorded_samples(self):
        tracker = appmod._LatencyTracker(window=100, report_interval=999)
        for v in [10, 20, 30, 40, 50]:
            tracker.record(v, "audio_only")
        p50 = tracker._percentile(sorted([10, 20, 30, 40, 50]), 50)
        self.assertEqual(p50, 30)

    def test_window_evicts_oldest_samples(self):
        tracker = appmod._LatencyTracker(window=3, report_interval=999)
        for v in [1, 2, 3, 4]:
            tracker.record(v, "both")
        self.assertEqual(len(tracker._samples["both"]), 3)
        self.assertEqual(list(tracker._samples["both"]), [2, 3, 4])


class HealthEndpointTests(unittest.TestCase):
    def test_health_returns_ok_without_model_load(self):
        client = TestClient(appmod.app)
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
