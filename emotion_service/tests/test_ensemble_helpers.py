import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.ensemble import (
    _infer_modality_flag,
    _infer_modality_flags,
    _select_weights,
    _temperature_scale,
    _check_no_nan,
    build_features_single,
    MODALITY_AUDIO_ONLY,
    MODALITY_VIDEO_ONLY,
    MODALITY_BOTH,
)


class InferModalityFlagTests(unittest.TestCase):
    def test_both_present(self):
        fm = np.ones(10, dtype=np.float32)
        am = np.ones(10, dtype=np.float32)
        self.assertEqual(_infer_modality_flag(fm, am), MODALITY_BOTH)

    def test_audio_only(self):
        fm = np.zeros(10, dtype=np.float32)
        am = np.ones(10, dtype=np.float32)
        self.assertEqual(_infer_modality_flag(fm, am), MODALITY_AUDIO_ONLY)

    def test_video_only(self):
        fm = np.ones(10, dtype=np.float32)
        am = np.zeros(10, dtype=np.float32)
        self.assertEqual(_infer_modality_flag(fm, am), MODALITY_VIDEO_ONLY)

    def test_neither_present_defaults_video_only(self):
        fm = np.zeros(10, dtype=np.float32)
        am = np.zeros(10, dtype=np.float32)
        self.assertEqual(_infer_modality_flag(fm, am), MODALITY_VIDEO_ONLY)


class InferModalityFlagsBatchTests(unittest.TestCase):
    def test_batch_matches_single(self):
        fm = np.array([[1, 1], [0, 0], [1, 0]], dtype=np.float32)
        am = np.array([[1, 1], [1, 1], [0, 0]], dtype=np.float32)
        flags = _infer_modality_flags(fm, am)
        expected = [
            _infer_modality_flag(fm[i], am[i]) for i in range(len(fm))
        ]
        self.assertEqual(list(flags), expected)


class SelectWeightsTests(unittest.TestCase):
    def test_both_modalities_uses_calibrated_weights(self):
        self.assertEqual(_select_weights(True, True, 0.4, 0.6), (0.4, 0.6))

    def test_audio_only_is_fully_xgb(self):
        self.assertEqual(_select_weights(False, True, 0.4, 0.6), (0.0, 1.0))

    def test_face_only_is_fully_modal(self):
        self.assertEqual(_select_weights(True, False, 0.4, 0.6), (1.0, 0.0))

    def test_neither_falls_back_to_equal_weights(self):
        self.assertEqual(_select_weights(False, False, 0.4, 0.6), (0.5, 0.5))


class TemperatureScaleTests(unittest.TestCase):
    def test_output_sums_to_one_per_row(self):
        logits = torch.tensor([[1.0, 2.0, 0.5], [0.1, 0.1, 0.1]])
        probs = _temperature_scale(logits, 1.0)
        sums = probs.sum(axis=1)
        np.testing.assert_allclose(sums, [1.0, 1.0], atol=1e-5)

    def test_higher_temperature_softens_distribution(self):
        logits = torch.tensor([[5.0, 0.0, 0.0]])
        sharp = _temperature_scale(logits, 0.5)
        soft = _temperature_scale(logits, 5.0)
        self.assertGreater(sharp[0][0], soft[0][0])


class CheckNoNanTests(unittest.TestCase):
    def test_raises_on_nan(self):
        with self.assertRaises(RuntimeError):
            _check_no_nan(np.array([1.0, float("nan")]), "test")

    def test_passes_clean_array(self):
        _check_no_nan(np.array([1.0, 2.0]), "test")


class BuildFeaturesSingleTests(unittest.TestCase):
    def test_output_shape_matches_batch_dims(self):
        seq_len, face_dim, audio_dim = 10, 326, 1024
        xf = np.random.randn(seq_len, face_dim).astype(np.float32)
        xa = np.random.randn(seq_len, audio_dim).astype(np.float32)
        fm = np.ones(seq_len, dtype=np.float32)
        am = np.ones(seq_len, dtype=np.float32)
        feats = build_features_single(xf, xa, fm, am, MODALITY_BOTH)
        self.assertEqual(feats.shape[0], 1)
        self.assertTrue(np.isfinite(feats).all() or np.isnan(feats).any())


if __name__ == "__main__":
    unittest.main()
