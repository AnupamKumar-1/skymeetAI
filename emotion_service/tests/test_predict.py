import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = Path(__file__).resolve().parent.parent

from inference.predict import (
    EmotionPredictor,
    validate_inputs,
    coerce,
    InputValidationError,
    error_response,
    _get_modality,
    load_npy,
    SEQ_LEN,
    FACE_DIM,
    AUDIO_DIM,
)


def _load_sample(tag):
    d = ROOT / "sample_inputs"
    xf = np.load(d / f"xf_{tag}.npy")
    xa = np.load(d / f"xa_{tag}.npy")
    fm = np.load(d / f"fm_{tag}.npy")
    am = np.load(d / f"am_{tag}.npy")
    return xf, xa, fm, am


class ValidateInputsTests(unittest.TestCase):
    def _valid(self):
        xf = np.zeros((SEQ_LEN, FACE_DIM), dtype=np.float32)
        xa = np.zeros((SEQ_LEN, AUDIO_DIM), dtype=np.float32)
        fm = np.ones(SEQ_LEN, dtype=np.float32)
        am = np.ones(SEQ_LEN, dtype=np.float32)
        return xf, xa, fm, am

    def test_valid_inputs_pass(self):
        xf, xa, fm, am = self._valid()
        validate_inputs(xf, xa, fm, am)

    def test_wrong_shape_raises(self):
        xf, xa, fm, am = self._valid()
        bad_xf = np.zeros((SEQ_LEN, FACE_DIM + 1), dtype=np.float32)
        with self.assertRaises(InputValidationError):
            validate_inputs(bad_xf, xa, fm, am)

    def test_non_binary_mask_raises(self):
        xf, xa, fm, am = self._valid()
        fm = fm.copy()
        fm[0] = 0.5
        with self.assertRaises(InputValidationError):
            validate_inputs(xf, xa, fm, am)

    def test_all_zero_masks_raise(self):
        xf, xa, fm, am = self._valid()
        fm = np.zeros(SEQ_LEN, dtype=np.float32)
        am = np.zeros(SEQ_LEN, dtype=np.float32)
        with self.assertRaises(InputValidationError):
            validate_inputs(xf, xa, fm, am)

    def test_nan_in_features_raises(self):
        xf, xa, fm, am = self._valid()
        xf = xf.copy()
        xf[0, 0] = float("nan")
        with self.assertRaises(InputValidationError):
            validate_inputs(xf, xa, fm, am)

    def test_inf_in_features_raises(self):
        xf, xa, fm, am = self._valid()
        xa = xa.copy()
        xa[0, 0] = float("inf")
        with self.assertRaises(InputValidationError):
            validate_inputs(xf, xa, fm, am)


class CoerceTests(unittest.TestCase):
    def test_casts_to_float32(self):
        xf = np.zeros((2, 2), dtype=np.float64)
        xa = np.zeros((2, 2), dtype=np.float64)
        fm = np.ones(2, dtype=np.int32)
        am = np.ones(2, dtype=np.int32)
        out_xf, out_xa, out_fm, out_am = coerce(xf, xa, fm, am)
        for arr in (out_xf, out_xa, out_fm, out_am):
            self.assertEqual(arr.dtype, np.float32)


class GetModalityTests(unittest.TestCase):
    def test_both(self):
        self.assertEqual(_get_modality(np.ones(3), np.ones(3)), "both")

    def test_audio_only(self):
        self.assertEqual(_get_modality(np.zeros(3), np.ones(3)), "audio_only")

    def test_video_only(self):
        self.assertEqual(_get_modality(np.ones(3), np.zeros(3)), "video_only")

    def test_none(self):
        self.assertEqual(_get_modality(np.zeros(3), np.zeros(3)), "none")


class ErrorResponseTests(unittest.TestCase):
    def test_shape_and_status(self):
        resp = error_response("bad input")
        self.assertEqual(resp["status"], "error")
        self.assertEqual(resp["error"], "bad input")
        self.assertIsNone(resp["emotion"])
        self.assertIsNone(resp["confidence"])


class LoadNpyTests(unittest.TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_npy(str(ROOT / "sample_inputs" / "does_not_exist.npy"))

    def test_existing_file_loads(self):
        arr = load_npy(str(ROOT / "sample_inputs" / "xf_angry_0.npy"))
        self.assertIsInstance(arr, np.ndarray)


@unittest.skipUnless(
    (ROOT / "models" / "modal" / "best_modal.pt").exists(),
    "committed model weights not present",
)
class EmotionPredictorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = EmotionPredictor()
        with open(ROOT / "sample_inputs" / "manifest.json") as f:
            cls.manifest = json.load(f)

    def test_predict_returns_ok_status_for_all_sample_tags(self):
        for entry in self.manifest:
            tag = entry["tag"]
            xf, xa, fm, am = _load_sample(tag)
            result = self.predictor.predict(xf, xa, fm, am)
            self.assertEqual(result["status"], "ok", msg=f"tag={tag} error={result.get('error')}")
            self.assertIn(result["emotion"], self.predictor.ensemble.class_names)
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)
            self.assertEqual(result["modality"], "both")
            self.assertIsInstance(result["probs"], dict)
            probs_sum = sum(result["probs"].values())
            self.assertAlmostEqual(probs_sum, 1.0, delta=0.02)

    def test_predict_is_deterministic_for_same_input(self):
        xf, xa, fm, am = _load_sample("angry_0")
        r1 = self.predictor.predict(xf, xa, fm, am)
        r2 = self.predictor.predict(xf, xa, fm, am)
        self.assertEqual(r1["emotion"], r2["emotion"])
        self.assertAlmostEqual(r1["confidence"], r2["confidence"], places=5)

    def test_predict_handles_invalid_shapes_gracefully(self):
        bad_xf = np.zeros((3, 3), dtype=np.float32)
        bad_xa = np.zeros((3, 3), dtype=np.float32)
        bad_fm = np.ones(3, dtype=np.float32)
        bad_am = np.ones(3, dtype=np.float32)
        result = self.predictor.predict(bad_xf, bad_xa, bad_fm, bad_am)
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["emotion"])

    def test_audio_only_modality_uses_full_xgb_weight(self):
        xf, xa, fm, am = _load_sample("happy_0")
        zero_fm = np.zeros_like(fm)
        result = self.predictor.predict(xf, xa, zero_fm, am)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["modality"], "audio_only")

    def test_video_only_modality_uses_full_modal_weight(self):
        xf, xa, fm, am = _load_sample("sad_0")
        zero_am = np.zeros_like(am)
        result = self.predictor.predict(xf, xa, fm, zero_am)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["modality"], "video_only")


if __name__ == "__main__":
    unittest.main()
