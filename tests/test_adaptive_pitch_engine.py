from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from hakyking.app_settings import DEFAULT_PITCH_ENGINE, normalize_pitch_engine
from hakyking.audio.audio_engine import process_blob


class AdaptivePitchEngineTests(unittest.TestCase):
    def test_adaptive_is_default_and_aliases_normalize(self) -> None:
        self.assertEqual(DEFAULT_PITCH_ENGINE, "adaptive")
        self.assertEqual(normalize_pitch_engine("auto"), "adaptive")
        self.assertEqual(normalize_pitch_engine("hybrid"), "adaptive")

    def test_adaptive_uses_psola_for_small_and_rubberband_for_large_shift(self) -> None:
        source = np.linspace(-0.2, 0.2, 2048, dtype=np.float32)
        old_engine = os.environ.get("HAKYKING_PITCH_ENGINE")
        os.environ["HAKYKING_PITCH_ENGINE"] = "adaptive"
        try:
            with (
                patch(
                    "hakyking.audio.audio_engine._parselmouth_psola_pitch_time_process",
                    return_value=source.copy(),
                ) as psola,
                patch(
                    "hakyking.audio.audio_engine._rubberband_pitch_shift",
                    return_value=source.copy(),
                ) as rubberband,
            ):
                process_blob(source, 44100, n_steps=2.0)
                self.assertEqual(psola.call_count, 1)
                self.assertEqual(rubberband.call_count, 0)

                process_blob(source, 44100, n_steps=12.0)
                self.assertEqual(rubberband.call_count, 1)
        finally:
            if old_engine is None:
                os.environ.pop("HAKYKING_PITCH_ENGINE", None)
            else:
                os.environ["HAKYKING_PITCH_ENGINE"] = old_engine

    def test_slice_gain_is_linear_and_silence_keeps_length(self) -> None:
        source = np.full(1024, 0.01, dtype=np.float32)
        gained = process_blob(source, 44100, gain_db=18.0)
        expected_ratio = 10.0 ** (18.0 / 20.0)
        ratio = float(np.sqrt(np.mean(gained**2)) / np.sqrt(np.mean(source**2)))
        self.assertAlmostEqual(ratio, expected_ratio, places=4)

        silence = np.zeros(2048, dtype=np.float32)
        rendered = process_blob(silence, 44100, n_steps=-12.0, rate=0.5)
        self.assertEqual(rendered.size, 4096)
        self.assertTrue(np.all(rendered == 0.0))


if __name__ == "__main__":
    unittest.main()
