from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from hakyking.audio.exporter import ExportClip, export_mixdown_to_wav
from hakyking.audio.gain import soft_limit_audio
from hakyking.audio.playback import prepare_playback_audio


class MixdownParityTests(unittest.TestCase):
    def test_soft_limiter_is_block_independent_below_threshold(self) -> None:
        source = np.asarray([-0.91, -0.2, 0.0, 0.2, 0.91, 1.4], dtype=np.float32)
        whole = soft_limit_audio(source)
        blocks = np.concatenate(
            [soft_limit_audio(source[:3]), soft_limit_audio(source[3:])]
        )
        np.testing.assert_array_equal(whole, blocks)
        np.testing.assert_array_equal(whole[:5], source[:5])
        self.assertLessEqual(float(np.max(np.abs(whole))), 0.98001)

    def test_export_uses_same_edge_fade_and_master_limiter_as_playback(self) -> None:
        sample_rate = 8_000
        first = np.full(800, 0.7, dtype=np.float32)
        second = np.full(800, 0.7, dtype=np.float32)
        clips = [
            ExportClip(0.0, first, sample_rate, 0),
            ExportClip(0.05, second, sample_rate, 1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "mix.wav"
            export_mixdown_to_wav(
                clips,
                str(output_path),
                sample_rate=sample_rate,
                fade_ms=5.0,
            )
            exported, exported_rate = sf.read(output_path, dtype="float32")

        expected = np.zeros(1_200, dtype=np.float32)
        prepared_first = prepare_playback_audio(
            first, sample_rate, fade_ms=5.0, limit_output=False
        )
        prepared_second = prepare_playback_audio(
            second, sample_rate, fade_ms=5.0, limit_output=False
        )
        expected[:800] += prepared_first
        expected[400:1_200] += prepared_second
        expected = soft_limit_audio(expected)

        self.assertEqual(exported_rate, sample_rate)
        np.testing.assert_allclose(exported, expected, atol=4.0 / 32768.0)


if __name__ == "__main__":
    unittest.main()
