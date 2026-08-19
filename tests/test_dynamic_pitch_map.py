from __future__ import annotations

import unittest
import shutil

import numpy as np

from hakyking.audio.audio_engine import (
    _build_dynamic_pitch_map,
    _normalize_pitch_control_points,
    _normalize_pitch_vibrato_regions,
    _rubberband_dynamic_pitch_process,
)


class DynamicPitchMapTests(unittest.TestCase):
    def test_map_covers_both_endpoints_and_control_curve(self) -> None:
        points = _normalize_pitch_control_points([(0.0, -1.0), (1.0, 2.0)])
        entries = _build_dynamic_pitch_map(48_000, 48_000, points, ())
        self.assertEqual(entries[0][0], 0)
        self.assertEqual(entries[-1][0], 47_999)
        self.assertAlmostEqual(entries[0][1], -1.0, places=4)
        self.assertAlmostEqual(entries[-1][1], 2.0, places=4)

    def test_waveforms_generate_distinct_dynamic_maps(self) -> None:
        values = {}
        for waveform in ("sine", "triangle", "square"):
            regions = _normalize_pitch_vibrato_regions(
                [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "cycles": 2.25,
                        "depth": 1.0,
                        "phase": 0.13,
                        "waveform": waveform,
                    }
                ]
            )
            entries = _build_dynamic_pitch_map(44_100, 44_100, (), regions)
            values[waveform] = tuple(round(offset, 4) for _frame, offset in entries)
        self.assertNotEqual(values["sine"], values["triangle"])
        self.assertNotEqual(values["triangle"], values["square"])
        self.assertNotEqual(values["sine"], values["square"])

    @unittest.skipUnless(
        shutil.which("rubberband") or shutil.which("rubberband.exe"),
        "Rubber Band is an optional external runtime dependency",
    )
    def test_rubberband_accepts_dynamic_pitch_map(self) -> None:
        sample_rate = 16_000
        time_axis = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
        source = (0.2 * np.sin(2.0 * np.pi * 220.0 * time_axis)).astype(np.float32)
        output = _rubberband_dynamic_pitch_process(
            source,
            sample_rate,
            ((0.0, 0.0), (1.0, 2.0)),
            (),
        )
        self.assertEqual(output.shape, source.shape)
        self.assertTrue(bool(np.all(np.isfinite(output))))
        self.assertGreater(float(np.max(np.abs(output))), 0.01)
        self.assertGreater(float(np.mean(np.abs(output - source))), 0.005)


if __name__ == "__main__":
    unittest.main()
