from __future__ import annotations

import unittest

from hakyking.models.audio_edit import AudioSliceEditModel
from hakyking.models.audio_slice import AudioSlice


class AudioEditModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audio_slice = AudioSlice(
            source_path="hakimi.wav",
            index=2,
            start_time=1.0,
            end_time=2.5,
            midi_note=60,
            f0_hz=261.6256,
        )
        self.model = AudioSliceEditModel.create(self.audio_slice, track_index=1)

    def test_control_points_do_not_change_audio_region(self) -> None:
        before = (self.model.clip.source_start, self.model.clip.source_end)
        self.model.pitch_automation.set_control_points(
            [
                {"x": 0.75, "offset": -1.0},
                {"x": 0.25, "offset": 2.0},
            ]
        )
        self.assertEqual(before, (self.model.clip.source_start, self.model.clip.source_end))
        self.assertEqual(
            self.model.pitch_automation.control_point_pairs(),
            [(0.25, 2.0), (0.75, -1.0)],
        )

    def test_vibrato_is_parametric_and_separate_from_control_points(self) -> None:
        self.model.pitch_automation.set_control_points([{"x": 0.5, "offset": 1.0}])
        self.model.pitch_automation.set_vibrato_regions(
            [
                {
                    "start": 0.2,
                    "end": 0.8,
                    "cycles": 3.5,
                    "depth": 0.4,
                    "phase": 0.5,
                    "waveform": "triangle",
                }
            ]
        )
        request = self.model.build_render_request()
        self.assertEqual(request.pitch_control_points, ((0.5, 1.0),))
        self.assertEqual(request.pitch_vibrato_regions[0][-1], "triangle")
        self.assertEqual(len(self.model.pitch_automation.control_points), 1)

    def test_project_payload_round_trip(self) -> None:
        self.model.clip.timeline_start = 4.25
        self.model.clip.target_duration = 2.0
        self.model.clip.pitch_center_midi = 64
        self.model.pitch_automation.set_control_points([{"x": 0.4, "offset": 1.25}])
        self.model.pitch_automation.set_vibrato_regions(
            [{"start": 0.3, "end": 0.7, "cycles": 2, "depth": 0.2}]
        )
        self.model.gain_db = 3.0

        restored = AudioSliceEditModel.create(self.audio_slice, track_index=0)
        restored.load_project_payload(self.model.to_project_payload())

        self.assertEqual(restored.clip.track_index, 1)
        self.assertAlmostEqual(restored.clip.timeline_start, 4.25)
        self.assertAlmostEqual(restored.clip.target_duration or 0.0, 2.0)
        self.assertEqual(restored.clip.pitch_center_midi, 64)
        self.assertEqual(restored.pitch_automation.control_point_pairs(), [(0.4, 1.25)])
        self.assertEqual(len(restored.pitch_automation.vibrato_regions), 1)
        self.assertAlmostEqual(restored.gain_db, 3.0)


if __name__ == "__main__":
    unittest.main()
