from __future__ import annotations

import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hakyking.models.audio_slice import AudioSlice
from hakyking.qt import QApplication, Qt
from hakyking.views.material_browser import SlicePitchMapWidget
from hakyking.views.workspace import WorkspaceView


def make_slices(count: int, source: str = "long.wav") -> list[AudioSlice]:
    return [
        AudioSlice(
            source_path=source,
            index=index,
            start_time=index * 0.1,
            end_time=(index + 1) * 0.1,
            midi_note=60 + index % 5,
            f0_hz=261.6256,
            analysis_backend="librosa_yin_fast",
        )
        for index in range(count)
    ]


class ProgressiveUiPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_material_pitch_map_renders_large_results_in_batches(self) -> None:
        widget = SlicePitchMapWidget()
        slices = make_slices(300)
        widget.set_slices(slices)

        immediate_count = sum(
            isinstance(item.data(Qt.UserRole), AudioSlice) for item in widget.scene().items()
        )
        self.assertLess(immediate_count, len(slices))
        self.assertEqual(widget.slices(), slices)

        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            self.app.processEvents()
            rendered = sum(
                isinstance(item.data(Qt.UserRole), AudioSlice) for item in widget.scene().items()
            )
            if rendered == len(slices):
                break
        self.assertEqual(rendered, len(slices))
        widget.close()

    def test_new_material_generation_discards_old_render_queue(self) -> None:
        widget = SlicePitchMapWidget()
        widget.set_slices(make_slices(300, "old.wav"))
        current = make_slices(1, "current.wav")
        widget.set_slices(current)
        for _ in range(10):
            self.app.processEvents()

        rendered = [
            item.data(Qt.UserRole)
            for item in widget.scene().items()
            if isinstance(item.data(Qt.UserRole), AudioSlice)
        ]
        self.assertEqual(rendered, current)
        widget.close()

    def test_workspace_chunk_placement_keeps_global_source_time(self) -> None:
        workspace = WorkspaceView()
        slices = make_slices(130)
        first = workspace.add_slice_items(
            slices[:64],
            100.0,
            200.0,
            source_first_start=0.0,
        )
        group_id = first[0].placement_group_id
        second = workspace.add_slice_items(
            slices[64:],
            100.0,
            200.0,
            source_first_start=0.0,
            placement_group_id=group_id,
        )
        items = sorted([*first, *second], key=lambda item: item.audio_slice.index)

        self.assertEqual({item.placement_group_id for item in items}, {group_id})
        self.assertAlmostEqual(items[0].scenePos().x(), 100.0)
        self.assertGreater(items[-1].scenePos().x(), items[64].scenePos().x())
        expected_last_x = 100.0 + slices[-1].start_time * workspace.pixels_per_second()
        self.assertAlmostEqual(items[-1].scenePos().x(), expected_last_x)
        workspace.close()


if __name__ == "__main__":
    unittest.main()
