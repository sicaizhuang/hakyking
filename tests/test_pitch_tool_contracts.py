from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hakyking.commands import ChangeParameterCommand
from hakyking.models.audio_slice import AudioSlice
from hakyking.qt import QApplication, QColor, QPointF, Qt, QUndoStack
from hakyking.views.workspace import AudioSliceGraphicsItem, WorkspaceView


class PitchToolContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.workspace = WorkspaceView()
        self.workspace.set_pitch_curve_edit_mode(True)
        audio_slice = AudioSlice(
            source_path="acceptance.wav",
            index=0,
            start_time=0.0,
            end_time=1.0,
            midi_note=60,
            f0_hz=261.6256,
        )
        self.item = AudioSliceGraphicsItem(
            audio_slice=audio_slice,
            track_index=0,
            width=400.0,
            height=20.0,
            color=QColor("#55aaff"),
        )
        self.workspace.scene().addItem(self.item)
        self.item.set_pitch_curve_edit_mode(True)
        self.item.pitch_contour = np.full(32, 60.0, dtype=np.float64)
        self.item.pitch_curve_center_midi = 60.0
        self.item.set_pitch_control_points(
            [{"x": 0.2, "offset": 0.0}, {"x": 0.8, "offset": 0.0}]
        )
        self.undo_stack = QUndoStack()
        self.workspace.slice_parameter_changed.connect(self._push_change)

    def tearDown(self) -> None:
        self.workspace.close()

    def _push_change(self, item, before, after) -> None:
        self.undo_stack.push(
            ChangeParameterCommand(
                item=item,
                before=dict(before),
                after=dict(after),
                initially_applied=True,
            )
        )

    def _point(self, index: int) -> QPointF:
        ratio, offset = self.item.pitch_control_points[index]
        return self.item._pitch_control_point_to_local(ratio, offset)

    def test_v_selects_points_and_derives_line_range_without_adding_points(self) -> None:
        self.workspace.set_pitch_curve_tool_mode("curve_select")
        original_points = list(self.item.pitch_control_points)

        self.assertTrue(
            self.item.handle_pitch_curve_tool_press(self._point(0), Qt.NoModifier)
        )
        self.item.finish_pitch_control_drag()
        self.assertFalse(self.item.has_selected_pitch_curve_range())

        self.assertTrue(
            self.item.handle_pitch_curve_tool_press(
                self._point(1), Qt.ControlModifier
            )
        )
        self.item.finish_pitch_control_drag()
        self.assertEqual(self.item.selected_pitch_curve_ranges(), [(0.2, 0.8)])

        midpoint = self.item._pitch_curve_point_at_ratio(0.5)
        self.assertIsNotNone(midpoint)
        self.assertFalse(
            self.item.handle_pitch_curve_tool_press(midpoint, Qt.NoModifier)
        )
        self.assertEqual(self.item.pitch_control_points, original_points)

    def test_v_vertical_move_is_undoable(self) -> None:
        self.workspace.set_pitch_curve_tool_mode("curve_select")
        original_points = list(self.item.pitch_control_points)
        start = self._point(0)
        self.assertTrue(
            self.item.handle_pitch_curve_tool_press(start, Qt.NoModifier)
        )
        self.item._update_pitch_control_drag(QPointF(start.x(), start.y() - 28.0))
        self.item.finish_pitch_control_drag()

        moved_points = list(self.item.pitch_control_points)
        self.assertNotEqual(moved_points, original_points)
        self.assertEqual(self.undo_stack.count(), 1)
        self.undo_stack.undo()
        self.assertEqual(self.item.pitch_control_points, original_points)
        self.undo_stack.redo()
        self.assertEqual(self.item.pitch_control_points, moved_points)

    def test_v_horizontal_move_preserves_multi_selection_and_is_one_undo(self) -> None:
        self.workspace.set_pitch_curve_tool_mode("curve_select")
        original_points = list(self.item.pitch_control_points)
        self.item._select_pitch_control_point(0)
        self.item._select_pitch_control_point(1, additive=True)
        start = self._point(0)
        self.assertTrue(
            self.item.handle_pitch_curve_tool_press(start, Qt.NoModifier)
        )
        self.item._pitch_control_drag_axis = "x"
        self.item._update_pitch_control_drag(QPointF(start.x() + 40.0, start.y()))
        self.item.finish_pitch_control_drag()

        moved_points = list(self.item.pitch_control_points)
        self.assertEqual(len(self.item._selected_pitch_control_indices), 2)
        self.assertAlmostEqual(
            moved_points[1][0] - moved_points[0][0],
            original_points[1][0] - original_points[0][0],
            places=6,
        )
        self.assertEqual(self.undo_stack.count(), 1)
        self.undo_stack.undo()
        self.assertEqual(self.item.pitch_control_points, original_points)
        self.undo_stack.redo()
        self.assertEqual(self.item.pitch_control_points, moved_points)

    def test_render_requirement_covers_gain_curve_and_vibrato(self) -> None:
        self.assertFalse(self.item.requires_rendered_audio())
        self.item.set_gain_db(3.0)
        self.assertTrue(self.item.requires_rendered_audio())
        self.item.set_gain_db(0.0)
        self.item.set_pitch_control_points([{"x": 0.5, "offset": 1.0}])
        self.assertTrue(self.item.requires_rendered_audio())
        self.item.set_pitch_control_points([{"x": 0.5, "offset": 0.0}])
        self.assertFalse(self.item.requires_rendered_audio())
        self.item.set_pitch_vibrato_regions(
            [{"start": 0.2, "end": 0.8, "cycles": 3.0, "depth": 0.5}]
        )
        self.assertTrue(self.item.requires_rendered_audio())

    def test_b_only_adds_or_deletes_points_and_both_are_undoable(self) -> None:
        self.workspace.set_pitch_curve_tool_mode("curve_point")
        existing = self._point(0)
        self.assertTrue(
            self.item.handle_pitch_curve_tool_press(existing, Qt.NoModifier)
        )
        self.assertFalse(self.item._selected_pitch_control_indices)

        midpoint = self.item._pitch_curve_point_at_ratio(0.5)
        self.assertIsNotNone(midpoint)
        self.assertTrue(
            self.item.handle_pitch_curve_tool_press(midpoint, Qt.NoModifier)
        )
        self.assertEqual(len(self.item.pitch_control_points), 3)
        self.assertEqual(self.undo_stack.count(), 1)
        self.undo_stack.undo()
        self.assertEqual(len(self.item.pitch_control_points), 2)
        self.undo_stack.redo()
        self.assertEqual(len(self.item.pitch_control_points), 3)

        middle_index = min(
            range(len(self.item.pitch_control_points)),
            key=lambda index: abs(self.item.pitch_control_points[index][0] - 0.5),
        )
        self.assertTrue(
            self.item._delete_pitch_control_point_at(self._point(middle_index))
        )
        self.assertEqual(len(self.item.pitch_control_points), 2)
        self.undo_stack.undo()
        self.assertEqual(len(self.item.pitch_control_points), 3)

    def test_n_requires_selected_range_and_only_writes_vibrato_region(self) -> None:
        self.workspace.set_pitch_curve_tool_mode("curve_vibrato")
        scene_midpoint = self.item.mapToScene(
            self.item._pitch_curve_point_at_ratio(0.5)
        )
        self.assertFalse(
            self.workspace._begin_pitch_vibrato_drag(
                QPointF(100.0, 100.0), scene_midpoint, target_hint=self.item
            )
        )

        self.item._select_pitch_control_point(0)
        self.item._select_pitch_control_point(1, additive=True)
        original_points = list(self.item.pitch_control_points)
        self.assertTrue(
            self.workspace._begin_pitch_vibrato_drag(
                QPointF(100.0, 100.0), scene_midpoint, target_hint=self.item
            )
        )
        self.workspace._update_pitch_vibrato_drag(QPointF(156.0, 148.0))
        self.workspace._finish_pitch_vibrato_drag(commit=True)

        self.assertEqual(self.item.pitch_control_points, original_points)
        self.assertEqual(len(self.item.pitch_vibrato_regions), 1)
        self.assertEqual(self.undo_stack.count(), 1)
        self.undo_stack.undo()
        self.assertEqual(self.item.pitch_vibrato_regions, [])
        self.undo_stack.redo()
        self.assertEqual(len(self.item.pitch_vibrato_regions), 1)

    def test_split_preserves_curve_value_and_vibrato_phase_at_boundary(self) -> None:
        self.item.set_pitch_control_points(
            [{"x": 0.2, "offset": 0.0}, {"x": 0.8, "offset": 2.0}]
        )
        self.item.set_pitch_vibrato_regions(
            [
                {
                    "start": 0.2,
                    "end": 0.8,
                    "cycles": 3.5,
                    "depth": 0.7,
                    "phase": 0.1,
                    "waveform": "sine",
                }
            ]
        )
        expected_offset = self.item._pitch_control_offset_at_ratio(0.5)
        children = self.workspace.split_slice_item(
            self.item,
            self.item.rect().width() * 0.5,
        )
        self.assertEqual(len(children), 2)
        left, right = children
        self.assertAlmostEqual(left.pitch_control_points[-1][0], 1.0)
        self.assertAlmostEqual(right.pitch_control_points[0][0], 0.0)
        self.assertAlmostEqual(left.pitch_control_points[-1][1], expected_offset)
        self.assertAlmostEqual(right.pitch_control_points[0][1], expected_offset)
        self.assertEqual(len(left.pitch_vibrato_regions), 1)
        self.assertEqual(len(right.pitch_vibrato_regions), 1)
        self.assertAlmostEqual(float(left.pitch_vibrato_regions[0]["cycles"]), 1.75)
        self.assertAlmostEqual(float(right.pitch_vibrato_regions[0]["cycles"]), 1.75)
        self.assertAlmostEqual(float(right.pitch_vibrato_regions[0]["phase"]), 0.85)


if __name__ == "__main__":
    unittest.main()
