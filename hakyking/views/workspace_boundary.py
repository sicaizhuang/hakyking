from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from hakyking.models.audio_slice import AudioSlice


Snapshot = dict[str, object]
ReplaceAudioSlice = Callable[[Any, AudioSlice], None]


@dataclass
class BoundaryDragState:
    """State and math for moving a cut line between two adjacent slices.

    This intentionally models the gesture as "merge the original adjacent
    source interval, then re-cut it at the new point" instead of as two
    independent edge-resize edits.
    """

    left_item: Any
    right_item: Any
    before: list[Snapshot]
    start_x: float
    left_x: float
    left_y: float
    right_y: float
    total_width: float
    height_left: float
    height_right: float
    source_start: float
    source_end: float

    @classmethod
    def from_items(
        cls,
        left_item: Any,
        right_item: Any,
        before: list[Snapshot],
        scene_x: float,
    ) -> BoundaryDragState:
        return cls(
            left_item=left_item,
            right_item=right_item,
            before=[dict(snapshot) for snapshot in before],
            start_x=float(scene_x),
            left_x=float(left_item.scenePos().x()),
            left_y=float(left_item.scenePos().y()),
            right_y=float(right_item.scenePos().y()),
            total_width=float(left_item.rect().width() + right_item.rect().width()),
            height_left=float(left_item.rect().height()),
            height_right=float(right_item.rect().height()),
            source_start=float(left_item.audio_slice.start_time),
            source_end=float(right_item.audio_slice.end_time),
        )

    def items(self) -> list[Any]:
        return [self.left_item, self.right_item]

    def set_edit_notifications_suppressed(self, suppressed: bool) -> None:
        for item in self.items():
            item._suppress_edit_notifications = bool(suppressed)

    def preview(self, scene_x: float, replace_audio_slice: ReplaceAudioSlice) -> None:
        left_width, right_width, split_time = self._split_geometry(scene_x)
        self.left_item.setRect(0, 0, left_width, self.height_left)
        self.right_item.setRect(0, 0, right_width, self.height_right)
        self.right_item.setPos(self.left_x + left_width, self.right_y)

        replace_audio_slice(
            self.left_item,
            AudioSlice(
                source_path=self.left_item.audio_slice.source_path,
                index=self.left_item.audio_slice.index,
                start_time=self.source_start,
                end_time=split_time,
                midi_note=self.left_item.audio_slice.midi_note,
                f0_hz=self.left_item.audio_slice.f0_hz,
            ),
        )
        replace_audio_slice(
            self.right_item,
            AudioSlice(
                source_path=self.right_item.audio_slice.source_path,
                index=self.right_item.audio_slice.index,
                start_time=split_time,
                end_time=self.source_end,
                midi_note=self.right_item.audio_slice.midi_note,
                f0_hz=self.right_item.audio_slice.f0_hz,
            ),
        )

    def after_snapshots(self, pixels_per_second: float) -> list[Snapshot] | None:
        if len(self.before) != 2:
            return None
        left_width, right_width, split_time = self._split_geometry(
            self.left_x + float(self.left_item.rect().width())
        )
        source_start = float(self.before[0]["original_start"])
        source_end = float(self.before[1]["original_end"])
        split_time = max(source_start + 0.001, min(source_end - 0.001, split_time))
        left_after = dict(self.before[0])
        right_after = dict(self.before[1])

        left_after.update(
            {
                "original_start": source_start,
                "original_end": split_time,
                "x": self.left_x,
                "y": self.left_y,
                "width": left_width,
                "height": self.height_left,
                "target_duration": max(0.001, left_width / pixels_per_second),
            }
        )
        right_after.update(
            {
                "original_start": split_time,
                "original_end": source_end,
                "x": self.left_x + left_width,
                "y": self.right_y,
                "width": right_width,
                "height": self.height_right,
                "target_duration": max(0.001, right_width / pixels_per_second),
            }
        )
        return [left_after, right_after]

    def _split_geometry(self, scene_x: float) -> tuple[float, float, float]:
        min_width = max(self.left_item.MIN_WIDTH, self.right_item.MIN_WIDTH)
        left_width = max(
            min_width,
            min(self.total_width - min_width, float(scene_x) - self.left_x),
        )
        right_width = max(min_width, self.total_width - left_width)
        if left_width + right_width > self.total_width:
            scale = self.total_width / max(1.0, left_width + right_width)
            left_width *= scale
            right_width *= scale

        ratio = left_width / max(1.0, self.total_width)
        split_time = self.source_start + (self.source_end - self.source_start) * ratio
        split_time = max(self.source_start + 0.001, min(self.source_end - 0.001, split_time))
        return left_width, right_width, split_time
