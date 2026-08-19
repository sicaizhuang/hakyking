from __future__ import annotations

from hakyking.qt import QUndoCommand
from hakyking.views.workspace import AudioSliceGraphicsItem, WorkspaceView


def _audio_state_changed(before: dict[str, object], after: dict[str, object]) -> bool:
    keys = (
        "width",
        "height",
        "target_midi_note",
        "target_duration",
        "gain_db",
        "pitch_flatten_amount",
        "formant_shift",
        "protect_transients",
        "pitch_control_points",
        "pitch_vibrato_regions",
    )
    return any(before.get(key) != after.get(key) for key in keys)


class MoveSliceCommand(QUndoCommand):
    def __init__(
        self,
        item: AudioSliceGraphicsItem,
        before: dict[str, object],
        after: dict[str, object],
        render_callback=None,
        initially_applied: bool = True,
    ) -> None:
        super().__init__("Move Slice")
        self.item = item
        self.before = dict(before)
        self.after = dict(after)
        self.render_callback = render_callback
        self._first_redo = initially_applied

    def undo(self) -> None:  # type: ignore[override]
        self.item.apply_edit_state(self.before)
        self._render_if_needed(self.after, self.before)

    def redo(self) -> None:  # type: ignore[override]
        if self._first_redo:
            self._first_redo = False
        else:
            self.item.apply_edit_state(self.after)
        self._render_if_needed(self.before, self.after)

    def _render_if_needed(
        self,
        before: dict[str, object],
        after: dict[str, object],
    ) -> None:
        if self.render_callback is not None and _audio_state_changed(before, after):
            self.render_callback(self.item)


class MoveSlicesCommand(QUndoCommand):
    def __init__(
        self,
        changes: list[tuple[AudioSliceGraphicsItem, dict[str, object], dict[str, object]]],
        render_callback=None,
        post_callback=None,
        initially_applied: bool = True,
        text: str = "Move Slices",
    ) -> None:
        super().__init__(text)
        self.changes = [
            (item, dict(before), dict(after))
            for item, before, after in changes
        ]
        self.render_callback = render_callback
        self.post_callback = post_callback
        self._first_redo = initially_applied

    def undo(self) -> None:  # type: ignore[override]
        for item, before, after in self.changes:
            item.apply_edit_state(before)
            self._render_if_needed(item, after, before)
        self._post()

    def redo(self) -> None:  # type: ignore[override]
        if self._first_redo:
            self._first_redo = False
        else:
            for item, before, after in self.changes:
                item.apply_edit_state(after)
                self._render_if_needed(item, before, after)
        self._post()

    def _render_if_needed(
        self,
        item: AudioSliceGraphicsItem,
        before: dict[str, object],
        after: dict[str, object],
    ) -> None:
        if self.render_callback is not None and _audio_state_changed(before, after):
            self.render_callback(item)

    def _post(self) -> None:
        if self.post_callback is not None:
            self.post_callback()


class ChangeParameterCommand(QUndoCommand):
    def __init__(
        self,
        item: AudioSliceGraphicsItem,
        before: dict[str, object],
        after: dict[str, object],
        render_callback=None,
        initially_applied: bool = False,
    ) -> None:
        super().__init__("Change Slice Parameters")
        self.item = item
        self.before = dict(before)
        self.after = dict(after)
        self.render_callback = render_callback
        self._first_redo = initially_applied

    def undo(self) -> None:  # type: ignore[override]
        self.item.apply_edit_state(self.before)
        self._render()

    def redo(self) -> None:  # type: ignore[override]
        if self._first_redo:
            self._first_redo = False
        else:
            self.item.apply_edit_state(self.after)
        self._render()

    def _render(self) -> None:
        if self.render_callback is not None:
            self.render_callback(self.item)


class MaterialSlicesCommand(QUndoCommand):
    def __init__(
        self,
        path: str,
        before: list[object],
        after: list[object],
        apply_callback,
        initially_applied: bool = True,
    ) -> None:
        super().__init__("Edit Material Slices")
        self.path = str(path)
        self.before = list(before)
        self.after = list(after)
        self.apply_callback = apply_callback
        self._first_redo = initially_applied

    def undo(self) -> None:  # type: ignore[override]
        self.apply_callback(self.path, self.before)

    def redo(self) -> None:  # type: ignore[override]
        if self._first_redo:
            self._first_redo = False
            return
        self.apply_callback(self.path, self.after)


class SplitSliceCommand(QUndoCommand):
    def __init__(
        self,
        workspace: WorkspaceView,
        item: AudioSliceGraphicsItem,
        local_x: float,
        render_callback=None,
    ) -> None:
        super().__init__("Split Slice")
        self.workspace = workspace
        self.local_x = float(local_x)
        self.render_callback = render_callback
        self.source_snapshot = workspace.snapshot_item(item)
        self.source_item: AudioSliceGraphicsItem | None = item
        self.left_item: AudioSliceGraphicsItem | None = None
        self.right_item: AudioSliceGraphicsItem | None = None
        self._applied = False

    def undo(self) -> None:  # type: ignore[override]
        if not self._applied:
            return
        self.workspace.remove_slice_item(self.left_item)
        self.workspace.remove_slice_item(self.right_item)
        self.left_item = None
        self.right_item = None
        self.source_item = self.workspace.restore_item_snapshot(self.source_snapshot)
        self.source_item.setSelected(True)
        self._applied = False
        self._render_if_needed(self.source_item)

    def redo(self) -> None:  # type: ignore[override]
        if self.source_item is None or self.source_item.scene() is not self.workspace.scene():
            self.source_item = self.workspace.restore_item_snapshot(self.source_snapshot)
        new_items = self.workspace.split_slice_item(self.source_item, self.local_x)
        if len(new_items) != 2:
            return
        self.left_item, self.right_item = new_items
        self.source_item = None
        self._applied = True
        self._render_if_needed(self.left_item)
        self._render_if_needed(self.right_item)

    def _render_if_needed(self, item: AudioSliceGraphicsItem | None) -> None:
        if item is not None and self.render_callback is not None:
            self.render_callback(item)


class BoundaryMoveCommand(QUndoCommand):
    def __init__(
        self,
        workspace: WorkspaceView,
        items: list[AudioSliceGraphicsItem],
        before_snapshots: list[dict[str, object]],
        after_snapshots: list[dict[str, object]],
        render_callback=None,
        post_callback=None,
        initially_applied: bool = True,
    ) -> None:
        super().__init__("Move Slice Boundary")
        self.workspace = workspace
        self.items = list(items)
        self.before_snapshots = [dict(snapshot) for snapshot in before_snapshots]
        self.after_snapshots = [dict(snapshot) for snapshot in after_snapshots]
        self.render_callback = render_callback
        self.post_callback = post_callback
        self._first_redo = initially_applied

    def undo(self) -> None:  # type: ignore[override]
        self._remove_current_items()
        self.items = self._restore(self.before_snapshots)
        self._post()

    def redo(self) -> None:  # type: ignore[override]
        if self._first_redo:
            self._first_redo = False
        else:
            self._remove_current_items()
            self.items = self._restore(self.after_snapshots)
        self._post()

    def _remove_current_items(self) -> None:
        for item in list(self.items):
            self.workspace.remove_slice_item(item)
        self.items = []

    def _restore(
        self,
        snapshots: list[dict[str, object]],
    ) -> list[AudioSliceGraphicsItem]:
        restored = [
            self.workspace.restore_item_snapshot(snapshot)
            for snapshot in snapshots
        ]
        for item in restored:
            item.setSelected(True)
            if self.render_callback is not None:
                self.render_callback(item)
        return restored

    def _post(self) -> None:
        if self.post_callback is not None:
            self.post_callback()


class MergeSlicesCommand(QUndoCommand):
    def __init__(
        self,
        workspace: WorkspaceView,
        items: list[AudioSliceGraphicsItem],
        before_snapshots: list[dict[str, object]],
        after_snapshots: list[dict[str, object]],
        render_callback=None,
        post_callback=None,
        initially_applied: bool = True,
    ) -> None:
        super().__init__("Merge Slices")
        self.workspace = workspace
        self.items = list(items)
        self.before_snapshots = [dict(snapshot) for snapshot in before_snapshots]
        self.after_snapshots = [dict(snapshot) for snapshot in after_snapshots]
        self.render_callback = render_callback
        self.post_callback = post_callback
        self._first_redo = initially_applied

    def undo(self) -> None:  # type: ignore[override]
        self._remove_current_items()
        self.items = self._restore(self.before_snapshots)
        self._post()

    def redo(self) -> None:  # type: ignore[override]
        if self._first_redo:
            self._first_redo = False
        else:
            self._remove_current_items()
            self.items = self._restore(self.after_snapshots)
        self._post()

    def _remove_current_items(self) -> None:
        for item in list(self.items):
            self.workspace.remove_slice_item(item)
        self.items = []

    def _restore(
        self,
        snapshots: list[dict[str, object]],
    ) -> list[AudioSliceGraphicsItem]:
        restored = [
            self.workspace.restore_item_snapshot(snapshot)
            for snapshot in snapshots
        ]
        for item in restored:
            item.setSelected(True)
            if self.render_callback is not None:
                self.render_callback(item)
        return restored

    def _post(self) -> None:
        if self.post_callback is not None:
            self.post_callback()


class DeleteSliceCommand(QUndoCommand):
    def __init__(
        self,
        workspace: WorkspaceView,
        items: list[AudioSliceGraphicsItem],
    ) -> None:
        super().__init__("Delete Slice")
        self.workspace = workspace
        self.snapshots = [
            workspace.snapshot_item(item)
            for item in sorted(items, key=lambda candidate: candidate.scenePos().x())
        ]
        self.items: list[AudioSliceGraphicsItem] = list(items)

    def undo(self) -> None:  # type: ignore[override]
        self.items = [
            self.workspace.restore_item_snapshot(snapshot)
            for snapshot in self.snapshots
        ]
        for item in self.items:
            item.setSelected(True)

    def redo(self) -> None:  # type: ignore[override]
        for item in list(self.items):
            self.workspace.remove_slice_item(item)
        self.items = []


class AddSliceCommand(QUndoCommand):
    def __init__(
        self,
        workspace: WorkspaceView,
        items: list[AudioSliceGraphicsItem],
        render_callback=None,
        post_callback=None,
        initially_applied: bool = True,
    ) -> None:
        super().__init__("Add Slice")
        self.workspace = workspace
        self.snapshots = [
            workspace.snapshot_item(item)
            for item in sorted(items, key=lambda candidate: candidate.scenePos().x())
        ]
        self.items: list[AudioSliceGraphicsItem] = list(items)
        self.render_callback = render_callback
        self.post_callback = post_callback
        self._first_redo = initially_applied

    def undo(self) -> None:  # type: ignore[override]
        for item in list(self.items):
            self.workspace.remove_slice_item(item)
        self.items = []
        self._post()

    def redo(self) -> None:  # type: ignore[override]
        if self._first_redo:
            self._first_redo = False
        else:
            self.items = [
                self.workspace.restore_item_snapshot(snapshot)
                for snapshot in self.snapshots
            ]
        for item in self.items:
            item.setSelected(True)
            if self.render_callback is not None:
                self.render_callback(item)
        self._post()

    def _post(self) -> None:
        if self.post_callback is not None:
            self.post_callback()


class PasteSliceCommand(QUndoCommand):
    def __init__(
        self,
        workspace: WorkspaceView,
        snapshots: list[dict[str, object]],
        render_callback=None,
        text: str = "Paste Slice",
    ) -> None:
        super().__init__(text)
        self.workspace = workspace
        self.snapshots = [dict(snapshot) for snapshot in snapshots]
        self.render_callback = render_callback
        self.items: list[AudioSliceGraphicsItem] = []

    def undo(self) -> None:  # type: ignore[override]
        for item in list(self.items):
            self.workspace.remove_slice_item(item)
        self.items = []

    def redo(self) -> None:  # type: ignore[override]
        self.items = [
            self.workspace.restore_item_snapshot(snapshot)
            for snapshot in self.snapshots
        ]
        for item in self.items:
            item.setSelected(True)
            if self.render_callback is not None:
                self.render_callback(item)
