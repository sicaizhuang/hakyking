from __future__ import annotations

import re
import zlib
from pathlib import Path

import numpy as np

from hakyking.dnd import (
    MIME_AUDIO_FILE,
    MIME_AUDIO_SLICES,
    encode_audio_file,
    encode_audio_slices,
)
from hakyking.models.audio_slice import AudioSlice, copy_audio_slice
from hakyking.models.material_file_system import MaterialFileSystemModel
from hakyking.qt import (
    QApplication,
    QByteArray,
    QBrush,
    QColor,
    QDrag,
    QEvent,
    QFileDialog,
    QFont,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QCheckBox,
    QHBoxLayout,
    QIcon,
    QLabel,
    QMenu,
    QMimeData,
    QPainter,
    QPen,
    QPixmap,
    QRect,
    QPushButton,
    QRubberBand,
    QSize,
    QSplitter,
    QToolButton,
    QTreeView,
    QTimer,
    Qt,
    Signal,
    QVBoxLayout,
    QWidget,
)
from hakyking.views.workspace import _workspace_cursor


class PreviewScrubBar(QWidget):
    """Compact video-style timeline with click and drag seeking."""

    position_previewed = Signal(float)
    seek_requested = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._duration = 0.0
        self._position = 0.0
        self._dragging = False
        self.setMinimumHeight(34)
        self.setMouseTracking(True)
        self.setCursor(_workspace_cursor("app_hand"))

    def set_duration(self, duration: float) -> None:
        self._duration = max(0.0, float(duration))
        self._position = min(self._position, self._duration)
        self.update()

    def set_position(self, position: float) -> None:
        self._position = max(0.0, min(float(position), self._duration))
        self.update()

    def position(self) -> float:
        return self._position

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        width = max(1, self.width())
        height = max(1, self.height())
        track_y = height // 2
        painter.fillRect(0, 0, width, height, QColor("#202124"))
        painter.setPen(QPen(QColor("#3b414b"), 1))
        painter.drawLine(8, track_y, width - 8, track_y)

        if self._duration > 0:
            ratio = self._position / self._duration
            x = int(8 + ratio * max(1, width - 16))
            painter.setPen(QPen(QColor("#3f8ec5"), 3))
            painter.drawLine(8, track_y, x, track_y)
            painter.setPen(QPen(QColor("#7fb6de"), 1))
            painter.setBrush(QColor("#7fb6de"))
            painter.drawEllipse(x - 3, track_y - 3, 6, 6)
        else:
            painter.setPen(QPen(QColor("#676b72"), 1))
        painter.end()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.LeftButton or self._duration <= 0:
            super().mousePressEvent(event)
            return
        self._dragging = True
        self._preview_event_position(event)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._dragging and event.buttons() & Qt.LeftButton:
            self._preview_event_position(event)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton and self._dragging:
            self._preview_event_position(event)
            self._dragging = False
            self.seek_requested.emit(self._position)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _preview_event_position(self, event) -> None:
        x = max(8, min(self.width() - 8, event.pos().x()))
        ratio = (x - 8) / max(1, self.width() - 16)
        self.set_position(ratio * self._duration)
        self.position_previewed.emit(self._position)


class MaterialPreviewPlayerWidget(QWidget):
    """Video-like player strip shown below the material parsing area."""

    play_requested = Signal(str, float)
    stop_requested = Signal()
    position_previewed = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._language = "zh"
        self._path = ""
        self._duration = 0.0
        self._playing = False
        self.setFixedHeight(44)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(5)

        self.play_button = QPushButton("▶")
        self.play_button.setFixedSize(34, 26)
        self.play_button.setEnabled(False)
        layout.addWidget(self.play_button)

        self.name_label = QLabel("未加载媒体")
        self.name_label.setVisible(False)
        self.name_label.setMinimumWidth(0)
        self.name_label.setWordWrap(False)
        layout.addWidget(self.name_label, 0)

        self.scrub_bar = PreviewScrubBar()
        self.scrub_bar.setMinimumHeight(30)
        layout.addWidget(self.scrub_bar, 2)

        self.time_label = QLabel("0:00")
        self.time_label.setFixedWidth(42)
        self.time_label.setVisible(False)
        layout.addWidget(self.time_label)

        self.play_button.clicked.connect(self._on_play_clicked)
        self.scrub_bar.position_previewed.connect(self._on_scrub_previewed)
        self.scrub_bar.seek_requested.connect(self._on_scrub_committed)

    def set_language(self, language: str) -> None:
        self._language = "zh" if language == "zh" else "en"
        if not self._path:
            self.name_label.setText("未加载媒体" if self._language == "zh" else "No Media")
        self.play_button.setToolTip("播放源媒体预览" if self._language == "zh" else "Play source preview")
        self._refresh_button_text()

    def set_media(self, path: str, duration: float | None = None) -> None:
        self._path = path
        self._duration = max(0.0, float(duration or 0.0))
        self._playing = False
        self.play_button.setEnabled(bool(path))
        self.name_label.setText(Path(path).name if path else ("未加载媒体" if self._language == "zh" else "No Media"))
        self.name_label.setToolTip(path)
        self.scrub_bar.set_duration(self._duration)
        self.scrub_bar.set_position(0.0)
        self._refresh_time_text()
        self._refresh_button_text()

    def update_duration(self, duration: float) -> None:
        self._duration = max(0.0, float(duration))
        self.scrub_bar.set_duration(self._duration)
        self._refresh_time_text()

    def set_position(self, position: float) -> None:
        self.scrub_bar.set_position(position)
        self._refresh_time_text()

    def position(self) -> float:
        return self.scrub_bar.position()

    def set_playing(self, playing: bool) -> None:
        self._playing = bool(playing)
        self._refresh_button_text()

    def _on_play_clicked(self) -> None:
        if not self._path:
            return
        if self._playing:
            self.stop_requested.emit()
            return
        self.play_requested.emit(self._path, self.position())

    def toggle_playback(self) -> None:
        self._on_play_clicked()

    def _on_scrub_previewed(self, position: float) -> None:
        self.set_position(position)
        self.position_previewed.emit(position)

    def _on_scrub_committed(self, position: float) -> None:
        self.set_position(position)
        self.position_previewed.emit(position)
        if self._playing and self._path:
            self.play_requested.emit(self._path, position)

    def _refresh_button_text(self) -> None:
        if self._playing:
            self.play_button.setText("■")
            self.play_button.setToolTip("停止源媒体预览" if self._language == "zh" else "Stop source preview")
        else:
            self.play_button.setText("▶")
            self.play_button.setToolTip("播放源媒体预览" if self._language == "zh" else "Play source preview")

    def _refresh_time_text(self) -> None:
        self.time_label.setText(
            self._format_time(self.position())
        )
        self.time_label.setToolTip(
            f"{self._format_time(self.position())}/{self._format_time(self._duration)}"
        )

    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        minutes = int(seconds // 60)
        whole_seconds = int(seconds % 60)
        return f"{minutes}:{whole_seconds:02d}"


class SlicePitchBlockItem(QGraphicsRectItem):
    """Lightweight material-slice block positioned by source time and pitch."""

    def __init__(self, audio_slice: AudioSlice, width: float, height: float) -> None:
        super().__init__(0, 0, width, height)
        self.audio_slice = audio_slice
        self.setData(Qt.UserRole, audio_slice)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.setAcceptHoverEvents(True)
        self.setBrush(QBrush(self._slice_color(audio_slice)))
        self.setPen(QPen(QColor("#d8e1ec"), 1))
        self.setToolTip(
            f"{audio_slice.source_path}\n"
            f"{audio_slice.start_time:.3f}s - {audio_slice.end_time:.3f}s\n"
            f"MIDI: {audio_slice.midi_note if audio_slice.midi_note is not None else 'N/A'}"
        )

    def contextMenuEvent(self, event) -> None:  # type: ignore[override]
        menu = QMenu()
        for index, line in enumerate(self.info_lines()):
            action = menu.addAction(line)
            action.setEnabled(False)
            if index == 0:
                menu.addSeparator()
        if hasattr(menu, "exec"):
            menu.exec(event.screenPos())
        else:
            menu.exec_(event.screenPos())
        event.accept()

    def info_lines(self) -> list[str]:
        f0_text = "N/A" if self.audio_slice.f0_hz is None else f"{self.audio_slice.f0_hz:.2f} Hz"
        confidence_text = (
            "N/A"
            if self.audio_slice.pitch_confidence is None
            else f"{self.audio_slice.pitch_confidence:.2f}"
        )
        backend_text = self.audio_slice.analysis_backend or "legacy"
        return [
            "源媒体片段属性",
            f"源文件: {Path(self.audio_slice.source_path).name}",
            f"片段编号: {self.audio_slice.index + 1}",
            f"源区间: {self.audio_slice.start_time:.3f}s - {self.audio_slice.end_time:.3f}s",
            f"时长: {self.audio_slice.duration:.3f}s",
            f"音高: {self.audio_slice.note_name}",
            f"MIDI: {self.audio_slice.midi_note if self.audio_slice.midi_note is not None else 'N/A'}",
            f"F0: {f0_text}",
            f"Analysis: {backend_text} / {confidence_text}",
            "增益: 原始媒体 0.0 dB",
        ]

    def paint(self, painter, option, widget=None) -> None:  # type: ignore[override]
        super().paint(painter, option, widget)
        rect = self.rect()
        if self.isSelected():
            painter.save()
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 3, 3)
            painter.restore()

    def _slice_color(self, audio_slice: AudioSlice) -> QColor:
        source_shift = zlib.crc32(audio_slice.source_path.encode("utf-8", errors="ignore")) % 360
        if audio_slice.midi_note is None:
            color = QColor()
            color.setHsv(source_shift, 45, 185)
            return color
        color = QColor()
        color.setHsv(int(((audio_slice.midi_note % 12) * 30 + source_shift * 0.35) % 360), 115, 220)
        return color


class SlicePitchMapWidget(QGraphicsView):
    """Mini pitch workspace for analyzed material slices.

    X maps to source time. Y maps to MIDI pitch. No waveform is loaded here, so
    browsing parsed material remains cheap even for large source files.
    """

    slice_preview_requested = Signal(object)
    playback_toggled = Signal()
    playhead_seek_requested = Signal(float)
    slices_edited = Signal(object)

    PIXELS_PER_SECOND = 96.0
    ROW_HEIGHT = 16.0
    BLOCK_HEIGHT = 11.0
    LEFT_GUTTER = 42.0
    TOP_RULER = 20.0
    MIN_BLOCK_WIDTH = 18.0
    ASYNC_RENDER_THRESHOLD = 160
    RENDER_BATCH_SIZE = 64

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._drag_start_position = None
        self._drag_slices_at_press: list[AudioSlice] = []
        self._drag_started_on_slice = False
        self._rubber_band_origin = None
        self._rubber_band = QRubberBand(QRubberBand.Rectangle, self.viewport())
        self._slices: list[AudioSlice] = []
        self._midi_min = 48
        self._midi_max = 72
        self._playhead_time = 0.0
        self._playhead_item = None
        self._slice_render_generation = 0
        self._slice_render_queue: list[AudioSlice] = []
        self.tool_mode = "select"
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setBackgroundBrush(QBrush(QColor("#202124")))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._refresh_tool_cursor()

    def set_slices(self, slices: list[AudioSlice]) -> None:
        self._slices = list(slices)
        self._slice_render_generation += 1
        generation = self._slice_render_generation
        self._slice_render_queue = list(self._slices)
        self.scene().clear()
        self._playhead_item = None
        self._configure_pitch_range(self._slices)
        self._draw_background(self._slices)
        self.set_playhead_time(self._playhead_time)
        if len(self._slice_render_queue) <= self.ASYNC_RENDER_THRESHOLD:
            while self._slice_render_queue:
                self._add_slice_graphics_item(self._slice_render_queue.pop(0))
            return
        self._render_next_slice_batch(generation)

    def _add_slice_graphics_item(self, audio_slice: AudioSlice) -> None:
        width = max(
            self.MIN_BLOCK_WIDTH,
            audio_slice.duration * self.PIXELS_PER_SECOND,
        )
        item = SlicePitchBlockItem(audio_slice, width, self.BLOCK_HEIGHT)
        item.setPos(self._x_for_time(audio_slice.start_time), self._y_for_slice(audio_slice))
        self.scene().addItem(item)

    def _render_next_slice_batch(self, generation: int) -> None:
        if generation != self._slice_render_generation:
            return
        batch = self._slice_render_queue[: self.RENDER_BATCH_SIZE]
        del self._slice_render_queue[: self.RENDER_BATCH_SIZE]
        for audio_slice in batch:
            self._add_slice_graphics_item(audio_slice)
        if self._slice_render_queue:
            QTimer.singleShot(0, lambda token=generation: self._render_next_slice_batch(token))

    def set_playhead_time(self, seconds: float) -> None:
        self._playhead_time = max(0.0, float(seconds))
        scene = self.scene()
        if self._playhead_item is None:
            self._playhead_item = scene.addLine(0, 0, 0, 1, QPen(QColor("#ff4a4a"), 2))
            self._playhead_item.setZValue(40)
        rect = scene.sceneRect()
        x = self._x_for_time(self._playhead_time)
        self._playhead_item.setLine(x, 0, x, max(1.0, rect.height()))

    def selected_audio_slices(self) -> list[AudioSlice]:
        slices: list[AudioSlice] = []
        for item in self.scene().selectedItems():
            audio_slice = item.data(Qt.UserRole)
            if isinstance(audio_slice, AudioSlice):
                slices.append(audio_slice)
        return sorted(
            slices,
            key=lambda audio_slice: (
                audio_slice.source_path,
                audio_slice.start_time,
                audio_slice.end_time,
                audio_slice.index,
            ),
        )

    def slices(self) -> list[AudioSlice]:
        return list(self._slices)

    def set_tool_mode(self, mode: str) -> None:
        self.tool_mode = mode if mode in {"select", "split", "fit_merge"} else "select"
        self._refresh_tool_cursor()

    def _effective_tool_mode(self, modifiers=None) -> str:
        if modifiers is not None:
            if modifiers & Qt.AltModifier and modifiers & Qt.ShiftModifier:
                return "fit_merge"
            if self.tool_mode == "split" and modifiers & Qt.AltModifier:
                return "fit_merge"
            if self.tool_mode == "fit_merge" and modifiers & Qt.AltModifier:
                return "split"
            if modifiers & Qt.AltModifier:
                return "split"
        return self.tool_mode

    def _refresh_tool_cursor(self) -> None:
        mode = self._effective_tool_mode(QApplication.keyboardModifiers())
        if mode == "split":
            self.viewport().setCursor(_workspace_cursor("scissors"))
        elif mode == "fit_merge":
            self.viewport().setCursor(_workspace_cursor("cut_merge"))
        else:
            self.viewport().setCursor(_workspace_cursor("move"))

    def split_at_playhead(self) -> bool:
        if not self._slices:
            return False

        split_time = float(self._playhead_time)
        candidates = self.selected_audio_slices()
        containing = [
            audio_slice
            for audio_slice in (candidates or self._slices)
            if audio_slice.start_time + 0.001 < split_time < audio_slice.end_time - 0.001
        ]
        if not containing:
            containing = [
                audio_slice
                for audio_slice in self._slices
                if audio_slice.start_time + 0.001 < split_time < audio_slice.end_time - 0.001
            ]
        if not containing:
            return False

        target = min(containing, key=lambda audio_slice: audio_slice.duration)
        new_slices: list[AudioSlice] = []
        for audio_slice in self._slices:
            if audio_slice is not target:
                new_slices.append(audio_slice)
                continue
            new_slices.append(
                copy_audio_slice(
                    audio_slice,
                    index=audio_slice.index,
                    start_time=audio_slice.start_time,
                    end_time=split_time,
                )
            )
            new_slices.append(
                copy_audio_slice(
                    audio_slice,
                    index=audio_slice.index + 1,
                    start_time=split_time,
                    end_time=audio_slice.end_time,
                )
            )

        reindexed = self._refine_manual_slice_pitches([
            copy_audio_slice(
                audio_slice,
                index=index,
                start_time=audio_slice.start_time,
                end_time=audio_slice.end_time,
            )
            for index, audio_slice in enumerate(
                sorted(new_slices, key=lambda item: (item.start_time, item.end_time))
            )
        ])
        self.set_slices(reindexed)
        self.set_playhead_time(split_time)
        for item in self.scene().items():
            audio_slice = item.data(Qt.UserRole)
            if not isinstance(audio_slice, AudioSlice):
                continue
            if abs(audio_slice.start_time - target.start_time) < 1e-6 or abs(audio_slice.end_time - target.end_time) < 1e-6:
                item.setSelected(True)
        return True

    def fit_or_merge_selected(self) -> str | None:
        selected = self.selected_audio_slices()
        if len(selected) != 2:
            return None
        left, right = sorted(selected, key=lambda item: (item.start_time, item.end_time))
        if left.source_path != right.source_path or left is right:
            return None

        tolerance = 0.001
        if abs(left.end_time - right.start_time) <= tolerance:
            replacement = copy_audio_slice(
                left,
                index=left.index,
                start_time=min(left.start_time, right.start_time),
                end_time=max(left.end_time, right.end_time),
            )
            result = "merged"
        else:
            boundary = (left.end_time + right.start_time) / 2.0
            left = copy_audio_slice(
                left,
                index=left.index,
                start_time=left.start_time,
                end_time=boundary,
            )
            right = copy_audio_slice(
                right,
                index=right.index,
                start_time=boundary,
                end_time=right.end_time,
            )
            replacement = None
            result = "fitted"

        selected_ids = {
            (item.source_path, item.index, item.start_time, item.end_time)
            for item in selected
        }
        new_slices = [
            item
            for item in self._slices
            if (item.source_path, item.index, item.start_time, item.end_time) not in selected_ids
        ]
        if replacement is not None:
            new_slices.append(replacement)
            selection_ranges = [(replacement.start_time, replacement.end_time)]
        else:
            new_slices.extend((left, right))
            selection_ranges = [
                (left.start_time, left.end_time),
                (right.start_time, right.end_time),
            ]

        reindexed = self._refine_manual_slice_pitches([
            copy_audio_slice(
                item,
                index=index,
                start_time=item.start_time,
                end_time=item.end_time,
            )
            for index, item in enumerate(sorted(new_slices, key=lambda value: (value.start_time, value.end_time)))
        ])
        self.set_slices(reindexed)
        for item in self.scene().items():
            audio_slice = item.data(Qt.UserRole)
            if not isinstance(audio_slice, AudioSlice):
                continue
            if any(
                abs(audio_slice.start_time - start) < 1e-6
                and abs(audio_slice.end_time - end) < 1e-6
                for start, end in selection_ranges
            ):
                item.setSelected(True)
        return result

    def cancel_split_at_position(self, position) -> bool:
        scene_position = self.mapToScene(position)
        margin = 7.0
        ordered = sorted(self._slices, key=lambda item: (item.start_time, item.end_time))
        for left, right in zip(ordered, ordered[1:]):
            if left.source_path != right.source_path:
                continue
            if abs(left.end_time - right.start_time) > 0.001:
                continue
            boundary_x = self._x_for_time(left.end_time)
            if abs(scene_position.x() - boundary_x) > margin:
                continue
            top = min(self._y_for_slice(left), self._y_for_slice(right)) - margin
            bottom = max(
                self._y_for_slice(left) + self.BLOCK_HEIGHT,
                self._y_for_slice(right) + self.BLOCK_HEIGHT,
            ) + margin
            if not (top <= scene_position.y() <= bottom):
                continue
            merged = copy_audio_slice(
                left,
                index=left.index,
                start_time=left.start_time,
                end_time=right.end_time,
            )
            new_slices = [
                item
                for item in self._slices
                if item is not left and item is not right
            ]
            new_slices.append(merged)
            reindexed = self._refine_manual_slice_pitches([
                copy_audio_slice(
                    item,
                    index=index,
                    start_time=item.start_time,
                    end_time=item.end_time,
                )
                for index, item in enumerate(
                    sorted(new_slices, key=lambda value: (value.start_time, value.end_time))
                )
            ])
            self.set_slices(reindexed)
            for item in self.scene().items():
                audio_slice = item.data(Qt.UserRole)
                if not isinstance(audio_slice, AudioSlice):
                    continue
                if (
                    abs(audio_slice.start_time - merged.start_time) < 1e-6
                    and abs(audio_slice.end_time - merged.end_time) < 1e-6
                ):
                    item.setSelected(True)
            return True
        return False

    def _refine_manual_slice_pitches(self, slices: list[AudioSlice]) -> list[AudioSlice]:
        refined: list[AudioSlice] = []
        source_cache: dict[str, tuple[object, int]] = {}
        for audio_slice in slices:
            try:
                from hakyking.audio.reader import AudioReader
                from hakyking.audio.slicer import _estimate_initial_f0, _hz_to_midi

                if audio_slice.source_path not in source_cache:
                    source_cache[audio_slice.source_path] = AudioReader.load_mono(audio_slice.source_path)
                audio, sample_rate = source_cache[audio_slice.source_path]
                start_sample = max(0, int(round(audio_slice.start_time * sample_rate)))
                end_sample = min(len(audio), int(round(audio_slice.end_time * sample_rate)))
                if end_sample <= start_sample:
                    refined.append(audio_slice)
                    continue
                f0_hz = _estimate_initial_f0(audio[start_sample:end_sample], sample_rate)
                midi_note = _hz_to_midi(f0_hz) if f0_hz is not None else audio_slice.midi_note
                refined.append(
                    copy_audio_slice(
                        audio_slice,
                        midi_note=midi_note,
                        f0_hz=f0_hz if f0_hz is not None else audio_slice.f0_hz,
                        analysis_backend=(
                            "manual-start-f0"
                            if f0_hz is not None
                            else audio_slice.analysis_backend
                        ),
                    )
                )
            except Exception:
                refined.append(audio_slice)
        return refined

    def _configure_pitch_range(self, slices: list[AudioSlice]) -> None:
        midi_notes = [item.midi_note for item in slices if item.midi_note is not None]
        visible_rows = max(
            30,
            int(
                max(260.0, float(self.viewport().height()) - self.TOP_RULER - 36.0)
                / self.ROW_HEIGHT
            ),
        )
        if not midi_notes:
            center = 60
            half = max(15, visible_rows // 2)
            self._midi_min = max(0, center - half)
            self._midi_max = min(127, center + half)
            return
        center = int(round(float(np.median(np.asarray(midi_notes, dtype=np.float32)))))
        span = max(30, visible_rows, int(max(midi_notes) - min(midi_notes) + 10))
        half = max(15, span // 2)
        self._midi_min = max(0, center - half)
        self._midi_max = min(127, center + half)
        if self._midi_min == 0:
            self._midi_max = min(127, max(self._midi_max, 30))
        if self._midi_max == 127:
            self._midi_min = max(0, min(self._midi_min, 97))

    def _draw_background(self, slices: list[AudioSlice]) -> None:
        scene = self.scene()
        max_time = max((item.end_time for item in slices), default=4.0)
        width = max(560.0, self._x_for_time(max_time) + 140.0)
        height = max(
            float(self.viewport().height()) - 4.0,
            self.TOP_RULER + (self._midi_max - self._midi_min + 1) * self.ROW_HEIGHT + 46.0,
        )
        scene.setSceneRect(0, 0, width, height)

        font = QFont("Segoe UI", 7)
        for midi_note in range(self._midi_min, self._midi_max + 1):
            y = self._y_for_midi(midi_note) + self.BLOCK_HEIGHT / 2.0
            is_c = midi_note % 12 == 0
            pen = QPen(QColor("#34383f") if is_c else QColor("#2a2d33"), 1)
            scene.addLine(self.LEFT_GUTTER, y, width, y, pen).setZValue(-10)
            if is_c:
                label = scene.addText(self._note_name(midi_note), font)
                label.setDefaultTextColor(QColor("#9ba7b4"))
                label.setPos(6, y - 8)
                label.setZValue(-5)

        second_count = int(max_time) + 2
        for second in range(second_count + 1):
            x = self._x_for_time(float(second))
            scene.addLine(
                x,
                0,
                x,
                height,
                QPen(QColor("#333841"), 1),
            ).setZValue(-9)
            label = scene.addText(f"{second}s", font)
            label.setDefaultTextColor(QColor("#9ba7b4"))
            label.setPos(x + 3, 2)
            label.setZValue(-4)

    def _x_for_time(self, seconds: float) -> float:
        first_start = min((item.start_time for item in self._slices), default=0.0)
        return self.LEFT_GUTTER + max(0.0, seconds - first_start) * self.PIXELS_PER_SECOND

    def _time_for_x(self, x: float) -> float:
        first_start = min((item.start_time for item in self._slices), default=0.0)
        return max(
            0.0,
            first_start + (float(x) - self.LEFT_GUTTER) / self.PIXELS_PER_SECOND,
        )

    def _y_for_slice(self, audio_slice: AudioSlice) -> float:
        midi_note = audio_slice.midi_note
        if midi_note is None:
            midi_note = int(round((self._midi_min + self._midi_max) / 2.0))
        return self._y_for_midi(midi_note)

    def _y_for_midi(self, midi_note: int) -> float:
        row = self._midi_max - max(self._midi_min, min(self._midi_max, midi_note))
        return self.TOP_RULER + row * self.ROW_HEIGHT + 1.5

    def _note_name(self, midi_note: int) -> str:
        names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
        return f"{names[midi_note % 12]}{midi_note // 12 - 1}"

    def _audio_slice_at(self, position) -> AudioSlice | None:
        item = self._audio_slice_item_at(position)
        if item is None:
            return None
        audio_slice = item.data(Qt.UserRole)
        return audio_slice if isinstance(audio_slice, AudioSlice) else None

    def _audio_slice_item_at(self, position) -> QGraphicsItem | None:
        for item in self.items(position):
            audio_slice = item.data(Qt.UserRole)
            if isinstance(audio_slice, AudioSlice):
                return item
        return None

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.RightButton:
            mode = self._effective_tool_mode(event.modifiers())
            before_slices = self.slices()
            if mode in {"split", "fit_merge"} and self.cancel_split_at_position(event.pos()):
                self.slices_edited.emit(
                    {"before": before_slices, "after": self.slices()}
                )
                event.accept()
                return
        if event.button() == Qt.LeftButton:
            self.setFocus(Qt.MouseFocusReason)
            clicked_item = self._audio_slice_item_at(event.pos())
            mode = self._effective_tool_mode(event.modifiers())
            if mode == "split":
                before_slices = self.slices()
                if clicked_item is None:
                    scene_position = self.mapToScene(event.pos())
                    seconds = self._time_for_x(scene_position.x())
                    self.set_playhead_time(seconds)
                    self.playhead_seek_requested.emit(seconds)
                    event.accept()
                    return
                self.scene().clearSelection()
                clicked_item.setSelected(True)
                scene_position = self.mapToScene(event.pos())
                seconds = self._time_for_x(scene_position.x())
                self.set_playhead_time(seconds)
                if self.split_at_playhead():
                    self.slices_edited.emit(
                        {"before": before_slices, "after": self.slices()}
                    )
                event.accept()
                return
            if mode == "fit_merge" and clicked_item is not None:
                before_slices = self.slices()
                selected = self.scene().selectedItems()
                if not clicked_item.isSelected():
                    if len(selected) >= 2:
                        self.scene().clearSelection()
                    clicked_item.setSelected(True)
                if len(self.selected_audio_slices()) == 2:
                    if self.fit_or_merge_selected() is not None:
                        self.slices_edited.emit(
                            {"before": before_slices, "after": self.slices()}
                        )
                event.accept()
                return
            self._drag_start_position = event.pos()
            self._drag_slices_at_press = []
            self._drag_started_on_slice = clicked_item is not None
            if clicked_item is None:
                self._rubber_band_origin = event.pos()
                self._rubber_band.setGeometry(QRect(event.pos(), event.pos()).normalized())
                self._rubber_band.show()
                event.accept()
                return
            else:
                selected_slices = self.selected_audio_slices()
                if (
                    clicked_item.isSelected()
                    and len(selected_slices) > 1
                    and not (event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier))
                ):
                    self._drag_slices_at_press = selected_slices
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        self._refresh_tool_cursor()
        if not (event.buttons() & Qt.LeftButton):
            super().mouseMoveEvent(event)
            return
        if self._drag_start_position is None:
            super().mouseMoveEvent(event)
            return
        if (event.pos() - self._drag_start_position).manhattanLength() < QApplication.startDragDistance():
            super().mouseMoveEvent(event)
            return

        if not self._drag_started_on_slice:
            if self._rubber_band_origin is not None:
                self._rubber_band.setGeometry(
                    QRect(self._rubber_band_origin, event.pos()).normalized()
                )
                event.accept()
                return
            super().mouseMoveEvent(event)
            return

        selected_slices = self._drag_slices_at_press or self.selected_audio_slices()
        if not selected_slices:
            super().mouseMoveEvent(event)
            return
        selected_slices = self._refine_manual_slice_pitches(selected_slices)

        mime_data = QMimeData()
        mime_data.setData(
            MIME_AUDIO_SLICES,
            QByteArray(encode_audio_slices(selected_slices)),
        )

        drag = QDrag(self)
        drag.setMimeData(mime_data)
        if hasattr(drag, "exec"):
            drag.exec(Qt.CopyAction)
        else:
            drag.exec_(Qt.CopyAction)
        self._drag_slices_at_press = []

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if (
            event.button() == Qt.LeftButton
            and not self._drag_started_on_slice
            and self._rubber_band_origin is not None
        ):
            band_geometry = self._rubber_band.geometry()
            self._rubber_band.hide()
            if (event.pos() - self._rubber_band_origin).manhattanLength() < QApplication.startDragDistance():
                if not (event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)):
                    self.scene().clearSelection()
                scene_position = self.mapToScene(event.pos())
                seconds = self._time_for_x(scene_position.x())
                self.set_playhead_time(seconds)
                self.playhead_seek_requested.emit(seconds)
            else:
                self._select_slices_in_view_rect(band_geometry, event.modifiers())
            self._drag_start_position = None
            self._drag_slices_at_press = []
            self._drag_started_on_slice = False
            self._rubber_band_origin = None
            event.accept()
            return

        self._drag_start_position = None
        self._drag_slices_at_press = []
        self._drag_started_on_slice = False
        self._rubber_band_origin = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key_Space:
            self.playback_toggled.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def _select_slices_in_view_rect(self, view_rect: QRect, modifiers) -> None:
        scene_rect = self.mapToScene(view_rect).boundingRect()
        if not (modifiers & (Qt.ControlModifier | Qt.ShiftModifier)):
            self.scene().clearSelection()

        for item in self.scene().items(scene_rect):
            audio_slice = item.data(Qt.UserRole)
            if not isinstance(audio_slice, AudioSlice):
                continue
            if item.sceneBoundingRect().intersects(scene_rect):
                item.setSelected(True)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        audio_slice = self._audio_slice_at(event.pos())
        if audio_slice is None:
            if event.button() == Qt.LeftButton:
                self.playback_toggled.emit()
                event.accept()
                return
            super().mouseDoubleClickEvent(event)
            return
        self.slice_preview_requested.emit(audio_slice)
        event.accept()
        return


class MaterialTreeView(QTreeView):
    """File tree that can drag whole media files into tracks or the workspace."""

    folders_dropped = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._drag_start_position = None
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setCursor(_workspace_cursor("app_arrow"))

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if _folder_paths_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if _folder_paths_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        folder_paths = _folder_paths_from_mime_data(event.mimeData())
        if folder_paths:
            self.folders_dropped.emit(folder_paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self._drag_start_position = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if not (event.buttons() & Qt.LeftButton):
            super().mouseMoveEvent(event)
            return
        if self._drag_start_position is None:
            super().mouseMoveEvent(event)
            return
        if (event.pos() - self._drag_start_position).manhattanLength() < QApplication.startDragDistance():
            super().mouseMoveEvent(event)
            return

        index = self.indexAt(self._drag_start_position)
        model = self.model()
        if (
            index is None
            or not index.isValid()
            or not hasattr(model, "is_supported_file")
            or not model.is_supported_file(index)
        ):
            super().mouseMoveEvent(event)
            return

        mime_data = QMimeData()
        mime_data.setData(
            MIME_AUDIO_FILE,
            QByteArray(encode_audio_file(model.file_path(index))),
        )

        drag = QDrag(self)
        drag.setMimeData(mime_data)
        if hasattr(drag, "exec"):
            drag.exec(Qt.CopyAction)
        else:
            drag.exec_(Qt.CopyAction)


def _folder_paths_from_mime_data(mime_data) -> list[str]:  # noqa: ANN001
    if mime_data is None or not mime_data.hasUrls():
        return []
    folder_paths: list[str] = []
    for url in mime_data.urls():
        if not url.isLocalFile():
            continue
        path = Path(url.toLocalFile())
        try:
            if path.is_dir():
                folder_paths.append(str(path))
        except OSError:
            continue
    return folder_paths


class MaterialBrowserWidget(QWidget):
    """Multi-folder material browser for supported audio and video assets."""

    audio_file_selected = Signal(str)
    audio_file_parse_requested = Signal(str)
    audio_file_preview_requested = Signal(str, float)
    slice_sequence_preview_requested = Signal(object)
    folder_added = Signal(str)
    folder_expanded = Signal(str)
    auto_slice_toggled = Signal(bool)
    slice_preview_requested = Signal(object)
    material_slices_changed = Signal(str, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.file_model = MaterialFileSystemModel(self)
        self._language = "zh"
        self._current_file_path = ""
        self._parser_cut_mode = "split"
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top_row = QWidget()
        top_layout = QVBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        self.add_folder_button = QPushButton("+ 媒体文件夹")
        self.add_folder_button.setMinimumWidth(118)
        self.add_folder_button.setMinimumHeight(28)
        self.add_folder_button.setCursor(_workspace_cursor("app_hand"))
        top_layout.addWidget(self.add_folder_button)

        self.auto_slice_checkbox = QCheckBox("自动分段")
        self.auto_slice_checkbox.setChecked(True)
        self.auto_slice_checkbox.setMinimumHeight(24)
        self.auto_slice_checkbox.setToolTip("开启后会后台分析媒体并生成音符片段；关闭后双击音频只生成整段媒体片段")
        self.auto_slice_checkbox.setCursor(_workspace_cursor("app_hand"))
        top_layout.addWidget(self.auto_slice_checkbox)
        layout.addWidget(top_row)

        self.browser_splitter = QSplitter(Qt.Vertical)
        self.browser_splitter.setChildrenCollapsible(False)
        layout.addWidget(self.browser_splitter, 1)

        self.tree_view = MaterialTreeView()
        self.tree_view.setModel(self.file_model)
        self.tree_view.setRootIsDecorated(True)
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setAnimated(True)
        self.tree_view.setHeaderHidden(True)
        self.browser_splitter.addWidget(self.tree_view)

        analysis_widget = QWidget()
        self.analysis_widget = analysis_widget
        self._analysis_min_height = 180
        self._analysis_max_height = 720
        self._analysis_height = 320
        analysis_widget.setMinimumHeight(self._analysis_min_height)
        analysis_widget.setMaximumHeight(self._analysis_max_height)
        analysis_layout = QVBoxLayout(analysis_widget)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        analysis_layout.setSpacing(3)

        header = QWidget()
        header.setMinimumHeight(38)
        header.setMaximumHeight(42)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(5)

        self.slice_title_label = QLabel("源片段编辑区")
        self.slice_title_label.setMinimumHeight(28)
        self.slice_title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.slice_title_label.setVisible(True)
        header_layout.addWidget(self.slice_title_label, 1)
        analysis_layout.addWidget(header)

        self.parser_tool_container = QWidget()
        parser_tool_layout = QVBoxLayout(self.parser_tool_container)
        parser_tool_layout.setContentsMargins(0, 0, 0, 0)
        parser_tool_layout.setSpacing(1)

        self.manual_split_button = QToolButton()
        self.manual_split_button.setObjectName("MaterialScissorsButton")
        self.manual_split_button.setIcon(self._make_scissors_icon())
        self.manual_split_button.setIconSize(QSize(25, 25))
        self.manual_split_button.setFixedSize(38, 30)
        self.manual_split_button.setCheckable(True)
        self.manual_split_button.setEnabled(False)
        self.manual_split_button.setCursor(_workspace_cursor("app_hand"))
        parser_tool_layout.addWidget(self.manual_split_button)

        self.manual_fit_merge_button = QToolButton()
        self.manual_fit_merge_button.setObjectName("ScissorsMenuButton")
        self.manual_fit_merge_button.setText("▼")
        self.manual_fit_merge_button.setFixedSize(38, 15)
        self.manual_fit_merge_button.setEnabled(False)
        self.manual_fit_merge_button.setCursor(_workspace_cursor("app_hand"))
        parser_tool_layout.addWidget(self.manual_fit_merge_button)
        header_layout.addWidget(self.parser_tool_container, 0, Qt.AlignRight | Qt.AlignVCenter)

        self.slice_list = SlicePitchMapWidget()
        self.slice_list.setMinimumHeight(130)
        self.slice_list.setMaximumHeight(720)
        analysis_layout.addWidget(self.slice_list, 1)

        self.preview_player = MaterialPreviewPlayerWidget()
        analysis_layout.addWidget(self.preview_player, 0)

        self.info_label = QLabel("")
        self.info_label.setMinimumHeight(24)
        self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.info_label.setWordWrap(False)
        self.info_label.setVisible(False)
        analysis_layout.addWidget(self.info_label, 0)

        self.browser_splitter.addWidget(analysis_widget)
        self.browser_splitter.setStretchFactor(0, 5)
        self.browser_splitter.setStretchFactor(1, 3)
        self.browser_splitter.setSizes([520, self._analysis_height])

        self.parser_cut_menu = QMenu(self)
        self.parser_split_action = self.parser_cut_menu.addAction(
            self._make_scissors_icon(),
            "分割",
        )
        self.parser_fit_merge_action = self.parser_cut_menu.addAction(
            self._make_cut_merge_icon(),
            "片段贴合",
        )

        self.add_folder_button.clicked.connect(self._choose_folder)
        self.auto_slice_checkbox.toggled.connect(self._on_auto_slice_toggled)
        self.manual_split_button.toggled.connect(self._on_parser_tool_toggled)
        self.manual_split_button.setContextMenuPolicy(Qt.CustomContextMenu)
        self.manual_split_button.customContextMenuRequested.connect(self._toggle_parser_cut_mode)
        self.manual_fit_merge_button.clicked.connect(self._toggle_parser_cut_mode)
        self.parser_split_action.triggered.connect(lambda: self._set_parser_cut_mode("split"))
        self.parser_fit_merge_action.triggered.connect(lambda: self._set_parser_cut_mode("fit_merge"))
        self.tree_view.clicked.connect(self._on_tree_clicked)
        self.tree_view.doubleClicked.connect(self._on_tree_double_clicked)
        self.tree_view.expanded.connect(self._on_tree_expanded)
        self.tree_view.folders_dropped.connect(self._add_dropped_folders)
        self.slice_list.slice_preview_requested.connect(self.slice_preview_requested)
        self.slice_list.playback_toggled.connect(self.preview_player.toggle_playback)
        self.slice_list.playhead_seek_requested.connect(self.set_material_preview_position)
        self.slice_list.slices_edited.connect(self._on_parser_slices_edited)
        self.preview_player.play_requested.connect(self._on_preview_play_requested)
        self.preview_player.stop_requested.connect(lambda: self.audio_file_preview_requested.emit("", 0.0))
        self.preview_player.position_previewed.connect(self.set_material_preview_position)
        self._refresh_manual_split_button()
        self._analysis_resize_targets = (
            self.analysis_widget,
            self.slice_list,
            self.slice_list.viewport(),
            self.preview_player,
        )
        for target in self._analysis_resize_targets:
            target.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if (
            hasattr(self, "_analysis_resize_targets")
            and any(watched is target for target in self._analysis_resize_targets)
            and event.type() == QEvent.Wheel
        ):
            delta = event.angleDelta().y()
            if delta:
                self._resize_analysis_panel(80 if delta > 0 else -80)
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _resize_analysis_panel(self, delta_height: int) -> None:
        current_height = self.analysis_widget.height() or self._analysis_height
        self._analysis_height = max(
            self._analysis_min_height,
            min(self._analysis_max_height, current_height + int(delta_height)),
        )
        sizes = self.browser_splitter.sizes()
        total = sum(sizes) if sizes else self.height()
        tree_height = max(120, total - self._analysis_height)
        self.browser_splitter.setSizes([tree_height, self._analysis_height])

    def set_language(self, language: str) -> None:
        self._language = "zh" if language == "zh" else "en"
        self.file_model.set_language(self._language)
        if self._language == "zh":
            self.add_folder_button.setText("+ 媒体文件夹")
            self.add_folder_button.setToolTip("添加媒体文件夹")
            self.auto_slice_checkbox.setText("自动分段")
            self.auto_slice_checkbox.setToolTip("开启后会后台分析媒体并生成音符片段；关闭后双击音频只生成整段媒体片段")
            self.parser_split_action.setText("分割")
            self.parser_fit_merge_action.setText("片段贴合")
            self.slice_title_label.setText("源片段编辑区")
            self.info_label.setText("")
        else:
            self.add_folder_button.setText("+ Folder")
            self.add_folder_button.setToolTip("Add Folder")
            self.auto_slice_checkbox.setText("Auto Slice")
            self.auto_slice_checkbox.setToolTip("When enabled, imported folders are pre-parsed in the background")
            self.parser_split_action.setText("Split")
            self.parser_fit_merge_action.setText("Cut Fit")
            self.slice_title_label.setText("Parser")
            self.info_label.setText("")
        self.preview_player.set_language(self._language)
        self._refresh_parser_cut_tool()

    def add_folder(self, folder_path: str) -> bool:
        added = self.file_model.add_folder(folder_path)
        if added:
            message = "已添加" if self._language == "zh" else "Added"
            self.info_label.setText(message)
            self.info_label.setToolTip(folder_path)
            self.folder_added.emit(folder_path)
        else:
            message = (
                "不可用"
                if self._language == "zh"
                else "Unavailable"
            )
            self.info_label.setText(f"{message}: {folder_path}")
        return added

    def clear_folders(self) -> None:
        self.file_model.clear_folders()
        self._current_file_path = ""
        self.slice_list.set_slices([])
        self.preview_player.set_media("", duration=None)
        self._refresh_manual_split_button()

    def folder_paths(self) -> list[str]:
        return self.file_model.folder_paths()

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if _folder_paths_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if _folder_paths_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        folder_paths = _folder_paths_from_mime_data(event.mimeData())
        if folder_paths:
            self._add_dropped_folders(folder_paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def _add_dropped_folders(self, folder_paths: object) -> None:
        paths = [
            str(path)
            for path in (folder_paths if isinstance(folder_paths, list) else [])
            if path
        ]
        added_count = 0
        for folder_path in paths:
            if self.add_folder(folder_path):
                added_count += 1
        if added_count > 1:
            message = (
                f"已添加 {added_count} 个文件夹"
                if self._language == "zh"
                else f"Added {added_count} folders"
            )
            self.info_label.setText(message)
            self.info_label.setToolTip("\n".join(paths))

    def set_probe_result(self, text: str) -> None:
        self.info_label.setToolTip(text)

    def set_parse_status(self, text: str) -> None:
        self.info_label.setToolTip(text)
        self.slice_title_label.setText(self._compact_parse_title(text))

    def set_background_parse_status(self, text: str) -> None:
        """Keep pre-parse progress out of the visible parser title."""
        self.info_label.setToolTip(text)

    def set_slices(self, path: str, slices: list[AudioSlice]) -> None:
        self.slice_list.set_slices(slices)
        unit = "个" if self._language == "zh" else "items"
        self.slice_title_label.setText(self._idle_parse_title())
        self.info_label.setText(f"{len(slices)} {unit}")
        self.info_label.setToolTip(path)
        max_end = max((audio_slice.end_time for audio_slice in slices), default=0.0)
        if path and path != self._current_file_path:
            self.load_material_into_preview(path, duration=max_end if max_end > 0 else None)
        elif path and not self.preview_player.play_button.isEnabled():
            self.load_material_into_preview(path, duration=max_end if max_end > 0 else None)
        elif path == self._current_file_path and max_end > 0:
            self.preview_player.update_duration(max_end)
        self._refresh_manual_split_button()

    def load_material_into_preview(self, path: str, duration: float | None = None) -> None:
        self._current_file_path = path
        self.preview_player.set_media(path, duration=duration)

    def set_material_duration(self, path: str, duration: float) -> None:
        if path == self._current_file_path:
            self.preview_player.update_duration(duration)

    def set_material_preview_position(self, position: float) -> None:
        self.preview_player.set_position(position)
        self.slice_list.set_playhead_time(position)

    def set_material_preview_playing(self, playing: bool) -> None:
        self.preview_player.set_playing(playing)

    def current_file_path(self) -> str:
        return self._current_file_path

    def auto_slicing_enabled(self) -> bool:
        return self.auto_slice_checkbox.isChecked()

    def _on_auto_slice_toggled(self, enabled: bool) -> None:
        self._refresh_manual_split_button()
        self.auto_slice_toggled.emit(bool(enabled))

    def _refresh_manual_split_button(self) -> None:
        slice_count = len(self.slice_list.slices())
        manual_mode = not self.auto_slice_checkbox.isChecked()
        has_active_material = bool(self._current_file_path) and slice_count > 0
        show_tools = manual_mode and has_active_material
        self.parser_tool_container.setVisible(show_tools)
        self.manual_split_button.setVisible(show_tools)
        self.manual_split_button.setEnabled(show_tools)
        self.manual_fit_merge_button.setVisible(show_tools)
        self.manual_fit_merge_button.setEnabled(show_tools)
        self.parser_fit_merge_action.setEnabled(slice_count >= 2)
        if not show_tools:
            self.manual_split_button.setChecked(False)
            self.slice_list.set_tool_mode("select")
        if slice_count < 2 and self._parser_cut_mode == "fit_merge":
            self._parser_cut_mode = "split"
        if slice_count <= 0 and self.manual_split_button.isChecked():
            self.manual_split_button.setChecked(False)
        self._refresh_parser_cut_tool()

    def _on_parser_tool_toggled(self, checked: bool) -> None:
        self.slice_list.set_tool_mode(self._parser_cut_mode if checked else "select")
        self._refresh_parser_cut_tool()

    def _on_parser_slices_edited(self, slices: object) -> None:
        before_slices: list[AudioSlice] = []
        if isinstance(slices, dict):
            before_slices = [
                audio_slice
                for audio_slice in list(slices.get("before", []))
                if isinstance(audio_slice, AudioSlice)
            ]
            slices = slices.get("after", [])
        audio_slices = [
            audio_slice for audio_slice in list(slices) if isinstance(audio_slice, AudioSlice)
        ]
        self.material_slices_changed.emit(
            self._current_file_path,
            {"before": before_slices, "after": audio_slices},
        )
        unit = "个" if self._language == "zh" else "items"
        self.slice_title_label.setText(self._idle_parse_title())
        self.info_label.setText(f"{len(audio_slices)} {unit}")
        self._refresh_manual_split_button()

    def _show_parser_cut_menu(self) -> None:
        self.parser_cut_menu.exec(
            self.manual_fit_merge_button.mapToGlobal(
                self.manual_fit_merge_button.rect().bottomLeft()
            )
        )

    def _toggle_parser_cut_mode(self, _pos=None) -> None:
        if self._parser_cut_mode == "split" and len(self.slice_list.slices()) >= 2:
            self._set_parser_cut_mode("fit_merge")
        else:
            self._set_parser_cut_mode("split")

    def _set_parser_cut_mode(self, mode: str) -> None:
        self._parser_cut_mode = "fit_merge" if mode == "fit_merge" else "split"
        self.manual_split_button.setChecked(True)
        self.slice_list.set_tool_mode(self._parser_cut_mode)
        self._refresh_parser_cut_tool()

    def _refresh_parser_cut_tool(self) -> None:
        is_fit = self._parser_cut_mode == "fit_merge"
        self.manual_split_button.setIcon(
            self._make_cut_merge_icon() if is_fit else self._make_scissors_icon()
        )
        if self._language == "zh":
            tooltip = (
                "片段贴合：选中两个音符片段，先贴合，再次使用可合并"
                if is_fit
                else "分割源媒体片段（先把播放头放到分割位置）"
            )
            menu_tip = "切换源片段分割模式"
        else:
            tooltip = (
                "Cut Fit: select two slices to fit, then use again to merge"
                if is_fit
                else "Split material in the parser at the playhead"
            )
            menu_tip = "Switch parser cut tool"
        self.manual_split_button.setToolTip(tooltip)
        self.manual_fit_merge_button.setToolTip(menu_tip)
        if self.manual_split_button.isChecked():
            self.manual_split_button.setStyleSheet("")

    def _split_material_at_playhead(self) -> None:
        before_slices = self.slice_list.slices()
        if not self.slice_list.split_at_playhead():
            self.info_label.setText("未分割" if self._language == "zh" else "No split")
            return
        slices = self.slice_list.slices()
        self.material_slices_changed.emit(
            self._current_file_path,
            {"before": before_slices, "after": slices},
        )
        unit = "个" if self._language == "zh" else "items"
        self.slice_title_label.setText(self._idle_parse_title())
        self.info_label.setText(f"{len(slices)} {unit}")
        self._refresh_manual_split_button()

    def _fit_or_merge_material_slices(self) -> None:
        before_slices = self.slice_list.slices()
        result = self.slice_list.fit_or_merge_selected()
        if result is None:
            self.info_label.setText(
                "请选择同一源媒体的两个音符片段"
                if self._language == "zh"
                else "Select two slices from the same material"
            )
            return
        slices = self.slice_list.slices()
        self.material_slices_changed.emit(
            self._current_file_path,
            {"before": before_slices, "after": slices},
        )
        if self._language == "zh":
            self.info_label.setText("已合并" if result == "merged" else "已贴合，再次使用可合并")
        else:
            self.info_label.setText("Merged" if result == "merged" else "Fitted; use again to merge")
        self.slice_title_label.setText(self._idle_parse_title())
        self._refresh_manual_split_button()

    def _make_scissors_icon(self) -> QIcon:
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(QColor("#d4d4d4"), 2))
        painter.drawEllipse(4, 15, 5, 5)
        painter.drawEllipse(15, 15, 5, 5)
        painter.drawLine(8, 16, 19, 5)
        painter.drawLine(16, 16, 5, 5)
        painter.end()
        return QIcon(pixmap)

    def _make_cut_merge_icon(self) -> QIcon:
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        light = QColor("#f4f8ff")
        painter.fillRect(11, 2, 3, 20, QColor("#ff9f2f"))
        painter.fillRect(2, 10, 6, 5, light)
        painter.fillRect(16, 10, 6, 5, light)
        painter.setPen(QPen(light, 2))
        painter.drawLine(8, 7, 12, 12)
        painter.drawLine(8, 17, 12, 12)
        painter.drawLine(16, 7, 13, 12)
        painter.drawLine(16, 17, 13, 12)
        painter.end()
        return QIcon(pixmap)

    def _choose_folder(self) -> None:
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "添加文件夹目录" if self._language == "zh" else "Add Folder",
            "",
        )
        if folder_path:
            self.add_folder(folder_path)

    def _on_tree_clicked(self, index) -> None:
        if not self.file_model.is_supported_file(index):
            return
        self._current_file_path = self.file_model.file_path(index)
        self.load_material_into_preview(self._current_file_path)
        self.audio_file_selected.emit(self._current_file_path)

    def _on_tree_double_clicked(self, index) -> None:
        if not self.file_model.is_supported_file(index):
            return
        self._current_file_path = self.file_model.file_path(index)
        self.load_material_into_preview(self._current_file_path)
        self.audio_file_selected.emit(self._current_file_path)
        self.audio_file_parse_requested.emit(self._current_file_path)

    def _on_tree_expanded(self, index) -> None:
        folder_path = self.file_model.file_path(index)
        if folder_path and Path(folder_path).is_dir():
            self.folder_expanded.emit(folder_path)

    def _on_preview_play_requested(self, path: str, start_time: float) -> None:
        selected_slices = self.slice_list.selected_audio_slices()
        if selected_slices:
            self.slice_sequence_preview_requested.emit(selected_slices)
            return
        self.audio_file_preview_requested.emit(path, start_time)

    def _compact_status_text(self, text: str) -> str:
        if not text:
            return ""
        lowered = text.lower()
        if "fail" in lowered or "失败" in text:
            return "!"
        duration_match = re.search(r"duration=([0-9.]+)s", text)
        if duration_match:
            return f"✓ {float(duration_match.group(1)):.2f}s"
        if "probe" in lowered or "探测" in text or "解析" in text or "parsing" in lowered:
            return "…"
        return "✓"

    def _compact_parse_title(self, text: str) -> str:
        lowered = text.lower()
        if "fail" in lowered or "失败" in text:
            return "分析失败" if self._language == "zh" else "Parse failed"
        if "解析" in text or "parsing" in lowered:
            return "分析中" if self._language == "zh" else "Parsing"
        return self._idle_parse_title()

    def _idle_parse_title(self) -> str:
        return "源片段编辑区" if self._language == "zh" else "Source Clip Editor"
