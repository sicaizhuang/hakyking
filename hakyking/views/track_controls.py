from __future__ import annotations

from pathlib import Path
from typing import Iterable

from hakyking.dnd import MIME_AUDIO_FILE, decode_audio_file
from hakyking.models.material_file_system import SUPPORTED_MEDIA_EXTENSIONS
from hakyking.models.project import TrackModel
from hakyking.qt import (
    QColor,
    QFrame,
    QFontMetrics,
    QHBoxLayout,
    QLabel,
    QPainter,
    QPen,
    QPushButton,
    QRectF,
    QScrollArea,
    Signal,
    Qt,
    QVBoxLayout,
    QWidget,
)


def _audio_path_from_mime_data(mime_data) -> str:  # noqa: ANN001 - Qt binding type varies.
    if mime_data.hasFormat(MIME_AUDIO_FILE):
        try:
            path = decode_audio_file(bytes(mime_data.data(MIME_AUDIO_FILE)))
        except Exception:
            path = ""
        if path:
            return path

    if mime_data.hasUrls():
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS:
                return str(path)
    return ""


class TrackClipStrip(QWidget):
    """Small timeline strip inside a track row for whole-file track clips."""

    clip_moved = Signal(int, float)
    file_dropped = Signal(int, str)
    PIXELS_PER_SECOND = 260.0

    def __init__(self, index: int, track: TrackModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.index = index
        self.track = track
        self._drag_origin_x: float | None = None
        self._drag_origin_global_x: float | None = None
        self._drag_start_time = 0.0
        self.setAcceptDrops(True)
        self.setMinimumHeight(26)
        self.setCursor(Qt.OpenHandCursor)

    def set_track(self, index: int, track: TrackModel) -> None:
        self.index = index
        self.track = track
        self.update()

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if _audio_path_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if _audio_path_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        path = _audio_path_from_mime_data(event.mimeData())
        if not path:
            super().dropEvent(event)
            return
        self.file_dropped.emit(self.index, path)
        event.acceptProposedAction()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton and self.track.clip_path:
            self._drag_origin_x = float(event.x())
            self._drag_origin_global_x = self._event_global_x(event)
            self._drag_start_time = max(0.0, float(self.track.clip_start))
            self.setCursor(Qt.ClosedHandCursor)
            self.grabMouse()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._drag_origin_global_x is None:
            super().mouseMoveEvent(event)
            return
        delta_seconds = (
            self._event_global_x(event) - self._drag_origin_global_x
        ) / self.PIXELS_PER_SECOND
        self.clip_moved.emit(self.index, max(0.0, self._drag_start_time + delta_seconds))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton and self._drag_origin_global_x is not None:
            self._drag_origin_x = None
            self._drag_origin_global_x = None
            self.setCursor(Qt.OpenHandCursor)
            self.releaseMouse()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _event_global_x(self, event) -> float:  # noqa: ANN001 - Qt binding type varies.
        if hasattr(event, "globalPosition"):
            return float(event.globalPosition().x())
        if hasattr(event, "globalPos"):
            return float(event.globalPos().x())
        return float(event.x())

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect().adjusted(0, 1, 0, -1)
        painter.setPen(QPen(QColor("#242428"), 1))
        painter.setBrush(QColor("#232327"))
        painter.drawRoundedRect(rect, 4, 4)

        if not self.track.clip_path:
            painter.setPen(QColor("#8a8f98"))
            painter.drawText(rect.adjusted(8, 0, -8, 0), Qt.AlignVCenter | Qt.AlignLeft, "拖入音频")
            painter.end()
            return

        start_x = max(0.0, float(self.track.clip_start) * self.PIXELS_PER_SECOND)
        width = max(32.0, float(self.track.clip_duration) * self.PIXELS_PER_SECOND)
        clip_rect = QRectF(start_x, rect.top() + 2, width, rect.height() - 4)
        visible_clip = clip_rect.intersected(QRectF(rect))
        fill = QColor("#4b5868") if not self.track.clip_editable else QColor("#3f8ec5")
        painter.setPen(QPen(QColor("#9fb3c8"), 1))
        painter.setBrush(fill)
        painter.drawRoundedRect(visible_clip, 4, 4)
        if visible_clip.width() >= 42:
            painter.save()
            painter.setClipRect(visible_clip.adjusted(2, 1, -2, -1))
            font = painter.font()
            if font.pointSize() > 0:
                font.setPointSize(max(8, font.pointSize() - 1))
            painter.setFont(font)
            painter.setPen(QColor("#eef3f8"))
            name = Path(self.track.clip_path).name or self.track.clip_path
            text_rect = visible_clip.adjusted(8, 0, -8, 0)
            name = QFontMetrics(painter.font()).elidedText(
                name,
                Qt.ElideMiddle,
                max(16, int(text_rect.width())),
            )
            painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, name)
            painter.restore()
        painter.end()


class TrackRowWidget(QFrame):
    selected = Signal(int)
    solo_changed = Signal(int, bool)
    lock_changed = Signal(int, bool)
    mute_changed = Signal(int, bool)
    clip_editable_changed = Signal(int, bool)
    clip_moved = Signal(int, float)
    file_dropped = Signal(int, str)
    clip_delete_requested = Signal(int)

    def __init__(
        self,
        index: int,
        track: TrackModel,
        selected: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("TrackRow")
        self.setProperty("selected", selected)
        self.index = index
        self.track = track
        self.setFrameShape(QFrame.NoFrame)
        self.setFixedHeight(92)
        self.setAcceptDrops(True)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 6, 8, 6)
        root_layout.setSpacing(6)

        header = QWidget(self)
        header.setFixedHeight(30)
        top_layout = QHBoxLayout(header)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)
        root_layout.addWidget(header)

        self.color_bar = QFrame()
        self.color_bar.setObjectName("TrackColorBar")
        self.color_bar.setFixedWidth(5)
        top_layout.addWidget(self.color_bar)

        self.name_label = QLabel(track.name)
        self.name_label.setObjectName("TrackName")
        self.name_label.setMinimumWidth(58)
        self.name_label.setMaximumWidth(96)
        self.name_label.setToolTip(track.name)
        top_layout.addWidget(self.name_label)

        self.status_label = QLabel(self._status_text())
        self.status_label.setObjectName("TrackStatus")
        self.status_label.setFixedWidth(18)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setVisible(bool(track.clip_path))
        self.status_label.setToolTip(self._status_tooltip())
        top_layout.addWidget(self.status_label)

        self.edit_button = QPushButton("编辑")
        self.edit_button.setObjectName("TrackEditButton")
        self.edit_button.setCheckable(True)
        self.edit_button.setFixedSize(44, 26)
        self.edit_button.setChecked(track.clip_editable)
        self.edit_button.setEnabled(bool(track.clip_path))
        self.edit_button.setToolTip("启用后，音高编辑区里的参考片段会转换为可编辑音符片段")
        top_layout.addWidget(self.edit_button)

        self.clear_button = QPushButton("×")
        self.clear_button.setObjectName("TrackClearButton")
        self.clear_button.setFixedSize(26, 26)
        self.clear_button.setVisible(bool(track.clip_path))
        self.clear_button.setEnabled(bool(track.clip_path))
        self.clear_button.setToolTip("移除这条音轨上的音频")
        top_layout.addWidget(self.clear_button)

        self.solo_button = QPushButton("S")
        self.solo_button.setObjectName("SoloButton")
        self.solo_button.setCheckable(True)
        self.solo_button.setFixedSize(30, 26)
        self.solo_button.setChecked(track.solo)
        self.solo_button.setToolTip("Solo")
        top_layout.addWidget(self.solo_button)

        self.mute_button = QPushButton("M")
        self.mute_button.setObjectName("MuteButton")
        self.mute_button.setCheckable(True)
        self.mute_button.setFixedSize(30, 26)
        self.mute_button.setChecked(track.muted)
        self.mute_button.setToolTip("Mute")
        top_layout.addWidget(self.mute_button)

        self.lock_button = QPushButton("L")
        self.lock_button.setObjectName("LockButton")
        self.lock_button.setCheckable(True)
        self.lock_button.setFixedSize(30, 26)
        self.lock_button.setChecked(track.locked)
        self.lock_button.setToolTip("Lock")
        top_layout.addWidget(self.lock_button)
        top_layout.addStretch(1)

        self.clip_strip = TrackClipStrip(index, track)
        self.clip_strip.setFixedHeight(38)
        root_layout.addWidget(self.clip_strip)

        self.solo_button.toggled.connect(self._on_solo_toggled)
        self.mute_button.toggled.connect(self._on_mute_toggled)
        self.lock_button.toggled.connect(self._on_lock_toggled)
        self.edit_button.toggled.connect(self._on_edit_toggled)
        self.clear_button.clicked.connect(lambda: self.clip_delete_requested.emit(self.index))
        self.clip_strip.clip_moved.connect(self.clip_moved)
        self.clip_strip.file_dropped.connect(self.file_dropped)

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if _audio_path_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if _audio_path_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        path = _audio_path_from_mime_data(event.mimeData())
        if not path:
            super().dropEvent(event)
            return
        self.file_dropped.emit(self.index, path)
        event.acceptProposedAction()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self.selected.emit(self.index)
        super().mousePressEvent(event)

    def _on_solo_toggled(self, checked: bool) -> None:
        self.track.solo = checked
        self.solo_changed.emit(self.index, checked)

    def _on_mute_toggled(self, checked: bool) -> None:
        self.track.muted = checked
        self.mute_changed.emit(self.index, checked)

    def _on_lock_toggled(self, checked: bool) -> None:
        self.track.locked = checked
        self.lock_changed.emit(self.index, checked)

    def _on_edit_toggled(self, checked: bool) -> None:
        self.track.clip_editable = checked
        self.clip_editable_changed.emit(self.index, checked)

    def _status_text(self) -> str:
        if not self.track.clip_path:
            return "空"
        if self.track.clip_duration <= 0.0:
            return "分析"
        return "编" if self.track.clip_editable else "参"

    def _status_tooltip(self) -> str:
        if not self.track.clip_path:
            return "空音轨：把媒体库或资源管理器里的音频拖到这里"
        name = Path(self.track.clip_path).name
        if self.track.clip_duration <= 0.0:
            return f"{name}\n正在分析音高与片段边界"
        mode = "可编辑片段层" if self.track.clip_editable else "参考片段层"
        return f"{name}\n{mode}\n起点 {self.track.clip_start:.2f}s，时长 {self.track.clip_duration:.2f}s"


    def refresh_track_state(self, index: int, track: TrackModel) -> None:
        self.index = index
        self.track = track
        self.name_label.setText(track.name)
        self.name_label.setToolTip(track.name)
        self.status_label.setText(self._status_text())
        self.status_label.setVisible(bool(track.clip_path))
        self.status_label.setToolTip(self._status_tooltip())
        self.edit_button.setChecked(track.clip_editable)
        self.edit_button.setEnabled(bool(track.clip_path))
        self.clear_button.setVisible(bool(track.clip_path))
        self.clear_button.setEnabled(bool(track.clip_path))
        self.solo_button.setChecked(track.solo)
        self.mute_button.setChecked(track.muted)
        self.lock_button.setChecked(track.locked)
        self.clip_strip.set_track(index, track)


class TrackControlPanel(QScrollArea):
    track_selected = Signal(int)
    track_solo_changed = Signal(int, bool)
    track_lock_changed = Signal(int, bool)
    track_mute_changed = Signal(int, bool)
    track_add_requested = Signal()
    track_clip_editable_changed = Signal(int, bool)
    track_clip_moved = Signal(int, float)
    track_audio_file_dropped = Signal(int, str)
    track_clip_delete_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._selected_index = 0
        self._track_count = 0
        self._row_widgets: list[TrackRowWidget] = []
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.container = QWidget()
        self.container.setAcceptDrops(True)
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(4, 4, 4, 4)
        self.layout.setSpacing(3)

        self.add_track_button = QPushButton("+ 新建音轨")
        self.add_track_button.setObjectName("AddTrackButton")
        self.add_track_button.setToolTip("新增一条音轨")
        self.add_track_button.setMinimumHeight(30)
        self.add_track_button.setMinimumWidth(110)
        self.layout.addWidget(self.add_track_button)
        self.layout.addStretch(1)
        self.setWidget(self.container)
        self.add_track_button.clicked.connect(self.track_add_requested)

    def set_tracks(
        self,
        tracks: Iterable[TrackModel],
        selected_index: int = 0,
    ) -> None:
        track_list = list(tracks)
        self._track_count = len(track_list)
        self._selected_index = max(0, min(int(selected_index), max(0, self._track_count - 1)))
        for row in self._row_widgets:
            self.layout.removeWidget(row)
            row.deleteLater()
        self._row_widgets = []

        for index, track in enumerate(track_list):
            row = TrackRowWidget(
                index=index,
                track=track,
                selected=index == self._selected_index,
            )
            row.selected.connect(self.track_selected)
            row.solo_changed.connect(self.track_solo_changed)
            row.lock_changed.connect(self.track_lock_changed)
            row.mute_changed.connect(self.track_mute_changed)
            row.clip_editable_changed.connect(self.track_clip_editable_changed)
            row.clip_moved.connect(self.track_clip_moved)
            row.file_dropped.connect(self.track_audio_file_dropped)
            row.clip_delete_requested.connect(self.track_clip_delete_requested)
            self.layout.insertWidget(index + 1, row)
            self._row_widgets.append(row)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.track_clip_delete_requested.emit(self._selected_index)
            event.accept()
            return
        super().keyPressEvent(event)

    def refresh_track_state(self, index: int, track: TrackModel) -> None:
        if 0 <= index < len(self._row_widgets):
            self._row_widgets[index].refresh_track_state(index, track)

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if _audio_path_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if _audio_path_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        path = _audio_path_from_mime_data(event.mimeData())
        if not path or not self._drop_audio_path(path):
            super().dropEvent(event)
            return
        event.acceptProposedAction()

    def _drop_audio_path(self, path: str) -> bool:
        if not path or self._track_count <= 0:
            return False
        index = max(0, min(self._selected_index, self._track_count - 1))
        self.track_audio_file_dropped.emit(index, path)
        return True
