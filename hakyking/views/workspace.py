from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import numpy as np

from hakyking.audio.gain import (
    MAX_GAIN_DB,
    MIN_GAIN_DB,
    apply_gain,
    format_dbfs,
    format_gain,
    measure_audio_dbfs,
)
from hakyking.dnd import (
    MIME_AUDIO_FILE,
    MIME_AUDIO_SLICES,
    decode_audio_file,
    decode_audio_slices,
)
from hakyking.models.audio_slice import AudioSlice, copy_audio_slice
from hakyking.models.audio_edit import AudioSliceEditModel, SliceRenderRequest
from hakyking.models.scale import (
    is_midi_in_scale,
    nearest_midi_in_scale,
    normalize_root,
    normalize_scale_type,
)
from hakyking.qt import (
    QApplication,
    QColor,
    QBrush,
    QCursor,
    QFont,
    QFontMetrics,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QMenu,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPointF,
    QPen,
    QPixmap,
    QRect,
    QRectF,
    QRubberBand,
    Signal,
    Qt,
    QTimer,
)
from hakyking.views.workspace_boundary import BoundaryDragState


_CURSOR_CACHE: dict[str, QCursor] = {}
_PLACEMENT_GROUP_COUNTER = itertools.count()


def _new_placement_group_id(prefix: str) -> str:
    return f"{prefix}-{time.monotonic_ns()}-{next(_PLACEMENT_GROUP_COUNTER)}"


def _cursor_triangle(painter: QPainter, points: list[tuple[float, float]], color: QColor) -> None:
    path = QPainterPath()
    path.moveTo(points[0][0], points[0][1])
    for x, y in points[1:]:
        path.lineTo(x, y)
    path.closeSubpath()
    painter.fillPath(path, QBrush(color))


def _cursor_music_note(
    painter: QPainter,
    x: float,
    y: float,
    color: QColor,
    shadow: QColor,
    scale: float = 1.0,
) -> None:
    head_w = 6.2 * scale
    head_h = 4.8 * scale
    stem_h = 14.0 * scale
    stem_x = x + head_w - 0.8 * scale
    stem_top = y - stem_h + 2.0 * scale
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(shadow, max(3.0, 3.8 * scale)))
    painter.drawEllipse(QRectF(x, y, head_w, head_h))
    painter.drawLine(int(stem_x), int(y + 1.0 * scale), int(stem_x), int(stem_top))
    painter.drawLine(int(stem_x), int(stem_top), int(stem_x + 5.0 * scale), int(stem_top + 2.5 * scale))
    painter.setPen(QPen(color, max(1.4, 1.8 * scale)))
    painter.drawEllipse(QRectF(x, y, head_w, head_h))
    painter.drawLine(int(stem_x), int(y + 1.0 * scale), int(stem_x), int(stem_top))
    painter.drawLine(int(stem_x), int(stem_top), int(stem_x + 5.0 * scale), int(stem_top + 2.5 * scale))


def _cursor_staff(painter: QPainter, x1: int, x2: int, y: int, color: QColor, width: float = 1.2) -> None:
    painter.setPen(QPen(color, width))
    painter.drawLine(x1, y - 3, x2, y - 3)
    painter.drawLine(x1, y, x2, y)
    painter.drawLine(x1, y + 3, x2, y + 3)


def _cursor_wave(painter: QPainter, x: float, y: float, color: QColor, shadow: QColor) -> None:
    wave = QPainterPath()
    wave.moveTo(x, y)
    wave.cubicTo(x + 3, y - 5, x + 5, y + 5, x + 8, y)
    wave.cubicTo(x + 11, y - 5, x + 13, y + 5, x + 16, y)
    painter.setPen(QPen(shadow, 4))
    painter.drawPath(wave)
    painter.setPen(QPen(color, 1.8))
    painter.drawPath(wave)


def _workspace_cursor(name: str) -> QCursor:
    cached = _CURSOR_CACHE.get(name)
    if cached is not None:
        return cached

    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    light = QColor("#eef5ff")
    accent = QColor("#66c2df")
    merge_accent = QColor("#f4f8ff")
    merge_line = QColor("#ff9f2f")
    shadow = QColor(0, 0, 0, 135)
    painter.setPen(QPen(shadow, 4))

    hot_x = 16
    hot_y = 16
    if name == "app_arrow":
        hot_x = 5
        hot_y = 3
        pointer = QPainterPath()
        pointer.moveTo(5, 3)
        pointer.cubicTo(14, 6, 22, 12, 22, 19)
        pointer.cubicTo(22, 26, 15, 29, 9, 24)
        pointer.cubicTo(5, 20, 4, 11, 5, 3)
        pointer.closeSubpath()
        painter.setPen(QPen(shadow, 4))
        painter.drawPath(pointer)
        painter.fillPath(pointer, QBrush(QColor("#f5f7fb")))
        painter.setPen(QPen(accent, 1.7))
        painter.drawPath(pointer)
        _cursor_staff(painter, 16, 31, 23, QColor(20, 36, 48, 185), 1.1)
        _cursor_music_note(painter, 22, 25, QColor("#61c9e8"), shadow, 0.72)
    elif name == "app_hand":
        hot_x = 10
        hot_y = 9
        painter.setPen(QPen(shadow, 4))
        painter.setBrush(QBrush(QColor("#f5f7fb")))
        note = QPainterPath()
        note.addEllipse(QRectF(7, 17, 9, 6))
        note.moveTo(15, 20)
        note.lineTo(15, 7)
        note.cubicTo(21, 8, 25, 10, 27, 14)
        note.cubicTo(23, 13, 20, 12, 17, 12)
        painter.drawPath(note)
        painter.fillPath(note, QBrush(QColor("#f5f7fb")))
        painter.setPen(QPen(accent, 2))
        painter.drawPath(note)
        painter.setBrush(QBrush(QColor("#61c9e8")))
        painter.setPen(QPen(shadow, 2))
        painter.drawEllipse(QRectF(7, 7, 6, 6))
        painter.setPen(QPen(QColor("#61c9e8"), 1.5))
        painter.drawEllipse(QRectF(7, 7, 6, 6))
    elif name == "horizontal_resize":
        _cursor_wave(painter, 8, 16, accent, shadow)
        for color in (shadow, light):
            _cursor_triangle(painter, [(4, 16), (10, 10), (10, 22)], color)
            _cursor_triangle(painter, [(28, 16), (22, 10), (22, 22)], color)
        _cursor_staff(painter, 9, 23, 25, QColor("#eef5ff"), 1.0)
    elif name == "gain_vertical":
        painter.setPen(QPen(shadow, 5))
        painter.drawLine(11, 5, 11, 27)
        painter.setPen(QPen(light, 2))
        painter.drawLine(11, 5, 11, 27)
        painter.setBrush(QBrush(accent))
        painter.setPen(QPen(shadow, 2))
        painter.drawRoundedRect(QRectF(7, 13, 8, 6), 2, 2)
        for color in (shadow, light):
            _cursor_triangle(painter, [(11, 3), (6, 9), (16, 9)], color)
            _cursor_triangle(painter, [(11, 29), (6, 23), (16, 23)], color)
        speaker = QPainterPath()
        speaker.moveTo(19, 13)
        speaker.lineTo(22, 13)
        speaker.lineTo(26, 9)
        speaker.lineTo(26, 24)
        speaker.lineTo(22, 20)
        speaker.lineTo(19, 20)
        speaker.closeSubpath()
        painter.setPen(QPen(shadow, 3))
        painter.drawPath(speaker)
        painter.setPen(QPen(light, 1.8))
        painter.drawPath(speaker)
        painter.setPen(QPen(shadow, 3))
        painter.drawArc(24, 12, 7, 9, -45 * 16, 90 * 16)
        painter.setPen(QPen(light, 1.6))
        painter.drawArc(24, 12, 7, 9, -45 * 16, 90 * 16)
    elif name == "scissors":
        hot_x = 7
        painter.setPen(QPen(shadow, 4))
        painter.drawLine(7, 3, 7, 29)
        painter.setPen(QPen(accent, 2))
        painter.drawLine(7, 3, 7, 29)
        painter.setPen(QPen(shadow, 4))
        painter.drawEllipse(14, 19, 5, 5)
        painter.drawEllipse(23, 19, 5, 5)
        painter.drawLine(18, 20, 27, 8)
        painter.drawLine(25, 20, 15, 8)
        painter.setPen(QPen(light, 2))
        painter.drawEllipse(14, 19, 5, 5)
        painter.drawEllipse(23, 19, 5, 5)
        painter.drawLine(18, 20, 27, 8)
        painter.drawLine(25, 20, 15, 8)
    elif name == "cut_merge":
        painter.setPen(QPen(shadow, 4))
        painter.drawLine(16, 3, 16, 29)
        painter.setPen(QPen(merge_line, 2))
        painter.drawLine(16, 3, 16, 29)
        painter.fillRect(3, 13, 8, 6, merge_accent)
        _cursor_triangle(painter, [(15, 16), (10, 8), (10, 24)], merge_accent)
        painter.fillRect(21, 13, 8, 6, merge_accent)
        _cursor_triangle(painter, [(17, 16), (22, 8), (22, 24)], merge_accent)
    elif name == "pitch_point":
        hot_x = 12
        hot_y = 12
        painter.setPen(QPen(shadow, 5))
        pitch_path = QPainterPath()
        pitch_path.moveTo(4, 19)
        pitch_path.cubicTo(8, 8, 12, 24, 17, 13)
        pitch_path.cubicTo(20, 7, 23, 9, 27, 6)
        painter.drawPath(pitch_path)
        painter.setPen(QPen(accent, 2.2))
        painter.drawPath(pitch_path)
        painter.setBrush(QBrush(QColor("#fff3a6")))
        painter.setPen(QPen(shadow, 2))
        painter.drawEllipse(QRectF(10, 9, 8, 8))
        painter.setPen(QPen(light, 2))
        painter.drawLine(20, 21, 30, 21)
        painter.drawLine(25, 16, 25, 26)
    elif name == "pitch_curve_select":
        hot_x = 5
        hot_y = 3
        pointer = QPainterPath()
        pointer.moveTo(5, 3)
        pointer.cubicTo(14, 6, 22, 12, 22, 19)
        pointer.cubicTo(22, 26, 15, 29, 9, 24)
        pointer.cubicTo(5, 20, 4, 11, 5, 3)
        pointer.closeSubpath()
        painter.setPen(QPen(shadow, 4))
        painter.drawPath(pointer)
        painter.fillPath(pointer, QBrush(QColor("#f5f7fb")))
        painter.setPen(QPen(accent, 1.7))
        painter.drawPath(pointer)
        _cursor_wave(painter, 15, 24, QColor("#42f5ff"), shadow)
    elif name == "pitch_curve_point":
        hot_x = 12
        hot_y = 12
        painter.setPen(QPen(shadow, 5))
        pitch_path = QPainterPath()
        pitch_path.moveTo(4, 19)
        pitch_path.cubicTo(8, 8, 12, 24, 17, 13)
        pitch_path.cubicTo(20, 7, 23, 9, 27, 6)
        painter.drawPath(pitch_path)
        painter.setPen(QPen(accent, 2.2))
        painter.drawPath(pitch_path)
        painter.setBrush(QBrush(QColor("#fff3a6")))
        painter.setPen(QPen(shadow, 2))
        painter.drawEllipse(QRectF(10, 9, 8, 8))
    elif name in {
        "pitch_vibrato",
        "pitch_vibrato_sine",
        "pitch_vibrato_triangle",
        "pitch_vibrato_square",
    }:
        hot_x = 12
        hot_y = 13
        waveform = name.removeprefix("pitch_vibrato_")
        if waveform == "pitch_vibrato":
            waveform = "sine"
        painter.setPen(QPen(shadow, 4))
        painter.drawLine(4, 19, 23, 19)
        painter.setPen(QPen(QColor("#fff3a6"), 1.6))
        painter.drawLine(4, 19, 23, 19)
        wave_path = QPainterPath()
        wave_path.moveTo(4, 13)
        wave_path.cubicTo(6, 5, 9, 21, 11, 13)
        wave_path.cubicTo(13, 5, 16, 21, 18, 13)
        wave_path.cubicTo(20, 8, 22, 10, 24, 9)
        painter.setPen(QPen(shadow, 4))
        painter.drawPath(wave_path)
        painter.setPen(QPen(accent, 2.2))
        painter.drawPath(wave_path)
        badge_rect = QRectF(21, 21, 10, 9)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(8, 12, 16, 220)))
        painter.drawRoundedRect(badge_rect.adjusted(-1, -1, 1, 1), 3, 3)
        painter.setPen(QPen(QColor("#fff3a6"), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if waveform == "triangle":
            marker = QPainterPath()
            marker.moveTo(21.5, 28.0)
            marker.lineTo(24.2, 22.0)
            marker.lineTo(27.0, 28.0)
            marker.lineTo(30.0, 22.0)
            painter.drawPath(marker)
        elif waveform == "square":
            marker = QPainterPath()
            marker.moveTo(21.5, 27.0)
            marker.lineTo(21.5, 23.0)
            marker.lineTo(25.5, 23.0)
            marker.lineTo(25.5, 27.0)
            marker.lineTo(30.0, 27.0)
            painter.drawPath(marker)
        else:
            marker = QPainterPath()
            marker.moveTo(21.5, 25.5)
            marker.cubicTo(23.0, 21.0, 25.0, 30.0, 26.8, 25.5)
            marker.cubicTo(28.0, 22.0, 29.0, 24.0, 30.0, 23.0)
            painter.drawPath(marker)
    elif name == "pitch_control_pitch":
        hot_x = 12
        hot_y = 16
        painter.setPen(QPen(shadow, 5))
        painter.drawLine(12, 5, 12, 27)
        painter.setPen(QPen(light, 2.2))
        painter.drawLine(12, 5, 12, 27)
        for color in (shadow, light):
            _cursor_triangle(painter, [(12, 3), (7, 10), (17, 10)], color)
            _cursor_triangle(painter, [(12, 29), (7, 22), (17, 22)], color)
        painter.setPen(QPen(shadow, 3))
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRectF(18, 4, 13, 11), Qt.AlignCenter, "♯")
        painter.drawText(QRectF(18, 17, 13, 11), Qt.AlignCenter, "♭")
        painter.setPen(QPen(QColor("#fff3a6"), 1.5))
        painter.drawText(QRectF(18, 4, 13, 11), Qt.AlignCenter, "♯")
        painter.drawText(QRectF(18, 17, 13, 11), Qt.AlignCenter, "♭")
    elif name in {
        "vibrato_frequency_up",
        "vibrato_frequency_down",
        "vibrato_amplitude_up",
        "vibrato_amplitude_down",
    }:
        hot_x = 16
        hot_y = 16
        arrow_color = QColor("#f5f7fb")
        painter.setPen(QPen(shadow, 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if name == "vibrato_frequency_up":
            painter.drawLine(7, 16, 25, 16)
            _cursor_triangle(painter, [(27, 16), (19, 10), (19, 22)], shadow)
        elif name == "vibrato_frequency_down":
            painter.drawLine(25, 16, 7, 16)
            _cursor_triangle(painter, [(5, 16), (13, 10), (13, 22)], shadow)
        elif name == "vibrato_amplitude_up":
            painter.drawLine(16, 25, 16, 7)
            _cursor_triangle(painter, [(16, 5), (10, 13), (22, 13)], shadow)
        elif name == "vibrato_amplitude_down":
            painter.drawLine(16, 7, 16, 25)
            _cursor_triangle(painter, [(16, 27), (10, 19), (22, 19)], shadow)
        painter.setPen(QPen(arrow_color, 2.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if name == "vibrato_frequency_up":
            painter.drawLine(7, 16, 25, 16)
            _cursor_triangle(painter, [(27, 16), (19, 10), (19, 22)], arrow_color)
        elif name == "vibrato_frequency_down":
            painter.drawLine(25, 16, 7, 16)
            _cursor_triangle(painter, [(5, 16), (13, 10), (13, 22)], arrow_color)
        elif name == "vibrato_amplitude_up":
            painter.drawLine(16, 25, 16, 7)
            _cursor_triangle(painter, [(16, 5), (10, 13), (22, 13)], arrow_color)
        elif name == "vibrato_amplitude_down":
            painter.drawLine(16, 7, 16, 25)
            _cursor_triangle(painter, [(16, 27), (10, 19), (22, 19)], arrow_color)
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        label = {
            "vibrato_frequency_up": "f",
            "vibrato_frequency_down": "f",
            "vibrato_amplitude_up": "A",
            "vibrato_amplitude_down": "A",
        }[name]
        badge = QRectF(22, 0, 10, 12)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(10, 14, 18, 210)))
        painter.drawRoundedRect(badge.adjusted(-2, 0, 1, 1), 3, 3)
        painter.setPen(QPen(shadow, 3))
        painter.drawText(badge, Qt.AlignCenter, label)
        painter.setPen(QPen(QColor("#fff3a6"), 1.8))
        painter.drawText(badge, Qt.AlignCenter, label)
        painter.setPen(QPen(shadow, 4))
        wave = QPainterPath()
        wave.moveTo(5, 28)
        wave.cubicTo(9, 22, 13, 32, 17, 27)
        wave.cubicTo(21, 22, 25, 32, 30, 26)
        painter.drawPath(wave)
        painter.setPen(QPen(accent, 2))
        painter.drawPath(wave)
    elif name == "move":
        hot_x = 5
        hot_y = 3
        pointer = QPainterPath()
        pointer.moveTo(5, 3)
        pointer.cubicTo(14, 6, 22, 12, 22, 19)
        pointer.cubicTo(22, 26, 15, 29, 9, 24)
        pointer.cubicTo(5, 20, 4, 11, 5, 3)
        pointer.closeSubpath()
        painter.setPen(QPen(shadow, 4))
        painter.drawPath(pointer)
        painter.fillPath(pointer, QBrush(QColor("#f5f7fb")))
        painter.setPen(QPen(accent, 1.7))
        painter.drawPath(pointer)

        painter.setPen(QPen(shadow, 4))
        painter.drawLine(15, 23, 31, 23)
        painter.drawLine(23, 15, 23, 31)
        painter.setPen(QPen(light, 2.3))
        painter.drawLine(15, 23, 31, 23)
        painter.drawLine(23, 15, 23, 31)
        for color in (shadow, light):
            _cursor_triangle(painter, [(13, 23), (19, 19), (19, 27)], color)
            _cursor_triangle(painter, [(32, 23), (26, 19), (26, 27)], color)
            _cursor_triangle(painter, [(23, 13), (19, 19), (27, 19)], color)
            _cursor_triangle(painter, [(23, 32), (19, 26), (27, 26)], color)
        painter.setBrush(QBrush(accent))
        painter.setPen(QPen(shadow, 1.4))
        painter.drawEllipse(QRectF(20.8, 20.8, 4.4, 4.4))
    else:
        painter.end()
        cursor = QCursor(Qt.ArrowCursor)
        _CURSOR_CACHE[name] = cursor
        return cursor

    painter.end()
    cursor = QCursor(pixmap, hot_x, hot_y)
    _CURSOR_CACHE[name] = cursor
    return cursor


class AudioSliceGraphicsItem(QGraphicsRectItem):
    """Editable audio slice block with track binding, lock visuals, and snap hooks."""

    PIXELS_PER_SECOND = 260.0
    PITCH_PIXELS_PER_SEMITONE = 20.0
    RESIZE_MARGIN = 5.0
    GAIN_EDGE_MARGIN = 5.0
    GAIN_HOVER_DELAY_SECONDS = 0.2
    GAIN_HOVER_STABLE_PIXELS = 2.5
    MIN_WIDTH = 24.0
    # Leave vertical room for the edit overlay. A pitch curve is more useful
    # when small drift can visibly leave the note body.
    PITCH_CURVE_OVERFLOW = 360.0
    PITCH_CURVE_PIXELS_PER_SEMITONE = 14.0
    PITCH_CURVE_HIT_RADIUS = 16.0
    PITCH_VIBRATO_HIT_RADIUS = 24.0
    PITCH_CONTROL_HIT_RADIUS = 20.0
    PITCH_CONTROL_HANDLE_SIZE = 24.0
    PITCH_CONTROL_HOVER_DELAY_SECONDS = 0.35
    PITCH_CONTROL_HOVER_STABLE_PIXELS = 1.5

    @property
    def edit_model(self) -> AudioSliceEditModel:
        return self._edit_model

    @property
    def audio_slice(self) -> AudioSlice:
        return self._edit_model.clip.audio_slice

    @audio_slice.setter
    def audio_slice(self, value: AudioSlice) -> None:
        self._edit_model.clip.audio_slice = value

    @property
    def track_index(self) -> int:
        return self._edit_model.clip.track_index

    @track_index.setter
    def track_index(self, value: int) -> None:
        self._edit_model.clip.track_index = max(0, int(value))

    @property
    def target_midi_note(self) -> int | None:
        return self._edit_model.clip.pitch_center_midi

    @target_midi_note.setter
    def target_midi_note(self, value: int | None) -> None:
        self._edit_model.clip.pitch_center_midi = None if value is None else int(value)

    @property
    def target_duration(self) -> float:
        return max(0.001, float(self._edit_model.clip.target_duration or 0.001))

    @target_duration.setter
    def target_duration(self, value: float) -> None:
        self._edit_model.clip.target_duration = max(0.001, float(value))

    @property
    def gain_db(self) -> float:
        return self._edit_model.gain_db

    @gain_db.setter
    def gain_db(self, value: float) -> None:
        self._edit_model.gain_db = float(value)

    @property
    def pitch_flatten_amount(self) -> float:
        return self._edit_model.pitch_flatten_amount

    @pitch_flatten_amount.setter
    def pitch_flatten_amount(self, value: float) -> None:
        self._edit_model.pitch_flatten_amount = float(value)

    @property
    def formant_shift(self) -> float:
        return self._edit_model.formant_shift

    @formant_shift.setter
    def formant_shift(self, value: float) -> None:
        self._edit_model.formant_shift = float(value)

    @property
    def protect_transients(self) -> bool:
        return self._edit_model.protect_transients

    @protect_transients.setter
    def protect_transients(self, value: bool) -> None:
        self._edit_model.protect_transients = bool(value)

    @property
    def pitch_control_points(self) -> list[tuple[float, float]]:
        return self._edit_model.pitch_automation.control_point_pairs()

    @pitch_control_points.setter
    def pitch_control_points(self, values: object) -> None:
        self._edit_model.pitch_automation.set_control_points(values)

    @property
    def pitch_vibrato_regions(self) -> list[dict[str, float | str]]:
        return self._edit_model.pitch_automation.vibrato_regions_payload()

    @pitch_vibrato_regions.setter
    def pitch_vibrato_regions(self, values: object) -> None:
        self._edit_model.pitch_automation.set_vibrato_regions(values)

    def __init__(
        self,
        audio_slice: AudioSlice,
        track_index: int,
        width: float,
        height: float,
        color: QColor,
        parent=None,
    ) -> None:
        super().__init__(0, 0, width, height, parent)
        self._edit_model = AudioSliceEditModel.create(audio_slice, track_index)
        self.track_type = "vocal_slice"
        self.is_locked = False
        self.snap_to_grid_enabled = False
        self.snap_grid_size = 120.0
        self.base_audio_cache = None
        self.base_audio_sample_rate: int | None = None
        self.base_audio_level_dbfs = None
        self.waveform_envelope = None
        self.render_cache_audio = None
        self.render_cache_sample_rate: int | None = None
        self.render_cache_parameters = None
        self.render_cache_level_dbfs = None
        self.is_rendering = False
        self.is_missing_source = False
        self.is_track_reference = False
        self.reference_editable = False
        self.pitch_contour = None
        self.pitch_curve_center_midi: float | None = None
        self.pitch_shape_regions: list[dict[str, float | str]] = []
        self.pitch_curve_edit_mode = False
        self.placement_group_id = _new_placement_group_id("item")
        self._active_render_key: str | None = None
        self._pitch_anchor_y: float | None = None
        self._resize_edge: str | None = None
        self._gain_dragging = False
        self._gain_start_scene_y = 0.0
        self._gain_start_db = 0.0
        self._gain_peer_start_states: dict[int, tuple[AudioSliceGraphicsItem, dict[str, object], float]] = {}
        self._gain_hover_edge: str | None = None
        self._gain_hover_started_at = 0.0
        self._gain_hover_armed = False
        self._gain_hover_position: QPointF | None = None
        self._gain_hover_token = 0
        self._gain_drag_cursor_hidden = False
        self._resize_drag_cursor_hidden = False
        self._pitch_control_drag_index: int | None = None
        self._pitch_control_drag_start_state: dict[str, object] | None = None
        self._pitch_control_drag_start_points: list[tuple[float, float]] | None = None
        self._pitch_control_drag_start_selection: set[int] = set()
        self._pitch_control_drag_start_local: QPointF | None = None
        self._pitch_control_drag_axis = "xy"
        self._hover_pitch_control_index: int | None = None
        self._selected_pitch_control_index: int | None = None
        self._selected_pitch_control_indices: set[int] = set()
        self._pitch_curve_hovered = False
        self._hover_pitch_curve_range: tuple[float, float] | None = None
        self._selected_pitch_curve_ranges: list[tuple[float, float]] = []
        self._pitch_control_hover_edge: str | None = None
        self._pitch_control_hover_started_at = 0.0
        self._pitch_control_hover_armed = False
        self._pitch_control_hover_position: QPointF | None = None
        self._pitch_control_hover_token = 0
        self._applying_saved_state = False
        self._suppress_edit_notifications = False
        self._edit_start_state: dict[str, object] | None = None
        self._resize_start_scene_x = 0.0
        self._resize_start_width = width
        self._resize_start_pos = QPointF()
        self._edit_start_pos = QPointF()
        self._edit_start_width = width
        self._edit_start_target_midi = self.target_midi_note
        self._edit_start_target_duration = self.target_duration
        self._edit_peer_items: list[AudioSliceGraphicsItem] = []
        self._cut_merge_marked_edge: str | None = None
        self.label = self._make_label()
        self.base_color = QColor(color)

        self.setAcceptHoverEvents(True)
        self.setPen(QPen(QColor("#d8e1ec")))
        self.setBrush(QBrush(self.base_color))
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setData(0, audio_slice.to_dict())
        self.setData(1, track_index)
        self.setToolTip(
            f"Track {track_index + 1}\n"
            f"{audio_slice.source_path}\n"
            f"{audio_slice.start_time:.3f}s - {audio_slice.end_time:.3f}s\n"
            f"MIDI: {audio_slice.midi_note if audio_slice.midi_note is not None else 'N/A'}"
        )

    def update(self, *args) -> None:  # type: ignore[override]
        super().update(*args)
        scene = self.scene()
        if scene is None:
            return
        overlay = scene.property("pitch_curve_overlay_item")
        if isinstance(overlay, QGraphicsItem):
            overlay.update()

    def set_locked(self, locked: bool) -> None:
        self.is_locked = locked
        self._apply_edit_flags()
        self.update()

    def set_track_reference(self, enabled: bool, editable: bool = False) -> None:
        self.is_track_reference = bool(enabled)
        self.reference_editable = bool(editable)
        if self.is_track_reference and not self.reference_editable:
            self.setBrush(QBrush(QColor("#69717d")))
            self.setPen(QPen(QColor("#9aa5b2"), 1))
            self.setZValue(-30.0)
        else:
            self.setBrush(QBrush(self.base_color))
            self.setPen(QPen(QColor("#d8e1ec")))
            self.setZValue(0.0)
        self._apply_edit_flags()
        self.label = self._make_label()
        self.update()

    def _apply_edit_flags(self) -> None:
        fully_locked = self._is_fully_locked()
        self.setFlag(QGraphicsItem.ItemIsSelectable, not fully_locked)
        self.setFlag(QGraphicsItem.ItemIsMovable, not fully_locked)
        if fully_locked:
            self.setSelected(False)
        self.update()

    def set_missing_source(self, missing: bool) -> None:
        self.is_missing_source = missing
        if missing:
            self.setBrush(QBrush(QColor("#7b2d35")))
            self.setPen(QPen(QColor("#ff6b7a"), 2))
            self.setToolTip(f"Missing source file:\n{self.audio_slice.source_path}")
        else:
            self.setBrush(QBrush(self.base_color))
            self.setPen(QPen(QColor("#d8e1ec")))
        self.update()

    def set_track_type(self, track_type: str) -> None:
        self.track_type = track_type
        if self._is_master_bgm():
            self.target_midi_note = self.audio_slice.midi_note
            self.target_duration = self.audio_slice.duration
            self.setRect(
                0,
                0,
                max(self.MIN_WIDTH, self.audio_slice.duration * self._pixels_per_second()),
                self.rect().height(),
            )
        self.set_locked(self.is_locked)
        self.label = self._make_label()
        self.update()

    def set_pitch_anchor_y(self, y: float) -> None:
        self._pitch_anchor_y = y

    def set_snap_to_grid(self, enabled: bool, grid_size: float | None = None) -> None:
        self.snap_to_grid_enabled = enabled
        if grid_size is not None and grid_size > 0:
            self.snap_grid_size = grid_size

    def _pixels_per_second(self) -> float:
        scene = self.scene()
        if scene is not None:
            for view in scene.views():
                if hasattr(view, "pixels_per_second"):
                    try:
                        return float(view.pixels_per_second())
                    except Exception:
                        break
        return self.PIXELS_PER_SECOND

    def render_cache_key(self) -> str:
        return f"{self.base_cache_key()}|{self.build_render_request().cache_signature()}"

    def base_cache_key(self) -> str:
        return (
            f"{self.audio_slice.source_path}|slice={self.audio_slice.index}|"
            f"start={self.audio_slice.start_time:.6f}|end={self.audio_slice.end_time:.6f}|"
            f"track={self.track_index}"
        )

    def build_render_request(
        self,
        advanced_controls_enabled: bool = True,
    ) -> SliceRenderRequest:
        return self._edit_model.build_render_request(advanced_controls_enabled)

    def current_render_parameters(self):
        from hakyking.audio.audio_engine import build_render_parameters_from_request

        return build_render_parameters_from_request(self.build_render_request())

    def requires_rendered_audio(self, advanced_controls_enabled: bool = True) -> bool:
        """Return whether base audio can no longer represent the edit state."""

        pitch_changed = (
            self.target_midi_note is not None
            and self.audio_slice.midi_note is not None
            and int(self.target_midi_note) != int(self.audio_slice.midi_note)
        )
        duration_changed = abs(self.target_duration - self.audio_slice.duration) > 1e-4
        control_curve_changed = any(
            abs(float(point.get("offset", 0.0))) > 1e-4
            for point in self.pitch_control_points_payload()
        )
        vibrato_changed = bool(self.pitch_vibrato_regions_payload())
        advanced_changed = bool(
            advanced_controls_enabled
            and (
                abs(self.pitch_flatten_amount) > 1e-4
                or abs(self.formant_shift) > 1e-4
            )
        )
        return bool(
            pitch_changed
            or duration_changed
            or abs(self.gain_db) > 1e-4
            or control_curve_changed
            or vibrato_changed
            or advanced_changed
        )

    def store_render_result(self, result) -> None:
        if result.cache_key != self._active_render_key:
            return
        if result.cache_key != self.render_cache_key():
            self.set_rendering(False)
            return
        self.render_cache_audio = result.audio
        self.render_cache_sample_rate = result.sample_rate
        self.render_cache_parameters = result.parameters
        self.render_cache_level_dbfs = measure_audio_dbfs(result.audio)
        self.set_rendering(False)
        self.setData(2, result.cache_key)
        self.setData(3, result.parameters.n_steps)
        self.setData(4, result.parameters.rate)

    def set_rendering(self, rendering: bool, cache_key: str | None = None) -> None:
        self.is_rendering = rendering
        self._active_render_key = cache_key if rendering else None
        self.update()

    def store_waveform_result(self, result) -> None:
        self.base_audio_cache = result.audio
        self.base_audio_sample_rate = result.sample_rate
        self.base_audio_level_dbfs = measure_audio_dbfs(result.audio)
        self.waveform_envelope = result.envelope
        incoming_pitch_contour = getattr(result, "pitch_contour", None)
        # A split inherits the already analysed parent contour. Re-analysing
        # each child has different boundary context and must not visually
        # rewrite the curve just because the user made a cut.
        if self.pitch_contour is None:
            self.pitch_contour = incoming_pitch_contour
        if self.pitch_curve_center_midi is None:
            self.pitch_curve_center_midi = self._detected_pitch_center_midi()
        self.update()

    def set_gain_db(self, gain_db: float) -> None:
        self._edit_model.set_gain_db(gain_db, MIN_GAIN_DB, MAX_GAIN_DB)
        self.label = self._make_label()
        self.update()

    def set_pitch_flatten_amount(self, amount: float) -> None:
        self._edit_model.set_pitch_flatten_amount(amount)
        self.label = self._make_label()
        self.update()

    def set_formant_shift(self, semitones: float) -> None:
        self._edit_model.set_formant_shift(semitones)
        self.label = self._make_label()
        self.update()

    def set_transient_protection(self, enabled: bool) -> None:
        self._edit_model.protect_transients = bool(enabled)
        self.label = self._make_label()
        self.update()

    def set_pitch_control_points(self, points: object) -> None:
        normalized: list[tuple[float, float]] = []
        if isinstance(points, (list, tuple)):
            for entry in points:
                try:
                    if isinstance(entry, dict):
                        raw_x = entry.get("x", entry.get("ratio", 0.0))
                        raw_offset = entry.get("offset", entry.get("semitones", 0.0))
                    else:
                        raw_x = entry[0]  # type: ignore[index]
                        raw_offset = entry[1]  # type: ignore[index]
                    x_value = max(0.0, min(1.0, float(raw_x)))
                    offset = max(-24.0, min(24.0, float(raw_offset)))
                except (TypeError, ValueError, IndexError, KeyError, OverflowError):
                    continue
                normalized.append((x_value, offset))
        normalized.sort(key=lambda point: point[0])
        collapsed: list[tuple[float, float]] = []
        for x_value, offset in normalized:
            if collapsed and abs(x_value - collapsed[-1][0]) < 0.001:
                collapsed[-1] = (x_value, offset)
            else:
                collapsed.append((x_value, offset))
        self._edit_model.pitch_automation.set_control_points(collapsed)
        if (
            self._selected_pitch_control_index is not None
            and self._selected_pitch_control_index >= len(self.pitch_control_points)
        ):
            self._selected_pitch_control_index = None
        self._selected_pitch_control_indices = {
            index
            for index in self._selected_pitch_control_indices
            if index < len(self.pitch_control_points)
        }
        if (
            self._hover_pitch_control_index is not None
            and self._hover_pitch_control_index >= len(self.pitch_control_points)
        ):
            self._hover_pitch_control_index = None
        self.update()

    def pitch_control_points_payload(self) -> list[dict[str, float]]:
        return self._edit_model.pitch_automation.control_points_payload()

    def set_pitch_vibrato_regions(self, regions: object) -> None:
        normalized: list[dict[str, float | str]] = []
        if isinstance(regions, (list, tuple)):
            for entry in regions:
                if not isinstance(entry, dict):
                    continue
                try:
                    start = max(0.0, min(1.0, float(entry.get("start", 0.0))))
                    end = max(0.0, min(1.0, float(entry.get("end", 1.0))))
                    if end < start:
                        start, end = end, start
                    if end - start <= 1e-5:
                        continue
                    cycles = max(0.0, float(entry.get("cycles", 0.0)))
                    depth = max(0.0, min(12.0, float(entry.get("depth", 0.0))))
                    phase = float(entry.get("phase", 0.0)) % 1.0
                except (TypeError, ValueError, OverflowError):
                    continue
                waveform = str(entry.get("waveform", "sine"))
                if waveform not in {"sine", "triangle", "square"}:
                    waveform = "sine"
                if cycles <= 0.0 or depth <= 0.0:
                    continue
                normalized.append(
                    {
                        "start": round(start, 6),
                        "end": round(end, 6),
                        "cycles": round(cycles, 3),
                        "depth": round(depth, 4),
                        "phase": round(phase, 4),
                        "waveform": waveform,
                    }
                )
        normalized.sort(key=lambda region: (float(region["start"]), float(region["end"])))
        self._edit_model.pitch_automation.set_vibrato_regions(normalized)
        self.update()

    def pitch_vibrato_regions_payload(self) -> list[dict[str, float | str]]:
        return self._edit_model.pitch_automation.vibrato_regions_payload()

    def set_pitch_shape_regions(self, regions: object) -> None:
        self.pitch_shape_regions = []
        self.update()

    def pitch_shape_regions_payload(self) -> list[dict[str, float | str]]:
        return []

    def render_pitch_control_points_payload(self) -> list[dict[str, float]]:
        ratios = {float(x_value) for x_value, _offset in self.pitch_control_points}
        payload: list[dict[str, float]] = []
        for ratio in sorted(ratios):
            offset = self._pitch_control_offset_at_ratio(ratio, include_vibrato=False)
            if abs(offset) >= 0.001 or any(abs(ratio - point[0]) <= 1e-5 for point in self.pitch_control_points):
                payload.append({"x": round(ratio, 6), "offset": round(offset, 4)})
        return payload

    def pitch_curve_clipboard_payload(self) -> list[dict[str, float]]:
        if self.pitch_control_points:
            return self.pitch_control_points_payload()
        if self.pitch_contour is None or len(self.pitch_contour) == 0:
            return []
        center_midi = self._pitch_curve_center_midi()
        ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
        return [
            {
                "x": ratio,
                "offset": round(
                    max(
                        -24.0,
                        min(24.0, self._interpolated_contour_midi(ratio) - center_midi),
                    ),
                    4,
                ),
            }
            for ratio in ratios
        ]

    def apply_pitch_curve_zero(self) -> None:
        self.set_pitch_control_points(
            [{"x": 0.0, "offset": 0.0}, {"x": 1.0, "offset": 0.0}]
        )

    def apply_pitch_curve_natural(self) -> None:
        self.set_pitch_control_points([])
        self.set_pitch_flatten_amount(0.0)

    def apply_pitch_curve_electro(self) -> None:
        self.apply_pitch_curve_zero()
        self.set_pitch_flatten_amount(1.0)

    def apply_pitch_curve_smooth_glide(self) -> None:
        center_midi = self._pitch_curve_center_midi()
        ratios = [index / 6.0 for index in range(7)]
        points: list[dict[str, float]] = []
        for ratio in ratios:
            nearby = [
                self._interpolated_contour_midi(max(0.0, min(1.0, ratio + delta)))
                - center_midi
                for delta in (-0.12, -0.06, 0.0, 0.06, 0.12)
            ]
            smoothed_offset = sum(nearby) / len(nearby)
            original_offset = self._interpolated_contour_midi(ratio) - center_midi
            points.append({"x": ratio, "offset": smoothed_offset - original_offset})
        self.set_pitch_control_points(points)

    def apply_pitch_curve_vibrato(
        self,
        depth: float = 0.35,
        cycles: float = 4.0,
        waveform: str = "sine",
        phase: float = 0.0,
        base_points: list[tuple[float, float]] | None = None,
        base_regions: list[dict[str, float | str]] | None = None,
    ) -> None:
        ranges = self.selected_pitch_curve_ranges()
        if not ranges:
            return
        original_points = list(base_points if base_points is not None else self.pitch_control_points)
        original_regions = list(
            base_regions if base_regions is not None else self.pitch_vibrato_regions_payload()
        )
        waveform = waveform if waveform in {"sine", "triangle", "square"} else "sine"
        cycles = max(0.0, float(cycles))
        depth = max(0.0, min(12.0, float(depth)))
        phase = float(phase) % 1.0
        if cycles <= 0.0 or depth <= 0.0:
            self.pitch_control_points = sorted(original_points, key=lambda point: point[0])
            self.set_pitch_vibrato_regions(original_regions)
            self._selected_pitch_curve_ranges = ranges
            self.update()
            return

        self.pitch_control_points = sorted(original_points, key=lambda point: point[0])
        next_regions: list[dict[str, float | str]] = []
        for region in original_regions:
            try:
                region_start = float(region["start"])
                region_end = float(region["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if any(not (region_end <= start + 1e-6 or region_start >= end - 1e-6) for start, end in ranges):
                continue
            next_regions.append(dict(region))
        for start, end in ranges:
            next_regions.append(
                {
                    "start": start,
                    "end": end,
                    "cycles": cycles,
                    "depth": depth,
                    "phase": phase,
                    "waveform": waveform,
                }
            )
        self.set_pitch_vibrato_regions(next_regions)
        self._selected_pitch_curve_ranges = ranges

    def apply_pitch_curve_segment_shape(self, shape: str) -> bool:
        return False

    def _pitch_curve_segment_shape_points(
        self,
        start: float,
        end: float,
        start_pitch: float,
        end_pitch: float,
        shape: str,
    ) -> list[tuple[float, float]]:
        start = max(0.0, min(1.0, float(start)))
        end = max(0.0, min(1.0, float(end)))
        if end < start:
            start, end = end, start
            start_pitch, end_pitch = end_pitch, start_pitch
        span = end - start
        if span <= 1e-5:
            return []

        shape = str(shape or "linear")
        if shape == "original":
            sample_ratios = [start, end]
            return [(ratio, self._offset_for_target_pitch(ratio, self._interpolated_contour_midi(ratio))) for ratio in sample_ratios]

        if shape == "instant":
            epsilon = max(0.0015, min(0.006, span * 0.08))
            middle = start + span * 0.5
            sample_ratios = [
                start,
                max(start + 0.0015, middle - epsilon),
                min(end - 0.0015, middle + epsilon),
                end,
            ]
            target_pitches = [start_pitch, start_pitch, end_pitch, end_pitch]
            return [
                (ratio, self._offset_for_target_pitch(ratio, target_pitch))
                for ratio, target_pitch in zip(sample_ratios, target_pitches)
            ]

        sample_count = max(9, min(49, int(math.ceil(span * 96.0))))
        points: list[tuple[float, float]] = []
        for index in range(sample_count):
            t = 0.0 if sample_count <= 1 else index / (sample_count - 1)
            ratio = start + span * t
            factor = self._pitch_curve_shape_factor(shape, t)
            target_pitch = start_pitch + (end_pitch - start_pitch) * factor
            points.append((ratio, self._offset_for_target_pitch(ratio, target_pitch)))
        return points

    def _offset_for_target_pitch(self, ratio: float, target_pitch: float) -> float:
        baseline = self._interpolated_contour_midi(ratio)
        return max(-24.0, min(24.0, float(target_pitch) - baseline))

    @staticmethod
    def _pitch_curve_shape_factor(shape: str, t: float) -> float:
        t = max(0.0, min(1.0, float(t)))
        if shape == "ease_in":
            return t * t
        if shape == "ease_out":
            return 1.0 - (1.0 - t) * (1.0 - t)
        if shape in {"smooth", "s_curve"}:
            return t * t * (3.0 - 2.0 * t)
        if shape == "instant":
            return 0.0 if t < 0.5 else 1.0
        return t

    @staticmethod
    def _pitch_control_offset_at_ratio_from_points(
        points: list[tuple[float, float]],
        ratio: float,
    ) -> float:
        if not points:
            return 0.0
        sorted_points = sorted(points)
        ratio = max(0.0, min(1.0, float(ratio)))
        if ratio <= sorted_points[0][0]:
            return float(sorted_points[0][1])
        if ratio >= sorted_points[-1][0]:
            return float(sorted_points[-1][1])
        for left, right in zip(sorted_points, sorted_points[1:]):
            left_x, left_offset = left
            right_x, right_offset = right
            if left_x <= ratio <= right_x:
                span = max(1e-6, right_x - left_x)
                t = (ratio - left_x) / span
                return float(left_offset + (right_offset - left_offset) * t)
        return 0.0

    def boundingRect(self) -> QRectF:  # type: ignore[override]
        rect = super().boundingRect()
        if self.pitch_curve_edit_mode:
            rect = rect.adjusted(
                -24.0,
                -self.PITCH_CURVE_OVERFLOW,
                24.0,
                self.PITCH_CURVE_OVERFLOW,
            )
        if self._should_show_gain_badge():
            rect = rect.united(self._gain_badge_rect().adjusted(-4.0, -4.0, 4.0, 4.0))
        return rect

    def shape(self) -> QPainterPath:  # type: ignore[override]
        path = super().shape()
        if not self.pitch_curve_edit_mode:
            return path
        curve_path = QPainterPath()
        has_curve = False
        for segment in self._pitch_curve_local_segments():
            if not segment:
                continue
            if len(segment) == 1:
                point = segment[0]
                curve_path.addEllipse(QRectF(point.x() - 12.0, point.y() - 12.0, 24.0, 24.0))
                has_curve = True
                continue
            segment_path = QPainterPath(segment[0])
            for point in segment[1:]:
                segment_path.lineTo(point)
            curve_path.addPath(segment_path)
            has_curve = True
        for ratio, offset in self.pitch_control_points:
            point = self._pitch_control_point_to_local(ratio, offset)
            curve_path.addEllipse(QRectF(point.x() - 16.0, point.y() - 16.0, 32.0, 32.0))
            has_curve = True
        if has_curve:
            stroker = QPainterPathStroker()
            stroker.setWidth(26.0)
            path = path.united(stroker.createStroke(curve_path)).united(curve_path)
        return path

    def set_pitch_curve_edit_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self.pitch_curve_edit_mode == enabled:
            return
        self.prepareGeometryChange()
        self.pitch_curve_edit_mode = enabled
        if not enabled:
            self._selected_pitch_control_index = None
            self._hover_pitch_control_index = None
            self._reset_pitch_control_hover_edge()
        self.update()

    def set_cut_merge_marked(self, edge: str | None) -> None:
        self._cut_merge_marked_edge = edge if edge in {"left", "right"} else None
        self.update()

    def clear_runtime_caches(self) -> None:
        self.base_audio_cache = None
        self.base_audio_sample_rate = None
        self.base_audio_level_dbfs = None
        self.waveform_envelope = None
        self.render_cache_audio = None
        self.render_cache_sample_rate = None
        self.render_cache_parameters = None
        self.render_cache_level_dbfs = None
        self.pitch_contour = None
        self.pitch_curve_center_midi = None
        self._active_render_key = None
        self.is_rendering = False

    def edit_state(self) -> dict[str, object]:
        return {
            "x": self.scenePos().x(),
            "y": self.scenePos().y(),
            "width": self.rect().width(),
            "height": self.rect().height(),
            "target_midi_note": self.target_midi_note,
            "target_duration": self.target_duration,
            "gain_db": self.gain_db,
            "pitch_flatten_amount": self.pitch_flatten_amount,
            "formant_shift": self.formant_shift,
            "protect_transients": self.protect_transients,
            "pitch_control_points": self.pitch_control_points_payload(),
            "pitch_vibrato_regions": self.pitch_vibrato_regions_payload(),
            "pitch_shape_regions": self.pitch_shape_regions_payload(),
            "is_track_reference": self.is_track_reference,
            "reference_editable": self.reference_editable,
        }

    def apply_edit_state(self, state: dict[str, object]) -> None:
        self._applying_saved_state = True
        try:
            width = max(self.MIN_WIDTH, float(state.get("width", self.rect().width())))
            height = max(1.0, float(state.get("height", self.rect().height())))
            self.setRect(0, 0, width, height)
            self.target_midi_note = (
                None
                if state.get("target_midi_note") is None
                else int(state["target_midi_note"])
            )
            self.target_duration = max(
                0.001,
                float(state.get("target_duration", self.target_duration)),
            )
            self.set_gain_db(float(state.get("gain_db", self.gain_db)))
            self.set_pitch_flatten_amount(
                float(state.get("pitch_flatten_amount", self.pitch_flatten_amount))
            )
            self.set_formant_shift(float(state.get("formant_shift", self.formant_shift)))
            self.set_transient_protection(
                bool(state.get("protect_transients", self.protect_transients))
            )
            self.set_pitch_control_points(
                state.get("pitch_control_points", self.pitch_control_points)
            )
            self.set_pitch_vibrato_regions(
                state.get("pitch_vibrato_regions", self.pitch_vibrato_regions)
            )
            self.set_pitch_shape_regions(
                state.get("pitch_shape_regions", self.pitch_shape_regions)
            )
            self.set_track_reference(
                bool(state.get("is_track_reference", self.is_track_reference)),
                bool(state.get("reference_editable", self.reference_editable)),
            )
            self.setPos(float(state.get("x", self.scenePos().x())), float(state.get("y", self.scenePos().y())))
            self._edit_model.clip.timeline_start = max(
                0.0,
                self.scenePos().x() / max(1.0, self._pixels_per_second()),
            )
            self.label = self._make_label()
            self.update()
        finally:
            self._applying_saved_state = False

    def has_current_render_cache(self) -> bool:
        if self.render_cache_audio is None or self.render_cache_sample_rate is None:
            return False
        if self.render_cache_parameters is None:
            return False
        current = self.current_render_parameters()
        return (
            self.render_cache_parameters.target_midi_note == current.target_midi_note
            and abs(self.render_cache_parameters.target_duration - current.target_duration) < 1e-4
            and abs(self.render_cache_parameters.gain_db - current.gain_db) < 1e-4
            and abs(self.render_cache_parameters.pitch_flatten_amount - current.pitch_flatten_amount) < 1e-4
            and abs(self.render_cache_parameters.formant_shift - current.formant_shift) < 1e-4
            and self.render_cache_parameters.protect_transients == current.protect_transients
            and self.render_cache_parameters.pitch_control_points == current.pitch_control_points
            and self.render_cache_parameters.pitch_vibrato_regions == current.pitch_vibrato_regions
        )

    def preview_audio(self):
        if self.is_missing_source:
            return None, None
        if self.has_current_render_cache():
            return self.render_cache_audio, self.render_cache_sample_rate
        if self.base_audio_cache is not None and self.base_audio_sample_rate is not None:
            if self._can_preview_from_base_cache():
                return self._base_preview_with_gain(), self.base_audio_sample_rate
        return None, None

    def measured_level_dbfs(
        self,
        gain_db_override: float | None = None,
    ) -> tuple[float, float] | None:
        target_gain = self.gain_db if gain_db_override is None else float(gain_db_override)
        if (
            self.render_cache_audio is not None
            and self.render_cache_parameters is not None
            and self.render_cache_level_dbfs is not None
        ):
            gain_delta = target_gain - float(self.render_cache_parameters.gain_db)
            return (
                float(self.render_cache_level_dbfs[0]) + gain_delta,
                min(0.0, float(self.render_cache_level_dbfs[1]) + gain_delta),
            )
        if self.base_audio_cache is not None and self.base_audio_level_dbfs is not None:
            return (
                float(self.base_audio_level_dbfs[0]) + target_gain,
                min(0.0, float(self.base_audio_level_dbfs[1]) + target_gain),
            )
        return None

    def hoverMoveEvent(self, event) -> None:  # type: ignore[override]
        mode = self._effective_tool_mode(event.modifiers())
        pitch_tool = self._effective_pitch_curve_tool_mode(event.modifiers())
        if self._is_fully_locked():
            self._reset_gain_hover()
            self._reset_pitch_control_hover()
            self._set_pitch_curve_hovered(False)
            self.unsetCursor()
        elif self.pitch_curve_edit_mode and pitch_tool == "curve_point":
            self._reset_gain_hover()
            hit_index = self._hit_pitch_control_point(event.pos())
            near_curve = hit_index is None and self._position_near_pitch_curve(event.pos())
            self._update_pitch_control_hover(hit_index)
            self._set_pitch_curve_hovered(near_curve)
            self._hover_pitch_curve_range = None
            self._reset_pitch_control_hover_edge()
            self.setCursor(_workspace_cursor("pitch_curve_point"))
        elif self.pitch_curve_edit_mode and pitch_tool == "curve_select":
            self._reset_gain_hover()
            hit_index = self._hit_pitch_control_point(event.pos())
            self._update_pitch_control_hover(hit_index)
            self._set_pitch_curve_hovered(False)
            self._hover_pitch_curve_range = None
            edge = self._pitch_control_handle_edge_at(hit_index, event.pos())
            if edge is not None and self._pitch_control_hover_is_armed(edge, event.pos()):
                self.setCursor(_workspace_cursor("horizontal_resize"))
            elif hit_index is not None:
                self.setCursor(_workspace_cursor("pitch_curve_select"))
            else:
                self._reset_pitch_control_hover_edge()
                self.setCursor(_workspace_cursor("pitch_curve_select"))
        elif self.pitch_curve_edit_mode and pitch_tool == "curve_vibrato":
            self._reset_gain_hover()
            self._reset_pitch_control_hover()
            near_curve = self._position_near_pitch_curve(
                event.pos(),
                max_distance=self.PITCH_VIBRATO_HIT_RADIUS,
            )
            self._set_pitch_curve_hovered(near_curve)
            self._hover_pitch_curve_range = None
            self._reset_pitch_control_hover_edge()
            self.setCursor(self._pitch_vibrato_cursor())
        elif mode == "cut_merge":
            self._reset_gain_hover()
            self._reset_pitch_control_hover()
            self._set_pitch_curve_hovered(False)
            self.setCursor(_workspace_cursor("cut_merge"))
        elif mode == "scissors":
            self._reset_gain_hover()
            self._reset_pitch_control_hover()
            self._set_pitch_curve_hovered(False)
            self.setCursor(_workspace_cursor("scissors"))
        elif mode == "amplitude" and self._can_edit_gain():
            self._reset_gain_hover()
            self._reset_pitch_control_hover()
            self._set_pitch_curve_hovered(False)
            self.setCursor(_workspace_cursor("gain_vertical"))
        elif mode == "select":
            self._reset_pitch_control_hover()
            self._set_pitch_curve_hovered(False)
            resize_edge = self._resize_edge_at(event.pos())
            gain_edge = self._gain_edge_at(event.pos())
            if resize_edge is not None and self._can_edit_pitch_and_duration():
                self._reset_gain_hover()
                self.setCursor(_workspace_cursor("horizontal_resize"))
            elif gain_edge is not None and self._gain_hover_is_armed(gain_edge, event.pos()):
                self.setCursor(_workspace_cursor("gain_vertical"))
            else:
                self.setCursor(_workspace_cursor("move"))
        elif not self._can_edit_pitch_and_duration():
            self._reset_gain_hover()
            self._reset_pitch_control_hover()
            self._set_pitch_curve_hovered(False)
            self.setCursor(_workspace_cursor("move"))
        elif self._resize_edge_at(event.pos()) is not None:
            self._reset_gain_hover()
            self._reset_pitch_control_hover()
            self._set_pitch_curve_hovered(False)
            self.setCursor(_workspace_cursor("horizontal_resize"))
        else:
            self._reset_gain_hover()
            self._reset_pitch_control_hover()
            self._set_pitch_curve_hovered(False)
            self.setCursor(_workspace_cursor("move"))
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # type: ignore[override]
        self._reset_gain_hover()
        self._reset_pitch_control_hover()
        self._hover_pitch_curve_range = None
        self._set_pitch_curve_hovered(False)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton and not self._is_fully_locked():
            if self._handle_pitch_control_press(event):
                event.accept()
                return
            pitch_mode = self._effective_pitch_curve_tool_mode(event.modifiers())
            if pitch_mode == "curve_vibrato":
                scene = self.scene()
                if scene is not None:
                    for view in scene.views():
                        if hasattr(view, "_begin_pitch_vibrato_drag"):
                            view_position = view.mapFromScene(event.scenePos())
                            if view._begin_pitch_vibrato_drag(
                                QPointF(view_position),
                                event.scenePos(),
                                target_hint=self,
                            ):
                                event.accept()
                                return
            if pitch_mode != "none":
                event.accept()
                return
            self._capture_edit_start_state()
            self._capture_selected_peer_edit_start_states()
            mode = self._effective_tool_mode(event.modifiers())
            if mode in {"scissors", "cut_merge"}:
                scene = self.scene()
                if scene is not None:
                    for view in scene.views():
                        if (
                            hasattr(view, "handle_item_tool_press")
                            and view.handle_item_tool_press(self, event.pos())
                        ):
                            event.accept()
                            return
            if mode == "amplitude" and self._can_edit_gain():
                self._begin_gain_drag(event.scenePos().y())
                event.accept()
                return
            resize_edge = self._resize_edge_at(event.pos())
            if resize_edge is not None and self._can_edit_pitch_and_duration():
                self._resize_edge = resize_edge
                self._resize_start_scene_x = event.scenePos().x()
                self._resize_start_width = self.rect().width()
                self._resize_start_pos = QPointF(self.pos())
                self._hide_resize_drag_cursor()
                event.accept()
                return
            if mode == "select" and self._gain_edge_is_armed(event.pos()):
                self._begin_gain_drag(event.scenePos().y())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        scene = self.scene()
        if scene is not None:
            for view in scene.views():
                if getattr(view, "_pitch_vibrato_drag", None) is not None:
                    view_position = view.mapFromScene(event.scenePos())
                    view._update_pitch_vibrato_drag(QPointF(view_position))
                    event.accept()
                    return
        if self._pitch_control_drag_index is not None:
            self._update_pitch_control_drag(event.pos())
            event.accept()
            return
        if self._effective_pitch_curve_tool_mode(event.modifiers()) != "none":
            event.accept()
            return
        if self._gain_dragging and self._can_edit_gain():
            delta_y = self._gain_start_scene_y - event.scenePos().y()
            self._update_group_gain_drag(delta_y * 0.16)
            event.accept()
            return
        if (
            self._resize_edge is not None
            and self._can_edit_pitch_and_duration()
        ):
            self._update_resize_preview(event.scenePos().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        scene = self.scene()
        if scene is not None:
            for view in scene.views():
                if getattr(view, "_pitch_vibrato_drag", None) is not None:
                    view_position = view.mapFromScene(event.scenePos())
                    view._update_pitch_vibrato_drag(QPointF(view_position))
                    view._finish_pitch_vibrato_drag(commit=True)
                    event.accept()
                    return
        if self._pitch_control_drag_index is not None:
            self.finish_pitch_control_drag()
            event.accept()
            return
        if self._gain_dragging:
            self._gain_dragging = False
            self._restore_gain_drag_cursor()
            self._finish_group_gain_drag()
            event.accept()
            return
        if self._resize_edge is not None:
            self._resize_edge = None
            self._restore_resize_drag_cursor()
            self._request_render_if_edit_changed()
            event.accept()
            return
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton and not self._is_fully_locked():
            self._request_render_for_edit_peers_if_changed()

    def finish_pitch_control_drag(self) -> bool:
        if self._pitch_control_drag_index is None:
            return False
        before = self._pitch_control_drag_start_state or self.edit_state()
        self._pitch_control_drag_index = None
        self._pitch_control_drag_start_state = None
        self._pitch_control_drag_start_points = None
        self._pitch_control_drag_start_selection.clear()
        self._pitch_control_drag_start_local = None
        self._pitch_control_drag_axis = "xy"
        self._sync_pitch_curve_ranges_from_selected_points()
        self._notify_pitch_curve_selection_changed()
        self._notify_parameter_change(before, self.edit_state())
        return True

    def _handle_pitch_control_press(self, event) -> bool:
        return self.handle_pitch_curve_tool_press(event.pos(), event.modifiers())

    def _pitch_vibrato_cursor(self) -> QCursor:
        scene = self.scene()
        if scene is not None:
            for view in scene.views():
                cursor_name = getattr(view, "_pitch_vibrato_cursor_name", None)
                if callable(cursor_name):
                    return _workspace_cursor(str(cursor_name()))
        return _workspace_cursor("pitch_vibrato_sine")

    def handle_pitch_curve_tool_press(self, position: QPointF, modifiers) -> bool:
        if not self.pitch_curve_edit_mode or not self._can_edit_pitch_and_duration():
            return False
        mode = self._effective_pitch_curve_tool_mode(modifiers)
        hit_index = self._hit_pitch_control_point(position)
        if mode == "curve_select" and hit_index is not None:
            self._select_pitch_control_point(
                hit_index,
                additive=bool(modifiers & (Qt.ControlModifier | Qt.ShiftModifier)),
            )
            self._pitch_control_drag_index = hit_index
            self._pitch_control_drag_start_state = self.edit_state()
            self._pitch_control_drag_start_points = list(self.pitch_control_points)
            self._pitch_control_drag_start_selection = set(
                self._selected_pitch_control_indices
            ) or {hit_index}
            self._pitch_control_drag_start_local = QPointF(position)
            edge = self._pitch_control_handle_edge_at(hit_index, position)
            self._pitch_control_drag_axis = (
                "x"
                if edge is not None
                and self._pitch_control_hover_edge == edge
                and self._pitch_control_hover_armed
                else "y"
            )
            return True
        if mode == "curve_point" and hit_index is not None:
            # B owns point creation/deletion only. Existing points are left
            # untouched so this tool cannot alter V's selection state.
            return True
        if mode == "curve_point" and self._position_near_pitch_curve(position):
            return self.add_pitch_control_point_at_position(position)
        return False

    def add_pitch_control_point_at_position(
        self,
        position: QPointF,
        max_distance: float = 12.0,
    ) -> bool:
        if not self.pitch_curve_edit_mode or not self._can_edit_pitch_and_duration():
            return False
        if self._pitch_curve_distance_to_position(position) > max_distance:
            return False
        before = self.edit_state()
        ratio = self._nearest_pitch_curve_ratio(position)
        offset = self._pitch_control_offset_at_ratio(ratio)
        self._remove_pitch_shape_regions_at_ratio(ratio)
        self.clear_pitch_control_selection()
        self._append_pitch_control_point(ratio, offset)
        self._notify_parameter_change(before, self.edit_state())
        return True

    def _remove_pitch_shape_regions_at_ratio(self, ratio: float) -> None:
        ratio = max(0.0, min(1.0, float(ratio)))
        kept: list[dict[str, float | str]] = []
        for region in self.pitch_shape_regions_payload():
            try:
                start = float(region["start"])
                end = float(region["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if start + 1e-4 < ratio < end - 1e-4:
                continue
            kept.append(region)
        self.set_pitch_shape_regions(kept)

    def _update_pitch_control_drag(self, position: QPointF) -> None:
        if self._pitch_control_drag_index is None:
            return
        points = list(self.pitch_control_points)
        if not (0 <= self._pitch_control_drag_index < len(points)):
            return
        ratio, offset = self._pitch_control_point_from_local(position)
        current_ratio, current_offset = points[self._pitch_control_drag_index]
        if self._pitch_control_drag_axis == "x":
            start_points = self._pitch_control_drag_start_points or points
            start_selection = {
                index
                for index in self._pitch_control_drag_start_selection
                if 0 <= index < len(start_points)
            } or {self._pitch_control_drag_index}
            anchor_index = self._pitch_control_drag_index
            if anchor_index >= len(start_points):
                return
            requested_delta = ratio - start_points[anchor_index][0]
            minimum_delta = -min(start_points[index][0] for index in start_selection)
            maximum_delta = 1.0 - max(
                start_points[index][0] for index in start_selection
            )
            delta = max(minimum_delta, min(maximum_delta, requested_delta))
            records = [
                (
                    point[0] + delta if index in start_selection else point[0],
                    point[1],
                    index in start_selection,
                    index == anchor_index,
                )
                for index, point in enumerate(start_points)
            ]
            records.sort(key=lambda record: record[0])
            points = [(record[0], record[1]) for record in records]
            self._selected_pitch_control_indices = {
                index for index, record in enumerate(records) if record[2]
            }
            self._pitch_control_drag_index = next(
                index for index, record in enumerate(records) if record[3]
            )
            ratio = points[self._pitch_control_drag_index][0]
        elif self._pitch_control_drag_axis == "y":
            ratio = current_ratio
            start_points = self._pitch_control_drag_start_points or points
            if self._pitch_control_drag_index < len(start_points):
                delta = offset - start_points[self._pitch_control_drag_index][1]
                selected = self._selected_pitch_control_indices or {
                    self._pitch_control_drag_index
                }
                for index in selected:
                    if 0 <= index < len(points) and index < len(start_points):
                        start_ratio, start_offset = start_points[index]
                        points[index] = (
                            start_ratio,
                            max(-24.0, min(24.0, start_offset + delta)),
                        )
        points.sort(key=lambda point: point[0])
        self.pitch_control_points = points
        if self._pitch_control_drag_axis != "x":
            self._pitch_control_drag_index = min(
                range(len(points)),
                key=lambda index: abs(points[index][0] - ratio),
            )
        self._selected_pitch_control_index = self._pitch_control_drag_index
        self._sync_pitch_curve_ranges_from_selected_points()
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            scene = self.scene()
            if scene is not None:
                for view in scene.views():
                    if (
                        hasattr(view, "handle_item_double_click")
                        and view.handle_item_double_click(self, event.pos())
                    ):
                        event.accept()
                        return
                    if hasattr(view, "request_preview_for_item"):
                        view.request_preview_for_item(self)
                        event.accept()
                        return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event) -> None:  # type: ignore[override]
        pitch_tool = self._effective_pitch_curve_tool_mode(event.modifiers())
        if pitch_tool != "none":
            if pitch_tool == "curve_point":
                self._delete_pitch_control_point_at(event.pos())
            event.accept()
            return
        normal_tool = self._effective_tool_mode(event.modifiers())
        if normal_tool in {"scissors", "cut_merge"}:
            scene = self.scene()
            if scene is not None:
                scene_position = self.mapToScene(event.pos())
                for view in scene.views():
                    if hasattr(view, "cancel_cut_at_scene_position"):
                        view.cancel_cut_at_scene_position(scene_position)
                        break
            event.accept()
            return
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

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if event.modifiers() & Qt.ControlModifier:
            event.ignore()
            return
        if self._view_tool_mode() == "formant" and self._can_edit_voice_effects():
            delta = event.delta() if hasattr(event, "delta") else event.angleDelta().y()
            if delta:
                before = self.edit_state()
                self.set_formant_shift(self.formant_shift + (float(delta) / 120.0) * 0.5)
                self._notify_parameter_change(before, self.edit_state())
                event.accept()
                return
        super().wheelEvent(event)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # type: ignore[override]
        rect = self.rect()
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        if self.pitch_curve_edit_mode:
            painter.setPen(QPen(QColor(233, 245, 255, 210), 1.5))
            painter.setBrush(QColor(244, 249, 255, 52))
        else:
            painter.setPen(self.pen())
            painter.setBrush(self.brush())
        painter.drawRoundedRect(rect, 4, 4)

        if self.pitch_curve_edit_mode:
            self._paint_waveform(painter)

        if self.isSelected() and not self._is_fully_locked():
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 4, 4)

        if self.is_rendering:
            painter.setPen(QPen(QColor("#ffcc66"), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 4, 4)

        if self._cut_merge_marked_edge is not None:
            painter.setPen(QPen(QColor("#8be9ff"), 3))
            x = rect.left() if self._cut_merge_marked_edge == "left" else rect.right()
            painter.drawLine(QPointF(x, rect.top() + 2), QPointF(x, rect.bottom() - 2))

        label_rect = rect.adjusted(7, 0, -7, 0)
        if self._should_show_gain_badge():
            label_rect = label_rect.adjusted(0, 0, -122, 0)
        if not self.pitch_curve_edit_mode and rect.width() >= 22.0:
            metrics = QFontMetrics(painter.font())
            label = self.label if rect.width() >= 72.0 else self._compact_label()
            label = metrics.elidedText(label, Qt.ElideRight, max(8, int(label_rect.width())))
            painter.setPen(QColor("#10141a"))
            painter.drawText(
                label_rect,
                Qt.AlignVCenter | Qt.AlignLeft,
                label,
            )

        if self.is_missing_source:
            painter.fillRect(rect, QColor(130, 0, 24, 95))
            painter.setPen(QColor("#ffd4da"))
            painter.drawText(
                rect.adjusted(7, 0, -7, 0),
                Qt.AlignVCenter | Qt.AlignRight,
                "MISSING",
            )
        elif self.is_track_reference and not self.reference_editable:
            painter.fillRect(rect, QColor(0, 0, 0, 110))
            painter.setPen(QColor("#d6dde6"))
            painter.drawText(
                rect.adjusted(7, 0, -7, 0),
                Qt.AlignVCenter | Qt.AlignRight,
                "REF",
            )
        elif self.is_locked:
            painter.fillRect(rect, QColor(0, 0, 0, 150))
            painter.setPen(QColor("#c5cfdb"))
            painter.drawText(
                rect.adjusted(7, 0, -7, 0),
                Qt.AlignVCenter | Qt.AlignRight,
                "BGM" if self._is_master_bgm() else "LOCK",
            )
        elif self.is_rendering:
            painter.fillRect(rect, QColor(0, 0, 0, 70))
            painter.setPen(QColor("#ffe1a3"))
            painter.drawText(
                rect.adjusted(7, 0, -7, 0),
                Qt.AlignVCenter | Qt.AlignRight,
                "LOADING",
            )
        scene = self.scene()
        overlay_enabled = (
            bool(scene.property("pitch_curve_overlay_enabled"))
            if scene is not None
            else False
        )
        if (
            self.pitch_curve_edit_mode
            and not overlay_enabled
            and not self.is_missing_source
            and not (self.is_track_reference and not self.reference_editable)
        ):
            self._paint_pitch_contour(painter)
        self._paint_gain_badge(painter)
        painter.restore()

    def itemChange(self, change, value):  # type: ignore[override]
        if self._applying_saved_state:
            return super().itemChange(change, value)
        if (
            change == QGraphicsItem.ItemPositionChange
            and not self._is_fully_locked()
            and isinstance(value, QPointF)
        ):
            x_value = value.x()
            y_value = value.y()
            if self.snap_to_grid_enabled:
                x_value = round(value.x() / self.snap_grid_size) * self.snap_grid_size
            if self._is_master_bgm() and self._pitch_anchor_y is not None:
                return QPointF(x_value, self._pitch_anchor_y)
            if self._can_edit_pitch_and_duration():
                y_value = self._snap_y_to_scale(y_value)
            if self.snap_to_grid_enabled:
                return QPointF(x_value, y_value)
            if y_value != value.y():
                return QPointF(x_value, y_value)
        if (
            change == QGraphicsItem.ItemPositionHasChanged
            and self._pitch_anchor_y is not None
            and self._can_edit_pitch_and_duration()
            and isinstance(value, QPointF)
        ):
            self._update_target_midi_from_y(value.y())
        if change == QGraphicsItem.ItemPositionHasChanged:
            if isinstance(value, QPointF):
                self._edit_model.clip.timeline_start = max(
                    0.0,
                    value.x() / max(1.0, self._pixels_per_second()),
                )
            scene = self.scene()
            if scene is not None:
                overlay = scene.property("pitch_curve_overlay_item")
                if isinstance(overlay, QGraphicsItem):
                    overlay.update()
        if change == QGraphicsItem.ItemSelectedHasChanged and not bool(value):
            self.clear_pitch_control_selection()
        return super().itemChange(change, value)

    def _update_target_midi_from_y(self, y: float) -> None:
        if self.audio_slice.midi_note is None or self._pitch_anchor_y is None:
            return
        if not self._can_edit_pitch_and_duration():
            return
        delta_semitones = round((self._pitch_anchor_y - y) / self.PITCH_PIXELS_PER_SEMITONE)
        target = max(0, min(127, self.audio_slice.midi_note + delta_semitones))
        root_note, scale_type = self._scale_settings()
        if scale_type != "Chromatic":
            target = nearest_midi_in_scale(target, root_note, scale_type)
        self.target_midi_note = target
        self.label = self._make_label()
        self.update()

    def _snap_y_to_scale(self, y: float) -> float:
        if self.audio_slice.midi_note is None or self._pitch_anchor_y is None:
            return y
        root_note, scale_type = self._scale_settings()
        if scale_type == "Chromatic":
            return y
        delta_semitones = round((self._pitch_anchor_y - y) / self.PITCH_PIXELS_PER_SEMITONE)
        proposed_midi = max(0, min(127, self.audio_slice.midi_note + delta_semitones))
        snapped_midi = nearest_midi_in_scale(proposed_midi, root_note, scale_type)
        return self._pitch_anchor_y - (snapped_midi - self.audio_slice.midi_note) * self.PITCH_PIXELS_PER_SEMITONE

    def _scale_settings(self) -> tuple[str, str]:
        scene = self.scene()
        if scene is None:
            return "C", "Chromatic"
        root_note = normalize_root(str(scene.property("scale_root") or "C"))
        scale_type = normalize_scale_type(str(scene.property("scale_type") or "Chromatic"))
        return root_note, scale_type

    def _make_label(self) -> str:
        note = self.audio_slice.note_name
        if (
            self._can_edit_pitch_and_duration()
            and self.target_midi_note is not None
            and self.target_midi_note != self.audio_slice.midi_note
        ):
            from hakyking.models.audio_slice import NOTE_NAMES

            octave = self.target_midi_note // 12 - 1
            note = f"{NOTE_NAMES[self.target_midi_note % 12]}{octave}"
        extras: list[str] = []
        if self.pitch_flatten_amount >= 0.01:
            extras.append(f"Flt {self.pitch_flatten_amount * 100:.0f}%")
        if abs(self.formant_shift) >= 0.05:
            extras.append(f"Fmt {self.formant_shift:+.1f}")
        suffix = "" if not extras else "  " + " ".join(extras)
        prefix = "REF" if self.is_track_reference and not self.reference_editable else f"T{self.track_index + 1}"
        return f"{prefix}  {note}  {self.target_duration:.2f}s{suffix}"

    def _compact_label(self) -> str:
        note = self._note_name_for_midi(self.target_midi_note)
        if note == "N/A":
            note = self.audio_slice.note_name
        if note == "N/A":
            return f"T{self.track_index + 1}"
        return note

    def _resize_edge_at(self, position: QPointF) -> str | None:
        if not self._can_edit_pitch_and_duration():
            return None
        rect = self.rect()
        if abs(position.x() - rect.left()) <= self.RESIZE_MARGIN:
            return "left"
        if abs(position.x() - rect.right()) <= self.RESIZE_MARGIN:
            return "right"
        return None

    def _merge_edge_at(self, position: QPointF) -> str | None:
        if not self._can_edit_pitch_and_duration():
            return None
        rect = self.rect()
        margin = max(self.RESIZE_MARGIN + 3.0, 8.0)
        if abs(position.x() - rect.left()) <= margin:
            return "left"
        if abs(position.x() - rect.right()) <= margin:
            return "right"
        return None

    def _gain_edge_at(self, position: QPointF) -> str | None:
        if not self._can_edit_gain():
            return None
        rect = self.rect()
        if rect.width() <= self.RESIZE_MARGIN * 2.0:
            return None
        if (
            abs(position.x() - rect.left()) <= self.RESIZE_MARGIN
            or abs(position.x() - rect.right()) <= self.RESIZE_MARGIN
        ):
            return None
        if abs(position.y() - rect.top()) <= self.GAIN_EDGE_MARGIN:
            return "top"
        if abs(position.y() - rect.bottom()) <= self.GAIN_EDGE_MARGIN:
            return "bottom"
        return None

    def _gain_hover_is_armed(self, edge: str, position: QPointF) -> bool:
        now = time.monotonic()
        if (
            edge != self._gain_hover_edge
            or self._gain_hover_position is None
            or abs(position.x() - self._gain_hover_position.x()) > self.GAIN_HOVER_STABLE_PIXELS
            or abs(position.y() - self._gain_hover_position.y()) > self.GAIN_HOVER_STABLE_PIXELS
        ):
            self._gain_hover_edge = edge
            self._gain_hover_started_at = now
            self._gain_hover_armed = False
            self._gain_hover_position = QPointF(position)
            self._gain_hover_token += 1
            token = self._gain_hover_token
            QTimer.singleShot(
                int(self.GAIN_HOVER_DELAY_SECONDS * 1000),
                lambda token=token: self._arm_gain_hover_if_still(token),
            )
            return False
        return self._gain_hover_armed

    def _arm_gain_hover_if_still(self, token: int) -> None:
        if (
            token == self._gain_hover_token
            and self._gain_hover_edge is not None
            and time.monotonic() - self._gain_hover_started_at >= self.GAIN_HOVER_DELAY_SECONDS
        ):
            self._gain_hover_armed = True
            if self.isUnderMouse():
                self.setCursor(_workspace_cursor("gain_vertical"))

    def _gain_edge_is_armed(self, position: QPointF) -> bool:
        edge = self._gain_edge_at(position)
        return (
            edge is not None
            and edge == self._gain_hover_edge
            and self._gain_hover_armed
        )

    def _reset_gain_hover(self) -> None:
        self._gain_hover_edge = None
        self._gain_hover_started_at = 0.0
        self._gain_hover_armed = False
        self._gain_hover_position = None
        self._gain_hover_token += 1

    def _begin_gain_drag(self, scene_y: float) -> None:
        self._gain_dragging = True
        self._gain_start_scene_y = scene_y
        self._gain_start_db = self.gain_db
        self._capture_gain_peer_start_states()
        if not self._gain_drag_cursor_hidden:
            QApplication.setOverrideCursor(QCursor(Qt.BlankCursor))
            self._gain_drag_cursor_hidden = True

    def _restore_gain_drag_cursor(self) -> None:
        if self._gain_drag_cursor_hidden:
            QApplication.restoreOverrideCursor()
            self._gain_drag_cursor_hidden = False

    def _capture_gain_peer_start_states(self) -> None:
        scene = self.scene()
        if scene is None or not self.isSelected():
            peers = [self]
        else:
            peers = [
                item
                for item in scene.selectedItems()
                if isinstance(item, AudioSliceGraphicsItem)
                and not item._is_fully_locked()
                and item._can_edit_gain()
            ]
            if self not in peers:
                peers.append(self)
        self._gain_peer_start_states = {
            id(item): (item, item.edit_state(), item.gain_db)
            for item in peers
        }

    def _update_group_gain_drag(self, delta_db: float) -> None:
        if not self._gain_peer_start_states:
            self.set_gain_db(self._gain_start_db + delta_db)
            return
        for item, _before, start_db in self._gain_peer_start_states.values():
            item.set_gain_db(start_db + delta_db)

    def _finish_group_gain_drag(self) -> None:
        changes: list[tuple[AudioSliceGraphicsItem, dict[str, object], dict[str, object]]] = []
        for item, before, _start_db in self._gain_peer_start_states.values():
            after = item.edit_state()
            if before != after:
                changes.append((item, before, after))
        self._gain_peer_start_states = {}
        if not changes:
            return
        scene = self.scene()
        if scene is not None:
            for view in scene.views():
                if len(changes) > 1 and hasattr(view, "handle_items_edit_finished"):
                    view.handle_items_edit_finished(changes)
                    return
                if hasattr(view, "handle_item_parameter_change"):
                    item, before, after = changes[0]
                    view.handle_item_parameter_change(item, before, after)
                    return
        for item, _before, _after in changes:
            item._request_render()

    def _hide_resize_drag_cursor(self) -> None:
        if not self._resize_drag_cursor_hidden:
            QApplication.setOverrideCursor(QCursor(Qt.BlankCursor))
            self._resize_drag_cursor_hidden = True

    def _restore_resize_drag_cursor(self) -> None:
        if self._resize_drag_cursor_hidden:
            QApplication.restoreOverrideCursor()
            self._resize_drag_cursor_hidden = False

    def _update_resize_preview(self, scene_x: float) -> None:
        delta_x = scene_x - self._resize_start_scene_x
        if self._resize_edge == "right":
            new_width = max(self.MIN_WIDTH, self._resize_start_width + delta_x)
            self.setRect(0, 0, new_width, self.rect().height())
        elif self._resize_edge == "left":
            bounded_delta = min(delta_x, self._resize_start_width - self.MIN_WIDTH)
            new_width = max(self.MIN_WIDTH, self._resize_start_width - bounded_delta)
            self.setPos(self._resize_start_pos.x() + bounded_delta, self._resize_start_pos.y())
            self.setRect(0, 0, new_width, self.rect().height())
        self.target_duration = max(0.001, self.rect().width() / self._pixels_per_second())
        self.label = self._make_label()
        self.update()

    def _capture_edit_start_state(self) -> None:
        self._edit_start_pos = QPointF(self.pos())
        self._edit_start_width = self.rect().width()
        self._edit_start_target_midi = self.target_midi_note
        self._edit_start_target_duration = self.target_duration
        self._edit_start_state = self.edit_state()

    def _capture_selected_peer_edit_start_states(self) -> None:
        scene = self.scene()
        if scene is None or not self.isSelected():
            self._edit_peer_items = [self]
            return

        peers: list[AudioSliceGraphicsItem] = []
        for item in scene.selectedItems():
            if not isinstance(item, AudioSliceGraphicsItem):
                continue
            if item._is_fully_locked():
                continue
            peers.append(item)
        if self not in peers:
            peers.append(self)
        self._edit_peer_items = peers
        for item in peers:
            if item is not self:
                item._capture_edit_start_state()

    def _request_render_for_edit_peers_if_changed(self) -> None:
        if self._suppress_edit_notifications:
            return
        peers = self._edit_peer_items or [self]
        self._edit_peer_items = []
        seen: set[int] = set()
        changes: list[tuple[AudioSliceGraphicsItem, dict[str, object], dict[str, object]]] = []
        for item in peers:
            item_id = id(item)
            if item_id in seen:
                continue
            seen.add(item_id)
            change = item._edit_change_if_any()
            if change is not None:
                changes.append((item, change[0], change[1]))
        if not changes:
            return
        scene = self.scene()
        if scene is not None:
            for view in scene.views():
                if len(changes) > 1 and hasattr(view, "handle_items_edit_finished"):
                    view.handle_items_edit_finished(changes)
                    return
                if hasattr(view, "handle_item_edit_finished"):
                    item, before, after = changes[0]
                    view.handle_item_edit_finished(item, before, after)
                    return
        for item, _, _ in changes:
            item._request_render()

    def _request_render_if_edit_changed(self) -> None:
        if self._suppress_edit_notifications:
            return
        change = self._edit_change_if_any()
        if change is not None:
            self._notify_edit_finished(change[0], change[1])

    def _edit_change_if_any(self) -> tuple[dict[str, object], dict[str, object]] | None:
        x_changed = abs(self.pos().x() - self._edit_start_pos.x()) > 0.5
        width_changed = abs(self.rect().width() - self._edit_start_width) > 0.5
        y_changed = abs(self.pos().y() - self._edit_start_pos.y()) > 0.5
        midi_changed = self.target_midi_note != self._edit_start_target_midi
        duration_changed = abs(self.target_duration - self._edit_start_target_duration) > 1e-4
        gain_changed = (
            self._edit_start_state is not None
            and abs(self.gain_db - float(self._edit_start_state.get("gain_db", self.gain_db))) > 1e-3
        )
        if x_changed or width_changed or y_changed or midi_changed or duration_changed or gain_changed:
            audio_edit_changed = width_changed or y_changed or midi_changed or duration_changed
            if audio_edit_changed and not self._can_edit_pitch_and_duration():
                return None
            before = self._edit_start_state or self.edit_state()
            return before, self.edit_state()
        return None

    def _request_render(self) -> None:
        scene = self.scene()
        if scene is not None:
            for view in scene.views():
                if hasattr(view, "request_render_for_item"):
                    view.request_render_for_item(self)
                    return

    def _notify_edit_finished(
        self,
        before: dict[str, object],
        after: dict[str, object],
    ) -> None:
        if before == after:
            return
        scene = self.scene()
        if scene is not None:
            for view in scene.views():
                if hasattr(view, "handle_item_edit_finished"):
                    view.handle_item_edit_finished(self, before, after)
                    return
        self._request_render()

    def _notify_parameter_change(
        self,
        before: dict[str, object],
        after: dict[str, object],
    ) -> None:
        if before == after:
            return
        scene = self.scene()
        if scene is not None:
            for view in scene.views():
                if hasattr(view, "handle_item_parameter_change"):
                    view.handle_item_parameter_change(self, before, after)
                    return
        self._request_render()

    def info_lines(self) -> list[str]:
        parameters = self.current_render_parameters()
        source_name = Path(self.audio_slice.source_path).name or self.audio_slice.source_path
        target_note = self._note_name_for_midi(self.target_midi_note)
        source_note = self.audio_slice.note_name
        f0_text = "N/A" if self.audio_slice.f0_hz is None else f"{self.audio_slice.f0_hz:.2f} Hz"
        confidence_text = (
            "N/A"
            if self.audio_slice.pitch_confidence is None
            else f"{self.audio_slice.pitch_confidence:.2f}"
        )
        backend_text = self.audio_slice.analysis_backend or "legacy"
        level = self.measured_level_dbfs()
        level_text = (
            "尚未分析"
            if level is None
            else f"RMS {format_dbfs(level[0])} / Peak {format_dbfs(level[1])}"
        )
        status = "缺失源媒体" if self.is_missing_source else ("渲染中" if self.is_rendering else "就绪")
        return [
            "音符片段属性",
            f"状态: {status}",
            f"音轨: {self.track_index + 1} ({'BGM' if self._is_master_bgm() else 'Vocal'})",
            f"源文件: {source_name}",
            f"源区间: {self.audio_slice.start_time:.3f}s - {self.audio_slice.end_time:.3f}s",
            f"片段时长: 原始 {self.audio_slice.duration:.3f}s / 当前 {self.target_duration:.3f}s",
            f"音高: 原始 {source_note} -> 目标 {target_note}",
            f"F0: {f0_text}",
            f"Analysis: {backend_text} / {confidence_text}",
            f"变调: {parameters.n_steps:+.2f} 半音",
            f"增益: {format_gain(self.gain_db)}",
            f"当前电平: {level_text}",
            f"颤音展平: {self.pitch_flatten_amount * 100:.0f}%",
            f"共振峰偏移: {self.formant_shift:+.1f} 半音",
            f"瞬态保护: {'开启' if self.protect_transients else '关闭'}",
        ]

    def _paint_gain_badge(self, painter: QPainter) -> None:
        if not self._should_show_gain_badge():
            return
        badge = self._gain_badge_rect()
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        font = QFont("Segoe UI", 9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.setBrush(QBrush(QColor(12, 14, 18, 238)))
        painter.drawRoundedRect(badge, 4, 4)
        painter.drawText(badge.adjusted(8, 0, -8, 0), Qt.AlignCenter, self._gain_badge_text())
        painter.restore()

    def _should_show_gain_badge(self) -> bool:
        return self._gain_dragging

    def _gain_badge_text(self) -> str:
        return format_gain(self.gain_db, compact=False).replace(" / ", " · ")

    def _gain_badge_rect(self) -> QRectF:
        rect = self.rect()
        font = QFont("Segoe UI", 9)
        font.setBold(True)
        metrics = QFontMetrics(font)
        text_width = metrics.horizontalAdvance(self._gain_badge_text())
        badge_width = max(132.0, float(text_width + 20))
        badge_height = 24.0
        x = rect.center().x() - badge_width / 2.0
        y = rect.top() - badge_height - 7.0
        return QRectF(x, y, badge_width, badge_height)

    def _can_preview_from_base_cache(self) -> bool:
        return not self.requires_rendered_audio()

    def _base_preview_with_gain(self):
        if abs(self.gain_db) <= 1e-4:
            return self.base_audio_cache
        return apply_gain(self.base_audio_cache, self.gain_db, soft_limit=True)

    @staticmethod
    def _note_name_for_midi(midi_note: int | None) -> str:
        if midi_note is None:
            return "N/A"
        from hakyking.models.audio_slice import NOTE_NAMES

        clipped = max(0, min(127, int(midi_note)))
        return f"{NOTE_NAMES[clipped % 12]}{clipped // 12 - 1}"

    def _is_master_bgm(self) -> bool:
        return self.track_type == "master_bgm"

    def _can_edit_pitch_and_duration(self) -> bool:
        if self.is_track_reference and not self.reference_editable:
            return False
        return self.track_type != "master_bgm" and not self.is_locked

    def _can_edit_gain(self) -> bool:
        if self.is_missing_source:
            return False
        if self.is_track_reference and not self.reference_editable:
            return False
        return not self.is_locked or self._is_master_bgm()

    def _can_edit_voice_effects(self) -> bool:
        if self.is_missing_source:
            return False
        if self.is_track_reference and not self.reference_editable:
            return False
        return self.track_type != "master_bgm" and not self.is_locked

    def _is_fully_locked(self) -> bool:
        if self.is_track_reference and not self.reference_editable:
            return True
        return self.is_locked and not self._is_master_bgm()

    def _paint_waveform(self, painter: QPainter) -> None:
        if self.waveform_envelope is None or len(self.waveform_envelope) == 0:
            return

        rect = self.rect().adjusted(6, 6, -6, -6)
        if rect.width() <= 2 or rect.height() <= 2:
            return

        painter.save()
        painter.setClipRect(rect)
        painter.setPen(QPen(QColor(255, 255, 255, 115), 1))

        center_y = rect.center().y()
        amplitude = rect.height() * 0.42
        count = len(self.waveform_envelope)
        if count == 1:
            x_values = [rect.center().x()]
        else:
            step = rect.width() / (count - 1)
            x_values = [rect.left() + index * step for index in range(count)]

        for x, (minimum, maximum) in zip(x_values, self.waveform_envelope):
            top = center_y - float(maximum) * amplitude
            bottom = center_y - float(minimum) * amplitude
            painter.drawLine(QPointF(x, top), QPointF(x, bottom))
        painter.restore()

    def _paint_pitch_contour(self, painter: QPainter) -> None:
        segments = self._pitch_curve_local_segments()
        if not segments:
            return

        rect = self._pitch_curve_rect()
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(QColor(255, 255, 255, 45), 1))
        painter.drawLine(
            QPointF(rect.left(), rect.center().y()),
            QPointF(rect.right(), rect.center().y()),
        )
        glow = self._pitch_curve_hovered
        paths: list[QPainterPath] = []
        for segment in segments:
            path = QPainterPath(segment[0])
            for point in segment[1:]:
                path.lineTo(point)
            paths.append(path)
        painter.setPen(QPen(QColor(10, 20, 26, 245), 5.6 if glow else 4.8))
        for path in paths:
            painter.drawPath(path)
        painter.setPen(QPen(QColor("#7cffff") if glow else QColor("#42f5ff"), 3.2 if glow else 2.2))
        for path in paths:
            painter.drawPath(path)
        self._paint_selected_pitch_curve_ranges(painter)
        self._paint_pitch_control_points(painter)
        painter.restore()

    def _paint_selected_pitch_curve_ranges(self, painter: QPainter) -> None:
        ranges = list(self._selected_pitch_curve_ranges)
        if self._hover_pitch_curve_range is not None:
            ranges.append(self._hover_pitch_curve_range)
        ranges = self._normalize_pitch_curve_ranges(ranges)
        if not ranges:
            return
        for start, end in ranges:
            path = self._pitch_curve_path_for_range(start, end)
            if path is None:
                continue
            painter.setPen(QPen(QColor(255, 243, 166, 110), 4.4))
            painter.drawPath(path)
            painter.setPen(QPen(QColor("#fff3a6"), 2.0))
            painter.drawPath(path)

    def _pitch_curve_path_for_range(self, start: float, end: float) -> QPainterPath | None:
        if end <= start:
            return None
        span = end - start
        steps = max(2, min(96, int(span * 160) + 2))
        path: QPainterPath | None = None
        for index in range(steps):
            t = 0.0 if steps <= 1 else index / (steps - 1)
            point = self._pitch_curve_point_at_ratio(start + span * t)
            if point is None:
                continue
            if path is None:
                path = QPainterPath(point)
            else:
                path.lineTo(point)
        return path

    def _paint_pitch_contour_connector(self, painter: QPainter, local_last_point: QPointF) -> None:
        if not self._pitch_edge_is_voiced(at_start=False):
            return
        next_item = self._next_contiguous_pitch_item()
        if next_item is None:
            return
        if not next_item._pitch_edge_is_voiced(at_start=True):
            return
        next_point = next_item._first_pitch_contour_scene_point()
        if next_point is None:
            return
        local_next = self.mapFromScene(next_point)
        if (
            local_next.x() <= local_last_point.x()
            or local_next.x() - local_last_point.x() > 16.0
            or abs(local_next.y() - local_last_point.y())
            > self.PITCH_CURVE_PIXELS_PER_SEMITONE * 1.5
        ):
            return
        path = QPainterPath(local_last_point)
        path.lineTo(local_next)
        painter.setPen(QPen(QColor(10, 20, 26, 210), 4.0))
        painter.drawPath(path)
        painter.setPen(QPen(QColor("#42f5ff"), 1.8))
        painter.drawPath(path)

    def _next_contiguous_pitch_item(self) -> "AudioSliceGraphicsItem | None":
        scene = self.scene()
        if scene is None:
            return None
        best: AudioSliceGraphicsItem | None = None
        best_score = 1e9
        right_edge = self.scenePos().x() + self.rect().width()
        for candidate in scene.items():
            if not isinstance(candidate, AudioSliceGraphicsItem) or candidate is self:
                continue
            if candidate.audio_slice.source_path != self.audio_slice.source_path:
                continue
            if candidate.track_index != self.track_index:
                continue
            visual_gap = candidate.scenePos().x() - right_edge
            if abs(visual_gap) > 2.0:
                continue
            source_delta = candidate.audio_slice.start_time - self.audio_slice.end_time
            if abs(source_delta) <= 0.002:
                score = abs(visual_gap) + abs(source_delta) * 1000.0
                if score < best_score:
                    best = candidate
                    best_score = score
        return best

    def _first_pitch_contour_scene_point(self) -> QPointF | None:
        segments = self._pitch_curve_local_segments()
        if not segments:
            return None
        return self.mapToScene(segments[0][0])

    def _pitch_edge_is_voiced(self, at_start: bool) -> bool:
        if self.pitch_contour is None or len(self.pitch_contour) == 0:
            return False
        value = self.pitch_contour[0] if at_start else self.pitch_contour[-1]
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def _pitch_curve_rect(self) -> QRectF:
        # Keep the X mapping exactly proportional to source time. Horizontal
        # padding would make the same contour change shape after a split.
        return self.rect().adjusted(0, 4, 0, -8)

    def _detected_pitch_center_midi(self) -> float | None:
        if self.pitch_contour is None:
            return None
        finite_values = [float(value) for value in self.pitch_contour if value == value]
        if not finite_values:
            return None
        return float(np.median(np.asarray(finite_values, dtype=np.float32)))

    def _pitch_curve_center_midi(self) -> float:
        if self.pitch_curve_center_midi is not None:
            return float(self.pitch_curve_center_midi)
        detected = self._detected_pitch_center_midi()
        if detected is not None:
            self.pitch_curve_center_midi = detected
            return detected
        if self.target_midi_note is not None:
            return float(self.target_midi_note)
        if self.audio_slice.midi_note is not None:
            return float(self.audio_slice.midi_note)
        return 60.0

    def _finite_pitch_samples(self) -> list[tuple[float, float]]:
        if self.pitch_contour is None or len(self.pitch_contour) == 0:
            return []
        count = len(self.pitch_contour)
        if count <= 1:
            return [(0.0, float(self.pitch_contour[0]))] if self.pitch_contour[0] == self.pitch_contour[0] else []
        samples: list[tuple[float, float]] = []
        for index, value in enumerate(self.pitch_contour):
            if value != value:
                continue
            samples.append((index / (count - 1), float(value)))
        return samples

    def _interpolated_contour_midi(self, ratio: float) -> float:
        ratio = max(0.0, min(1.0, float(ratio)))
        samples = self._finite_pitch_samples()
        if not samples:
            return self._pitch_curve_center_midi()
        if ratio <= samples[0][0]:
            return samples[0][1]
        if ratio >= samples[-1][0]:
            return samples[-1][1]
        for (left_x, left_value), (right_x, right_value) in zip(samples, samples[1:]):
            if left_x <= ratio <= right_x:
                if right_x <= left_x:
                    return left_value
                t = (ratio - left_x) / (right_x - left_x)
                return left_value + (right_value - left_value) * t
        return samples[-1][1]

    @staticmethod
    def _pitch_vibrato_wave_value(waveform: str, phase: float) -> float:
        phase %= 1.0
        if waveform == "triangle":
            return (2.0 / math.pi) * math.asin(math.sin(math.tau * phase))
        if waveform == "square":
            return 1.0 if phase < 0.5 else -1.0
        return math.sin(math.tau * phase)

    def _pitch_vibrato_offset_at_ratio(self, ratio: float) -> float:
        if not self.pitch_vibrato_regions:
            return 0.0
        ratio = max(0.0, min(1.0, float(ratio)))
        total = 0.0
        for region in self.pitch_vibrato_regions:
            try:
                start = float(region["start"])
                end = float(region["end"])
                cycles = float(region["cycles"])
                depth = float(region["depth"])
                phase = float(region.get("phase", 0.0))
                waveform = str(region["waveform"])
            except (KeyError, TypeError, ValueError):
                continue
            if end <= start or not (start - 1e-6 <= ratio <= end + 1e-6):
                continue
            t = (ratio - start) / max(1e-6, end - start)
            total += self._pitch_vibrato_wave_value(waveform, cycles * t + phase) * depth
        return max(-24.0, min(24.0, total))

    def _pitch_shape_for_interval(
        self,
        left_x: float,
        right_x: float,
        ratio: float,
    ) -> str | None:
        if right_x <= left_x:
            return None
        for region in self.pitch_shape_regions:
            try:
                start = float(region["start"])
                end = float(region["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                abs(start - left_x) <= 0.002
                and abs(end - right_x) <= 0.002
                and start - 1e-6 <= ratio <= end + 1e-6
            ):
                return str(region.get("shape", "linear"))
        return None

    def _pitch_control_offset_at_ratio(
        self,
        ratio: float,
        include_vibrato: bool = True,
    ) -> float:
        ratio = max(0.0, min(1.0, float(ratio)))
        vibrato_offset = (
            self._pitch_vibrato_offset_at_ratio(ratio)
            if include_vibrato
            else 0.0
        )
        points = self.pitch_control_points
        if not points:
            return vibrato_offset
        segment_bounds = self._raw_pitch_segment_bounds_for_ratio(ratio)
        if segment_bounds is None:
            segment_bounds = (0.0, 1.0)
        start, end = segment_bounds
        anchors: list[tuple[float, float]] = [(start, 0.0), (end, 0.0)]
        anchors.extend(
            (float(x_value), float(offset))
            for x_value, offset in points
            if start - 1e-5 <= float(x_value) <= end + 1e-5
        )
        anchors.sort(key=lambda point: point[0])
        collapsed: list[tuple[float, float]] = []
        for x_value, offset in anchors:
            if collapsed and abs(x_value - collapsed[-1][0]) <= 1e-5:
                collapsed[-1] = (collapsed[-1][0], offset)
            else:
                collapsed.append((x_value, offset))
        if ratio <= collapsed[0][0]:
            return collapsed[0][1] + vibrato_offset
        if ratio >= collapsed[-1][0]:
            return collapsed[-1][1] + vibrato_offset
        for (left_x, left_offset), (right_x, right_offset) in zip(collapsed, collapsed[1:]):
            if left_x <= ratio <= right_x:
                if right_x <= left_x:
                    return left_offset + vibrato_offset
                t = (ratio - left_x) / (right_x - left_x)
                shape = self._pitch_shape_for_interval(left_x, right_x, ratio)
                t = self._pitch_curve_shape_factor(shape or "smooth", t)
                manual = left_offset + (right_offset - left_offset) * t
                return manual + vibrato_offset
        return vibrato_offset

    def _pitch_curve_local_segments(self) -> list[list[QPointF]]:
        if (
            (self.pitch_contour is None or len(self.pitch_contour) == 0)
            and not self.pitch_control_points
            and not self.pitch_vibrato_regions
        ):
            return []
        rect = self._pitch_curve_rect()
        if rect.width() <= 2 or rect.height() <= 2:
            return []
        center_midi = self._pitch_curve_center_midi()
        segments: list[list[QPointF]] = []
        for start, end in self._raw_pitch_segment_bounds():
            current: list[QPointF] = []
            for ratio in self._pitch_curve_sample_ratios(start, end, rect.width()):
                midi_value = self._interpolated_contour_midi(ratio)
                if not math.isfinite(midi_value):
                    if current:
                        segments.append(current)
                        current = []
                    continue
                offset = self._pitch_control_offset_at_ratio(ratio)
                x = rect.left() + rect.width() * ratio
                y = (
                    rect.center().y()
                    - (midi_value - center_midi + offset)
                    * self.PITCH_CURVE_PIXELS_PER_SEMITONE
                )
                current.append(QPointF(x, y))
            if current:
                segments.append(current)
        return segments

    def _pitch_curve_local_points(self) -> list[QPointF]:
        return [
            point
            for segment in self._pitch_curve_local_segments()
            for point in segment
        ]

    def _pitch_curve_point_at_ratio(self, ratio: float) -> QPointF | None:
        rect = self._pitch_curve_rect()
        if rect.width() <= 0:
            return None
        ratio = max(0.0, min(1.0, float(ratio)))
        center_midi = self._pitch_curve_center_midi()
        midi_value = self._interpolated_contour_midi(ratio)
        offset = self._pitch_control_offset_at_ratio(ratio)
        x = rect.left() + rect.width() * ratio
        y = rect.center().y() - (midi_value - center_midi + offset) * self.PITCH_CURVE_PIXELS_PER_SEMITONE
        return QPointF(x, y)

    def _pitch_control_point_to_local(self, ratio: float, offset: float) -> QPointF:
        rect = self._pitch_curve_rect()
        clipped_ratio = max(0.0, min(1.0, float(ratio)))
        x = rect.left() + rect.width() * clipped_ratio
        baseline = self._interpolated_contour_midi(clipped_ratio)
        center = self._pitch_curve_center_midi()
        y = rect.center().y() - (baseline - center + float(offset)) * self.PITCH_CURVE_PIXELS_PER_SEMITONE
        return QPointF(x, y)

    def _pitch_control_point_from_local(self, position: QPointF) -> tuple[float, float]:
        rect = self._pitch_curve_rect()
        ratio = 0.0 if rect.width() <= 0 else (position.x() - rect.left()) / rect.width()
        ratio = max(0.0, min(1.0, ratio))
        center = self._pitch_curve_center_midi()
        baseline = self._interpolated_contour_midi(ratio)
        visual_semitones = (rect.center().y() - position.y()) / self.PITCH_CURVE_PIXELS_PER_SEMITONE
        offset = visual_semitones - (baseline - center)
        return ratio, max(-24.0, min(24.0, offset))

    def _hit_pitch_control_point(self, position: QPointF) -> int | None:
        if not self.pitch_control_points:
            return None
        position_view = self._local_point_to_view(position)
        best_index: int | None = None
        best_distance = self.PITCH_CONTROL_HIT_RADIUS * self.PITCH_CONTROL_HIT_RADIUS
        for index, (ratio, offset) in enumerate(self.pitch_control_points):
            point_view = self._local_point_to_view(
                self._pitch_control_point_to_local(ratio, offset)
            )
            distance = (
                (point_view.x() - position_view.x()) ** 2
                + (point_view.y() - position_view.y()) ** 2
            )
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        return best_index

    def _append_pitch_control_point(self, ratio: float, offset: float) -> int:
        points = self.pitch_control_points_payload()
        points.append({"x": ratio, "offset": offset})
        self.set_pitch_control_points(points)
        if not self.pitch_control_points:
            return 0
        return min(
            range(len(self.pitch_control_points)),
            key=lambda index: abs(self.pitch_control_points[index][0] - ratio),
        )

    def _ensure_pitch_control_point_at_ratio(self, ratio: float) -> bool:
        ratio = max(0.0, min(1.0, float(ratio)))
        for existing_ratio, _offset in self.pitch_control_points:
            if abs(existing_ratio - ratio) <= 1e-4:
                return False
        offset = self._pitch_control_offset_at_ratio(ratio)
        points = self.pitch_control_points_payload()
        points.append({"x": ratio, "offset": offset})
        self.set_pitch_control_points(points)
        return True

    def _ensure_pitch_curve_range_boundary_points(
        self,
        selected_range: tuple[float, float],
    ) -> tuple[float, float]:
        start, end = selected_range
        changed = False
        changed = self._ensure_pitch_control_point_at_ratio(start) or changed
        changed = self._ensure_pitch_control_point_at_ratio(end) or changed
        if changed:
            self.update()
        return (start, end)

    def _ensure_pitch_control_bounds_around_ratio(self, ratio: float) -> bool:
        bounds = self._pitch_segment_bounds_for_ratio(ratio)
        if bounds is None:
            return False
        start, end = bounds
        changed = False
        changed = self._ensure_pitch_control_point_at_ratio(start) or changed
        changed = self._ensure_pitch_control_point_at_ratio(end) or changed
        if changed:
            self.update()
        return changed

    def _select_pitch_control_point(self, index: int | None, additive: bool = False) -> None:
        scene = self.scene()
        preserve_group = bool(
            not additive
            and index is not None
            and index in self._selected_pitch_control_indices
            and self.isSelected()
        )
        if scene is not None and not additive and not preserve_group:
            scene.clearSelection()
            for item in scene.items():
                if isinstance(item, AudioSliceGraphicsItem) and item is not self:
                    item.clear_pitch_control_selection()
        if not additive:
            self._selected_pitch_curve_ranges.clear()
        self._selected_pitch_control_index = index
        if index is None:
            if not additive:
                self._selected_pitch_control_indices.clear()
        elif additive:
            self._selected_pitch_control_indices.add(index)
        elif index not in self._selected_pitch_control_indices:
            self._selected_pitch_control_indices = {index}
        if index is not None:
            self.setSelected(True)
        self._sync_pitch_curve_ranges_from_selected_points()
        self.update()
        self._notify_pitch_curve_selection_changed()

    def _sync_pitch_curve_ranges_from_selected_points(self) -> None:
        """Derive N-editable line ranges from V's selected control points.

        Control points are selection markers, not audio boundaries. Two or more
        selected points delimit curve ranges, while real unvoiced gaps remain
        separate so vibrato never bridges a discontinuity.
        """

        indices = sorted(
            index
            for index in self._selected_pitch_control_indices
            if 0 <= index < len(self.pitch_control_points)
        )
        if len(indices) < 2:
            self._selected_pitch_curve_ranges.clear()
            return

        ranges: list[tuple[float, float]] = []
        voiced_segments = self._raw_pitch_segment_bounds()
        for left_index, right_index in zip(indices, indices[1:], strict=False):
            start = float(self.pitch_control_points[left_index][0])
            end = float(self.pitch_control_points[right_index][0])
            if end < start:
                start, end = end, start
            for voiced_start, voiced_end in voiced_segments:
                clipped_start = max(start, voiced_start)
                clipped_end = min(end, voiced_end)
                if clipped_end - clipped_start > 0.004:
                    ranges.append((clipped_start, clipped_end))
        self._selected_pitch_curve_ranges = self._normalize_pitch_curve_ranges(ranges)

    def _select_pitch_curve_range_at(
        self,
        position: QPointF,
        additive: bool = False,
        max_distance: float = 12.0,
    ) -> bool:
        selected_range = self._pitch_curve_range_bounds_at_position(
            position,
            max_distance=max_distance,
        )
        if selected_range is None:
            return False
        selected_range = self._ensure_pitch_curve_range_boundary_points(selected_range)
        scene = self.scene()
        if scene is not None and not additive:
            scene.clearSelection()
            for item in scene.items():
                if isinstance(item, AudioSliceGraphicsItem) and item is not self:
                    item.clear_pitch_control_selection()
        if not additive:
            self._selected_pitch_control_index = None
            self._selected_pitch_control_indices.clear()
            self._selected_pitch_curve_ranges.clear()
        if not any(
            abs(start - selected_range[0]) <= 1e-5 and abs(end - selected_range[1]) <= 1e-5
            for start, end in self._selected_pitch_curve_ranges
        ):
            self._selected_pitch_curve_ranges.append(selected_range)
            self._selected_pitch_curve_ranges = self._normalize_pitch_curve_ranges(
                self._selected_pitch_curve_ranges
            )
        self.setSelected(True)
        self.update()
        self._notify_pitch_curve_selection_changed()
        return True

    def _pitch_curve_range_bounds_at_position(
        self,
        position: QPointF,
        max_distance: float = 12.0,
    ) -> tuple[float, float] | None:
        if self._pitch_curve_distance_to_position(position) > max_distance:
            return None
        ratio = self._nearest_pitch_curve_ratio(position)
        segment_bounds = self._pitch_segment_bounds_for_ratio(ratio)
        if segment_bounds is None:
            return None
        start, end = segment_bounds
        control_ratios = sorted(
            max(start, min(end, float(x_value)))
            for x_value, _offset in self.pitch_control_points
            if start + 1e-5 < float(x_value) < end - 1e-5
        )
        left_candidates = [x_value for x_value in control_ratios if x_value <= ratio]
        right_candidates = [x_value for x_value in control_ratios if x_value > ratio]
        if left_candidates:
            start = max(left_candidates)
        if right_candidates:
            end = min(right_candidates)
        if end - start <= 0.004:
            return None
        return (max(0.0, start), min(1.0, end))

    def _pitch_segment_bounds_for_ratio(self, ratio: float) -> tuple[float, float] | None:
        return self._raw_pitch_segment_bounds_for_ratio(ratio)

    def _raw_pitch_segment_bounds(self) -> list[tuple[float, float]]:
        if self.pitch_contour is None or len(self.pitch_contour) == 0:
            return [(0.0, 1.0)] if self.pitch_control_points or self.pitch_vibrato_regions else []
        count = len(self.pitch_contour)
        if count <= 1:
            try:
                value = float(self.pitch_contour[0])
            except (TypeError, ValueError):
                return []
            return [(0.0, 1.0)] if math.isfinite(value) else []
        segments: list[tuple[float, float]] = []
        start_index: int | None = None
        for index, value in enumerate(self.pitch_contour):
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                finite = False
            if finite and start_index is None:
                start_index = index
            elif not finite and start_index is not None:
                end_index = max(start_index, index - 1)
                segments.append((start_index / (count - 1), end_index / (count - 1)))
                start_index = None
        if start_index is not None:
            segments.append((start_index / (count - 1), 1.0))
        return segments

    def _raw_pitch_segment_bounds_for_ratio(self, ratio: float) -> tuple[float, float] | None:
        ratio = max(0.0, min(1.0, float(ratio)))
        best_bounds: tuple[float, float] | None = None
        best_distance = float("inf")
        for start, end in self._raw_pitch_segment_bounds():
            if end < start:
                start, end = end, start
            if end - start <= 1e-6:
                contour_length = 0 if self.pitch_contour is None else len(self.pitch_contour)
                pad = 0.5 / max(1, contour_length)
                start = max(0.0, start - pad)
                end = min(1.0, end + pad)
            if start - 1e-5 <= ratio <= end + 1e-5:
                return (start, end)
            distance = min(abs(ratio - start), abs(ratio - end))
            if distance < best_distance:
                best_distance = distance
                best_bounds = (start, end)
        return best_bounds if best_distance <= 0.01 else None

    def _pitch_curve_sample_ratios(
        self,
        start: float,
        end: float,
        rect_width: float,
    ) -> list[float]:
        start = max(0.0, min(1.0, float(start)))
        end = max(0.0, min(1.0, float(end)))
        if end < start:
            start, end = end, start
        if end - start <= 1e-6:
            return [start]

        ratios: set[float] = {start, end}
        width_ratio = end - start
        visible_width = max(1.0, rect_width * width_ratio)
        base_count = max(12, int(math.ceil(visible_width / 4.0)))
        for index in range(base_count + 1):
            ratios.add(start + width_ratio * index / base_count)

        if self.pitch_contour is not None and len(self.pitch_contour) > 1:
            contour_count = len(self.pitch_contour)
            first = max(0, int(math.floor(start * (contour_count - 1))))
            last = min(contour_count - 1, int(math.ceil(end * (contour_count - 1))))
            for index in range(first, last + 1):
                try:
                    finite = math.isfinite(float(self.pitch_contour[index]))
                except (TypeError, ValueError):
                    finite = False
                if finite:
                    ratios.add(index / (contour_count - 1))

        for region in self.pitch_shape_regions_payload():
            try:
                region_start = max(start, float(region["start"]))
                region_end = min(end, float(region["end"]))
                shape = str(region["shape"])
            except (KeyError, TypeError, ValueError):
                continue
            if region_end <= region_start:
                continue
            ratios.add(region_start)
            ratios.add(region_end)
            region_width = region_end - region_start
            if shape == "instant":
                midpoint = region_start + region_width * 0.5
                epsilon = max(1e-5, min(0.003, region_width * 0.04))
                ratios.add(max(region_start, midpoint - epsilon))
                ratios.add(min(region_end, midpoint + epsilon))
            else:
                samples = 7 if shape == "linear" else 17
                for step in range(1, samples - 1):
                    ratios.add(region_start + region_width * step / (samples - 1))

        pixel_epsilon = max(1e-5, min(0.001, 0.45 / max(1.0, rect_width)))
        for region in self.pitch_vibrato_regions_payload():
            try:
                region_start = max(start, float(region["start"]))
                region_end = min(end, float(region["end"]))
                cycles = max(0.0, float(region["cycles"]))
                phase = float(region.get("phase", 0.0)) % 1.0
                waveform = str(region["waveform"])
            except (KeyError, TypeError, ValueError):
                continue
            if region_end <= region_start or cycles <= 0.0:
                continue
            region_width = region_end - region_start
            ratios.add(region_start)
            ratios.add(region_end)
            if waveform == "square":
                transitions = max(1, int(math.ceil(cycles * 2.0)) + 3)
                for step in range(-2, transitions + 1):
                    t = (step * 0.5 - phase) / cycles
                    if not 0.0 < t < 1.0:
                        continue
                    x = region_start + region_width * t
                    ratios.add(max(region_start, x - pixel_epsilon))
                    ratios.add(min(region_end, x + pixel_epsilon))
            elif waveform == "triangle":
                samples = max(4, int(math.ceil(cycles * 4.0)))
                for step in range(samples + 1):
                    ratios.add(region_start + region_width * step / samples)
            else:
                samples = max(24, int(math.ceil(cycles * 24.0)))
                for step in range(samples + 1):
                    ratios.add(region_start + region_width * step / samples)

        return [
            max(start, min(end, ratio))
            for ratio in sorted(ratios)
            if start - 1e-6 <= ratio <= end + 1e-6
        ]

    @staticmethod
    def _normalize_pitch_curve_ranges(
        ranges: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        normalized: list[tuple[float, float]] = []
        for start, end in sorted((min(a, b), max(a, b)) for a, b in ranges):
            start = max(0.0, min(1.0, start))
            end = max(0.0, min(1.0, end))
            if end - start <= 0.004:
                continue
            if normalized and start < normalized[-1][1] - 1e-5:
                normalized[-1] = (normalized[-1][0], max(normalized[-1][1], end))
            else:
                normalized.append((start, end))
        return normalized

    def selected_pitch_curve_ranges(self) -> list[tuple[float, float]]:
        return list(self._selected_pitch_curve_ranges)

    def has_selected_pitch_curve_range(self) -> bool:
        return bool(self._selected_pitch_curve_ranges)

    def _notify_pitch_curve_selection_changed(self) -> None:
        scene = self.scene()
        if scene is None:
            return
        for view in scene.views():
            if hasattr(view, "handle_pitch_curve_selection_changed"):
                view.handle_pitch_curve_selection_changed()

    def select_pitch_control_points_in_scene_rect(
        self,
        scene_rect: QRectF,
        additive: bool,
    ) -> int:
        if not additive:
            self._selected_pitch_control_indices.clear()
            self._selected_pitch_control_index = None
        selected = 0
        for index, (ratio, offset) in enumerate(self.pitch_control_points):
            scene_point = self.mapToScene(
                self._pitch_control_point_to_local(ratio, offset)
            )
            if scene_rect.contains(scene_point):
                self._selected_pitch_control_indices.add(index)
                self._selected_pitch_control_index = index
                selected += 1
        self._sync_pitch_curve_ranges_from_selected_points()
        self.update()
        self._notify_pitch_curve_selection_changed()
        return selected

    def clear_pitch_control_selection(self) -> None:
        had_selection = (
            self._selected_pitch_control_index is not None
            or bool(self._selected_pitch_control_indices)
            or bool(self._selected_pitch_curve_ranges)
        )
        if not had_selection:
            return
        self._selected_pitch_control_index = None
        self._selected_pitch_control_indices.clear()
        self._selected_pitch_curve_ranges.clear()
        self.update()
        self._notify_pitch_curve_selection_changed()

    def delete_selected_pitch_control_point(self) -> bool:
        change = self.remove_selected_pitch_control_points()
        if change is None:
            return False
        before, after = change
        self._notify_parameter_change(before, after)
        return True

    def remove_selected_pitch_control_points(
        self,
    ) -> tuple[dict[str, object], dict[str, object]] | None:
        indices = {
            index
            for index in self._selected_pitch_control_indices
            if 0 <= index < len(self.pitch_control_points)
        }
        if not indices and self._selected_pitch_control_index is not None:
            indices.add(self._selected_pitch_control_index)
        if not indices:
            return None
        before = self.edit_state()
        deleted_ratios = [
            float(self.pitch_control_points[index][0])
            for index in indices
            if 0 <= index < len(self.pitch_control_points)
        ]
        points = [
            point
            for index, point in enumerate(self.pitch_control_points)
            if index not in indices
        ]
        self._selected_pitch_control_index = None
        self._selected_pitch_control_indices.clear()
        self._selected_pitch_curve_ranges.clear()
        self.set_pitch_control_points(points)
        self._remove_pitch_shape_regions_touching_ratios(deleted_ratios)
        return before, self.edit_state()

    def _delete_pitch_control_point_at(self, position: QPointF) -> bool:
        index = self._hit_pitch_control_point(position)
        if index is None:
            return False
        before = self.edit_state()
        points = list(self.pitch_control_points)
        deleted_ratio = float(points[index][0])
        del points[index]
        self._selected_pitch_control_index = None
        self._selected_pitch_control_indices.clear()
        self.set_pitch_control_points(points)
        self._remove_pitch_shape_regions_touching_ratios([deleted_ratio])
        self._notify_parameter_change(before, self.edit_state())
        return True

    def _remove_pitch_shape_regions_touching_ratios(self, ratios: list[float]) -> None:
        if not ratios:
            return
        kept: list[dict[str, float | str]] = []
        for region in self.pitch_shape_regions_payload():
            try:
                start = float(region["start"])
                end = float(region["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if any(abs(ratio - start) <= 0.002 or abs(ratio - end) <= 0.002 for ratio in ratios):
                continue
            kept.append(region)
        self.set_pitch_shape_regions(kept)

    def _pitch_control_handle_rect(self, index: int) -> QRectF:
        ratio, offset = self.pitch_control_points[index]
        point = self._pitch_control_point_to_local(ratio, offset)
        size = self.PITCH_CONTROL_HANDLE_SIZE
        return QRectF(point.x() - size / 2.0, point.y() - size / 2.0, size, size)

    def _pitch_control_handle_edge_at(self, index: int | None, position: QPointF) -> str | None:
        if index is None or not (0 <= index < len(self.pitch_control_points)):
            return None
        ratio, offset = self.pitch_control_points[index]
        point_view = self._local_point_to_view(
            self._pitch_control_point_to_local(ratio, offset)
        )
        position_view = self._local_point_to_view(position)
        half_size = self.PITCH_CONTROL_HANDLE_SIZE / 2.0
        if abs(position_view.y() - point_view.y()) > 6.0:
            return None
        left = point_view.x() - half_size
        right = point_view.x() + half_size
        margin = 3.5
        if position_view.x() < point_view.x() and abs(position_view.x() - left) <= margin:
            return "left"
        if position_view.x() > point_view.x() and abs(position_view.x() - right) <= margin:
            return "right"
        return None

    def _position_near_pitch_curve(
        self,
        position: QPointF,
        max_distance: float | None = None,
    ) -> bool:
        distance = self.PITCH_CURVE_HIT_RADIUS if max_distance is None else float(max_distance)
        return self._pitch_curve_distance_to_position(position) <= distance

    def _nearest_pitch_curve_ratio(self, position: QPointF) -> float:
        segments = self._pitch_curve_local_segments()
        rect = self._pitch_curve_rect()
        if not segments or rect.width() <= 0:
            return 0.0
        position_view = self._local_point_to_view(position)
        nearest = segments[0][0]
        nearest_view = self._local_point_to_view(nearest)
        best_distance = math.hypot(
            position_view.x() - nearest_view.x(),
            position_view.y() - nearest_view.y(),
        )
        for segment in segments:
            for left, right in zip(segment, segment[1:]):
                left_view = self._local_point_to_view(left)
                right_view = self._local_point_to_view(right)
                dx = right_view.x() - left_view.x()
                dy = right_view.y() - left_view.y()
                length_sq = dx * dx + dy * dy
                if length_sq <= 1e-9:
                    t = 0.0
                else:
                    t = (
                        (position_view.x() - left_view.x()) * dx
                        + (position_view.y() - left_view.y()) * dy
                    ) / length_sq
                    t = max(0.0, min(1.0, t))
                candidate = QPointF(
                    left.x() + t * (right.x() - left.x()),
                    left.y() + t * (right.y() - left.y()),
                )
                candidate_view = self._local_point_to_view(candidate)
                distance = math.hypot(
                    position_view.x() - candidate_view.x(),
                    position_view.y() - candidate_view.y(),
                )
                if distance < best_distance:
                    nearest = candidate
                    best_distance = distance
        return max(0.0, min(1.0, (nearest.x() - rect.left()) / rect.width()))

    def _pitch_curve_distance_to_position(self, position: QPointF) -> float:
        segments = self._pitch_curve_local_segments()
        if not segments:
            return float("inf")
        position_view = self._local_point_to_view(position)
        best = float("inf")
        for segment in segments:
            view_points = [self._local_point_to_view(point) for point in segment]
            if len(view_points) == 1:
                best = min(
                    best,
                    math.hypot(
                        view_points[0].x() - position_view.x(),
                        view_points[0].y() - position_view.y(),
                    ),
                )
                continue
            for left, right in zip(view_points, view_points[1:]):
                best = min(best, self._distance_to_segment(position_view, left, right))
        return best

    def _local_point_to_view(self, point: QPointF) -> QPointF:
        scene = self.scene()
        if scene is None or not scene.views():
            return QPointF(point)
        mapped = scene.views()[0].mapFromScene(self.mapToScene(point))
        return QPointF(float(mapped.x()), float(mapped.y()))

    @staticmethod
    def _distance_to_segment(point: QPointF, left: QPointF, right: QPointF) -> float:
        dx = right.x() - left.x()
        dy = right.y() - left.y()
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-6:
            return math.hypot(point.x() - left.x(), point.y() - left.y())
        t = ((point.x() - left.x()) * dx + (point.y() - left.y()) * dy) / length_sq
        t = max(0.0, min(1.0, t))
        proj_x = left.x() + dx * t
        proj_y = left.y() + dy * t
        return math.hypot(point.x() - proj_x, point.y() - proj_y)

    def _update_pitch_control_hover(self, index: int | None) -> None:
        if self._hover_pitch_control_index == index:
            return
        self._hover_pitch_control_index = index
        self._reset_pitch_control_hover_edge()
        self.update()

    def _pitch_control_hover_is_armed(self, edge: str, position: QPointF) -> bool:
        now = time.monotonic()
        if (
            edge != self._pitch_control_hover_edge
            or self._pitch_control_hover_position is None
            or abs(position.x() - self._pitch_control_hover_position.x()) > self.PITCH_CONTROL_HOVER_STABLE_PIXELS
            or abs(position.y() - self._pitch_control_hover_position.y()) > self.PITCH_CONTROL_HOVER_STABLE_PIXELS
        ):
            self._pitch_control_hover_edge = edge
            self._pitch_control_hover_started_at = now
            self._pitch_control_hover_armed = False
            self._pitch_control_hover_position = QPointF(position)
            self._pitch_control_hover_token += 1
            token = self._pitch_control_hover_token
            QTimer.singleShot(
                int(self.PITCH_CONTROL_HOVER_DELAY_SECONDS * 1000),
                lambda token=token: self._arm_pitch_control_hover_if_still(token),
            )
            return False
        return self._pitch_control_hover_armed

    def _arm_pitch_control_hover_if_still(self, token: int) -> None:
        if (
            token == self._pitch_control_hover_token
            and self._pitch_control_hover_edge is not None
            and time.monotonic() - self._pitch_control_hover_started_at >= self.PITCH_CONTROL_HOVER_DELAY_SECONDS
        ):
            self._pitch_control_hover_armed = True
            if self.isUnderMouse():
                self.setCursor(_workspace_cursor("horizontal_resize"))

    def _reset_pitch_control_hover_edge(self) -> None:
        self._pitch_control_hover_edge = None
        self._pitch_control_hover_started_at = 0.0
        self._pitch_control_hover_armed = False
        self._pitch_control_hover_position = None
        self._pitch_control_hover_token += 1

    def _set_pitch_curve_hovered(self, hovered: bool) -> None:
        if not hovered:
            self._hover_pitch_curve_range = None
        if self._pitch_curve_hovered == hovered:
            return
        self._pitch_curve_hovered = hovered
        self.update()

    def _reset_pitch_control_hover(self) -> None:
        changed = self._hover_pitch_control_index is not None
        self._hover_pitch_control_index = None
        self._reset_pitch_control_hover_edge()
        if changed:
            self.update()

    def _paint_pitch_control_points(self, painter: QPainter) -> None:
        if not self.pitch_control_points:
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        control_path: QPainterPath | None = None
        for index, (ratio, offset) in enumerate(self.pitch_control_points):
            point = self._pitch_control_point_to_local(ratio, offset)
            if index == 0:
                control_path = QPainterPath(point)
            elif control_path is not None:
                control_path.lineTo(point)
        if control_path is not None:
            painter.setPen(QPen(QColor(255, 255, 255, 150), 1.2))
            painter.drawPath(control_path)
        for index, (ratio, offset) in enumerate(self.pitch_control_points):
            point = self._pitch_control_point_to_local(ratio, offset)
            selected = (
                index == self._selected_pitch_control_index
                or index in self._selected_pitch_control_indices
            )
            hovered = index == self._hover_pitch_control_index
            radius = 5.2 if selected or hovered else 4.0
            handle_rect = QRectF(point.x() - radius, point.y() - radius, radius * 2.0, radius * 2.0)
            if hovered:
                painter.setPen(QPen(QColor(255, 243, 166, 95), 8.0))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(handle_rect.adjusted(-4.0, -4.0, 4.0, 4.0))
            painter.setPen(QPen(QColor("#10141a"), 1.4 if selected else 1.2))
            painter.setBrush(QColor("#fff3a6") if not selected else QColor("#ffffff"))
            painter.drawEllipse(handle_rect)
        painter.restore()

    def _view_tool_mode(self) -> str:
        scene = self.scene()
        if scene is None:
            return "select"
        scene_tool = scene.property("current_tool")
        if scene_tool:
            return str(scene_tool)
        for view in scene.views():
            if hasattr(view, "tool_mode"):
                return str(view.tool_mode)
        return "select"

    def _view_pitch_curve_tool_mode(self) -> str:
        scene = self.scene()
        if scene is None:
            return "none"
        scene_tool = scene.property("pitch_curve_tool")
        if scene_tool:
            return str(scene_tool)
        for view in scene.views():
            if hasattr(view, "pitch_curve_tool_mode"):
                return str(view.pitch_curve_tool_mode)
        return "none"

    def _effective_pitch_curve_tool_mode(self, modifiers=None) -> str:
        if not self.pitch_curve_edit_mode:
            return "none"
        mode = self._view_pitch_curve_tool_mode()
        scene = self.scene()
        alt_toggle = bool(modifiers is not None and modifiers & Qt.AltModifier)
        if alt_toggle or (scene is not None and bool(scene.property("pitch_curve_space_toggle"))):
            if mode == "curve_select":
                return "curve_point"
            if mode == "curve_point":
                return "curve_select"
        return mode if mode in {"curve_select", "curve_point", "curve_vibrato"} else "none"

    def _effective_tool_mode(self, modifiers=None) -> str:
        mode = self._view_tool_mode()
        scene = self.scene()
        merge_enabled = bool(scene.property("scissors_merge_enabled")) if scene is not None else False
        if modifiers is not None:
            if mode == "select" and modifiers & Qt.AltModifier and modifiers & Qt.ShiftModifier:
                return "cut_merge"
            if mode == "scissors" and modifiers & Qt.AltModifier:
                return "scissors" if merge_enabled else "cut_merge"
            if mode == "select" and modifiers & Qt.AltModifier:
                return "scissors"
        if mode == "scissors" and merge_enabled:
            return "cut_merge"
        return mode


class PitchCurveOverlayItem(QGraphicsItem):
    """Scene-level pitch curve layer, independent from clip block painting."""

    def __init__(self, workspace: "WorkspaceView", bounds: QRectF) -> None:
        super().__init__()
        self.workspace = workspace
        self._bounds = QRectF(bounds)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setZValue(95.0)

    def boundingRect(self) -> QRectF:  # type: ignore[override]
        return QRectF(self._bounds)

    def set_bounds(self, bounds: QRectF) -> None:
        if self._bounds == bounds:
            return
        self.prepareGeometryChange()
        self._bounds = QRectF(bounds)
        self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:  # type: ignore[override]
        if not self.workspace.pitch_curve_edit_mode:
            return
        items = sorted(
            self.workspace.slice_items(),
            key=lambda item: (item.track_index, item.scenePos().x(), item.scenePos().y()),
        )
        for item in items:
            if item.is_missing_source:
                continue
            if item.is_track_reference and not item.reference_editable:
                continue
            if not item.pitch_curve_edit_mode:
                continue
            painter.save()
            painter.translate(item.scenePos())
            item._paint_pitch_contour(painter)
            painter.restore()


class WorkspaceView(QGraphicsView):
    """Main QGraphicsView canvas reserved for waveform slices and pitch blobs."""

    slice_items_created = Signal(object)
    slice_items_dropped = Signal(object)
    preview_requested = Signal(object)
    render_requested = Signal(object)
    global_playback_toggled = Signal()
    playhead_changed = Signal(float)
    playhead_seek_requested = Signal(float)
    slice_edit_finished = Signal(object, object, object)
    slice_edits_finished = Signal(object)
    slice_parameter_changed = Signal(object, object, object)
    slice_boundary_changed = Signal(object)
    slices_merged = Signal(object)
    split_requested = Signal(object, float)
    bgm_file_dropped = Signal(str, float)
    audio_file_dropped = Signal(str, float, float, int)
    delete_requested = Signal(object)
    pitch_curve_view_changed = Signal(bool)
    pitch_curve_selection_changed = Signal(bool)
    horizontal_zoom_changed = Signal(float)

    RULER_HEIGHT = 0.0
    MIDI_NOTE_COUNT = 128
    SCENE_WIDTH = 2400.0
    SCENE_HEIGHT = (
        RULER_HEIGHT
        + MIDI_NOTE_COUNT * AudioSliceGraphicsItem.PITCH_PIXELS_PER_SEMITONE
    )
    BASE_PIXELS_PER_SECOND = AudioSliceGraphicsItem.PIXELS_PER_SECOND
    MIN_HORIZONTAL_ZOOM = 0.08
    MAX_HORIZONTAL_ZOOM = 64.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.active_track_index = 0
        self.tool_mode = "select"
        self.pitch_curve_tool_mode = "none"
        self._pitch_curve_space_toggle = False
        self._split_index_counter = 100000
        self.scale_root = "C"
        self.scale_type = "Chromatic"
        self._grid_items = []
        self._ruler_items = []
        self._scale_band_items = []
        self.snap_to_grid_enabled = False
        self.snap_grid_size = 120.0
        self.default_protect_transients = True
        self.pitch_curve_edit_mode = False
        self.scissors_merge_enabled = False
        self._horizontal_zoom = 1.0
        self._track_lock_states: dict[int, bool] = {}
        self._track_types: dict[int, str] = {}
        self._source_timeline_offsets: dict[str, float] = {}
        self._playhead_time = 0.0
        self._playhead_item = None
        self._rubber_band_origin = None
        self._rubber_band_selects_pitch_points = False
        self._pitch_drag_item: AudioSliceGraphicsItem | None = None
        self._playhead_seeked_on_press = False
        self._rubber_band = QRubberBand(QRubberBand.Rectangle, self.viewport())
        self._boundary_drag: BoundaryDragState | None = None
        self._cut_merge_edge_candidate: tuple[AudioSliceGraphicsItem, str] | None = None
        self._cut_merge_body_candidate: AudioSliceGraphicsItem | None = None
        self.pitch_curve_vibrato_waveform = "sine"
        self._pitch_vibrato_drag: dict[str, object] | None = None
        self._pitch_vibrato_hint_item: QGraphicsSimpleTextItem | None = None

        self.setScene(QGraphicsScene(self))
        self._pitch_curve_overlay_item = PitchCurveOverlayItem(
            self,
            QRectF(0, 0, self.SCENE_WIDTH, self.SCENE_HEIGHT),
        )
        self.scene().addItem(self._pitch_curve_overlay_item)
        self.scene().setProperty("current_tool", self.tool_mode)
        self.scene().setProperty("pitch_curve_tool", self.pitch_curve_tool_mode)
        self.scene().setProperty("pitch_curve_space_toggle", self._pitch_curve_space_toggle)
        self.scene().setProperty("pitch_curve_overlay_enabled", True)
        self.scene().setProperty("pitch_curve_overlay_item", self._pitch_curve_overlay_item)
        self.scene().setProperty("scissors_merge_enabled", self.scissors_merge_enabled)
        self.scene().setProperty("scale_root", self.scale_root)
        self.scene().setProperty("scale_type", self.scale_type)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setRenderHints(self.renderHints())
        self.setFocusPolicy(Qt.StrongFocus)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._build_placeholder_grid()
        self._refresh_viewport_cursor()

    def set_tool_mode(self, mode: str) -> None:
        if mode not in {"select", "scissors", "amplitude", "flatten", "formant"}:
            mode = "select"
        self.tool_mode = mode
        self.scene().setProperty("current_tool", mode)
        if mode not in {"select", "scissors"}:
            self._clear_cut_merge_candidates()
        if mode == "select":
            self.setDragMode(QGraphicsView.RubberBandDrag)
        elif mode == "scissors":
            self.setDragMode(QGraphicsView.NoDrag)
        elif mode == "amplitude":
            self.setDragMode(QGraphicsView.NoDrag)
        elif mode in {"flatten", "formant"}:
            self.setDragMode(QGraphicsView.RubberBandDrag)
        self._refresh_viewport_cursor()

    def set_pitch_curve_tool_mode(self, mode: str) -> None:
        if mode not in {"none", "curve_select", "curve_point", "curve_vibrato"}:
            mode = "curve_select"
        if not self.pitch_curve_edit_mode and mode != "none":
            mode = "none"
        self.pitch_curve_tool_mode = mode
        self.scene().setProperty("pitch_curve_tool", mode)
        self._refresh_viewport_cursor()

    def _set_pitch_curve_space_toggle(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._pitch_curve_space_toggle == enabled:
            return
        self._pitch_curve_space_toggle = enabled
        self.scene().setProperty("pitch_curve_space_toggle", enabled)
        self._refresh_viewport_cursor(QApplication.keyboardModifiers())

    def _effective_pitch_curve_tool_mode(self, modifiers=None) -> str:
        if not self.pitch_curve_edit_mode:
            return "none"
        alt_toggle = bool(modifiers is not None and modifiers & Qt.AltModifier)
        if alt_toggle or self._pitch_curve_space_toggle:
            if self.pitch_curve_tool_mode == "curve_select":
                return "curve_point"
            if self.pitch_curve_tool_mode == "curve_point":
                return "curve_select"
        return (
            self.pitch_curve_tool_mode
            if self.pitch_curve_tool_mode in {"curve_select", "curve_point", "curve_vibrato"}
            else "none"
        )

    def set_pitch_curve_vibrato_waveform(self, waveform: str) -> None:
        self.pitch_curve_vibrato_waveform = (
            waveform if waveform in {"sine", "triangle", "square"} else "sine"
        )
        self._refresh_viewport_cursor()

    def _pitch_vibrato_cursor_name(self) -> str:
        waveform = (
            self.pitch_curve_vibrato_waveform
            if self.pitch_curve_vibrato_waveform in {"sine", "triangle", "square"}
            else "sine"
        )
        return f"pitch_vibrato_{waveform}"

    def set_pitch_curve_edit_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self.pitch_curve_edit_mode == enabled:
            return
        self.pitch_curve_edit_mode = enabled
        if not enabled:
            self.set_pitch_curve_tool_mode("none")
            self._set_pitch_curve_space_toggle(False)
            self._finish_pitch_vibrato_drag(commit=False)
            self._hide_pitch_vibrato_hint()
            self._clear_pitch_control_selection()
        for item in self.slice_items():
            item.set_pitch_curve_edit_mode(enabled)
        self.viewport().update()
        self.pitch_curve_view_changed.emit(enabled)

    def set_scissors_merge_enabled(self, enabled: bool) -> None:
        self.scissors_merge_enabled = bool(enabled)
        self.scene().setProperty("scissors_merge_enabled", self.scissors_merge_enabled)
        if not self.scissors_merge_enabled:
            self._clear_cut_merge_candidates()
        if self.tool_mode == "scissors":
            self._refresh_viewport_cursor()

    def _refresh_viewport_cursor(self, modifiers=None) -> None:
        active_modifiers = (
            QApplication.keyboardModifiers() if modifiers is None else modifiers
        )
        pitch_mode = self._effective_pitch_curve_tool_mode(active_modifiers)
        mode = self._effective_tool_mode(active_modifiers)
        if pitch_mode == "curve_point":
            self.viewport().setCursor(_workspace_cursor("pitch_curve_point"))
        elif pitch_mode == "curve_select":
            self.viewport().setCursor(_workspace_cursor("pitch_curve_select"))
        elif pitch_mode == "curve_vibrato":
            self.viewport().setCursor(_workspace_cursor(self._pitch_vibrato_cursor_name()))
        elif mode == "cut_merge":
            self.viewport().setCursor(_workspace_cursor("cut_merge"))
        elif mode == "scissors":
            self.viewport().setCursor(_workspace_cursor("scissors"))
        elif mode == "amplitude":
            self.viewport().setCursor(_workspace_cursor("gain_vertical"))
        elif mode == "select":
            self.viewport().setCursor(_workspace_cursor("move"))
        else:
            self.viewport().unsetCursor()

    def set_scale(self, root_note: str, scale_type: str) -> None:
        self.scale_root = normalize_root(root_note)
        self.scale_type = normalize_scale_type(scale_type)
        self.scene().setProperty("scale_root", self.scale_root)
        self.scene().setProperty("scale_type", self.scale_type)
        self._rebuild_scale_bands()
        for item in self.slice_items():
            item.update()

    def set_active_track_index(self, track_index: int) -> None:
        self.active_track_index = max(0, track_index)

    def set_track_locked(self, track_index: int, locked: bool) -> None:
        self._track_lock_states[track_index] = locked
        for item in self.scene().items():
            if isinstance(item, AudioSliceGraphicsItem) and item.track_index == track_index:
                item.set_locked(locked)

    def set_track_type(self, track_index: int, track_type: str) -> None:
        self._track_types[track_index] = track_type
        for item in self.scene().items():
            if isinstance(item, AudioSliceGraphicsItem) and item.track_index == track_index:
                item.set_track_type(track_type)

    def clear_track_bindings(self) -> None:
        self._track_lock_states.clear()
        self._track_types.clear()
        self._source_timeline_offsets.clear()

    def set_source_timeline_offset(self, source_path: str, start_time: float) -> None:
        if not source_path:
            return
        self._source_timeline_offsets[str(source_path)] = max(0.0, float(start_time))

    def set_snap_to_grid(self, enabled: bool, grid_size: float | None = None) -> None:
        self.snap_to_grid_enabled = enabled
        if grid_size is not None and grid_size > 0:
            self.snap_grid_size = grid_size
        for item in self.scene().items():
            if isinstance(item, AudioSliceGraphicsItem):
                item.set_snap_to_grid(self.snap_to_grid_enabled, self.snap_grid_size)

    def set_default_transient_protection(self, enabled: bool) -> None:
        self.default_protect_transients = bool(enabled)

    def set_horizontal_zoom(self, zoom: float) -> None:
        old_pps = self.pixels_per_second()
        scene_duration = self._scene_duration_for_zoom(old_pps)
        minimum_zoom = self._minimum_horizontal_zoom(scene_duration)
        resolved_zoom = max(
            minimum_zoom,
            min(self.MAX_HORIZONTAL_ZOOM, float(zoom)),
        )
        if abs(resolved_zoom - self._horizontal_zoom) < 1e-6:
            return
        self.resetTransform()
        self._horizontal_zoom = resolved_zoom
        self._relayout_items_for_zoom(old_pps)
        self._build_placeholder_grid(scene_duration * self.pixels_per_second())
        self.set_playhead_time(self._playhead_time)
        self.horizontal_zoom_changed.emit(self._horizontal_zoom)

    def zoom_in(self) -> float:
        self.set_horizontal_zoom(self._horizontal_zoom * 1.25)
        return self._horizontal_zoom

    def zoom_out(self) -> float:
        self.set_horizontal_zoom(self._horizontal_zoom / 1.25)
        return self._horizontal_zoom

    def reset_horizontal_zoom(self) -> float:
        self.set_horizontal_zoom(1.0)
        return self._horizontal_zoom

    def horizontal_zoom(self) -> float:
        return self._horizontal_zoom

    def pixels_per_second(self) -> float:
        return self.BASE_PIXELS_PER_SECOND * self._horizontal_zoom

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if event.modifiers() & Qt.ControlModifier:
            angle = event.angleDelta().y() if hasattr(event, "angleDelta") else event.delta()
            if angle:
                cursor_pos = self._event_position(event)
                factor = 1.12 if angle > 0 else 1 / 1.12
                self.set_horizontal_zoom_at_cursor(self._horizontal_zoom * factor, cursor_pos)
                event.accept()
                return
        super().wheelEvent(event)

    def set_horizontal_zoom_at_cursor(self, zoom: float, viewport_position) -> float:
        anchored_seconds = self.x_to_seconds(self.mapToScene(viewport_position).x())
        viewport_x = float(viewport_position.x())
        self.set_horizontal_zoom(zoom)
        target_scroll = self.seconds_to_x(anchored_seconds) - viewport_x
        bar = self.horizontalScrollBar()
        bar.setValue(int(round(target_scroll)))
        return self._horizontal_zoom

    def _scene_duration_for_zoom(self, pixels_per_second: float | None = None) -> float:
        pps = max(1.0, float(pixels_per_second or self.pixels_per_second()))
        scene_width = float(self.scene().sceneRect().width()) if self.scene() is not None else self.SCENE_WIDTH
        return max(
            self.SCENE_WIDTH / self.BASE_PIXELS_PER_SECOND,
            scene_width / pps,
            self.timeline_end_time(),
            self.playhead_time() + 0.1,
            4.0,
        )

    def _minimum_horizontal_zoom(self, scene_duration: float | None = None) -> float:
        duration = max(0.1, float(scene_duration or self._scene_duration_for_zoom()))
        viewport_width = max(1.0, float(self.viewport().width()) - 4.0)
        fit_zoom = viewport_width / max(1.0, duration * self.BASE_PIXELS_PER_SECOND)
        return max(self.MIN_HORIZONTAL_ZOOM, min(1.0, fit_zoom))

    def _relayout_items_for_zoom(self, old_pixels_per_second: float) -> None:
        old_pps = max(1.0, float(old_pixels_per_second))
        new_pps = self.pixels_per_second()
        for item in self.slice_items():
            start_seconds = max(0.0, item.scenePos().x() / old_pps)
            width = max(item.MIN_WIDTH, float(item.target_duration) * new_pps)
            item.setPos(start_seconds * new_pps, item.scenePos().y())
            item.setRect(0, 0, width, item.rect().height())
            item.update()

    def _build_placeholder_grid(self, width: float | None = None) -> None:
        scene = self.scene()
        minimum_width = self.seconds_to_x(self.SCENE_WIDTH / self.BASE_PIXELS_PER_SECOND)
        resolved_width = max(
            minimum_width,
            float(width if width is not None else scene.sceneRect().width()),
        )
        for item in self._grid_items:
            scene.removeItem(item)
        for item in self._ruler_items:
            scene.removeItem(item)
        self._grid_items = []
        self._ruler_items = []
        if hasattr(self, "_pitch_curve_overlay_item"):
            self._pitch_curve_overlay_item.set_bounds(
                QRectF(0, 0, resolved_width, self.SCENE_HEIGHT)
            )
        scene.setSceneRect(0, 0, resolved_width, self.SCENE_HEIGHT)
        scene.setBackgroundBrush(QBrush(QColor("#1b1b1d")))
        beat_pen = QPen(QColor("#27282b"))
        pitch_pen = QPen(QColor("#222326"))
        natural_pitch_pen = QPen(QColor("#353b42"))
        do_pen = QPen(QColor("#4a535d"))
        beat_pen.setWidth(0)
        pitch_pen.setWidth(0)
        natural_pitch_pen.setWidthF(0.75)
        do_pen.setWidthF(0.95)

        total_seconds = resolved_width / max(1.0, self.pixels_per_second())
        half_second_count = int(total_seconds * 2) + 3
        for tick in range(half_second_count):
            x = self.seconds_to_x(tick * 0.5)
            if x > resolved_width + 1:
                break
            line = scene.addLine(x, self.RULER_HEIGHT, x, self.SCENE_HEIGHT, beat_pen)
            line.setZValue(-48)
            self._grid_items.append(line)

        row_height = AudioSliceGraphicsItem.PITCH_PIXELS_PER_SEMITONE
        natural_pitch_classes = {0, 2, 4, 5, 7, 9, 11}
        for octave_c in range(0, self.MIDI_NOTE_COUNT, 12):
            high_midi = min(127, octave_c + 11)
            top_y = self.y_for_midi_note(high_midi)
            bottom_y = self.y_for_midi_note(octave_c) + row_height
            octave_band = scene.addRect(
                0,
                top_y,
                resolved_width,
                max(row_height, bottom_y - top_y),
                QPen(Qt.NoPen),
                QBrush(
                    QColor("#20262a")
                    if (octave_c // 12) % 2 == 0
                    else QColor("#1d2225")
                ),
            )
            octave_band.setOpacity(0.14)
            octave_band.setZValue(-72)
            self._grid_items.append(octave_band)

        for midi_note in range(self.MIDI_NOTE_COUNT):
            y = self.y_for_midi_note(midi_note)
            pitch_class = midi_note % 12
            if pitch_class in natural_pitch_classes:
                band_color = (
                    QColor(73, 86, 99, 22)
                    if pitch_class == 0
                    else QColor(56, 66, 75, 15)
                )
                band = scene.addRect(
                    0,
                    y,
                    resolved_width,
                    row_height,
                    QPen(Qt.NoPen),
                    QBrush(band_color),
                )
                band.setZValue(-68)
                self._grid_items.append(band)

            line_y = y + row_height if pitch_class == 0 else y
            line = scene.addLine(
                0,
                line_y,
                resolved_width,
                line_y,
                do_pen if pitch_class == 0 else (
                    natural_pitch_pen if pitch_class in natural_pitch_classes else pitch_pen
                ),
            )
            line.setZValue(-46 if pitch_class == 0 else -50)
            self._grid_items.append(line)

        if self._playhead_item is None:
            self._playhead_item = scene.addLine(
                0,
                0,
                0,
                self.SCENE_HEIGHT,
                QPen(QColor("#ff4a4a"), 2),
            )
            self._playhead_item.setZValue(100)
        self.set_playhead_time(self._playhead_time)
        self._rebuild_scale_bands()

    def _rebuild_scale_bands(self) -> None:
        scene = self.scene()
        for item in self._scale_band_items:
            scene.removeItem(item)
        self._scale_band_items = []
        if self.scale_type == "Chromatic":
            return

        scene_width = max(
            self.seconds_to_x(self.SCENE_WIDTH / self.BASE_PIXELS_PER_SECOND),
            float(scene.sceneRect().width()),
        )
        row_height = AudioSliceGraphicsItem.PITCH_PIXELS_PER_SEMITONE
        for row in range(self.MIDI_NOTE_COUNT):
            midi_note = 127 - row
            y = self.RULER_HEIGHT + row * row_height
            in_scale = is_midi_in_scale(midi_note, self.scale_root, self.scale_type)
            color = QColor(35, 42, 39, 70) if in_scale else QColor(16, 17, 19, 70)
            band = scene.addRect(
                0,
                y,
                scene_width,
                row_height,
                QPen(Qt.NoPen),
                QBrush(color),
            )
            band.setZValue(-66)
            self._scale_band_items.append(band)

    def seconds_to_x(self, seconds: float) -> float:
        return max(0.0, seconds * self.pixels_per_second())

    def x_to_seconds(self, x: float) -> float:
        return max(0.0, x / max(1.0, self.pixels_per_second()))

    def set_playhead_time(self, seconds: float) -> None:
        self._playhead_time = max(0.0, seconds)
        if self._playhead_item is not None:
            x = self.seconds_to_x(self._playhead_time)
            self._playhead_item.setLine(x, 0, x, self.SCENE_HEIGHT)
        self.playhead_changed.emit(self._playhead_time)

    def playhead_time(self) -> float:
        return self._playhead_time

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasFormat(MIME_AUDIO_FILE):
            event.acceptProposedAction()
            return
        if event.mimeData().hasFormat(MIME_AUDIO_SLICES) and self._active_track_accepts_slices():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasFormat(MIME_AUDIO_FILE):
            event.acceptProposedAction()
            return
        if event.mimeData().hasFormat(MIME_AUDIO_SLICES) and self._active_track_accepts_slices():
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasFormat(MIME_AUDIO_FILE):
            try:
                payload = bytes(event.mimeData().data(MIME_AUDIO_FILE))
                path = decode_audio_file(payload)
            except Exception:
                event.ignore()
                return
            scene_position = self.mapToScene(self._event_position(event))
            if self._active_track_accepts_slices():
                self.audio_file_dropped.emit(
                    path,
                    scene_position.x(),
                    scene_position.y(),
                    self.active_track_index,
                )
            else:
                self.bgm_file_dropped.emit(path, scene_position.x())
            event.acceptProposedAction()
            return

        if not event.mimeData().hasFormat(MIME_AUDIO_SLICES) or not self._active_track_accepts_slices():
            super().dropEvent(event)
            return

        try:
            payload = bytes(event.mimeData().data(MIME_AUDIO_SLICES))
            slices = decode_audio_slices(payload)
        except Exception:
            event.ignore()
            return

        scene_position = self.mapToScene(self._event_position(event))
        created_items = self.add_slice_items(slices, scene_position.x(), scene_position.y())
        if created_items:
            self.slice_items_dropped.emit(created_items)
        event.acceptProposedAction()

    def add_slice_items(
        self,
        slices: list[AudioSlice],
        x: float,
        y: float,
        track_index: int | None = None,
        *,
        source_first_start: float | None = None,
        placement_group_id: str | None = None,
    ) -> list[AudioSliceGraphicsItem]:
        scene = self.scene()
        target_track_index = self.active_track_index if track_index is None else track_index
        if not self._track_accepts_slices(target_track_index):
            return []
        created_items: list[AudioSliceGraphicsItem] = []
        placement_group_id = placement_group_id or _new_placement_group_id("drop")
        ordered_slices = sorted(
            slices,
            key=lambda audio_slice: (
                audio_slice.source_path,
                audio_slice.start_time,
                audio_slice.end_time,
                audio_slice.index,
            ),
        )
        first_start = (
            min((audio_slice.start_time for audio_slice in ordered_slices), default=0.0)
            if source_first_start is None
            else float(source_first_start)
        )
        source_paths = {audio_slice.source_path for audio_slice in ordered_slices}
        source_offset = None
        if len(source_paths) == 1:
            source_offset = self._source_timeline_offsets.get(next(iter(source_paths)))
        pixels_per_second = self.pixels_per_second()
        for audio_slice in ordered_slices:
            if source_offset is None:
                item_x = x + max(0.0, audio_slice.start_time - first_start) * pixels_per_second
            else:
                item_x = self.seconds_to_x(source_offset + audio_slice.start_time)
            item_y = self.y_for_midi_note(audio_slice.midi_note, fallback_y=y)
            width = max(
                AudioSliceGraphicsItem.MIN_WIDTH,
                audio_slice.duration * pixels_per_second,
            )
            height = 30.0
            color = self._slice_color(audio_slice)

            slice_item = AudioSliceGraphicsItem(
                audio_slice=audio_slice,
                track_index=target_track_index,
                width=width,
                height=height,
                color=color,
            )
            slice_item.set_pitch_curve_edit_mode(self.pitch_curve_edit_mode)
            slice_item.placement_group_id = placement_group_id
            slice_item.set_track_type(self._track_types.get(target_track_index, "vocal_slice"))
            slice_item.set_transient_protection(self.default_protect_transients)
            slice_item.setPos(item_x, item_y)
            slice_item.set_pitch_anchor_y(item_y)
            slice_item.set_snap_to_grid(self.snap_to_grid_enabled, self.snap_grid_size)
            slice_item.set_locked(self._track_lock_states.get(target_track_index, False))
            scene.addItem(slice_item)
            self._ensure_item_inside_scene(slice_item)
            created_items.append(slice_item)
        if created_items:
            self.slice_items_created.emit(created_items)
        return created_items

    def y_for_midi_note(self, midi_note: int | None, fallback_y: float | None = None) -> float:
        if midi_note is None:
            if fallback_y is None:
                return self.RULER_HEIGHT
            return max(self.RULER_HEIGHT, float(fallback_y))
        clipped = max(0, min(127, int(midi_note)))
        return (
            self.RULER_HEIGHT
            + (127 - clipped) * AudioSliceGraphicsItem.PITCH_PIXELS_PER_SEMITONE
        )

    def restore_slice_item(
        self,
        audio_slice: AudioSlice,
        track_index: int,
        x: float,
        y: float,
        width: float,
        height: float,
        target_midi_note: int | None,
        target_duration: float,
        missing_source: bool = False,
        gain_db: float = 0.0,
        pitch_flatten_amount: float = 0.0,
        formant_shift: float = 0.0,
        protect_transients: bool = True,
        pitch_control_points: object | None = None,
        pitch_vibrato_regions: object | None = None,
        pitch_shape_regions: object | None = None,
        track_reference: bool = False,
        reference_editable: bool = False,
        emit_created: bool = True,
        placement_group_id: str | None = None,
    ) -> AudioSliceGraphicsItem:
        color = self._slice_color(audio_slice)
        slice_item = AudioSliceGraphicsItem(
            audio_slice=audio_slice,
            track_index=track_index,
            width=width,
            height=height,
            color=color,
        )
        slice_item.set_pitch_curve_edit_mode(self.pitch_curve_edit_mode)
        if placement_group_id is not None:
            slice_item.placement_group_id = placement_group_id
        slice_item.set_track_type(self._track_types.get(track_index, "vocal_slice"))
        slice_item.setRect(0, 0, max(slice_item.MIN_WIDTH, width), max(1.0, height))
        slice_item.target_midi_note = (
            audio_slice.midi_note if target_midi_note is None else target_midi_note
        )
        slice_item.target_duration = max(0.001, target_duration)
        slice_item.set_gain_db(gain_db)
        slice_item.set_pitch_flatten_amount(pitch_flatten_amount)
        slice_item.set_formant_shift(formant_shift)
        slice_item.set_transient_protection(protect_transients)
        if pitch_control_points is not None:
            slice_item.set_pitch_control_points(pitch_control_points)
        if pitch_vibrato_regions is not None:
            slice_item.set_pitch_vibrato_regions(pitch_vibrato_regions)
        if pitch_shape_regions is not None:
            slice_item.set_pitch_shape_regions(pitch_shape_regions)
        slice_item.set_track_reference(track_reference, reference_editable)
        slice_item.setPos(x, y)
        slice_item.set_pitch_anchor_y(y)
        slice_item.set_snap_to_grid(self.snap_to_grid_enabled, self.snap_grid_size)
        slice_item.set_missing_source(missing_source)
        slice_item.set_locked(self._track_lock_states.get(track_index, False))
        slice_item.label = slice_item._make_label()
        self.scene().addItem(slice_item)
        self._ensure_item_inside_scene(slice_item)
        if emit_created and not missing_source:
            self.slice_items_created.emit([slice_item])
        return slice_item

    def clear_slice_items(self) -> None:
        scene = self.scene()
        for item in list(self.slice_items()):
            scene.removeItem(item)

    def snapshot_item(self, item: AudioSliceGraphicsItem) -> dict[str, object]:
        state = item.edit_state()
        return {
            "source_path": item.audio_slice.source_path,
            "slice_index": item.audio_slice.index,
            "original_start": item.audio_slice.start_time,
            "original_end": item.audio_slice.end_time,
            "midi_note": item.audio_slice.midi_note,
            "f0_hz": item.audio_slice.f0_hz,
            "pitch_confidence": item.audio_slice.pitch_confidence,
            "analysis_backend": item.audio_slice.analysis_backend,
            "placement_group_id": item.placement_group_id,
            "track_index": item.track_index,
            "missing_source": item.is_missing_source,
            "is_track_reference": item.is_track_reference,
            "reference_editable": item.reference_editable,
            **state,
        }

    def restore_item_snapshot(self, snapshot: dict[str, object]) -> AudioSliceGraphicsItem:
        audio_slice = AudioSlice(
            source_path=str(snapshot["source_path"]),
            index=int(snapshot["slice_index"]),
            start_time=float(snapshot["original_start"]),
            end_time=float(snapshot["original_end"]),
            midi_note=(
                None if snapshot.get("midi_note") is None else int(snapshot["midi_note"])
            ),
            f0_hz=None if snapshot.get("f0_hz") is None else float(snapshot["f0_hz"]),
            pitch_confidence=(
                None
                if snapshot.get("pitch_confidence") is None
                else float(snapshot["pitch_confidence"])
            ),
            analysis_backend=(
                None
                if snapshot.get("analysis_backend") is None
                else str(snapshot["analysis_backend"])
            ),
        )
        return self.restore_slice_item(
            audio_slice=audio_slice,
            track_index=int(snapshot["track_index"]),
            x=float(snapshot["x"]),
            y=float(snapshot["y"]),
            width=float(snapshot["width"]),
            height=float(snapshot["height"]),
            target_midi_note=(
                None
                if snapshot.get("target_midi_note") is None
                else int(snapshot["target_midi_note"])
            ),
            target_duration=float(snapshot["target_duration"]),
            missing_source=bool(snapshot.get("missing_source", False)),
            gain_db=float(snapshot.get("gain_db", 0.0)),
            pitch_flatten_amount=float(snapshot.get("pitch_flatten_amount", 0.0)),
            formant_shift=float(snapshot.get("formant_shift", 0.0)),
            protect_transients=bool(snapshot.get("protect_transients", True)),
            pitch_control_points=snapshot.get("pitch_control_points", []),
            pitch_vibrato_regions=snapshot.get("pitch_vibrato_regions", []),
            pitch_shape_regions=snapshot.get("pitch_shape_regions", []),
            track_reference=bool(snapshot.get("is_track_reference", False)),
            reference_editable=bool(snapshot.get("reference_editable", False)),
            placement_group_id=str(snapshot.get("placement_group_id") or _new_placement_group_id("restore")),
        )

    def remove_slice_item(self, item: AudioSliceGraphicsItem | None) -> None:
        if item is not None and item.scene() is self.scene():
            item.clear_runtime_caches()
            self.scene().removeItem(item)

    def remove_track_reference_items(self, track_index: int) -> None:
        for item in list(self.slice_items()):
            if item.track_index == track_index and item.is_track_reference:
                self.remove_slice_item(item)

    def add_track_reference_items(
        self,
        slices: list[AudioSlice],
        track_index: int,
        start_time: float,
        editable: bool = False,
    ) -> list[AudioSliceGraphicsItem]:
        self.remove_track_reference_items(track_index)
        created_items: list[AudioSliceGraphicsItem] = []
        placement_group_id = _new_placement_group_id(f"track-ref-{track_index}")
        pixels_per_second = self.pixels_per_second()
        for audio_slice in sorted(slices, key=lambda candidate: candidate.start_time):
            x = self.seconds_to_x(max(0.0, float(start_time) + audio_slice.start_time))
            y = self.y_for_midi_note(
                audio_slice.midi_note,
                fallback_y=self.RULER_HEIGHT + 40.0,
            )
            item = self.restore_slice_item(
                audio_slice=audio_slice,
                track_index=track_index,
                x=x,
                y=y,
                width=max(
                    AudioSliceGraphicsItem.MIN_WIDTH,
                    audio_slice.duration * pixels_per_second,
                ),
                height=26.0,
                target_midi_note=audio_slice.midi_note,
                target_duration=audio_slice.duration,
                track_reference=True,
                reference_editable=editable,
                emit_created=False,
                placement_group_id=placement_group_id,
            )
            created_items.append(item)
        return created_items

    def move_track_reference_items(self, track_index: int, start_time: float) -> None:
        for item in self.slice_items():
            if item.track_index != track_index or not item.is_track_reference:
                continue
            item.setPos(
                self.seconds_to_x(max(0.0, float(start_time) + item.audio_slice.start_time)),
                item.scenePos().y(),
            )
            self._ensure_item_inside_scene(item)

    def set_track_reference_editable(self, track_index: int, editable: bool) -> None:
        for item in self.slice_items():
            if item.track_index == track_index and item.is_track_reference:
                item.set_track_reference(True, editable)

    def _replace_item_audio_slice(
        self,
        item: AudioSliceGraphicsItem,
        audio_slice: AudioSlice,
    ) -> None:
        item.audio_slice = audio_slice
        item.setData(0, audio_slice.to_dict())
        item.target_duration = max(0.001, item.rect().width() / self.pixels_per_second())
        item.label = item._make_label()
        item.update()

    def _ensure_item_inside_scene(self, item: AudioSliceGraphicsItem) -> None:
        scene = self.scene()
        scene_rect = scene.sceneRect()
        needed_right = item.scenePos().x() + item.rect().width() + 240.0
        if needed_right > scene_rect.width():
            self._build_placeholder_grid(needed_right)

    def handle_item_double_click(
        self,
        item: AudioSliceGraphicsItem,
        local_position: QPointF,
    ) -> bool:
        pitch_mode = self._effective_pitch_curve_tool_mode(QApplication.keyboardModifiers())
        if pitch_mode != "none":
            return True
        if self._effective_tool_mode(QApplication.keyboardModifiers()) not in {"scissors", "select"}:
            return False
        self.split_requested.emit(item, float(local_position.x()))
        return True

    def handle_item_tool_press(
        self,
        item: AudioSliceGraphicsItem,
        local_position: QPointF,
    ) -> bool:
        mode = self._effective_tool_mode(QApplication.keyboardModifiers())
        if mode == "cut_merge":
            return self._handle_cut_merge_press(item, local_position)
        if mode != "scissors":
            return False
        self.split_requested.emit(item, float(local_position.x()))
        return True

    def _effective_tool_mode(self, modifiers=None) -> str:
        mode = self.tool_mode
        if modifiers is not None:
            if mode == "select" and modifiers & Qt.AltModifier and modifiers & Qt.ShiftModifier:
                return "cut_merge"
            if mode == "scissors" and modifiers & Qt.AltModifier:
                return "scissors" if self.scissors_merge_enabled else "cut_merge"
            if mode == "select" and modifiers & Qt.AltModifier:
                return "scissors"
        if mode == "scissors" and self.scissors_merge_enabled:
            return "cut_merge"
        return mode

    def _handle_cut_merge_press(
        self,
        item: AudioSliceGraphicsItem,
        local_position: QPointF,
    ) -> bool:
        scene_position = item.mapToScene(local_position)
        boundary = self._merge_boundary_at_scene_pos(scene_position)
        if boundary is not None:
            self._clear_cut_merge_candidates()
            self._apply_cut_merge_joined_pair(boundary[0], boundary[1])
            return True

        edge = item._merge_edge_at(local_position)
        if edge is None:
            return self._handle_cut_merge_body_press(item)

        candidate = self._cut_merge_edge_candidate
        if (
            candidate is None
            or candidate[0] is item
            or candidate[0].scene() is not self.scene()
        ):
            self._set_cut_merge_edge_candidate(item, edge)
            return True

        first_item, first_edge = candidate
        operation_done = self._apply_cut_merge_edge_pair(first_item, first_edge, item, edge)
        if operation_done:
            self._clear_cut_merge_candidates()
            return True

        self._set_cut_merge_edge_candidate(item, edge)
        return True

    def _handle_cut_merge_body_press(self, item: AudioSliceGraphicsItem) -> bool:
        candidate = self._cut_merge_body_candidate
        if (
            candidate is None
            or candidate is item
            or candidate.scene() is not self.scene()
        ):
            self._set_cut_merge_body_candidate(item)
            return True

        operation_done = self._apply_cut_merge_body_pair(candidate, item)
        if operation_done:
            self._clear_cut_merge_candidates()
            return True

        self._set_cut_merge_body_candidate(item)
        return True

    def _cut_merge_pair_for_item_click(
        self,
        item: AudioSliceGraphicsItem,
        local_position: QPointF,
    ) -> tuple[AudioSliceGraphicsItem, str, AudioSliceGraphicsItem, str] | None:
        item_left = item.scenePos().x()
        item_right = item_left + item.rect().width()
        left_candidates: list[tuple[float, AudioSliceGraphicsItem]] = []
        right_candidates: list[tuple[float, AudioSliceGraphicsItem]] = []
        for candidate in self.slice_items():
            if candidate is item:
                continue
            if not self._items_can_cut_merge(item, candidate):
                continue
            candidate_left = candidate.scenePos().x()
            candidate_right = candidate_left + candidate.rect().width()
            if candidate_right <= item_left + 2.0:
                left_candidates.append((item_left - candidate_right, candidate))
            elif candidate_left >= item_right - 2.0:
                right_candidates.append((candidate_left - item_right, candidate))

        left_candidates.sort(key=lambda pair: pair[0])
        right_candidates.sort(key=lambda pair: pair[0])
        prefer_right = local_position.x() >= item.rect().width() * 0.5

        if prefer_right and right_candidates:
            return item, "right", right_candidates[0][1], "left"
        if not prefer_right and left_candidates:
            return left_candidates[0][1], "right", item, "left"
        if right_candidates and (
            not left_candidates or right_candidates[0][0] <= left_candidates[0][0]
        ):
            return item, "right", right_candidates[0][1], "left"
        if left_candidates:
            return left_candidates[0][1], "right", item, "left"
        return None

    def split_slice_item(
        self,
        item: AudioSliceGraphicsItem,
        local_x: float,
    ) -> list[AudioSliceGraphicsItem]:
        if item.track_type == "master_bgm":
            return []
        rect = item.rect()
        if rect.width() <= item.MIN_WIDTH * 2:
            return []
        ratio = max(0.05, min(0.95, float(local_x) / max(1.0, rect.width())))
        left_width = max(item.MIN_WIDTH, rect.width() * ratio)
        right_width = max(item.MIN_WIDTH, rect.width() - left_width)
        if left_width + right_width > rect.width():
            scale = rect.width() / (left_width + right_width)
            left_width *= scale
            right_width *= scale

        split_time = item.audio_slice.start_time + item.audio_slice.duration * ratio
        if split_time <= item.audio_slice.start_time or split_time >= item.audio_slice.end_time:
            return []

        left_index = self._split_index_counter
        right_index = self._split_index_counter + 1
        self._split_index_counter += 2
        left_slice = copy_audio_slice(
            item.audio_slice,
            index=left_index,
            start_time=item.audio_slice.start_time,
            end_time=split_time,
        )
        right_slice = copy_audio_slice(
            item.audio_slice,
            index=right_index,
            start_time=split_time,
            end_time=item.audio_slice.end_time,
        )

        scene_position = item.scenePos()
        track_index = item.track_index
        target_midi_note = item.target_midi_note
        left_duration = max(0.001, item.target_duration * ratio)
        right_duration = max(0.001, item.target_duration - left_duration)
        height = item.rect().height()
        missing_source = item.is_missing_source
        gain_db = item.gain_db
        pitch_flatten_amount = item.pitch_flatten_amount
        formant_shift = item.formant_shift
        protect_transients = item.protect_transients
        split_pitch_offset = item._pitch_control_offset_at_ratio(ratio)
        left_pitch_points = [
            {"x": max(0.0, min(1.0, x_value / ratio)), "offset": offset}
            for x_value, offset in item.pitch_control_points
            if x_value <= ratio
        ]
        right_pitch_points = [
            {
                "x": max(0.0, min(1.0, (x_value - ratio) / max(0.001, 1.0 - ratio))),
                "offset": offset,
            }
            for x_value, offset in item.pitch_control_points
            if x_value >= ratio
        ]
        if item.pitch_control_points:
            left_pitch_points.append({"x": 1.0, "offset": split_pitch_offset})
            right_pitch_points.append({"x": 0.0, "offset": split_pitch_offset})
        left_vibrato_regions: list[dict[str, float | str]] = []
        right_vibrato_regions: list[dict[str, float | str]] = []
        left_shape_regions: list[dict[str, float | str]] = []
        right_shape_regions: list[dict[str, float | str]] = []
        right_span = max(0.001, 1.0 - ratio)
        for region in item.pitch_vibrato_regions_payload():
            start = float(region["start"])
            end = float(region["end"])
            span = max(1e-6, end - start)
            cycles = float(region.get("cycles", 0.0))
            phase = float(region.get("phase", 0.0)) % 1.0
            if start < ratio:
                clipped_start = start
                clipped_end = min(end, ratio)
                mapped_start = max(0.0, min(1.0, clipped_start / ratio))
                mapped_end = max(0.0, min(1.0, clipped_end / ratio))
                if mapped_end - mapped_start > 1e-5:
                    left_region = dict(region)
                    left_region["start"] = mapped_start
                    left_region["end"] = mapped_end
                    left_region["cycles"] = cycles * (clipped_end - clipped_start) / span
                    left_region["phase"] = (
                        phase + cycles * (clipped_start - start) / span
                    ) % 1.0
                    left_vibrato_regions.append(left_region)
            if end > ratio:
                clipped_start = max(start, ratio)
                clipped_end = end
                mapped_start = max(0.0, min(1.0, (clipped_start - ratio) / right_span))
                mapped_end = max(0.0, min(1.0, (clipped_end - ratio) / right_span))
                if mapped_end - mapped_start > 1e-5:
                    right_region = dict(region)
                    right_region["start"] = mapped_start
                    right_region["end"] = mapped_end
                    right_region["cycles"] = cycles * (clipped_end - clipped_start) / span
                    right_region["phase"] = (
                        phase + cycles * (clipped_start - start) / span
                    ) % 1.0
                    right_vibrato_regions.append(right_region)
        for region in item.pitch_shape_regions_payload():
            start = float(region["start"])
            end = float(region["end"])
            if start < ratio:
                mapped_start = max(0.0, min(1.0, start / ratio))
                mapped_end = max(0.0, min(1.0, min(end, ratio) / ratio))
                if mapped_end - mapped_start > 1e-5:
                    left_region = dict(region)
                    left_region["start"] = mapped_start
                    left_region["end"] = mapped_end
                    left_shape_regions.append(left_region)
            if end > ratio:
                mapped_start = max(0.0, min(1.0, (max(start, ratio) - ratio) / right_span))
                mapped_end = max(0.0, min(1.0, (end - ratio) / right_span))
                if mapped_end - mapped_start > 1e-5:
                    right_region = dict(region)
                    right_region["start"] = mapped_start
                    right_region["end"] = mapped_end
                    right_shape_regions.append(right_region)

        self.scene().removeItem(item)
        left_item = self.restore_slice_item(
            audio_slice=left_slice,
            track_index=track_index,
            x=scene_position.x(),
            y=scene_position.y(),
            width=left_width,
            height=height,
            target_midi_note=target_midi_note,
            target_duration=left_duration,
            missing_source=missing_source,
            gain_db=gain_db,
            pitch_flatten_amount=pitch_flatten_amount,
            formant_shift=formant_shift,
            protect_transients=protect_transients,
            pitch_control_points=left_pitch_points,
            pitch_vibrato_regions=left_vibrato_regions,
            pitch_shape_regions=left_shape_regions,
            placement_group_id=item.placement_group_id,
        )
        right_item = self.restore_slice_item(
            audio_slice=right_slice,
            track_index=track_index,
            x=scene_position.x() + left_width,
            y=scene_position.y(),
            width=right_width,
            height=height,
            target_midi_note=target_midi_note,
            target_duration=right_duration,
            missing_source=missing_source,
            gain_db=gain_db,
            pitch_flatten_amount=pitch_flatten_amount,
            formant_shift=formant_shift,
            protect_transients=protect_transients,
            pitch_control_points=right_pitch_points,
            pitch_vibrato_regions=right_vibrato_regions,
            pitch_shape_regions=right_shape_regions,
            placement_group_id=item.placement_group_id,
        )
        self._copy_split_caches(item, left_item, right_item, ratio)
        left_item.setSelected(True)
        right_item.setSelected(True)
        return [left_item, right_item]

    def _copy_split_caches(
        self,
        source_item: AudioSliceGraphicsItem,
        left_item: AudioSliceGraphicsItem,
        right_item: AudioSliceGraphicsItem,
        ratio: float,
    ) -> None:
        if source_item.pitch_contour is not None and len(source_item.pitch_contour) > 0:
            contour = np.asarray(source_item.pitch_contour, dtype=np.float32)
            split_index = int(round((len(contour) - 1) * max(0.0, min(1.0, ratio))))
            split_index = max(0, min(len(contour) - 1, split_index))
            left_item.pitch_contour = contour[: split_index + 1].copy()
            right_item.pitch_contour = contour[split_index:].copy()
            left_item.pitch_curve_center_midi = source_item.pitch_curve_center_midi
            right_item.pitch_curve_center_midi = source_item.pitch_curve_center_midi

        if source_item.waveform_envelope is not None and len(source_item.waveform_envelope) > 0:
            envelope = np.asarray(source_item.waveform_envelope)
            split_index = int(round(len(envelope) * max(0.0, min(1.0, ratio))))
            split_index = max(1, min(len(envelope) - 1, split_index))
            left_item.waveform_envelope = envelope[:split_index].copy()
            right_item.waveform_envelope = envelope[split_index:].copy()

        if (
            source_item.base_audio_cache is not None
            and source_item.base_audio_sample_rate is not None
        ):
            split_sample = int(
                round(len(source_item.base_audio_cache) * max(0.0, min(1.0, ratio)))
            )
            left_item.base_audio_cache = source_item.base_audio_cache[:split_sample].copy()
            right_item.base_audio_cache = source_item.base_audio_cache[split_sample:].copy()
            left_item.base_audio_sample_rate = source_item.base_audio_sample_rate
            right_item.base_audio_sample_rate = source_item.base_audio_sample_rate
            left_item.base_audio_level_dbfs = measure_audio_dbfs(left_item.base_audio_cache)
            right_item.base_audio_level_dbfs = measure_audio_dbfs(right_item.base_audio_cache)

        if (
            source_item.render_cache_audio is not None
            and source_item.render_cache_sample_rate is not None
            and source_item.has_current_render_cache()
        ):
            split_sample = int(
                round(len(source_item.render_cache_audio) * max(0.0, min(1.0, ratio)))
            )
            left_item.render_cache_audio = source_item.render_cache_audio[:split_sample].copy()
            right_item.render_cache_audio = source_item.render_cache_audio[split_sample:].copy()
            left_item.render_cache_sample_rate = source_item.render_cache_sample_rate
            right_item.render_cache_sample_rate = source_item.render_cache_sample_rate
            left_item.render_cache_parameters = left_item.current_render_parameters()
            right_item.render_cache_parameters = right_item.current_render_parameters()
            left_item.render_cache_level_dbfs = measure_audio_dbfs(left_item.render_cache_audio)
            right_item.render_cache_level_dbfs = measure_audio_dbfs(right_item.render_cache_audio)

    def request_preview_for_item(self, item: AudioSliceGraphicsItem) -> None:
        self.preview_requested.emit(item)

    def request_render_for_item(self, item: AudioSliceGraphicsItem) -> None:
        self.render_requested.emit(item)

    def handle_item_edit_finished(
        self,
        item: AudioSliceGraphicsItem,
        before: dict[str, object],
        after: dict[str, object],
    ) -> None:
        self.slice_edit_finished.emit(item, before, after)

    def handle_items_edit_finished(
        self,
        changes: list[tuple[AudioSliceGraphicsItem, dict[str, object], dict[str, object]]],
    ) -> None:
        self.slice_edits_finished.emit(changes)

    def handle_item_parameter_change(
        self,
        item: AudioSliceGraphicsItem,
        before: dict[str, object],
        after: dict[str, object],
    ) -> None:
        self.slice_parameter_changed.emit(item, before, after)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self.setFocus(Qt.MouseFocusReason)
        event_position = self._event_position(event)
        scene_position = self.mapToScene(event_position)
        effective_mode = self._effective_tool_mode(event.modifiers())
        pitch_mode = self._effective_pitch_curve_tool_mode(event.modifiers())
        if event.button() == Qt.RightButton and pitch_mode != "none":
            pitch_item = self._pitch_curve_item_at_scene_pos(
                scene_position,
                modifiers=event.modifiers(),
            )
            if pitch_mode == "curve_point" and pitch_item is not None:
                pitch_item._delete_pitch_control_point_at(
                    pitch_item.mapFromScene(scene_position)
                )
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.pitch_curve_edit_mode:
            pitch_mode = self._effective_pitch_curve_tool_mode(event.modifiers())
            if pitch_mode == "curve_vibrato":
                if self._begin_pitch_vibrato_drag(event_position, scene_position):
                    event.accept()
                    return
                event.accept()
                return
            pitch_item = self._pitch_curve_item_at_scene_pos(
                scene_position,
                modifiers=event.modifiers(),
            )
            if pitch_item is not None:
                local_position = pitch_item.mapFromScene(scene_position)
                if pitch_item.handle_pitch_curve_tool_press(local_position, event.modifiers()):
                    if pitch_item._pitch_control_drag_index is not None:
                        self._pitch_drag_item = pitch_item
                    event.accept()
                    return
            if pitch_mode == "curve_select":
                if not (event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)):
                    self._clear_pitch_control_selection()
                self._rubber_band_origin = event_position
                self._rubber_band_selects_pitch_points = True
                self._rubber_band.setGeometry(
                    QRect(event_position, event_position).normalized()
                )
                self._rubber_band.show()
                event.accept()
                return
            if pitch_mode != "none":
                event.accept()
                return
        if event.button() == Qt.RightButton and effective_mode in {"scissors", "cut_merge"}:
            boundary = self._merge_boundary_at_scene_pos(scene_position)
            if boundary is not None and self._items_can_source_merge(boundary[0], boundary[1]):
                self._clear_cut_merge_candidates()
                self.merge_slice_items(boundary[0], boundary[1])
                event.accept()
                return
        if event.button() == Qt.LeftButton and effective_mode in {"scissors", "cut_merge"}:
            for scene_item in self.items(event_position):
                if isinstance(scene_item, AudioSliceGraphicsItem):
                    local_position = scene_item.mapFromScene(scene_position)
                    if self.handle_item_tool_press(scene_item, local_position):
                        event.accept()
                        return
        if (
            event.button() == Qt.LeftButton
            and self.tool_mode == "select"
            and not (event.modifiers() & Qt.ControlModifier)
        ):
            boundary = self._boundary_at_scene_pos(scene_position)
            if boundary is not None:
                left_item, right_item = boundary
                self._begin_boundary_drag(left_item, right_item, scene_position.x())
                event.accept()
                return
        if (
            event.button() == Qt.LeftButton
            and effective_mode == "cut_merge"
        ):
            boundary = self._merge_boundary_at_scene_pos(scene_position)
            if boundary is not None:
                self._clear_cut_merge_candidates()
                self._apply_cut_merge_joined_pair(boundary[0], boundary[1])
                event.accept()
                return
        if event.button() == Qt.LeftButton and scene_position.y() <= self.RULER_HEIGHT:
            seconds = self.x_to_seconds(scene_position.x())
            self.set_playhead_time(seconds)
            self.playhead_seek_requested.emit(seconds)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.tool_mode in {"select", "flatten", "formant"}:
            clicked_slice = any(
                isinstance(item, AudioSliceGraphicsItem)
                for item in self.items(event_position)
            )
            if not clicked_slice:
                self._playhead_seeked_on_press = False
                if not (event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)):
                    seconds = self.x_to_seconds(scene_position.x())
                    self.set_playhead_time(seconds)
                    self.playhead_seek_requested.emit(seconds)
                    self._playhead_seeked_on_press = True
                self._rubber_band_origin = event_position
                self._rubber_band.setGeometry(
                    QRect(event_position, event_position).normalized()
                )
                self._rubber_band.show()
                event.accept()
                return
        super().mousePressEvent(event)

    def _pitch_curve_item_at_scene_pos(
        self,
        scene_position: QPointF,
        max_distance: float = 12.0,
        modifiers=None,
    ) -> AudioSliceGraphicsItem | None:
        mode = self._effective_pitch_curve_tool_mode(modifiers)
        if mode == "none":
            return None
        best_item: AudioSliceGraphicsItem | None = None
        best_distance = float("inf")
        for item in self.slice_items():
            if item._is_fully_locked() or not item.pitch_curve_edit_mode:
                continue
            local_position = item.mapFromScene(scene_position)
            hit_index = item._hit_pitch_control_point(local_position)
            if hit_index is not None:
                return item
            distance = item._pitch_curve_distance_to_position(local_position)
            if distance < best_distance:
                best_distance = distance
                best_item = item
        if best_item is not None and best_distance <= max_distance:
            return best_item
        return None

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._pitch_vibrato_drag is not None:
            self._update_pitch_vibrato_drag(self._event_position(event))
            event.accept()
            return
        if (
            self._pitch_drag_item is not None
            and self._pitch_drag_item.scene() is self.scene()
            and event.buttons() & Qt.LeftButton
        ):
            scene_position = self.mapToScene(self._event_position(event))
            self._pitch_drag_item._update_pitch_control_drag(
                self._pitch_drag_item.mapFromScene(scene_position)
            )
            event.accept()
            return
        if self._boundary_drag is not None:
            self._update_boundary_drag(self.mapToScene(self._event_position(event)).x())
            event.accept()
            return
        if (
            self._rubber_band_origin is not None
            and event.buttons() & Qt.LeftButton
        ):
            event_position = self._event_position(event)
            self._rubber_band.setGeometry(
                QRect(self._rubber_band_origin, event_position).normalized()
            )
            event.accept()
            return
        if (
            event.buttons() & Qt.LeftButton
            and self.pitch_curve_edit_mode
            and self._effective_pitch_curve_tool_mode(event.modifiers()) != "none"
        ):
            event.accept()
            return
        self._refresh_viewport_cursor(event.modifiers())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton and self._pitch_vibrato_drag is not None:
            self._update_pitch_vibrato_drag(self._event_position(event))
            self._finish_pitch_vibrato_drag(commit=True)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._pitch_drag_item is not None:
            drag_item = self._pitch_drag_item
            self._pitch_drag_item = None
            drag_item.finish_pitch_control_drag()
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._boundary_drag is not None:
            self._finish_boundary_drag()
            event.accept()
            return
        if (
            event.button() == Qt.LeftButton
            and self._rubber_band_origin is not None
        ):
            event_position = self._event_position(event)
            band_geometry = self._rubber_band.geometry()
            self._rubber_band.hide()
            if (
                event_position - self._rubber_band_origin
            ).manhattanLength() < QApplication.startDragDistance():
                if self._rubber_band_selects_pitch_points:
                    if not (event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)):
                        self._clear_pitch_control_selection()
                elif not (event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)):
                    self.scene().clearSelection()
                if (
                    not self._rubber_band_selects_pitch_points
                    and not self._playhead_seeked_on_press
                ):
                    scene_position = self.mapToScene(event_position)
                    seconds = self.x_to_seconds(scene_position.x())
                    self.set_playhead_time(seconds)
                    self.playhead_seek_requested.emit(seconds)
            else:
                if self._rubber_band_selects_pitch_points:
                    self._select_pitch_points_in_view_rect(
                        band_geometry,
                        event.modifiers(),
                    )
                else:
                    self._select_slices_in_view_rect(band_geometry, event.modifiers())
            self._rubber_band_origin = None
            self._rubber_band_selects_pitch_points = False
            self._playhead_seeked_on_press = False
            event.accept()
            return

        self._rubber_band_origin = None
        self._rubber_band_selects_pitch_points = False
        self._playhead_seeked_on_press = False
        super().mouseReleaseEvent(event)

    def _boundary_at_scene_pos(
        self,
        scene_position: QPointF,
    ) -> tuple[AudioSliceGraphicsItem, AudioSliceGraphicsItem] | None:
        margin = 7.0
        candidates = sorted(
            self.slice_items(),
            key=lambda item: (item.track_index, item.scenePos().x()),
        )
        for left, right in zip(candidates, candidates[1:], strict=False):
            if left.track_index != right.track_index:
                continue
            if left.track_type == "master_bgm" or right.track_type == "master_bgm":
                continue
            if left.is_locked or right.is_locked:
                continue
            if left.audio_slice.source_path != right.audio_slice.source_path:
                continue
            if abs(left.audio_slice.end_time - right.audio_slice.start_time) > 1e-4:
                continue
            boundary_x = left.scenePos().x() + left.rect().width()
            if abs(boundary_x - right.scenePos().x()) > 2.0:
                continue
            if abs(scene_position.x() - boundary_x) > margin:
                continue
            top = max(left.scenePos().y(), right.scenePos().y())
            bottom = min(
                left.scenePos().y() + left.rect().height(),
                right.scenePos().y() + right.rect().height(),
            )
            if top - margin <= scene_position.y() <= bottom + margin:
                return left, right
        return None

    def _merge_boundary_at_scene_pos(
        self,
        scene_position: QPointF,
    ) -> tuple[AudioSliceGraphicsItem, AudioSliceGraphicsItem] | None:
        margin = 8.0
        candidates = sorted(
            self.slice_items(),
            key=lambda item: (item.track_index, item.scenePos().x()),
        )
        for left, right in zip(candidates, candidates[1:], strict=False):
            if not self._items_can_cut_merge(left, right):
                continue
            boundary_x = left.scenePos().x() + left.rect().width()
            if abs(boundary_x - right.scenePos().x()) > 2.0:
                continue
            if abs(scene_position.x() - boundary_x) > margin:
                continue
            top = min(left.scenePos().y(), right.scenePos().y())
            bottom = max(
                left.scenePos().y() + left.rect().height(),
                right.scenePos().y() + right.rect().height(),
            )
            if top - margin <= scene_position.y() <= bottom + margin:
                return left, right
        return None

    def cancel_cut_at_scene_position(self, scene_position: QPointF) -> bool:
        boundary = self._merge_boundary_at_scene_pos(scene_position)
        if boundary is None:
            return False
        left_item, right_item = boundary
        if not self._items_can_source_merge(left_item, right_item):
            return False
        self._clear_cut_merge_candidates()
        return self.merge_slice_items(left_item, right_item) is not None

    def _items_can_cut_merge(
        self,
        left_item: AudioSliceGraphicsItem,
        right_item: AudioSliceGraphicsItem,
    ) -> bool:
        if left_item is right_item:
            return False
        if left_item.track_index != right_item.track_index:
            return False
        if left_item.track_type == "master_bgm" or right_item.track_type == "master_bgm":
            return False
        if left_item.is_locked or right_item.is_locked:
            return False
        if left_item.scene() is not self.scene() or right_item.scene() is not self.scene():
            return False
        return True

    def _items_can_source_merge(
        self,
        left_item: AudioSliceGraphicsItem,
        right_item: AudioSliceGraphicsItem,
    ) -> bool:
        if not self._items_can_cut_merge(left_item, right_item):
            return False
        if left_item.scenePos().x() > right_item.scenePos().x():
            left_item, right_item = right_item, left_item
        if left_item.audio_slice.source_path != right_item.audio_slice.source_path:
            return False
        if abs(left_item.audio_slice.end_time - right_item.audio_slice.start_time) > 1e-4:
            return False
        return True

    def _items_have_same_target_pitch(
        self,
        left_item: AudioSliceGraphicsItem,
        right_item: AudioSliceGraphicsItem,
    ) -> bool:
        if left_item.target_midi_note is None or right_item.target_midi_note is None:
            return abs(left_item.scenePos().y() - right_item.scenePos().y()) <= 1.0
        return int(left_item.target_midi_note) == int(right_item.target_midi_note)

    def _set_cut_merge_edge_candidate(
        self,
        item: AudioSliceGraphicsItem,
        edge: str,
    ) -> None:
        self._clear_cut_merge_candidates()
        self._cut_merge_edge_candidate = (item, edge)
        item.set_cut_merge_marked(edge)
        item.setSelected(True)

    def _set_cut_merge_body_candidate(self, item: AudioSliceGraphicsItem) -> None:
        self._clear_cut_merge_candidates()
        self._cut_merge_body_candidate = item
        item.setSelected(True)

    def _clear_cut_merge_edge_candidate(self) -> None:
        candidate = self._cut_merge_edge_candidate
        self._cut_merge_edge_candidate = None
        if candidate is not None:
            item, _edge = candidate
            if isinstance(item, AudioSliceGraphicsItem):
                item.set_cut_merge_marked(None)

    def _clear_cut_merge_body_candidate(self) -> None:
        self._cut_merge_body_candidate = None

    def _clear_cut_merge_candidates(self) -> None:
        self._clear_cut_merge_edge_candidate()
        self._clear_cut_merge_body_candidate()

    def _move_cut_merge_target_x(
        self,
        anchor_item: AudioSliceGraphicsItem,
        target_item: AudioSliceGraphicsItem,
    ) -> bool:
        if target_item._is_fully_locked() or target_item.track_type == "master_bgm":
            return False

        before = target_item.edit_state()
        target_x = anchor_item.scenePos().x() + anchor_item.rect().width()
        preserved_target_midi = target_item.target_midi_note
        target_item.setPos(target_x, target_item.scenePos().y())
        target_item.target_midi_note = preserved_target_midi
        target_item.label = target_item._make_label()
        target_item.update()
        self._ensure_item_inside_scene(target_item)
        after = target_item.edit_state()
        if before != after:
            self.slice_edit_finished.emit(target_item, before, after)
        return True

    def _apply_cut_merge_body_pair(
        self,
        first_item: AudioSliceGraphicsItem,
        second_item: AudioSliceGraphicsItem,
    ) -> bool:
        if not self._items_can_cut_merge(first_item, second_item):
            return False
        if first_item.scenePos().x() <= second_item.scenePos().x():
            anchor_item, target_item = first_item, second_item
        else:
            anchor_item, target_item = second_item, first_item
        if self._items_are_visually_joined(anchor_item, target_item):
            return True
        return self._move_cut_merge_target_x(anchor_item, target_item)

    def _apply_cut_merge_edge_pair(
        self,
        first_item: AudioSliceGraphicsItem,
        first_edge: str,
        second_item: AudioSliceGraphicsItem,
        second_edge: str,
    ) -> bool:
        if first_edge == "right" and second_edge == "left":
            anchor_item, target_item = first_item, second_item
        elif first_edge == "left" and second_edge == "right":
            anchor_item, target_item = second_item, first_item
        else:
            return False

        if self._items_are_visually_joined(anchor_item, target_item):
            return self._apply_cut_merge_joined_pair(anchor_item, target_item)

        return self._move_cut_merge_target_x(anchor_item, target_item)

    def _apply_cut_merge_joined_pair(
        self,
        first_item: AudioSliceGraphicsItem,
        second_item: AudioSliceGraphicsItem,
    ) -> bool:
        if first_item.scenePos().x() <= second_item.scenePos().x():
            left_item, right_item = first_item, second_item
        else:
            left_item, right_item = second_item, first_item
        if not self._items_are_visually_joined(left_item, right_item):
            return False

        if not self._items_have_same_target_pitch(left_item, right_item):
            return self._align_cut_merge_pitch(left_item, right_item)

        if self._items_can_source_merge(left_item, right_item):
            return self.merge_slice_items(left_item, right_item) is not None
        return True

    def _align_cut_merge_pitch(
        self,
        anchor_item: AudioSliceGraphicsItem,
        target_item: AudioSliceGraphicsItem,
    ) -> bool:
        if target_item._is_fully_locked() or target_item.track_type == "master_bgm":
            return False
        before = target_item.edit_state()
        target_y = self.y_for_midi_note(
            anchor_item.target_midi_note,
            fallback_y=anchor_item.scenePos().y(),
        )
        target_item.setPos(target_item.scenePos().x(), target_y)
        target_item.target_midi_note = anchor_item.target_midi_note
        target_item.label = target_item._make_label()
        target_item.update()
        self._ensure_item_inside_scene(target_item)
        after = target_item.edit_state()
        if before != after:
            self.slice_edit_finished.emit(target_item, before, after)
        return True

    def _items_are_visually_joined(
        self,
        left_item: AudioSliceGraphicsItem,
        right_item: AudioSliceGraphicsItem,
    ) -> bool:
        if not self._items_can_cut_merge(left_item, right_item):
            return False
        if left_item.scenePos().x() > right_item.scenePos().x():
            left_item, right_item = right_item, left_item
        return abs(left_item.scenePos().x() + left_item.rect().width() - right_item.scenePos().x()) <= 2.0

    def merge_slice_items(
        self,
        first_item: AudioSliceGraphicsItem,
        second_item: AudioSliceGraphicsItem,
    ) -> AudioSliceGraphicsItem | None:
        if first_item.scenePos().x() <= second_item.scenePos().x():
            left_item, right_item = first_item, second_item
        else:
            left_item, right_item = second_item, first_item
        if not self._items_are_visually_joined(left_item, right_item):
            return None
        if not self._items_can_source_merge(left_item, right_item):
            return None
        if not self._items_have_same_target_pitch(left_item, right_item):
            return None

        before = [self.snapshot_item(left_item), self.snapshot_item(right_item)]
        scene_x = min(left_item.scenePos().x(), right_item.scenePos().x())
        scene_y = self.y_for_midi_note(
            left_item.target_midi_note,
            fallback_y=left_item.scenePos().y(),
        )
        right_edge = max(
            left_item.scenePos().x() + left_item.rect().width(),
            right_item.scenePos().x() + right_item.rect().width(),
        )
        width = max(AudioSliceGraphicsItem.MIN_WIDTH, right_edge - scene_x)
        height = max(left_item.rect().height(), right_item.rect().height())
        source_start = min(left_item.audio_slice.start_time, right_item.audio_slice.start_time)
        source_end = max(left_item.audio_slice.end_time, right_item.audio_slice.end_time)
        merged_slice = copy_audio_slice(
            left_item.audio_slice,
            index=self._split_index_counter,
            start_time=source_start,
            end_time=source_end,
        )
        self._split_index_counter += 1
        total_width = max(1.0, left_item.rect().width() + right_item.rect().width())
        left_ratio = left_item.rect().width() / total_width
        merged_pitch_points = [
            {"x": x_value * left_ratio, "offset": offset}
            for x_value, offset in left_item.pitch_control_points
        ]
        merged_pitch_points.extend(
            {
                "x": left_ratio + x_value * (1.0 - left_ratio),
                "offset": offset,
            }
            for x_value, offset in right_item.pitch_control_points
        )
        merged_vibrato_regions: list[dict[str, float | str]] = []
        for region in left_item.pitch_vibrato_regions_payload():
            merged_region = dict(region)
            merged_region["start"] = float(region["start"]) * left_ratio
            merged_region["end"] = float(region["end"]) * left_ratio
            merged_vibrato_regions.append(merged_region)
        for region in right_item.pitch_vibrato_regions_payload():
            merged_region = dict(region)
            merged_region["start"] = left_ratio + float(region["start"]) * (1.0 - left_ratio)
            merged_region["end"] = left_ratio + float(region["end"]) * (1.0 - left_ratio)
            merged_vibrato_regions.append(merged_region)
        merged_shape_regions: list[dict[str, float | str]] = []
        for region in left_item.pitch_shape_regions_payload():
            merged_region = dict(region)
            merged_region["start"] = float(region["start"]) * left_ratio
            merged_region["end"] = float(region["end"]) * left_ratio
            merged_shape_regions.append(merged_region)
        for region in right_item.pitch_shape_regions_payload():
            merged_region = dict(region)
            merged_region["start"] = left_ratio + float(region["start"]) * (1.0 - left_ratio)
            merged_region["end"] = left_ratio + float(region["end"]) * (1.0 - left_ratio)
            merged_shape_regions.append(merged_region)

        self.scene().removeItem(left_item)
        self.scene().removeItem(right_item)
        merged_item = self.restore_slice_item(
            audio_slice=merged_slice,
            track_index=left_item.track_index,
            x=scene_x,
            y=scene_y,
            width=width,
            height=height,
            target_midi_note=left_item.target_midi_note,
            target_duration=max(0.001, width / self.pixels_per_second()),
            missing_source=left_item.is_missing_source or right_item.is_missing_source,
            gain_db=left_item.gain_db,
            pitch_flatten_amount=left_item.pitch_flatten_amount,
            formant_shift=left_item.formant_shift,
            protect_transients=left_item.protect_transients,
            pitch_control_points=merged_pitch_points,
            pitch_vibrato_regions=merged_vibrato_regions,
            pitch_shape_regions=merged_shape_regions,
            track_reference=left_item.is_track_reference and right_item.is_track_reference,
            reference_editable=left_item.reference_editable and right_item.reference_editable,
            placement_group_id=left_item.placement_group_id,
        )
        self._copy_merge_caches(left_item, right_item, merged_item)
        merged_item.setSelected(True)
        after = [self.snapshot_item(merged_item)]
        self.slices_merged.emit(
            {
                "items": [merged_item],
                "before": before,
                "after": after,
            }
        )
        return merged_item

    def _copy_merge_caches(
        self,
        left_item: AudioSliceGraphicsItem,
        right_item: AudioSliceGraphicsItem,
        merged_item: AudioSliceGraphicsItem,
    ) -> None:
        if (
            left_item.base_audio_cache is not None
            and right_item.base_audio_cache is not None
            and left_item.base_audio_sample_rate == right_item.base_audio_sample_rate
            and left_item.base_audio_sample_rate is not None
        ):
            try:
                import numpy as np

                merged_item.base_audio_cache = np.concatenate(
                    [left_item.base_audio_cache, right_item.base_audio_cache]
                )
                merged_item.base_audio_sample_rate = left_item.base_audio_sample_rate
                merged_item.base_audio_level_dbfs = measure_audio_dbfs(merged_item.base_audio_cache)
            except Exception:
                merged_item.base_audio_cache = None
                merged_item.base_audio_sample_rate = None
                merged_item.base_audio_level_dbfs = None

        if (
            left_item.render_cache_audio is not None
            and right_item.render_cache_audio is not None
            and left_item.render_cache_sample_rate == right_item.render_cache_sample_rate
            and left_item.render_cache_sample_rate is not None
            and left_item.has_current_render_cache()
            and right_item.has_current_render_cache()
        ):
            try:
                import numpy as np

                merged_item.render_cache_audio = np.concatenate(
                    [left_item.render_cache_audio, right_item.render_cache_audio]
                )
                merged_item.render_cache_sample_rate = left_item.render_cache_sample_rate
                merged_item.render_cache_parameters = merged_item.current_render_parameters()
                merged_item.render_cache_level_dbfs = measure_audio_dbfs(
                    merged_item.render_cache_audio
                )
            except Exception:
                merged_item.render_cache_audio = None
                merged_item.render_cache_sample_rate = None
                merged_item.render_cache_parameters = None
                merged_item.render_cache_level_dbfs = None

    def _begin_boundary_drag(
        self,
        left_item: AudioSliceGraphicsItem,
        right_item: AudioSliceGraphicsItem,
        scene_x: float,
    ) -> None:
        self._boundary_drag = BoundaryDragState.from_items(
            left_item=left_item,
            right_item=right_item,
            before=[
                self.snapshot_item(left_item),
                self.snapshot_item(right_item),
            ],
            scene_x=scene_x,
        )
        self._boundary_drag.set_edit_notifications_suppressed(True)
        self.viewport().setCursor(_workspace_cursor("horizontal_resize"))

    def _update_boundary_drag(self, scene_x: float) -> None:
        if self._boundary_drag is None:
            return
        self._boundary_drag.preview(scene_x, self._replace_item_audio_slice)

    def _finish_boundary_drag(self) -> None:
        if self._boundary_drag is None:
            return
        drag_state = self._boundary_drag
        left_item, right_item = drag_state.items()
        before = drag_state.before
        after = drag_state.after_snapshots(self.pixels_per_second())
        self._boundary_drag = None
        self._refresh_viewport_cursor()
        if not isinstance(left_item, AudioSliceGraphicsItem) or not isinstance(right_item, AudioSliceGraphicsItem):
            return
        drag_state.set_edit_notifications_suppressed(False)
        if after is None:
            return
        if before != after:
            self.remove_slice_item(left_item)
            self.remove_slice_item(right_item)
            replacement_items = [
                self.restore_item_snapshot(snapshot)
                for snapshot in after
            ]
            for item in replacement_items:
                item.setSelected(True)
            self.slice_boundary_changed.emit(
                {
                    "items": replacement_items,
                    "before": before,
                    "after": after,
                }
            )

    def _select_slices_in_view_rect(self, view_rect: QRect, modifiers) -> None:
        scene_rect = self.mapToScene(view_rect).boundingRect()
        if not (modifiers & (Qt.ControlModifier | Qt.ShiftModifier)):
            self.scene().clearSelection()

        for item in self.scene().items(scene_rect):
            if not isinstance(item, AudioSliceGraphicsItem):
                continue
            if item.sceneBoundingRect().intersects(scene_rect):
                item.setSelected(True)

    def _clear_pitch_control_selection(self) -> None:
        for item in self.slice_items():
            item.clear_pitch_control_selection()

    def handle_pitch_curve_selection_changed(self) -> None:
        self.pitch_curve_selection_changed.emit(self.has_selected_pitch_curve_range())

    def has_selected_pitch_curve_range(self) -> bool:
        return any(item.has_selected_pitch_curve_range() for item in self.slice_items())

    def selected_pitch_curve_line_items(self) -> list[AudioSliceGraphicsItem]:
        return [
            item
            for item in self.slice_items()
            if item.has_selected_pitch_curve_range()
        ]

    def apply_pitch_curve_segment_shape(self, shape: str) -> int:
        changed = 0
        for item in self.selected_pitch_curve_line_items():
            if item.apply_pitch_curve_segment_shape(shape):
                changed += 1
        if changed:
            self.handle_pitch_curve_selection_changed()
        return changed

    def _begin_pitch_vibrato_drag(
        self,
        view_position: QPointF,
        scene_position: QPointF,
        target_hint: AudioSliceGraphicsItem | None = None,
    ) -> bool:
        hit_distance = AudioSliceGraphicsItem.PITCH_VIBRATO_HIT_RADIUS
        target_item: AudioSliceGraphicsItem | None = None
        if (
            isinstance(target_hint, AudioSliceGraphicsItem)
            and target_hint.scene() is self.scene()
            and target_hint.pitch_curve_edit_mode
            and not target_hint._is_fully_locked()
            and target_hint.has_selected_pitch_curve_range()
        ):
            target_item = target_hint
        if target_item is None:
            target_item = self._selected_pitch_curve_item_at_scene_pos(scene_position)
        if target_item is None:
            return False
        local_position = target_item.mapFromScene(scene_position)
        if target_item._pitch_curve_distance_to_position(local_position) > hit_distance:
            return False
        rect = target_item._pitch_curve_rect()
        click_ratio = (
            0.0
            if rect.width() <= 0
            else max(0.0, min(1.0, (local_position.x() - rect.left()) / rect.width()))
        )
        click_inside_selected_range = any(
            start - 0.02 <= click_ratio <= end + 0.02
            for start, end in target_item.selected_pitch_curve_ranges()
        )
        if not click_inside_selected_range:
            return False
        items = self.selected_pitch_curve_line_items()
        if not items:
            items = [target_item] if target_item.has_selected_pitch_curve_range() else []
        if not items:
            return False
        self._pitch_vibrato_drag = {
            "origin": QPointF(view_position),
            "items": [
                {
                    "item": item,
                    "before": item.edit_state(),
                    "base_points": list(item.pitch_control_points),
                    "base_regions": item.pitch_vibrato_regions_payload(),
                }
                for item in items
            ],
            "cycles": 0.0,
            "depth": 0.0,
        }
        self._update_pitch_vibrato_drag(view_position)
        return True

    def _selected_pitch_curve_item_at_scene_pos(
        self,
        scene_position: QPointF,
    ) -> AudioSliceGraphicsItem | None:
        best_item: AudioSliceGraphicsItem | None = None
        best_distance = float("inf")
        for item in self.selected_pitch_curve_line_items():
            local_position = item.mapFromScene(scene_position)
            rect = item._pitch_curve_rect().adjusted(-8.0, -72.0, 8.0, 72.0)
            if not rect.contains(local_position):
                continue
            ratio = (
                0.0
                if item._pitch_curve_rect().width() <= 0
                else (local_position.x() - item._pitch_curve_rect().left())
                / item._pitch_curve_rect().width()
            )
            if not any(
                start - 0.02 <= ratio <= end + 0.02
                for start, end in item.selected_pitch_curve_ranges()
            ):
                continue
            distance = item._pitch_curve_distance_to_position(local_position)
            if distance < best_distance:
                best_distance = distance
                best_item = item
        return best_item

    def _update_pitch_vibrato_drag(self, view_position: QPointF) -> None:
        if self._pitch_vibrato_drag is None:
            return
        origin = self._pitch_vibrato_drag.get("origin")
        if not hasattr(origin, "x") or not hasattr(origin, "y"):
            return
        origin = QPointF(origin)
        view_position = QPointF(view_position)
        dx = float(view_position.x() - origin.x())
        dy = float(view_position.y() - origin.y())
        horizontal = abs(dx)
        vertical = abs(dy)
        cycles = 0.0
        if horizontal >= 3.0:
            cycles = max(0.25, horizontal / 28.0)
            cycles = round(cycles * 4.0) / 4.0
        elif vertical >= 3.0:
            cycles = 1.0
        depth = 0.0
        if cycles >= 0.25:
            depth = max(0.2, min(24.0, 0.2 + vertical / 48.0))
            depth = round(depth * 20.0) / 20.0
        waveform = self.pitch_curve_vibrato_waveform
        phase = 0.5 if dy > 0.0 else 0.0
        entries = list(self._pitch_vibrato_drag.get("items", []))
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            item = entry.get("item")
            base_points = entry.get("base_points")
            base_regions = entry.get("base_regions")
            if (
                isinstance(item, AudioSliceGraphicsItem)
                and isinstance(base_points, list)
                and isinstance(base_regions, list)
            ):
                item.apply_pitch_curve_vibrato(
                    depth=depth,
                    cycles=cycles,
                    waveform=waveform,
                    phase=phase,
                    base_points=base_points,
                    base_regions=base_regions,
                )
        self._pitch_vibrato_drag["cycles"] = cycles
        self._pitch_vibrato_drag["depth"] = depth
        self._pitch_vibrato_drag["phase"] = phase
        self.viewport().setCursor(_workspace_cursor(self._pitch_vibrato_cursor_name()))
        self._show_pitch_vibrato_hint(view_position, cycles, depth, waveform)

    def _finish_pitch_vibrato_drag(self, commit: bool = True) -> None:
        drag = self._pitch_vibrato_drag
        if drag is None:
            return
        self._pitch_vibrato_drag = None
        entries = list(drag.get("items", []))
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            item = entry.get("item")
            before = entry.get("before")
            base_points = entry.get("base_points")
            base_regions = entry.get("base_regions")
            if not isinstance(item, AudioSliceGraphicsItem):
                continue
            if not commit and isinstance(base_points, list):
                item.pitch_control_points = sorted(base_points, key=lambda point: point[0])
                if isinstance(base_regions, list):
                    item.set_pitch_vibrato_regions(base_regions)
                item.update()
                continue
            if isinstance(before, dict):
                item._notify_parameter_change(before, item.edit_state())
        self._hide_pitch_vibrato_hint()
        self._refresh_viewport_cursor()

    def _show_pitch_vibrato_hint(
        self,
        view_position: QPointF,
        cycles: float,
        depth: float,
        waveform: str,
    ) -> None:
        label = {
            "sine": "正弦",
            "triangle": "三角",
            "square": "矩形",
        }.get(waveform, "正弦")
        text = f"{label}  周期 {cycles:g}  振幅 {depth:.2f} st"
        if self._pitch_vibrato_hint_item is None:
            self._pitch_vibrato_hint_item = QGraphicsSimpleTextItem()
            self._pitch_vibrato_hint_item.setBrush(QBrush(QColor("#fff3a6")))
            font = QFont()
            font.setPointSize(10)
            font.setBold(True)
            self._pitch_vibrato_hint_item.setFont(font)
            self._pitch_vibrato_hint_item.setZValue(100000.0)
            self.scene().addItem(self._pitch_vibrato_hint_item)
        self._pitch_vibrato_hint_item.setText(text)
        self._pitch_vibrato_hint_item.setPos(
            self.mapToScene(int(view_position.x()) + 14, int(view_position.y()) - 28)
        )
        self._pitch_vibrato_hint_item.show()

    def _hide_pitch_vibrato_hint(self) -> None:
        if self._pitch_vibrato_hint_item is not None:
            self._pitch_vibrato_hint_item.hide()

    def _select_pitch_points_in_view_rect(self, view_rect: QRect, modifiers) -> None:
        scene_rect = self.mapToScene(view_rect).boundingRect()
        additive = bool(modifiers & (Qt.ControlModifier | Qt.ShiftModifier))
        if not additive:
            self._clear_pitch_control_selection()
        for item in self.slice_items():
            item.select_pitch_control_points_in_scene_rect(scene_rect, additive=True)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            scene_position = self.mapToScene(event.pos())
            pitch_mode = self._effective_pitch_curve_tool_mode(event.modifiers())
            if pitch_mode != "none":
                event.accept()
                return
            if pitch_mode == "none" and self._effective_tool_mode(event.modifiers()) == "select":
                for scene_item in self.items(event.pos()):
                    if isinstance(scene_item, AudioSliceGraphicsItem):
                        local_position = scene_item.mapFromScene(scene_position)
                        self.split_requested.emit(scene_item, float(local_position.x()))
                        event.accept()
                        return
            clicked_slice = any(
                isinstance(item, AudioSliceGraphicsItem)
                for item in self.items(event.pos())
            )
            if not clicked_slice:
                seconds = self.x_to_seconds(scene_position.x())
                self.set_playhead_time(seconds)
                self.playhead_seek_requested.emit(seconds)
                self.global_playback_toggled.emit()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() in (Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt):
            self._refresh_viewport_cursor(QApplication.keyboardModifiers())
            if (
                event.key() == Qt.Key_Alt
                and self.pitch_curve_edit_mode
                and self.pitch_curve_tool_mode in {"curve_select", "curve_point"}
            ):
                self._set_pitch_curve_space_toggle(True)
                event.accept()
                return
            super().keyPressEvent(event)
            return
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if (
                self.pitch_curve_edit_mode
                and self._effective_pitch_curve_tool_mode(event.modifiers()) == "curve_select"
                and self._delete_selected_pitch_control_point()
            ):
                event.accept()
                return
            selected_items = self.selected_slice_items()
            if selected_items:
                self.delete_requested.emit(selected_items)
                event.accept()
                return
        if event.key() == Qt.Key_Space:
            self.global_playback_toggled.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.key() in (Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt):
            self._refresh_viewport_cursor(QApplication.keyboardModifiers())
            if event.key() == Qt.Key_Alt and self._pitch_curve_space_toggle:
                self._set_pitch_curve_space_toggle(False)
                event.accept()
                return
            super().keyReleaseEvent(event)
            return
        super().keyReleaseEvent(event)

    def selected_slice_items(self) -> list[AudioSliceGraphicsItem]:
        return [
            item
            for item in self.scene().selectedItems()
            if isinstance(item, AudioSliceGraphicsItem)
        ]

    def _delete_selected_pitch_control_point(self) -> bool:
        changes: list[
            tuple[AudioSliceGraphicsItem, dict[str, object], dict[str, object]]
        ] = []
        for item in self.slice_items():
            change = item.remove_selected_pitch_control_points()
            if change is not None:
                changes.append((item, change[0], change[1]))
        if not changes:
            return False
        self.slice_edits_finished.emit(changes)
        return True

    def slice_items(self) -> list[AudioSliceGraphicsItem]:
        return [
            item
            for item in self.scene().items()
            if isinstance(item, AudioSliceGraphicsItem)
        ]

    def visible_slice_items(self, margin: float = 120.0) -> list[AudioSliceGraphicsItem]:
        visible_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        visible_rect = visible_rect.adjusted(-margin, -margin, margin, margin)
        return [
            item
            for item in self.slice_items()
            if item.sceneBoundingRect().intersects(visible_rect)
        ]

    def delete_selected_items(self) -> int:
        selected_items = self.selected_slice_items()
        for item in selected_items:
            item.clear_runtime_caches()
            self.scene().removeItem(item)
        return len(selected_items)

    def item_start_time(self, item: AudioSliceGraphicsItem) -> float:
        return self.x_to_seconds(item.scenePos().x())

    def timeline_end_time(self) -> float:
        end_time = 0.0
        for item in self.slice_items():
            end_time = max(end_time, self.item_start_time(item) + item.target_duration)
        return end_time

    def _slice_color(self, audio_slice: AudioSlice) -> QColor:
        if audio_slice.midi_note is None:
            return QColor("#8e99a8")
        hue = int((audio_slice.midi_note % 12) * 30)
        color = QColor()
        color.setHsv(hue, 115, 220)
        return color

    def _active_track_locked(self) -> bool:
        return self._track_lock_states.get(self.active_track_index, False)

    def _active_track_accepts_slices(self) -> bool:
        return self._track_accepts_slices(self.active_track_index)

    def _track_accepts_slices(self, track_index: int) -> bool:
        if self._track_types.get(track_index) == "master_bgm":
            return False
        return not self._track_lock_states.get(track_index, False)

    def _event_position(self, event):
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()
