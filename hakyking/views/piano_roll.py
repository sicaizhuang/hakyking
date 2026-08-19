from __future__ import annotations

from hakyking.models.scale import is_midi_in_scale, normalize_root, normalize_scale_type
from hakyking.qt import (
    QColor,
    QLinearGradient,
    QPainter,
    QPen,
    QRectF,
    QSize,
    Qt,
    QWidget,
)


class PianoRollWidget(QWidget):
    """Reserved piano roll area representing MIDI pitch rows 0-127."""

    NOTE_COUNT = 128
    ROW_HEIGHT = 20
    BLACK_KEYS = {1, 3, 6, 8, 10}

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._hover_midi: int | None = None
        self._scale_root = "C"
        self._scale_type = "Chromatic"
        self.setMouseTracking(True)

    def minimumSizeHint(self) -> QSize:
        return QSize(96, self.NOTE_COUNT * self.ROW_HEIGHT)

    def sizeHint(self) -> QSize:
        return QSize(120, self.NOTE_COUNT * self.ROW_HEIGHT)

    def set_scale(self, root_note: str, scale_type: str) -> None:
        self._scale_root = normalize_root(root_note)
        self._scale_type = normalize_scale_type(scale_type)
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#202020"))

        row_height = max(4.0, self.height() / self.NOTE_COUNT)
        white_top = QColor("#e2e2e2")
        white_bottom = QColor("#c8c8c8")
        black = QColor("#151515")
        black_edge = QColor("#2b2b2b")
        grid = QPen(QColor("#363636"))
        text = QColor("#1d1d1d")
        black_key_width = self.width() * 0.68

        for midi in range(self.NOTE_COUNT):
            y = self.height() - (midi + 1) * row_height
            pitch_class = midi % 12
            if pitch_class in self.BLACK_KEYS:
                continue

            key_rect = QRectF(0, y, self.width(), row_height + 1)
            in_scale = is_midi_in_scale(midi, self._scale_root, self._scale_type)
            gradient = QLinearGradient(0, key_rect.top(), 0, key_rect.bottom())
            gradient.setColorAt(0.0, white_top if in_scale else QColor("#8d8d8d"))
            gradient.setColorAt(1.0, white_bottom if in_scale else QColor("#6f6f6f"))
            painter.fillRect(key_rect, gradient)
            painter.setPen(grid)
            painter.drawLine(0, int(y), self.width(), int(y))

            if pitch_class == 0 and row_height >= 6:
                painter.setPen(text)
                octave = midi // 12 - 1
                painter.drawText(8, int(y + row_height - 2), f"C{octave}")

        for midi in range(self.NOTE_COUNT):
            pitch_class = midi % 12
            if pitch_class not in self.BLACK_KEYS:
                continue
            y = self.height() - (midi + 1) * row_height
            key_rect = QRectF(0, y + 1, black_key_width, max(1.0, row_height - 1))
            in_scale = is_midi_in_scale(midi, self._scale_root, self._scale_type)
            painter.fillRect(key_rect, black if in_scale else QColor("#080808"))
            painter.setPen(QPen(black_edge))
            painter.drawRect(key_rect.adjusted(0, 0, -1, -1))

        if self._hover_midi is not None:
            hover_y = self.height() - (self._hover_midi + 1) * row_height
            painter.fillRect(
                QRectF(0, hover_y, self.width(), row_height + 1),
                QColor(63, 142, 197, 92),
            )
            painter.setPen(QPen(QColor("#8ed0ff")))
            painter.drawLine(0, int(hover_y), self.width(), int(hover_y))
            painter.drawLine(
                0,
                int(hover_y + row_height),
                self.width(),
                int(hover_y + row_height),
            )

        painter.setPen(QPen(QColor("#3b3b3b")))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        position_y = event.position().y() if hasattr(event, "position") else event.y()
        next_hover = self._midi_from_y(float(position_y))
        if next_hover != self._hover_midi:
            self._hover_midi = next_hover
            self.update()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            position_y = event.position().y() if hasattr(event, "position") else event.y()
            midi_note = self._midi_from_y(float(position_y))
            if midi_note is not None:
                self._hover_midi = midi_note
                self.update()
                self._play_midi_note(midi_note)
                event.accept()
                return
        super().mousePressEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._hover_midi = None
        self.update()
        super().leaveEvent(event)

    def _midi_from_y(self, y: float) -> int | None:
        if self.height() <= 0:
            return None
        midi = int(self.NOTE_COUNT - 1 - (y / self.height()) * self.NOTE_COUNT)
        return max(0, min(self.NOTE_COUNT - 1, midi))

    def _play_midi_note(self, midi_note: int) -> None:
        try:
            import sounddevice as sd

            from hakyking.audio.piano_preview import synthesize_piano_note

            sample_rate = 44100
            tone = synthesize_piano_note(midi_note, sample_rate=sample_rate)
            sd.stop()
            sd.play(tone, sample_rate, blocking=False)
        except Exception:
            return
