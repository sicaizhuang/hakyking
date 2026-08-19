from __future__ import annotations

from hakyking.app_settings import DEFAULT_PITCH_ENGINE, PITCH_ENGINE_LABELS, normalize_pitch_engine
from hakyking.audio.playback import AudioOutputDevice, PlaybackManager
from hakyking.qt import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


class AudioSettingsDialog(QDialog):
    """Small playback/rendering dialog for safer real-world audio preview."""

    TEXT = {
        "zh": {
            "title": "音频设置",
            "device": "输出设备",
            "blocksize": "时间线缓冲块",
            "fade": "防爆音淡入淡出",
            "pitch_engine": "变调引擎",
            "ms": " ms",
            "device_error": "无法读取音频设备：",
            "system_default": "系统默认",
            "ok": "确定",
            "cancel": "取消",
        },
        "en": {
            "title": "Audio Settings",
            "device": "Output Device",
            "blocksize": "Timeline Block Size",
            "fade": "Anti-click Fade",
            "pitch_engine": "Pitch Engine",
            "ms": " ms",
            "device_error": "Could not read audio devices: ",
            "system_default": "System Default",
            "ok": "OK",
            "cancel": "Cancel",
        },
    }

    def __init__(
        self,
        parent=None,
        output_device_index: int | None = None,
        blocksize: int = 1024,
        fade_ms: float = 5.0,
        pitch_engine: str = DEFAULT_PITCH_ENGINE,
        language: str = "zh",
    ) -> None:
        super().__init__(parent)
        self._language = "zh" if language == "zh" else "en"
        self._device_read_error = ""
        self._devices = self._read_output_devices()
        self.setWindowTitle(self._text("title"))
        self.resize(460, 260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.form_layout = QFormLayout()
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        self.form_layout.setSpacing(8)

        self.output_device_combo = QComboBox()
        self._populate_devices(output_device_index)
        self.form_layout.addRow(self._text("device"), self.output_device_combo)

        self.blocksize_spin = QSpinBox()
        self.blocksize_spin.setRange(128, 8192)
        self.blocksize_spin.setSingleStep(128)
        self.blocksize_spin.setValue(max(128, min(8192, int(blocksize))))
        self.form_layout.addRow(self._text("blocksize"), self.blocksize_spin)

        self.fade_spin = QDoubleSpinBox()
        self.fade_spin.setRange(0.0, 50.0)
        self.fade_spin.setSingleStep(0.5)
        self.fade_spin.setDecimals(1)
        self.fade_spin.setSuffix(self._text("ms"))
        self.fade_spin.setValue(max(0.0, min(50.0, float(fade_ms))))
        self.form_layout.addRow(self._text("fade"), self.fade_spin)

        self.pitch_engine_combo = QComboBox()
        self._populate_pitch_engines(pitch_engine)
        self.form_layout.addRow(self._text("pitch_engine"), self.pitch_engine_combo)

        layout.addLayout(self.form_layout)

        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        if self._device_read_error:
            self.message_label.setText(self._text("device_error") + self._device_read_error)
        layout.addWidget(self.message_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.button(QDialogButtonBox.Ok).setText(self._text("ok"))
        self.button_box.button(QDialogButtonBox.Cancel).setText(self._text("cancel"))
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def selected_output_device_index(self) -> int | None:
        data = self.output_device_combo.currentData()
        return None if data is None else int(data)

    def selected_blocksize(self) -> int:
        return int(self.blocksize_spin.value())

    def selected_fade_ms(self) -> float:
        return float(self.fade_spin.value())

    def selected_pitch_engine(self) -> str:
        return normalize_pitch_engine(self.pitch_engine_combo.currentData())

    def _read_output_devices(self) -> list[AudioOutputDevice]:
        try:
            devices = PlaybackManager.available_output_devices()
        except Exception as exc:  # noqa: BLE001 - dialog should stay usable without devices
            self._device_read_error = str(exc)
            return [
                AudioOutputDevice(
                    index=None,
                    name=self._text("system_default"),
                    hostapi="",
                    max_output_channels=0,
                    default_samplerate=0.0,
                )
            ]
        if devices:
            return devices
        return [
            AudioOutputDevice(
                index=None,
                name=self._text("system_default"),
                hostapi="",
                max_output_channels=0,
                default_samplerate=0.0,
            )
        ]

    def _populate_devices(self, output_device_index: int | None) -> None:
        selected_row = 0
        for row, device in enumerate(self._devices):
            label = self._device_label(device)
            self.output_device_combo.addItem(label, device.index)
            if device.index == output_device_index:
                selected_row = row
        self.output_device_combo.setCurrentIndex(selected_row)

    def _populate_pitch_engines(self, pitch_engine: str) -> None:
        selected_engine = normalize_pitch_engine(pitch_engine)
        selected_row = 0
        for row, (engine, label) in enumerate(PITCH_ENGINE_LABELS.items()):
            self.pitch_engine_combo.addItem(label, engine)
            if engine == selected_engine:
                selected_row = row
        self.pitch_engine_combo.setCurrentIndex(selected_row)

    def _device_label(self, device: AudioOutputDevice) -> str:
        if device.index is None:
            return self._text("system_default")
        detail = f"{device.name}"
        if device.hostapi:
            detail += f" [{device.hostapi}]"
        if device.max_output_channels:
            detail += f" | {device.max_output_channels} out"
        if device.default_samplerate:
            detail += f" | {device.default_samplerate:.0f} Hz"
        return detail

    def _text(self, key: str) -> str:
        return self.TEXT[self._language][key]
