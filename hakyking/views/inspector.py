from __future__ import annotations

from hakyking.audio.gain import (
    MAX_GAIN_DB,
    MIN_GAIN_DB,
    format_dbfs,
    gain_db_to_percent,
)
from hakyking.qt import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QAbstractSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    Qt,
    Signal,
    QVBoxLayout,
    QWidget,
)


class InspectorWidget(QWidget):
    """Compact tuning panel for the selected audio slice(s)."""

    # Advanced tuning controls are temporarily disabled while the DSP workflow
    # is being stabilized. Keep their API alive for project compatibility, but
    # do not update or expose them during normal editing.
    ADVANCED_CONTROLS_ENABLED = False

    parameter_change_committed = Signal(object, object, object)
    transient_protection_toggled = Signal(bool)
    pitch_curve_view_toggled = Signal(bool)
    tuning_preset_requested = Signal(str)

    TEXT = {
        "zh": {
            "title": "片段属性",
            "no_slice": "未选中片段",
            "one_slice": "T{track}  {note}",
            "many_slices": "已选 {count} 个片段",
            "preset": "调音预设",
            "reset": "参数归零",
            "natural": "自然校正",
            "flat": "电音压平",
            "bright": "明亮化",
            "deep": "厚声化",
            "flatten": "颤音展平",
            "formant": "共振峰偏移",
            "gain": "增益",
            "level": "电平",
            "transients": "瞬态保护",
            "transients_tip": "变速时尽量保留辅音、气声与爆破瞬态。",
        },
        "en": {
            "title": "Clip Properties",
            "no_slice": "No clip",
            "one_slice": "T{track}  {note}",
            "many_slices": "{count} clips selected",
            "preset": "Tuning Preset",
            "reset": "Reset Parameters",
            "natural": "Natural Correction",
            "flat": "Flat Tune",
            "bright": "Brighten",
            "deep": "Thicken",
            "flatten": "Vibrato Flatten",
            "formant": "Formant Shift",
            "gain": "Gain",
            "level": "Level",
            "transients": "Transient Protect",
            "transients_tip": "Protect consonant transients during time stretching.",
        },
    }

    PRESET_ORDER = ("reset", "natural", "flat", "bright", "deep")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._item = None
        self._selection_count = 0
        self._updating = False
        self._language = "zh"
        self._transient_protection_enabled = True

        self.setMinimumWidth(128)
        self.setMaximumWidth(180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        self.title_label = QLabel()
        self.title_label.setObjectName("InspectorTitle")
        self.status_label = QLabel()
        self.status_label.setObjectName("InspectorStatus")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.status_label, 1)
        layout.addWidget(header)

        self.transient_protection_checkbox = QCheckBox()
        self.transient_protection_checkbox.setChecked(True)
        self.pitch_curve_view_checkbox = QCheckBox()
        self.pitch_curve_view_checkbox.setChecked(False)
        layout.addWidget(self.pitch_curve_view_checkbox)
        layout.addWidget(self.transient_protection_checkbox)

        self.preset_combo = QComboBox()
        self.preset_combo.setObjectName("TuningPresetCombo")
        layout.addWidget(self.preset_combo)

        self.flatten_slider = QSlider(Qt.Horizontal)
        self.flatten_slider.setRange(0, 100)
        self.flatten_spin = QSpinBox()
        self.flatten_spin.setRange(0, 100)
        self.flatten_spin.setSuffix("%")
        self.flatten_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.flatten_spin.setAlignment(Qt.AlignCenter)
        self.flatten_label = QLabel()
        self.flatten_label.setToolTip("颤音展平 / Vibrato flatten")
        self.flatten_row = self._control_row(
            self.flatten_label,
            self.flatten_slider,
            self.flatten_spin,
        )
        layout.addWidget(self.flatten_row)

        self.formant_slider = QSlider(Qt.Horizontal)
        self.formant_slider.setRange(-120, 120)
        self.formant_spin = QDoubleSpinBox()
        self.formant_spin.setRange(-12.0, 12.0)
        self.formant_spin.setDecimals(1)
        self.formant_spin.setSingleStep(0.1)
        self.formant_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.formant_spin.setAlignment(Qt.AlignCenter)
        self.formant_label = QLabel()
        self.formant_label.setToolTip("共振峰偏移 / Formant shift")
        self.formant_row = self._control_row(
            self.formant_label,
            self.formant_slider,
            self.formant_spin,
        )
        layout.addWidget(self.formant_row)

        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(int(MIN_GAIN_DB * 10), int(MAX_GAIN_DB * 10))
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(MIN_GAIN_DB, MAX_GAIN_DB)
        self.gain_spin.setDecimals(1)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.setSuffix(" dB")
        self.gain_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.gain_spin.setAlignment(Qt.AlignCenter)
        self.gain_label = QLabel()
        self.gain_label.setToolTip("片段增益 / Gain")
        self.gain_row = self._control_row(
            self.gain_label,
            self.gain_slider,
            self.gain_spin,
        )
        layout.addWidget(self.gain_row)

        self.gain_level_label = QLabel()
        self.gain_level_label.setObjectName("InspectorHint")
        layout.addWidget(self.gain_level_label)
        layout.addStretch(1)

        self.preset_combo.activated.connect(self._on_preset_activated)
        self.flatten_slider.valueChanged.connect(self._on_flatten_slider_changed)
        self.flatten_spin.valueChanged.connect(self._on_flatten_spin_changed)
        self.formant_slider.valueChanged.connect(self._on_formant_slider_changed)
        self.formant_spin.valueChanged.connect(self._on_formant_spin_changed)
        self.gain_slider.valueChanged.connect(self._on_gain_slider_changed)
        self.gain_spin.valueChanged.connect(self._on_gain_spin_changed)
        self.transient_protection_checkbox.toggled.connect(
            self._on_transient_protection_toggled
        )
        self.pitch_curve_view_checkbox.toggled.connect(
            lambda checked: self.pitch_curve_view_toggled.emit(bool(checked))
        )

        self.flatten_slider.sliderReleased.connect(self.commit_current_values)
        self.formant_slider.sliderReleased.connect(self.commit_current_values)
        self.gain_slider.sliderReleased.connect(self.commit_current_values)
        self.flatten_spin.editingFinished.connect(self.commit_current_values)
        self.formant_spin.editingFinished.connect(self.commit_current_values)
        self.gain_spin.editingFinished.connect(self.commit_current_values)

        if not self.ADVANCED_CONTROLS_ENABLED:
            self.title_label.hide()
            self.status_label.hide()
            self.preset_combo.hide()
            self.flatten_row.hide()
            self.formant_row.hide()
            self.gain_row.hide()
            self.gain_level_label.hide()

        self.set_language("zh")
        self.set_item(None)

    def set_language(self, language: str) -> None:
        self._language = "zh" if language == "zh" else "en"
        text = self.TEXT[self._language]
        self.title_label.setText(text["title"])
        self.pitch_curve_view_checkbox.setText(
            "音高曲线" if self._language == "zh" else "Pitch Curve"
        )
        self.pitch_curve_view_checkbox.setToolTip(
            "显示波形与音高曲线编辑层" if self._language == "zh"
            else "Show waveform and pitch contour"
        )
        self.transient_protection_checkbox.setText(text["transients"])
        self.transient_protection_checkbox.setToolTip(text["transients_tip"])
        if not self.ADVANCED_CONTROLS_ENABLED:
            return
        self.flatten_label.setText(text["flatten"])
        self.formant_label.setText(text["formant"])
        self._populate_presets()
        self._refresh_gain_label(self.gain_spin.value())
        self.set_item(self._item, self._selection_count)

    def set_item(self, item, selection_count: int | None = None) -> None:
        self._item = item
        self._selection_count = (
            max(0, int(selection_count))
            if selection_count is not None
            else (1 if item is not None else 0)
        )
        if not self.ADVANCED_CONTROLS_ENABLED:
            self._updating = True
            self.transient_protection_checkbox.setEnabled(True)
            self.transient_protection_checkbox.setChecked(
                self._transient_protection_enabled
                if item is None
                else bool(item.protect_transients)
            )
            self._updating = False
            return
        self._updating = True
        voice_enabled = (
            item is not None
            and item.track_type != "master_bgm"
            and not item.is_locked
            and not item.is_missing_source
        )
        gain_enabled = (
            item is not None
            and not item.is_missing_source
            and (not item.is_locked or item.track_type == "master_bgm")
        )
        has_selection = item is not None
        text = self.TEXT[self._language]
        if item is None:
            self.status_label.setText(text["no_slice"])
        elif self._selection_count > 1:
            self.status_label.setText(text["many_slices"].format(count=self._selection_count))
        else:
            self.status_label.setText(
                text["one_slice"].format(
                    track=item.track_index + 1,
                    note=item.audio_slice.note_name,
                )
            )

        self.preset_combo.setEnabled(voice_enabled)
        for widget in (
            self.flatten_slider,
            self.flatten_spin,
            self.formant_slider,
            self.formant_spin,
        ):
            widget.setEnabled(voice_enabled)
        for widget in (self.gain_slider, self.gain_spin):
            widget.setEnabled(gain_enabled)

        self.transient_protection_checkbox.setEnabled(has_selection)
        if item is None:
            self.flatten_slider.setValue(0)
            self.flatten_spin.setValue(0)
            self.formant_slider.setValue(0)
            self.formant_spin.setValue(0.0)
            self.gain_slider.setValue(0)
            self.gain_spin.setValue(0.0)
            self.transient_protection_checkbox.setChecked(self._transient_protection_enabled)
        else:
            self.flatten_slider.setValue(int(round(item.pitch_flatten_amount * 100.0)))
            self.flatten_spin.setValue(int(round(item.pitch_flatten_amount * 100.0)))
            self.formant_slider.setValue(int(round(item.formant_shift * 10.0)))
            self.formant_spin.setValue(float(item.formant_shift))
            self.gain_slider.setValue(int(round(item.gain_db * 10.0)))
            self.gain_spin.setValue(float(item.gain_db))
            self.transient_protection_checkbox.setChecked(bool(item.protect_transients))
        self._refresh_gain_label(self.gain_spin.value())
        self._updating = False

    def commit_current_values(self) -> None:
        if self._item is None or self._updating:
            return
        before = self._item.edit_state()
        after = dict(before)
        if self._item.track_type != "master_bgm" and not self._item.is_locked:
            after["pitch_flatten_amount"] = self.flatten_spin.value() / 100.0
            after["formant_shift"] = self.formant_spin.value()
        if not self._item.is_missing_source and (
            not self._item.is_locked or self._item.track_type == "master_bgm"
        ):
            after["gain_db"] = self.gain_spin.value()
        if before == after:
            return
        self.parameter_change_committed.emit(self._item, before, after)

    def _control_row(self, label: QLabel, slider: QSlider, spinbox: QWidget) -> QFrame:
        frame = QFrame()
        frame.setObjectName("InspectorRow")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(2, 3, 2, 3)
        layout.setSpacing(4)
        label.setFixedWidth(76)
        spinbox.setFixedWidth(64)
        slider.setMinimumWidth(36)
        layout.addWidget(label)
        layout.addWidget(slider, 1)
        layout.addWidget(spinbox)
        return frame

    def _populate_presets(self) -> None:
        text = self.TEXT[self._language]
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem(text["preset"], "")
        for preset_name in self.PRESET_ORDER:
            self.preset_combo.addItem(text[preset_name], preset_name)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)

    def _on_preset_activated(self, index: int) -> None:
        preset_name = self.preset_combo.itemData(index)
        self.preset_combo.setCurrentIndex(0)
        if preset_name:
            self.tuning_preset_requested.emit(str(preset_name))

    def _on_flatten_slider_changed(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self.flatten_spin.setValue(value)
        self._updating = False

    def _on_flatten_spin_changed(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self.flatten_slider.setValue(value)
        self._updating = False

    def _on_formant_slider_changed(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self.formant_spin.setValue(value / 10.0)
        self._updating = False

    def _on_formant_spin_changed(self, value: float) -> None:
        if self._updating:
            return
        self._updating = True
        self.formant_slider.setValue(int(round(value * 10.0)))
        self._updating = False

    def _on_gain_slider_changed(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self.gain_spin.setValue(value / 10.0)
        self._refresh_gain_label(value / 10.0)
        self._updating = False

    def _on_gain_spin_changed(self, value: float) -> None:
        if self._updating:
            return
        self._updating = True
        self.gain_slider.setValue(int(round(value * 10.0)))
        self._refresh_gain_label(value)
        self._updating = False

    def _on_transient_protection_toggled(self, checked: bool) -> None:
        if self._updating:
            return
        self._transient_protection_enabled = bool(checked)
        self.transient_protection_toggled.emit(bool(checked))

    def transient_protection_enabled(self) -> bool:
        return self.transient_protection_checkbox.isChecked()

    def set_transient_protection_enabled(self, enabled: bool) -> None:
        self._transient_protection_enabled = bool(enabled)
        self.transient_protection_checkbox.blockSignals(True)
        self.transient_protection_checkbox.setChecked(bool(enabled))
        self.transient_protection_checkbox.blockSignals(False)

    def set_pitch_curve_view_enabled(self, enabled: bool) -> None:
        self.pitch_curve_view_checkbox.blockSignals(True)
        self.pitch_curve_view_checkbox.setChecked(bool(enabled))
        self.pitch_curve_view_checkbox.blockSignals(False)

    def _refresh_gain_label(self, gain_db: float) -> None:
        percent = gain_db_to_percent(gain_db)
        percent_text = f"{percent:.1f}" if percent < 10.0 else f"{percent:.0f}"
        self.gain_label.setText(self.TEXT[self._language]["gain"])
        self.gain_label.setToolTip(f"{self.TEXT[self._language]['gain']} {percent_text}%")
        level = (
            None
            if self._item is None
            else self._item.measured_level_dbfs(gain_db_override=gain_db)
        )
        if level is None:
            level_text = "--"
        else:
            level_text = f"RMS {format_dbfs(level[0])} / 峰 {format_dbfs(level[1])}"
        self.gain_level_label.setText(
            f"{self.TEXT[self._language]['level']} {level_text}"
        )
