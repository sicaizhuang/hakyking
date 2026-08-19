from __future__ import annotations

from collections import OrderedDict
import math
import os
import time
from pathlib import Path

from hakyking.app_settings import (
    AudioPlaybackSettings,
    load_audio_settings,
    save_audio_settings,
)
from hakyking.audio.exporter import ExportClip
from hakyking.audio.playback import PlaybackManager, TimelineClip, mix_playback_buffers
from hakyking.audio.reader import AudioInfo
from hakyking.audio.slicer import build_full_audio_slice
from hakyking.commands import (
    AddSliceCommand,
    BoundaryMoveCommand,
    ChangeParameterCommand,
    DeleteSliceCommand,
    MaterialSlicesCommand,
    MergeSlicesCommand,
    MoveSliceCommand,
    MoveSlicesCommand,
    PasteSliceCommand,
    SplitSliceCommand,
)
from hakyking.controllers.audio_worker import (
    AudioProbeWorker,
    ExportWorker,
    FilePreviewWorker,
    FolderMediaScanWorker,
    ParseWorker,
    RenderWorker,
    SlicePreviewWorker,
    SliceSequencePreviewWorker,
    WaveformWorker,
    WholeSliceWorker,
)
from hakyking.models.audio_slice import AudioSlice
from hakyking.models.project import ProjectModel, TrackModel, TrackRole, TrackType
from hakyking.models.scale import nearest_midi_in_scale
from hakyking.project_manager import ProjectManager
from hakyking.qt import QFileDialog, QGraphicsItem, QMessageBox, QObject, QThread, QTimer
from hakyking.runtime import user_data_root
from hakyking.views.audio_settings import AudioSettingsDialog
from hakyking.views.main_window import MainWindow
from hakyking.views.workspace import AudioSliceGraphicsItem


class MainController(QObject):
    """Coordinates models and views without putting business logic in widgets."""

    PREPARSE_MAX_DURATION_SECONDS = 20.0
    PREPARSE_DRAIN_DELAY_MS = 350
    TRACK_FINE_PARSE_MAX_DURATION_SECONDS = 45.0
    MAX_WAVEFORM_WORKERS = 1
    MAX_RENDER_WORKERS = 1
    WAVEFORM_PREVIEW_POINTS = 128
    WAVEFORM_RESULT_CACHE_SIZE = 256
    RENDER_RESULT_CACHE_MAX_BYTES = 128 * 1024 * 1024
    RENDER_RESULT_CACHE_MAX_ITEM_BYTES = 16 * 1024 * 1024
    MAX_SELECTED_WAVEFORM_PREFETCH = 16
    WORKSPACE_INSERT_BATCH_SIZE = 64

    def __init__(self, project: ProjectModel, main_window: MainWindow) -> None:
        super().__init__()
        self.project = project
        self.main_window = main_window
        self._probe_threads: list[QThread] = []
        self._probe_workers: list[AudioProbeWorker] = []
        self._bgm_probe_threads: list[QThread] = []
        self._bgm_probe_workers: list[AudioProbeWorker] = []
        self._folder_scan_threads: list[QThread] = []
        self._folder_scan_workers: list[FolderMediaScanWorker] = []
        self._parse_threads: list[QThread] = []
        self._parse_workers: list[ParseWorker] = []
        self._parse_inflight: set[str] = set()
        self._parse_generations: dict[str, int] = {}
        self._parse_worker_context: dict[ParseWorker, tuple[str, int, bool, bool]] = {}
        self._slice_cache: dict[str, list[AudioSlice]] = {}
        self._workspace_drop_requests: dict[str, list[dict[str, object]]] = {}
        self._workspace_insert_generation = 0
        self._workspace_insert_active: set[int] = set()
        self._track_parse_requests: dict[str, set[int]] = {}
        self._preparse_queue: list[str] = []
        self._preparse_enqueued: set[str] = set()
        self._expanded_preparse_folders: set[str] = set()
        self._slice_preview_threads: list[QThread] = []
        self._slice_preview_workers: list[SlicePreviewWorker] = []
        self._slice_sequence_preview_workers: list[SliceSequencePreviewWorker] = []
        self._file_preview_threads: list[QThread] = []
        self._file_preview_workers: list[FilePreviewWorker] = []
        self._whole_slice_threads: list[QThread] = []
        self._whole_slice_workers: list[WholeSliceWorker] = []
        self._whole_slice_cache: dict[str, AudioSlice] = {}
        self._audio_info_cache: dict[str, AudioInfo] = {}
        self._render_threads: list[QThread] = []
        self._render_workers: list[RenderWorker] = []
        self._render_targets: dict[str, list[AudioSliceGraphicsItem]] = {}
        self._render_queue: list[tuple[object, ...]] = []
        self._render_queued_keys: set[str] = set()
        self._render_active_keys: set[str] = set()
        self._render_result_cache: OrderedDict[str, object] = OrderedDict()
        self._render_result_cache_bytes = 0
        self._waveform_threads: list[QThread] = []
        self._waveform_workers: list[WaveformWorker] = []
        self._waveform_targets: dict[str, list[AudioSliceGraphicsItem]] = {}
        self._waveform_queue: list[tuple[str, AudioSlice]] = []
        self._waveform_queued_keys: set[str] = set()
        self._waveform_active_keys: set[str] = set()
        self._waveform_result_cache: OrderedDict[str, object] = OrderedDict()
        self._clipboard_snapshots: list[dict[str, object]] = []
        self._pitch_curve_clipboard: list[dict[str, float]] = []
        self._pending_preview_groups: list[list[AudioSliceGraphicsItem]] = []
        self._pending_global_playback_start_time: float | None = None
        self._pending_global_playback_resume = False
        self._timeline_playback_return_time = 0.0
        self._timeline_playback_paused = False
        self._pending_export_path: str | None = None
        self._export_thread: QThread | None = None
        self._export_worker: ExportWorker | None = None
        self.project_manager = ProjectManager()
        self.current_project_path: str | None = None
        self.autosave_path = user_data_root() / "autosave" / "hakyking_autosave.haky"
        self.playback_manager = PlaybackManager()
        self.audio_settings = load_audio_settings()
        self._apply_pitch_engine_setting(self.audio_settings.pitch_engine)
        self.playback_manager.configure(
            output_device_index=self.audio_settings.output_device_index,
            blocksize=self.audio_settings.blocksize,
            fade_ms=self.audio_settings.fade_ms,
        )
        self._playhead_timer = QTimer(self)
        self._playhead_timer.setInterval(30)
        self._playhead_timer.timeout.connect(self._sync_playhead_from_playback)
        self._material_preview_timer = QTimer(self)
        self._material_preview_timer.setInterval(30)
        self._material_preview_timer.timeout.connect(self._sync_material_preview_position)
        self._material_preview_started_at = 0.0
        self._material_preview_start_time = 0.0
        self._material_preview_duration = 0.0
        self._preparse_drain_timer = QTimer(self)
        self._preparse_drain_timer.setSingleShot(True)
        self._preparse_drain_timer.timeout.connect(self._drain_preparse_queue)
        self._visible_waveform_timer = QTimer(self)
        self._visible_waveform_timer.setSingleShot(True)
        self._visible_waveform_timer.setInterval(120)
        self._visible_waveform_timer.timeout.connect(self._queue_visible_waveforms)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(60_000)
        self._autosave_timer.timeout.connect(self._autosave_project)
        self._is_shutting_down = False

    def initialize(self) -> None:
        self.project.bootstrap_default_tracks()
        self._sync_project_tracks_to_views()

        self._show_status(
            "就绪。音频读写、分析、渲染都会走后台线程。",
            "Ready. Audio I/O and analysis will be handled by QThread workers.",
        )

        self.main_window.track_control_panel.track_selected.connect(
            self._on_track_selected
        )
        self.main_window.track_control_panel.track_solo_changed.connect(
            self._on_track_solo_changed
        )
        self.main_window.track_control_panel.track_lock_changed.connect(
            self._on_track_lock_changed
        )
        self.main_window.track_control_panel.track_mute_changed.connect(
            self._on_track_mute_changed
        )
        self.main_window.track_control_panel.track_add_requested.connect(
            self._on_track_add_requested
        )
        self.main_window.track_control_panel.track_audio_file_dropped.connect(
            self._on_track_audio_file_dropped
        )
        self.main_window.track_control_panel.track_clip_delete_requested.connect(
            self._on_track_clip_delete_requested
        )
        self.main_window.track_control_panel.track_clip_moved.connect(
            self._on_track_clip_moved
        )
        self.main_window.track_control_panel.track_clip_editable_changed.connect(
            self._on_track_clip_editable_changed
        )
        self.main_window.material_browser.audio_file_selected.connect(
            self._on_material_file_selected
        )
        self.main_window.material_browser.audio_file_parse_requested.connect(
            self._on_material_file_parse_requested
        )
        self.main_window.material_browser.audio_file_preview_requested.connect(
            self._on_material_file_preview_requested
        )
        self.main_window.material_browser.folder_added.connect(
            self._on_material_folder_added
        )
        self.main_window.material_browser.folder_expanded.connect(
            self._on_material_folder_expanded
        )
        self.main_window.material_browser.auto_slice_toggled.connect(
            self._on_auto_slice_toggled
        )
        self.main_window.material_browser.slice_preview_requested.connect(
            self._on_material_slice_preview_requested
        )
        self.main_window.material_browser.slice_sequence_preview_requested.connect(
            self._on_material_slice_sequence_preview_requested
        )
        self.main_window.material_browser.material_slices_changed.connect(
            self._on_material_slices_changed
        )
        self.main_window.workspace.slice_items_created.connect(
            self._on_slice_items_created
        )
        self.main_window.workspace.horizontalScrollBar().valueChanged.connect(
            self._schedule_visible_waveforms
        )
        self.main_window.workspace.verticalScrollBar().valueChanged.connect(
            self._schedule_visible_waveforms
        )
        self.main_window.workspace.pitch_curve_view_changed.connect(
            self._on_pitch_curve_view_changed
        )
        self.main_window.workspace.slice_items_dropped.connect(
            self._on_slice_items_dropped
        )
        self.main_window.workspace.preview_requested.connect(
            self._on_preview_requested
        )
        self.main_window.workspace.render_requested.connect(
            self.render_slice_item
        )
        self.main_window.workspace.global_playback_toggled.connect(
            self._toggle_global_playback
        )
        self.main_window.workspace.playhead_seek_requested.connect(
            self._on_playhead_seek_requested
        )
        self.main_window.workspace.slice_edit_finished.connect(
            self._on_slice_edit_finished
        )
        self.main_window.workspace.slice_edits_finished.connect(
            self._on_slice_edits_finished
        )
        self.main_window.workspace.slice_parameter_changed.connect(
            self._on_slice_parameter_changed
        )
        self.main_window.workspace.slice_boundary_changed.connect(
            self._on_slice_boundary_changed
        )
        self.main_window.workspace.slices_merged.connect(
            self._on_slices_merged
        )
        self.main_window.workspace.split_requested.connect(
            self._on_split_requested
        )
        self.main_window.workspace.delete_requested.connect(
            self._on_delete_requested
        )
        self.main_window.workspace.bgm_file_dropped.connect(
            self._on_bgm_file_dropped
        )
        self.main_window.workspace.audio_file_dropped.connect(
            self._on_audio_file_dropped_as_slice
        )
        self.main_window.workspace.scene().selectionChanged.connect(
            self._on_workspace_selection_changed
        )
        self.main_window.inspector_widget.parameter_change_committed.connect(
            self._on_inspector_parameter_change_committed
        )
        self.main_window.inspector_widget.transient_protection_toggled.connect(
            self._on_transient_protection_requested
        )
        self.main_window.inspector_widget.tuning_preset_requested.connect(
            self._on_tuning_preset_requested
        )
        self.main_window.root_note_combo.currentTextChanged.connect(
            self._on_scale_changed
        )
        self.main_window.scale_type_combo.currentTextChanged.connect(
            self._on_scale_changed
        )
        self.main_window.tool_action_group.triggered.connect(self._on_tool_action_triggered)
        self.main_window.scissors_merge_action.toggled.connect(
            self._on_scissors_merge_toggled
        )
        self.main_window.transient_protection_requested.connect(
            self._on_transient_protection_requested
        )
        self.main_window.delete_selected_action.triggered.connect(
            self._delete_selected_slices
        )
        self.main_window.select_all_action.triggered.connect(self._select_all_slices)
        self.main_window.copy_action.triggered.connect(self._copy_selected_slices)
        self.main_window.paste_action.triggered.connect(self._paste_slices)
        self.main_window.duplicate_action.triggered.connect(self._duplicate_selected_slices)
        self.main_window.copy_pitch_curve_action.triggered.connect(self._copy_pitch_curve)
        self.main_window.paste_pitch_curve_action.triggered.connect(self._paste_pitch_curve)
        self.main_window.smooth_pitch_curve_action.triggered.connect(
            self._smooth_selected_pitch_curves
        )
        self.main_window.pitch_zero_action.triggered.connect(self._zero_selected_pitch_curves)
        self.main_window.pitch_natural_action.triggered.connect(
            self._naturalize_selected_pitch_curves
        )
        self.main_window.pitch_electro_action.triggered.connect(
            self._electro_selected_pitch_curves
        )
        self.main_window.flatten_slider.valueChanged.connect(
            self._on_flatten_slider_changed
        )
        self.main_window.flatten_slider.sliderReleased.connect(
            self._render_selected_effect_items
        )
        self.main_window.formant_slider.valueChanged.connect(
            self._on_formant_slider_changed
        )
        self.main_window.formant_slider.sliderReleased.connect(
            self._render_selected_effect_items
        )
        self.main_window.new_project_action.triggered.connect(self._new_project)
        self.main_window.open_project_action.triggered.connect(self._open_project_dialog)
        self.main_window.open_autosave_action.triggered.connect(self._open_autosave_project)
        self.main_window.save_project_action.triggered.connect(self._save_project)
        self.main_window.save_project_as_action.triggered.connect(self._save_project_as)
        self.main_window.export_action.triggered.connect(self._choose_export_path)
        self.main_window.audio_settings_action.triggered.connect(
            self._open_audio_settings_dialog
        )
        self.main_window.zoom_in_action.triggered.connect(self._zoom_timeline_in)
        self.main_window.zoom_out_action.triggered.connect(self._zoom_timeline_out)
        self.main_window.zoom_reset_action.triggered.connect(self._zoom_timeline_reset)
        self.main_window.play_pause_action.triggered.connect(self._toggle_global_playback)
        self.main_window.stop_action.triggered.connect(self._stop_all_playback)
        self.main_window.return_start_action.triggered.connect(self._return_playhead_to_start)
        self.main_window.about_to_close.connect(self.shutdown)
        self._autosave_timer.start()
        self._on_scale_changed()
        self._on_workspace_selection_changed()

    def _on_tool_action_triggered(self, action) -> None:
        mode = action.data() if hasattr(action, "data") else "select"
        mode = str(mode or "select")
        if mode in {"curve_select", "curve_point", "curve_vibrato"}:
            self.main_window.workspace.set_pitch_curve_tool_mode(mode)
            names = {
                "curve_select": ("音高控制点工具", "Pitch control point tool"),
                "curve_point": ("添加控制点工具", "Add pitch control point tool"),
                "curve_vibrato": ("颤音工具", "Vibrato tool"),
            }
            zh, en = names[mode]
            self._show_status(zh, en)
            return
        self.main_window.workspace.set_pitch_curve_tool_mode("none")
        self.main_window.workspace.set_tool_mode(mode)
        names = {
            "select": ("选择工具", "Select tool"),
            "scissors": ("分割工具", "Split tool"),
            "amplitude": ("增益工具", "Gain tool"),
            "flatten": ("颤音展平工具", "Pitch flatten tool"),
            "formant": ("共振峰工具", "Formant tool"),
        }
        zh, en = names.get(mode, names["select"])
        self._show_status(zh, en)

    def _on_scissors_merge_toggled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled:
            self.main_window.scissors_tool_action.setChecked(True)
            self.main_window.workspace.set_tool_mode("scissors")
        self.main_window.workspace.set_scissors_merge_enabled(enabled)
        self.main_window.update_scissors_merge_ui()
        self._show_status(
            f"片段贴合 / 合并：{'开启' if enabled else '关闭'}",
            f"Cut merge {'enabled' if enabled else 'disabled'}.",
        )

    def _on_transient_protection_requested(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self.main_window.workspace.set_default_transient_protection(enabled)
        self.main_window.set_transient_protection_enabled(enabled)
        changes: list[
            tuple[AudioSliceGraphicsItem, dict[str, object], dict[str, object]]
        ] = []
        for item in self._selected_effect_editable_items():
            before = item.edit_state()
            item.set_transient_protection(enabled)
            after = item.edit_state()
            if before != after:
                changes.append((item, before, after))
        if changes:
            self.main_window.undo_stack.push(
                MoveSlicesCommand(
                    changes=changes,
                    render_callback=self._render_item_if_required,
                    post_callback=self._after_timeline_item_command,
                    initially_applied=True,
                    text="Change Consonant Protection",
                )
            )
        self._show_status(
            f"瞬态保护：{'开启' if enabled else '关闭'}（已应用到 {len(changes)} 个所选片段）",
            f"Transient protection {'on' if enabled else 'off'} "
            f"({len(changes)} selected slice(s) updated).",
        )

    def _on_flatten_slider_changed(self, value: int) -> None:
        self.main_window.flatten_tool_action.setChecked(True)
        self.main_window.workspace.set_tool_mode("flatten")
        amount = max(0.0, min(1.0, float(value) / 100.0))
        items = self._selected_effect_editable_items()
        for item in items:
            item.set_pitch_flatten_amount(amount)
        self._show_status(
            f"颤音展平：{amount * 100:.0f}%（{len(items)} 个片段）",
            f"Pitch flatten: {amount * 100:.0f}% on {len(items)} clip(s).",
        )

    def _on_formant_slider_changed(self, value: int) -> None:
        self.main_window.formant_tool_action.setChecked(True)
        self.main_window.workspace.set_tool_mode("formant")
        semitones = max(-12.0, min(12.0, float(value) / 10.0))
        items = self._selected_effect_editable_items()
        for item in items:
            item.set_formant_shift(semitones)
        self._show_status(
            f"共振峰偏移：{semitones:+.1f}（{len(items)} 个片段）",
            f"Formant shift: {semitones:+.1f} on {len(items)} clip(s).",
        )

    def _render_selected_effect_items(self) -> None:
        for item in self._selected_effect_editable_items():
            self.render_slice_item(item)

    def _selected_effect_editable_items(self) -> list[AudioSliceGraphicsItem]:
        return [
            item
            for item in self.main_window.workspace.selected_slice_items()
            if not item.is_missing_source
            and not item.is_locked
            and item.track_type != "master_bgm"
        ]

    def _copy_pitch_curve(self) -> None:
        items = self._selected_effect_editable_items()
        if not items:
            self._show_status("请先选中一个音符片段。", "Select a clip first.")
            return
        self._pitch_curve_clipboard = items[0].pitch_curve_clipboard_payload()
        self._show_status(
            f"已复制音高曲线控制点：{len(self._pitch_curve_clipboard)} 个",
            f"Copied {len(self._pitch_curve_clipboard)} pitch control point(s).",
        )

    def _paste_pitch_curve(self) -> None:
        if not self._pitch_curve_clipboard:
            self._show_status("没有可粘贴的音高曲线。", "No pitch curve to paste.")
            return
        self._apply_pitch_curve_mutation(
            "Paste Pitch Line",
            lambda item: item.set_pitch_control_points(self._pitch_curve_clipboard),
            "已粘贴音高曲线",
            "Pasted pitch curve",
        )

    def _smooth_selected_pitch_curves(self) -> None:
        self._apply_pitch_curve_mutation(
            "Smooth Pitch Line",
            lambda item: item.apply_pitch_curve_smooth_glide(),
            "已自动平滑滑音",
            "Smoothed pitch glide",
        )

    def _add_vibrato_to_selected_pitch_curves(self) -> None:
        self._apply_pitch_curve_mutation(
            "Add Vibrato",
            lambda item: item.apply_pitch_curve_vibrato(),
            "已添加颤音线",
            "Added vibrato",
        )

    def _zero_selected_pitch_curves(self) -> None:
        self._apply_pitch_curve_mutation(
            "Zero Pitch Line",
            lambda item: item.apply_pitch_curve_zero(),
            "音高曲线已归零",
            "Pitch curve zeroed",
        )

    def _naturalize_selected_pitch_curves(self) -> None:
        self._apply_pitch_curve_mutation(
            "Naturalize Pitch Line",
            lambda item: item.apply_pitch_curve_natural(),
            "已恢复自然音高曲线",
            "Natural pitch curve restored",
        )

    def _electro_selected_pitch_curves(self) -> None:
        self._apply_pitch_curve_mutation(
            "Electro Pitch Line",
            lambda item: item.apply_pitch_curve_electro(),
            "已电音化音高曲线",
            "Electro pitch curve applied",
        )

    def _apply_pitch_curve_mutation(
        self,
        command_text: str,
        mutator,
        zh_message: str,
        en_message: str,
    ) -> None:
        items = self._selected_effect_editable_items()
        if not items:
            self._show_status("请先选中可编辑音符片段。", "Select editable clip(s) first.")
            return
        changes: list[
            tuple[AudioSliceGraphicsItem, dict[str, object], dict[str, object]]
        ] = []
        for item in items:
            before = item.edit_state()
            mutator(item)
            after = item.edit_state()
            if before != after:
                changes.append((item, before, after))
        if not changes:
            self._show_status("音高曲线没有变化。", "Pitch curve unchanged.")
            return
        self.main_window.undo_stack.push(
            MoveSlicesCommand(
                changes=changes,
                render_callback=self._render_item_if_required,
                post_callback=self._after_timeline_item_command,
                initially_applied=True,
                text=command_text,
            )
        )
        self.main_window.workspace.set_pitch_curve_edit_mode(True)
        self.main_window.inspector_widget.set_pitch_curve_view_enabled(True)
        self.main_window.inspector_widget.set_item(changes[0][0], len(items))
        self._show_status(
            f"{zh_message}：{len(changes)} 个音符片段",
            f"{en_message}: {len(changes)} clip(s).",
        )

    def _on_workspace_selection_changed(self) -> None:
        selected_items = self.main_window.workspace.selected_slice_items()
        item = selected_items[0] if selected_items else None
        self.main_window.inspector_widget.set_item(item, len(selected_items))
        if self.main_window.workspace.pitch_curve_edit_mode:
            for selected_item in self._selected_waveform_prefetch_items(selected_items):
                if not selected_item.is_missing_source:
                    self._start_waveform_worker(selected_item)

    def _on_tuning_preset_requested(self, preset_name: str) -> None:
        presets: dict[str, tuple[str, str, dict[str, object]]] = {
            "reset": (
                "调音参数已归零",
                "Tuning parameters reset.",
                {
                    "pitch_flatten_amount": 0.0,
                    "formant_shift": 0.0,
                    "gain_db": 0.0,
                    "protect_transients": True,
                },
            ),
            "natural": (
                "已应用自然修音",
                "Natural tuning preset applied.",
                {
                    "pitch_flatten_amount": 0.25,
                    "formant_shift": 0.0,
                    "protect_transients": True,
                },
            ),
            "flat": (
                "已应用电音压平",
                "Flat-tune preset applied.",
                {
                    "pitch_flatten_amount": 1.0,
                    "formant_shift": 0.0,
                    "protect_transients": True,
                },
            ),
            "bright": (
                "已应用明亮音色",
                "Bright tone preset applied.",
                {
                    "pitch_flatten_amount": 0.45,
                    "formant_shift": 3.0,
                    "protect_transients": True,
                },
            ),
            "deep": (
                "已应用厚声音色",
                "Deep tone preset applied.",
                {
                    "pitch_flatten_amount": 0.2,
                    "formant_shift": -3.0,
                    "protect_transients": True,
                },
            ),
        }
        zh_message, en_message, values = presets.get(str(preset_name), presets["natural"])
        items = self._selected_effect_editable_items()
        if not items:
            self._show_status("请先选中可调音符片段。", "Select editable clip(s) first.")
            return

        changes: list[
            tuple[AudioSliceGraphicsItem, dict[str, object], dict[str, object]]
        ] = []
        root_note = self.main_window.root_note_combo.currentText()
        scale_type = self.main_window.current_scale_type()
        for item in items:
            before = item.edit_state()
            after = dict(before)
            for key, value in values.items():
                after[key] = value
            if preset_name == "reset":
                self._reset_pitch_center_in_state(item, after)
            elif preset_name == "natural":
                self._snap_pitch_center_in_state(item, after, root_note, scale_type)
            if before != after:
                changes.append((item, before, after))
        if not changes:
            self._show_status("调音参数没有变化。", "Tuning parameters unchanged.")
            return

        self.main_window.undo_stack.push(
            MoveSlicesCommand(
                changes=changes,
                render_callback=self._render_item_if_required,
                post_callback=self._after_timeline_item_command,
                initially_applied=False,
                text="Apply Tuning Preset",
            )
        )
        self.main_window.inspector_widget.set_item(items[0], len(items))
        self._show_status(
            f"{zh_message}（{len(changes)} 个音符片段）",
            f"{en_message} ({len(changes)} clip(s))",
        )

    def _reset_pitch_center_in_state(
        self,
        item: AudioSliceGraphicsItem,
        state: dict[str, object],
    ) -> None:
        if item.audio_slice.midi_note is None:
            return
        target = max(0, min(127, int(item.audio_slice.midi_note)))
        state["target_midi_note"] = target
        state["y"] = self.main_window.workspace.y_for_midi_note(
            target,
            fallback_y=float(state.get("y", item.scenePos().y())),
        )

    def _snap_pitch_center_in_state(
        self,
        item: AudioSliceGraphicsItem,
        state: dict[str, object],
        root_note: str,
        scale_type: str,
    ) -> None:
        source_note = state.get("target_midi_note")
        if source_note is None:
            source_note = item.target_midi_note
        if source_note is None:
            source_note = item.audio_slice.midi_note
        if source_note is None:
            return
        target = nearest_midi_in_scale(int(round(float(source_note))), root_note, scale_type)
        state["target_midi_note"] = target
        state["y"] = self.main_window.workspace.y_for_midi_note(
            target,
            fallback_y=float(state.get("y", item.scenePos().y())),
        )

    def _on_scale_changed(self, *args) -> None:
        root_note = self.main_window.root_note_combo.currentText()
        scale_type = self.main_window.current_scale_type()
        self.main_window.piano_roll.set_scale(root_note, scale_type)
        self.main_window.workspace.set_scale(root_note, scale_type)
        self._show_status(
            f"音阶：{root_note} {scale_type}",
            f"Scale: {root_note} {scale_type}",
        )

    def _on_slice_edit_finished(
        self,
        item: AudioSliceGraphicsItem,
        before: object,
        after: object,
    ) -> None:
        before_state = dict(before)
        after_state = dict(after)
        if before_state == after_state:
            return
        self.main_window.undo_stack.push(
            MoveSliceCommand(
                item=item,
                before=before_state,
                after=after_state,
                render_callback=self._render_item_if_required,
                initially_applied=True,
            )
        )
        self.main_window.refresh_timeline_transport_duration()
        self.main_window.inspector_widget.set_item(item)

    def _on_slice_edits_finished(self, payload: object) -> None:
        if not isinstance(payload, list):
            return
        changes: list[
            tuple[AudioSliceGraphicsItem, dict[str, object], dict[str, object]]
        ] = []
        for entry in payload:
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                continue
            item, before, after = entry
            if (
                not isinstance(item, AudioSliceGraphicsItem)
                or not isinstance(before, dict)
                or not isinstance(after, dict)
                or before == after
            ):
                continue
            changes.append((item, dict(before), dict(after)))
        if not changes:
            return
        self.main_window.undo_stack.push(
            MoveSlicesCommand(
                changes=changes,
                render_callback=self._render_item_if_required,
                post_callback=self._after_timeline_item_command,
                initially_applied=True,
            )
        )
        self.main_window.inspector_widget.set_item(changes[0][0])

    def _on_slice_parameter_changed(
        self,
        item: AudioSliceGraphicsItem,
        before: object,
        after: object,
    ) -> None:
        before_state = dict(before)
        after_state = dict(after)
        if before_state == after_state:
            return
        self.main_window.undo_stack.push(
            ChangeParameterCommand(
                item=item,
                before=before_state,
                after=after_state,
                render_callback=self._render_item_if_required,
                initially_applied=True,
            )
        )
        self.main_window.inspector_widget.set_item(item)

    def _on_slice_boundary_changed(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        items = [
            item
            for item in payload.get("items", [])
            if isinstance(item, AudioSliceGraphicsItem)
            and item.scene() is self.main_window.workspace.scene()
        ]
        before = payload.get("before")
        after = payload.get("after")
        if (
            len(items) != 2
            or not isinstance(before, list)
            or not isinstance(after, list)
            or before == after
        ):
            return
        self.main_window.undo_stack.push(
            BoundaryMoveCommand(
                workspace=self.main_window.workspace,
                items=items,
                before_snapshots=before,
                after_snapshots=after,
                render_callback=self._render_item_if_required,
                post_callback=self._after_timeline_item_command,
                initially_applied=True,
            )
        )
        for item in items:
            self._start_waveform_worker(item)
        self.main_window.inspector_widget.set_item(items[0])

    def _on_slices_merged(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        items = [
            item
            for item in payload.get("items", [])
            if isinstance(item, AudioSliceGraphicsItem)
            and item.scene() is self.main_window.workspace.scene()
        ]
        before = payload.get("before")
        after = payload.get("after")
        if (
            len(items) != 1
            or not isinstance(before, list)
            or not isinstance(after, list)
            or before == after
        ):
            return
        self.main_window.undo_stack.push(
            MergeSlicesCommand(
                workspace=self.main_window.workspace,
                items=items,
                before_snapshots=before,
                after_snapshots=after,
                render_callback=self._render_item_if_required,
                post_callback=self._after_timeline_item_command,
                initially_applied=True,
            )
        )
        for item in items:
            self._start_waveform_worker(item)
        self.main_window.inspector_widget.set_item(items[0])

    def _on_inspector_parameter_change_committed(
        self,
        item: AudioSliceGraphicsItem,
        before: object,
        after: object,
    ) -> None:
        before_state = dict(before)
        after_state = dict(after)
        if before_state == after_state:
            return
        self.main_window.undo_stack.push(
            ChangeParameterCommand(
                item=item,
                before=before_state,
                after=after_state,
                render_callback=self._render_item_if_required,
                initially_applied=False,
            )
        )
        self.main_window.inspector_widget.set_item(item)

    def _on_split_requested(self, item: AudioSliceGraphicsItem, local_x: float) -> None:
        self.main_window.undo_stack.push(
            SplitSliceCommand(
                workspace=self.main_window.workspace,
                item=item,
                local_x=local_x,
                render_callback=self._render_item_if_required,
            )
        )

    def _on_delete_requested(self, payload: object) -> None:
        if not isinstance(payload, (list, tuple)):
            return
        items = [
            item
            for item in payload
            if isinstance(item, AudioSliceGraphicsItem)
            and item.scene() is self.main_window.workspace.scene()
        ]
        if not items:
            return
        self.main_window.undo_stack.push(
            DeleteSliceCommand(
                workspace=self.main_window.workspace,
                items=items,
            )
        )
        self.main_window.inspector_widget.set_item(None)
        self.main_window.refresh_timeline_transport_duration()
        self._show_status(f"已删除 {len(items)} 个片段。", f"Deleted {len(items)} clip(s).")

    def _delete_selected_slices(self) -> None:
        self._on_delete_requested(self.main_window.workspace.selected_slice_items())

    def _select_all_slices(self) -> None:
        items = [
            item
            for item in self.main_window.workspace.slice_items()
            if item.flags() & QGraphicsItem.ItemIsSelectable
        ]
        for item in items:
            item.setSelected(True)
        self._show_status(f"已选中 {len(items)} 个片段。", f"Selected {len(items)} clip(s).")

    def _copy_selected_slices(self) -> None:
        snapshots = self._selected_item_snapshots()
        self._clipboard_snapshots = snapshots
        self._show_status(
            f"已复制 {len(snapshots)} 个片段。",
            f"Copied {len(snapshots)} clip(s).",
        )

    def _paste_slices(self) -> None:
        if not self._clipboard_snapshots:
            self._show_status("剪贴板里没有片段。", "No clips in the clipboard.")
            return
        snapshots = self._offset_snapshots(self._clipboard_snapshots, dx=80.0, dy=0.0)
        self._push_paste_command(snapshots, "粘贴", "Paste", "Paste Slice")
        self._clipboard_snapshots = snapshots

    def _duplicate_selected_slices(self) -> None:
        snapshots = self._selected_item_snapshots()
        if not snapshots:
            self._show_status("没有选中的片段。", "No selected clips.")
            return
        self._clipboard_snapshots = snapshots
        snapshots = self._offset_snapshots(snapshots, dx=80.0, dy=0.0)
        self._push_paste_command(snapshots, "复制一份", "Duplicate", "Duplicate Slice")
        self._clipboard_snapshots = snapshots

    def _selected_item_snapshots(self) -> list[dict[str, object]]:
        items = sorted(
            self.main_window.workspace.selected_slice_items(),
            key=lambda item: (item.scenePos().x(), item.scenePos().y()),
        )
        return [self.main_window.workspace.snapshot_item(item) for item in items]

    def _offset_snapshots(
        self,
        snapshots: list[dict[str, object]],
        dx: float,
        dy: float,
    ) -> list[dict[str, object]]:
        return [
            {
                **snapshot,
                "x": float(snapshot.get("x", 0.0)) + dx,
                "y": float(snapshot.get("y", 0.0)) + dy,
            }
            for snapshot in snapshots
        ]

    def _push_paste_command(
        self,
        snapshots: list[dict[str, object]],
        zh_action: str,
        en_action: str,
        command_text: str,
    ) -> None:
        if not snapshots:
            return
        for item in self.main_window.workspace.selected_slice_items():
            item.setSelected(False)
        self.main_window.undo_stack.push(
            PasteSliceCommand(
                workspace=self.main_window.workspace,
                snapshots=snapshots,
                render_callback=self._render_item_if_required,
                text=command_text,
            )
        )
        self._show_status(
            f"已{zh_action} {len(snapshots)} 个片段。",
            f"{en_action}d {len(snapshots)} clip(s).",
        )

    def _render_item_if_required(self, item: AudioSliceGraphicsItem) -> None:
        if item.scene() is None or item.is_missing_source:
            return
        if self._item_requires_render(item):
            self.render_slice_item(item)

    def _sync_project_tracks_to_views(self) -> None:
        self.main_window.workspace.clear_track_bindings()
        selected_index = self.project.selected_track_index
        if selected_index is None:
            selected_index = 0
            self.project.selected_track_index = selected_index
        selected_index = max(0, min(int(selected_index), max(0, len(self.project.tracks) - 1)))
        self.project.selected_track_index = selected_index
        self.main_window.track_control_panel.set_tracks(self.project.tracks, selected_index)
        self.main_window.workspace.set_active_track_index(selected_index)
        for index, track in enumerate(self.project.tracks):
            self.main_window.workspace.set_track_type(index, track.track_type.value)
            self.main_window.workspace.set_track_locked(index, track.locked)
            if track.clip_path:
                self.main_window.workspace.set_source_timeline_offset(
                    track.clip_path,
                    track.clip_start,
                )

    def _new_project(self) -> None:
        self._workspace_insert_active.clear()
        self._reset_timeline_runtime()
        self.project = ProjectModel()
        self.project.bootstrap_default_tracks()
        self._slice_cache.clear()
        self._whole_slice_cache.clear()
        self.current_project_path = None
        self.main_window.workspace.clear_slice_items()
        self.main_window.material_browser.clear_folders()
        self.main_window.workspace.set_playhead_time(0.0)
        self._sync_project_tracks_to_views()
        self.main_window.undo_stack.clear()
        self.main_window.inspector_widget.set_item(None)
        self._show_status("已新建 Hakyking 工程。", "New Hakyking project.")

    def _open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            self._ui_text("打开 Hakyking 工程", "Open Hakyking Project"),
            str(user_data_root()),
            "Hakyking projects (*.haky);;JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return
        self._load_project_path(path)

    def _open_autosave_project(self) -> None:
        if not self.autosave_path.exists():
            QMessageBox.information(
                self.main_window,
                "没有自动保存",
                f"还没有找到自动保存文件:\n{self.autosave_path}",
            )
            return
        self._load_project_path(str(self.autosave_path))

    def load_project_path(self, path: str) -> None:
        self._load_project_path(path)

    def _open_audio_settings_dialog(self) -> None:
        dialog = AudioSettingsDialog(
            parent=self.main_window,
            output_device_index=self.playback_manager.output_device_index,
            blocksize=self.playback_manager.blocksize,
            fade_ms=self.playback_manager.fade_ms,
            pitch_engine=self.audio_settings.pitch_engine,
            language=self.main_window.current_language(),
        )
        accepted = dialog.exec() if hasattr(dialog, "exec") else dialog.exec_()
        if not accepted:
            return
        self.playback_manager.configure(
            output_device_index=dialog.selected_output_device_index(),
            blocksize=dialog.selected_blocksize(),
            fade_ms=dialog.selected_fade_ms(),
        )
        self.audio_settings = AudioPlaybackSettings(
            output_device_index=dialog.selected_output_device_index(),
            blocksize=dialog.selected_blocksize(),
            fade_ms=dialog.selected_fade_ms(),
            pitch_engine=dialog.selected_pitch_engine(),
        )
        self._apply_pitch_engine_setting(self.audio_settings.pitch_engine)
        save_audio_settings(self.audio_settings)
        self.main_window.statusBar().showMessage(
            self._ui_text(
                f"音频设置已更新并保存：{self.playback_manager.settings_summary()} · 变调引擎 {self.audio_settings.pitch_engine}",
                f"Audio settings updated and saved: {self.playback_manager.settings_summary()} · pitch engine {self.audio_settings.pitch_engine}",
            )
        )

    def _apply_pitch_engine_setting(self, pitch_engine: str) -> None:
        os.environ["HAKYKING_PITCH_ENGINE"] = pitch_engine

    def _zoom_timeline_in(self) -> None:
        zoom = self.main_window.workspace.zoom_in()
        self.main_window.timeline_ruler.sync_from_workspace()
        self._show_status(f"时间线缩放：{zoom:.0%}", f"Timeline zoom: {zoom:.0%}")

    def _zoom_timeline_out(self) -> None:
        zoom = self.main_window.workspace.zoom_out()
        self.main_window.timeline_ruler.sync_from_workspace()
        self._show_status(f"时间线缩放：{zoom:.0%}", f"Timeline zoom: {zoom:.0%}")

    def _zoom_timeline_reset(self) -> None:
        zoom = self.main_window.workspace.reset_horizontal_zoom()
        self.main_window.timeline_ruler.sync_from_workspace()
        self._show_status(f"时间线缩放：{zoom:.0%}", f"Timeline zoom: {zoom:.0%}")

    def _save_project(self) -> None:
        if self.current_project_path is None:
            self._save_project_as()
            return
        self._save_project_path(self.current_project_path)

    def _save_project_as(self) -> None:
        default_path = self.current_project_path or str(Path.home() / "untitled.haky")
        path, _ = QFileDialog.getSaveFileName(
            self.main_window,
            self._ui_text("保存 Hakyking 工程", "Save Hakyking Project"),
            default_path,
            "Hakyking projects (*.haky)",
        )
        if not path:
            return
        self._save_project_path(path)

    def _save_project_path(self, path: str) -> None:
        try:
            self.project.material_folders = self.main_window.material_browser.folder_paths()
            output_path = self.project_manager.save(
                path,
                self.project,
                self.main_window.workspace,
            )
        except Exception as exc:  # noqa: BLE001 - file dialogs surface the failure
            QMessageBox.critical(self.main_window, "保存工程失败", str(exc))
            self._show_status(f"保存失败：{exc}", f"Save failed: {exc}")
            return
        self.current_project_path = str(output_path)
        self._show_status(f"工程已保存：{output_path}", f"Project saved: {output_path}")

    def _autosave_project(self) -> None:
        if self._is_shutting_down and self.main_window is None:
            return
        try:
            self.project.material_folders = self.main_window.material_browser.folder_paths()
            output_path = self.project_manager.save(
                self.autosave_path,
                self.project,
                self.main_window.workspace,
            )
        except Exception as exc:  # noqa: BLE001 - autosave should never crash the app
            print(f"Autosave failed: {exc}", flush=True)
            return
        print(f"Autosaved project: {output_path}", flush=True)

    def shutdown(self) -> None:
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        self._autosave_project()
        self._autosave_timer.stop()
        self._playhead_timer.stop()
        self._material_preview_timer.stop()
        self._visible_waveform_timer.stop()
        self._workspace_insert_active.clear()
        for worker in list(self._parse_workers):
            worker.cancel()
        self._reset_timeline_runtime()
        self._wait_for_worker_threads()

    def _wait_for_worker_threads(self, timeout_ms: int = 5000) -> None:
        thread_groups = (
            self._probe_threads,
            self._bgm_probe_threads,
            self._folder_scan_threads,
            self._parse_threads,
            self._slice_preview_threads,
            self._file_preview_threads,
            self._whole_slice_threads,
            self._render_threads,
            self._waveform_threads,
            [self._export_thread] if self._export_thread is not None else [],
        )
        for threads in thread_groups:
            for thread in list(threads):
                if thread is None:
                    continue
                if thread.isRunning():
                    thread.quit()
                    if not thread.wait(timeout_ms):
                        print(
                            f"Worker thread did not stop within {timeout_ms} ms.",
                            flush=True,
                        )

    def _load_project_path(self, path: str) -> None:
        try:
            loaded_project = self.project_manager.load(path)
        except Exception as exc:  # noqa: BLE001 - invalid projects should not crash UI
            QMessageBox.critical(self.main_window, "打开工程失败", str(exc))
            self._show_status(f"打开工程失败：{exc}", f"Open project failed: {exc}")
            return

        self._workspace_insert_active.clear()
        self._reset_timeline_runtime()
        self.project = loaded_project.project
        self._slice_cache.clear()
        self._whole_slice_cache.clear()
        self.current_project_path = path
        self.main_window.workspace.clear_slice_items()
        self.main_window.material_browser.clear_folders()
        for folder_path in list(self.project.material_folders):
            if Path(folder_path).is_dir():
                self.main_window.material_browser.add_folder(folder_path)
        self.main_window.workspace.set_playhead_time(0.0)
        self._sync_project_tracks_to_views()
        self.main_window.undo_stack.clear()
        self.main_window.inspector_widget.set_item(None)

        restored_count = 0
        for restored_slice in loaded_project.slices:
            item = self.main_window.workspace.restore_slice_item(
                audio_slice=restored_slice.audio_slice,
                track_index=restored_slice.track_index,
                x=restored_slice.x,
                y=restored_slice.y,
                width=restored_slice.width,
                height=restored_slice.height,
                target_midi_note=restored_slice.target_midi_note,
                target_duration=restored_slice.target_duration,
                missing_source=restored_slice.missing_source,
                gain_db=restored_slice.gain_db,
                pitch_flatten_amount=restored_slice.pitch_flatten_amount,
                formant_shift=restored_slice.formant_shift,
                protect_transients=restored_slice.protect_transients,
                pitch_control_points=restored_slice.pitch_control_points,
                pitch_vibrato_regions=restored_slice.pitch_vibrato_regions,
                pitch_shape_regions=restored_slice.pitch_shape_regions,
                track_reference=restored_slice.track_reference,
                reference_editable=restored_slice.reference_editable,
                emit_created=not restored_slice.track_reference,
            )
            item.setData(3, restored_slice.n_steps)
            item.setData(4, restored_slice.rate)
            restored_count += 1
            if not restored_slice.missing_source and self._item_requires_render(item):
                self.render_slice_item(item)

        if loaded_project.missing_paths:
            self._show_missing_material_warning(loaded_project.missing_paths)

        self._show_status(
            f"工程已打开：{path} | 片段={restored_count}",
            f"Project opened: {path} | clips={restored_count}",
        )
        if loaded_project.recovered_from:
            self._show_status(
                f"主工程损坏，已从备份恢复：{loaded_project.recovered_from}",
                f"Primary project was damaged; recovered from {loaded_project.recovered_from}",
            )
        elif loaded_project.migrated_from_version is not None:
            self._show_status(
                f"已兼容读取旧版工程 v{loaded_project.migrated_from_version}",
                f"Loaded legacy project v{loaded_project.migrated_from_version}",
            )

    def _show_missing_material_warning(self, paths: list[str]) -> None:
        preview_paths = paths[:8]
        suffix = "" if len(paths) <= 8 else f"\n...and {len(paths) - 8} more"
        QMessageBox.warning(
            self.main_window,
            "部分源媒体丢失",
            "部分源媒体丢失，相关片段已标记为红色警告色：\n"
            + "\n".join(preview_paths)
            + suffix,
        )

    def _reset_timeline_runtime(self) -> None:
        self.playback_manager.stop()
        self.playback_manager.stop_timeline()
        self._playhead_timer.stop()
        self._pending_preview_groups.clear()
        self._pending_global_playback_start_time = None
        self._pending_global_playback_resume = False
        self._timeline_playback_return_time = 0.0
        self._timeline_playback_paused = False
        self._pending_export_path = None
        self._render_targets.clear()
        self._waveform_targets.clear()
        self._render_queue.clear()
        self._render_queued_keys.clear()
        self._waveform_queue.clear()
        self._waveform_queued_keys.clear()

    def _on_track_selected(self, index: int) -> None:
        self.project.selected_track_index = index
        track = self.project.selected_track
        if track is None:
            self._show_status("没有选中的音轨。", "No track selected.")
            return
        self.main_window.workspace.set_active_track_index(index)
        self.main_window.track_control_panel.set_tracks(self.project.tracks, index)
        self._show_status(f"已选择：{track.name}", f"Selected: {track.name}")

    def _on_track_lock_changed(self, index: int, locked: bool) -> None:
        if 0 <= index < len(self.project.tracks):
            track = self.project.tracks[index]
            locked = True if track.track_type.value == "master_bgm" else locked
            self.project.tracks[index].locked = locked
            track_name = self.project.tracks[index].name
        else:
            track_name = f"Track {index + 1}"
        self.main_window.workspace.set_track_locked(index, locked)
        zh_state = "已锁定" if locked else "已解锁"
        en_state = "locked" if locked else "unlocked"
        self._show_status(f"{track_name} {zh_state}。", f"{track_name} {en_state}.")

    def _on_track_solo_changed(self, index: int, solo: bool) -> None:
        if 0 <= index < len(self.project.tracks):
            self.project.tracks[index].solo = solo
            track_name = self.project.tracks[index].name
        else:
            track_name = f"Track {index + 1}"
        zh_state = "已独奏" if solo else "取消独奏"
        en_state = "solo" if solo else "unsolo"
        self._show_status(f"{track_name} {zh_state}。", f"{track_name} {en_state}.")
        if self.playback_manager.is_timeline_playing:
            self._start_global_playback(self.playback_manager.timeline_current_time)

    def _on_track_mute_changed(self, index: int, muted: bool) -> None:
        if 0 <= index < len(self.project.tracks):
            self.project.tracks[index].muted = muted
            track_name = self.project.tracks[index].name
        else:
            track_name = f"Track {index + 1}"
        zh_state = "已静音" if muted else "取消静音"
        en_state = "muted" if muted else "unmuted"
        self._show_status(f"{track_name} {zh_state}。", f"{track_name} {en_state}.")
        if self.playback_manager.is_timeline_playing:
            self._start_global_playback(self.playback_manager.timeline_current_time)

    def _on_track_add_requested(self) -> None:
        index = len(self.project.tracks)
        self.project.tracks.append(
            TrackModel(
                name=f"Track {index + 1}",
                role=TrackRole.AUX if index > 0 else TrackRole.MAIN,
                track_type=TrackType.VOCAL_SLICE,
            )
        )
        self.project.selected_track_index = index
        self._sync_project_tracks_to_views()
        self._show_status(f"已新增音轨 {index + 1}", f"Added track {index + 1}.")

    def _on_track_audio_file_dropped(self, index: int, path: str) -> None:
        if index < 0 or index >= len(self.project.tracks):
            return
        try:
            path = str(Path(path).expanduser().resolve())
        except Exception:
            path = str(path)
        track = self.project.tracks[index]
        track.clip_path = path
        track.clip_start = 0.0
        track.clip_duration = 0.0
        track.clip_editable = False
        self.project.selected_track_index = index
        self.main_window.workspace.set_source_timeline_offset(path, track.clip_start)
        self.main_window.workspace.remove_track_reference_items(index)
        self._sync_project_tracks_to_views()

        cached = self._slice_cache.get(path)
        if cached is not None:
            self._apply_track_parse_result(index, path, cached)
            return

        self._show_status(
            f"音轨 {index + 1} 正在读取音频信息：{path}",
            f"Track {index + 1} probing audio: {path}",
        )
        self._start_track_audio_probe(index, path)

    def _start_track_audio_probe(self, track_index: int, path: str) -> None:
        thread = QThread(self)
        worker = AudioProbeWorker(path)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(
            lambda info, index=track_index: self._on_track_audio_info_ready(index, info)
        )
        worker.failed.connect(
            lambda message, index=track_index, probe_path=path: (
                self._on_track_audio_info_failed(index, probe_path, message)
            )
        )
        worker.completed.connect(thread.quit)
        worker.completed.connect(lambda: self._remove_probe_worker(worker))
        worker.completed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._remove_probe_thread(thread))

        self._probe_threads.append(thread)
        self._probe_workers.append(worker)
        thread.start()

    def _on_track_audio_info_ready(self, track_index: int, info: AudioInfo) -> None:
        if track_index < 0 or track_index >= len(self.project.tracks):
            return
        track = self.project.tracks[track_index]
        if track.clip_path != info.path:
            return
        self._audio_info_cache[info.path] = info
        self._apply_track_full_reference(track_index, info.path, info.duration)

        if info.duration <= self.TRACK_FINE_PARSE_MAX_DURATION_SECONDS:
            self._track_parse_requests.setdefault(info.path, set()).add(track_index)
            self._show_status(
                f"音轨 {track_index + 1} 正在细解析短音频：{info.duration:.1f}s",
                f"Track {track_index + 1} fine-parsing short audio: {info.duration:.1f}s",
            )
            self._start_parse_worker(info.path, display_result=False, preparse=False)
            return

        self._show_status(
            f"音轨 {track_index + 1} 已快速载入。长音频先作为整段参考，避免卡死。",
            f"Track {track_index + 1} loaded as full reference to avoid long parse stalls.",
        )

    def _on_track_audio_info_failed(self, track_index: int, path: str, message: str) -> None:
        if 0 <= track_index < len(self.project.tracks):
            track = self.project.tracks[track_index]
            if track.clip_path == path:
                track.clip_path = ""
                track.clip_duration = 0.0
                track.clip_start = 0.0
                track.clip_editable = False
        self.main_window.workspace.remove_track_reference_items(track_index)
        self.main_window.track_control_panel.set_tracks(
            self.project.tracks,
            self.project.selected_track_index or 0,
        )
        self.main_window.refresh_timeline_transport_duration()
        self._show_status(f"音轨音频读取失败：{message}", f"Track audio probe failed: {message}")

    def _apply_track_full_reference(
        self,
        track_index: int,
        path: str,
        duration: float,
    ) -> None:
        if track_index < 0 or track_index >= len(self.project.tracks):
            return
        track = self.project.tracks[track_index]
        if track.clip_path != path:
            return
        track.clip_duration = max(0.001, float(duration))
        audio_slice = AudioSlice(
            source_path=path,
            index=0,
            start_time=0.0,
            end_time=track.clip_duration,
            midi_note=None,
            f0_hz=None,
        )
        self.main_window.workspace.set_source_timeline_offset(path, track.clip_start)
        self.main_window.workspace.add_track_reference_items(
            [audio_slice],
            track_index=track_index,
            start_time=track.clip_start,
            editable=track.clip_editable,
        )
        self.main_window.track_control_panel.set_tracks(
            self.project.tracks,
            self.project.selected_track_index or 0,
        )
        self.main_window.refresh_timeline_transport_duration()

    def _on_track_clip_delete_requested(self, index: int) -> None:
        if index < 0 or index >= len(self.project.tracks):
            return
        track = self.project.tracks[index]
        if not track.clip_path:
            return
        removed_name = Path(track.clip_path).name
        pending = self._track_parse_requests.get(track.clip_path)
        if pending is not None:
            pending.discard(index)
        track.clip_path = ""
        track.clip_start = 0.0
        track.clip_duration = 0.0
        track.clip_editable = False
        self.main_window.workspace.remove_track_reference_items(index)
        self.main_window.track_control_panel.set_tracks(
            self.project.tracks,
            self.project.selected_track_index or 0,
        )
        self.main_window.refresh_timeline_transport_duration()
        if self.playback_manager.is_timeline_playing:
            self._start_global_playback(self.playback_manager.timeline_current_time)
        self._show_status(
            f"已移除音轨 {index + 1} 的音频：{removed_name}",
            f"Removed audio from track {index + 1}: {removed_name}",
        )

    def _on_track_clip_moved(self, index: int, start_time: float) -> None:
        if index < 0 or index >= len(self.project.tracks):
            return
        track = self.project.tracks[index]
        if not track.clip_path:
            return
        track.clip_start = max(0.0, float(start_time))
        self.main_window.workspace.set_source_timeline_offset(track.clip_path, track.clip_start)
        self.main_window.workspace.move_track_reference_items(index, track.clip_start)
        self.main_window.track_control_panel.refresh_track_state(index, track)
        self.main_window.refresh_timeline_transport_duration()

    def _on_track_clip_editable_changed(self, index: int, editable: bool) -> None:
        if index < 0 or index >= len(self.project.tracks):
            return
        track = self.project.tracks[index]
        track.clip_editable = bool(editable)
        self.main_window.workspace.set_track_reference_editable(index, track.clip_editable)
        self.main_window.track_control_panel.set_tracks(
            self.project.tracks,
            self.project.selected_track_index or 0,
        )
        if track.clip_editable:
            for item in self.main_window.workspace.slice_items():
                if item.track_index == index and item.is_track_reference:
                    self._start_waveform_worker(item)
        self._show_status(
            f"音轨 {index + 1} {'已启用编辑' if editable else '已设为参考'}",
            f"Track {index + 1} {'editable' if editable else 'reference-only'}.",
        )

    def _apply_track_parse_result(
        self,
        track_index: int,
        path: str,
        audio_slices: list[AudioSlice],
    ) -> None:
        if track_index < 0 or track_index >= len(self.project.tracks):
            return
        track = self.project.tracks[track_index]
        if track.clip_path != path:
            return
        audio_slices = list(audio_slices)
        if not audio_slices:
            audio_slices = [build_full_audio_slice(path)]
            self._slice_cache[path] = audio_slices
        self.main_window.workspace.set_source_timeline_offset(path, track.clip_start)
        track.clip_duration = max(
            (audio_slice.end_time for audio_slice in audio_slices),
            default=0.0,
        )
        self.main_window.workspace.add_track_reference_items(
            audio_slices,
            track_index=track_index,
            start_time=track.clip_start,
            editable=track.clip_editable,
        )
        self.main_window.track_control_panel.set_tracks(
            self.project.tracks,
            self.project.selected_track_index or 0,
        )
        self.main_window.refresh_timeline_transport_duration()
        self._show_status(
            f"音轨 {track_index + 1} 已分析 {len(audio_slices)} 个参考片段",
            f"Track {track_index + 1} analyzed {len(audio_slices)} reference clip(s).",
        )

    def _on_material_file_selected(self, path: str) -> None:
        if self.main_window.material_browser.auto_slicing_enabled():
            cached_slices = self._slice_cache.get(path)
            if cached_slices is not None:
                self.main_window.material_browser.set_slices(path, cached_slices)
        else:
            manual_slices = self.project.material_slice_overrides.get(path)
            if manual_slices:
                self.main_window.material_browser.set_slices(path, manual_slices)
            else:
                self._show_whole_material_slice(path)

        cached_info = self._audio_info_cache.get(path)
        if cached_info is not None:
            self.main_window.material_browser.set_material_duration(path, cached_info.duration)
            return

        self._show_status(f"正在分析媒体：{path}", f"Analyzing media: {path}")
        self.main_window.material_browser.set_probe_result(
            self._ui_text(f"探测：{path}", f"Probe: {path}")
        )

        thread = QThread(self)
        worker = AudioProbeWorker(path)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_audio_info_ready)
        worker.failed.connect(self._on_audio_info_failed)
        worker.completed.connect(thread.quit)
        worker.completed.connect(lambda: self._remove_probe_worker(worker))
        worker.completed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._remove_probe_thread(thread))

        self._probe_threads.append(thread)
        self._probe_workers.append(worker)
        thread.start()

    def _on_material_folder_added(self, folder_path: str) -> None:
        normalized_folder = str(Path(folder_path).expanduser().resolve())
        if normalized_folder not in self.project.material_folders:
            self.project.material_folders.append(normalized_folder)
        self._expanded_preparse_folders.discard(str(folder_path))
        if not self.main_window.material_browser.auto_slicing_enabled():
            self._show_status(
                "已添加文件夹。自动分段已关闭，双击媒体将生成整段媒体片段。",
                "Folder added. Auto Slice is off; double-click creates a full clip.",
            )
            return

        self.main_window.material_browser.set_background_parse_status(
            self._ui_text(
                "已添加文件夹 · 展开目录时预解析短音频",
                "Folder added · expand folders to pre-parse short audio",
            )
        )
        self._show_status(
            "已添加文件夹。展开某个目录时，才会预解析这一层的短音频。",
            "Folder added. Short audio is pre-parsed only when a folder is expanded.",
        )
        QTimer.singleShot(0, lambda path=folder_path: self._on_material_folder_expanded(path))

    def _on_material_folder_expanded(self, folder_path: str) -> None:
        if not self.main_window.material_browser.auto_slicing_enabled():
            return
        normalized_path = str(Path(folder_path).expanduser().resolve())
        if normalized_path in self._expanded_preparse_folders:
            return
        self._expanded_preparse_folders.add(normalized_path)

        self.main_window.material_browser.set_background_parse_status(
            self._ui_text(
                f"扫描展开目录：{folder_path}",
                f"Scanning expanded folder: {folder_path}",
            )
        )
        thread = QThread(self)
        worker = FolderMediaScanWorker(folder_path, recursive=False)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_material_folder_scan_ready)
        worker.failed.connect(self._on_material_folder_scan_failed)
        worker.completed.connect(thread.quit)
        worker.completed.connect(lambda: self._remove_folder_scan_worker(worker))
        worker.completed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._remove_folder_scan_thread(thread))

        self._folder_scan_threads.append(thread)
        self._folder_scan_workers.append(worker)
        thread.start()

    def _on_material_folder_scan_ready(self, folder_path: str, paths: object) -> None:
        material_paths = [str(path) for path in paths] if isinstance(paths, list) else []
        queued = 0
        for path in material_paths:
            if (
                path in self._slice_cache
                or path in self._parse_inflight
                or path in self._preparse_enqueued
            ):
                continue
            self._preparse_queue.append(path)
            self._preparse_enqueued.add(path)
            queued += 1

        if queued:
            self.main_window.material_browser.set_background_parse_status(
                self._ui_text(
                    f"展开目录预分析：{queued} 个短媒体",
                    f"Expanded folder pre-parse: {queued} short file(s)",
                )
            )
            self._show_status(
                f"已将展开目录这一层的 {queued} 个媒体文件加入预分析队列。",
                f"Queued {queued} file(s) from the expanded folder for pre-parse.",
            )
        else:
            self._show_status(
                "这个展开目录里没有新的可预解析短音频。",
                "No new short audio files in this expanded folder.",
            )
        self._schedule_preparse_drain()

    def _on_material_folder_scan_failed(self, message: str) -> None:
        print(f"Material folder scan failed: {message}", flush=True)
        self._show_status(f"媒体文件夹扫描失败：{message}", f"Media folder scan failed: {message}")

    def _on_auto_slice_toggled(self, enabled: bool) -> None:
        if enabled:
            self._show_status(
                "自动分段已开启。展开文件夹时才会预分析这一层短音频。",
                "Auto Slice enabled. Expand folders to pre-parse short audio.",
            )
        else:
            for worker, (_path, _generation, _display, preparse) in list(
                self._parse_worker_context.items()
            ):
                if preparse:
                    worker.cancel()
            self._preparse_queue.clear()
            self._preparse_enqueued.clear()
            self._expanded_preparse_folders.clear()
            self._preparse_drain_timer.stop()
            self._show_status(
                "自动分段已关闭。双击媒体会生成整段媒体片段，可拖进时间线后手动分割。",
                "Auto Slice disabled. Double-click creates a full clip for manual splitting.",
            )
        current_path = self.main_window.material_browser.current_file_path()
        if current_path:
            self._on_material_file_parse_requested(current_path)

    def _on_material_file_parse_requested(self, path: str) -> None:
        self._cancel_obsolete_display_parses(path)
        manual_slices = self.project.material_slice_overrides.get(path)
        if not self.main_window.material_browser.auto_slicing_enabled():
            if manual_slices:
                self.main_window.material_browser.set_slices(path, manual_slices)
                self._show_status(
                    f"已恢复 {len(manual_slices)} 个手动分段。",
                    f"Restored {len(manual_slices)} manual material segment(s).",
                )
                return
            self._show_whole_material_slice(path)
            return

        cached = self._slice_cache.get(path)
        if cached is not None:
            self.main_window.material_browser.set_slices(path, cached)
            self._show_status(
                f"已从预分析缓存读取 {len(cached)} 个片段。",
                f"Loaded {len(cached)} clip(s) from pre-analysis cache.",
            )
            return

        if manual_slices:
            # Manual cuts are project data, but automatic mode should show the
            # current automatic analysis instead of the user's saved cut marks.
            self.main_window.material_browser.set_background_parse_status(
                self._ui_text("手动分段已保存；自动分段模式将重新分析", "Manual segments saved; auto mode will analyze again")
            )

        if path in self._parse_inflight:
            self.main_window.material_browser.set_parse_status(
                self._ui_text(f"正在后台预解析：{path}", f"Pre-parsing in background: {path}")
            )
            self._show_status(
                "这个媒体文件正在后台预分析，完成后会自动显示。",
                "This media file is already being pre-analyzed and will appear when ready.",
            )
            return

        if path in self._preparse_queue:
            self._preparse_queue = [
                queued_path for queued_path in self._preparse_queue if queued_path != path
            ]
            self._preparse_enqueued.discard(path)
        self._show_status(f"正在分析片段：{path}", f"Analyzing clips: {path}")
        self.main_window.material_browser.set_parse_status(
            self._ui_text(f"正在分析片段：{path}", f"Analyzing clips: {path}")
        )
        self.main_window.material_browser.set_slices(path, [])
        self._start_parse_worker(path, display_result=True, preparse=False)
        return

    def _on_material_file_preview_requested(self, path: str, start_time: float) -> None:
        if not path:
            self.playback_manager.stop()
            self._material_preview_timer.stop()
            self.main_window.material_browser.set_material_preview_playing(False)
            self._show_status("源媒体试听已停止。", "Source preview stopped.")
            return

        start_time = max(0.0, float(start_time))
        self._show_status(
            f"正在准备源媒体试听：{path} @ {start_time:.2f}s",
            f"Preparing source preview: {path} @ {start_time:.2f}s",
        )
        thread = QThread(self)
        worker = FilePreviewWorker(path, start_time=start_time)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_file_preview_audio_ready)
        worker.failed.connect(self._on_file_preview_failed)
        worker.completed.connect(thread.quit)
        worker.completed.connect(lambda: self._remove_file_preview_worker(worker))
        worker.completed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._remove_file_preview_thread(thread))

        self._file_preview_threads.append(thread)
        self._file_preview_workers.append(worker)
        thread.start()

    def _on_material_slices_changed(self, path: str, slices: object) -> None:
        if not path:
            return
        before_slices: list[AudioSlice] = []
        if isinstance(slices, dict):
            before_slices = [
                audio_slice
                for audio_slice in list(slices.get("before", []))
                if isinstance(audio_slice, AudioSlice)
            ]
            slices = slices.get("after", [])
        audio_slices = [
            audio_slice
            for audio_slice in list(slices)
            if isinstance(audio_slice, AudioSlice)
        ]
        if before_slices == audio_slices:
            return
        # This is an explicit user edit. Keep it separate from automatic
        # pre-parse output, and persist it as part of the project.
        self.project.material_slice_overrides[path] = list(audio_slices)
        self._whole_slice_cache.pop(path, None)
        self.main_window.undo_stack.push(
            MaterialSlicesCommand(
                path=path,
                before=before_slices,
                after=audio_slices,
                apply_callback=self._apply_material_slices_state,
                initially_applied=True,
            )
        )
        self._show_status(
            f"源片段编辑区已手动分成 {len(audio_slices)} 个音符片段。",
            f"Source editor manually split into {len(audio_slices)} clip(s).",
        )

    def _apply_material_slices_state(self, path: str, slices: list[object]) -> None:
        audio_slices = [
            audio_slice
            for audio_slice in slices
            if isinstance(audio_slice, AudioSlice)
        ]
        if audio_slices:
            self.project.material_slice_overrides[path] = list(audio_slices)
        else:
            self.project.material_slice_overrides.pop(path, None)
        self._whole_slice_cache.pop(path, None)
        if (
            path == self.main_window.material_browser.current_file_path()
            and not self.main_window.material_browser.auto_slicing_enabled()
        ):
            self.main_window.material_browser.set_slices(path, audio_slices)

    def _start_parse_worker(
        self,
        path: str,
        display_result: bool,
        preparse: bool,
    ) -> None:
        if path in self._parse_inflight:
            return
        self._parse_inflight.add(path)
        generation = self._parse_generations.get(path, 0) + 1
        self._parse_generations[path] = generation

        thread = QThread(self)
        worker = ParseWorker(
            path,
            max_duration_seconds=self.PREPARSE_MAX_DURATION_SECONDS if preparse else None,
            skip_video=preparse,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(
            lambda ready_path, slices, display=display_result, is_preparse=preparse, token=generation: (
                self._on_audio_slices_ready_if_current(
                    ready_path, slices, display, is_preparse, token
                )
            )
        )
        worker.failed.connect(
            lambda message, failed_path=path, display=display_result, is_preparse=preparse, token=generation: (
                self._on_audio_slices_failed_if_current(
                    message, failed_path, display, is_preparse, token
                )
            )
        )
        worker.skipped.connect(self._on_preparse_skipped)
        worker.completed.connect(thread.quit)
        worker.completed.connect(lambda failed_path=path: self._parse_inflight.discard(failed_path))
        worker.completed.connect(
            lambda completed_path=path, token=generation: (
                self._restart_current_parse_if_superseded(completed_path, token)
            )
        )
        worker.completed.connect(lambda failed_path=path: self._preparse_enqueued.discard(failed_path))
        worker.completed.connect(lambda completed_path=path: self._continue_workspace_drop_parse_if_needed(completed_path))
        worker.completed.connect(self._schedule_preparse_drain)
        worker.completed.connect(lambda: self._remove_parse_worker(worker))
        worker.completed.connect(lambda: self._parse_worker_context.pop(worker, None))
        worker.completed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._remove_parse_thread(thread))

        self._parse_threads.append(thread)
        self._parse_workers.append(worker)
        self._parse_worker_context[worker] = (
            path,
            generation,
            bool(display_result),
            bool(preparse),
        )
        if preparse:
            thread.start(QThread.LowPriority)
        else:
            thread.start()

    def _cancel_obsolete_display_parses(self, keep_path: str) -> None:
        """Stop obsolete foreground parses without disturbing queued drop work."""

        for worker, (path, generation, display_result, preparse) in list(
            self._parse_worker_context.items()
        ):
            if (
                path == keep_path
                or preparse
                or not display_result
                or path in self._workspace_drop_requests
                or path in self._track_parse_requests
            ):
                continue
            worker.cancel()
            if self._parse_generations.get(path) == generation:
                self._parse_generations[path] = generation + 1

    def _restart_current_parse_if_superseded(self, path: str, generation: int) -> None:
        if self._parse_generations.get(path) == generation:
            return
        if path != self.main_window.material_browser.current_file_path():
            return
        if not self.main_window.material_browser.auto_slicing_enabled():
            return
        if path in self._slice_cache or path in self._parse_inflight:
            return
        QTimer.singleShot(
            0,
            lambda pending_path=path: self._start_parse_worker(
                pending_path,
                display_result=True,
                preparse=False,
            ),
        )

    def _on_audio_slices_ready_if_current(
        self,
        path: str,
        slices: object,
        display_result: bool,
        preparse: bool,
        generation: int,
    ) -> None:
        if self._parse_generations.get(path) != generation:
            return
        self._on_audio_slices_ready(path, slices, display_result, preparse)

    def _on_audio_slices_failed_if_current(
        self,
        message: str,
        path: str,
        display_result: bool,
        preparse: bool,
        generation: int,
    ) -> None:
        if self._parse_generations.get(path) != generation:
            return
        self._on_audio_slices_failed(message, path, display_result, preparse)

    def _continue_workspace_drop_parse_if_needed(self, path: str) -> None:
        if path not in self._workspace_drop_requests:
            return
        if path in self._parse_inflight or path in self._slice_cache:
            return
        self._start_parse_worker(path, display_result=False, preparse=False)

    def _schedule_preparse_drain(self) -> None:
        if self._is_shutting_down:
            return
        if not self.main_window.material_browser.auto_slicing_enabled():
            return
        if self._preparse_drain_timer.isActive():
            return
        self._preparse_drain_timer.start(self.PREPARSE_DRAIN_DELAY_MS)

    def _drain_preparse_queue(self) -> None:
        if not self.main_window.material_browser.auto_slicing_enabled():
            return
        if self._parse_inflight:
            return

        while self._preparse_queue:
            path = self._preparse_queue.pop(0)
            if path in self._slice_cache or path in self._parse_inflight:
                self._preparse_enqueued.discard(path)
                continue
            self._start_parse_worker(path, display_result=False, preparse=True)
            remaining = len(self._preparse_queue)
            self.main_window.material_browser.set_background_parse_status(
                self._ui_text(
                    f"预解析中 · 剩余 {remaining}",
                    f"Pre-parsing short files · {remaining} left",
                )
            )
            return

    def _on_material_slice_preview_requested(self, audio_slice: AudioSlice) -> None:
        self._show_status(
            f"正在准备媒体片段试听：{audio_slice.display_text()}",
            f"Preparing material clip preview: {audio_slice.display_text()}",
        )
        thread = QThread(self)
        worker = SlicePreviewWorker(audio_slice)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_slice_preview_audio_ready)
        worker.failed.connect(self._on_slice_preview_failed)
        worker.completed.connect(thread.quit)
        worker.completed.connect(lambda: self._remove_slice_preview_worker(worker))
        worker.completed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._remove_slice_preview_thread(thread))

        self._slice_preview_threads.append(thread)
        self._slice_preview_workers.append(worker)
        thread.start()

    def _on_material_slice_sequence_preview_requested(self, audio_slices: object) -> None:
        slices = [
            audio_slice
            for audio_slice in (audio_slices if isinstance(audio_slices, list) else [])
            if isinstance(audio_slice, AudioSlice)
        ]
        if not slices:
            return

        self._show_status(
            f"正在准备所选片段试听：{len(slices)} 个",
            f"Preparing selected material clips: {len(slices)} item(s).",
        )
        thread = QThread(self)
        worker = SliceSequencePreviewWorker(slices)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_slice_sequence_preview_audio_ready)
        worker.failed.connect(self._on_slice_sequence_preview_failed)
        worker.completed.connect(thread.quit)
        worker.completed.connect(lambda: self._remove_slice_sequence_preview_worker(worker))
        worker.completed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._remove_slice_preview_thread(thread))

        self._slice_preview_threads.append(thread)
        self._slice_sequence_preview_workers.append(worker)
        thread.start()

    def _on_slice_preview_audio_ready(self, audio: object, sample_rate: int) -> None:
        try:
            self.playback_manager.play(audio, int(sample_rate))
        except Exception as exc:  # noqa: BLE001 - playback errors are device-dependent
            self._show_status(f"片段试听失败：{exc}", f"Clip preview failed: {exc}")
            return
        self._show_status("正在试听媒体片段。", "Previewing material clip.")

    def _on_slice_sequence_preview_audio_ready(
        self,
        path: str,
        audio: object,
        sample_rate: int,
        start_time: float,
        end_time: float,
        slice_count: int,
    ) -> None:
        try:
            self._material_preview_timer.stop()
            self.playback_manager.play(audio, int(sample_rate))
        except Exception as exc:  # noqa: BLE001 - playback errors are device-dependent
            self._show_status(f"所选片段试听失败：{exc}", f"Selected clip preview failed: {exc}")
            self.main_window.material_browser.set_material_preview_playing(False)
            return

        self._material_preview_started_at = time.perf_counter()
        self._material_preview_start_time = max(0.0, float(start_time))
        self._material_preview_duration = max(self._material_preview_start_time, float(end_time))
        self.main_window.material_browser.set_material_preview_position(self._material_preview_start_time)
        self.main_window.material_browser.set_material_preview_playing(True)
        self._material_preview_timer.start()
        self._show_status(
            f"正在试听选中的 {slice_count} 个连续片段。",
            f"Previewing {slice_count} selected contiguous material clip(s).",
        )

    def _on_slice_sequence_preview_failed(self, message: str) -> None:
        print(f"Selected material slice preview failed: {message}", flush=True)
        self.main_window.material_browser.set_material_preview_playing(False)
        self._show_status(
            "选中的片段不连续，或不属于同一个源媒体。",
            f"Selected material clip preview failed: {message}",
        )

    def _on_slice_preview_failed(self, message: str) -> None:
        print(f"Slice preview failed: {message}", flush=True)
        self._show_status(f"片段试听失败：{message}", f"Clip preview failed: {message}")

    def _on_file_preview_audio_ready(
        self,
        path: str,
        audio: object,
        sample_rate: int,
        start_time: float,
        duration: float,
    ) -> None:
        try:
            self.playback_manager.play(audio, int(sample_rate))
        except Exception as exc:  # noqa: BLE001 - playback errors are device-dependent
            self._show_status(f"源媒体试听失败：{exc}", f"Source preview failed: {exc}")
            self.main_window.material_browser.set_material_preview_playing(False)
            return
        self._material_preview_started_at = time.perf_counter()
        self._material_preview_start_time = max(0.0, float(start_time))
        self._material_preview_duration = max(0.0, float(duration))
        self.main_window.material_browser.set_material_duration(path, duration)
        self.main_window.material_browser.set_material_preview_position(start_time)
        self.main_window.material_browser.set_material_preview_playing(True)
        self._material_preview_timer.start()
        self._show_status("正在试听源媒体。", "Previewing source media.")

    def _on_file_preview_failed(self, message: str) -> None:
        print(f"Material preview failed: {message}", flush=True)
        self._show_status(f"源媒体试听失败：{message}", f"Source preview failed: {message}")
        self.main_window.material_browser.set_material_preview_playing(False)

    def _sync_material_preview_position(self) -> None:
        elapsed = max(0.0, time.perf_counter() - self._material_preview_started_at)
        position = self._material_preview_start_time + elapsed
        if self._material_preview_duration > 0 and position >= self._material_preview_duration:
            position = self._material_preview_start_time
            self._material_preview_timer.stop()
            self.main_window.material_browser.set_material_preview_playing(False)
        self.main_window.material_browser.set_material_preview_position(position)

    def _show_whole_material_slice(self, path: str) -> None:
        cached = self._whole_slice_for_path(path)
        if cached is not None:
            self.main_window.material_browser.set_slices(path, [cached])
            self._show_status(
                "已生成整段媒体片段，可拖进时间线后手动分割。",
                "Full source clip ready; drag it to the timeline for manual splitting.",
            )
            return

        self.main_window.material_browser.set_parse_status(
            self._ui_text(f"正在准备整段媒体片段：{path}", f"Preparing full clip: {path}")
        )
        self.main_window.material_browser.set_slices(path, [])
        self._start_whole_slice_worker(path, {"mode": "display"})

    def _on_audio_file_dropped_as_slice(
        self,
        path: str,
        x: float,
        y: float,
        track_index: int,
    ) -> None:
        request = {
            "x": float(x),
            "y": float(y),
            "track_index": int(track_index),
        }
        manual_slices = self.project.material_slice_overrides.get(path)
        cached_slices = (
            list(manual_slices)
            if manual_slices and not self.main_window.material_browser.auto_slicing_enabled()
            else self._slice_cache.get(path)
        )
        if cached_slices is not None:
            self._add_parsed_slices_to_workspace(path, cached_slices, request)
            return
        self._workspace_drop_requests.setdefault(path, []).append(request)
        if path in self._preparse_queue:
            self._preparse_queue = [
                queued_path for queued_path in self._preparse_queue if queued_path != path
            ]
            self._preparse_enqueued.discard(path)
        if path in self._parse_inflight:
            self._show_status(
                "Workspace import is waiting for the current slice analysis.",
                "Workspace import is waiting for the current slice analysis.",
            )
            return
        self._show_status(
            f"Parsing material into pitch slices for workspace: {path}",
            f"Parsing material into pitch slices for workspace: {path}",
        )
        self._start_parse_worker(path, display_result=False, preparse=False)
        return

        cached = self._whole_slice_for_path(path)
        if cached is not None:
            self._add_whole_slice_to_workspace(cached, x, y, track_index)
            return
        self._show_status(
            f"正在准备整段媒体片段：{path}",
            f"Preparing full source clip: {path}",
        )
        self._start_whole_slice_worker(
            path,
            {
                "mode": "drop",
                "x": float(x),
                "y": float(y),
                "track_index": int(track_index),
            },
        )

    def _add_parsed_slices_to_workspace(
        self,
        path: str,
        slices: object,
        request: dict[str, object],
    ) -> None:
        audio_slices = [
            audio_slice for audio_slice in list(slices) if isinstance(audio_slice, AudioSlice)
        ]
        if not audio_slices:
            cached = self._whole_slice_for_path(path)
            if cached is not None:
                self._add_whole_slice_to_workspace(
                    cached,
                    float(request.get("x", 0.0)),
                    float(request.get("y", self.main_window.workspace.RULER_HEIGHT + 40.0)),
                    int(request.get("track_index", self.main_window.workspace.active_track_index)),
                )
            return

        workspace = self.main_window.workspace
        track_index = int(request.get("track_index", workspace.active_track_index))
        if not workspace._track_accepts_slices(track_index):
            track_index = workspace.active_track_index
        self._workspace_insert_generation += 1
        generation = self._workspace_insert_generation
        self._workspace_insert_active.add(generation)
        ordered_slices = sorted(
            audio_slices,
            key=lambda item: (
                item.source_path,
                item.start_time,
                item.end_time,
                item.index,
            ),
        )
        workspace.scene().clearSelection()
        self._insert_workspace_slice_batch(
            generation=generation,
            remaining=ordered_slices,
            created_items=[],
            x=float(request.get("x", 0.0)),
            y=float(request.get("y", workspace.RULER_HEIGHT + 40.0)),
            track_index=track_index,
            source_first_start=min(item.start_time for item in ordered_slices),
            placement_group_id=None,
        )

    def _insert_workspace_slice_batch(
        self,
        *,
        generation: int,
        remaining: list[AudioSlice],
        created_items: list[AudioSliceGraphicsItem],
        x: float,
        y: float,
        track_index: int,
        source_first_start: float,
        placement_group_id: str | None,
    ) -> None:
        if generation not in self._workspace_insert_active:
            for item in created_items:
                if item.scene() is self.main_window.workspace.scene():
                    self.main_window.workspace.scene().removeItem(item)
            return

        batch = remaining[: self.WORKSPACE_INSERT_BATCH_SIZE]
        del remaining[: self.WORKSPACE_INSERT_BATCH_SIZE]
        batch_items = self.main_window.workspace.add_slice_items(
            batch,
            x,
            y,
            track_index,
            source_first_start=source_first_start,
            placement_group_id=placement_group_id,
        )
        if batch_items and placement_group_id is None:
            placement_group_id = batch_items[0].placement_group_id
        created_items.extend(batch_items)

        if remaining:
            QTimer.singleShot(
                0,
                lambda: self._insert_workspace_slice_batch(
                    generation=generation,
                    remaining=remaining,
                    created_items=created_items,
                    x=x,
                    y=y,
                    track_index=track_index,
                    source_first_start=source_first_start,
                    placement_group_id=placement_group_id,
                ),
            )
            return

        for item in created_items:
            item.setSelected(True)
        self._workspace_insert_active.discard(generation)
        if created_items:
            y_positions = sorted(item.sceneBoundingRect().center().y() for item in created_items)
            median_y = y_positions[len(y_positions) // 2]
            anchor_item = min(
                created_items,
                key=lambda item: abs(item.sceneBoundingRect().center().y() - median_y),
            )
            self.main_window.workspace.ensureVisible(anchor_item, 80, 120)
        self._on_slice_items_dropped(created_items)
        self._show_status(
            f"Added {len(created_items)} pitch slice(s) to the workspace.",
            f"Added {len(created_items)} pitch slice(s) to the workspace.",
        )

    def _whole_slice_for_path(self, path: str) -> AudioSlice | None:
        cached = self._whole_slice_cache.get(path)
        if cached is not None:
            return cached

        try:
            full_slice = build_full_audio_slice(path)
            self._whole_slice_cache[path] = full_slice
            return full_slice
        except Exception as exc:  # noqa: BLE001 - fall back to parsed cache below.
            print(f"Full clip fallback failed for {path}: {exc}", flush=True)

        slices = self._slice_cache.get(path)
        if not slices:
            return None

        end_time = max((audio_slice.end_time for audio_slice in slices), default=0.0)
        weighted_f0 = [
            (audio_slice.f0_hz, max(0.001, audio_slice.duration))
            for audio_slice in slices
            if audio_slice.f0_hz is not None and audio_slice.f0_hz > 0.0
        ]
        f0_hz = None
        midi_note = None
        if weighted_f0:
            total_weight = sum(weight for _f0, weight in weighted_f0)
            if total_weight > 0:
                f0_hz = sum(float(f0) * weight for f0, weight in weighted_f0) / total_weight
                midi_note = max(0, min(127, int(round(69 + 12 * math.log2(f0_hz / 440.0)))))

        full_slice = AudioSlice(
            source_path=path,
            index=0,
            start_time=0.0,
            end_time=max(0.001, end_time),
            midi_note=midi_note,
            f0_hz=f0_hz,
            pitch_confidence=None,
            analysis_backend="weighted-cache" if f0_hz is not None else None,
        )
        self._whole_slice_cache[path] = full_slice
        return full_slice

    def _start_whole_slice_worker(self, path: str, request: dict[str, object]) -> None:
        thread = QThread(self)
        worker = WholeSliceWorker(path)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(
            lambda ready_path, audio_slice, pending_request=dict(request): (
                self._on_whole_slice_ready(ready_path, audio_slice, pending_request)
            )
        )
        worker.failed.connect(self._on_whole_slice_failed)
        worker.completed.connect(thread.quit)
        worker.completed.connect(lambda: self._remove_whole_slice_worker(worker))
        worker.completed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._remove_whole_slice_thread(thread))

        self._whole_slice_threads.append(thread)
        self._whole_slice_workers.append(worker)
        thread.start()

    def _on_whole_slice_ready(
        self,
        path: str,
        audio_slice: object,
        request: dict[str, object],
    ) -> None:
        if not isinstance(audio_slice, AudioSlice):
            return
        self._whole_slice_cache[path] = audio_slice
        mode = request.get("mode")
        if mode == "drop":
            self._add_whole_slice_to_workspace(
                audio_slice,
                float(request.get("x", 0.0)),
                float(request.get("y", self.main_window.workspace.RULER_HEIGHT + 40.0)),
                int(request.get("track_index", self.main_window.workspace.active_track_index)),
            )
            return
        self.main_window.material_browser.set_slices(path, [audio_slice])
        self._show_status(
            "已生成整段媒体片段，可拖进时间线后手动分割。",
            "Full source clip ready; drag it to the timeline for manual splitting.",
        )

    def _on_whole_slice_failed(self, message: str) -> None:
        print(f"Full clip preparation failed: {message}", flush=True)
        self._show_status(f"整段媒体片段准备失败：{message}", f"Full clip preparation failed: {message}")

    def _add_whole_slice_to_workspace(
        self,
        audio_slice: AudioSlice,
        x: float,
        y: float,
        track_index: int,
    ) -> None:
        fallback_y = max(self.main_window.workspace.RULER_HEIGHT + 1.0, float(y))
        item_y = self.main_window.workspace.y_for_midi_note(
            audio_slice.midi_note,
            fallback_y=fallback_y,
        )
        item = self.main_window.workspace.restore_slice_item(
            audio_slice=audio_slice,
            track_index=track_index,
            x=max(0.0, float(x)),
            y=item_y,
            width=max(
                72.0,
                audio_slice.duration * self.main_window.workspace.pixels_per_second(),
            ),
            height=30.0,
            target_midi_note=audio_slice.midi_note,
            target_duration=audio_slice.duration,
            protect_transients=self.main_window.workspace.default_protect_transients,
        )
        item.setSelected(True)
        self._show_status(
            "整段媒体片段已加入人声音轨，可用分割工具手动分段。",
            "Full source clip added to the vocal track; use Split to segment it.",
        )

    def _on_bgm_file_dropped(self, path: str, x: float) -> None:
        self._show_status(f"正在准备 BGM 音轨：{path}", f"Preparing BGM track: {path}")
        thread = QThread(self)
        worker = AudioProbeWorker(path)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(
            lambda info, drop_x=float(x): self._on_bgm_audio_info_ready(info, drop_x)
        )
        worker.failed.connect(self._on_bgm_audio_info_failed)
        worker.completed.connect(thread.quit)
        worker.completed.connect(lambda: self._remove_bgm_probe_worker(worker))
        worker.completed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._remove_bgm_probe_thread(thread))

        self._bgm_probe_threads.append(thread)
        self._bgm_probe_workers.append(worker)
        thread.start()

    def _on_bgm_audio_info_ready(self, info: AudioInfo, x: float) -> None:
        self._audio_info_cache[info.path] = info
        audio_slice = AudioSlice(
            source_path=info.path,
            index=0,
            start_time=0.0,
            end_time=max(0.001, float(info.duration)),
            midi_note=None,
            f0_hz=None,
        )
        item = self.main_window.workspace.restore_slice_item(
            audio_slice=audio_slice,
            track_index=0,
            x=max(0.0, float(x)),
            y=self.main_window.workspace.RULER_HEIGHT + 8.0,
            width=max(72.0, audio_slice.duration * self.main_window.workspace.pixels_per_second()),
            height=34.0,
            target_midi_note=None,
            target_duration=audio_slice.duration,
        )
        item.set_track_type("master_bgm")
        item.set_locked(True)
        self._show_status(
            f"BGM 已加入：{info.path} | 时长={info.duration:.2f}s",
            f"BGM added: {info.path} | duration={info.duration:.2f}s",
        )

    def _on_bgm_audio_info_failed(self, message: str) -> None:
        print(f"BGM probe failed: {message}", flush=True)
        self._show_status(f"BGM 导入失败：{message}", f"BGM import failed: {message}")

    def _on_audio_info_ready(self, info: AudioInfo) -> None:
        self._audio_info_cache[info.path] = info
        message = (
            f"{info.path} | sample_rate={info.sample_rate} Hz | "
            f"duration={info.duration:.3f}s | channels={info.channels}"
        )
        print(message, flush=True)
        self.main_window.statusBar().showMessage(message)
        self.main_window.material_browser.set_probe_result(message)
        self.main_window.material_browser.set_material_duration(info.path, info.duration)

    def _on_audio_info_failed(self, message: str) -> None:
        print(f"Audio probe failed: {message}", flush=True)
        self._show_status("音频探测失败。", "Audio probe failed.")
        self.main_window.material_browser.set_probe_result(
            self._ui_text(f"探测失败：{message}", f"Probe failed: {message}")
        )

    def _on_audio_slices_ready(
        self,
        path: str,
        slices: object,
        display_result: bool = True,
        preparse: bool = False,
    ) -> None:
        audio_slices = list(slices)
        manual_slices = self.project.material_slice_overrides.get(path)
        auto_slice_mode = self.main_window.material_browser.auto_slicing_enabled()
        effective_slices = (
            list(manual_slices)
            if (manual_slices and not auto_slice_mode)
            else audio_slices
        )
        self._slice_cache[path] = audio_slices
        self._whole_slice_cache.pop(path, None)
        track_requests = self._track_parse_requests.pop(path, set())
        for track_index in sorted(track_requests):
            self._apply_track_parse_result(track_index, path, audio_slices)
        workspace_requests = self._workspace_drop_requests.pop(path, [])
        for request in workspace_requests:
            self._add_parsed_slices_to_workspace(path, audio_slices, request)
        if display_result or self.main_window.material_browser.current_file_path() == path:
            self.main_window.material_browser.set_slices(path, effective_slices)
        en_message = f"Analyzed {len(audio_slices)} clips: {path}"
        zh_message = f"已分析 {len(audio_slices)} 个片段：{path}"
        print(en_message, flush=True)
        if preparse and not display_result:
            self._show_status(
                f"预分析完成：{len(audio_slices)} 个片段。",
                f"Pre-analysis ready: {len(audio_slices)} clip(s).",
            )
        else:
            self._show_status(zh_message, en_message)

    def _on_audio_slices_failed(
        self,
        message: str,
        path: str | None = None,
        display_result: bool = True,
        preparse: bool = False,
    ) -> None:
        print(f"Slice parse failed: {message}", flush=True)
        if path is not None:
            if path not in self.project.material_slice_overrides:
                self._slice_cache.pop(path, None)
            self._whole_slice_cache.pop(path, None)
            self._track_parse_requests.pop(path, None)
            self._workspace_drop_requests.pop(path, None)
        if display_result or not preparse:
            if (
                path is not None
                and self.main_window.material_browser.current_file_path() != path
            ):
                return
            self._show_status("片段分析失败。", "Clip analysis failed.")
            self.main_window.material_browser.set_parse_status(
                self._ui_text(f"解析失败：{message}", f"Parse failed: {message}")
            )
            return
        self._show_status("后台预分析跳过了一个媒体文件。", "Background pre-analysis skipped one file.")

    def _on_preparse_skipped(self, path: str, reason: str) -> None:
        print(f"Pre-parse skipped: {path} ({reason})", flush=True)
        self._slice_cache.pop(path, None)
        self._whole_slice_cache.pop(path, None)
        if path in self._workspace_drop_requests:
            QTimer.singleShot(
                0,
                lambda pending_path=path: self._start_parse_worker(
                    pending_path,
                    display_result=False,
                    preparse=False,
                ),
            )
        self.main_window.material_browser.set_background_parse_status(
            self._ui_text(
                f"预分析跳过长媒体 · 剩余 {len(self._preparse_queue)}",
                f"Skipped long pre-analysis · {len(self._preparse_queue)} left",
            )
        )

    def render_slice_item(self, item: AudioSliceGraphicsItem) -> None:
        cache_key = item.render_cache_key()
        cached_result = self._render_result_cache.get(cache_key)
        if cached_result is not None:
            self._render_result_cache.move_to_end(cache_key)
            item.set_rendering(True, cache_key)
            item.store_render_result(cached_result)
            if item.isSelected():
                self.main_window.inspector_widget.set_item(item)
            self._try_pending_preview_groups()
            self._try_pending_global_playback()
            self._try_pending_export()
            return
        if cache_key in self._render_targets:
            if item not in self._render_targets[cache_key]:
                self._render_targets[cache_key].append(item)
            item.set_rendering(True, cache_key)
            return

        # A quickly repeated edit can supersede a queued render before it has
        # started. Detach this item from stale jobs so they do not consume time.
        for stale_key, targets in list(self._render_targets.items()):
            if stale_key == cache_key or item not in targets:
                continue
            targets.remove(item)
            if not targets and stale_key not in self._render_active_keys:
                self._render_targets.pop(stale_key, None)
        self._render_targets[cache_key] = [item]
        item.set_rendering(True, cache_key)
        self.main_window.statusBar().showMessage(f"Rendering slice: {cache_key}")

        advanced_enabled = bool(
            self.main_window.inspector_widget.ADVANCED_CONTROLS_ENABLED
        )
        job = (cache_key, item.build_render_request(advanced_enabled))
        self._render_queue.append(job)
        self._render_queued_keys.add(cache_key)
        self._drain_render_queue()

    def _drain_render_queue(self) -> None:
        while (
            len(self._render_active_keys) < self.MAX_RENDER_WORKERS
            and self._render_queue
        ):
            job = self._render_queue.pop(0)
            cache_key = str(job[0])
            self._render_queued_keys.discard(cache_key)
            targets = [
                target
                for target in self._render_targets.get(cache_key, [])
                if target._active_render_key == cache_key
            ]
            if not targets:
                self._render_targets.pop(cache_key, None)
                continue
            self._render_targets[cache_key] = targets
            self._launch_render_worker(job)

    def _launch_render_worker(self, job: tuple[object, ...]) -> None:
        cache_key, request = job
        cache_key = str(cache_key)
        self._render_active_keys.add(cache_key)

        thread = QThread(self)
        worker = RenderWorker(
            cache_key=cache_key,
            audio_slice=request.audio_slice,
            target_midi_note=request.target_midi_note,
            target_duration=request.target_duration,
            gain_db=float(request.gain_db),
            pitch_flatten_amount=float(request.pitch_flatten_amount),
            formant_shift=float(request.formant_shift),
            protect_transients=bool(request.protect_transients),
            pitch_control_points=request.pitch_control_points,
            pitch_vibrato_regions=request.pitch_vibrato_regions,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_render_finished)
        worker.failed.connect(self._on_render_failed)
        worker.completed.connect(thread.quit)
        worker.completed.connect(lambda: self._remove_render_worker(worker))
        worker.completed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda key=cache_key: self._on_render_thread_finished(thread, key)
        )

        self._render_threads.append(thread)
        self._render_workers.append(worker)
        thread.start()

    def _on_render_thread_finished(self, thread: QThread, cache_key: str) -> None:
        self._remove_render_thread(thread)
        self._render_active_keys.discard(cache_key)
        QTimer.singleShot(0, self._drain_render_queue)

    def _on_render_finished(self, result: object) -> None:
        cache_key = result.cache_key
        self._cache_render_result(result)
        targets = self._render_targets.pop(cache_key, [])
        for target in targets:
            target.store_render_result(result)
            if target.isSelected():
                self.main_window.inspector_widget.set_item(target)
        message = (
            f"Rendered {cache_key} | n_steps={result.parameters.n_steps:.3f} | "
            f"rate={result.parameters.rate:.3f} | samples={len(result.audio)}"
        )
        print(message, flush=True)
        self.main_window.statusBar().showMessage(message)
        self._try_pending_preview_groups()
        self._try_pending_global_playback()
        self._try_pending_export()

    def _cache_render_result(self, result: object) -> None:
        cache_key = str(getattr(result, "cache_key", ""))
        audio = getattr(result, "audio", None)
        item_bytes = int(getattr(audio, "nbytes", 0))
        if (
            not cache_key
            or item_bytes <= 0
            or item_bytes > self.RENDER_RESULT_CACHE_MAX_ITEM_BYTES
        ):
            return
        previous = self._render_result_cache.pop(cache_key, None)
        if previous is not None:
            self._render_result_cache_bytes -= int(
                getattr(getattr(previous, "audio", None), "nbytes", 0)
            )
        self._render_result_cache[cache_key] = result
        self._render_result_cache_bytes += item_bytes
        while (
            self._render_result_cache
            and self._render_result_cache_bytes > self.RENDER_RESULT_CACHE_MAX_BYTES
        ):
            _old_key, old_result = self._render_result_cache.popitem(last=False)
            self._render_result_cache_bytes -= int(
                getattr(getattr(old_result, "audio", None), "nbytes", 0)
            )

    def _on_render_failed(self, message: str) -> None:
        cache_key = message.split(": ", 1)[0]
        targets = self._render_targets.pop(cache_key, [])
        for target in targets:
            if target._active_render_key == cache_key:
                target.set_rendering(False)
        print(f"Render failed: {message}", flush=True)
        self.main_window.statusBar().showMessage(f"Render failed: {message}")
        self._pending_preview_groups.clear()
        self._pending_global_playback_start_time = None
        self._pending_global_playback_resume = False
        self._pending_export_path = None
        self._try_pending_preview_groups()

    def _on_slice_items_created(self, items: object) -> None:
        if self.main_window.workspace.pitch_curve_edit_mode:
            self._schedule_visible_waveforms()
        self.main_window.refresh_timeline_transport_duration()

    def _on_pitch_curve_view_changed(self, enabled: bool) -> None:
        if enabled:
            self._queue_visible_waveforms()
            self._show_status(
                "Preparing pitch curves in the background.",
                "Preparing pitch curves in the background.",
            )
        else:
            self._show_status(
                "Pitch edit view hidden; analysis now runs only for preview.",
                "Pitch edit view hidden; analysis now runs only for preview.",
            )

    def _schedule_visible_waveforms(self, *_args: object) -> None:
        if self._is_shutting_down or not self.main_window.workspace.pitch_curve_edit_mode:
            return
        self._visible_waveform_timer.start()

    def _queue_visible_waveforms(self) -> None:
        if self._is_shutting_down or not self.main_window.workspace.pitch_curve_edit_mode:
            return
        visible_items = self.main_window.workspace.visible_slice_items()
        selected_items = self.main_window.workspace.selected_slice_items()
        candidates = list(visible_items)
        for item in self._selected_waveform_prefetch_items(selected_items):
            if item not in candidates:
                candidates.append(item)
        for item in candidates:
            if not item.is_missing_source:
                self._start_waveform_worker(item)

    def _selected_waveform_prefetch_items(
        self,
        selected_items: list[AudioSliceGraphicsItem],
    ) -> list[AudioSliceGraphicsItem]:
        if len(selected_items) <= self.MAX_SELECTED_WAVEFORM_PREFETCH:
            return selected_items
        visible = self.main_window.workspace.visible_slice_items()
        return [item for item in selected_items if item in visible]

    def _on_slice_items_dropped(self, items: object) -> None:
        if not isinstance(items, (list, tuple)):
            return
        created_items = [
            item
            for item in items
            if isinstance(item, AudioSliceGraphicsItem)
            and item.scene() is self.main_window.workspace.scene()
        ]
        if not created_items:
            return
        self.main_window.undo_stack.push(
            AddSliceCommand(
                workspace=self.main_window.workspace,
                items=created_items,
                render_callback=self._render_item_if_required,
                post_callback=self._after_timeline_item_command,
                initially_applied=True,
            )
        )

    def _after_timeline_item_command(self) -> None:
        self.main_window.refresh_timeline_transport_duration()
        self._on_workspace_selection_changed()

    def _start_waveform_worker(self, item: AudioSliceGraphicsItem) -> None:
        cache_key = item.base_cache_key()
        cached_result = self._waveform_result_cache.get(cache_key)
        if cached_result is not None:
            self._waveform_result_cache.move_to_end(cache_key)
            item.store_waveform_result(cached_result)
            return
        if cache_key in self._waveform_targets:
            if item not in self._waveform_targets[cache_key]:
                self._waveform_targets[cache_key].append(item)
            return
        self._waveform_targets[cache_key] = [item]

        self._waveform_queue.append((cache_key, item.audio_slice))
        self._waveform_queued_keys.add(cache_key)
        self._drain_waveform_queue()

    def _drain_waveform_queue(self) -> None:
        while (
            len(self._waveform_active_keys) < self.MAX_WAVEFORM_WORKERS
            and self._waveform_queue
        ):
            cache_key, audio_slice = self._waveform_queue.pop(0)
            self._waveform_queued_keys.discard(cache_key)
            targets = self._waveform_targets.get(cache_key, [])
            if not targets:
                continue
            self._waveform_active_keys.add(cache_key)
            self._launch_waveform_worker(cache_key, audio_slice)

    def _launch_waveform_worker(
        self,
        cache_key: str,
        audio_slice: AudioSlice,
    ) -> None:

        thread = QThread(self)
        worker = WaveformWorker(
            cache_key=cache_key,
            audio_slice=audio_slice,
            max_points=self.WAVEFORM_PREVIEW_POINTS,
            prefer_cached_pitch=audio_slice.analysis_backend != "librosa_yin_fast",
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_waveform_ready)
        worker.failed.connect(self._on_waveform_failed)
        worker.completed.connect(thread.quit)
        worker.completed.connect(lambda: self._remove_waveform_worker(worker))
        worker.completed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda key=cache_key: self._on_waveform_thread_finished(thread, key)
        )

        self._waveform_threads.append(thread)
        self._waveform_workers.append(worker)
        thread.start(QThread.LowPriority)

    def _on_waveform_thread_finished(self, thread: QThread, cache_key: str) -> None:
        self._remove_waveform_thread(thread)
        self._waveform_active_keys.discard(cache_key)
        QTimer.singleShot(0, self._drain_waveform_queue)

    def _on_waveform_ready(self, result: object) -> None:
        self._waveform_result_cache[result.cache_key] = result
        self._waveform_result_cache.move_to_end(result.cache_key)
        while len(self._waveform_result_cache) > self.WAVEFORM_RESULT_CACHE_SIZE:
            self._waveform_result_cache.popitem(last=False)
        items = self._waveform_targets.pop(result.cache_key, [])
        for item in items:
            item.store_waveform_result(result)
            if item.isSelected():
                self.main_window.inspector_widget.set_item(item)
        self.main_window.statusBar().showMessage(f"Waveform ready: {result.cache_key}")
        self._try_pending_preview_groups()
        self._try_pending_global_playback()
        self._try_pending_export()

    def _on_waveform_failed(self, message: str) -> None:
        cache_key = message.split(": ", 1)[0]
        self._waveform_targets.pop(cache_key, None)
        print(f"Waveform failed: {message}", flush=True)
        self.main_window.statusBar().showMessage(f"Waveform failed: {message}")
        self._pending_preview_groups.clear()
        self._pending_global_playback_start_time = None
        self._pending_export_path = None
        self._try_pending_preview_groups()

    def _on_preview_requested(self, payload: object) -> None:
        if isinstance(payload, AudioSliceGraphicsItem):
            items = [payload]
        elif isinstance(payload, list):
            items = [item for item in payload if isinstance(item, AudioSliceGraphicsItem)]
        else:
            items = []
        if not items:
            return

        if self._prepare_items_for_preview(items):
            self._play_preview_items(items)
        else:
            self._pending_preview_groups.append(items)
            self._show_status("正在准备试听音频...", "Preparing preview audio...")

    def _prepare_items_for_preview(self, items: list[AudioSliceGraphicsItem]) -> bool:
        ready = True
        for item in items:
            if self._item_requires_render(item):
                if not item.has_current_render_cache():
                    self.render_slice_item(item)
                    ready = False
            elif item.base_audio_cache is None or item.base_audio_sample_rate is None:
                self._start_waveform_worker(item)
                ready = False
        return ready

    def _try_pending_preview_groups(self) -> None:
        remaining: list[list[AudioSliceGraphicsItem]] = []
        for items in self._pending_preview_groups:
            if self._prepare_items_for_preview(items):
                self._play_preview_items(items)
            else:
                remaining.append(items)
        self._pending_preview_groups = remaining

    def _play_preview_items(self, items: list[AudioSliceGraphicsItem]) -> None:
        buffers = []
        sample_rate = None
        for item in items:
            audio, item_sample_rate = item.preview_audio()
            if audio is None or item_sample_rate is None:
                continue
            if sample_rate is None:
                sample_rate = item_sample_rate
            if item_sample_rate != sample_rate:
                print("Preview skipped mixed sample-rate selection.", flush=True)
                self._show_status(
                    "已跳过试听：选中的片段采样率不同。",
                    "Preview skipped: selected clips use different sample rates.",
                )
                return
            buffers.append(audio)

        if not buffers or sample_rate is None:
            self._show_status("没有可试听的音频。", "No preview audio is ready.")
            return

        audio = buffers[0] if len(buffers) == 1 else mix_playback_buffers(buffers)
        try:
            self._material_preview_timer.stop()
            self.main_window.material_browser.set_material_preview_playing(False)
            self.playback_manager.play(audio, sample_rate)
        except Exception as exc:  # noqa: BLE001 - playback errors are device-dependent
            print(f"Playback failed: {exc}", flush=True)
            self._show_status(f"播放失败：{exc}", f"Playback failed: {exc}")
            return
        self._show_status(
            f"正在试听 {len(buffers)} 个片段。",
            f"Previewing {len(buffers)} clip{'s' if len(buffers) != 1 else ''}.",
        )

    def _toggle_global_playback(self) -> None:
        if self.playback_manager.is_timeline_playing:
            self._stop_global_playback()
            return
        self._start_global_playback(
            self.main_window.workspace.playhead_time(),
            resume=self._timeline_playback_paused,
        )

    def _start_global_playback(self, start_time: float, *, resume: bool = False) -> None:
        if self.playback_manager.is_timeline_playing:
            self.playback_manager.stop_timeline()
            self._playhead_timer.stop()
        self._material_preview_timer.stop()
        self.main_window.material_browser.set_material_preview_playing(False)

        timeline_end_time = self._project_timeline_end_time()
        candidate_items = self._playable_timeline_items()
        if not candidate_items:
            if timeline_end_time > start_time:
                try:
                    self.main_window.refresh_timeline_transport_duration()
                    self.playback_manager.play_timeline(
                        [],
                        start_time=start_time,
                        sample_rate=self.project.sample_rate,
                        timeline_end_time=timeline_end_time,
                    )
                except Exception as exc:  # noqa: BLE001 - playback errors are device-dependent
                    print(f"Timeline playback failed: {exc}", flush=True)
                    self._show_status(
                        f"时间线播放失败：{exc}",
                        f"Timeline playback failed: {exc}",
                    )
                    return
                if not resume:
                    self._timeline_playback_return_time = max(0.0, float(start_time))
                self._timeline_playback_paused = False
                self.main_window.workspace.set_playhead_time(start_time)
                self.main_window.timeline_transport.set_playing(True)
                self._playhead_timer.start()
                self._show_status(
                    f"所有轨道已静音，播放头从 {start_time:.2f}s 静音播放。",
                    f"All tracks are muted; playhead running silently from {start_time:.2f}s.",
                )
                return
            self._show_status(
                "没有可播放的未静音时间线片段。",
                "No unmuted timeline clips to play.",
            )
            return

        if not self._prepare_items_for_preview(candidate_items):
            self._pending_global_playback_start_time = start_time
            self._pending_global_playback_resume = resume
            self._show_status("正在准备时间线播放...", "Preparing timeline playback...")
            return

        clips = self._build_timeline_clips(candidate_items)
        if not clips:
            self._show_status("时间线音频还没有准备好。", "Timeline audio is not ready.")
            return

        try:
            self.main_window.refresh_timeline_transport_duration()
            self.playback_manager.play_timeline(
                clips,
                start_time=start_time,
                timeline_end_time=timeline_end_time,
            )
        except Exception as exc:  # noqa: BLE001 - playback errors are device-dependent
            print(f"Timeline playback failed: {exc}", flush=True)
            self._show_status(
                f"时间线播放失败：{exc}",
                f"Timeline playback failed: {exc}",
            )
            return

        if not resume:
            self._timeline_playback_return_time = max(0.0, float(start_time))
        self._timeline_playback_paused = False
        self.main_window.workspace.set_playhead_time(start_time)
        self.main_window.timeline_transport.set_playing(True)
        self._playhead_timer.start()
        self._show_status(
            f"时间线从 {start_time:.2f}s 开始播放。",
            f"Timeline playing from {start_time:.2f}s.",
        )

    def _stop_global_playback(self) -> None:
        self.playback_manager.stop_timeline()
        self._playhead_timer.stop()
        self.main_window.timeline_transport.set_playing(False)
        self._timeline_playback_paused = True
        self._show_status(
            f"时间线暂停在 {self.main_window.workspace.playhead_time():.2f}s。",
            f"Timeline paused at {self.main_window.workspace.playhead_time():.2f}s.",
        )

    def _stop_all_playback(self) -> None:
        self.playback_manager.stop()
        self._playhead_timer.stop()
        self._material_preview_timer.stop()
        self.main_window.material_browser.set_material_preview_playing(False)
        self.main_window.timeline_transport.set_playing(False)
        self.main_window.workspace.set_playhead_time(self._timeline_playback_return_time)
        self._timeline_playback_paused = False
        self._pending_preview_groups.clear()
        self._pending_global_playback_start_time = None
        self._show_status("播放已停止。", "Playback stopped.")

    def _return_playhead_to_start(self) -> None:
        was_playing = self.playback_manager.is_timeline_playing
        self._stop_all_playback()
        self.main_window.workspace.set_playhead_time(0.0)
        if was_playing:
            self._show_status("播放已停止在 0.00s。", "Playback stopped at 0.00s.")
        else:
            self._show_status("播放头已回到 0.00s。", "Playhead returned to 0.00s.")

    def _on_playhead_seek_requested(self, seconds: float) -> None:
        if self.playback_manager.is_timeline_playing:
            self._start_global_playback(seconds, resume=True)
        else:
            self._show_status(f"播放头：{seconds:.2f}s", f"Playhead: {seconds:.2f}s")

    def _sync_playhead_from_playback(self) -> None:
        if self.playback_manager.is_timeline_playing:
            self.main_window.workspace.set_playhead_time(
                self.playback_manager.timeline_current_time
            )
            return
        self._playhead_timer.stop()
        self.main_window.workspace.set_playhead_time(self._timeline_playback_return_time)
        self.main_window.timeline_transport.set_playing(False)
        self._timeline_playback_paused = False

    def _try_pending_global_playback(self) -> None:
        if self._pending_global_playback_start_time is None:
            return
        start_time = self._pending_global_playback_start_time
        resume = self._pending_global_playback_resume
        candidate_items = self._playable_timeline_items()
        if self._prepare_items_for_preview(candidate_items):
            self._pending_global_playback_start_time = None
            self._pending_global_playback_resume = False
            self._start_global_playback(start_time, resume=resume)

    def _playable_timeline_items(self) -> list[AudioSliceGraphicsItem]:
        items: list[AudioSliceGraphicsItem] = []
        for item in self.main_window.workspace.slice_items():
            if not self._track_is_audible(item.track_index):
                continue
            items.append(item)
        return sorted(items, key=self.main_window.workspace.item_start_time)

    def _project_timeline_end_time(self) -> float:
        end_time = self.main_window.workspace.timeline_end_time()
        for track in self.project.tracks:
            if not track.clip_path:
                continue
            end_time = max(
                end_time,
                float(track.clip_start) + max(0.0, float(track.clip_duration)),
            )
        return end_time

    def _build_timeline_clips(
        self,
        items: list[AudioSliceGraphicsItem],
    ) -> list[TimelineClip]:
        clips: list[TimelineClip] = []
        for item in items:
            audio, sample_rate = item.preview_audio()
            if audio is None or sample_rate is None:
                continue
            clips.append(
                TimelineClip(
                    start_time=self.main_window.workspace.item_start_time(item),
                    audio=audio,
                    sample_rate=sample_rate,
                    track_index=item.track_index,
                )
            )
        return clips

    def _track_is_muted(self, track_index: int) -> bool:
        if 0 <= track_index < len(self.project.tracks):
            return self.project.tracks[track_index].muted
        return False

    def _track_is_audible(self, track_index: int) -> bool:
        if self._track_is_muted(track_index):
            return False
        solo_indices = {
            index
            for index, track in enumerate(self.project.tracks)
            if track.solo and not track.muted
        }
        if not solo_indices:
            return True
        return track_index in solo_indices

    def _choose_export_path(self) -> None:
        output_path, _ = QFileDialog.getSaveFileName(
            self.main_window,
            self._ui_text("导出音频", "Export Audio"),
            str(Path.home() / "hakyking_export.wav"),
            "WAV files (*.wav)",
        )
        if not output_path:
            return
        if not output_path.lower().endswith(".wav"):
            output_path += ".wav"
        self._request_export(output_path)

    def _request_export(self, output_path: str) -> None:
        items = self._exportable_timeline_items()
        if not items:
            self._show_status("没有可导出的时间线片段。", "No timeline clips to export.")
            return

        if not self._prepare_items_for_preview(items):
            self._pending_export_path = output_path
            self._show_status("正在准备导出缓存...", "Preparing export buffers...")
            return

        self._start_export_worker(output_path, items)

    def _try_pending_export(self) -> None:
        if self._pending_export_path is None:
            return
        output_path = self._pending_export_path
        items = self._exportable_timeline_items()
        if self._prepare_items_for_preview(items):
            self._pending_export_path = None
            self._start_export_worker(output_path, items)

    def _exportable_timeline_items(self) -> list[AudioSliceGraphicsItem]:
        return sorted(
            [
                item
                for item in self.main_window.workspace.slice_items()
                if self._track_is_audible(item.track_index)
            ],
            key=self.main_window.workspace.item_start_time,
        )

    def _start_export_worker(
        self,
        output_path: str,
        items: list[AudioSliceGraphicsItem],
    ) -> None:
        if self._export_thread is not None:
            self._show_status("导出已经在运行。", "Export is already running.")
            return

        clips = self._build_export_clips(items)
        if not clips:
            self._show_status(
                "没有准备好的导出音频缓存。",
                "No export audio buffers are ready.",
            )
            return

        self.playback_manager.stop()
        thread = QThread(self)
        worker = ExportWorker(
            clips=clips,
            output_path=output_path,
            sample_rate=44100,
            fade_ms=self.playback_manager.fade_ms,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_export_finished)
        worker.failed.connect(self._on_export_failed)
        worker.completed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_export_worker)

        self._export_thread = thread
        self._export_worker = worker
        self._show_status(f"正在导出 WAV：{output_path}", f"Exporting WAV: {output_path}")
        thread.start()

    def _build_export_clips(self, items: list[AudioSliceGraphicsItem]) -> list[ExportClip]:
        clips: list[ExportClip] = []
        for item in items:
            audio, sample_rate = item.preview_audio()
            if audio is None or sample_rate is None:
                continue
            clips.append(
                ExportClip(
                    start_time=self.main_window.workspace.item_start_time(item),
                    audio=audio,
                    sample_rate=sample_rate,
                    track_index=item.track_index,
                )
            )
        return clips

    def _on_export_finished(self, result: object) -> None:
        message = (
            f"Exported WAV: {result.output_path} | duration={result.duration:.2f}s | "
            f"peak={result.peak_before_limit:.3f} | normalized={result.normalized}"
        )
        print(message, flush=True)
        self._show_status(
            f"已导出 WAV：{result.output_path} | 时长={result.duration:.2f}s | "
            f"峰值={result.peak_before_limit:.3f} | 已标准化={result.normalized}",
            message,
        )

    def _on_export_failed(self, message: str) -> None:
        print(f"Export failed: {message}", flush=True)
        self._show_status(f"导出失败：{message}", f"Export failed: {message}")

    def _clear_export_worker(self) -> None:
        self._export_thread = None
        self._export_worker = None

    def _item_requires_render(self, item: AudioSliceGraphicsItem) -> bool:
        advanced_enabled = bool(
            self.main_window.inspector_widget.ADVANCED_CONTROLS_ENABLED
        )
        return item.requires_rendered_audio(
            advanced_controls_enabled=advanced_enabled
        )

    def _ui_text(self, zh: str, en: str) -> str:
        return zh if self.main_window.current_language() == "zh" else en

    def _show_status(self, zh: str, en: str) -> None:
        self.main_window.statusBar().showMessage(self._ui_text(zh, en))

    def _remove_probe_thread(self, thread: QThread) -> None:
        if thread in self._probe_threads:
            self._probe_threads.remove(thread)

    def _remove_probe_worker(self, worker: AudioProbeWorker) -> None:
        if worker in self._probe_workers:
            self._probe_workers.remove(worker)

    def _remove_bgm_probe_thread(self, thread: QThread) -> None:
        if thread in self._bgm_probe_threads:
            self._bgm_probe_threads.remove(thread)

    def _remove_bgm_probe_worker(self, worker: AudioProbeWorker) -> None:
        if worker in self._bgm_probe_workers:
            self._bgm_probe_workers.remove(worker)

    def _remove_folder_scan_thread(self, thread: QThread) -> None:
        if thread in self._folder_scan_threads:
            self._folder_scan_threads.remove(thread)

    def _remove_folder_scan_worker(self, worker: FolderMediaScanWorker) -> None:
        if worker in self._folder_scan_workers:
            self._folder_scan_workers.remove(worker)

    def _remove_parse_thread(self, thread: QThread) -> None:
        if thread in self._parse_threads:
            self._parse_threads.remove(thread)

    def _remove_parse_worker(self, worker: ParseWorker) -> None:
        if worker in self._parse_workers:
            self._parse_workers.remove(worker)

    def _remove_slice_preview_thread(self, thread: QThread) -> None:
        if thread in self._slice_preview_threads:
            self._slice_preview_threads.remove(thread)

    def _remove_slice_preview_worker(self, worker: SlicePreviewWorker) -> None:
        if worker in self._slice_preview_workers:
            self._slice_preview_workers.remove(worker)

    def _remove_slice_sequence_preview_worker(self, worker: SliceSequencePreviewWorker) -> None:
        if worker in self._slice_sequence_preview_workers:
            self._slice_sequence_preview_workers.remove(worker)

    def _remove_file_preview_thread(self, thread: QThread) -> None:
        if thread in self._file_preview_threads:
            self._file_preview_threads.remove(thread)

    def _remove_file_preview_worker(self, worker: FilePreviewWorker) -> None:
        if worker in self._file_preview_workers:
            self._file_preview_workers.remove(worker)

    def _remove_whole_slice_thread(self, thread: QThread) -> None:
        if thread in self._whole_slice_threads:
            self._whole_slice_threads.remove(thread)

    def _remove_whole_slice_worker(self, worker: WholeSliceWorker) -> None:
        if worker in self._whole_slice_workers:
            self._whole_slice_workers.remove(worker)

    def _remove_render_thread(self, thread: QThread) -> None:
        if thread in self._render_threads:
            self._render_threads.remove(thread)

    def _remove_render_worker(self, worker: RenderWorker) -> None:
        if worker in self._render_workers:
            self._render_workers.remove(worker)

    def _remove_waveform_thread(self, thread: QThread) -> None:
        if thread in self._waveform_threads:
            self._waveform_threads.remove(thread)

    def _remove_waveform_worker(self, worker: WaveformWorker) -> None:
        if worker in self._waveform_workers:
            self._waveform_workers.remove(worker)
