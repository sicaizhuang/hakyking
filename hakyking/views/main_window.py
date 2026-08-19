from __future__ import annotations

import math

from hakyking.models.scale import NOTE_NAMES
from hakyking.qt import (
    QAction,
    QActionGroup,
    QAbstractButton,
    QColor,
    QComboBox,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QIcon,
    QLabel,
    QMainWindow,
    QMenu,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPoint,
    QRectF,
    QProxyStyle,
    QPushButton,
    QScrollArea,
    Signal,
    QSize,
    QSplitter,
    QSlider,
    QTimer,
    QToolBar,
    QToolButton,
    QStyle,
    Qt,
    QUndoStack,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)
from hakyking.views.diagnostics import DiagnosticsDialog
from hakyking.views.inspector import InspectorWidget
from hakyking.views.material_browser import MaterialBrowserWidget, PreviewScrubBar
from hakyking.views.panel import Panel
from hakyking.views.piano_roll import PianoRollWidget
from hakyking.views.track_controls import TrackControlPanel
from hakyking.views.workspace import WorkspaceView, _workspace_cursor


_TRANSPORT_ICON_CACHE: dict[str, QIcon] = {}


class ToolPopupStyle(QProxyStyle):
    """Give compact tool popup menus a legible icon size at high DPI."""

    def pixelMetric(self, metric, option=None, widget=None):  # type: ignore[override]
        if metric == QStyle.PM_SmallIconSize:
            return 24
        return super().pixelMetric(metric, option, widget)


def _mini_transport_icon(action_name: str) -> QIcon:
    cached = _TRANSPORT_ICON_CACHE.get(action_name)
    if cached is not None:
        return cached
    pixmap = QPixmap(22, 22)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    color = QColor("#d8e4ef")
    painter.setPen(QPen(color, 2))
    painter.setBrush(color)
    if action_name == "play":
        path = QPainterPath()
        path.moveTo(7, 5)
        path.lineTo(17, 11)
        path.lineTo(7, 17)
        path.closeSubpath()
        painter.fillPath(path, color)
    elif action_name == "pause":
        painter.fillRect(6, 5, 4, 12, color)
        painter.fillRect(13, 5, 4, 12, color)
    elif action_name == "stop":
        painter.fillRect(6, 6, 10, 10, color)
    painter.end()
    icon = QIcon(pixmap)
    _TRANSPORT_ICON_CACHE[action_name] = icon
    return icon


class TimelineTransportWidget(QWidget):
    """Compact transport strip shown below the main timeline canvas."""

    play_pause_requested = Signal()
    stop_requested = Signal()
    position_previewed = Signal(float)
    seek_requested = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._duration = 8.0
        self._position = 0.0
        self._playing = False
        self.setFixedHeight(34)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(5)

        self.play_button = QPushButton()
        self.play_button.setFixedSize(28, 26)
        self.play_button.setIconSize(QSize(18, 18))
        self.play_button.setToolTip("播放 / 暂停")
        layout.addWidget(self.play_button)

        self.stop_button = QPushButton()
        self.stop_button.setFixedSize(28, 26)
        self.stop_button.setIconSize(QSize(18, 18))
        self.stop_button.setToolTip("停止")
        layout.addWidget(self.stop_button)

        self.scrub_bar = PreviewScrubBar()
        layout.addWidget(self.scrub_bar, 1)

        self.time_label = QLabel("0:00")
        self.time_label.setFixedWidth(42)
        self.time_label.setVisible(False)
        layout.addWidget(self.time_label)

        self.play_button.clicked.connect(self.play_pause_requested)
        self.stop_button.clicked.connect(self.stop_requested)
        self.scrub_bar.position_previewed.connect(self._on_position_previewed)
        self.scrub_bar.seek_requested.connect(self._on_seek_requested)
        self.scrub_bar.set_duration(self._duration)
        self._refresh()

    def set_duration(self, duration: float) -> None:
        self._duration = max(0.1, float(duration))
        self.scrub_bar.set_duration(self._duration)
        self._position = min(self._position, self._duration)
        self._refresh()

    def set_position(self, position: float) -> None:
        self._position = max(0.0, min(float(position), self._duration))
        self.scrub_bar.set_position(self._position)
        self._refresh()

    def set_playing(self, playing: bool) -> None:
        self._playing = bool(playing)
        self._refresh()

    def _on_seek_requested(self, position: float) -> None:
        self._position = max(0.0, min(float(position), self._duration))
        self._refresh()
        self.seek_requested.emit(self._position)

    def _on_position_previewed(self, position: float) -> None:
        self._position = max(0.0, min(float(position), self._duration))
        self._refresh()
        self.position_previewed.emit(self._position)

    def _refresh(self) -> None:
        self.play_button.setIcon(_mini_transport_icon("pause" if self._playing else "play"))
        self.stop_button.setIcon(_mini_transport_icon("stop"))
        self.time_label.setText(self._format_time(self._position))
        self.time_label.setToolTip(
            f"{self._format_time(self._position)}/{self._format_time(self._duration)}"
        )

    def _format_time(self, seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        minutes = int(seconds // 60)
        whole_seconds = int(seconds % 60)
        return f"{minutes}:{whole_seconds:02d}"


class TimelineRulerWidget(QWidget):
    """Fixed time ruler that stays visible while the workspace scrolls vertically."""

    seek_requested = Signal(float)

    def __init__(self, workspace: WorkspaceView, parent=None) -> None:
        super().__init__(parent)
        self.workspace = workspace
        self._duration = 8.0
        self._left_scene_x = 0.0
        self._right_scene_x = 1.0
        self._viewport_left = 0
        self._viewport_width = 1
        self.setFixedHeight(34)
        self.setMouseTracking(True)

    def set_duration(self, duration: float) -> None:
        self._duration = max(0.1, float(duration))
        self.sync_from_workspace()

    def sync_from_workspace(self) -> None:
        viewport = self.workspace.viewport()
        viewport_width = max(1, viewport.width())
        left = self.workspace.mapToScene(0, 0).x()
        right = self.workspace.mapToScene(viewport_width, 0).x()
        if right <= left:
            right = left + 1.0
        self._left_scene_x = max(0.0, float(left))
        self._right_scene_x = max(self._left_scene_x + 1.0, float(right))
        viewport_top_left = self.mapFromGlobal(viewport.mapToGlobal(QPoint(0, 0)))
        self._viewport_left = int(viewport_top_left.x())
        self._viewport_width = viewport_width
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#242529"))
        painter.setPen(QPen(QColor("#393c42"), 1))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)

        scene_span = max(1.0, self._right_scene_x - self._left_scene_x)
        label_pad = 10
        start_time = max(0.0, self.workspace.x_to_seconds(self._left_scene_x))
        end_time = min(
            self._duration + 1.0,
            self.workspace.x_to_seconds(self._right_scene_x) + 2.0,
        )
        pps = max(1.0, self.workspace.pixels_per_second())
        major_step = self._major_tick_step(pps)
        minor_step = major_step / 4.0
        first_minor = math.floor(start_time / minor_step) * minor_step
        tick = first_minor
        while tick <= end_time + minor_step:
            scene_x = self.workspace.seconds_to_x(tick)
            x = self._x_for_scene(scene_x, scene_span)
            if -48 <= x <= self.width() + 48:
                major = abs((tick / major_step) - round(tick / major_step)) < 1e-4
                painter.setPen(QPen(QColor("#59606a") if major else QColor("#3f444b"), 1))
                painter.drawLine(x, 12 if major else 20, x, self.height() - 1)
                if major:
                    painter.setPen(QColor("#b8bec8"))
                    text_left = max(label_pad, self._viewport_left + 6)
                    text_right = min(self.width() - 62, self._viewport_left + self._viewport_width - 62)
                    text_x = max(text_left, min(text_right, x + 6))
                    painter.drawText(
                        QRectF(text_x, 8, 62, 20),
                        Qt.AlignLeft | Qt.AlignVCenter,
                        self._format_tick_label(tick, major_step),
                    )
            tick += minor_step

        playhead_x = self._x_for_scene(
            self.workspace.seconds_to_x(self.workspace.playhead_time()),
            scene_span,
        )
        painter.setPen(QPen(QColor("#ff4a4a"), 2))
        painter.drawLine(playhead_x, 0, playhead_x, self.height())
        painter.end()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self._seek_from_x(event.pos().x())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.buttons() & Qt.LeftButton:
            self._seek_from_x(event.pos().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.sync_from_workspace()

    def _seek_from_x(self, x: int) -> None:
        span = max(1.0, self._right_scene_x - self._left_scene_x)
        viewport_x = max(0, min(self._viewport_width, x - self._viewport_left))
        scene_x = self._left_scene_x + viewport_x / max(1, self._viewport_width) * span
        seconds = self.workspace.x_to_seconds(scene_x)
        self.workspace.set_playhead_time(seconds)
        self.seek_requested.emit(seconds)

    def _x_for_scene(self, scene_x: float, scene_span: float | None = None) -> int:
        span = max(1.0, float(scene_span or (self._right_scene_x - self._left_scene_x)))
        ratio = (float(scene_x) - self._left_scene_x) / span
        return int(round(self._viewport_left + ratio * max(1, self._viewport_width)))

    @staticmethod
    def _major_tick_step(pixels_per_second: float) -> float:
        for step in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
            if step * pixels_per_second >= 90.0:
                return step
        return 20.0

    @staticmethod
    def _format_tick_label(seconds: float, step: float) -> str:
        if step < 0.1:
            return f"{seconds:.2f}s"
        if step < 1.0:
            return f"{seconds:.1f}s"
        if abs(seconds - round(seconds)) < 1e-5:
            return f"{int(round(seconds))}s"
        return f"{seconds:.1f}s"


class MainWindow(QMainWindow):
    about_to_close = Signal()
    transient_protection_requested = Signal(bool)

    TEXT = {
        "zh": {
            "file": "文件",
            "edit": "编辑",
            "tools": "工具",
            "transport": "传输",
            "settings": "设置",
            "view": "视图",
            "help": "帮助",
            "new": "新建",
            "open": "打开工程...",
            "open_autosave": "打开自动保存...",
            "save": "保存工程",
            "save_as": "另存为...",
            "export": "导出音频...",
            "undo": "撤销",
            "redo": "重做",
            "select_all": "全选片段",
            "copy": "复制片段",
            "paste": "粘贴片段",
            "duplicate": "复制一份",
            "delete": "删除所选片段",
            "play_pause": "播放 / 暂停",
            "stop": "停止",
            "return_start": "回到开头",
            "piano_roll": "",
            "workspace": "音高编辑区",
            "tracks": "音轨",
            "materials": "媒体库",
            "inspector": "属性面板",
            "project_toolbar": "工程",
            "toolbox": "编辑工具",
            "show_toolbox": "显示编辑工具栏",
            "show_inspector": "显示片段属性面板",
            "show_materials": "显示媒体库",
            "show_tracks": "显示音轨面板",
            "zoom_in": "时间线放大",
            "zoom_out": "时间线缩小",
            "zoom_reset": "重置时间线缩放",
            "diagnostics": "诊断中心...",
            "root": "主音",
            "scale": "音阶",
            "language": "语言",
            "audio_settings": "音频设置...",
            "select_tool": "选择 / 移动",
            "scissors_tool": "分割",
            "cut_merge_tool": "片段贴合 / 合并",
            "amplitude_tool": "增益",
            "flatten_tool": "颤音展平",
            "formant_tool": "共振峰偏移",
            "pitch_curve_select_tool": "音高控制点",
            "pitch_curve_point_tool": "添加控制点",
            "flatten_tip": "颤音展平 0% - 100%",
            "formant_tip": "共振峰偏移 -12 到 +12 半音",
            "copy_pitch_curve": "复制音高曲线",
            "paste_pitch_curve": "粘贴音高曲线",
            "smooth_pitch_curve": "自动平滑滑音",
            "vibrato_pitch_curve": "颤音",
            "vibrato_sine": "正弦波",
            "vibrato_triangle": "三角波",
            "vibrato_square": "矩形波",
            "pitch_segment_shape": "段形状",
            "pitch_segment_original": "原始曲线",
            "pitch_segment_linear": "直线过渡",
            "pitch_segment_smooth": "平滑过渡",
            "pitch_glide_shape": "滑音形状",
            "pitch_glide_ease_in": "先慢后快",
            "pitch_glide_ease_out": "先快后慢",
            "pitch_glide_s_curve": "S 型滑音",
            "pitch_glide_instant": "瞬间跳音",
            "pitch_zero": "一键归零",
            "pitch_natural": "自然化",
            "pitch_electro": "电音化",
            "root_tip": "主音",
            "scale_tip": "音阶类型",
            "language_tip": "界面语言",
            "initialized": "Hakyking 已初始化。",
            "major": "大调",
            "minor": "自然小调",
            "pentatonic": "五声音阶",
            "chromatic": "半音阶",
        },
        "en": {
            "file": "File",
            "edit": "Edit",
            "tools": "Tools",
            "transport": "Transport",
            "settings": "Settings",
            "view": "View",
            "help": "Help",
            "new": "New",
            "open": "Open Project...",
            "open_autosave": "Open Autosave...",
            "save": "Save Project",
            "save_as": "Save As...",
            "export": "Export Audio...",
            "undo": "Undo",
            "redo": "Redo",
            "select_all": "Select All Clips",
            "copy": "Copy Clips",
            "paste": "Paste Clips",
            "duplicate": "Duplicate Clips",
            "delete": "Delete Selected Clips",
            "play_pause": "Play / Pause",
            "stop": "Stop",
            "return_start": "Return to Start",
            "piano_roll": "",
            "workspace": "Pitch Editor",
            "tracks": "Tracks",
            "materials": "Media Library",
            "inspector": "Properties",
            "project_toolbar": "Project",
            "toolbox": "Edit Tools",
            "show_toolbox": "Show Edit Toolbar",
            "show_inspector": "Show Clip Properties",
            "show_materials": "Show Media Library",
            "show_tracks": "Show Track Panel",
            "zoom_in": "Zoom Timeline In",
            "zoom_out": "Zoom Timeline Out",
            "zoom_reset": "Reset Timeline Zoom",
            "diagnostics": "Diagnostics...",
            "root": "Root",
            "scale": "Scale",
            "language": "Language",
            "audio_settings": "Audio Settings...",
            "select_tool": "Arrow / Select",
            "scissors_tool": "Split",
            "cut_merge_tool": "Fit / Merge Clips",
            "amplitude_tool": "Gain",
            "flatten_tool": "Vibrato Flatten",
            "formant_tool": "Formant Shift",
            "pitch_curve_select_tool": "Pitch Control Points",
            "pitch_curve_point_tool": "Add Control Point",
            "flatten_tip": "Vibrato flatten 0% - 100%",
            "formant_tip": "Formant shift -12 to +12 semitones",
            "copy_pitch_curve": "Copy Pitch Curve",
            "paste_pitch_curve": "Paste Pitch Curve",
            "smooth_pitch_curve": "Smooth Glide",
            "vibrato_pitch_curve": "Vibrato",
            "vibrato_sine": "Sine",
            "vibrato_triangle": "Triangle",
            "vibrato_square": "Square",
            "pitch_segment_shape": "Segment Shape",
            "pitch_segment_original": "Original Curve",
            "pitch_segment_linear": "Linear",
            "pitch_segment_smooth": "Smooth",
            "pitch_glide_shape": "Glide Shape",
            "pitch_glide_ease_in": "Slow-Fast",
            "pitch_glide_ease_out": "Fast-Slow",
            "pitch_glide_s_curve": "S Glide",
            "pitch_glide_instant": "Pitch Jump",
            "pitch_zero": "Reset Pitch Curve",
            "pitch_natural": "Naturalize",
            "pitch_electro": "Electro Tune",
            "root_tip": "Root Note",
            "scale_tip": "Scale Type",
            "language_tip": "Interface Language",
            "initialized": "Hakyking initialized.",
            "major": "Major",
            "minor": "Minor",
            "pentatonic": "Pentatonic",
            "chromatic": "Chromatic",
        },
    }
    SCALE_ORDER = ("Major", "Minor", "Pentatonic", "Chromatic")
    SCALE_TEXT_KEYS = {
        "Major": "major",
        "Minor": "minor",
        "Pentatonic": "pentatonic",
        "Chromatic": "chromatic",
    }

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hakyking")
        self.resize(1480, 880)
        self.setMinimumSize(1180, 720)
        self._transient_protection_enabled = True

        self.undo_stack = QUndoStack(self)
        self.undo_action = self.undo_stack.createUndoAction(self, "Undo")
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.setShortcutContext(Qt.ApplicationShortcut)
        self.redo_action = self.undo_stack.createRedoAction(self, "Redo")
        self.redo_action.setShortcut("Ctrl+Y")
        self.redo_action.setShortcutContext(Qt.ApplicationShortcut)
        self.select_all_action = QAction(self)
        self.select_all_action.setShortcut("Ctrl+A")
        self.copy_action = QAction(self)
        self.copy_action.setShortcut("Ctrl+C")
        self.paste_action = QAction(self)
        self.paste_action.setShortcut("Ctrl+V")
        self.duplicate_action = QAction(self)
        self.duplicate_action.setShortcut("Ctrl+D")
        self.delete_selected_action = QAction(self)
        self.delete_selected_action.setShortcut("Del")

        self.piano_roll = PianoRollWidget()
        self.workspace = WorkspaceView()
        self.timeline_ruler = TimelineRulerWidget(self.workspace)
        self.timeline_transport = TimelineTransportWidget()
        self.track_control_panel = TrackControlPanel()
        self.track_control_panel.setMinimumHeight(96)
        self.material_browser = MaterialBrowserWidget()
        self.inspector_widget = InspectorWidget()

        self.root_note_combo = QComboBox()
        self.root_note_combo.addItems(NOTE_NAMES)
        self.root_note_combo.setFixedWidth(66)
        self.scale_type_combo = QComboBox()
        for scale_type in self.SCALE_ORDER:
            self.scale_type_combo.addItem(scale_type, scale_type)
        self.scale_type_combo.setCurrentIndex(self.scale_type_combo.findData("Chromatic"))
        self.scale_type_combo.setFixedWidth(112)
        self.language_combo = QComboBox()
        self.language_combo.addItem("中文", "zh")
        self.language_combo.addItem("EN", "en")
        self.language_combo.setFixedWidth(76)

        self.new_project_action = QAction(self)
        self.open_project_action = QAction(self)
        self.open_autosave_action = QAction(self)
        self.save_project_action = QAction(self)
        self.save_project_as_action = QAction(self)
        self.export_action = QAction(self)
        self.diagnostics_action = QAction(self)
        self.audio_settings_action = QAction(self)
        self.zoom_in_action = QAction(self)
        self.zoom_in_action.setShortcut("Ctrl++")
        self.zoom_out_action = QAction(self)
        self.zoom_out_action.setShortcut("Ctrl+-")
        self.zoom_reset_action = QAction(self)
        self.zoom_reset_action.setShortcut("Ctrl+0")
        self.play_pause_action = QAction(self)
        self.play_pause_action.setShortcut("Ctrl+Space")
        self.stop_action = QAction(self)
        self.stop_action.setShortcut("Esc")
        self.return_start_action = QAction(self)
        self.return_start_action.setShortcut("Home")

        self.tool_action_group = QActionGroup(self)
        self.tool_action_group.setExclusive(True)
        self.play_pause_action.setIcon(self._make_transport_icon("play"))
        self.stop_action.setIcon(self._make_transport_icon("stop"))
        self.select_tool_action = self._create_tool_action("select")
        self.scissors_tool_action = self._create_tool_action("scissors")
        self.amplitude_tool_action = self._create_tool_action("amplitude")
        self.scissors_merge_action = QAction(self._make_tool_icon("cut_merge"), "", self)
        self.scissors_merge_action.setCheckable(True)
        self.scissors_merge_action.setShortcut("D")
        self.scissors_merge_action.setShortcutContext(Qt.ApplicationShortcut)
        self.scissors_merge_action.toggled.connect(self.update_scissors_merge_ui)
        self.scissors_plain_menu_action = QAction(self._make_tool_icon("scissors"), "", self)
        self.scissors_merge_menu_action = QAction(self._make_tool_icon("cut_merge"), "", self)
        self.scissors_plain_menu_action.triggered.connect(self._activate_plain_scissors)
        self.scissors_merge_menu_action.triggered.connect(self._activate_scissors_merge)
        self.select_tool_action.setShortcut("Z")
        self.scissors_tool_action.setShortcut("X")
        self.amplitude_tool_action.setShortcut("C")
        for action in (
            self.select_tool_action,
            self.scissors_tool_action,
            self.amplitude_tool_action,
        ):
            action.setShortcutContext(Qt.ApplicationShortcut)
        self.flatten_tool_action = self._create_tool_action("flatten")
        self.formant_tool_action = self._create_tool_action("formant")
        # Pitch-curve tools are part of the same visible tool selector as the
        # block tools.  Pitch view adds capabilities, but there is still only
        # one active mouse tool at a time.
        self.pitch_curve_tool_action_group = self.tool_action_group
        self.pitch_curve_select_action = self._create_pitch_curve_tool_action("pitch_curve_select")
        self.pitch_curve_point_action = self._create_pitch_curve_tool_action("pitch_curve_point")
        self.pitch_curve_select_action.setData("curve_select")
        self.pitch_curve_point_action.setData("curve_point")
        self.pitch_curve_select_action.setShortcut("V")
        self.pitch_curve_point_action.setShortcut("B")
        self.pitch_curve_select_action.setShortcutContext(Qt.ApplicationShortcut)
        self.pitch_curve_point_action.setShortcutContext(Qt.ApplicationShortcut)
        self.pitch_curve_select_action.setVisible(False)
        self.pitch_curve_point_action.setVisible(False)
        self.pitch_curve_select_action.setEnabled(False)
        self.pitch_curve_point_action.setEnabled(False)
        self.copy_pitch_curve_action = QAction(self)
        self.copy_pitch_curve_action.setShortcut("Ctrl+Alt+C")
        self.copy_pitch_curve_action.setShortcutContext(Qt.ApplicationShortcut)
        self.paste_pitch_curve_action = QAction(self)
        self.paste_pitch_curve_action.setShortcut("Ctrl+Alt+V")
        self.paste_pitch_curve_action.setShortcutContext(Qt.ApplicationShortcut)
        self.smooth_pitch_curve_action = QAction(self)
        self.vibrato_pitch_curve_action = self._create_pitch_curve_tool_action("pitch_vibrato")
        self.vibrato_pitch_curve_action.setData("curve_vibrato")
        self.vibrato_pitch_curve_action.setShortcut("N")
        self.vibrato_pitch_curve_action.setShortcutContext(Qt.ApplicationShortcut)
        self.vibrato_pitch_curve_action.setEnabled(False)
        self.vibrato_pitch_curve_action.setVisible(False)
        self.vibrato_waveform_group = QActionGroup(self)
        self.vibrato_waveform_group.setExclusive(True)
        self.vibrato_sine_action = QAction(self)
        self.vibrato_triangle_action = QAction(self)
        self.vibrato_square_action = QAction(self)
        for action, waveform in (
            (self.vibrato_sine_action, "sine"),
            (self.vibrato_triangle_action, "triangle"),
            (self.vibrato_square_action, "square"),
        ):
            action.setCheckable(True)
            action.setData(waveform)
            self.vibrato_waveform_group.addAction(action)
        self.vibrato_sine_action.setChecked(True)
        self.vibrato_sine_action.setShortcut("Alt+1")
        self.vibrato_triangle_action.setShortcut("Alt+2")
        self.vibrato_square_action.setShortcut("Alt+3")
        for action in (
            self.vibrato_sine_action,
            self.vibrato_triangle_action,
            self.vibrato_square_action,
        ):
            action.setShortcutContext(Qt.ApplicationShortcut)
            self.addAction(action)
        self.pitch_segment_shape_action = QAction(self._make_tool_icon("pitch_segment_shape"), "", self)
        self.pitch_segment_original_action = QAction(self)
        self.pitch_segment_linear_action = QAction(self)
        self.pitch_segment_smooth_action = QAction(self)
        self.pitch_glide_shape_action = QAction(self._make_tool_icon("pitch_glide_shape"), "", self)
        self.pitch_glide_ease_in_action = QAction(self)
        self.pitch_glide_ease_out_action = QAction(self)
        self.pitch_glide_s_curve_action = QAction(self)
        self.pitch_glide_instant_action = QAction(self)
        self._last_pitch_segment_shape = "linear"
        self._last_pitch_glide_shape = "s_curve"
        self.pitch_segment_shape_action.triggered.connect(
            lambda _checked=False: self._apply_pitch_curve_segment_shape(
                self._last_pitch_segment_shape
            )
        )
        self.pitch_glide_shape_action.triggered.connect(
            lambda _checked=False: self._apply_pitch_curve_segment_shape(
                self._last_pitch_glide_shape
            )
        )
        for action, shape in (
            (self.pitch_segment_original_action, "original"),
            (self.pitch_segment_linear_action, "linear"),
            (self.pitch_segment_smooth_action, "smooth"),
            (self.pitch_glide_ease_in_action, "ease_in"),
            (self.pitch_glide_ease_out_action, "ease_out"),
            (self.pitch_glide_s_curve_action, "s_curve"),
            (self.pitch_glide_instant_action, "instant"),
        ):
            action.setData(shape)
            action.triggered.connect(
                lambda _checked=False, action=action: self._choose_pitch_curve_segment_shape(
                    str(action.data() or "linear")
                )
            )
        for action in (
            self.select_tool_action,
            self.scissors_tool_action,
            self.amplitude_tool_action,
            self.pitch_curve_select_action,
            self.pitch_curve_point_action,
            self.vibrato_pitch_curve_action,
            self.scissors_merge_action,
        ):
            self.addAction(action)
        self.vibrato_waveform_group.triggered.connect(
            self._on_vibrato_waveform_changed
        )
        self.pitch_zero_action = QAction(self)
        self.pitch_natural_action = QAction(self)
        self.pitch_electro_action = QAction(self)
        self.select_tool_action.setChecked(True)

        self.flatten_slider = QSlider(Qt.Horizontal)
        self.flatten_slider.setRange(0, 100)
        self.flatten_slider.setFixedWidth(110)

        self.formant_slider = QSlider(Qt.Horizontal)
        self.formant_slider.setRange(-120, 120)
        self.formant_slider.setFixedWidth(120)

        self._build_layout()
        self._build_docks()
        self._build_toolbox_toolbar()
        self._build_menu_bar()
        for action in (
            self.pitch_curve_select_action,
            self.pitch_curve_point_action,
            self.vibrato_pitch_curve_action,
        ):
            action.triggered.connect(
                lambda _checked=False, action=action: self._on_pitch_curve_tool_action_triggered(action)
            )
            action.toggled.connect(
                lambda checked=False, action=action: (
                    self._on_pitch_curve_tool_action_triggered(action) if checked else None
                )
            )
        self.inspector_widget.pitch_curve_view_toggled.connect(
            self.workspace.set_pitch_curve_edit_mode
        )
        self.inspector_widget.pitch_curve_view_toggled.connect(
            self._on_pitch_curve_view_toggled
        )
        self.workspace.pitch_curve_selection_changed.connect(
            self._on_pitch_curve_selection_changed
        )
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        self.diagnostics_action.triggered.connect(self._show_diagnostics_dialog)
        self.set_language("zh")
        self._apply_application_cursors()

    def _build_layout(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        root_layout.addWidget(self.main_splitter, 1)

        self.piano_roll.setFixedHeight(
            int(WorkspaceView.SCENE_HEIGHT - WorkspaceView.RULER_HEIGHT)
        )
        self.piano_roll_header_spacer = QWidget()
        self.piano_roll_header_spacer.setFixedHeight(self.timeline_ruler.height())
        piano_roll_content = QWidget()
        piano_roll_layout = QVBoxLayout(piano_roll_content)
        piano_roll_layout.setContentsMargins(0, 0, 0, 0)
        piano_roll_layout.setSpacing(0)
        piano_roll_layout.addWidget(self.piano_roll_header_spacer)
        piano_roll_layout.addWidget(self.piano_roll)
        piano_roll_content.setFixedHeight(int(WorkspaceView.SCENE_HEIGHT))

        self.piano_scroll_area = QScrollArea()
        self.piano_scroll_area.setWidgetResizable(False)
        self.piano_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.piano_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.piano_scroll_area.setWidget(piano_roll_content)

        self.left_column = QWidget()
        self.left_column.setMinimumWidth(128)
        self.left_column.setMaximumWidth(170)
        left_column_layout = QVBoxLayout(self.left_column)
        left_column_layout.setContentsMargins(0, 0, 0, 0)
        left_column_layout.setSpacing(0)

        self.left_panel = Panel("", self.piano_scroll_area)
        self.left_panel.setMinimumWidth(118)
        self.left_panel.setMaximumWidth(170)
        self.left_track_spacer = QFrame()
        self.left_track_spacer.setObjectName("LeftTrackSpacer")
        self.left_track_spacer.setMinimumHeight(0)
        self.left_track_spacer.setVisible(True)
        left_column_layout.addWidget(self.left_panel)
        left_column_layout.addWidget(self.left_track_spacer, 1)
        self.main_splitter.addWidget(self.left_column)

        self.center_splitter = QSplitter(Qt.Vertical)
        self.center_splitter.setChildrenCollapsible(False)
        workspace_content = QWidget()
        workspace_layout = QVBoxLayout(workspace_content)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)
        workspace_layout.addWidget(self.timeline_ruler)
        workspace_layout.addWidget(self.workspace, 1)
        workspace_layout.addWidget(self.timeline_transport)
        self.workspace_panel = Panel("", workspace_content)
        self.track_panel = Panel("", self.track_control_panel)
        self.workspace_panel.setMinimumHeight(360)
        self.track_panel.setMinimumHeight(120)
        self.center_splitter.addWidget(self.workspace_panel)
        self.center_splitter.addWidget(self.track_panel)
        self.center_splitter.setStretchFactor(0, 5)
        self.center_splitter.setStretchFactor(1, 1)
        self.center_splitter.setSizes([670, 136])
        self.center_splitter.splitterMoved.connect(self._on_center_splitter_moved)
        self.main_splitter.addWidget(self.center_splitter)

        self.material_panel = Panel("", self.material_browser)
        self.material_panel.setMinimumWidth(430)
        self.material_panel.setMaximumWidth(720)
        self.main_splitter.addWidget(self.material_panel)

        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([150, 880, 480])

        self.setCentralWidget(root)
        QTimer.singleShot(0, self._apply_default_splitter_sizes)
        QTimer.singleShot(120, self._sync_left_column_heights)
        self.statusBar().showMessage("")
        self.refresh_timeline_transport_duration()
        self.timeline_transport.play_pause_requested.connect(self.play_pause_action.trigger)
        self.timeline_transport.stop_requested.connect(self.stop_action.trigger)
        self.timeline_transport.position_previewed.connect(self.workspace.set_playhead_time)
        self.timeline_transport.seek_requested.connect(self._seek_timeline_from_transport)
        self.timeline_ruler.seek_requested.connect(self._seek_timeline_from_transport)
        self.workspace.playhead_changed.connect(self.timeline_transport.set_position)
        self.workspace.playhead_changed.connect(lambda _seconds: self.timeline_ruler.update())
        self.workspace.horizontalScrollBar().valueChanged.connect(
            lambda _value: self.timeline_ruler.sync_from_workspace()
        )
        self.workspace.horizontal_zoom_changed.connect(
            lambda _zoom: (self.timeline_ruler.sync_from_workspace(), self.refresh_timeline_transport_duration())
        )
        self._sync_piano_workspace_scrollbars()

    def _apply_default_splitter_sizes(self) -> None:
        total_width = max(1, self.main_splitter.size().width())
        left_width = min(170, max(140, self.left_column.minimumWidth()))
        right_width = min(520, max(460, self.material_panel.minimumWidth()))
        if total_width < left_width + right_width + 520:
            right_width = max(self.material_panel.minimumWidth(), total_width - left_width - 520)
        center_width = max(520, total_width - left_width - right_width)
        self.main_splitter.setSizes([left_width, center_width, right_width])

        total_height = max(1, self.center_splitter.size().height())
        track_height = min(190, max(150, self.track_panel.minimumHeight()))
        if total_height < self.workspace_panel.minimumHeight() + track_height:
            track_height = max(self.track_panel.minimumHeight(), total_height - self.workspace_panel.minimumHeight())
        workspace_height = max(self.workspace_panel.minimumHeight(), total_height - track_height)
        self.center_splitter.setSizes([workspace_height, track_height])
        self._sync_left_column_heights()

    def _sync_left_column_heights(self, *_args) -> None:
        """Keep the piano viewport exactly aligned with the workspace canvas."""
        if not hasattr(self, "center_splitter"):
            return
        canvas_height = max(1, int(self.workspace.viewport().height()))
        scroll_frame_height = max(
            0,
            int(self.piano_scroll_area.height() - self.piano_scroll_area.viewport().height()),
        )
        scroll_height = canvas_height + scroll_frame_height
        self.piano_scroll_area.setFixedHeight(scroll_height)
        self.left_panel.setFixedHeight(self.left_panel.title_label.height() + scroll_height)

    def _on_center_splitter_moved(self, *_args) -> None:
        self._sync_left_column_heights()
        # QSplitter emits before nested viewport layouts have always settled.
        QTimer.singleShot(0, self._sync_left_column_heights)
        QTimer.singleShot(30, self._sync_left_column_heights)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        QTimer.singleShot(0, self._sync_left_column_heights)

    def _seek_timeline_from_transport(self, seconds: float) -> None:
        self.workspace.set_playhead_time(seconds)
        self.workspace.playhead_seek_requested.emit(seconds)

    def refresh_timeline_transport_duration(self) -> None:
        scene_duration = self.workspace.x_to_seconds(self.workspace.scene().sceneRect().width())
        duration = max(
            4.0,
            scene_duration,
            self.workspace.timeline_end_time(),
            self.workspace.playhead_time() + 0.1,
        )
        self.timeline_transport.set_duration(duration)
        self.timeline_ruler.set_duration(duration)

    def _sync_piano_workspace_scrollbars(self) -> None:
        workspace_bar = self.workspace.verticalScrollBar()
        piano_bar = self.piano_scroll_area.verticalScrollBar()
        workspace_bar.valueChanged.connect(piano_bar.setValue)
        piano_bar.valueChanged.connect(workspace_bar.setValue)

    def _apply_application_cursors(self) -> None:
        self.setCursor(_workspace_cursor("app_arrow"))
        central = self.centralWidget()
        if central is not None:
            central.setCursor(_workspace_cursor("app_arrow"))
        for button in self.findChildren(QAbstractButton):
            button.setCursor(_workspace_cursor("app_hand"))
        for combo in self.findChildren(QComboBox):
            combo.setCursor(_workspace_cursor("app_hand"))
        for slider in self.findChildren(QSlider):
            slider.setCursor(_workspace_cursor("app_hand"))
        self.workspace.viewport().setCursor(_workspace_cursor("move"))
        self.material_browser.slice_list._refresh_tool_cursor()

    def _build_docks(self) -> None:
        self.inspector_dock = QDockWidget("", self)
        self.inspector_dock.setObjectName("SliceInspectorDock")
        self.inspector_dock.setWidget(self.inspector_widget)
        self.inspector_dock.setMinimumWidth(128)
        self.inspector_dock.setMaximumWidth(180)
        if not self.inspector_widget.ADVANCED_CONTROLS_ENABLED:
            compact_title_bar = QWidget()
            compact_title_bar.setFixedHeight(1)
            self.inspector_dock.setTitleBarWidget(compact_title_bar)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.inspector_dock)

    def _build_toolbox_toolbar(self) -> None:
        self.toolbox_toolbar = QToolBar("", self)
        self.toolbox_toolbar.setMovable(True)
        self.toolbox_toolbar.setIconSize(QSize(22, 22))
        self.toolbox_toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.toolbox_toolbar.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.toolbox_toolbar.addAction(self.select_tool_action)
        self.scissors_tool_container = QWidget(self.toolbox_toolbar)
        scissors_layout = QVBoxLayout(self.scissors_tool_container)
        scissors_layout.setContentsMargins(0, 0, 0, 0)
        scissors_layout.setSpacing(1)
        self.scissors_tool_button = QToolButton(self.scissors_tool_container)
        self.scissors_tool_button.setDefaultAction(self.scissors_tool_action)
        self.scissors_tool_button.setIconSize(QSize(25, 25))
        self.scissors_tool_button.setFixedSize(QSize(38, 30))
        scissors_layout.addWidget(self.scissors_tool_button)
        self.scissors_menu_button = QToolButton(self.scissors_tool_container)
        self.scissors_menu_button.setObjectName("ScissorsMenuButton")
        self.scissors_menu_button.setText("▾")
        self.scissors_menu_button.setFixedSize(QSize(38, 14))
        scissors_layout.addWidget(self.scissors_menu_button)
        self.toolbox_toolbar.addWidget(self.scissors_tool_container)
        self.toolbox_toolbar.addAction(self.amplitude_tool_action)
        self.toolbox_toolbar.addAction(self.pitch_curve_select_action)
        self.toolbox_toolbar.addAction(self.pitch_curve_point_action)
        self.vibrato_tool_container = QWidget(self.toolbox_toolbar)
        vibrato_layout = QVBoxLayout(self.vibrato_tool_container)
        vibrato_layout.setContentsMargins(0, 0, 0, 0)
        vibrato_layout.setSpacing(1)
        self.vibrato_tool_button = QToolButton(self.vibrato_tool_container)
        self.vibrato_tool_button.setDefaultAction(self.vibrato_pitch_curve_action)
        self.vibrato_tool_button.setIconSize(QSize(25, 25))
        self.vibrato_tool_button.setFixedSize(QSize(38, 30))
        vibrato_layout.addWidget(self.vibrato_tool_button)
        self.vibrato_menu_button = QToolButton(self.vibrato_tool_container)
        self.vibrato_menu_button.setObjectName("VibratoMenuButton")
        self.vibrato_menu_button.setText("▾")
        self.vibrato_menu_button.setFixedSize(QSize(38, 14))
        vibrato_layout.addWidget(self.vibrato_menu_button)
        self.vibrato_toolbar_action = self.toolbox_toolbar.addWidget(
            self.vibrato_tool_container
        )
        self.vibrato_toolbar_action.setVisible(False)
        self.pitch_segment_shape_container = QWidget(self.toolbox_toolbar)
        pitch_segment_layout = QVBoxLayout(self.pitch_segment_shape_container)
        pitch_segment_layout.setContentsMargins(0, 0, 0, 0)
        pitch_segment_layout.setSpacing(1)
        self.pitch_segment_shape_button = QToolButton(self.pitch_segment_shape_container)
        self.pitch_segment_shape_button.setDefaultAction(self.pitch_segment_shape_action)
        self.pitch_segment_shape_button.setIconSize(QSize(25, 25))
        self.pitch_segment_shape_button.setFixedSize(QSize(38, 30))
        pitch_segment_layout.addWidget(self.pitch_segment_shape_button)
        self.pitch_segment_menu_button = QToolButton(self.pitch_segment_shape_container)
        self.pitch_segment_menu_button.setObjectName("PitchSegmentMenuButton")
        self.pitch_segment_menu_button.setText("▾")
        self.pitch_segment_menu_button.setFixedSize(QSize(38, 14))
        pitch_segment_layout.addWidget(self.pitch_segment_menu_button)
        self.pitch_segment_toolbar_action = self.toolbox_toolbar.addWidget(
            self.pitch_segment_shape_container
        )
        self.pitch_segment_toolbar_action.setVisible(False)
        self.pitch_glide_shape_container = QWidget(self.toolbox_toolbar)
        pitch_glide_layout = QVBoxLayout(self.pitch_glide_shape_container)
        pitch_glide_layout.setContentsMargins(0, 0, 0, 0)
        pitch_glide_layout.setSpacing(1)
        self.pitch_glide_shape_button = QToolButton(self.pitch_glide_shape_container)
        self.pitch_glide_shape_button.setDefaultAction(self.pitch_glide_shape_action)
        self.pitch_glide_shape_button.setIconSize(QSize(25, 25))
        self.pitch_glide_shape_button.setFixedSize(QSize(38, 30))
        pitch_glide_layout.addWidget(self.pitch_glide_shape_button)
        self.pitch_glide_menu_button = QToolButton(self.pitch_glide_shape_container)
        self.pitch_glide_menu_button.setObjectName("PitchGlideMenuButton")
        self.pitch_glide_menu_button.setText("▾")
        self.pitch_glide_menu_button.setFixedSize(QSize(38, 14))
        pitch_glide_layout.addWidget(self.pitch_glide_menu_button)
        self.pitch_glide_toolbar_action = self.toolbox_toolbar.addWidget(
            self.pitch_glide_shape_container
        )
        self.pitch_glide_toolbar_action.setVisible(False)
        self.scissors_tool_menu = QMenu(self.scissors_tool_container)
        self.scissors_tool_menu_style = ToolPopupStyle()
        self.scissors_tool_menu_style.setParent(self.scissors_tool_menu)
        self.scissors_tool_menu.setStyle(self.scissors_tool_menu_style)
        self.scissors_tool_menu.addAction(self.scissors_plain_menu_action)
        self.scissors_tool_menu.addAction(self.scissors_merge_menu_action)
        self.scissors_menu_button.clicked.connect(self._show_scissors_menu)
        self.vibrato_waveform_menu = QMenu(self.vibrato_tool_container)
        self.vibrato_waveform_menu_style = ToolPopupStyle()
        self.vibrato_waveform_menu_style.setParent(self.vibrato_waveform_menu)
        self.vibrato_waveform_menu.setStyle(self.vibrato_waveform_menu_style)
        self.vibrato_waveform_menu.addAction(self.vibrato_sine_action)
        self.vibrato_waveform_menu.addAction(self.vibrato_triangle_action)
        self.vibrato_waveform_menu.addAction(self.vibrato_square_action)
        self.vibrato_menu_button.clicked.connect(self._show_vibrato_waveform_menu)
        self.pitch_segment_shape_menu = QMenu(self.pitch_segment_shape_button)
        self.pitch_segment_shape_menu_style = ToolPopupStyle()
        self.pitch_segment_shape_menu_style.setParent(self.pitch_segment_shape_menu)
        self.pitch_segment_shape_menu.setStyle(self.pitch_segment_shape_menu_style)
        self.pitch_segment_shape_menu.addAction(self.pitch_segment_original_action)
        self.pitch_segment_shape_menu.addAction(self.pitch_segment_linear_action)
        self.pitch_segment_shape_menu.addAction(self.pitch_segment_smooth_action)
        self.pitch_segment_menu_button.clicked.connect(self._show_pitch_segment_shape_menu)
        self.pitch_glide_shape_menu = QMenu(self.pitch_glide_shape_button)
        self.pitch_glide_shape_menu_style = ToolPopupStyle()
        self.pitch_glide_shape_menu_style.setParent(self.pitch_glide_shape_menu)
        self.pitch_glide_shape_menu.setStyle(self.pitch_glide_shape_menu_style)
        self.pitch_glide_shape_menu.addAction(self.pitch_glide_ease_in_action)
        self.pitch_glide_shape_menu.addAction(self.pitch_glide_ease_out_action)
        self.pitch_glide_shape_menu.addAction(self.pitch_glide_s_curve_action)
        self.pitch_glide_shape_menu.addAction(self.pitch_glide_instant_action)
        self.pitch_glide_menu_button.clicked.connect(self._show_pitch_glide_shape_menu)
        self.vibrato_tool_button.setContextMenuPolicy(Qt.CustomContextMenu)
        self.vibrato_tool_button.customContextMenuRequested.connect(
            lambda _pos: self._cycle_vibrato_waveform()
        )
        self.scissors_tool_button.setContextMenuPolicy(Qt.CustomContextMenu)
        self.scissors_tool_button.customContextMenuRequested.connect(
            lambda _pos: self.scissors_merge_action.trigger()
        )
        for action in (self.select_tool_action, self.amplitude_tool_action):
            button = self.toolbox_toolbar.widgetForAction(action)
            if button is not None:
                button.setContextMenuPolicy(Qt.PreventContextMenu)
        self.root_label = QLabel()
        self.scale_label = QLabel()
        self.language_label = QLabel()
        self.addToolBar(Qt.TopToolBarArea, self.toolbox_toolbar)

    def _show_scissors_menu(self) -> None:
        self.scissors_tool_menu.exec(
            self.scissors_menu_button.mapToGlobal(self.scissors_menu_button.rect().bottomLeft())
        )

    def _show_vibrato_waveform_menu(self) -> None:
        self.vibrato_waveform_menu.exec(
            self.vibrato_menu_button.mapToGlobal(self.vibrato_menu_button.rect().bottomLeft())
        )

    def _show_pitch_segment_shape_menu(self) -> None:
        self.pitch_segment_shape_menu.exec(
            self.pitch_segment_menu_button.mapToGlobal(
                self.pitch_segment_menu_button.rect().bottomLeft()
            )
        )

    def _show_pitch_glide_shape_menu(self) -> None:
        self.pitch_glide_shape_menu.exec(
            self.pitch_glide_menu_button.mapToGlobal(
                self.pitch_glide_menu_button.rect().bottomLeft()
            )
        )

    def _cycle_vibrato_waveform(self) -> None:
        actions = [
            self.vibrato_sine_action,
            self.vibrato_triangle_action,
            self.vibrato_square_action,
        ]
        current = self.vibrato_waveform_group.checkedAction()
        try:
            next_action = actions[(actions.index(current) + 1) % len(actions)]
        except ValueError:
            next_action = actions[0]
        next_action.setChecked(True)
        self._on_vibrato_waveform_changed(next_action)

    def _build_menu_bar(self) -> None:
        self.file_menu = self.menuBar().addMenu("")
        self.file_menu.addAction(self.new_project_action)
        self.file_menu.addAction(self.open_project_action)
        self.file_menu.addAction(self.open_autosave_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.save_project_action)
        self.file_menu.addAction(self.save_project_as_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.export_action)

        self.edit_menu = self.menuBar().addMenu("")
        self.edit_menu.addAction(self.undo_action)
        self.edit_menu.addAction(self.redo_action)
        self.edit_menu.addSeparator()
        self.edit_menu.addAction(self.select_all_action)
        self.edit_menu.addAction(self.copy_action)
        self.edit_menu.addAction(self.paste_action)
        self.edit_menu.addAction(self.duplicate_action)
        self.edit_menu.addSeparator()
        self.edit_menu.addAction(self.delete_selected_action)

        self.tools_menu = self.menuBar().addMenu("")
        self.tools_menu.addAction(self.select_tool_action)
        self.tools_menu.addAction(self.scissors_tool_action)
        self.tools_menu.addAction(self.scissors_merge_action)
        self.tools_menu.addAction(self.amplitude_tool_action)
        self.tools_menu.addSeparator()
        self.tools_menu.addAction(self.flatten_tool_action)
        self.flatten_menu_label = QLabel()
        self.tools_menu.addAction(
            self._create_menu_control_action(self.flatten_menu_label, self.flatten_slider)
        )
        self.tools_menu.addSeparator()
        self.tools_menu.addAction(self.formant_tool_action)
        self.formant_menu_label = QLabel()
        self.tools_menu.addAction(
            self._create_menu_control_action(self.formant_menu_label, self.formant_slider)
        )
        self.transport_menu = self.menuBar().addMenu("")
        self.transport_menu.addAction(self.play_pause_action)
        self.transport_menu.addAction(self.stop_action)
        self.transport_menu.addAction(self.return_start_action)

        self.settings_menu = self.menuBar().addMenu("")
        self.settings_menu.addAction(
            self._create_menu_control_action(self.root_label, self.root_note_combo)
        )
        self.settings_menu.addAction(
            self._create_menu_control_action(self.scale_label, self.scale_type_combo)
        )
        self.settings_menu.addAction(
            self._create_menu_control_action(self.language_label, self.language_combo)
        )
        self.settings_menu.addSeparator()
        self.settings_menu.addAction(self.audio_settings_action)

        self.toolbox_toggle_action = self.toolbox_toolbar.toggleViewAction()
        self.inspector_toggle_action = self.inspector_dock.toggleViewAction()
        self.material_toggle_action = QAction(self)
        self.material_toggle_action.setCheckable(True)
        self.material_toggle_action.setChecked(True)
        self.material_toggle_action.toggled.connect(self.material_panel.setVisible)
        self.track_toggle_action = QAction(self)
        self.track_toggle_action.setCheckable(True)
        self.track_toggle_action.setChecked(True)
        self.track_toggle_action.toggled.connect(self.track_panel.setVisible)

        self.view_menu = self.menuBar().addMenu("")
        self.view_menu.addAction(self.toolbox_toggle_action)
        self.view_menu.addAction(self.inspector_toggle_action)
        self.view_menu.addAction(self.material_toggle_action)
        self.view_menu.addAction(self.track_toggle_action)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.zoom_in_action)
        self.view_menu.addAction(self.zoom_out_action)
        self.view_menu.addAction(self.zoom_reset_action)

        self.help_menu = self.menuBar().addMenu("")
        self.help_menu.addAction(self.diagnostics_action)

    def _create_menu_control_action(
        self,
        label: QLabel,
        control: QWidget,
    ) -> QWidgetAction:
        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)
        label.setMinimumWidth(58)
        layout.addWidget(label)
        layout.addWidget(control, 1)
        action = QWidgetAction(self)
        action.setDefaultWidget(widget)
        return action

    def _on_language_changed(self) -> None:
        language = self.language_combo.currentData() or "zh"
        self.set_language(str(language))

    def _on_pitch_curve_view_toggled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self.pitch_curve_select_action.setVisible(enabled)
        self.pitch_curve_point_action.setVisible(enabled)
        self.vibrato_pitch_curve_action.setVisible(enabled)
        self.vibrato_toolbar_action.setVisible(enabled)
        self.pitch_segment_toolbar_action.setVisible(False)
        self.pitch_glide_toolbar_action.setVisible(False)
        self.pitch_segment_shape_container.setEnabled(False)
        self.pitch_glide_shape_container.setEnabled(False)
        self.vibrato_tool_container.setEnabled(enabled)
        self.pitch_curve_select_action.setEnabled(enabled)
        self.pitch_curve_point_action.setEnabled(enabled)
        self.vibrato_menu_button.setEnabled(enabled)
        self._on_pitch_curve_selection_changed(
            self.workspace.has_selected_pitch_curve_range()
        )
        if not enabled:
            if (
                self.pitch_curve_select_action.isChecked()
                or self.pitch_curve_point_action.isChecked()
                or self.vibrato_pitch_curve_action.isChecked()
            ):
                self.select_tool_action.setChecked(True)
                self.workspace.set_tool_mode("select")
            self.workspace.set_pitch_curve_tool_mode("none")

    def _on_pitch_curve_tool_action_triggered(self, action) -> None:
        mode = action.data() if hasattr(action, "data") else "curve_select"
        self.workspace.set_pitch_curve_tool_mode(str(mode or "curve_select"))

    def _on_pitch_curve_selection_changed(self, has_line_selection: bool) -> None:
        view_enabled = self.pitch_curve_select_action.isVisible()
        vibrato_enabled = view_enabled and bool(has_line_selection)
        self.vibrato_pitch_curve_action.setEnabled(vibrato_enabled)
        self.vibrato_tool_button.setEnabled(vibrato_enabled)
        self.vibrato_tool_container.setEnabled(vibrato_enabled)
        self.vibrato_menu_button.setEnabled(vibrato_enabled)
        segment_enabled = False
        self.pitch_segment_shape_button.setEnabled(segment_enabled)
        self.pitch_segment_menu_button.setEnabled(segment_enabled)
        self.pitch_segment_shape_container.setEnabled(segment_enabled)
        self.pitch_glide_shape_button.setEnabled(segment_enabled)
        self.pitch_glide_menu_button.setEnabled(segment_enabled)
        self.pitch_glide_shape_container.setEnabled(segment_enabled)
        for action in (
            self.pitch_segment_shape_action,
            self.pitch_segment_original_action,
            self.pitch_segment_linear_action,
            self.pitch_segment_smooth_action,
            self.pitch_glide_shape_action,
            self.pitch_glide_ease_in_action,
            self.pitch_glide_ease_out_action,
            self.pitch_glide_s_curve_action,
            self.pitch_glide_instant_action,
        ):
            action.setEnabled(segment_enabled)

    def _on_vibrato_waveform_changed(self, action: QAction) -> None:
        waveform = str(action.data() or "sine")
        self.workspace.set_pitch_curve_vibrato_waveform(waveform)

    def _choose_pitch_curve_segment_shape(self, shape: str) -> None:
        shape = str(shape or "linear")
        if shape in {"original", "linear", "smooth"}:
            self._last_pitch_segment_shape = shape
        elif shape in {"ease_in", "ease_out", "s_curve", "instant"}:
            self._last_pitch_glide_shape = shape
        self._apply_pitch_curve_segment_shape(shape)

    def _apply_pitch_curve_segment_shape(self, shape: str) -> None:
        changed = self.workspace.apply_pitch_curve_segment_shape(shape)
        language = str(self.language_combo.currentData() or "zh")
        text = self.TEXT.get(language, self.TEXT["zh"])
        label = {
            "original": text["pitch_segment_original"],
            "linear": text["pitch_segment_linear"],
            "smooth": text["pitch_segment_smooth"],
            "ease_in": text["pitch_glide_ease_in"],
            "ease_out": text["pitch_glide_ease_out"],
            "s_curve": text["pitch_glide_s_curve"],
            "instant": text["pitch_glide_instant"],
        }.get(shape, shape)
        if changed:
            self.statusBar().showMessage(f"{label} · {changed} 段", 1800)
        else:
            self.statusBar().showMessage("请先用 V 选中一段音高线", 1800)

    def set_language(self, language: str) -> None:
        language = "zh" if language == "zh" else "en"
        text = self.TEXT[language]

        self.file_menu.setTitle(text["file"])
        self.edit_menu.setTitle(text["edit"])
        self.tools_menu.setTitle(text["tools"])
        self.transport_menu.setTitle(text["transport"])
        self.settings_menu.setTitle(text["settings"])
        self.view_menu.setTitle(text["view"])
        self.help_menu.setTitle(text["help"])
        self.new_project_action.setText(text["new"])
        self.open_project_action.setText(text["open"])
        self.open_autosave_action.setText(text["open_autosave"])
        self.save_project_action.setText(text["save"])
        self.save_project_as_action.setText(text["save_as"])
        self.export_action.setText(text["export"])
        self.diagnostics_action.setText(text["diagnostics"])
        self.audio_settings_action.setText(text["audio_settings"])
        self.play_pause_action.setText(text["play_pause"])
        self.stop_action.setText(text["stop"])
        self.return_start_action.setText(text["return_start"])
        self.play_pause_action.setToolTip(text["play_pause"])
        self.stop_action.setToolTip(text["stop"])
        self.undo_action.setText(text["undo"])
        self.redo_action.setText(text["redo"])
        self.select_all_action.setText(text["select_all"])
        self.copy_action.setText(text["copy"])
        self.paste_action.setText(text["paste"])
        self.duplicate_action.setText(text["duplicate"])
        self.delete_selected_action.setText(text["delete"])

        self.left_panel.set_title(text["piano_roll"])
        self.workspace_panel.set_title(text["workspace"])
        self.track_panel.set_title(text["tracks"])
        self.material_panel.set_title(text["materials"])
        self.inspector_dock.setWindowTitle(text["inspector"])
        self.toolbox_toolbar.setWindowTitle(text["toolbox"])
        self.toolbox_toggle_action.setText(text["show_toolbox"])
        self.inspector_toggle_action.setText(text["show_inspector"])
        self.material_toggle_action.setText(text["show_materials"])
        self.track_toggle_action.setText(text["show_tracks"])
        self.zoom_in_action.setText(text["zoom_in"])
        self.zoom_out_action.setText(text["zoom_out"])
        self.zoom_reset_action.setText(text["zoom_reset"])

        self.select_tool_action.setText(text["select_tool"])
        self.scissors_tool_action.setText(text["scissors_tool"])
        self.scissors_merge_action.setText(f"{text['cut_merge_tool']} (D)")
        self.scissors_plain_menu_action.setText(f"{text['scissors_tool']} (X)")
        self.scissors_merge_menu_action.setText(f"{text['cut_merge_tool']} (D)")
        self.amplitude_tool_action.setText(text["amplitude_tool"])
        self.flatten_tool_action.setText(text["flatten_tool"])
        self.formant_tool_action.setText(text["formant_tool"])
        self.pitch_curve_select_action.setText(text["pitch_curve_select_tool"])
        self.pitch_curve_point_action.setText(text["pitch_curve_point_tool"])
        self.copy_pitch_curve_action.setText(f"{text['copy_pitch_curve']} (Ctrl+Alt+C)")
        self.paste_pitch_curve_action.setText(f"{text['paste_pitch_curve']} (Ctrl+Alt+V)")
        self.smooth_pitch_curve_action.setText(text["smooth_pitch_curve"])
        self.vibrato_pitch_curve_action.setText(text["vibrato_pitch_curve"])
        self.vibrato_sine_action.setText(f"{text['vibrato_sine']} (Alt+1)")
        self.vibrato_triangle_action.setText(f"{text['vibrato_triangle']} (Alt+2)")
        self.vibrato_square_action.setText(f"{text['vibrato_square']} (Alt+3)")
        self.pitch_segment_shape_action.setText(text["pitch_segment_shape"])
        self.pitch_segment_original_action.setText(text["pitch_segment_original"])
        self.pitch_segment_linear_action.setText(text["pitch_segment_linear"])
        self.pitch_segment_smooth_action.setText(text["pitch_segment_smooth"])
        self.pitch_glide_shape_action.setText(text["pitch_glide_shape"])
        self.pitch_glide_ease_in_action.setText(text["pitch_glide_ease_in"])
        self.pitch_glide_ease_out_action.setText(text["pitch_glide_ease_out"])
        self.pitch_glide_s_curve_action.setText(text["pitch_glide_s_curve"])
        self.pitch_glide_instant_action.setText(text["pitch_glide_instant"])
        self.pitch_zero_action.setText(text["pitch_zero"])
        self.pitch_natural_action.setText(text["pitch_natural"])
        self.pitch_electro_action.setText(text["pitch_electro"])
        self.select_tool_action.setToolTip(f"{text['select_tool']} (Z)")
        self.scissors_tool_action.setToolTip(
            f"{text['scissors_tool']} (X)\n"
            f"{text['cut_merge_tool']}：右键 / D\n"
            f"临时切换：Shift"
        )
        self.scissors_merge_action.setToolTip(
            f"{text['cut_merge_tool']} (D)\n"
            f"分割工具下按住 Alt，或选择工具下 Alt+Shift，可临时使用。"
        )
        self.amplitude_tool_action.setToolTip(f"{text['amplitude_tool']} (C)")
        for action in (
            self.select_tool_action,
            self.scissors_tool_action,
            self.amplitude_tool_action,
        ):
            button = self.toolbox_toolbar.widgetForAction(action)
            if button is not None:
                button.setToolTip(action.toolTip())
        self.flatten_tool_action.setToolTip(text["flatten_tool"])
        self.formant_tool_action.setToolTip(text["formant_tool"])
        self.pitch_curve_select_action.setToolTip(
            f"{text['pitch_curve_select_tool']} (V)\nAlt: V/B"
        )
        self.pitch_curve_point_action.setToolTip(
            f"{text['pitch_curve_point_tool']} (B)\nAlt: V/B"
        )
        self.copy_pitch_curve_action.setToolTip(text["copy_pitch_curve"])
        self.paste_pitch_curve_action.setToolTip(text["paste_pitch_curve"])
        self.smooth_pitch_curve_action.setToolTip(text["smooth_pitch_curve"])
        self.vibrato_pitch_curve_action.setToolTip(
            f"{text['vibrato_pitch_curve']} (N)\n"
            "按住音高线段拖动；左右调周期，上下调振幅"
        )
        self.vibrato_tool_button.setToolTip(self.vibrato_pitch_curve_action.toolTip())
        self.vibrato_menu_button.setToolTip("选择颤音波形")
        self.pitch_segment_shape_action.setToolTip(
            f"{text['pitch_segment_shape']}\n"
            f"{text['pitch_segment_original']} / {text['pitch_segment_linear']} / {text['pitch_segment_smooth']}"
        )
        self.pitch_segment_shape_button.setToolTip(self.pitch_segment_shape_action.toolTip())
        self.pitch_glide_shape_action.setToolTip(
            f"{text['pitch_glide_shape']}\n"
            f"{text['pitch_glide_ease_in']} / {text['pitch_glide_ease_out']} / "
            f"{text['pitch_glide_s_curve']} / {text['pitch_glide_instant']}"
        )
        self.pitch_glide_shape_button.setToolTip(self.pitch_glide_shape_action.toolTip())
        self.pitch_zero_action.setToolTip(text["pitch_zero"])
        self.pitch_natural_action.setToolTip(text["pitch_natural"])
        self.pitch_electro_action.setToolTip(text["pitch_electro"])
        self.flatten_slider.setToolTip(text["flatten_tip"])
        self.formant_slider.setToolTip(text["formant_tip"])
        self.root_label.setText(text["root"])
        self.scale_label.setText(text["scale"])
        self.language_label.setText(text["language"])
        self.flatten_menu_label.setText(text["flatten_tool"])
        self.formant_menu_label.setText(text["formant_tool"])
        self.root_note_combo.setToolTip(text["root_tip"])
        self.scale_type_combo.setToolTip(text["scale_tip"])
        self.language_combo.setToolTip(text["language_tip"])

        current_scale = self.current_scale_type()
        self.scale_type_combo.blockSignals(True)
        for index, scale_type in enumerate(self.SCALE_ORDER):
            label_key = self.SCALE_TEXT_KEYS[scale_type]
            self.scale_type_combo.setItemText(index, text[label_key])
        scale_index = self.scale_type_combo.findData(current_scale)
        if scale_index >= 0:
            self.scale_type_combo.setCurrentIndex(scale_index)
        self.scale_type_combo.blockSignals(False)

        target_language_index = self.language_combo.findData(language)
        if target_language_index >= 0:
            self.language_combo.blockSignals(True)
            self.language_combo.setCurrentIndex(target_language_index)
            self.language_combo.blockSignals(False)

        self.inspector_widget.set_language(language)
        self.material_browser.set_language(language)
        self.statusBar().showMessage(text["initialized"])
        self.update_scissors_merge_ui()

    def current_language(self) -> str:
        data = self.language_combo.currentData()
        return "zh" if data == "zh" else "en"

    def current_scale_type(self) -> str:
        data = self.scale_type_combo.currentData()
        if data:
            return str(data)
        return self.scale_type_combo.currentText()

    def set_transient_protection_enabled(self, enabled: bool) -> None:
        self._transient_protection_enabled = bool(enabled)
        self.inspector_widget.set_transient_protection_enabled(enabled)

    def update_scissors_merge_ui(self, *_args) -> None:
        if not hasattr(self, "scissors_merge_action"):
            return
        enabled = self.scissors_merge_action.isChecked()
        self.scissors_tool_action.setIcon(
            self._make_tool_icon("cut_merge" if enabled else "scissors")
        )
        if hasattr(self, "scissors_tool_button"):
            self.scissors_tool_button.update()

    def _activate_plain_scissors(self) -> None:
        self.scissors_merge_action.setChecked(False)
        self.scissors_tool_action.trigger()

    def _activate_scissors_merge(self) -> None:
        self.scissors_merge_action.setChecked(True)
        self.scissors_tool_action.setChecked(True)

    def _show_diagnostics_dialog(self) -> None:
        dialog = DiagnosticsDialog(self)
        dialog.exec() if hasattr(dialog, "exec") else dialog.exec_()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.about_to_close.emit()
        super().closeEvent(event)

    def _create_tool_action(self, tool_name: str) -> QAction:
        action = QAction(self._make_tool_icon(tool_name), "", self)
        action.setCheckable(True)
        action.setData(tool_name)
        self.tool_action_group.addAction(action)
        return action

    def _create_pitch_curve_tool_action(self, tool_name: str) -> QAction:
        action = QAction(self._make_tool_icon(tool_name), "", self)
        action.setCheckable(True)
        action.setData(tool_name)
        self.tool_action_group.addAction(action)
        return action

    def _make_tool_icon(self, tool_name: str) -> QIcon:
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        light = QColor("#d4d4d4")
        accent = QColor("#3f8ec5")
        orange = QColor("#ff9f2f")
        painter.setPen(QPen(light, 2))

        if tool_name == "select":
            path = QPainterPath()
            path.moveTo(6, 4)
            path.lineTo(17, 13)
            path.lineTo(12, 14)
            path.lineTo(15, 20)
            path.lineTo(12, 21)
            path.lineTo(9, 15)
            path.lineTo(6, 18)
            path.closeSubpath()
            painter.fillPath(path, light)
        elif tool_name == "scissors":
            painter.drawEllipse(4, 15, 5, 5)
            painter.drawEllipse(15, 15, 5, 5)
            painter.drawLine(8, 16, 19, 5)
            painter.drawLine(16, 16, 5, 5)
        elif tool_name == "cut_merge":
            merge_accent = QColor("#f4f8ff")
            painter.fillRect(11, 2, 3, 20, orange)
            painter.fillRect(2, 10, 6, 5, merge_accent)
            left_arrow = QPainterPath()
            left_arrow.moveTo(11, 12.5)
            left_arrow.lineTo(7, 6)
            left_arrow.lineTo(7, 19)
            left_arrow.closeSubpath()
            painter.fillPath(left_arrow, merge_accent)
            painter.fillRect(16, 10, 6, 5, merge_accent)
            right_arrow = QPainterPath()
            right_arrow.moveTo(14, 12.5)
            right_arrow.lineTo(18, 6)
            right_arrow.lineTo(18, 19)
            right_arrow.closeSubpath()
            painter.fillPath(right_arrow, merge_accent)
        elif tool_name == "amplitude":
            painter.setPen(QPen(accent, 3))
            painter.drawLine(6, 17, 6, 8)
            painter.drawLine(12, 19, 12, 5)
            painter.drawLine(18, 16, 18, 10)
        elif tool_name == "pitch_curve_select":
            pointer = QPainterPath()
            pointer.moveTo(4, 3)
            pointer.lineTo(14, 12)
            pointer.lineTo(10, 13)
            pointer.lineTo(13, 20)
            pointer.lineTo(10, 21)
            pointer.lineTo(7, 14)
            pointer.lineTo(4, 17)
            pointer.closeSubpath()
            painter.fillPath(pointer, light)
            painter.setPen(QPen(accent, 2.0))
            path = QPainterPath()
            path.moveTo(11, 18)
            path.cubicTo(14, 12, 17, 22, 20, 16)
            path.cubicTo(22, 11, 24, 13, 26, 10)
            painter.drawPath(path)
        elif tool_name == "pitch_curve_point":
            painter.setPen(QPen(accent, 2.2))
            path = QPainterPath()
            path.moveTo(3, 16)
            path.cubicTo(7, 7, 11, 21, 15, 12)
            path.cubicTo(18, 6, 20, 9, 22, 7)
            painter.drawPath(path)
            painter.setBrush(QColor("#fff3a6"))
            painter.setPen(QPen(QColor("#10141a"), 1.1))
            painter.drawEllipse(QRectF(10, 9, 7, 7))
        elif tool_name == "pitch_point":
            painter.setPen(QPen(accent, 2.2))
            path = QPainterPath()
            path.moveTo(3, 16)
            path.cubicTo(7, 7, 11, 21, 15, 12)
            path.cubicTo(18, 6, 20, 9, 22, 7)
            painter.drawPath(path)
            painter.setBrush(QColor("#fff3a6"))
            painter.setPen(QPen(QColor("#10141a"), 1.1))
            painter.drawEllipse(QRectF(10, 9, 7, 7))
        elif tool_name == "pitch_vibrato":
            painter.setPen(QPen(QColor("#fff3a6"), 1.4))
            painter.drawLine(3, 18, 21, 18)
            painter.setPen(QPen(accent, 2.2))
            path = QPainterPath()
            path.moveTo(3, 13)
            path.cubicTo(5, 5, 8, 21, 10, 13)
            path.cubicTo(12, 5, 15, 21, 17, 13)
            path.cubicTo(18.5, 8, 20, 10, 21, 9)
            painter.drawPath(path)
        elif tool_name == "pitch_segment_shape":
            painter.setPen(QPen(QColor("#fff3a6"), 1.6))
            painter.setBrush(QColor("#fff3a6"))
            painter.drawEllipse(QRectF(3.5, 15.5, 4.5, 4.5))
            painter.drawEllipse(QRectF(16.0, 5.0, 4.5, 4.5))
            painter.setPen(QPen(accent, 2.2))
            path = QPainterPath()
            path.moveTo(6, 18)
            path.cubicTo(9, 18, 13, 7, 18, 7)
            painter.drawPath(path)
            painter.setPen(QPen(light, 1.3))
            painter.drawLine(4, 21, 21, 21)
        elif tool_name == "pitch_glide_shape":
            painter.setPen(QPen(QColor("#fff3a6"), 1.6))
            painter.setBrush(QColor("#fff3a6"))
            painter.drawEllipse(QRectF(3.5, 16.0, 4.5, 4.5))
            painter.drawEllipse(QRectF(16.0, 4.5, 4.5, 4.5))
            painter.setPen(QPen(accent, 2.2))
            path = QPainterPath()
            path.moveTo(6, 18)
            path.cubicTo(6, 16, 9, 9, 18, 7)
            painter.drawPath(path)
            painter.setPen(QPen(light, 1.2))
            painter.drawLine(4, 12, 8, 12)
            painter.drawLine(16, 12, 20, 12)
        elif tool_name == "flatten":
            painter.setPen(QPen(orange, 2))
            path = QPainterPath()
            path.moveTo(3, 13)
            path.cubicTo(6, 5, 9, 21, 12, 13)
            path.cubicTo(15, 5, 18, 21, 21, 13)
            painter.drawPath(path)
            painter.setPen(QPen(light, 1.5))
            painter.drawLine(4, 18, 20, 18)
        elif tool_name == "formant":
            painter.setPen(QPen(light, 1.8))
            painter.drawLine(4, 19, 20, 19)
            painter.setPen(QPen(accent, 2))
            path = QPainterPath()
            path.moveTo(4, 17)
            path.cubicTo(8, 5, 12, 6, 15, 12)
            path.cubicTo(17, 15, 19, 10, 21, 7)
            painter.drawPath(path)

        painter.end()
        return self._stable_icon(pixmap, self._white_icon_pixmap(pixmap))

    def _make_scissors_button_icon(self, tool_name: str) -> QIcon:
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        light = QColor("#d4d4d4")
        painter.setPen(QPen(light, 2.5))
        if tool_name == "cut_merge":
            painter.fillRect(14, 1, 4, 21, QColor("#ff9f2f"))
            painter.fillRect(3, 9, 8, 5, light)
            painter.fillRect(21, 9, 8, 5, light)
            left_arrow = QPainterPath()
            left_arrow.moveTo(14, 11.5)
            left_arrow.lineTo(9, 5)
            left_arrow.lineTo(9, 18)
            left_arrow.closeSubpath()
            painter.fillPath(left_arrow, light)
            right_arrow = QPainterPath()
            right_arrow.moveTo(18, 11.5)
            right_arrow.lineTo(23, 5)
            right_arrow.lineTo(23, 18)
            right_arrow.closeSubpath()
            painter.fillPath(right_arrow, light)
        else:
            painter.drawEllipse(4, 15, 7, 7)
            painter.drawEllipse(21, 15, 7, 7)
            painter.drawLine(9, 16, 27, 3)
            painter.drawLine(23, 16, 5, 3)
        triangle = QPainterPath()
        triangle.moveTo(10, 25)
        triangle.lineTo(22, 25)
        triangle.lineTo(16, 31)
        triangle.closeSubpath()
        painter.fillPath(triangle, light)
        painter.end()
        return self._stable_icon(pixmap, self._white_icon_pixmap(pixmap))

    @staticmethod
    def _white_icon_pixmap(pixmap: QPixmap) -> QPixmap:
        selected_pixmap = QPixmap(pixmap.size())
        selected_pixmap.fill(Qt.transparent)
        painter = QPainter(selected_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.drawPixmap(0, 0, pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(selected_pixmap.rect(), QColor("#ffffff"))
        painter.end()
        return selected_pixmap

    @staticmethod
    def _stable_icon(pixmap: QPixmap, selected_pixmap: QPixmap | None = None) -> QIcon:
        """Keep custom artwork stable, with a dedicated visible checked state."""
        checked_pixmap = selected_pixmap or pixmap
        icon = QIcon()
        for mode in (QIcon.Normal, QIcon.Active, QIcon.Selected):
            icon.addPixmap(pixmap, mode, QIcon.Off)
            icon.addPixmap(checked_pixmap, mode, QIcon.On)
        return icon

    def _make_transport_icon(self, action_name: str) -> QIcon:
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        light = QColor("#d4d4d4")
        accent = QColor("#3f8ec5")
        painter.setPen(QPen(light, 2))
        painter.setBrush(accent)

        if action_name == "play":
            path = QPainterPath()
            path.moveTo(7, 5)
            path.lineTo(19, 12)
            path.lineTo(7, 19)
            path.closeSubpath()
            painter.fillPath(path, accent)
        elif action_name == "stop":
            painter.fillRect(7, 7, 10, 10, accent)

        painter.end()
        return QIcon(pixmap)
