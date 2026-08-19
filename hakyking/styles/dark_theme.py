from __future__ import annotations

from hakyking.qt import QApplication, QColor, QPalette


def apply_dark_theme(app: QApplication) -> None:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#1e1e1e"))
    palette.setColor(QPalette.WindowText, QColor("#d4d4d4"))
    palette.setColor(QPalette.Base, QColor("#252526"))
    palette.setColor(QPalette.AlternateBase, QColor("#2d2d30"))
    palette.setColor(QPalette.ToolTipBase, QColor("#2d2d30"))
    palette.setColor(QPalette.ToolTipText, QColor("#d4d4d4"))
    palette.setColor(QPalette.Text, QColor("#d4d4d4"))
    palette.setColor(QPalette.Button, QColor("#333337"))
    palette.setColor(QPalette.ButtonText, QColor("#d4d4d4"))
    palette.setColor(QPalette.Highlight, QColor("#3f8ec5"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    app.setStyleSheet(
        """
        * {
            selection-background-color: #3f8ec5;
            selection-color: #ffffff;
        }
        QMainWindow, QWidget {
            background: #1e1e1e;
            color: #d4d4d4;
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
            font-size: 10pt;
        }
        QToolTip {
            background: #2d2d30;
            color: #f0f0f0;
            border: 1px solid #505055;
            padding: 5px 7px;
        }
        QLabel {
            color: #d4d4d4;
            background: transparent;
        }
        QFrame#Panel {
            background: #2d2d30;
            border: none;
            border-radius: 3px;
        }
        QLabel#PanelTitle {
            color: #dcdcdc;
            font-weight: 600;
            padding: 0px 8px;
            background: #252526;
            border: none;
            border-bottom: 1px solid #383838;
        }
        QToolBar {
            background: #252526;
            border: none;
            border-bottom: 1px solid #343434;
            spacing: 6px;
            padding: 5px 8px;
        }
        QToolButton {
            background: #333337;
            color: #d4d4d4;
            border: none;
            border-radius: 4px;
            padding: 5px 10px;
        }
        QToolButton:hover {
            background: #3f3f46;
        }
        QToolButton:pressed {
            background: #4b4b52;
        }
        QToolButton:checked {
            background: #3f8ec5;
            color: #ffffff;
        }
        QToolButton:checked:hover {
            background: #4ea0d6;
        }
        QToolButton::menu-indicator {
            width: 8px;
            height: 8px;
            subcontrol-origin: padding;
            subcontrol-position: bottom center;
            bottom: 0px;
        }
        QToolButton#ScissorsToolButton::menu-indicator {
            image: none;
            width: 0px;
            height: 0px;
        }
        QToolButton#ScissorsMenuButton {
            background: #3b3b40;
            border: none;
            border-radius: 2px;
            padding: 0px;
            font-size: 13px;
            color: #eeeeee;
        }
        QToolButton#ScissorsMenuButton:hover {
            background: #505058;
        }
        QMenuBar {
            background: #252526;
            color: #d4d4d4;
            border: none;
            padding: 2px 6px;
        }
        QMenuBar::item {
            background: transparent;
            border-radius: 4px;
            padding: 5px 9px;
        }
        QMenuBar::item:selected {
            background: #3a3a3d;
        }
        QMenu {
            background: #2d2d30;
            color: #d4d4d4;
            border: 1px solid #414145;
            padding: 5px;
        }
        QMenu::item {
            border-radius: 4px;
            padding: 6px 28px 6px 18px;
        }
        QMenu::item:selected {
            background: #3f8ec5;
            color: #ffffff;
        }
        QMenu::separator {
            height: 1px;
            background: #3d3d42;
            margin: 5px 8px;
        }
        QDockWidget {
            background: #252526;
            color: #d4d4d4;
            titlebar-close-icon: none;
            titlebar-normal-icon: none;
        }
        QDockWidget::title {
            background: #252526;
            color: #dcdcdc;
            padding: 7px 9px;
            border-bottom: 1px solid #383838;
            font-weight: 600;
        }
        QPushButton {
            background: #333337;
            color: #d4d4d4;
            border: none;
            border-radius: 4px;
            padding: 6px 10px;
            min-height: 22px;
        }
        QPushButton:hover {
            background: #3f3f46;
            color: #ffffff;
        }
        QPushButton:pressed {
            background: #4b4b52;
        }
        QPushButton:checked {
            background: #3f8ec5;
            color: #ffffff;
        }
        QPushButton:disabled {
            background: #28282c;
            color: #77777e;
        }
        QPushButton#SoloButton:checked {
            background: #d6a526;
            color: #161616;
            font-weight: 700;
        }
        QPushButton#MuteButton:checked {
            background: #c94c4c;
            color: #ffffff;
            font-weight: 700;
        }
        QPushButton#LockButton:checked {
            background: #38506f;
            color: #eef5ff;
            font-weight: 700;
        }
        QPushButton#LockButton:disabled {
            background: #243247;
            color: #aeb8c6;
        }
        QGraphicsView {
            background: #1b1b1d;
            border: none;
        }
        QScrollArea {
            background: #2d2d30;
            border: none;
        }
        QComboBox, QSpinBox, QDoubleSpinBox {
            background: #303034;
            color: #d4d4d4;
            border: none;
            border-radius: 4px;
            padding: 4px 7px;
            min-height: 22px;
        }
        QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
            background: #3a3a3f;
        }
        QComboBox::drop-down {
            border: none;
            width: 18px;
        }
        QComboBox QAbstractItemView {
            background: #2d2d30;
            color: #d4d4d4;
            border: 1px solid #414145;
            selection-background-color: #3f8ec5;
        }
        QFrame#TrackRow {
            background: #303034;
            border: none;
            border-radius: 4px;
        }
        QFrame#TrackRow[selected="true"] {
            background: #344253;
            border-left: 3px solid #3f8ec5;
        }
        QFrame#TrackRow:hover {
            background: #36363b;
        }
        QFrame#TrackColorBar {
            background: #3f8ec5;
            border-radius: 2px;
        }
        QFrame#LeftTrackSpacer {
            background: #1b1b1d;
            border-top: 1px solid #33363c;
        }
        QLabel#TrackName {
            color: #d8d8d8;
            background: transparent;
        }
        QLabel#TrackStatus {
            color: #aeb6c2;
            background: transparent;
            font-size: 11px;
        }
        QLabel#InspectorTitle {
            color: #f0f3f7;
            font-weight: 700;
            font-size: 10pt;
        }
        QLabel#InspectorHint {
            color: #929aa5;
            font-size: 8pt;
        }
        QLabel#InspectorStatus {
            color: #9fa8b3;
            font-size: 8pt;
            font-weight: 500;
        }
        QFrame#InspectorRow {
            background: transparent;
            border: none;
            border-radius: 4px;
            padding: 0px;
        }
        QComboBox#TuningPresetCombo {
            min-height: 24px;
        }
        QPushButton#TuningPresetButton {
            background: #2f343b;
            color: #dce5ee;
            border: 1px solid #3d4650;
            border-radius: 4px;
            padding: 4px 6px;
            min-height: 20px;
            font-size: 9pt;
        }
        QPushButton#TuningPresetButton:hover {
            background: #3c4d5f;
            border-color: #5b7891;
            color: #ffffff;
        }
        QPushButton#TuningPresetButton:pressed {
            background: #3f8ec5;
            color: #ffffff;
        }
        QPushButton#TuningPresetButton:disabled {
            background: #252529;
            border-color: #2c2c31;
            color: #6f747b;
        }
        QPushButton#MaterialScissorsButton, QToolButton#MaterialScissorsButton {
            background: #333337;
            padding: 2px;
            min-width: 0;
            min-height: 0;
        }
        QPushButton#MaterialScissorsButton:hover, QToolButton#MaterialScissorsButton:hover {
            background: #3f8ec5;
        }
        QToolButton#MaterialScissorsButton:checked {
            background: #3f8ec5;
            color: #ffffff;
        }
        QPushButton#AddTrackButton {
            background: #2f3a45;
            color: #e1edf8;
            font-weight: 600;
        }
        QPushButton#TrackEditButton,
        QPushButton#TrackClearButton,
        QPushButton#SoloButton,
        QPushButton#MuteButton,
        QPushButton#LockButton {
            padding: 2px 4px;
            min-height: 0;
        }
        QPushButton#TrackClearButton {
            background: #3a3030;
            color: #e8c6c6;
            font-weight: 700;
        }
        QPushButton#TrackClearButton:hover {
            background: #8a3f3f;
            color: #ffffff;
        }
        QPushButton#TrackEditButton:checked {
            background: #3f8ec5;
            color: #ffffff;
            font-weight: 700;
        }
        QTreeView {
            background: #252526;
            alternate-background-color: #2a2a2d;
            border: none;
            border-radius: 3px;
            outline: 0;
            padding: 4px;
        }
        QTreeView::item {
            min-height: 23px;
            padding: 3px 5px;
            border-radius: 3px;
        }
        QTreeView::item:hover {
            background: #34343a;
        }
        QTreeView::item:selected {
            background: #3f8ec5;
            color: #ffffff;
        }
        QListWidget {
            background: #252526;
            alternate-background-color: #2a2a2d;
            border: none;
            border-radius: 3px;
            outline: 0;
            padding: 4px;
        }
        QListWidget::item {
            min-height: 24px;
            padding: 3px 5px;
            border-radius: 3px;
        }
        QListWidget::item:hover {
            background: #34343a;
        }
        QListWidget::item:selected {
            background: #3f8ec5;
            color: #ffffff;
        }
        QHeaderView::section {
            background: #2d2d30;
            color: #cfcfcf;
            border: none;
            border-bottom: 1px solid #3b3b3f;
            padding: 5px 6px;
        }
        QSplitter::handle {
            background: #1e1e1e;
        }
        QSplitter::handle:hover {
            background: #3f8ec5;
        }
        QScrollBar:vertical, QScrollBar:horizontal {
            background: #252526;
            border: none;
            margin: 0;
        }
        QScrollBar:vertical {
            width: 10px;
        }
        QScrollBar:horizontal {
            height: 10px;
        }
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
            background: #505055;
            border-radius: 5px;
            min-height: 28px;
            min-width: 28px;
        }
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
            background: #626269;
        }
        QScrollBar::add-line, QScrollBar::sub-line {
            width: 0;
            height: 0;
        }
        QScrollBar::add-page, QScrollBar::sub-page {
            background: none;
        }
        QSlider {
            background: transparent;
        }
        QSlider::groove:horizontal {
            height: 4px;
            background: #3a3a3d;
            border-radius: 2px;
        }
        QSlider::sub-page:horizontal {
            background: #3f8ec5;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #d4d4d4;
            border: none;
            width: 12px;
            height: 12px;
            margin: -4px 0;
            border-radius: 6px;
        }
        QSlider::handle:horizontal:hover {
            background: #ffffff;
        }
        QStatusBar {
            background: #181818;
            color: #bfbfbf;
            border-top: 1px solid #333333;
        }
        """
    )
