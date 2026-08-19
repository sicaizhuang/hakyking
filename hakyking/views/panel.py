from __future__ import annotations

from hakyking.qt import QFrame, QLabel, Qt, QVBoxLayout, QWidget


class Panel(QFrame):
    def __init__(self, title: str, content: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("PanelTitle")
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.title_label.setFixedHeight(32)
        self.title_label.setVisible(True)
        layout.addWidget(self.title_label)
        layout.addWidget(content, 1)

    def set_title(self, title: str) -> None:
        self.title_label.setText(title)
        self.setToolTip(title)
