from __future__ import annotations

from hakyking.diagnostics import collect_diagnostics, diagnostics_as_text
from hakyking.qt import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class DiagnosticsDialog(QDialog):
    """Small self-check window for environment and runtime dependencies."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Hakyking Diagnostics")
        self.resize(680, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit, 1)

        button_row = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.copy_button = QPushButton("Copy")
        self.close_button = QPushButton("Close")
        button_row.addWidget(self.refresh_button)
        button_row.addWidget(self.copy_button)
        button_row.addStretch(1)
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

        self.refresh_button.clicked.connect(self.refresh)
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.close_button.clicked.connect(self.accept)
        self.refresh()

    def refresh(self) -> None:
        self.text_edit.setPlainText(diagnostics_as_text(collect_diagnostics()))

    def copy_to_clipboard(self) -> None:
        QApplication.clipboard().setText(self.text_edit.toPlainText())
