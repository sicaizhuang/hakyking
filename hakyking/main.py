# Hakyking main entry point.
#
# Core third-party dependencies:
#   PyQt5     - Qt desktop UI runtime for this stage
#   PySide6   - optional Qt binding when available
#   numpy     - numerical buffers and future DSP utilities
#   librosa   - future pitch/onset analysis
#   soundfile - future audio file I/O
#   pyrubberband - Rubber Band Python bridge for pitch/time processing
#   sounddevice - non-blocking audio preview playback
#   ffmpeg    - external executable for extracting MP4 audio tracks
#   ffprobe   - external executable for detecting MP4 audio streams
#   rubberband - external executable used by pyrubberband
#
# Install:
#   python -m pip install PyQt5 numpy librosa soundfile pyrubberband sounddevice
#   or
#   python -m pip install -r requirements.txt
#   MP4 support also requires FFmpeg/FFprobe on PATH.
#   DSP pitch/time processing requires Rubber Band on PATH.
#
# Non-blocking rule:
#   All future audio file I/O, slicing, pitch detection, waveform rendering,
#   time-stretching, pitch-shifting, and export work must run in QThread
#   workers. Never block the main UI thread.

from __future__ import annotations

import sys

from hakyking.runtime_preload import preload_neural_pitch_runtime


preload_neural_pitch_runtime()

from hakyking.app_logging import configure_logging, install_excepthook  # noqa: E402
from hakyking.controllers.main_controller import MainController  # noqa: E402
from hakyking.models.project import ProjectModel  # noqa: E402
from hakyking.qt import QApplication, QTimer  # noqa: E402
from hakyking.runtime import configure_external_tool_paths  # noqa: E402
from hakyking.styles.dark_theme import apply_dark_theme  # noqa: E402
from hakyking.views.main_window import MainWindow  # noqa: E402


def main() -> int:
    configure_logging()
    install_excepthook()
    configure_external_tool_paths()
    app = QApplication(sys.argv)
    app.setApplicationName("Hakyking")
    app.setOrganizationName("Hakyking")
    apply_dark_theme(app)

    project = ProjectModel()
    window = MainWindow()
    controller = MainController(project=project, main_window=window)
    controller.initialize()

    window.show()
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".haky"):
        project_path = sys.argv[1]
        QTimer.singleShot(0, lambda: controller.load_project_path(project_path))
    return app.exec() if hasattr(app, "exec") else app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
