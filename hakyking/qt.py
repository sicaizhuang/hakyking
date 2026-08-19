"""Qt compatibility layer.

The app prefers PySide6, but falls back to PyQt5 when PySide6 is not
available. Keep all Qt imports behind this module so the rest of the
application is not tied to one binding.
"""

try:
    from PySide6.QtCore import *  # type: ignore # noqa: F401,F403
    from PySide6.QtGui import *  # type: ignore # noqa: F401,F403
    from PySide6.QtWidgets import *  # type: ignore # noqa: F401,F403

    QT_API = "PySide6"
except ImportError:
    from PyQt5.QtCore import *  # type: ignore # noqa: F401,F403
    from PyQt5.QtCore import pyqtSignal as Signal  # type: ignore # noqa: F401
    from PyQt5.QtCore import pyqtSlot as Slot  # type: ignore # noqa: F401
    from PyQt5.QtGui import *  # type: ignore # noqa: F401,F403
    from PyQt5.QtWidgets import *  # type: ignore # noqa: F401,F403

    QT_API = "PyQt5"
