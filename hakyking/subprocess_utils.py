from __future__ import annotations

import subprocess
import sys
from typing import Any


def hidden_subprocess_kwargs() -> dict[str, Any]:
    """Return subprocess options that prevent console flashes on Windows."""

    if sys.platform != "win32":
        return {}

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000),
        "startupinfo": startupinfo,
    }
