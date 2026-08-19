from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def app_root() -> Path:
    """Return the project root in source runs, or PyInstaller's data root in exe runs."""
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root).resolve()
    return Path(__file__).resolve().parents[1]


def user_data_root() -> Path:
    """Return the per-user writable directory for settings, caches, and autosave."""
    override = os.environ.get("HAKYKING_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif os.environ.get("XDG_DATA_HOME"):
        base = Path(os.environ["XDG_DATA_HOME"])
    else:
        base = Path.home() / ".local" / "share"
    return base / "Hakyking"


def prepend_to_path(directory: Path) -> None:
    resolved = str(directory.resolve())
    current_path = os.environ.get("PATH", "")
    if resolved.lower() not in current_path.lower():
        os.environ["PATH"] = resolved + os.pathsep + current_path


def find_bundled_executable(executable_name: str) -> Path | None:
    tools_root = app_root() / "tools"
    if not tools_root.exists():
        return None
    for candidate in tools_root.rglob(executable_name):
        if candidate.is_file():
            return candidate
    return None


def which_executable(tool_name: str) -> str | None:
    configure_external_tool_paths()
    return shutil.which(tool_name)


def configure_external_tool_paths() -> None:
    for executable_name in (
        "rubberband.exe",
        "rubberband",
        "ffmpeg.exe",
        "ffmpeg",
        "ffprobe.exe",
        "ffprobe",
    ):
        candidate = find_bundled_executable(executable_name)
        if candidate is not None:
            prepend_to_path(candidate.parent)
