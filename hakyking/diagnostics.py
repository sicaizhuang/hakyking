from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from hakyking.runtime_preload import preload_neural_pitch_runtime


preload_neural_pitch_runtime()

from hakyking.app_settings import SETTINGS_FILE  # noqa: E402
from hakyking.qt import QT_API  # noqa: E402
from hakyking.app_logging import LOG_FILE  # noqa: E402
from hakyking.runtime import which_executable  # noqa: E402
from hakyking.subprocess_utils import hidden_subprocess_kwargs  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTOSAVE_FILE = PROJECT_ROOT / "autosave" / "hakyking_autosave.haky"
QUALITY_GATE_SCRIPT = PROJECT_ROOT / "dev_tools" / "run_quality_gate.ps1"


@dataclass(frozen=True)
class DiagnosticCheck:
    name: str
    status: str
    detail: str


def collect_diagnostics() -> list[DiagnosticCheck]:
    checks: list[DiagnosticCheck] = [
        DiagnosticCheck(
            "Python",
            "PASS",
            f"{sys.version.split()[0]} | {platform.platform()}",
        ),
        DiagnosticCheck("Qt Binding", "PASS", QT_API),
        DiagnosticCheck("Project Root", "PASS", str(PROJECT_ROOT)),
        DiagnosticCheck("Log File", "PASS" if LOG_FILE.exists() else "WARN", str(LOG_FILE)),
        DiagnosticCheck(
            "Autosave File",
            "PASS" if AUTOSAVE_FILE.exists() else "WARN",
            str(AUTOSAVE_FILE),
        ),
        DiagnosticCheck(
            "Quality Gate",
            "PASS" if QUALITY_GATE_SCRIPT.exists() else "WARN",
            str(QUALITY_GATE_SCRIPT),
        ),
        DiagnosticCheck(
            "Settings File",
            "PASS" if SETTINGS_FILE.exists() else "WARN",
            str(SETTINGS_FILE),
        ),
    ]
    checks.extend(_package_checks())
    checks.extend(_tool_checks())
    checks.append(_sounddevice_check())
    return checks


def diagnostics_as_text(checks: list[DiagnosticCheck] | None = None) -> str:
    resolved_checks = collect_diagnostics() if checks is None else checks
    lines = ["Hakyking Diagnostics", ""]
    for check in resolved_checks:
        lines.append(f"[{check.status}] {check.name}: {check.detail}")
    return "\n".join(lines)


def _package_checks() -> list[DiagnosticCheck]:
    package_names = [
        "PyQt5",
        "numpy",
        "librosa",
        "soundfile",
        "pyrubberband",
        "sounddevice",
        "pyworld",
        "praat-parselmouth",
        "rmvpe-onnx",
        "onnxruntime",
        "audioflux",
    ]
    checks: list[DiagnosticCheck] = []
    optional_native = {"rmvpe-onnx", "onnxruntime"}
    for package_name in package_names:
        import_name = {
            "praat-parselmouth": "parselmouth",
            "rmvpe-onnx": "rmvpe_onnx",
        }.get(package_name, package_name)
        if package_name in optional_native and os.environ.get(
            "HAKYKING_ENABLE_NEURAL_PITCH", "0"
        ) != "1":
            installed = importlib.util.find_spec(import_name) is not None
            try:
                version = importlib.metadata.version(package_name) if installed else "not installed"
            except importlib.metadata.PackageNotFoundError:
                version = "not installed"
            checks.append(
                DiagnosticCheck(
                    package_name,
                    "WARN",
                    f"{version}; optional neural backend disabled",
                )
            )
            continue
        try:
            importlib.import_module(import_name)
            version = importlib.metadata.version(package_name)
        except Exception as exc:  # noqa: BLE001 - diagnostic should report all failures
            checks.append(DiagnosticCheck(package_name, "FAIL", str(exc)))
        else:
            checks.append(DiagnosticCheck(package_name, "PASS", version))
    return checks


def _tool_checks() -> list[DiagnosticCheck]:
    checks: list[DiagnosticCheck] = []
    for tool_name in ("ffmpeg", "ffprobe", "rubberband"):
        tool_path = which_executable(tool_name)
        if not tool_path:
            checks.append(DiagnosticCheck(tool_name, "FAIL", "not found on PATH"))
            continue
        version = _tool_version(tool_path)
        detail = tool_path if not version else f"{tool_path} | {version}"
        checks.append(DiagnosticCheck(tool_name, "PASS", detail))
    return checks


def _tool_version(tool_path: str) -> str:
    try:
        result = subprocess.run(
            [tool_path, "-version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=4,
            **hidden_subprocess_kwargs(),
        )
    except Exception:
        return ""
    first_line = (result.stdout or result.stderr).splitlines()
    return first_line[0].strip() if first_line else ""


def _sounddevice_check() -> DiagnosticCheck:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        output_count = sum(1 for device in devices if int(device.get("max_output_channels", 0)) > 0)
        input_count = sum(1 for device in devices if int(device.get("max_input_channels", 0)) > 0)
    except Exception as exc:  # noqa: BLE001 - audio host errors are environment-specific
        return DiagnosticCheck("Audio Devices", "WARN", str(exc))
    return DiagnosticCheck(
        "Audio Devices",
        "PASS" if output_count > 0 else "WARN",
        f"outputs={output_count}, inputs={input_count}",
    )
