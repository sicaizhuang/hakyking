from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hakyking.runtime import user_data_root


CONFIG_DIR = user_data_root() / "config"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
DEFAULT_PITCH_ENGINE = "adaptive"
PITCH_ENGINE_LABELS = {
    "adaptive": "Adaptive (Praat / RubberBand)",
    "rubberband": "RubberBand",
    "parselmouth_psola": "Praat PSOLA",
    "pyworld_hpss": "PyWorld HPSS",
    "librosa": "Librosa",
}


@dataclass(frozen=True)
class AudioPlaybackSettings:
    output_device_index: int | None = None
    blocksize: int = 1024
    fade_ms: float = 5.0
    pitch_engine: str = DEFAULT_PITCH_ENGINE


def load_audio_settings(path: str | Path = SETTINGS_FILE) -> AudioPlaybackSettings:
    settings_path = Path(path)
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AudioPlaybackSettings()
    if not isinstance(payload, dict):
        return AudioPlaybackSettings()
    audio_payload = payload.get("audio", payload)
    if not isinstance(audio_payload, dict):
        return AudioPlaybackSettings()
    return AudioPlaybackSettings(
        output_device_index=_optional_nonnegative_int(audio_payload.get("output_device_index")),
        blocksize=_int_value(audio_payload.get("blocksize"), 1024, minimum=128, maximum=8192),
        fade_ms=_float_value(audio_payload.get("fade_ms"), 5.0, minimum=0.0, maximum=50.0),
        pitch_engine=normalize_pitch_engine(audio_payload.get("pitch_engine")),
    )


def save_audio_settings(
    settings: AudioPlaybackSettings,
    path: str | Path = SETTINGS_FILE,
) -> Path:
    settings_path = Path(path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"audio": asdict(settings)}
    settings_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return settings_path


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def normalize_pitch_engine(value: Any) -> str:
    if not isinstance(value, str):
        return DEFAULT_PITCH_ENGINE
    engine = value.strip().lower()
    aliases = {
        "rubber": "rubberband",
        "rubber_band": "rubberband",
        "auto": "adaptive",
        "automatic": "adaptive",
        "hybrid": "adaptive",
        "praat": "parselmouth_psola",
        "psola": "parselmouth_psola",
        "parselmouth": "parselmouth_psola",
        "world": "pyworld_hpss",
        "pyworld": "pyworld_hpss",
        "librosa_phase_vocoder": "librosa",
    }
    engine = aliases.get(engine, engine)
    return engine if engine in PITCH_ENGINE_LABELS else DEFAULT_PITCH_ENGINE


def _int_value(
    value: Any,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        number = default
    return max(minimum, min(maximum, number))


def _float_value(
    value: Any,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        number = default
    if not math.isfinite(number):
        number = default
    return max(minimum, min(maximum, number))
