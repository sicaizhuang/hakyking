from __future__ import annotations

import math

import numpy as np


MIN_GAIN_DB = -60.0
MAX_GAIN_DB = 48.0
DEFAULT_SOFT_LIMIT_THRESHOLD = 0.92


def gain_db_to_percent(gain_db: float) -> float:
    return 100.0 * math.pow(10.0, float(gain_db) / 20.0)


def db_to_gain(gain_db: float) -> float:
    return math.pow(10.0, float(gain_db) / 20.0)


def apply_gain(
    audio: np.ndarray,
    gain_db: float,
    *,
    soft_limit: bool = True,
    threshold: float = DEFAULT_SOFT_LIMIT_THRESHOLD,
) -> np.ndarray:
    source = np.nan_to_num(
        np.asarray(audio, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if source.size == 0:
        return source
    if abs(gain_db) <= 1e-4:
        output = source.copy()
    else:
        output = np.asarray(source * db_to_gain(gain_db), dtype=np.float32)
    return soft_limit_audio(output, threshold=threshold) if soft_limit else output


def soft_limit_audio(
    audio: np.ndarray,
    *,
    threshold: float = DEFAULT_SOFT_LIMIT_THRESHOLD,
) -> np.ndarray:
    """Apply a transparent tanh soft clip above threshold.

    Gain itself is still simple dB multiplication. The limiter only catches
    peaks that would otherwise clip and sound like crackle/noise.
    """

    source = np.nan_to_num(
        np.asarray(audio, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if source.size == 0:
        return source
    resolved_threshold = float(min(0.99, max(0.05, threshold)))
    magnitude = np.abs(source)
    over = magnitude > resolved_threshold
    if not np.any(over):
        return source
    headroom = 0.98 - resolved_threshold
    limited = source.copy()
    excess = (magnitude[over] - resolved_threshold) / max(1e-6, headroom)
    limited_magnitude = resolved_threshold + headroom * np.tanh(excess)
    limited[over] = np.copysign(limited_magnitude, source[over])
    return np.asarray(limited, dtype=np.float32)


def format_gain(gain_db: float, compact: bool = False) -> str:
    percent = gain_db_to_percent(gain_db)
    percent_text = f"{percent:.1f}" if percent < 10.0 else f"{percent:.0f}"
    separator = "/" if compact else " / "
    return f"{float(gain_db):+.1f} dB{separator}{percent_text}%"


def measure_audio_dbfs(audio: np.ndarray | None) -> tuple[float, float] | None:
    if audio is None:
        return None
    source = np.asarray(audio, dtype=np.float64)
    if source.size == 0:
        return None
    source = source[np.isfinite(source)]
    if source.size == 0:
        return None
    rms = float(np.sqrt(np.mean(np.square(source))))
    peak = float(np.max(np.abs(source)))
    return amplitude_to_dbfs(rms), amplitude_to_dbfs(peak)


def amplitude_to_dbfs(amplitude: float) -> float:
    return max(-120.0, 20.0 * math.log10(max(1e-12, float(amplitude))))


def format_dbfs(value: float) -> str:
    return f"{float(value):+.1f} dBFS"
