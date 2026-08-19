from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from hakyking.audio.gain import soft_limit_audio
from hakyking.audio.playback import prepare_playback_audio


@dataclass(frozen=True)
class ExportClip:
    start_time: float
    audio: np.ndarray
    sample_rate: int
    track_index: int


@dataclass(frozen=True)
class ExportResult:
    output_path: str
    duration: float
    sample_rate: int
    peak_before_limit: float
    normalized: bool


def export_mixdown_to_wav(
    clips: list[ExportClip],
    output_path: str,
    sample_rate: int = 44100,
    fade_ms: float = 5.0,
) -> ExportResult:
    prepared_clips = _prepare_clips(clips, sample_rate, fade_ms=fade_ms)
    if not prepared_clips:
        raise ValueError("No audio clips are ready for export.")

    total_frames = 0
    for clip in prepared_clips:
        start_frame = int(round(clip.start_time * sample_rate))
        total_frames = max(total_frames, start_frame + clip.audio.shape[0])
    if total_frames <= 0:
        raise ValueError("Export duration is empty.")

    master_buffer = np.zeros(total_frames, dtype=np.float32)
    for clip in prepared_clips:
        start_frame = int(round(clip.start_time * sample_rate))
        end_frame = start_frame + clip.audio.shape[0]
        master_buffer[start_frame:end_frame] += clip.audio

    peak_before_limit = float(np.max(np.abs(master_buffer))) if master_buffer.size else 0.0
    master_buffer, normalized = _limit_master_buffer(master_buffer)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), master_buffer, sample_rate, subtype="PCM_16")
    return ExportResult(
        output_path=str(output),
        duration=master_buffer.shape[0] / sample_rate,
        sample_rate=sample_rate,
        peak_before_limit=peak_before_limit,
        normalized=normalized,
    )


def _prepare_clips(
    clips: list[ExportClip],
    target_rate: int,
    fade_ms: float = 5.0,
) -> list[ExportClip]:
    prepared: list[ExportClip] = []
    for clip in clips:
        audio = _to_mono_float32(clip.audio)
        if audio.size == 0:
            continue
        if clip.sample_rate != target_rate:
            audio = _resample_audio(audio, clip.sample_rate, target_rate)
        audio = prepare_playback_audio(
            audio,
            target_rate,
            fade_ms=fade_ms,
            limit_output=False,
        )
        prepared.append(
            ExportClip(
                start_time=max(0.0, clip.start_time),
                audio=audio,
                sample_rate=target_rate,
                track_index=clip.track_index,
            )
        )
    return prepared


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float32)
    if source.ndim == 2:
        source = np.mean(source, axis=1, dtype=np.float32)
    source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(source, dtype=np.float32)


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    import librosa

    return np.asarray(
        librosa.resample(
            audio,
            orig_sr=source_rate,
            target_sr=target_rate,
        ),
        dtype=np.float32,
    )


def _limit_master_buffer(audio: np.ndarray) -> tuple[np.ndarray, bool]:
    output = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    limited = soft_limit_audio(output)
    return limited, peak > 0.92
