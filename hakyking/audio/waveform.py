from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hakyking.audio.reader import AudioReader
from hakyking.models.audio_slice import AudioSlice


@dataclass(frozen=True)
class WaveformResult:
    cache_key: str
    audio: np.ndarray
    sample_rate: int
    envelope: np.ndarray
    pitch_contour: np.ndarray


def load_slice_audio(audio_slice: AudioSlice) -> tuple[np.ndarray, int]:
    """Load a slice from its source file without modifying the source file."""

    source_audio, sample_rate = AudioReader.load_mono(audio_slice.source_path)
    start_sample = max(0, int(round(audio_slice.start_time * sample_rate)))
    end_sample = max(start_sample, int(round(audio_slice.end_time * sample_rate)))
    return np.asarray(source_audio[start_sample:end_sample], dtype=np.float32).copy(), sample_rate


def build_waveform_result(
    cache_key: str,
    audio_slice: AudioSlice,
    max_points: int = 256,
    prefer_cached_pitch: bool = True,
) -> WaveformResult:
    audio, sample_rate = load_slice_audio(audio_slice)
    envelope = compute_waveform_envelope(audio, max_points=max_points)
    pitch_contour = None
    if prefer_cached_pitch:
        try:
            from hakyking.audio.vocal_analysis import cached_pitch_contour_for_slice

            pitch_contour = cached_pitch_contour_for_slice(
                audio_slice,
                max_points=max_points,
            )
        except Exception:
            pitch_contour = None
    if pitch_contour is None:
        pitch_contour = compute_pitch_contour(audio, sample_rate, max_points=max_points)
    return WaveformResult(
        cache_key=cache_key,
        audio=audio,
        sample_rate=sample_rate,
        envelope=envelope,
        pitch_contour=pitch_contour,
    )


def compute_waveform_envelope(audio: np.ndarray, max_points: int = 256) -> np.ndarray:
    """
    Convert raw audio into a downsampled min/max envelope.

    The returned array has shape (N, 2), where column 0 is min and column 1 is
    max, normalized to [-1, 1]. It is compact enough to paint directly.
    """

    source = np.asarray(audio, dtype=np.float32)
    if source.ndim == 2:
        source = np.mean(source, axis=1, dtype=np.float32)
    if source.size == 0 or max_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(source)))
    if peak > 0:
        source = source / peak

    point_count = min(max_points, source.size)
    block_size = int(np.ceil(source.size / point_count))
    padded_size = block_size * point_count
    padded = np.zeros(padded_size, dtype=np.float32)
    padded[: source.size] = source
    blocks = padded.reshape(point_count, block_size)
    mins = np.min(blocks, axis=1)
    maxes = np.max(blocks, axis=1)
    return np.stack([mins, maxes], axis=1).astype(np.float32)


def compute_pitch_contour(
    audio: np.ndarray,
    sample_rate: int,
    max_points: int = 256,
) -> np.ndarray:
    """Return a compact MIDI F0 contour with NaN values for unvoiced frames."""

    source = np.asarray(audio, dtype=np.float32)
    if source.ndim == 2:
        source = np.mean(source, axis=1, dtype=np.float32)
    if source.size < 256 or sample_rate <= 0 or max_points <= 0:
        return np.zeros(0, dtype=np.float32)

    pyworld_contour = _compute_pitch_contour_pyworld(
        source,
        sample_rate,
        max_points=max_points,
    )
    if pyworld_contour.size:
        return pyworld_contour

    return _compute_pitch_contour_librosa(
        source,
        sample_rate,
        max_points=max_points,
    )


def _compute_pitch_contour_pyworld(
    source: np.ndarray,
    sample_rate: int,
    max_points: int,
) -> np.ndarray:
    try:
        import pyworld as pw

        world_source = _as_world_float64(source)
        if world_source.size < int(sample_rate * 0.04):
            return np.zeros(0, dtype=np.float32)
        frame_period = 5.0
        f0, time_axis = pw.dio(
            world_source,
            sample_rate,
            f0_floor=65.0,
            f0_ceil=2093.0,
            frame_period=frame_period,
        )
        f0 = pw.stonemask(world_source, f0, time_axis, sample_rate)
        midi = _hz_to_midi_array(f0)
        return _compact_midi_contour(midi, max_points=max_points)
    except Exception:
        return np.zeros(0, dtype=np.float32)


def _compute_pitch_contour_librosa(
    source: np.ndarray,
    sample_rate: int,
    max_points: int,
) -> np.ndarray:
    try:
        import librosa

        frame_length = min(2048, int(2 ** np.floor(np.log2(max(256, source.size)))))
        frame_length = max(256, frame_length)
        hop_length = max(128, frame_length // 4)
        f0, _, _ = librosa.pyin(
            source,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        midi = librosa.hz_to_midi(f0).astype(np.float32)
        return _compact_midi_contour(midi, max_points=max_points)
    except Exception:
        return np.zeros(0, dtype=np.float32)


def _hz_to_midi_array(f0: np.ndarray) -> np.ndarray:
    source = np.asarray(f0, dtype=np.float64)
    midi = np.full(source.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(source) & (source > 0.0)
    midi[valid] = (69.0 + 12.0 * np.log2(source[valid] / 440.0)).astype(np.float32)
    return midi


def _compact_midi_contour(midi: np.ndarray, max_points: int) -> np.ndarray:
    source = np.asarray(midi, dtype=np.float32)
    if source.size == 0:
        return np.zeros(0, dtype=np.float32)
    if source.size <= max_points:
        return source

    positions = np.linspace(0, source.size - 1, max_points)
    compact = np.empty(max_points, dtype=np.float32)
    for index, position in enumerate(positions):
        left = int(max(0, np.floor(position) - 1))
        right = int(min(source.size, np.ceil(position) + 2))
        values = source[left:right]
        values = values[np.isfinite(values)]
        compact[index] = np.nan if values.size == 0 else float(np.median(values))
    return compact


def _as_world_float64(audio: np.ndarray) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float64)
    source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(source))) if source.size else 0.0
    if peak > 1.0:
        source = source / peak
    return np.ascontiguousarray(source, dtype=np.float64)
