from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hakyking.app_settings import DEFAULT_PITCH_ENGINE
from hakyking.audio.gain import apply_gain
from hakyking.audio.reader import AudioReader
from hakyking.models.audio_slice import AudioSlice
from hakyking.models.audio_edit import SliceRenderRequest
from hakyking.runtime import configure_external_tool_paths, find_bundled_executable
from hakyking.subprocess_utils import hidden_subprocess_kwargs


MIN_AUDIO_RATE = 0.05
MAX_AUDIO_RATE = 8.0
LARGE_PITCH_SHIFT_SEMITONES = 5.0
RUBBERBAND_NORMALIZED_PEAK = 0.92
MIN_EXTERNAL_DSP_SAMPLES = 64
SILENCE_PEAK_EPSILON = 1e-7
PITCH_LOUDNESS_MATCH_LIMIT_DB = 6.0


@dataclass(frozen=True)
class RenderParameters:
    target_midi_note: int | None
    target_duration: float
    n_steps: float
    rate: float
    gain_db: float = 0.0
    pitch_flatten_amount: float = 0.0
    formant_shift: float = 0.0
    protect_transients: bool = True
    pitch_control_points: tuple[tuple[float, float], ...] = ()
    pitch_vibrato_regions: tuple[tuple[float, float, float, float, float, str], ...] = ()


@dataclass(frozen=True)
class RenderResult:
    cache_key: str
    audio: np.ndarray
    sample_rate: int
    parameters: RenderParameters


def frequency_to_midi(f0_hz: float | None) -> float | None:
    if f0_hz is None or f0_hz <= 0 or not math.isfinite(f0_hz):
        return None
    return 69.0 + 12.0 * math.log2(f0_hz / 440.0)


def calculate_n_steps(original_f0_hz: float | None, target_midi_note: int | None) -> float:
    original_midi = frequency_to_midi(original_f0_hz)
    if original_midi is None or target_midi_note is None:
        return 0.0
    return float(target_midi_note) - original_midi


def calculate_time_rate(original_duration: float, target_duration: float) -> float:
    if original_duration <= 0 or target_duration <= 0:
        return 1.0
    rate = original_duration / target_duration
    return min(MAX_AUDIO_RATE, max(MIN_AUDIO_RATE, float(rate)))


def build_render_parameters(
    audio_slice: AudioSlice,
    target_midi_note: int | None,
    target_duration: float | None,
    gain_db: float = 0.0,
    pitch_flatten_amount: float = 0.0,
    formant_shift: float = 0.0,
    pitch_center_hz: float | None = None,
    protect_transients: bool = True,
    pitch_control_points: object = (),
    pitch_vibrato_regions: object = (),
) -> RenderParameters:
    duration = audio_slice.duration
    resolved_duration = duration if target_duration is None else max(0.001, target_duration)
    reference_f0 = pitch_center_hz if pitch_center_hz is not None else audio_slice.f0_hz
    return RenderParameters(
        target_midi_note=target_midi_note,
        target_duration=resolved_duration,
        n_steps=calculate_n_steps(reference_f0, target_midi_note),
        rate=calculate_time_rate(duration, resolved_duration),
        gain_db=float(gain_db),
        pitch_flatten_amount=min(1.0, max(0.0, float(pitch_flatten_amount))),
        formant_shift=min(12.0, max(-12.0, float(formant_shift))),
        protect_transients=bool(protect_transients),
        pitch_control_points=_normalize_pitch_control_points(pitch_control_points),
        pitch_vibrato_regions=_normalize_pitch_vibrato_regions(pitch_vibrato_regions),
    )


def build_render_parameters_from_request(
    request: SliceRenderRequest,
    pitch_center_hz: float | None = None,
) -> RenderParameters:
    """Build DSP parameters from the canonical non-destructive edit snapshot."""

    return build_render_parameters(
        audio_slice=request.audio_slice,
        target_midi_note=request.target_midi_note,
        target_duration=request.target_duration,
        gain_db=request.gain_db,
        pitch_flatten_amount=request.pitch_flatten_amount,
        formant_shift=request.formant_shift,
        pitch_center_hz=pitch_center_hz,
        protect_transients=request.protect_transients,
        pitch_control_points=request.pitch_control_points,
        pitch_vibrato_regions=request.pitch_vibrato_regions,
    )


def render_slice_from_file(
    audio_slice: AudioSlice,
    target_midi_note: int | None,
    target_duration: float | None,
    cache_key: str,
    gain_db: float = 0.0,
    pitch_flatten_amount: float = 0.0,
    formant_shift: float = 0.0,
    protect_transients: bool = True,
    pitch_control_points: object = (),
    pitch_vibrato_regions: object = (),
) -> RenderResult:
    source_audio, sample_rate = AudioReader.load_mono(audio_slice.source_path)
    start_sample = max(0, int(round(audio_slice.start_time * sample_rate)))
    end_sample = max(start_sample, int(round(audio_slice.end_time * sample_rate)))
    source_blob = source_audio[start_sample:end_sample]
    pitch_center_hz = _estimate_pitch_center_hz(source_blob, sample_rate) or audio_slice.f0_hz
    parameters = build_render_parameters(
        audio_slice,
        target_midi_note,
        target_duration,
        gain_db=gain_db,
        pitch_flatten_amount=pitch_flatten_amount,
        formant_shift=formant_shift,
        pitch_center_hz=pitch_center_hz,
        protect_transients=protect_transients,
        pitch_control_points=pitch_control_points,
        pitch_vibrato_regions=pitch_vibrato_regions,
    )
    rendered_audio = process_blob(
        source_blob,
        sample_rate,
        n_steps=parameters.n_steps,
        rate=parameters.rate,
        gain_db=parameters.gain_db,
        pitch_flatten_amount=parameters.pitch_flatten_amount,
        formant_shift=parameters.formant_shift,
        protect_transients=parameters.protect_transients,
        pitch_control_points=parameters.pitch_control_points,
        pitch_vibrato_regions=parameters.pitch_vibrato_regions,
    )
    return RenderResult(
        cache_key=cache_key,
        audio=rendered_audio,
        sample_rate=sample_rate,
        parameters=parameters,
    )


def process_blob(
    audio: np.ndarray,
    sr: int,
    n_steps: float = 0.0,
    rate: float = 1.0,
    gain_db: float = 0.0,
    pitch_flatten_amount: float = 0.0,
    formant_shift: float = 0.0,
    protect_transients: bool = True,
    pitch_control_points: object = (),
    pitch_vibrato_regions: object = (),
) -> np.ndarray:
    """
    Non-destructively pitch/time process an audio blob.

    Ordinary pitch edits use the selected pitch engine. The default engine is
    RubberBand with formant preservation and peak normalization on large shifts.
    Alternative engines remain available through settings or the
    HAKYKING_PITCH_ENGINE environment variable for A/B testing.
    """

    source = _as_float32_copy(audio)
    if source.size == 0:
        return source
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")

    if source.ndim == 1:
        return _process_mono_blob(
            source,
            sr,
            n_steps=n_steps,
            rate=rate,
            gain_db=gain_db,
            pitch_flatten_amount=pitch_flatten_amount,
            formant_shift=formant_shift,
            protect_transients=protect_transients,
            pitch_control_points=pitch_control_points,
            pitch_vibrato_regions=pitch_vibrato_regions,
        )
    if source.ndim == 2:
        return _process_multichannel_blob(
            source,
            sr,
            n_steps=n_steps,
            rate=rate,
            gain_db=gain_db,
            pitch_flatten_amount=pitch_flatten_amount,
            formant_shift=formant_shift,
            protect_transients=protect_transients,
            pitch_control_points=pitch_control_points,
            pitch_vibrato_regions=pitch_vibrato_regions,
        )
    raise ValueError(f"Unsupported audio dimensions: {source.ndim}")


def _normalize_pitch_control_points(value: object) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    if not isinstance(value, (list, tuple)):
        return ()
    for entry in value:
        try:
            if isinstance(entry, dict):
                raw_x = entry.get("x", entry.get("ratio", 0.0))
                raw_offset = entry.get("offset", entry.get("semitones", 0.0))
            else:
                raw_x = entry[0]  # type: ignore[index]
                raw_offset = entry[1]  # type: ignore[index]
            x_value = max(0.0, min(1.0, float(raw_x)))
            offset = max(-24.0, min(24.0, float(raw_offset)))
        except (TypeError, ValueError, IndexError, KeyError, OverflowError):
            continue
        points.append((x_value, offset))
    points.sort(key=lambda point: point[0])
    collapsed: list[tuple[float, float]] = []
    for x_value, offset in points:
        if collapsed and abs(x_value - collapsed[-1][0]) < 0.001:
            collapsed[-1] = (x_value, offset)
        else:
            collapsed.append((x_value, offset))
    return tuple(collapsed)


def _normalize_pitch_vibrato_regions(
    value: object,
) -> tuple[tuple[float, float, float, float, float, str], ...]:
    regions: list[tuple[float, float, float, float, float, str]] = []
    if not isinstance(value, (list, tuple)):
        return ()
    for entry in value:
        try:
            if isinstance(entry, dict):
                start = float(entry.get("start", 0.0))
                end = float(entry.get("end", 1.0))
                cycles = float(entry.get("cycles", 0.0))
                depth = float(entry.get("depth", 0.0))
                phase = float(entry.get("phase", 0.0)) % 1.0
                waveform = str(entry.get("waveform", "sine"))
            else:
                start = float(entry[0])  # type: ignore[index]
                end = float(entry[1])  # type: ignore[index]
                cycles = float(entry[2])  # type: ignore[index]
                depth = float(entry[3])  # type: ignore[index]
                if len(entry) > 5:  # type: ignore[arg-type]
                    phase = float(entry[4]) % 1.0  # type: ignore[index]
                    waveform = str(entry[5])  # type: ignore[index]
                else:
                    phase = 0.0
                    waveform = str(entry[4]) if len(entry) > 4 else "sine"  # type: ignore[arg-type]
        except (TypeError, ValueError, IndexError, KeyError, OverflowError):
            continue
        if not all(math.isfinite(number) for number in (start, end, cycles, depth)):
            continue
        start = max(0.0, min(1.0, start))
        end = max(0.0, min(1.0, end))
        if end < start:
            start, end = end, start
        if end - start <= 1e-5 or cycles <= 0.0 or depth <= 0.0:
            continue
        if waveform not in {"sine", "triangle", "square"}:
            waveform = "sine"
        regions.append((start, end, cycles, min(24.0, depth), phase, waveform))
    regions.sort(key=lambda region: (region[0], region[1]))
    return tuple(regions)


def _pitch_control_offset_at(
    points: tuple[tuple[float, float], ...],
    ratio: float,
) -> float:
    if not points:
        return 0.0
    ratio = max(0.0, min(1.0, float(ratio)))
    if ratio <= points[0][0]:
        return float(points[0][1])
    if ratio >= points[-1][0]:
        return float(points[-1][1])
    for (left_x, left_offset), (right_x, right_offset) in zip(points, points[1:]):
        if left_x <= ratio <= right_x:
            if right_x <= left_x:
                return float(left_offset)
            t = (ratio - left_x) / (right_x - left_x)
            t = t * t * (3.0 - 2.0 * t)
            return float(left_offset + (right_offset - left_offset) * t)
    return 0.0


def _pitch_vibrato_wave_value(waveform: str, phase: float) -> float:
    phase %= 1.0
    if waveform == "triangle":
        return (2.0 / math.pi) * math.asin(math.sin(math.tau * phase))
    if waveform == "square":
        return 1.0 if phase < 0.5 else -1.0
    return math.sin(math.tau * phase)


def _pitch_vibrato_offset_at(
    regions: tuple[tuple[float, float, float, float, float, str], ...],
    ratio: float,
) -> float:
    if not regions:
        return 0.0
    ratio = max(0.0, min(1.0, float(ratio)))
    total = 0.0
    for start, end, cycles, depth, phase, waveform in regions:
        if end <= start or not (start - 1e-6 <= ratio <= end + 1e-6):
            continue
        t = (ratio - start) / max(1e-6, end - start)
        total += _pitch_vibrato_wave_value(waveform, cycles * t + phase) * depth
    return max(-24.0, min(24.0, total))


def _apply_pitch_control_curve(
    audio: np.ndarray,
    sr: int,
    pitch_control_points: object,
    pitch_vibrato_regions: object = (),
) -> np.ndarray:
    points = _normalize_pitch_control_points(pitch_control_points)
    regions = _normalize_pitch_vibrato_regions(pitch_vibrato_regions)
    has_points = bool(points) and max(abs(offset) for _x, offset in points) >= 0.03
    has_regions = bool(regions)
    if not has_points and not has_regions:
        return np.asarray(audio, dtype=np.float32)

    source = np.asarray(audio, dtype=np.float32)
    if source.size <= 1 or sr <= 0:
        return source

    try:
        return _rubberband_dynamic_pitch_process(source, sr, points, regions)
    except Exception:
        return _apply_segmented_pitch_control_curve(source, sr, points, regions)


def _build_dynamic_pitch_map(
    sample_count: int,
    sr: int,
    points: tuple[tuple[float, float], ...],
    regions: tuple[tuple[float, float, float, float, float, str], ...],
) -> list[tuple[int, float]]:
    if sample_count <= 1 or sr <= 0:
        return [(0, 0.0)]
    duration = sample_count / float(sr)
    requested_cycles = sum(max(0.0, region[2]) for region in regions)
    by_time = int(math.ceil(duration / 0.008)) + 1
    by_vibrato = int(math.ceil(requested_cycles * 32.0)) + 1
    count = max(2, min(8192, max(by_time, by_vibrato)))
    frames = np.linspace(0, sample_count - 1, count, dtype=np.int64)
    entries: list[tuple[int, float]] = []
    previous_frame = -1
    for frame in frames:
        frame_number = int(frame)
        if frame_number == previous_frame:
            continue
        ratio = frame_number / max(1, sample_count - 1)
        offset = _pitch_control_offset_at(points, ratio) + _pitch_vibrato_offset_at(
            regions,
            ratio,
        )
        entries.append((frame_number, max(-24.0, min(24.0, float(offset)))))
        previous_frame = frame_number
    return entries


def _rubberband_dynamic_pitch_process(
    audio: np.ndarray,
    sr: int,
    points: tuple[tuple[float, float], ...],
    regions: tuple[tuple[float, float, float, float, float, str], ...],
) -> np.ndarray:
    entries = _build_dynamic_pitch_map(len(audio), sr, points, regions)
    if not entries:
        return np.asarray(audio, dtype=np.float32)

    map_path = ""
    try:
        map_handle = tempfile.NamedTemporaryFile(
            suffix=".pitchmap",
            mode="w",
            encoding="ascii",
            newline="\n",
            delete=False,
        )
        map_path = map_handle.name
        with map_handle:
            for frame, offset in entries:
                map_handle.write(f"{frame} {offset:.8f}\n")
        processed = _run_rubberband_cli(
            np.asarray(audio, dtype=np.float32),
            sr,
            {
                "--fine": "",
                "--quiet": "",
                "--formant": "",
                "--pitchmap": map_path,
            },
        )
        return _pad_or_crop(processed, len(audio))
    finally:
        if map_path:
            Path(map_path).unlink(missing_ok=True)


def _apply_segmented_pitch_control_curve(
    source: np.ndarray,
    sr: int,
    points: tuple[tuple[float, float], ...],
    regions: tuple[tuple[float, float, float, float, float, str], ...],
) -> np.ndarray:
    """Compatibility fallback for systems where Rubber Band is unavailable."""

    requested_cycles = sum(max(0.0, region[2]) for region in regions)
    min_segment_seconds = 0.012 if requested_cycles > 0 else 0.07
    min_segment_samples = max(256, int(round(sr * min_segment_seconds)))
    max_segments_by_length = max(1, source.size // min_segment_samples)
    segment_count = min(
        max_segments_by_length,
        max(
            3,
            len(points) - 1,
            int(math.ceil(requested_cycles * 16.0)) if requested_cycles > 0 else 0,
            int(math.ceil(source.size / max(1, int(sr * 0.25)))),
        ),
    )
    if segment_count <= 1:
        offset = _pitch_control_offset_at(points, 0.5) + _pitch_vibrato_offset_at(regions, 0.5)
        return _pad_or_crop(
            _direct_pitch_time_process(source, sr, n_steps=offset, rate=1.0),
            source.size,
        )

    boundaries = np.linspace(0, source.size, segment_count + 1, dtype=np.int64)
    processed_segments: list[np.ndarray] = []
    for index in range(segment_count):
        start = int(boundaries[index])
        end = int(boundaries[index + 1])
        if end <= start:
            continue
        segment = source[start:end]
        center_ratio = ((start + end) * 0.5) / max(1, source.size)
        offset = _pitch_control_offset_at(points, center_ratio) + _pitch_vibrato_offset_at(
            regions,
            center_ratio,
        )
        if abs(offset) >= 0.03:
            try:
                segment = _direct_pitch_time_process(
                    segment,
                    sr,
                    n_steps=offset,
                    rate=1.0,
                )
            except Exception:
                segment = source[start:end]
        processed_segments.append(_pad_or_crop(segment, end - start))

    if not processed_segments:
        return source
    joined = _crossfade_join(processed_segments, int(round(sr * 0.004)))
    return _pad_or_crop(joined, source.size)


def _process_mono_blob(
    audio: np.ndarray,
    sr: int,
    n_steps: float,
    rate: float,
    gain_db: float,
    pitch_flatten_amount: float,
    formant_shift: float,
    protect_transients: bool,
    pitch_control_points: object = (),
    pitch_vibrato_regions: object = (),
) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float32)
    safe_rate = min(MAX_AUDIO_RATE, max(MIN_AUDIO_RATE, float(rate)))
    target_length = max(1, int(round(source.size / safe_rate)))
    source_peak = float(np.max(np.abs(source))) if source.size else 0.0
    if source.size < MIN_EXTERNAL_DSP_SAMPLES or source_peak <= SILENCE_PEAK_EPSILON:
        bypassed = _resize_tiny_or_silent_audio(source, target_length)
        return _apply_output_gain(bypassed, gain_db)

    processed = source
    if pitch_flatten_amount > 1e-4:
        processed = _flatten_pitch_variation(
            processed,
            sr,
            amount=pitch_flatten_amount,
        )

    use_transient_stretch = protect_transients and abs(safe_rate - 1.0) > 1e-4
    processed = _direct_pitch_time_process(
        processed,
        sr,
        n_steps=n_steps,
        rate=1.0 if use_transient_stretch else safe_rate,
    )
    if abs(n_steps) > 1e-4:
        processed = _match_pitch_shift_loudness(source, processed)

    if abs(formant_shift) > 1e-4:
        processed = _apply_formant_shift_approx(
            processed,
            sr,
            semitones=formant_shift,
        )

    if use_transient_stretch:
        processed = transient_protected_time_stretch(
            processed,
            sr,
            stretch_factor=1.0 / safe_rate,
        )

    processed = _apply_pitch_control_curve(
        processed,
        sr,
        pitch_control_points=pitch_control_points,
        pitch_vibrato_regions=pitch_vibrato_regions,
    )

    return _apply_output_gain(processed, gain_db)


def process_advanced_blob(
    audio: np.ndarray,
    sr: int,
    n_steps: float = 0.0,
    rate: float = 1.0,
    gain_db: float = 0.0,
    flatten_amount: float = 0.0,
    formant_shift: float = 0.0,
) -> np.ndarray:
    """
    WORLD-vocoder advanced processing path.

    WORLD separates a vocal slice into F0, spectral envelope, and aperiodicity.
    Hakyking uses those axes independently: pitch edits scale F0, flattening
    smooths F0 movement, and formant edits warp only the spectral envelope.
    """

    source = _as_float32_copy(audio)
    if source.size == 0:
        return source
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")
    if source.ndim == 1:
        return _process_advanced_mono_blob(
            source,
            sr,
            n_steps=n_steps,
            rate=rate,
            gain_db=gain_db,
            flatten_amount=flatten_amount,
            formant_shift=formant_shift,
        )
    if source.ndim == 2:
        channel_first = source.shape[0] <= 8 and source.shape[0] < source.shape[1]
        channels = source if channel_first else source.T
        processed_channels = [
            _process_advanced_mono_blob(
                channel,
                sr,
                n_steps=n_steps,
                rate=rate,
                gain_db=gain_db,
                flatten_amount=flatten_amount,
                formant_shift=formant_shift,
            )
            for channel in channels
        ]
        target_len = max(len(channel) for channel in processed_channels)
        aligned = [_pad_or_crop(channel, target_len) for channel in processed_channels]
        stacked = np.stack(aligned, axis=0)
        return stacked if channel_first else stacked.T
    raise ValueError(f"Unsupported audio dimensions: {source.ndim}")


def _process_advanced_mono_blob(
    audio: np.ndarray,
    sr: int,
    n_steps: float,
    rate: float,
    gain_db: float,
    flatten_amount: float,
    formant_shift: float,
) -> np.ndarray:
    import pyworld as pw

    source = _as_world_float64(audio)
    if source.size < int(sr * 0.04):
        return _apply_output_gain(audio, gain_db)

    frame_period = 5.0
    f0, sp, ap = pw.wav2world(source, sr, frame_period=frame_period)
    if f0.ndim != 1 or sp.ndim != 2 or ap.ndim != 2:
        raise ValueError(
            f"Unexpected WORLD shapes: f0={f0.shape}, sp={sp.shape}, ap={ap.shape}"
        )
    if sp.shape != ap.shape:
        raise ValueError(f"WORLD SP/AP shape mismatch: sp={sp.shape}, ap={ap.shape}")
    if sp.shape[0] != f0.shape[0]:
        raise ValueError(f"WORLD frame mismatch: f0={f0.shape}, sp={sp.shape}")

    modified_f0 = _modify_world_f0(
        f0,
        n_steps=n_steps,
        flatten_amount=flatten_amount,
    )
    modified_sp = _shift_world_formants(
        sp,
        semitones=formant_shift,
    )
    synthesized = pw.synthesize(
        modified_f0.astype(np.float64, copy=False),
        modified_sp.astype(np.float64, copy=False),
        ap.astype(np.float64, copy=False),
        sr,
        frame_period=frame_period,
    )
    output = np.asarray(synthesized, dtype=np.float32)

    safe_rate = min(MAX_AUDIO_RATE, max(MIN_AUDIO_RATE, float(rate)))
    if abs(safe_rate - 1.0) > 1e-4 and output.size:
        _ensure_rubberband_available()
        output = _rubberband_time_stretch(output, sr, safe_rate)
    elif output.size:
        output = _pad_or_crop(output, len(audio))

    output = _apply_gain(output, gain_db)
    return _finite_float32(output)


def _process_advanced_with_hpss(
    audio: np.ndarray,
    sr: int,
    n_steps: float,
    rate: float,
    flatten_amount: float,
    formant_shift: float,
) -> np.ndarray | None:
    try:
        import librosa

        if len(audio) < 64:
            harmonic = np.asarray(audio, dtype=np.float32)
            percussive = np.zeros_like(harmonic)
        else:
            n_fft = min(2048, int(2 ** math.floor(math.log2(len(audio)))))
            n_fft = max(64, n_fft)
            harmonic, percussive = librosa.effects.hpss(
                np.asarray(audio, dtype=np.float32),
                n_fft=n_fft,
                hop_length=max(16, n_fft // 4),
            )

        processed_harmonic = process_advanced_blob(
            np.asarray(harmonic, dtype=np.float32),
            sr,
            n_steps=n_steps,
            rate=rate,
            gain_db=0.0,
            flatten_amount=flatten_amount,
            formant_shift=formant_shift,
        )
        aligned_percussive = _align_percussive(percussive, len(processed_harmonic))
        return _safe_float32_mix(processed_harmonic + aligned_percussive)
    except Exception:
        return None


def _direct_pitch_time_process(
    audio: np.ndarray,
    sr: int,
    n_steps: float,
    rate: float,
) -> np.ndarray:
    processed = np.asarray(audio, dtype=np.float32)
    engine = _pitch_engine_name()
    if engine in {
        "adaptive",
        "auto",
        "hybrid",
        "parselmouth",
        "praat",
        "psola",
        "parselmouth_psola",
    }:
        if _is_large_pitch_shift(n_steps):
            try:
                shifted = _rubberband_pitch_shift(
                    processed,
                    sr,
                    n_steps=n_steps,
                    preserve_formant=True,
                    normalize_peak=True,
                )
                if abs(rate - 1.0) > 1e-4:
                    shifted = _time_stretch_with_selected_engine(shifted, sr, rate=rate)
                return np.asarray(shifted, dtype=np.float32)
            except Exception:
                pass
        else:
            psola = _parselmouth_psola_pitch_time_process(
                processed,
                sr,
                n_steps=n_steps,
                rate=rate,
            )
            if psola is not None:
                return np.asarray(psola, dtype=np.float32)

    if engine in {"pyworld", "world", "pyworld_hpss", "world_hpss"}:
        advanced = _process_advanced_with_hpss(
            processed,
            sr,
            n_steps=n_steps,
            rate=rate,
            flatten_amount=0.0,
            formant_shift=0.0,
        )
        if advanced is not None:
            return np.asarray(advanced, dtype=np.float32)

    if abs(n_steps) > 1e-4:
        processed = _pitch_shift_with_selected_engine(
            processed,
            sr,
            n_steps=n_steps,
        )
    if abs(rate - 1.0) > 1e-4:
        processed = _time_stretch_with_selected_engine(
            processed,
            sr,
            rate=rate,
        )
    return np.asarray(processed, dtype=np.float32)


def _modify_world_f0(
    f0: np.ndarray,
    n_steps: float,
    flatten_amount: float,
) -> np.ndarray:
    source = np.asarray(f0, dtype=np.float64).copy()
    voiced = np.isfinite(source) & (source > 0.0)
    if not np.any(voiced):
        return source

    amount = min(1.0, max(0.0, float(flatten_amount)))
    if amount > 1e-4 and int(np.sum(voiced)) >= 3:
        indices = np.arange(source.size)
        voiced_indices = indices[voiced]
        log_f0 = np.log(np.maximum(source[voiced], 1e-12))
        interpolated = np.interp(indices, voiced_indices, log_f0)
        kernel_size = _odd_kernel_size(min(source.size, 15))
        kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
        smoothed = np.convolve(interpolated, kernel, mode="same")
        source[voiced] = np.exp(
            (1.0 - amount) * np.log(np.maximum(source[voiced], 1e-12))
            + amount * smoothed[voiced]
        )

    if abs(n_steps) > 1e-4:
        source[voiced] *= 2.0 ** (float(n_steps) / 12.0)

    source[~voiced] = 0.0
    return source


def _shift_world_formants(sp: np.ndarray, semitones: float) -> np.ndarray:
    source = np.asarray(sp, dtype=np.float64)
    shift = float(np.clip(semitones, -12.0, 12.0))
    if abs(shift) <= 1e-4 or source.size == 0:
        return source.copy()
    if source.ndim != 2:
        raise ValueError(f"WORLD spectral envelope must be 2D, got {source.ndim}D")

    frame_count, bin_count = source.shape
    if frame_count <= 0 or bin_count <= 1:
        return source.copy()

    scale = 2.0 ** (shift / 12.0)
    bins = np.arange(bin_count, dtype=np.float64)
    source_bins = np.clip(bins / scale, 0.0, float(bin_count - 1))
    shifted = np.empty_like(source)
    for frame_index in range(frame_count):
        shifted[frame_index, :] = np.interp(source_bins, bins, source[frame_index, :])
    shifted = np.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(shifted, 1e-12)


def _odd_kernel_size(size: int) -> int:
    resolved = max(3, int(size))
    if resolved % 2 == 0:
        resolved += 1
    return resolved


def _estimate_pitch_center_hz(audio: np.ndarray, sr: int) -> float | None:
    """Estimate a stable Melodyne-like pitch center for render calculations."""

    source = np.asarray(audio, dtype=np.float32)
    if source.ndim == 2:
        source = np.mean(source, axis=1, dtype=np.float32)
    if source.size < max(128, int(sr * 0.035)) or sr <= 0:
        return None

    world_pitch = _estimate_pitch_center_world(source, sr)
    if world_pitch is not None:
        return world_pitch

    pyin_pitch = _estimate_pitch_center_pyin(source, sr)
    if pyin_pitch is not None:
        return pyin_pitch

    return _estimate_pitch_center_yin(source, sr)


def _estimate_pitch_center_world(source: np.ndarray, sr: int) -> float | None:
    try:
        import pyworld as pw

        world_source = _as_world_float64(source)
        if world_source.size < int(sr * 0.04):
            return None
        f0, time_axis = pw.dio(
            world_source,
            sr,
            f0_floor=50.0,
            f0_ceil=min(2093.0, sr * 0.45),
            frame_period=5.0,
        )
        f0 = pw.stonemask(world_source, f0, time_axis, sr)
        return _voiced_log_median_hz(f0)
    except Exception:
        return None


def _estimate_pitch_center_pyin(source: np.ndarray, sr: int) -> float | None:
    try:
        import librosa

        frame_length = min(2048, int(2 ** math.floor(math.log2(max(256, source.size)))))
        frame_length = max(256, frame_length)
        hop_length = max(64, frame_length // 4)
        f0, voiced_flag, _ = librosa.pyin(
            source,
            fmin=librosa.note_to_hz("C2"),
            fmax=min(librosa.note_to_hz("C7"), sr * 0.45),
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        if voiced_flag is not None:
            f0 = np.asarray(f0)[np.asarray(voiced_flag, dtype=bool)]
        return _voiced_log_median_hz(f0)
    except Exception:
        return None


def _estimate_pitch_center_yin(source: np.ndarray, sr: int) -> float | None:
    try:
        import librosa

        frame_length = min(2048, int(2 ** math.floor(math.log2(max(256, source.size)))))
        frame_length = max(256, frame_length)
        hop_length = max(64, frame_length // 4)
        f0 = librosa.yin(
            source,
            fmin=librosa.note_to_hz("C2"),
            fmax=min(librosa.note_to_hz("C7"), sr * 0.45),
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        return _voiced_log_median_hz(f0)
    except Exception:
        return None


def _voiced_log_median_hz(f0: np.ndarray | None) -> float | None:
    if f0 is None:
        return None
    values = np.asarray(f0, dtype=np.float64)
    values = values[np.isfinite(values)]
    values = values[(values >= 50.0) & (values <= 2200.0)]
    if values.size == 0:
        return None
    if values.size >= 5:
        lower, upper = np.percentile(values, [10, 90])
        trimmed = values[(values >= lower) & (values <= upper)]
        if trimmed.size:
            values = trimmed
    return float(np.exp(np.median(np.log(np.maximum(values, 1e-12)))))


def _should_use_world_engine(
    pitch_flatten_amount: float,
    formant_shift: float,
) -> bool:
    return (
        abs(pitch_flatten_amount) > 1e-4
        or abs(formant_shift) > 1e-4
    )


def _process_multichannel_blob(
    audio: np.ndarray,
    sr: int,
    n_steps: float,
    rate: float,
    gain_db: float,
    pitch_flatten_amount: float,
    formant_shift: float,
    protect_transients: bool,
    pitch_control_points: object = (),
    pitch_vibrato_regions: object = (),
) -> np.ndarray:
    channel_first = audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]
    channels = audio if channel_first else audio.T
    processed_channels = [
        _process_mono_blob(
            channel,
            sr,
            n_steps=n_steps,
            rate=rate,
            gain_db=gain_db,
            pitch_flatten_amount=pitch_flatten_amount,
            formant_shift=formant_shift,
            protect_transients=protect_transients,
            pitch_control_points=pitch_control_points,
            pitch_vibrato_regions=pitch_vibrato_regions,
        )
        for channel in channels
    ]
    target_len = max(len(channel) for channel in processed_channels)
    aligned = [_pad_or_crop(channel, target_len) for channel in processed_channels]
    stacked = np.stack(aligned, axis=0)
    return stacked if channel_first else stacked.T


def _align_percussive(percussive: np.ndarray, target_length: int) -> np.ndarray:
    return _pad_or_crop(np.asarray(percussive, dtype=np.float32), target_length)


def _pad_or_crop(audio: np.ndarray, target_length: int) -> np.ndarray:
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(audio) == target_length:
        return np.asarray(audio, dtype=np.float32)
    if len(audio) > target_length:
        return np.asarray(audio[:target_length], dtype=np.float32)
    output = np.zeros(target_length, dtype=np.float32)
    output[: len(audio)] = audio
    return output


def _safe_float32_mix(audio: np.ndarray) -> np.ndarray:
    output = np.asarray(audio, dtype=np.float32)
    if output.size == 0:
        return output
    peak = float(np.max(np.abs(output)))
    if peak > 1.0:
        output = output / peak
    return np.asarray(output, dtype=np.float32)


def _is_large_pitch_shift(n_steps: float) -> bool:
    return abs(float(n_steps)) > LARGE_PITCH_SHIFT_SEMITONES


def _peak_normalize(
    audio: np.ndarray,
    target_peak: float = RUBBERBAND_NORMALIZED_PEAK,
) -> np.ndarray:
    output = np.asarray(audio, dtype=np.float32)
    if output.size == 0:
        return output
    peak = float(np.max(np.abs(output)))
    if peak <= 1e-8:
        return output
    resolved_peak = min(0.98, max(0.05, float(target_peak)))
    return np.asarray((output / peak) * resolved_peak, dtype=np.float32)


def _apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    # Slice gain is part of the editable signal, not a mastering stage. Limiting
    # every slice separately changes its timbre and makes the requested dB value
    # inaccurate. Playback and export apply one limiter after the full mix.
    return apply_gain(audio, gain_db, soft_limit=False)


def _apply_output_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    return _finite_float32(_apply_gain(audio, gain_db))


def _resize_tiny_or_silent_audio(audio: np.ndarray, target_length: int) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float32)
    resolved_length = max(1, int(target_length))
    if source.size == resolved_length:
        return source.copy()
    if source.size == 0 or float(np.max(np.abs(source))) <= SILENCE_PEAK_EPSILON:
        return np.zeros(resolved_length, dtype=np.float32)
    if source.size == 1:
        return np.full(resolved_length, source[0], dtype=np.float32)
    source_axis = np.linspace(0.0, 1.0, source.size, dtype=np.float64)
    target_axis = np.linspace(0.0, 1.0, resolved_length, dtype=np.float64)
    return np.asarray(np.interp(target_axis, source_axis, source), dtype=np.float32)


def _match_pitch_shift_loudness(
    reference: np.ndarray,
    rendered: np.ndarray,
) -> np.ndarray:
    """Keep pitch-engine level changes from becoming audible gain edits."""

    source = np.asarray(reference, dtype=np.float32)
    output = np.asarray(rendered, dtype=np.float32)
    if source.size == 0 or output.size == 0:
        return output
    source_rms = float(np.sqrt(np.mean(np.square(source, dtype=np.float64))))
    output_rms = float(np.sqrt(np.mean(np.square(output, dtype=np.float64))))
    if source_rms <= 1e-8 or output_rms <= 1e-8:
        return output
    adjustment_db = 20.0 * math.log10(source_rms / output_rms)
    adjustment_db = float(
        np.clip(
            adjustment_db,
            -PITCH_LOUDNESS_MATCH_LIMIT_DB,
            PITCH_LOUDNESS_MATCH_LIMIT_DB,
        )
    )
    return apply_gain(output, adjustment_db, soft_limit=False)


def _finite_float32(audio: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(audio, dtype=np.float32),
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=np.finfo(np.float32).min,
    )


def _rubberband_pitch_shift(
    audio: np.ndarray,
    sr: int,
    n_steps: float,
    preserve_formant: bool = True,
    normalize_peak: bool = False,
) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float32)
    rbargs = {"--fine": "", "--quiet": ""}
    if preserve_formant:
        rbargs["--formant"] = ""
        try:
            rbargs["--pitch"] = f"{float(n_steps):g}"
            shifted = _run_rubberband_cli(source, sr, rbargs)
            return _peak_normalize(shifted) if normalize_peak else shifted
        except Exception:
            pass
    shifted = _run_rubberband_cli(
        source,
        sr,
        {"--fine": "", "--quiet": "", "--pitch": f"{float(n_steps):g}"},
    )
    return _peak_normalize(shifted) if normalize_peak else shifted


def _pitch_shift_with_selected_engine(
    audio: np.ndarray,
    sr: int,
    n_steps: float,
) -> np.ndarray:
    engine = _pitch_engine_name()
    if engine in {
        "adaptive",
        "auto",
        "hybrid",
        "parselmouth",
        "praat",
        "psola",
        "parselmouth_psola",
    }:
        if _is_large_pitch_shift(n_steps):
            try:
                return _rubberband_pitch_shift(
                    audio,
                    sr,
                    n_steps=n_steps,
                    preserve_formant=True,
                    normalize_peak=True,
                )
            except Exception:
                pass
        psola = _parselmouth_psola_pitch_time_process(
            np.asarray(audio, dtype=np.float32),
            sr,
            n_steps=n_steps,
            rate=1.0,
        )
        if psola is not None:
            return np.asarray(psola, dtype=np.float32)

    if engine in {"pyworld", "world", "pyworld_hpss", "world_hpss"}:
        advanced = _process_advanced_with_hpss(
            np.asarray(audio, dtype=np.float32),
            sr,
            n_steps=n_steps,
            rate=1.0,
            flatten_amount=0.0,
            formant_shift=0.0,
        )
        if advanced is not None:
            return np.asarray(advanced, dtype=np.float32)

    if engine in {"librosa", "librosa_phase_vocoder"}:
        try:
            return _librosa_pitch_shift(audio, sr, n_steps=n_steps)
        except Exception:
            return _rubberband_pitch_shift(audio, sr, n_steps=n_steps, preserve_formant=True)

    try:
        return _rubberband_pitch_shift(
            audio,
            sr,
            n_steps=n_steps,
            preserve_formant=True,
            normalize_peak=_is_large_pitch_shift(n_steps),
        )
    except Exception:
        return _librosa_pitch_shift(audio, sr, n_steps=n_steps)


def _rubberband_time_stretch(
    audio: np.ndarray,
    sr: int,
    rate: float,
) -> np.ndarray:
    return _run_rubberband_cli(
        np.asarray(audio, dtype=np.float32),
        sr,
        {"--fine": "", "--quiet": "", "--tempo": f"{float(rate):g}"},
    )


def _run_rubberband_cli(audio: np.ndarray, sr: int, rbargs: dict[str, str]) -> np.ndarray:
    import soundfile as sf

    if sr <= 0:
        raise ValueError("Sample rate must be positive.")

    executable = _rubberband_executable()
    source = np.asarray(audio, dtype=np.float32)
    infile = ""
    outfile = ""
    try:
        in_handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        infile = in_handle.name
        in_handle.close()
        out_handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        outfile = out_handle.name
        out_handle.close()

        sf.write(infile, source, int(sr))
        command = [executable, "-q"]
        for key, value in rbargs.items():
            command.append(str(key))
            if str(value).strip():
                command.append(str(value))
        command.extend([infile, outfile])

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            **hidden_subprocess_kwargs(),
        )
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(error or f"rubberband exited with code {result.returncode}")

        output, _ = sf.read(outfile, dtype="float32", always_2d=True)
        if source.ndim == 1:
            output = np.squeeze(output)
        return np.asarray(output, dtype=np.float32)
    finally:
        for path in (infile, outfile):
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass


def _rubberband_executable() -> str:
    candidate = find_bundled_executable("rubberband.exe")
    if candidate is not None:
        return str(candidate)
    resolved = shutil.which("rubberband")
    if resolved:
        return resolved
    raise RuntimeError(
        "rubberband executable was not found. Install Rubber Band or keep "
        "the bundled tools/rubberband directory available."
    )


def _time_stretch_with_selected_engine(
    audio: np.ndarray,
    sr: int,
    rate: float,
) -> np.ndarray:
    engine = _pitch_engine_name()
    if engine in {"librosa", "librosa_phase_vocoder"}:
        try:
            return _librosa_time_stretch(audio, rate=rate)
        except Exception:
            return _rubberband_time_stretch(audio, sr, rate)

    try:
        return _rubberband_time_stretch(audio, sr, rate)
    except Exception:
        return _librosa_time_stretch(audio, rate=rate)


def transient_protected_time_stretch(
    audio: np.ndarray,
    sr: int,
    stretch_factor: float,
    transient_ms: float = 80.0,
    crossfade_ms: float = 7.0,
    minimum_transient_ms: float = 40.0,
    minimum_vowel_ratio: float = 0.10,
) -> np.ndarray:
    """
    Stretch speech non-linearly while keeping consonant transients intact.

    The normal-path vowel stretch factor is:

        (target_total_length - protected_transient_length) / original_vowel_length

    When an extreme contraction leaves too little room for vowels, the
    protected regions also contract, with a best-effort 40 ms floor.
    """

    source = np.asarray(audio, dtype=np.float32)
    if source.ndim != 1:
        raise ValueError("Transient-protected time stretching expects mono audio.")
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")
    if source.size == 0:
        return source.copy()

    factor = max(0.01, min(100.0, float(stretch_factor)))
    target_length = max(1, int(round(source.size * factor)))
    if target_length == source.size:
        return source.copy()

    try:
        protected_intervals = _detect_transient_intervals(
            source,
            sr,
            transient_samples=max(1, int(round(sr * transient_ms / 1000.0))),
        )
        segments = _segments_from_protected_intervals(source.size, protected_intervals)
        if len(segments) <= 1:
            return _stretch_to_exact_length(source, sr, target_length)

        original_lengths = np.asarray(
            [end - start for start, end, _ in segments],
            dtype=np.int64,
        )
        transient_mask = np.asarray(
            [is_transient for _, _, is_transient in segments],
            dtype=bool,
        )
        target_lengths = _allocate_transient_protected_lengths(
            original_lengths,
            transient_mask,
            target_length,
            sr,
            minimum_transient_ms=minimum_transient_ms,
            minimum_vowel_ratio=minimum_vowel_ratio,
        )

        crossfade_samples = max(0, int(round(sr * crossfade_ms / 1000.0)))
        overlaps = _planned_crossfade_overlaps(target_lengths, crossfade_samples)
        if overlaps:
            compensation = int(sum(overlaps))
            receiver_mask = ~transient_mask
            if not np.any(receiver_mask):
                receiver_mask = np.ones_like(transient_mask)
            target_lengths += _distribute_integer_lengths(
                compensation,
                np.where(receiver_mask, original_lengths, 0),
                minimum=0,
            )

        processed_segments: list[np.ndarray] = []
        for (start, end, _), segment_target in zip(
            segments,
            target_lengths,
            strict=False,
        ):
            processed_segments.append(
                _stretch_to_exact_length(source[start:end], sr, int(segment_target))
            )

        joined = _crossfade_join(processed_segments, crossfade_samples)
        return _pad_or_crop(joined, target_length)
    except Exception as exc:
        print(f"Transient-protected time stretch failed; using full-band fallback: {exc}", flush=True)
        return _stretch_to_exact_length(source, sr, target_length)


def _detect_transient_intervals(
    audio: np.ndarray,
    sr: int,
    transient_samples: int,
) -> list[tuple[int, int]]:
    import librosa

    source = np.asarray(audio, dtype=np.float32)
    hop_length = max(64, min(512, int(round(sr * 0.01))))
    onset_samples = librosa.onset.onset_detect(
        y=source,
        sr=sr,
        hop_length=hop_length,
        units="samples",
        backtrack=True,
    )
    onsets = {0}
    onsets.update(int(np.clip(sample, 0, max(0, source.size - 1))) for sample in onset_samples)
    intervals = [
        (onset, min(source.size, onset + transient_samples))
        for onset in sorted(onsets)
    ]
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _segments_from_protected_intervals(
    source_length: int,
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int, bool]]:
    segments: list[tuple[int, int, bool]] = []
    cursor = 0
    for start, end in intervals:
        start = max(cursor, min(source_length, int(start)))
        end = max(start, min(source_length, int(end)))
        if start > cursor:
            segments.append((cursor, start, False))
        if end > start:
            segments.append((start, end, True))
        cursor = end
    if cursor < source_length:
        segments.append((cursor, source_length, False))
    return segments


def _allocate_transient_protected_lengths(
    original_lengths: np.ndarray,
    transient_mask: np.ndarray,
    target_length: int,
    sr: int,
    minimum_transient_ms: float,
    minimum_vowel_ratio: float,
) -> np.ndarray:
    lengths = np.asarray(original_lengths, dtype=np.int64)
    protected_total = int(np.sum(lengths[transient_mask]))
    vowel_total = int(np.sum(lengths[~transient_mask]))
    desired_vowel_target = int(target_length - protected_total)
    minimum_vowel_target = int(math.ceil(vowel_total * max(0.0, minimum_vowel_ratio)))

    if vowel_total > 0 and desired_vowel_target >= minimum_vowel_target:
        result = np.zeros_like(lengths)
        result[transient_mask] = lengths[transient_mask]
        result[~transient_mask] = _distribute_integer_lengths(
            desired_vowel_target,
            lengths[~transient_mask],
            minimum=1,
        )
        return result

    # Extreme shortening: keep consonants at 40 ms where the target budget
    # permits it, then let all regions share the remaining compression.
    result = np.zeros_like(lengths)
    transient_floor = max(1, int(round(sr * minimum_transient_ms / 1000.0)))
    floors = np.where(transient_mask, np.minimum(lengths, transient_floor), 1)
    floor_total = int(np.sum(floors))
    if floor_total >= target_length:
        return _distribute_integer_lengths(target_length, lengths, minimum=0)

    result[:] = floors
    remaining = target_length - floor_total
    result += _distribute_integer_lengths(remaining, lengths, minimum=0)
    return result


def _distribute_integer_lengths(
    total: int,
    weights: np.ndarray,
    minimum: int = 0,
) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    output = np.zeros(values.size, dtype=np.int64)
    if output.size == 0 or total <= 0:
        return output

    eligible = values > 0
    eligible_count = int(np.sum(eligible))
    minimum = max(0, int(minimum))
    if minimum and eligible_count and total >= eligible_count * minimum:
        output[eligible] = minimum
        total -= eligible_count * minimum
    if total <= 0:
        return output

    weight_total = float(np.sum(values[eligible]))
    if weight_total <= 0.0:
        output[0] += total
        return output
    exact = np.where(eligible, values / weight_total * total, 0.0)
    additions = np.floor(exact).astype(np.int64)
    output += additions
    remainder = int(total - np.sum(additions))
    if remainder > 0:
        order = np.argsort(-(exact - additions))
        output[order[:remainder]] += 1
    return output


def _planned_crossfade_overlaps(
    lengths: np.ndarray,
    crossfade_samples: int,
) -> list[int]:
    if crossfade_samples <= 0:
        return []
    return [
        max(0, min(crossfade_samples, int(left) // 2, int(right) // 2))
        for left, right in zip(lengths[:-1], lengths[1:], strict=False)
    ]


def _stretch_to_exact_length(
    audio: np.ndarray,
    sr: int,
    target_length: int,
) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float32)
    target_length = max(0, int(target_length))
    if target_length == 0:
        return np.zeros(0, dtype=np.float32)
    if source.size == 0:
        return np.zeros(target_length, dtype=np.float32)
    if source.size == target_length:
        return source.copy()
    if source.size < 64 or target_length < 64:
        source_positions = np.linspace(0.0, 1.0, source.size, endpoint=True)
        target_positions = np.linspace(0.0, 1.0, target_length, endpoint=True)
        return np.asarray(np.interp(target_positions, source_positions, source), dtype=np.float32)
    stretched = _time_stretch_with_selected_engine(
        source,
        sr,
        rate=float(source.size) / float(target_length),
    )
    return _pad_or_crop(stretched, target_length)


def _crossfade_join(
    segments: list[np.ndarray],
    crossfade_samples: int,
) -> np.ndarray:
    if not segments:
        return np.zeros(0, dtype=np.float32)
    output = np.asarray(segments[0], dtype=np.float32).copy()
    for segment in segments[1:]:
        right = np.asarray(segment, dtype=np.float32)
        overlap = max(
            0,
            min(crossfade_samples, output.size // 2, right.size // 2),
        )
        if overlap <= 0:
            output = np.concatenate((output, right))
            continue
        fade_in = np.linspace(0.0, 1.0, overlap, endpoint=True, dtype=np.float32)
        fade_out = 1.0 - fade_in
        blend = output[-overlap:] * fade_out + right[:overlap] * fade_in
        output = np.concatenate((output[:-overlap], blend, right[overlap:]))
    return np.asarray(output, dtype=np.float32)


def _pitch_engine_name() -> str:
    return os.environ.get("HAKYKING_PITCH_ENGINE", DEFAULT_PITCH_ENGINE).strip().lower()


def _parselmouth_psola_pitch_time_process(
    audio: np.ndarray,
    sr: int,
    n_steps: float,
    rate: float,
) -> np.ndarray | None:
    try:
        import parselmouth
        from parselmouth.praat import call
        from scipy.signal import medfilt

        source = np.asarray(audio, dtype=np.float32)
        if source.size < max(64, int(sr * 0.04)):
            return source.copy()

        sound = parselmouth.Sound(
            np.asarray(source, dtype=np.float64),
            sampling_frequency=float(sr),
        )
        pitch_floor, pitch_ceiling = _praat_pitch_bounds(source, sr)
        manipulation = call(sound, "To Manipulation", 0.01, pitch_floor, pitch_ceiling)
        pitch_tier = call(manipulation, "Extract pitch tier")
        point_count = int(call(pitch_tier, "Get number of points"))
        if point_count <= 0:
            return None

        times = [
            float(call(pitch_tier, "Get time from index", index))
            for index in range(1, point_count + 1)
        ]
        frequencies = np.asarray(
            [
                float(call(pitch_tier, "Get value at index", index))
                for index in range(1, point_count + 1)
            ],
            dtype=np.float64,
        )
        smoothed = _median_filter_pitch_points(frequencies, medfilt)
        factor = 2.0 ** (float(n_steps) / 12.0)
        shifted = np.clip(smoothed * factor, 40.0, min(2400.0, sr * 0.45))

        for index in range(point_count, 0, -1):
            call(pitch_tier, "Remove point", index)
        for time_value, frequency in zip(times, shifted, strict=False):
            if np.isfinite(frequency) and frequency > 0.0:
                call(pitch_tier, "Add point", float(time_value), float(frequency))

        call([pitch_tier, manipulation], "Replace pitch tier")
        resynthesized = call(manipulation, "Get resynthesis (overlap-add)")
        output = np.asarray(resynthesized.values, dtype=np.float32)
        if output.ndim == 2:
            output = output[0] if output.shape[0] == 1 else np.mean(output, axis=0, dtype=np.float32)

        safe_rate = min(MAX_AUDIO_RATE, max(MIN_AUDIO_RATE, float(rate)))
        if abs(safe_rate - 1.0) > 1e-4:
            output = _time_stretch_with_selected_engine(output, sr, safe_rate)
        else:
            output = _pad_or_crop(output, len(source))
        return _safe_float32_mix(output)
    except Exception as exc:
        print(f"Parselmouth PSOLA pitch shift failed: {exc}", flush=True)
        return None


def _median_filter_pitch_points(frequencies: np.ndarray, medfilt) -> np.ndarray:
    values = np.asarray(frequencies, dtype=np.float64)
    if values.size < 3:
        return values.copy()
    kernel_size = min(5, values.size if values.size % 2 == 1 else values.size - 1)
    kernel_size = max(3, kernel_size)
    return np.asarray(medfilt(values, kernel_size=kernel_size), dtype=np.float64)


def _praat_pitch_bounds(audio: np.ndarray, sr: int) -> tuple[float, float]:
    center = _estimate_pitch_center_hz(audio, sr)
    if center is None:
        return 60.0, min(800.0, sr * 0.45)
    floor = max(40.0, center / 3.0)
    ceiling = min(max(600.0, center * 4.0), sr * 0.45, 2400.0)
    if ceiling <= floor + 100.0:
        ceiling = min(max(floor + 100.0, 600.0), sr * 0.45, 2400.0)
    return float(floor), float(ceiling)


def _librosa_pitch_shift(
    audio: np.ndarray,
    sr: int,
    n_steps: float,
) -> np.ndarray:
    import librosa

    source = np.asarray(audio, dtype=np.float32)
    shifted = librosa.effects.pitch_shift(
        source,
        sr=sr,
        n_steps=n_steps,
    )
    return np.asarray(shifted, dtype=np.float32)


def _librosa_time_stretch(
    audio: np.ndarray,
    rate: float,
) -> np.ndarray:
    import librosa

    return np.asarray(
        librosa.effects.time_stretch(np.asarray(audio, dtype=np.float32), rate=rate),
        dtype=np.float32,
    )


def _flatten_pitch_variation(
    audio: np.ndarray,
    sr: int,
    amount: float,
) -> np.ndarray:
    """
    Lightweight vibrato flattening fallback.

    It estimates local F0 drift with the shared vocal analysis stack
    (RMVPE-ONNX when available, then pyworld/librosa fallbacks) and applies
    small opposite pitch shifts to overlapping chunks. This is intentionally
    conservative, but the pitch tracking layer is now much stronger than the
    older pYIN-only path.
    """

    source = np.asarray(audio, dtype=np.float32)
    amount = min(1.0, max(0.0, float(amount)))
    if source.size < int(sr * 0.18) or amount <= 1e-4:
        return source

    try:
        import librosa

        midi, frame_times, frame_length = _flatten_pitch_track(source, sr)
        valid = np.isfinite(midi)
        if int(np.sum(valid)) < 3:
            return source

        center_midi = float(np.nanmedian(midi[valid]))
        window_length = max(int(sr * 0.18), frame_length)
        hop = max(1, window_length // 2)
        output = np.zeros(len(source) + window_length, dtype=np.float32)
        weights = np.zeros_like(output)
        window = np.hanning(window_length).astype(np.float32)
        if float(np.max(window)) <= 0:
            window = np.ones(window_length, dtype=np.float32)

        starts = range(0, max(1, len(source) - window_length + hop), hop)
        for start in starts:
            end = min(len(source), start + window_length)
            segment = source[start:end]
            if segment.size < frame_length // 2:
                continue
            mid_time = (start + end) * 0.5 / sr
            frame_index = int(np.argmin(np.abs(frame_times - mid_time)))
            left = max(0, frame_index - 1)
            right = min(len(midi), frame_index + 2)
            local_values = midi[left:right]
            local_values = local_values[np.isfinite(local_values)]
            if local_values.size == 0:
                shifted = segment
            else:
                local_midi = float(np.median(local_values))
                correction = float(np.clip((center_midi - local_midi) * amount, -4.0, 4.0))
                if abs(correction) <= 1e-3:
                    shifted = segment
                else:
                    shifted = librosa.effects.pitch_shift(
                        segment,
                        sr=sr,
                        n_steps=correction,
                    )
                    shifted = _pad_or_crop(np.asarray(shifted, dtype=np.float32), len(segment))

            local_window = window[: len(segment)]
            output[start:end] += shifted * local_window
            weights[start:end] += local_window

        output = output[: len(source)]
        weights = weights[: len(source)]
        untouched = weights <= 1e-6
        weights[untouched] = 1.0
        output = output / weights
        output[untouched] = source[untouched]
        return _safe_float32_mix(output)
    except Exception:
        return source


def _flatten_pitch_track(source: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, int]:
    try:
        from hakyking.audio.vocal_analysis import analyze_vocal_audio

        old_rmvpe_enabled = os.environ.get("HAKYKING_ENABLE_RMVPE")
        os.environ["HAKYKING_ENABLE_RMVPE"] = "0"
        try:
            analysis = analyze_vocal_audio(source, sr, source_path="", note_intervals=[])
        finally:
            if old_rmvpe_enabled is None:
                os.environ.pop("HAKYKING_ENABLE_RMVPE", None)
            else:
                os.environ["HAKYKING_ENABLE_RMVPE"] = old_rmvpe_enabled
        f0 = np.asarray(analysis.pitch_track.f0_hz, dtype=np.float64)
        times = np.asarray(analysis.pitch_track.times, dtype=np.float64)
        if f0.size and times.size == f0.size:
            midi = np.full(f0.shape, np.nan, dtype=np.float32)
            valid = np.isfinite(f0) & (f0 > 0.0)
            midi[valid] = (69.0 + 12.0 * np.log2(f0[valid] / 440.0)).astype(np.float32)
            if int(np.sum(np.isfinite(midi))) >= 3:
                frame_length = max(256, min(2048, int(round(sr * 0.05))))
                return midi, times, frame_length
    except Exception:
        pass

    import librosa

    frame_length = min(2048, int(2 ** math.floor(math.log2(max(256, len(source))))))
    frame_length = max(256, frame_length)
    hop_length = max(128, frame_length // 4)
    f0, _, _ = librosa.pyin(
        source,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    midi = librosa.hz_to_midi(f0)
    frame_times = librosa.frames_to_time(
        np.arange(len(midi)),
        sr=sr,
        hop_length=hop_length,
    )
    return np.asarray(midi, dtype=np.float32), np.asarray(frame_times, dtype=np.float64), frame_length


def _apply_formant_shift_approx(
    audio: np.ndarray,
    sr: int,
    semitones: float,
) -> np.ndarray:
    """
    Shift the spectral envelope without changing the explicit pitch parameter.

    This fallback keeps the fine harmonic structure and phase in place, then
    warps only a smoothed log-magnitude envelope. It is not as clean as WORLD or
    Praat, but it gives the formant tool a non-destructive, decoupled engine.
    """

    source = np.asarray(audio, dtype=np.float32)
    semitones = float(np.clip(semitones, -12.0, 12.0))
    if source.size < 128 or abs(semitones) <= 1e-4:
        return source

    try:
        import librosa

        n_fft = min(2048, int(2 ** math.floor(math.log2(max(128, len(source))))))
        n_fft = max(128, n_fft)
        hop_length = max(32, n_fft // 4)
        stft = librosa.stft(source, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft).astype(np.float32)
        phase = np.exp(1j * np.angle(stft))
        if magnitude.size == 0:
            return source

        log_mag = np.log(np.maximum(magnitude, 1e-7))
        kernel_width = max(7, min(45, magnitude.shape[0] // 10 * 2 + 1))
        kernel = np.ones(kernel_width, dtype=np.float32) / kernel_width
        smooth = np.apply_along_axis(
            lambda column: np.convolve(column, kernel, mode="same"),
            0,
            log_mag,
        )

        factor = 2.0 ** (semitones / 12.0)
        bins = np.arange(magnitude.shape[0], dtype=np.float32)
        source_bins = np.clip(bins / factor, 0, magnitude.shape[0] - 1)
        warped_smooth = np.empty_like(smooth)
        for frame in range(smooth.shape[1]):
            warped_smooth[:, frame] = np.interp(source_bins, bins, smooth[:, frame])

        envelope_delta = np.exp(np.clip(warped_smooth - smooth, -2.0, 2.0))
        warped_mag = magnitude * envelope_delta
        reconstructed = librosa.istft(
            warped_mag * phase,
            hop_length=hop_length,
            length=len(source),
        )
        return _safe_float32_mix(reconstructed)
    except Exception:
        return source


def _as_float32_copy(audio: np.ndarray) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float32)
    if not np.all(np.isfinite(source)):
        source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
    return source.copy()


def _as_world_float64(audio: np.ndarray) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float64)
    source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(source))) if source.size else 0.0
    if peak > 1.0:
        source = source / peak
    return np.ascontiguousarray(source, dtype=np.float64)


def _ensure_rubberband_available() -> None:
    configure_external_tool_paths()
    local_rubberband = _local_rubberband_dir()
    if local_rubberband is not None:
        local_path = str(local_rubberband)
        current_path = os.environ.get("PATH", "")
        if local_path.lower() not in current_path.lower():
            os.environ["PATH"] = local_path + os.pathsep + current_path

    if shutil.which("rubberband") is None:
        raise RuntimeError(
            "rubberband executable was not found. Install Rubber Band or keep "
            "the bundled tools/rubberband directory available."
        )


def _local_rubberband_dir() -> Path | None:
    candidate = find_bundled_executable("rubberband.exe")
    return candidate.parent if candidate is not None else None


def render_parameters_to_dict(parameters: RenderParameters) -> dict[str, Any]:
    return {
        "target_midi_note": parameters.target_midi_note,
        "target_duration": parameters.target_duration,
        "n_steps": parameters.n_steps,
        "rate": parameters.rate,
        "gain_db": parameters.gain_db,
        "pitch_flatten_amount": parameters.pitch_flatten_amount,
        "formant_shift": parameters.formant_shift,
        "protect_transients": parameters.protect_transients,
        "pitch_control_points": parameters.pitch_control_points,
        "pitch_vibrato_regions": parameters.pitch_vibrato_regions,
    }
