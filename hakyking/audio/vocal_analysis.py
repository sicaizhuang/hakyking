from __future__ import annotations

import importlib.util
import math
import os
import threading
from collections import OrderedDict
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from hakyking.audio.reader import AudioReader
from hakyking.models.audio_slice import AudioSlice
from hakyking.runtime import user_data_root


DEFAULT_FRAME_PERIOD_MS = 5.0
DEFAULT_FMIN_HZ = 50.0
DEFAULT_FMAX_HZ = 2093.0
DEFAULT_RMVPE_MODEL_PATH = user_data_root() / "models" / "rmvpe" / "rmvpe.onnx"
DEFAULT_RMVPE_CONFIDENCE = 0.03
DEFAULT_RMVPE_MAX_SECONDS = 12.0

_RMVPE_MODEL = None


@dataclass(frozen=True)
class PitchTrack:
    times: np.ndarray
    f0_hz: np.ndarray
    confidence: np.ndarray
    backend: str

    @property
    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.f0_hz) & (self.f0_hz > 0.0)

    @property
    def valid_ratio(self) -> float:
        if self.f0_hz.size == 0:
            return 0.0
        return float(np.count_nonzero(self.valid_mask) / self.f0_hz.size)

    @property
    def median_confidence(self) -> float:
        values = self.confidence[np.isfinite(self.confidence)]
        return 0.0 if values.size == 0 else float(np.median(values))


@dataclass(frozen=True)
class PitchFrame:
    time: float
    f0_hz: float | None
    midi: float | None
    confidence: float
    backend: str


@dataclass(frozen=True)
class VocalNote:
    source_path: str
    index: int
    start_time: float
    end_time: float
    midi_note: int | None
    f0_hz: float | None
    confidence: float
    backend: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_audio_slice(self) -> AudioSlice:
        return AudioSlice(
            source_path=self.source_path,
            index=self.index,
            start_time=self.start_time,
            end_time=self.end_time,
            midi_note=self.midi_note,
            f0_hz=self.f0_hz,
            pitch_confidence=self.confidence,
            analysis_backend=self.backend,
        )


@dataclass(frozen=True)
class VocalAnalysisResult:
    source_path: str
    sample_rate: int
    duration: float
    pitch_track: PitchTrack
    frames: tuple[PitchFrame, ...]
    notes: tuple[VocalNote, ...]
    attempted_backends: tuple[str, ...]
    active_backends: tuple[str, ...]
    warnings: tuple[str, ...]


_SOURCE_PITCH_CACHE_MAX_ENTRIES = 12
_SOURCE_PITCH_CACHE: OrderedDict[
    tuple[str, int, int], PitchTrack
] = OrderedDict()
_SOURCE_PITCH_CACHE_LOCK = threading.RLock()


def cache_source_pitch_track(path: str, pitch_track: PitchTrack) -> None:
    """Keep one reusable source-level F0 track for slice waveform workers."""

    cache_key = _source_pitch_cache_key(path)
    if cache_key is None or pitch_track.f0_hz.size == 0:
        return
    with _SOURCE_PITCH_CACHE_LOCK:
        stale_keys = [key for key in _SOURCE_PITCH_CACHE if key[0] == cache_key[0]]
        for stale_key in stale_keys:
            _SOURCE_PITCH_CACHE.pop(stale_key, None)
        _SOURCE_PITCH_CACHE[cache_key] = pitch_track
        _SOURCE_PITCH_CACHE.move_to_end(cache_key)
        while len(_SOURCE_PITCH_CACHE) > _SOURCE_PITCH_CACHE_MAX_ENTRIES:
            _SOURCE_PITCH_CACHE.popitem(last=False)


def cached_source_pitch_track(path: str) -> PitchTrack | None:
    cache_key = _source_pitch_cache_key(path)
    if cache_key is None:
        return None
    with _SOURCE_PITCH_CACHE_LOCK:
        pitch_track = _SOURCE_PITCH_CACHE.get(cache_key)
        if pitch_track is not None:
            _SOURCE_PITCH_CACHE.move_to_end(cache_key)
        return pitch_track


def cached_pitch_contour_for_slice(
    audio_slice: AudioSlice,
    max_points: int,
) -> np.ndarray | None:
    """Return a compact contour without re-analyzing an individual slice."""

    pitch_track = cached_source_pitch_track(audio_slice.source_path)
    if pitch_track is None or max_points <= 0:
        return None
    mask = (
        (pitch_track.times >= float(audio_slice.start_time))
        & (pitch_track.times <= float(audio_slice.end_time))
    )
    if not np.any(mask):
        return np.zeros(0, dtype=np.float32)
    midi = _hz_to_midi_array(pitch_track.f0_hz[mask])
    return _compact_midi_contour(midi, max_points=max_points)


def clear_source_pitch_cache() -> None:
    with _SOURCE_PITCH_CACHE_LOCK:
        _SOURCE_PITCH_CACHE.clear()


def _source_pitch_cache_key(path: str) -> tuple[str, int, int] | None:
    try:
        source = Path(path).expanduser().resolve(strict=True)
        stat = source.stat()
    except (OSError, RuntimeError, ValueError):
        return None
    return str(source), int(stat.st_mtime_ns), int(stat.st_size)


def available_analysis_backends() -> dict[str, bool]:
    """Report optional analysis engines without importing heavy modules."""

    return {
        "rmvpe_onnx": importlib.util.find_spec("rmvpe_onnx") is not None,
        "pyworld": importlib.util.find_spec("pyworld") is not None,
        "librosa_pyin": importlib.util.find_spec("librosa") is not None,
        "audioflux_onset": importlib.util.find_spec("audioflux") is not None,
        "torchcrepe": importlib.util.find_spec("torchcrepe") is not None,
        "crepe": importlib.util.find_spec("crepe") is not None,
        "basic_pitch": importlib.util.find_spec("basic_pitch") is not None,
        "rmvpe_external": bool(os.environ.get("HAKYKING_RMVPE_COMMAND")),
    }


def analyze_vocal_file(path: str) -> VocalAnalysisResult:
    audio, sample_rate = AudioReader.load_mono(path)
    return analyze_vocal_audio(audio, sample_rate, source_path=path)


def analyze_vocal_audio(
    audio: np.ndarray,
    sample_rate: int,
    source_path: str = "",
    note_intervals: Iterable[tuple[float, float]] | None = None,
) -> VocalAnalysisResult:
    source = _as_mono_float32(audio)
    duration = float(source.size / sample_rate) if sample_rate > 0 else 0.0
    warnings: list[str] = []
    attempted: list[str] = []
    tracks: list[PitchTrack] = []

    for backend_name, extractor in _enabled_pitch_extractors():
        attempted.append(backend_name)
        try:
            track = extractor(source, sample_rate)
        except Exception as exc:  # noqa: BLE001 - optional backends must not break the app
            warnings.append(f"{backend_name}: {exc}")
            continue
        if track.f0_hz.size and track.valid_ratio > 0.02:
            tracks.append(track)
        elif backend_name == "rmvpe_onnx" and _rmvpe_should_skip(source, sample_rate):
            continue
        else:
            warnings.append(f"{backend_name}: no usable voiced frames")

    pitch_track = _fuse_pitch_tracks(tracks, sample_rate, duration)
    frames = _frames_from_track(pitch_track)

    if note_intervals is None:
        note_intervals = suggest_note_intervals(source, sample_rate, pitch_track)
    notes = tuple(
        _note_from_interval(
            source_path=source_path,
            index=index,
            start_time=float(start_time),
            end_time=float(end_time),
            pitch_track=pitch_track,
        )
        for index, (start_time, end_time) in enumerate(note_intervals)
        if end_time > start_time
    )

    return VocalAnalysisResult(
        source_path=source_path,
        sample_rate=sample_rate,
        duration=duration,
        pitch_track=pitch_track,
        frames=frames,
        notes=notes,
        attempted_backends=tuple(attempted),
        active_backends=tuple(track.backend for track in tracks),
        warnings=tuple(warnings),
    )


def analyze_fast_pitch_audio(audio: np.ndarray, sample_rate: int) -> PitchTrack:
    """Build a coarse source F0 track for long-media first display.

    The long-file path deliberately favors bounded latency. It runs one YIN
    pass at roughly 20 ms resolution and uses an RMS gate to preserve unvoiced
    gaps. Fine per-slice rendering can still use the configured DSP engine.
    """

    import librosa

    source = _as_mono_float32(audio)
    if source.size < max(256, int(sample_rate * 0.04)) or sample_rate <= 0:
        return _empty_track("librosa_yin_fast")

    frame_length = 1024 if sample_rate <= 24000 else 2048
    frame_length = min(frame_length, int(2 ** np.floor(np.log2(source.size))))
    frame_length = max(256, frame_length)
    hop_length = max(64, int(round(sample_rate * 0.02)))
    fmax = min(DEFAULT_FMAX_HZ, sample_rate * 0.45)
    if fmax <= DEFAULT_FMIN_HZ:
        return _empty_track("librosa_yin_fast")

    f0 = librosa.yin(
        source,
        fmin=DEFAULT_FMIN_HZ,
        fmax=fmax,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    rms = librosa.feature.rms(
        y=source,
        frame_length=frame_length,
        hop_length=hop_length,
    ).reshape(-1)
    frame_count = min(f0.size, rms.size)
    f0 = np.asarray(f0[:frame_count], dtype=np.float32)
    rms = np.asarray(rms[:frame_count], dtype=np.float32)
    peak_rms = float(np.max(rms)) if rms.size else 0.0
    relative_db = np.full(rms.shape, -120.0, dtype=np.float32)
    if peak_rms > 1e-9:
        relative_db = (20.0 * np.log10(np.maximum(rms, 1e-12) / peak_rms)).astype(
            np.float32
        )
    valid = (
        np.isfinite(f0)
        & (f0 >= DEFAULT_FMIN_HZ)
        & (f0 <= fmax)
        & (relative_db >= -42.0)
    )
    f0 = np.where(valid, f0, np.nan).astype(np.float32)
    confidence = np.where(
        valid,
        np.clip((relative_db + 42.0) / 42.0, 0.25, 0.78),
        0.0,
    ).astype(np.float32)
    times = librosa.frames_to_time(
        np.arange(frame_count),
        sr=sample_rate,
        hop_length=hop_length,
    )
    return PitchTrack(
        times=np.asarray(times, dtype=np.float64),
        f0_hz=f0,
        confidence=confidence,
        backend="librosa_yin_fast",
    )


def annotate_slices_with_pitch_track(
    path: str,
    audio_slices: list[AudioSlice],
    pitch_track: PitchTrack,
) -> list[AudioSlice]:
    if not audio_slices or pitch_track.f0_hz.size == 0:
        return list(audio_slices)

    enriched: list[AudioSlice] = []
    for original in audio_slices:
        note = _note_from_interval(
            source_path=path,
            index=original.index,
            start_time=original.start_time,
            end_time=original.end_time,
            pitch_track=pitch_track,
        )
        enriched.append(
            AudioSlice(
                source_path=original.source_path,
                index=original.index,
                start_time=original.start_time,
                end_time=original.end_time,
                midi_note=note.midi_note if note.midi_note is not None else original.midi_note,
                f0_hz=note.f0_hz if note.f0_hz is not None else original.f0_hz,
                pitch_confidence=note.confidence,
                analysis_backend=note.backend,
            )
        )
    cache_source_pitch_track(path, pitch_track)
    return enriched


def annotate_slices_with_vocal_analysis(
    path: str,
    audio: np.ndarray,
    sample_rate: int,
    audio_slices: list[AudioSlice],
) -> list[AudioSlice]:
    if not audio_slices:
        return []

    intervals = [(item.start_time, item.end_time) for item in audio_slices]
    analysis = analyze_vocal_audio(
        audio,
        sample_rate,
        source_path=path,
        note_intervals=intervals,
    )
    if not analysis.pitch_track.f0_hz.size:
        return audio_slices
    return annotate_slices_with_pitch_track(path, audio_slices, analysis.pitch_track)


def compact_pitch_contour_for_audio(
    audio: np.ndarray,
    sample_rate: int,
    max_points: int,
) -> np.ndarray:
    analysis = analyze_vocal_audio(audio, sample_rate, source_path="", note_intervals=[])
    midi = _hz_to_midi_array(analysis.pitch_track.f0_hz)
    return _compact_midi_contour(midi, max_points=max_points)


def summarize_pitch_track(
    pitch_track: PitchTrack,
    start_time: float,
    end_time: float,
) -> tuple[float | None, int | None, float]:
    if pitch_track.f0_hz.size == 0:
        return None, None, 0.0
    start = max(0.0, float(start_time))
    end = max(start, float(end_time))
    mask = (
        (pitch_track.times >= start)
        & (pitch_track.times <= end)
        & pitch_track.valid_mask
    )
    values = pitch_track.f0_hz[mask]
    if values.size == 0:
        return None, None, 0.0
    confidence = pitch_track.confidence[mask]
    f0_hz = _robust_center_hz(values, confidence)
    midi_note = hz_to_midi_int(f0_hz)
    confidence_value = _confidence_center(confidence)
    return f0_hz, midi_note, confidence_value


def suggest_note_intervals(
    audio: np.ndarray,
    sample_rate: int,
    pitch_track: PitchTrack | None = None,
) -> list[tuple[float, float]]:
    source = _as_mono_float32(audio)
    if source.size == 0 or sample_rate <= 0:
        return []

    try:
        import librosa

        raw_intervals = librosa.effects.split(source, top_db=35)
    except Exception:
        raw_intervals = np.asarray([[0, source.size]], dtype=np.int64)

    if raw_intervals.size == 0:
        return []

    intervals: list[tuple[int, int]] = []
    min_samples = max(1, int(round(sample_rate * 0.045)))
    for start_sample, end_sample in raw_intervals:
        start_sample = int(start_sample)
        end_sample = int(end_sample)
        if end_sample - start_sample < min_samples:
            continue
        boundaries = _suggest_internal_boundaries(
            source,
            sample_rate,
            start_sample,
            end_sample,
            pitch_track,
        )
        points = [start_sample, *boundaries, end_sample]
        for left, right in zip(points, points[1:], strict=False):
            if right - left >= min_samples:
                intervals.append((left, right))

    return [
        (float(start_sample / sample_rate), float(end_sample / sample_rate))
        for start_sample, end_sample in _make_contiguous_intervals(intervals, source.size)
    ]


def hz_to_midi_float(f0_hz: float | None) -> float | None:
    if f0_hz is None or f0_hz <= 0.0 or not math.isfinite(f0_hz):
        return None
    return 69.0 + 12.0 * math.log2(f0_hz / 440.0)


def hz_to_midi_int(f0_hz: float | None) -> int | None:
    midi = hz_to_midi_float(f0_hz)
    if midi is None:
        return None
    return max(0, min(127, int(round(midi))))


def midi_to_hz(midi_note: float | None) -> float | None:
    if midi_note is None or not math.isfinite(float(midi_note)):
        return None
    return float(440.0 * (2.0 ** ((float(midi_note) - 69.0) / 12.0)))


def _enabled_pitch_extractors():
    # Neural trackers are useful but their native runtimes are large and can
    # fail to initialize on Windows. They are opt-in until diagnostics confirm
    # the local runtime; WORLD + pYIN remain the stable default combination.
    if os.environ.get("HAKYKING_ENABLE_NEURAL_PITCH", "0") == "1":
        if (
            os.environ.get("HAKYKING_ENABLE_RMVPE", "1") != "0"
            and importlib.util.find_spec("rmvpe_onnx") is not None
        ):
            yield "rmvpe_onnx", _pitch_track_rmvpe_onnx
        if importlib.util.find_spec("torchcrepe") is not None:
            yield "torchcrepe", _pitch_track_torchcrepe
        if importlib.util.find_spec("crepe") is not None:
            yield "crepe", _pitch_track_crepe
    if importlib.util.find_spec("pyworld") is not None:
        yield "pyworld", _pitch_track_pyworld
    if (
        os.environ.get("HAKYKING_ENABLE_PYIN_ANALYSIS", "0") == "1"
        and importlib.util.find_spec("librosa") is not None
    ):
        yield "librosa_pyin", _pitch_track_librosa_pyin


def _pitch_track_rmvpe_onnx(source: np.ndarray, sample_rate: int) -> PitchTrack:
    from rmvpe_onnx import RMVPE

    if source.size < int(sample_rate * 0.04):
        return _empty_track("rmvpe_onnx")
    if _rmvpe_should_skip(source, sample_rate):
        return _empty_track("rmvpe_onnx")

    global _RMVPE_MODEL
    if _RMVPE_MODEL is None:
        model_path = Path(
            os.environ.get("HAKYKING_RMVPE_MODEL", str(DEFAULT_RMVPE_MODEL_PATH))
        )
        device = os.environ.get("HAKYKING_RMVPE_DEVICE", "cpu").strip() or "cpu"
        _RMVPE_MODEL = RMVPE(model_path=model_path, device=device)

    times, f0, confidence, _activation = _RMVPE_MODEL.predict(
        np.asarray(source, dtype=np.float32),
        int(sample_rate),
    )
    confidence = np.nan_to_num(np.asarray(confidence, dtype=np.float32), nan=0.0)
    f0 = np.asarray(f0, dtype=np.float32)
    threshold = _float_env(
        "HAKYKING_RMVPE_CONFIDENCE",
        DEFAULT_RMVPE_CONFIDENCE,
        minimum=0.0,
        maximum=1.0,
    )
    valid = (
        np.isfinite(f0)
        & (f0 >= DEFAULT_FMIN_HZ)
        & (f0 <= min(DEFAULT_FMAX_HZ, sample_rate * 0.45))
        & (confidence >= threshold)
    )
    f0 = np.where(valid, f0, np.nan).astype(np.float32)
    return PitchTrack(
        times=np.asarray(times, dtype=np.float64),
        f0_hz=f0,
        confidence=confidence,
        backend="rmvpe_onnx",
    )


def _pitch_track_pyworld(source: np.ndarray, sample_rate: int) -> PitchTrack:
    import pyworld as pw

    if source.size < int(sample_rate * 0.04):
        return _empty_track("pyworld")
    world_source = _as_world_float64(source)
    f0, time_axis = pw.dio(
        world_source,
        sample_rate,
        f0_floor=DEFAULT_FMIN_HZ,
        f0_ceil=min(DEFAULT_FMAX_HZ, sample_rate * 0.45),
        frame_period=DEFAULT_FRAME_PERIOD_MS,
    )
    f0 = pw.stonemask(world_source, f0, time_axis, sample_rate)
    confidence = np.where(np.asarray(f0) > 0.0, 0.82, 0.0).astype(np.float32)
    return PitchTrack(
        times=np.asarray(time_axis, dtype=np.float64),
        f0_hz=np.asarray(f0, dtype=np.float32),
        confidence=confidence,
        backend="pyworld",
    )


def _pitch_track_librosa_pyin(source: np.ndarray, sample_rate: int) -> PitchTrack:
    import librosa

    if source.size < int(sample_rate * 0.04):
        return _empty_track("librosa_pyin")
    frame_length = min(2048, int(2 ** np.floor(np.log2(max(256, source.size)))))
    frame_length = max(256, frame_length)
    hop_length = max(64, min(512, frame_length // 4))
    f0, voiced_flag, voiced_prob = librosa.pyin(
        source,
        fmin=DEFAULT_FMIN_HZ,
        fmax=min(DEFAULT_FMAX_HZ, sample_rate * 0.45),
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    times = librosa.frames_to_time(
        np.arange(len(f0)),
        sr=sample_rate,
        hop_length=hop_length,
    )
    confidence = np.nan_to_num(voiced_prob, nan=0.0).astype(np.float32)
    f0 = np.where(voiced_flag, f0, np.nan)
    return PitchTrack(
        times=np.asarray(times, dtype=np.float64),
        f0_hz=np.asarray(f0, dtype=np.float32),
        confidence=confidence,
        backend="librosa_pyin",
    )


def _pitch_track_torchcrepe(source: np.ndarray, sample_rate: int) -> PitchTrack:
    import torch
    import torchcrepe

    if source.size < int(sample_rate * 0.04):
        return _empty_track("torchcrepe")
    hop_length = max(80, int(round(sample_rate * DEFAULT_FRAME_PERIOD_MS / 1000.0)))
    tensor = torch.tensor(source, dtype=torch.float32)[None]
    with torch.no_grad():
        result = torchcrepe.predict(
            tensor,
            sample_rate,
            hop_length,
            DEFAULT_FMIN_HZ,
            min(DEFAULT_FMAX_HZ, sample_rate * 0.45),
            model="tiny",
            batch_size=512,
            device="cpu",
            return_periodicity=True,
            pad=True,
        )
    if isinstance(result, tuple):
        f0_tensor, confidence_tensor = result[0], result[1]
    else:
        f0_tensor = result
        confidence_tensor = torch.ones_like(result)
    f0 = np.asarray(f0_tensor.squeeze().cpu().numpy(), dtype=np.float32)
    confidence = np.asarray(confidence_tensor.squeeze().cpu().numpy(), dtype=np.float32)
    times = np.arange(f0.size, dtype=np.float64) * hop_length / float(sample_rate)
    f0 = np.where(confidence >= 0.25, f0, np.nan)
    return PitchTrack(times=times, f0_hz=f0, confidence=confidence, backend="torchcrepe")


def _pitch_track_crepe(source: np.ndarray, sample_rate: int) -> PitchTrack:
    import crepe

    if source.size < int(sample_rate * 0.04):
        return _empty_track("crepe")
    times, f0, confidence, _activation = crepe.predict(
        source,
        sample_rate,
        viterbi=True,
        step_size=10,
        verbose=0,
    )
    confidence = np.asarray(confidence, dtype=np.float32)
    f0 = np.where(confidence >= 0.25, f0, np.nan)
    return PitchTrack(
        times=np.asarray(times, dtype=np.float64),
        f0_hz=np.asarray(f0, dtype=np.float32),
        confidence=confidence,
        backend="crepe",
    )


def _fuse_pitch_tracks(
    tracks: list[PitchTrack],
    sample_rate: int,
    duration: float,
) -> PitchTrack:
    useful = [track for track in tracks if track.f0_hz.size and track.valid_ratio > 0.02]
    if not useful:
        return _empty_track("none")
    if len(useful) == 1:
        return useful[0]

    step = DEFAULT_FRAME_PERIOD_MS / 1000.0
    frame_count = max(1, int(math.ceil(max(duration, step) / step)))
    target_times = np.arange(frame_count, dtype=np.float64) * step
    midi_layers: list[np.ndarray] = []
    confidence_layers: list[np.ndarray] = []
    names: list[str] = []
    for track in useful:
        midi = _hz_to_midi_array(track.f0_hz)
        valid = np.isfinite(midi)
        if np.count_nonzero(valid) < 2:
            continue
        interpolated = np.interp(
            target_times,
            track.times[valid],
            midi[valid],
            left=np.nan,
            right=np.nan,
        )
        outside = (target_times < track.times[valid][0]) | (target_times > track.times[valid][-1])
        interpolated[outside] = np.nan
        midi_layers.append(interpolated.astype(np.float32))
        confidence_layers.append(
            np.interp(
                target_times,
                track.times,
                np.nan_to_num(track.confidence, nan=0.0),
                left=0.0,
                right=0.0,
            ).astype(np.float32)
        )
        names.append(track.backend)

    if not midi_layers:
        return useful[0]

    midi_stack = np.stack(midi_layers, axis=0)
    fused_midi = np.full(target_times.shape, np.nan, dtype=np.float32)
    for index in range(target_times.size):
        values = midi_stack[:, index]
        values = values[np.isfinite(values)]
        if values.size:
            fused_midi[index] = float(np.median(values))
    f0 = np.full(fused_midi.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(fused_midi)
    f0[valid] = (440.0 * (2.0 ** ((fused_midi[valid] - 69.0) / 12.0))).astype(np.float32)
    confidence = (
        np.nanmean(np.stack(confidence_layers, axis=0), axis=0).astype(np.float32)
        if confidence_layers
        else np.where(valid, 0.7, 0.0).astype(np.float32)
    )
    return PitchTrack(
        times=target_times,
        f0_hz=f0,
        confidence=confidence,
        backend="ensemble:" + "+".join(names),
    )


def _frames_from_track(pitch_track: PitchTrack) -> tuple[PitchFrame, ...]:
    frames: list[PitchFrame] = []
    for time_value, f0_value, confidence in zip(
        pitch_track.times,
        pitch_track.f0_hz,
        pitch_track.confidence,
        strict=False,
    ):
        f0 = float(f0_value) if np.isfinite(f0_value) and f0_value > 0 else None
        frames.append(
            PitchFrame(
                time=float(time_value),
                f0_hz=f0,
                midi=hz_to_midi_float(f0),
                confidence=float(confidence) if np.isfinite(confidence) else 0.0,
                backend=pitch_track.backend,
            )
        )
    return tuple(frames)


def _note_from_interval(
    source_path: str,
    index: int,
    start_time: float,
    end_time: float,
    pitch_track: PitchTrack,
) -> VocalNote:
    f0_hz, midi_note, confidence = summarize_pitch_track(
        pitch_track,
        start_time,
        end_time,
    )
    return VocalNote(
        source_path=source_path,
        index=index,
        start_time=start_time,
        end_time=end_time,
        midi_note=midi_note,
        f0_hz=f0_hz,
        confidence=confidence,
        backend=pitch_track.backend,
    )


def _suggest_internal_boundaries(
    source: np.ndarray,
    sample_rate: int,
    start_sample: int,
    end_sample: int,
    pitch_track: PitchTrack | None,
) -> list[int]:
    duration = (end_sample - start_sample) / max(1, sample_rate)
    if duration < 0.18 or duration > 12.0:
        return []
    minimum_gap = max(1, int(round(sample_rate * 0.075)))
    candidates: list[int] = []
    candidates.extend(
        _onset_boundary_candidates(
            source[start_sample:end_sample],
            sample_rate,
            minimum_gap,
            offset=start_sample,
        )
    )
    if pitch_track is not None and pitch_track.f0_hz.size:
        candidates.extend(
            _pitch_jump_boundary_candidates(
                pitch_track,
                sample_rate,
                start_sample,
                end_sample,
                minimum_gap,
            )
        )
    return _dedupe_and_refine_boundaries(
        source,
        sample_rate,
        candidates,
        start_sample,
        end_sample,
        minimum_gap,
    )


def _onset_boundary_candidates(
    segment: np.ndarray,
    sample_rate: int,
    minimum_gap: int,
    offset: int,
) -> list[int]:
    try:
        import librosa

        hop_length = max(64, min(256, int(sample_rate * 0.006)))
        onset_env = librosa.onset.onset_strength(
            y=segment,
            sr=sample_rate,
            hop_length=hop_length,
            aggregate=np.median,
        )
        if onset_env.size < 3 or float(np.max(onset_env)) <= 1e-8:
            return []
        wait_frames = max(1, int(round(minimum_gap / hop_length)))
        frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=hop_length,
            units="frames",
            backtrack=False,
            pre_max=max(1, wait_frames // 2),
            post_max=max(1, wait_frames // 2),
            pre_avg=max(2, wait_frames),
            post_avg=max(2, wait_frames),
            delta=0.06,
            wait=wait_frames,
        )
        return [
            offset + int(librosa.frames_to_samples(int(frame), hop_length=hop_length))
            for frame in frames
        ]
    except Exception:
        return []


def _pitch_jump_boundary_candidates(
    pitch_track: PitchTrack,
    sample_rate: int,
    start_sample: int,
    end_sample: int,
    minimum_gap: int,
) -> list[int]:
    start_time = start_sample / max(1, sample_rate)
    end_time = end_sample / max(1, sample_rate)
    midi = _hz_to_midi_array(pitch_track.f0_hz)
    mask = (
        (pitch_track.times >= start_time)
        & (pitch_track.times <= end_time)
        & np.isfinite(midi)
    )
    if np.count_nonzero(mask) < 5:
        return []

    times = pitch_track.times[mask]
    values = midi[mask].astype(np.float64)
    values = _median_smooth(values, kernel_size=5)
    diffs = np.abs(np.diff(values))
    if diffs.size == 0:
        return []
    threshold = max(1.35, float(np.percentile(diffs, 88)) * 1.4)
    raw = np.where(diffs >= threshold)[0] + 1
    candidates: list[int] = []
    last = start_sample - minimum_gap
    for index in raw:
        sample = int(round(times[int(index)] * sample_rate))
        if sample - last < minimum_gap:
            continue
        candidates.append(sample)
        last = sample
    return candidates


def _dedupe_and_refine_boundaries(
    source: np.ndarray,
    sample_rate: int,
    candidates: list[int],
    start_sample: int,
    end_sample: int,
    minimum_gap: int,
) -> list[int]:
    clean: list[int] = []
    for sample in sorted({int(value) for value in candidates}):
        if sample - start_sample < minimum_gap:
            continue
        if end_sample - sample < minimum_gap:
            continue
        refined = _refine_to_energy_valley(source, sample, sample_rate)
        if refined - start_sample < minimum_gap or end_sample - refined < minimum_gap:
            continue
        if clean and refined - clean[-1] < minimum_gap:
            previous = clean[-1]
            clean[-1] = _refine_to_energy_valley(
                source,
                int(round((previous + refined) / 2.0)),
                sample_rate,
            )
            continue
        clean.append(refined)
    return clean


def _refine_to_energy_valley(source: np.ndarray, sample: int, sample_rate: int) -> int:
    search_before = int(round(sample_rate * 0.035))
    search_after = int(round(sample_rate * 0.015))
    start = max(0, sample - search_before)
    end = min(source.size, sample + search_after)
    if end - start < 4:
        return max(0, min(source.size, sample))
    envelope = np.abs(source[start:end]).astype(np.float64)
    kernel_size = max(3, int(round(sample_rate * 0.004)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if envelope.size >= kernel_size:
        kernel = np.ones(kernel_size, dtype=np.float64) / float(kernel_size)
        envelope = np.convolve(envelope, kernel, mode="same")
    return int(start + int(np.argmin(envelope)))


def _make_contiguous_intervals(
    intervals: list[tuple[int, int]],
    audio_size: int,
) -> list[tuple[int, int]]:
    clean = [
        (max(0, int(start)), min(int(audio_size), int(end)))
        for start, end in sorted(intervals)
        if int(end) > int(start)
    ]
    if not clean:
        return []
    if len(clean) == 1:
        return [(0, int(audio_size))]
    boundaries = [0]
    for left, right in zip(clean, clean[1:], strict=False):
        boundary = int(round((left[1] + right[0]) / 2.0))
        boundary = max(boundaries[-1] + 1, min(int(audio_size) - 1, boundary))
        boundaries.append(boundary)
    boundaries.append(int(audio_size))
    return [
        (boundaries[index], boundaries[index + 1])
        for index in range(len(boundaries) - 1)
        if boundaries[index + 1] > boundaries[index]
    ]


def _robust_center_hz(values: np.ndarray, confidence: np.ndarray | None = None) -> float | None:
    source = np.asarray(values, dtype=np.float64)
    source = source[np.isfinite(source) & (source > 0.0)]
    if source.size == 0:
        return None
    midi = _hz_to_midi_array(source).astype(np.float64)
    midi = midi[np.isfinite(midi)]
    if midi.size == 0:
        return None
    lower, upper = np.percentile(midi, [12, 88])
    trimmed = midi[(midi >= lower) & (midi <= upper)]
    if trimmed.size == 0:
        trimmed = midi
    return midi_to_hz(float(np.median(trimmed)))


def _confidence_center(confidence: np.ndarray) -> float:
    values = np.asarray(confidence, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(max(0.0, min(1.0, np.median(values))))


def _median_smooth(values: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if source.size < kernel_size:
        return source
    if kernel_size % 2 == 0:
        kernel_size += 1
    try:
        from scipy.signal import medfilt

        return np.asarray(medfilt(source, kernel_size=kernel_size), dtype=np.float64)
    except Exception:
        radius = kernel_size // 2
        smoothed = source.copy()
        for index in range(source.size):
            left = max(0, index - radius)
            right = min(source.size, index + radius + 1)
            smoothed[index] = float(np.median(source[left:right]))
        return smoothed


def _as_mono_float32(audio: np.ndarray) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float32)
    if source.ndim == 2:
        source = np.mean(source, axis=1, dtype=np.float32)
    source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(source, dtype=np.float32)


def _as_world_float64(audio: np.ndarray) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float64)
    source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(source))) if source.size else 0.0
    if peak > 1.0:
        source = source / peak
    return np.ascontiguousarray(source, dtype=np.float64)


def _hz_to_midi_array(f0: np.ndarray) -> np.ndarray:
    source = np.asarray(f0, dtype=np.float64)
    midi = np.full(source.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(source) & (source > 0.0)
    midi[valid] = (69.0 + 12.0 * np.log2(source[valid] / 440.0)).astype(np.float32)
    return midi


def _compact_midi_contour(midi: np.ndarray, max_points: int) -> np.ndarray:
    source = np.asarray(midi, dtype=np.float32)
    if source.size == 0 or max_points <= 0:
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


def _empty_track(backend: str) -> PitchTrack:
    return PitchTrack(
        times=np.zeros(0, dtype=np.float64),
        f0_hz=np.zeros(0, dtype=np.float32),
        confidence=np.zeros(0, dtype=np.float32),
        backend=backend,
    )


def _float_env(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = float(default)
    return max(float(minimum), min(float(maximum), value))


def _rmvpe_should_skip(source: np.ndarray, sample_rate: int) -> bool:
    if sample_rate <= 0:
        return False
    max_seconds = _float_env(
        "HAKYKING_RMVPE_MAX_SECONDS",
        DEFAULT_RMVPE_MAX_SECONDS,
        minimum=0.5,
        maximum=120.0,
    )
    return float(source.size / sample_rate) > max_seconds
