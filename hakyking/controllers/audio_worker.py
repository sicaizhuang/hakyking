from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import numpy as np

from hakyking.audio.audio_engine import render_slice_from_file
from hakyking.audio.analysis_cache import load_cached_slices, store_cached_slices
from hakyking.audio.exporter import ExportClip, export_mixdown_to_wav
from hakyking.audio.reader import AudioReader
from hakyking.audio.slicer import build_full_audio_slice, parse_audio_slices
from hakyking.audio.waveform import build_waveform_result, load_slice_audio
from hakyking.models.audio_slice import AudioSlice
from hakyking.qt import QObject, Signal, Slot


LOGGER = logging.getLogger(__name__)


class AudioWorker(QObject):
    """
    Placeholder worker for future audio loading and analysis.

    Non-blocking rule:
        Move this worker to a QThread before calling any method that touches
        disk, decodes audio, computes waveform previews, detects pitch, slices
        material, renders, or exports. The main UI thread must remain responsive.
    """

    progress_changed = Signal(float)
    finished = Signal(object)
    failed = Signal(str)

    @Slot(str)
    def load_audio(self, path: str) -> None:
        self.failed.emit(
            f"Audio loading is not implemented yet. Requested path: {path}"
        )


class AudioProbeWorker(QObject):
    """Runs media probing in a worker thread so the UI stays responsive."""

    finished = Signal(object)
    failed = Signal(str)
    completed = Signal()

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    @Slot()
    def run(self) -> None:
        try:
            self.finished.emit(AudioReader.read_info(self.path))
        except Exception as exc:  # noqa: BLE001 - worker reports user-facing errors
            self.failed.emit(f"{self.path}: {exc}")
        finally:
            self.completed.emit()


class FolderMediaScanWorker(QObject):
    """Scans an imported material folder off the UI thread."""

    finished = Signal(str, object)
    failed = Signal(str)
    completed = Signal()

    def __init__(self, folder_path: str, recursive: bool = True) -> None:
        super().__init__()
        self.folder_path = folder_path
        self.recursive = recursive

    @Slot()
    def run(self) -> None:
        try:
            root = Path(self.folder_path).expanduser().resolve(strict=True)
            if not root.is_dir():
                raise NotADirectoryError(str(root))
            supported = AudioReader.supported_extensions
            candidates = root.rglob("*") if self.recursive else root.iterdir()
            paths = [
                str(path)
                for path in candidates
                if path.is_file() and path.suffix.lower() in supported
            ]
            self.finished.emit(self.folder_path, sorted(paths, key=lambda value: value.lower()))
        except Exception as exc:  # noqa: BLE001 - worker reports user-facing errors
            self.failed.emit(f"{self.folder_path}: {exc}")
        finally:
            self.completed.emit()


class ParseWorker(QObject):
    """Runs librosa slicing and pitch analysis in a background thread."""

    finished = Signal(str, object)
    skipped = Signal(str, str)
    failed = Signal(str)
    completed = Signal()

    def __init__(
        self,
        path: str,
        max_duration_seconds: float | None = None,
        skip_video: bool = False,
    ) -> None:
        super().__init__()
        self.path = path
        self.max_duration_seconds = max_duration_seconds
        self.skip_video = skip_video
        self._cancel_requested = False

    def cancel(self) -> None:
        """Request cooperative cancellation at the next analysis checkpoint."""

        self._cancel_requested = True

    def is_cancelled(self) -> bool:
        return self._cancel_requested

    @Slot()
    def run(self) -> None:
        started_at = perf_counter()
        try:
            if self.is_cancelled():
                return
            suffix = Path(self.path).suffix.lower()
            if self.skip_video and suffix in AudioReader.video_extensions:
                self.skipped.emit(self.path, "video files are parsed on demand")
                return
            if self.max_duration_seconds is not None:
                info = AudioReader.read_info(self.path)
                if self.is_cancelled():
                    return
                if info.duration > self.max_duration_seconds:
                    self.skipped.emit(
                        self.path,
                        f"duration {info.duration:.1f}s > {self.max_duration_seconds:.1f}s",
                    )
                    return
            slices = load_cached_slices(self.path)
            cache_hit = slices is not None
            if slices is None:
                slices = parse_audio_slices(self.path, cancel_check=self.is_cancelled)
            if self.is_cancelled():
                return
            if not slices:
                slices = [build_full_audio_slice(self.path)]
            if not cache_hit:
                try:
                    if self.is_cancelled():
                        return
                    store_cached_slices(self.path, slices)
                except Exception:  # noqa: BLE001 - cache failure must not fail parsing
                    LOGGER.exception("Could not cache slice analysis for %s", self.path)
            LOGGER.info(
                "Slice analysis %s in %.3fs: %s (%d slices)",
                "cache hit" if cache_hit else "completed",
                perf_counter() - started_at,
                self.path,
                len(slices),
            )
            if not self.is_cancelled():
                self.finished.emit(self.path, slices)
        except Exception as exc:  # noqa: BLE001 - worker reports user-facing errors
            LOGGER.exception("Slice parsing failed for %s", self.path)
            self.failed.emit(f"{self.path}: {exc}")
        finally:
            self.completed.emit()


class RenderWorker(QObject):
    """Runs non-destructive HPSS/Rubber Band rendering in a background thread."""

    finished = Signal(object)
    failed = Signal(str)
    completed = Signal()

    def __init__(
        self,
        cache_key: str,
        audio_slice: AudioSlice,
        target_midi_note: int | None,
        target_duration: float | None,
        gain_db: float = 0.0,
        pitch_flatten_amount: float = 0.0,
        formant_shift: float = 0.0,
        protect_transients: bool = True,
        pitch_control_points: object = (),
        pitch_vibrato_regions: object = (),
    ) -> None:
        super().__init__()
        self.cache_key = cache_key
        self.audio_slice = audio_slice
        self.target_midi_note = target_midi_note
        self.target_duration = target_duration
        self.gain_db = gain_db
        self.pitch_flatten_amount = pitch_flatten_amount
        self.formant_shift = formant_shift
        self.protect_transients = bool(protect_transients)
        self.pitch_control_points = pitch_control_points
        self.pitch_vibrato_regions = pitch_vibrato_regions

    @Slot()
    def run(self) -> None:
        try:
            self.finished.emit(
                render_slice_from_file(
                    audio_slice=self.audio_slice,
                    target_midi_note=self.target_midi_note,
                    target_duration=self.target_duration,
                    cache_key=self.cache_key,
                    gain_db=self.gain_db,
                    pitch_flatten_amount=self.pitch_flatten_amount,
                    formant_shift=self.formant_shift,
                    protect_transients=self.protect_transients,
                    pitch_control_points=self.pitch_control_points,
                    pitch_vibrato_regions=self.pitch_vibrato_regions,
                )
            )
        except Exception as exc:  # noqa: BLE001 - worker reports user-facing errors
            self.failed.emit(f"{self.cache_key}: {exc}")
        finally:
            self.completed.emit()


class WaveformWorker(QObject):
    """Loads a slice and computes its compact waveform envelope off the UI thread."""

    finished = Signal(object)
    failed = Signal(str)
    completed = Signal()

    def __init__(
        self,
        cache_key: str,
        audio_slice: AudioSlice,
        max_points: int = 256,
        prefer_cached_pitch: bool = True,
    ) -> None:
        super().__init__()
        self.cache_key = cache_key
        self.audio_slice = audio_slice
        self.max_points = max_points
        self.prefer_cached_pitch = bool(prefer_cached_pitch)

    @Slot()
    def run(self) -> None:
        try:
            self.finished.emit(
                build_waveform_result(
                    cache_key=self.cache_key,
                    audio_slice=self.audio_slice,
                    max_points=self.max_points,
                    prefer_cached_pitch=self.prefer_cached_pitch,
                )
            )
        except Exception as exc:  # noqa: BLE001 - worker reports user-facing errors
            self.failed.emit(f"{self.cache_key}: {exc}")
        finally:
            self.completed.emit()


class SlicePreviewWorker(QObject):
    """Loads one material-browser slice for non-blocking preview playback."""

    finished = Signal(object, int)
    failed = Signal(str)
    completed = Signal()

    def __init__(self, audio_slice: AudioSlice) -> None:
        super().__init__()
        self.audio_slice = audio_slice

    @Slot()
    def run(self) -> None:
        try:
            audio, sample_rate = load_slice_audio(self.audio_slice)
            self.finished.emit(audio, sample_rate)
        except Exception as exc:  # noqa: BLE001 - worker reports user-facing errors
            self.failed.emit(f"{self.audio_slice.source_path}: {exc}")
        finally:
            self.completed.emit()


def _normalize_contiguous_slice_selection(audio_slices: list[AudioSlice]) -> list[AudioSlice]:
    slices = sorted(audio_slices, key=lambda item: (item.source_path, item.start_time, item.end_time, item.index))
    if not slices:
        raise ValueError("No selected material slices.")

    source_paths = {
        str(Path(audio_slice.source_path).expanduser().resolve())
        for audio_slice in slices
    }
    if len(source_paths) != 1:
        raise ValueError("Selected material slices must come from the same source file.")

    for left, right in zip(slices, slices[1:]):
        if right.index != left.index + 1:
            raise ValueError("Selected material slices must be adjacent.")
        boundary_gap = abs(float(right.start_time) - float(left.end_time))
        if boundary_gap > 0.025:
            raise ValueError("Selected material slices are not time-contiguous.")
    return slices


class SliceSequencePreviewWorker(QObject):
    """Loads a contiguous material-browser slice selection for preview playback."""

    finished = Signal(str, object, int, float, float, int)
    failed = Signal(str)
    completed = Signal()

    def __init__(self, audio_slices: list[AudioSlice]) -> None:
        super().__init__()
        self.audio_slices = list(audio_slices)

    @Slot()
    def run(self) -> None:
        try:
            slices = _normalize_contiguous_slice_selection(self.audio_slices)
            source_path = slices[0].source_path
            source_audio, sample_rate = AudioReader.load_mono(source_path)
            start_time = max(0.0, float(slices[0].start_time))
            end_time = max(start_time, float(slices[-1].end_time))
            start_sample = max(0, min(source_audio.shape[0], int(round(start_time * sample_rate))))
            end_sample = max(start_sample, min(source_audio.shape[0], int(round(end_time * sample_rate))))
            audio = np.asarray(source_audio[start_sample:end_sample], dtype=np.float32).copy()
            self.finished.emit(source_path, audio, sample_rate, start_time, end_time, len(slices))
        except Exception as exc:  # noqa: BLE001 - worker reports user-facing errors
            self.failed.emit(str(exc))
        finally:
            self.completed.emit()


class FilePreviewWorker(QObject):
    """Loads a full material file for non-blocking browser preview playback."""

    finished = Signal(str, object, int, float, float)
    failed = Signal(str)
    completed = Signal()

    def __init__(self, path: str, start_time: float = 0.0) -> None:
        super().__init__()
        self.path = path
        self.start_time = max(0.0, float(start_time))

    @Slot()
    def run(self) -> None:
        try:
            audio, sample_rate = AudioReader.load_mono(self.path)
            duration = float(audio.shape[0] / sample_rate) if sample_rate > 0 else 0.0
            start_sample = max(0, min(audio.shape[0], int(round(self.start_time * sample_rate))))
            self.finished.emit(
                self.path,
                audio[start_sample:].copy(),
                sample_rate,
                min(self.start_time, duration),
                duration,
            )
        except Exception as exc:  # noqa: BLE001 - worker reports user-facing errors
            self.failed.emit(f"{self.path}: {exc}")
        finally:
            self.completed.emit()


class WholeSliceWorker(QObject):
    """Creates one full-length editable slice for manual slicing workflows."""

    finished = Signal(str, object)
    failed = Signal(str)
    completed = Signal()

    def __init__(self, path: str, index: int = 0) -> None:
        super().__init__()
        self.path = path
        self.index = index

    @Slot()
    def run(self) -> None:
        try:
            self.finished.emit(self.path, build_full_audio_slice(self.path, index=self.index))
        except Exception as exc:  # noqa: BLE001 - worker reports user-facing errors
            self.failed.emit(f"{self.path}: {exc}")
        finally:
            self.completed.emit()


class ExportWorker(QObject):
    """Offline mixdown worker that writes a 44.1 kHz / 16-bit WAV file."""

    finished = Signal(object)
    failed = Signal(str)
    completed = Signal()

    def __init__(
        self,
        clips: list[ExportClip],
        output_path: str,
        sample_rate: int = 44100,
        fade_ms: float = 5.0,
    ) -> None:
        super().__init__()
        self.clips = clips
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.fade_ms = float(fade_ms)

    @Slot()
    def run(self) -> None:
        try:
            self.finished.emit(
                export_mixdown_to_wav(
                    clips=self.clips,
                    output_path=self.output_path,
                    sample_rate=self.sample_rate,
                    fade_ms=self.fade_ms,
                )
            )
        except Exception as exc:  # noqa: BLE001 - worker reports user-facing errors
            self.failed.emit(f"{self.output_path}: {exc}")
        finally:
            self.completed.emit()
