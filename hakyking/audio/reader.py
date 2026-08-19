from __future__ import annotations

from collections import OrderedDict
import io
import json
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hakyking.runtime import which_executable
from hakyking.subprocess_utils import hidden_subprocess_kwargs


@dataclass(frozen=True)
class AudioInfo:
    path: str
    sample_rate: int
    duration: float
    channels: int


class AudioReader:
    """Reads audio metadata for standalone audio files and video files with audio."""

    audio_extensions = {
        ".aac",
        ".aif",
        ".aiff",
        ".flac",
        ".m4a",
        ".mp3",
        ".ogg",
        ".opus",
        ".wav",
        ".wma",
    }
    video_extensions = {".mkv", ".mov", ".mp4", ".webm"}
    supported_extensions = audio_extensions | video_extensions
    ffprobe_timeout_seconds = 8.0
    ffmpeg_timeout_seconds = 90.0

    # Slice analysis, waveform drawing and rendering often request different
    # regions from the same source file. Decoding the complete source once per
    # slice quickly saturates both CPU and disk, so keep a small shared LRU of
    # decoded mono sources. Entries are invalidated when file size/mtime changes.
    _mono_cache_limit_bytes = 384 * 1024 * 1024
    _mono_cache_max_entries = 8
    _mono_cache: OrderedDict[
        tuple[str, int, int], tuple[np.ndarray, int]
    ] = OrderedDict()
    _mono_cache_bytes = 0
    _mono_cache_lock = threading.RLock()
    _mono_load_locks: dict[str, threading.Lock] = {}

    @classmethod
    def read_info(cls, path: str) -> AudioInfo:
        media_path = Path(path).expanduser().resolve(strict=True)
        suffix = media_path.suffix.lower()
        if suffix not in cls.supported_extensions:
            raise ValueError(f"Unsupported media format: {suffix}")
        if suffix in cls.video_extensions:
            return cls._read_ffmpeg_audio_info(media_path)
        return cls._read_audio_info(media_path)

    @classmethod
    def load_mono(cls, path: str) -> tuple[np.ndarray, int]:
        media_path = Path(path).expanduser().resolve(strict=True)
        suffix = media_path.suffix.lower()
        if suffix not in cls.supported_extensions:
            raise ValueError(f"Unsupported media format: {suffix}")
        stat = media_path.stat()
        cache_key = (str(media_path), int(stat.st_mtime_ns), int(stat.st_size))

        cached = cls._cached_mono(cache_key)
        if cached is not None:
            return cached

        path_key = str(media_path)
        with cls._mono_cache_lock:
            load_lock = cls._mono_load_locks.setdefault(path_key, threading.Lock())

        # Only one worker decodes a particular file at a time. Other files can
        # still decode concurrently and callers re-check the cache after waiting.
        with load_lock:
            cached = cls._cached_mono(cache_key)
            if cached is not None:
                return cached
            if suffix in cls.video_extensions:
                audio, sample_rate = cls._load_ffmpeg_audio_mono(media_path)
            else:
                audio, sample_rate = cls._load_audio_mono(media_path)
            source = np.ascontiguousarray(audio, dtype=np.float32)
            cls._store_cached_mono(cache_key, source, int(sample_rate))
            return source, int(sample_rate)

    @classmethod
    def clear_mono_cache(cls) -> None:
        """Release decoded source buffers, primarily for project lifecycle/tests."""

        with cls._mono_cache_lock:
            cls._mono_cache.clear()
            cls._mono_cache_bytes = 0
            cls._mono_load_locks.clear()

    @classmethod
    def mono_cache_stats(cls) -> tuple[int, int]:
        with cls._mono_cache_lock:
            return len(cls._mono_cache), cls._mono_cache_bytes

    @classmethod
    def _cached_mono(
        cls,
        cache_key: tuple[str, int, int],
    ) -> tuple[np.ndarray, int] | None:
        with cls._mono_cache_lock:
            cached = cls._mono_cache.get(cache_key)
            if cached is None:
                return None
            cls._mono_cache.move_to_end(cache_key)
            return cached

    @classmethod
    def _store_cached_mono(
        cls,
        cache_key: tuple[str, int, int],
        audio: np.ndarray,
        sample_rate: int,
    ) -> None:
        with cls._mono_cache_lock:
            path_key = cache_key[0]
            stale_keys = [key for key in cls._mono_cache if key[0] == path_key]
            for stale_key in stale_keys:
                stale_audio, _ = cls._mono_cache.pop(stale_key)
                cls._mono_cache_bytes -= int(stale_audio.nbytes)

            cls._mono_cache[cache_key] = (audio, sample_rate)
            cls._mono_cache_bytes += int(audio.nbytes)
            cls._mono_cache.move_to_end(cache_key)
            while (
                len(cls._mono_cache) > cls._mono_cache_max_entries
                or cls._mono_cache_bytes > cls._mono_cache_limit_bytes
            ):
                _, (evicted_audio, _) = cls._mono_cache.popitem(last=False)
                cls._mono_cache_bytes -= int(evicted_audio.nbytes)

    @classmethod
    def has_audio_stream(cls, path: str) -> bool:
        media_path = Path(path).expanduser().resolve(strict=True)
        suffix = media_path.suffix.lower()
        if suffix not in cls.supported_extensions:
            return False
        if suffix not in cls.video_extensions:
            return True

        ffprobe_path = cls._find_ffprobe()
        command = [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(media_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                check=False,
                timeout=cls.ffprobe_timeout_seconds,
                **hidden_subprocess_kwargs(),
            )
        except subprocess.TimeoutExpired:
            return False
        if result.returncode != 0:
            return False
        return "audio" in result.stdout.decode("utf-8", errors="replace").lower()

    @classmethod
    def _read_audio_info(cls, path: Path) -> AudioInfo:
        import soundfile as sf

        try:
            info = sf.info(str(path))
            if info.samplerate > 0 and info.frames > 0:
                return AudioInfo(
                    path=str(path),
                    sample_rate=int(info.samplerate),
                    duration=float(info.frames / info.samplerate),
                    channels=int(info.channels),
                )
        except RuntimeError:
            pass

        import librosa

        audio, sample_rate = librosa.load(str(path), sr=None, mono=False)
        sample_count = audio.shape[-1]
        channels = 1 if audio.ndim == 1 else audio.shape[0]
        return AudioInfo(
            path=str(path),
            sample_rate=int(sample_rate),
            duration=float(sample_count / sample_rate),
            channels=int(channels),
        )

    @classmethod
    def _load_audio_mono(cls, path: Path) -> tuple[np.ndarray, int]:
        # libsndfile starts far faster than importing librosa and directly
        # handles the common WAV/FLAC/OGG/MP3 formats in this application.
        # Keep librosa as a compatibility fallback for uncommon codecs.
        try:
            import soundfile as sf

            audio, sample_rate = sf.read(
                str(path),
                dtype="float32",
                always_2d=True,
            )
            source = np.asarray(audio, dtype=np.float32)
            mono = (
                source[:, 0]
                if source.shape[1] == 1
                else np.mean(source, axis=1, dtype=np.float32)
            )
            return np.ascontiguousarray(mono, dtype=np.float32), int(sample_rate)
        except (OSError, RuntimeError, ValueError):
            import librosa

            audio, sample_rate = librosa.load(str(path), sr=None, mono=True)
            return np.asarray(audio, dtype=np.float32), int(sample_rate)

    @classmethod
    def _read_ffmpeg_audio_info(cls, path: Path) -> AudioInfo:
        ffprobe_path = cls._find_ffprobe()
        command = [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,duration:format=duration",
            "-of",
            "json",
            str(path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                timeout=cls.ffprobe_timeout_seconds,
                **hidden_subprocess_kwargs(),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Timed out probing media audio after {cls.ffprobe_timeout_seconds:.0f}s"
            ) from exc
        if result.returncode == 0 and result.stdout:
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                payload = {}
            streams = payload.get("streams", []) if isinstance(payload, dict) else []
            stream = streams[0] if streams and isinstance(streams[0], dict) else {}
            format_payload = payload.get("format", {}) if isinstance(payload, dict) else {}
            sample_rate = cls._optional_int(stream.get("sample_rate"))
            channels = cls._optional_int(stream.get("channels")) or 1
            duration = cls._optional_float(stream.get("duration"))
            if duration is None and isinstance(format_payload, dict):
                duration = cls._optional_float(format_payload.get("duration"))
            if sample_rate and duration and duration > 0:
                return AudioInfo(
                    path=str(path),
                    sample_rate=sample_rate,
                    duration=duration,
                    channels=max(1, channels),
                )

        return cls._read_ffmpeg_audio_info_by_extract(path)

    @classmethod
    def _read_ffmpeg_audio_info_by_extract(cls, path: Path) -> AudioInfo:
        import soundfile as sf

        with sf.SoundFile(io.BytesIO(cls._extract_media_wav_bytes(path))) as audio_file:
            sample_rate = int(audio_file.samplerate)
            frames = int(len(audio_file))
            channels = int(audio_file.channels)

        return AudioInfo(
            path=str(path),
            sample_rate=sample_rate,
            duration=float(frames / sample_rate),
            channels=channels,
        )

    @staticmethod
    def _optional_int(value: object) -> int | None:
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if number > 0 else None

    @staticmethod
    def _optional_float(value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if number > 0 else None

    @classmethod
    def _load_ffmpeg_audio_mono(cls, path: Path) -> tuple[np.ndarray, int]:
        import soundfile as sf

        wav_data = cls._extract_media_wav_bytes(path)
        audio, sample_rate = sf.read(io.BytesIO(wav_data), dtype="float32", always_2d=False)
        audio_array = np.asarray(audio, dtype=np.float32)
        if audio_array.ndim == 2:
            audio_array = np.mean(audio_array, axis=1, dtype=np.float32)
        return audio_array, int(sample_rate)

    @classmethod
    def _extract_media_wav_bytes(cls, path: Path) -> bytes:
        ffmpeg_path = cls._find_ffmpeg()
        command = [
            ffmpeg_path,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-f",
            "wav",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                check=False,
                timeout=cls.ffmpeg_timeout_seconds,
                **hidden_subprocess_kwargs(),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Timed out extracting media audio after {cls.ffmpeg_timeout_seconds:.0f}s"
            ) from exc
        if result.returncode != 0 or not result.stdout:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"Unable to extract media audio track: {error}")
        return result.stdout

    @staticmethod
    def _find_ffmpeg() -> str:
        ffmpeg_path = which_executable("ffmpeg")
        if ffmpeg_path:
            return ffmpeg_path
        raise RuntimeError(
            "ffmpeg executable was not found. Install FFmpeg and add it to PATH "
            "to read MP4 audio tracks."
        )

    @staticmethod
    def _find_ffprobe() -> str:
        ffprobe_path = which_executable("ffprobe")
        if ffprobe_path:
            return ffprobe_path
        raise RuntimeError(
            "ffprobe executable was not found. Install FFmpeg and add it to PATH "
            "to filter MP4 files by audio stream."
        )
