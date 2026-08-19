from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from hakyking.audio.gain import soft_limit_audio


@dataclass(frozen=True)
class TimelineClip:
    start_time: float
    audio: np.ndarray
    sample_rate: int
    track_index: int


@dataclass(frozen=True)
class AudioOutputDevice:
    index: int | None
    name: str
    hostapi: str
    max_output_channels: int
    default_samplerate: float


class PlaybackManager:
    """Non-blocking preview and timeline playback backed by sounddevice."""

    def __init__(
        self,
        fade_ms: float = 5.0,
        output_device_index: int | None = None,
        blocksize: int = 1024,
    ) -> None:
        self.fade_ms = fade_ms
        self.output_device_index = output_device_index
        self.blocksize = blocksize
        self._stream = None
        self._control_lock = threading.RLock()
        self._timeline_lock = threading.Lock()
        self._timeline_playing = False
        self._timeline_current_time = 0.0
        self._timeline_end_time = 0.0

    def configure(
        self,
        output_device_index: int | None = None,
        blocksize: int | None = None,
        fade_ms: float | None = None,
    ) -> None:
        """Apply playback settings used by future preview/timeline starts."""

        with self._control_lock:
            self.output_device_index = output_device_index
            if blocksize is not None:
                self.blocksize = max(128, min(8192, int(blocksize)))
            if fade_ms is not None:
                self.fade_ms = max(0.0, min(50.0, float(fade_ms)))

    def settings_summary(self) -> str:
        device = "System Default" if self.output_device_index is None else str(self.output_device_index)
        return f"device={device}, blocksize={self.blocksize}, fade={self.fade_ms:.1f}ms"

    @staticmethod
    def available_output_devices() -> list[AudioOutputDevice]:
        import sounddevice as sd

        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        output_devices: list[AudioOutputDevice] = [
            AudioOutputDevice(
                index=None,
                name="System Default",
                hostapi="",
                max_output_channels=0,
                default_samplerate=0.0,
            )
        ]
        for index, device in enumerate(devices):
            channels = int(device.get("max_output_channels", 0))
            if channels <= 0:
                continue
            hostapi_index = int(device.get("hostapi", -1))
            hostapi_name = ""
            if 0 <= hostapi_index < len(hostapis):
                hostapi_name = str(hostapis[hostapi_index].get("name", ""))
            output_devices.append(
                AudioOutputDevice(
                    index=index,
                    name=str(device.get("name", f"Device {index}")),
                    hostapi=hostapi_name,
                    max_output_channels=channels,
                    default_samplerate=float(device.get("default_samplerate", 0.0)),
                )
            )
        return output_devices

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        import sounddevice as sd

        with self._control_lock:
            self.stop_timeline()
            prepared = prepare_playback_audio(audio, sample_rate, fade_ms=self.fade_ms)
            if prepared.size == 0:
                return
            sd.stop()
            sd.play(
                prepared,
                sample_rate,
                device=self.output_device_index,
                blocking=False,
            )

    def stop(self) -> None:
        import sounddevice as sd

        with self._control_lock:
            self.stop_timeline()
            sd.stop()

    def play_timeline(
        self,
        clips: list[TimelineClip],
        start_time: float,
        sample_rate: int | None = None,
        blocksize: int | None = None,
        timeline_end_time: float | None = None,
    ) -> None:
        import sounddevice as sd

        with self._control_lock:
            self.stop_timeline()
            prepared_clips, master_rate, end_time = self._prepare_timeline_clips(
                clips,
                sample_rate=sample_rate,
            )
            if timeline_end_time is not None:
                end_time = max(end_time, float(timeline_end_time))
            if not prepared_clips and end_time <= start_time:
                return

            current_frame = max(0, int(round(start_time * master_rate)))
            end_frame = max(current_frame, int(round(end_time * master_rate)))
            if current_frame >= end_frame:
                return

            with self._timeline_lock:
                self._timeline_playing = True
                self._timeline_current_time = current_frame / master_rate
                self._timeline_end_time = end_frame / master_rate

            def callback(outdata, frames, time_info, status):  # noqa: ANN001
                nonlocal current_frame
                block = self._mix_timeline_block(
                    prepared_clips,
                    block_start_frame=current_frame,
                    frames=frames,
                    sample_rate=master_rate,
                )
                outdata[:, 0] = block
                current_frame += frames
                with self._timeline_lock:
                    self._timeline_current_time = min(
                        current_frame / master_rate,
                        self._timeline_end_time,
                    )
                if current_frame >= end_frame:
                    with self._timeline_lock:
                        self._timeline_playing = False
                    raise sd.CallbackStop

            self._stream = sd.OutputStream(
                channels=1,
                samplerate=master_rate,
                blocksize=blocksize or self.blocksize,
                dtype="float32",
                device=self.output_device_index,
                callback=callback,
            )
            self._stream.start()

    def stop_timeline(self) -> None:
        with self._control_lock:
            stream = self._stream
            self._stream = None
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
            with self._timeline_lock:
                self._timeline_playing = False

    @property
    def is_timeline_playing(self) -> bool:
        with self._timeline_lock:
            return self._timeline_playing

    @property
    def timeline_current_time(self) -> float:
        with self._timeline_lock:
            return self._timeline_current_time

    def _prepare_timeline_clips(
        self,
        clips: list[TimelineClip],
        sample_rate: int | None,
    ) -> tuple[list[TimelineClip], int, float]:
        if not clips:
            return [], 44100 if sample_rate is None else sample_rate, 0.0

        master_rate = sample_rate or clips[0].sample_rate
        prepared: list[TimelineClip] = []
        end_time = 0.0
        for clip in clips:
            audio = prepare_playback_audio(
                clip.audio,
                clip.sample_rate,
                fade_ms=self.fade_ms,
                limit_output=False,
            )
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1, dtype=np.float32)
            if clip.sample_rate != master_rate:
                audio = _resample_audio(audio, clip.sample_rate, master_rate)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.size == 0:
                continue
            prepared_clip = TimelineClip(
                start_time=max(0.0, clip.start_time),
                audio=audio,
                sample_rate=master_rate,
                track_index=clip.track_index,
            )
            prepared.append(prepared_clip)
            end_time = max(end_time, prepared_clip.start_time + audio.shape[0] / master_rate)
        return prepared, master_rate, end_time

    def _mix_timeline_block(
        self,
        clips: list[TimelineClip],
        block_start_frame: int,
        frames: int,
        sample_rate: int,
    ) -> np.ndarray:
        block_end_frame = block_start_frame + frames
        mixed = np.zeros(frames, dtype=np.float32)

        for clip in clips:
            clip_start_frame = int(round(clip.start_time * sample_rate))
            clip_end_frame = clip_start_frame + clip.audio.shape[0]
            if clip_end_frame <= block_start_frame or clip_start_frame >= block_end_frame:
                continue

            dest_start = max(0, clip_start_frame - block_start_frame)
            source_start = max(0, block_start_frame - clip_start_frame)
            length = min(frames - dest_start, clip.audio.shape[0] - source_start)
            if length <= 0:
                continue
            mixed[dest_start : dest_start + length] += clip.audio[source_start : source_start + length]

        return soft_limit_audio(mixed).astype(np.float32)


def prepare_playback_audio(
    audio: np.ndarray,
    sample_rate: int,
    fade_ms: float = 5.0,
    limit_output: bool = True,
) -> np.ndarray:
    """Return a float32 playback buffer with short anti-click fades."""

    source = np.asarray(audio, dtype=np.float32)
    if source.size == 0:
        return source.copy()
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    output = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0).copy()
    _apply_edge_fade_in_place(output, sample_rate, fade_ms=fade_ms)
    if limit_output:
        output = soft_limit_audio(output)
    return np.asarray(output, dtype=np.float32)


def mix_playback_buffers(buffers: list[np.ndarray]) -> np.ndarray:
    """Mix several preview buffers that already share a sample rate."""

    if not buffers:
        return np.zeros(0, dtype=np.float32)
    max_len = max(buffer.shape[0] for buffer in buffers)
    if max_len == 0:
        return np.zeros(0, dtype=np.float32)

    mixed = np.zeros(max_len, dtype=np.float32)
    for buffer in buffers:
        mono = np.asarray(buffer, dtype=np.float32)
        if mono.ndim == 2:
            mono = np.mean(mono, axis=1, dtype=np.float32)
        mixed[: mono.shape[0]] += mono
    return soft_limit_audio(mixed).astype(np.float32)


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    import librosa

    return np.asarray(
        librosa.resample(
            np.asarray(audio, dtype=np.float32),
            orig_sr=source_rate,
            target_sr=target_rate,
        ),
        dtype=np.float32,
    )


def _apply_edge_fade_in_place(audio: np.ndarray, sample_rate: int, fade_ms: float) -> None:
    fade_samples = int(round(sample_rate * fade_ms / 1000.0))
    if fade_samples <= 0:
        return
    fade_samples = min(fade_samples, audio.shape[0] // 2)
    if fade_samples <= 0:
        return

    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    if audio.ndim == 1:
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
    else:
        audio[:fade_samples, :] *= fade_in[:, None]
        audio[-fade_samples:, :] *= fade_out[:, None]


def _normalize_if_needed(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        return np.asarray(audio / peak, dtype=np.float32)
    return np.asarray(audio, dtype=np.float32)
