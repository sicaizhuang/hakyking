from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from hakyking.audio.reader import AudioReader
from hakyking.models.audio_slice import AudioSlice


SYLLABLE_BOUNDARY_LEAD_SECONDS = 0.012
SYLLABLE_ONSET_MIN_STRENGTH = 0.16
SYLLABLE_NUCLEUS_MIN_PROMINENCE_DB = 1.5
SYLLABLE_NUCLEUS_MIN_GAP_SECONDS = 0.08
FULL_SLICE_PITCH_SCAN_SECONDS = 30.0
LONG_AUDIO_FAST_THRESHOLD_SECONDS = 30.0
LONG_AUDIO_ANALYSIS_SAMPLE_RATE = 16000


def parse_audio_slices(
    path: str,
    top_db: int = 35,
    cancel_check: Callable[[], bool] | None = None,
) -> list[AudioSlice]:
    """
    Split media into voiced slices and estimate pitch with librosa.

    The heavy work belongs in a QThread worker. This function intentionally
    contains no UI code so it can be tested independently.
    """

    import librosa

    def cancelled() -> bool:
        return bool(cancel_check is not None and cancel_check())

    if cancelled():
        return []
    source_audio, source_sample_rate = AudioReader.load_mono(path)
    if source_audio.size == 0:
        return []
    if cancelled():
        return []

    source_duration = float(source_audio.size / source_sample_rate)
    use_fast_long_path = source_duration >= LONG_AUDIO_FAST_THRESHOLD_SECONDS
    audio, sample_rate = (
        _resample_analysis_audio(
            source_audio,
            source_sample_rate,
            LONG_AUDIO_ANALYSIS_SAMPLE_RATE,
        )
        if use_fast_long_path
        else (source_audio, source_sample_rate)
    )
    analysis_duration = float(audio.size / sample_rate)
    time_scale = source_duration / analysis_duration if analysis_duration > 0 else 1.0

    intervals = librosa.effects.split(audio, top_db=top_db)
    if intervals.size == 0:
        return []
    if cancelled():
        return []

    min_samples = max(1, int(sample_rate * 0.03))
    if use_fast_long_path:
        syllable_intervals = _subdivide_voiced_intervals_fast(
            audio=audio,
            sample_rate=sample_rate,
            intervals=intervals,
        )
    else:
        syllable_intervals: list[tuple[int, int]] = []
        for start_sample, end_sample in intervals:
            if cancelled():
                return []
            if end_sample <= start_sample or (end_sample - start_sample) < min_samples:
                continue
            syllable_intervals.extend(
                _subdivide_voiced_interval(
                    audio=audio,
                    sample_rate=sample_rate,
                    start_sample=int(start_sample),
                    end_sample=int(end_sample),
                )
            )

    slices: list[AudioSlice] = []
    merged_intervals = _merge_tiny_intervals(
        syllable_intervals,
        sample_rate,
    )
    contiguous_intervals = _make_contiguous_slice_intervals(
        merged_intervals,
        audio_size=audio.size,
    )

    for start_sample, end_sample in contiguous_intervals:
        if cancelled():
            return []
        slices.append(
            AudioSlice(
                source_path=path,
                index=len(slices),
                start_time=float(start_sample / sample_rate) * time_scale,
                end_time=min(
                    source_duration,
                    float(end_sample / sample_rate) * time_scale,
                ),
                midi_note=None,
                f0_hz=None,
            )
        )
    try:
        if cancelled():
            return []
        if use_fast_long_path:
            from hakyking.audio.vocal_analysis import (
                PitchTrack,
                analyze_fast_pitch_audio,
                annotate_slices_with_pitch_track,
            )

            pitch_track = analyze_fast_pitch_audio(audio, sample_rate)
            if not math.isclose(time_scale, 1.0, rel_tol=0.0, abs_tol=1e-9):
                pitch_track = PitchTrack(
                    times=np.asarray(pitch_track.times, dtype=np.float64) * time_scale,
                    f0_hz=pitch_track.f0_hz,
                    confidence=pitch_track.confidence,
                    backend=pitch_track.backend,
                )
            slices = annotate_slices_with_pitch_track(path, slices, pitch_track)
        else:
            from hakyking.audio.vocal_analysis import annotate_slices_with_vocal_analysis

            slices = annotate_slices_with_vocal_analysis(
                path,
                audio,
                sample_rate,
                slices,
            )
    except Exception:
        pass
    if cancelled():
        return []
    if not use_fast_long_path and any(item.midi_note is None for item in slices):
        slices = _fill_missing_slice_pitch(slices, audio, sample_rate)
    return slices


def _resample_analysis_audio(
    audio: np.ndarray,
    sample_rate: int,
    target_sample_rate: int,
) -> tuple[np.ndarray, int]:
    if sample_rate <= target_sample_rate:
        return np.asarray(audio, dtype=np.float32), int(sample_rate)
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(int(sample_rate), int(target_sample_rate))
    resampled = resample_poly(
        np.asarray(audio, dtype=np.float32),
        int(target_sample_rate) // divisor,
        int(sample_rate) // divisor,
    )
    return np.ascontiguousarray(resampled, dtype=np.float32), int(target_sample_rate)


def _fill_missing_slice_pitch(
    slices: list[AudioSlice],
    audio: np.ndarray,
    sample_rate: int,
) -> list[AudioSlice]:
    """Rare fallback used only when the source-level analyzer found no pitch."""

    filled: list[AudioSlice] = []
    for audio_slice in slices:
        if audio_slice.midi_note is not None:
            filled.append(audio_slice)
            continue
        start_sample = max(0, int(round(audio_slice.start_time * sample_rate)))
        end_sample = min(audio.size, int(round(audio_slice.end_time * sample_rate)))
        segment = audio[start_sample:end_sample]
        average_f0_hz = _estimate_average_f0(segment, sample_rate)
        start_f0_hz = _estimate_initial_f0(segment, sample_rate)
        f0_hz = start_f0_hz if start_f0_hz is not None else average_f0_hz
        filled.append(
            AudioSlice(
                source_path=audio_slice.source_path,
                index=audio_slice.index,
                start_time=audio_slice.start_time,
                end_time=audio_slice.end_time,
                midi_note=_hz_to_midi(f0_hz),
                f0_hz=f0_hz,
                pitch_confidence=audio_slice.pitch_confidence,
                analysis_backend=audio_slice.analysis_backend,
            )
        )
    return filled


def build_full_audio_slice(
    path: str,
    index: int = 0,
    pitch_scan_seconds: float = FULL_SLICE_PITCH_SCAN_SECONDS,
) -> AudioSlice:
    """Create one editable material block for manual slicing workflows.

    This keeps automatic syllable slicing optional: the source is still analyzed
    for an average pitch, but no internal boundaries are generated.
    """

    audio, sample_rate = AudioReader.load_mono(path)
    duration = float(audio.size / sample_rate) if sample_rate > 0 else 0.0
    if audio.size == 0 or sample_rate <= 0:
        return AudioSlice(
            source_path=path,
            index=index,
            start_time=0.0,
            end_time=0.001,
            midi_note=None,
            f0_hz=None,
        )

    max_samples = max(1, int(round(max(0.1, pitch_scan_seconds) * sample_rate)))
    pitch_source = audio[: min(audio.size, max_samples)]
    f0_hz = _estimate_average_f0(pitch_source, sample_rate)
    midi_note = _hz_to_midi(f0_hz) if f0_hz is not None else None
    full_slice = AudioSlice(
        source_path=path,
        index=index,
        start_time=0.0,
        end_time=max(0.001, duration),
        midi_note=midi_note,
        f0_hz=f0_hz,
    )
    try:
        from hakyking.audio.vocal_analysis import annotate_slices_with_vocal_analysis

        return annotate_slices_with_vocal_analysis(
            path,
            pitch_source,
            sample_rate,
            [full_slice],
        )[0]
    except Exception:
        return full_slice


def _subdivide_voiced_interval(
    audio: np.ndarray,
    sample_rate: int,
    start_sample: int,
    end_sample: int,
) -> list[tuple[int, int]]:
    """Split a continuous voiced region into likely syllable-sized chunks.

    `librosa.effects.split` only notices real silence. For connected phrases,
    use open-source onset detectors first and keep local boundary cleanup small.
    """

    interval_samples = max(0, end_sample - start_sample)
    minimum_syllable_samples = max(1, int(sample_rate * 0.07))
    if interval_samples < minimum_syllable_samples * 3:
        return [(start_sample, end_sample)]

    segment = np.asarray(audio[start_sample:end_sample], dtype=np.float32)
    if not np.any(np.abs(segment) > 1e-5):
        return [(start_sample, end_sample)]

    try:
        split_points = _detect_internal_onsets(
            segment=segment,
            sample_rate=sample_rate,
            minimum_syllable_samples=minimum_syllable_samples,
        )
    except Exception:
        split_points = []

    return _intervals_from_split_points(
        start_sample=start_sample,
        end_sample=end_sample,
        sample_rate=sample_rate,
        split_points=split_points,
        minimum_syllable_samples=minimum_syllable_samples,
    )


def _subdivide_voiced_intervals_fast(
    audio: np.ndarray,
    sample_rate: int,
    intervals: np.ndarray,
) -> list[tuple[int, int]]:
    """Subdivide long media with one shared onset transform.

    The former path recomputed an STFT/onset envelope for every voiced region.
    A song can contain hundreds of regions, so that repeated setup dominated
    import time. This path detects candidates once and then refines them inside
    each voiced interval.
    """

    minimum_syllable_samples = max(1, int(sample_rate * 0.07))
    global_onsets = _detect_global_onset_samples(
        audio,
        sample_rate,
        minimum_syllable_samples,
    )
    subdivisions: list[tuple[int, int]] = []
    for raw_start, raw_end in intervals:
        start_sample = int(raw_start)
        end_sample = int(raw_end)
        interval_samples = end_sample - start_sample
        if interval_samples < minimum_syllable_samples:
            continue
        if interval_samples < minimum_syllable_samples * 3:
            subdivisions.append((start_sample, end_sample))
            continue

        split_points = [
            point - start_sample
            for point in global_onsets
            if start_sample < point < end_sample
        ]
        segment = np.asarray(audio[start_sample:end_sample], dtype=np.float32)
        fallback_points = _detect_regular_valley_splits(
            segment=segment,
            sample_rate=sample_rate,
            minimum_syllable_samples=minimum_syllable_samples,
        )
        if len(split_points) < len(fallback_points):
            split_points = _dedupe_split_points(
                [*split_points, *fallback_points],
                minimum_gap=minimum_syllable_samples,
            )
        subdivisions.extend(
            _intervals_from_split_points(
                start_sample=start_sample,
                end_sample=end_sample,
                sample_rate=sample_rate,
                split_points=split_points,
                minimum_syllable_samples=minimum_syllable_samples,
            )
        )
    return subdivisions


def _detect_global_onset_samples(
    audio: np.ndarray,
    sample_rate: int,
    minimum_syllable_samples: int,
) -> list[int]:
    import librosa

    hop_length = max(128, min(512, int(sample_rate * 0.012)))
    onset_env = librosa.onset.onset_strength(
        y=np.asarray(audio, dtype=np.float32),
        sr=sample_rate,
        hop_length=hop_length,
        aggregate=np.median,
    )
    if onset_env.size < 3 or float(np.max(onset_env)) <= 1e-8:
        return []
    normalized = np.asarray(onset_env, dtype=np.float64)
    normalized /= float(np.max(normalized)) + 1e-12
    wait_frames = max(1, int(round(minimum_syllable_samples / hop_length)))
    detected_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=hop_length,
        units="frames",
        backtrack=False,
        pre_max=max(1, wait_frames // 2),
        post_max=max(1, wait_frames // 2),
        pre_avg=max(2, wait_frames),
        post_avg=max(2, wait_frames),
        delta=0.07,
        wait=wait_frames,
    )
    split_samples: list[int] = []
    for frame in detected_frames:
        frame = max(0, min(int(frame), normalized.size - 1))
        if normalized[frame] < 0.12:
            continue
        sample = int(librosa.frames_to_samples(frame, hop_length=hop_length))
        split_samples.append(
            _refine_split_to_energy_valley(audio, sample, sample_rate)
        )
    return _dedupe_split_points(
        split_samples,
        minimum_gap=minimum_syllable_samples,
    )


def _intervals_from_split_points(
    start_sample: int,
    end_sample: int,
    sample_rate: int,
    split_points: list[int],
    minimum_syllable_samples: int,
) -> list[tuple[int, int]]:
    interval_samples = max(0, end_sample - start_sample)
    if not split_points or interval_samples <= 0:
        return [(start_sample, end_sample)]

    boundary_lead_samples = max(
        0,
        int(round(sample_rate * SYLLABLE_BOUNDARY_LEAD_SECONDS)),
    )
    boundaries = [0]
    for split_point in split_points:
        adjusted_split_point = max(0, int(split_point) - boundary_lead_samples)
        if adjusted_split_point - boundaries[-1] < minimum_syllable_samples:
            continue
        if interval_samples - adjusted_split_point < minimum_syllable_samples:
            continue
        boundaries.append(adjusted_split_point)
    boundaries.append(interval_samples)
    if len(boundaries) <= 2:
        return [(start_sample, end_sample)]
    return [
        (start_sample + boundaries[index], start_sample + boundaries[index + 1])
        for index in range(len(boundaries) - 1)
    ]


def _detect_internal_onsets(
    segment: np.ndarray,
    sample_rate: int,
    minimum_syllable_samples: int,
) -> list[int]:
    split_samples = _detect_librosa_onset_samples(
        segment=segment,
        sample_rate=sample_rate,
        minimum_syllable_samples=minimum_syllable_samples,
    )
    if len(split_samples) <= 1:
        split_samples = _detect_audioflux_onset_samples(
            segment=segment,
            sample_rate=sample_rate,
            minimum_syllable_samples=minimum_syllable_samples,
        )
    split_samples = _dedupe_split_points(
        split_samples,
        minimum_gap=minimum_syllable_samples,
    )
    return _filter_onsets_by_syllable_nuclei(
        segment=segment,
        sample_rate=sample_rate,
        split_samples=split_samples,
    )


def _detect_librosa_onset_samples(
    segment: np.ndarray,
    sample_rate: int,
    minimum_syllable_samples: int,
) -> list[int]:
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

    normalized = np.asarray(onset_env, dtype=np.float64)
    normalized /= float(np.max(normalized)) + 1e-12
    wait_frames = max(1, int(round(minimum_syllable_samples / hop_length)))

    detected_frames = librosa.onset.onset_detect(
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

    split_samples: list[int] = []
    for frame in detected_frames:
        frame = max(0, min(int(frame), normalized.size - 1))
        if normalized[frame] < SYLLABLE_ONSET_MIN_STRENGTH:
            continue
        sample = int(librosa.frames_to_samples(frame, hop_length=hop_length))
        refined = _refine_split_to_energy_valley(segment, sample, sample_rate)
        split_samples.append(refined)
    return split_samples


def _filter_onsets_by_syllable_nuclei(
    segment: np.ndarray,
    sample_rate: int,
    split_samples: list[int],
) -> list[int]:
    """Keep onset boundaries that actually separate two vowel-energy nuclei.

    Spectral onset detectors also react to vibrato, breath releases and other
    within-syllable changes. A speech boundary is useful only when a vocal
    energy peak exists on both sides. This follows the open syllable-nucleus
    approach used by established speech-analysis workflows while retaining
    librosa's accurate spectral onset timing.
    """

    if not split_samples:
        return []
    nuclei = _detect_syllable_nuclei(segment, sample_rate)
    if len(nuclei) < 2:
        return []

    edge_margin = max(1, int(round(sample_rate * 0.012)))
    filtered: list[int] = []
    for split_sample in split_samples:
        has_left_nucleus = any(
            nucleus <= split_sample - edge_margin for nucleus in nuclei
        )
        has_right_nucleus = any(
            nucleus >= split_sample + edge_margin for nucleus in nuclei
        )
        if has_left_nucleus and has_right_nucleus:
            filtered.append(split_sample)
    return filtered


def _detect_syllable_nuclei(
    segment: np.ndarray,
    sample_rate: int,
) -> list[int]:
    """Return prominent vocal-energy peaks for onset validation."""

    import librosa
    from scipy.signal import find_peaks

    if segment.size < max(32, int(sample_rate * 0.04)):
        return []
    frame_length = max(64, int(round(sample_rate * 0.04)))
    hop_length = max(32, int(round(sample_rate * 0.005)))
    rms = librosa.feature.rms(
        y=np.asarray(segment, dtype=np.float32),
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    if rms.size < 3 or float(np.max(rms)) <= 1e-8:
        return []
    intensity_db = librosa.amplitude_to_db(rms, ref=np.max)
    minimum_distance = max(
        1,
        int(round(SYLLABLE_NUCLEUS_MIN_GAP_SECONDS * sample_rate / hop_length)),
    )
    peak_frames, _properties = find_peaks(
        intensity_db,
        distance=minimum_distance,
        prominence=SYLLABLE_NUCLEUS_MIN_PROMINENCE_DB,
        height=-25.0,
    )
    return [
        min(segment.size - 1, int(frame) * hop_length)
        for frame in peak_frames
    ]


def _detect_audioflux_onset_samples(
    segment: np.ndarray,
    sample_rate: int,
    minimum_syllable_samples: int,
) -> list[int]:
    try:
        import audioflux as af
        from audioflux.type import (
            NoveltyType,
            SpectralDataType,
            SpectralFilterBankScaleType,
        )
        from audioflux.utils import power_to_db
    except Exception:
        return []

    try:
        radix2_exp = 11
        fft_length = 1 << radix2_exp
        if segment.size < fft_length:
            radix2_exp = max(7, int(math.floor(math.log2(max(128, segment.size)))))
        hop_length = max(64, min(256, int(sample_rate * 0.006)))
        bft = af.BFT(
            num=128,
            samplate=sample_rate,
            radix2_exp=radix2_exp,
            slide_length=hop_length,
            scale_type=SpectralFilterBankScaleType.MEL,
            data_type=SpectralDataType.POWER,
        )
        spec = bft.bft(np.asarray(segment, dtype=np.float32))
        spec_db = power_to_db(np.abs(spec))
        onset = af.Onset(
            time_length=spec_db.shape[-1],
            fre_length=spec_db.shape[-2],
            slide_length=hop_length,
            samplate=sample_rate,
            novelty_type=NoveltyType.FLUX,
        )
        _points, _envelope, times, values = onset.onset(spec_db)
    except Exception:
        return []

    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if values.size == 0 or times.size == 0:
        return []

    threshold = max(float(np.max(values)) * 0.55, float(np.percentile(values, 72)))
    split_samples: list[int] = []
    last_sample = -minimum_syllable_samples
    for time_value, strength in sorted(zip(times, values, strict=False)):
        if not np.isfinite(time_value) or not np.isfinite(strength):
            continue
        if strength < threshold:
            continue
        sample = int(round(float(time_value) * sample_rate))
        if sample - last_sample < minimum_syllable_samples:
            continue
        refined = _refine_split_to_energy_valley(segment, sample, sample_rate)
        split_samples.append(refined)
        last_sample = refined
    return split_samples


def _detect_regular_valley_splits(
    segment: np.ndarray,
    sample_rate: int,
    minimum_syllable_samples: int,
) -> list[int]:
    """Fallback for connected syllables whose onsets are too soft.

    Some voice-pack phrases have vowel-to-vowel or very soft consonant
    transitions. Onset detectors can merge them, so this fallback estimates a
    likely syllable count from duration and snaps regular target points to local
    energy valleys. It only engages for short vocal phrases, not long songs.
    """

    duration = segment.size / max(1, sample_rate)
    if duration < 0.38 or duration > 2.4:
        return []

    target_count = int(round(duration / 0.235))
    target_count = max(2, min(7, target_count))
    if target_count <= 1:
        return []

    split_points: list[int] = []
    for index in range(1, target_count):
        sample = int(round(segment.size * index / target_count))
        refined = _refine_split_to_energy_valley(segment, sample, sample_rate)
        if refined < minimum_syllable_samples:
            continue
        if segment.size - refined < minimum_syllable_samples:
            continue
        split_points.append(refined)
    return _dedupe_split_points(split_points, minimum_gap=minimum_syllable_samples)


def _refine_split_to_energy_valley(
    segment: np.ndarray,
    sample: int,
    sample_rate: int,
) -> int:
    search_before = int(sample_rate * 0.035)
    search_after = int(sample_rate * 0.012)
    start = max(0, sample - search_before)
    end = min(segment.size, sample + search_after)
    if end - start < 4:
        return max(0, min(segment.size, sample))

    envelope = np.abs(segment[start:end]).astype(np.float64)
    kernel_size = max(3, int(sample_rate * 0.004))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if envelope.size >= kernel_size:
        kernel = np.ones(kernel_size, dtype=np.float64) / float(kernel_size)
        envelope = np.convolve(envelope, kernel, mode="same")
    return int(start + np.argmin(envelope))


def _dedupe_split_points(
    split_points: list[int],
    minimum_gap: int,
) -> list[int]:
    clean: list[int] = []
    for point in sorted({int(point) for point in split_points}):
        if point <= 0:
            continue
        if clean and point - clean[-1] < minimum_gap:
            previous = clean[-1]
            clean[-1] = int(round((previous + point) / 2.0))
            continue
        clean.append(point)
    return clean


def _merge_tiny_intervals(
    intervals: list[tuple[int, int]],
    sample_rate: int,
) -> list[tuple[int, int]]:
    minimum_samples = max(1, int(sample_rate * 0.045))
    merged: list[tuple[int, int]] = []
    for start_sample, end_sample in sorted(intervals):
        if end_sample <= start_sample:
            continue
        if end_sample - start_sample < minimum_samples and merged:
            previous_start, _previous_end = merged[-1]
            merged[-1] = (previous_start, end_sample)
            continue
        merged.append((start_sample, end_sample))
    return merged


def _make_contiguous_slice_intervals(
    intervals: list[tuple[int, int]],
    audio_size: int,
) -> list[tuple[int, int]]:
    """Convert detected syllable ranges into a clean partition of the source.

    Automatic slicing is an editing operation, not a comfort-preview operation:
    the resulting blocks must not overlap, must not add preroll/release tails,
    and must not leave gaps. If the source contains leading/trailing silence or
    silence between voiced regions, that silence is assigned to the adjacent
    slice boundary so the slice durations add up exactly to the original file
    duration.
    """

    clean_intervals = [
        (max(0, int(start_sample)), min(int(audio_size), int(end_sample)))
        for start_sample, end_sample in sorted(intervals)
        if int(end_sample) > int(start_sample)
    ]
    if not clean_intervals:
        return []
    if len(clean_intervals) == 1:
        return [(0, int(audio_size))]

    boundaries = [0]
    for left, right in zip(clean_intervals, clean_intervals[1:]):
        boundary = int(round((left[1] + right[0]) / 2.0))
        boundary = max(boundaries[-1] + 1, min(int(audio_size) - 1, boundary))
        boundaries.append(boundary)
    boundaries.append(int(audio_size))

    return [
        (boundaries[index], boundaries[index + 1])
        for index in range(len(boundaries) - 1)
        if boundaries[index + 1] > boundaries[index]
    ]


def _estimate_average_f0(segment: np.ndarray, sample_rate: int) -> float | None:
    import librosa

    if segment.size < max(32, int(sample_rate * 0.02)):
        return None

    try:
        from hakyking.audio.audio_engine import _estimate_pitch_center_hz

        pitch_center = _estimate_pitch_center_hz(segment, sample_rate)
        if pitch_center is not None:
            return pitch_center
    except Exception:
        pass

    fmin = 50.0
    fmax = min(float(librosa.note_to_hz("C7")), sample_rate * 0.45)
    if fmax <= fmin:
        return None

    frame_length = min(2048, max(128, int(2 ** math.floor(math.log2(segment.size)))))
    if frame_length < 128:
        return None
    hop_length = max(32, frame_length // 4)

    try:
        f0_values = librosa.yin(
            segment,
            fmin=fmin,
            fmax=fmax,
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    except Exception:
        return None

    valid = np.asarray(f0_values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    valid = valid[(valid >= fmin) & (valid <= fmax)]
    if valid.size == 0:
        return None

    lower, upper = np.percentile(valid, [10, 90])
    trimmed = valid[(valid >= lower) & (valid <= upper)]
    if trimmed.size == 0:
        trimmed = valid
    return float(np.mean(trimmed))


def _estimate_initial_f0(segment: np.ndarray, sample_rate: int) -> float | None:
    """Estimate the first stable voiced pitch for timeline placement."""

    if segment.size < max(32, int(sample_rate * 0.02)):
        return None
    scan_samples = min(segment.size, max(int(sample_rate * 0.22), int(sample_rate * 0.06)))
    return _estimate_average_f0(segment[:scan_samples], sample_rate)


def _hz_to_midi(f0_hz: float | None) -> int | None:
    if f0_hz is None or f0_hz <= 0:
        return None
    midi = int(round(69 + 12 * math.log2(f0_hz / 440.0)))
    return max(0, min(127, midi))
