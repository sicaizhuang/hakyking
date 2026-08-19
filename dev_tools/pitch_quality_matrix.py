from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "qa_artifacts" / "pitch_quality_matrix"
MANIFEST = ROOT / "qa" / "acceptance_samples.json"
ENGINES = ("rubberband", "parselmouth_psola", "pyworld_hpss")
STEPS = (-12.0, -5.0, -2.0, 2.0, 5.0, 12.0)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hakyking.audio.audio_engine import process_blob  # noqa: E402
from hakyking.audio.playback import prepare_playback_audio  # noqa: E402
from hakyking.audio.reader import AudioReader  # noqa: E402


@dataclass(frozen=True)
class MatrixResult:
    sample_id: str
    engine: str
    steps: float
    elapsed_seconds: float
    pitch_error_semitones: float
    duration_error_ms: float
    rms_delta_db: float
    onset_correlation: float
    high_frequency_delta_db: float
    peak: float
    clip_fraction: float
    finite: bool
    output_path: str


def _db_ratio(numerator: float, denominator: float) -> float:
    return 20.0 * math.log10(max(1e-12, numerator) / max(1e-12, denominator))


def _rms(audio: np.ndarray) -> float:
    source = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(source)))) if source.size else 0.0


def _median_midi(
    audio: np.ndarray,
    sample_rate: int,
    expected_midi: float | None = None,
) -> float | None:
    import librosa
    import pyworld as pw

    source = np.asarray(audio, dtype=np.float64)
    if source.size < int(sample_rate * 0.04):
        return None
    f0, time_axis = pw.dio(
        source,
        sample_rate,
        f0_floor=45.0,
        f0_ceil=min(1600.0, sample_rate * 0.45),
        frame_period=10.0,
    )
    f0 = pw.stonemask(source, f0, time_axis, sample_rate)
    voiced = np.asarray(f0, dtype=np.float64)
    voiced = voiced[np.isfinite(voiced) & (voiced > 0.0)]
    if voiced.size == 0:
        return None
    world_midi = 69.0 + 12.0 * np.log2(voiced / 440.0)
    lower, upper = np.percentile(world_midi, [10.0, 90.0])
    central = world_midi[(world_midi >= lower) & (world_midi <= upper)]
    world_center = float(np.median(central if central.size else world_midi))

    # WORLD can lock to a strong second harmonic after a full-octave downward
    # shift. YIN is used as an octave-error referee, not the primary tracker.
    yin_f0 = librosa.yin(
        np.asarray(audio, dtype=np.float32),
        fmin=45.0,
        fmax=min(1200.0, sample_rate * 0.45),
        sr=sample_rate,
        frame_length=4096,
        hop_length=512,
    )
    yin_f0 = np.asarray(yin_f0, dtype=np.float64)
    yin_f0 = yin_f0[np.isfinite(yin_f0) & (yin_f0 > 0.0)]
    candidates = [world_center]
    if yin_f0.size:
        yin_midi = 69.0 + 12.0 * np.log2(yin_f0 / 440.0)
        yin_center = float(np.median(yin_midi))
        candidates.append(yin_center)
    if expected_midi is not None:
        return min(candidates, key=lambda value: abs(value - expected_midi))
    return world_center


def _onset_correlation(source: np.ndarray, rendered: np.ndarray, sample_rate: int) -> float:
    import librosa

    left = librosa.onset.onset_strength(
        y=np.asarray(source, dtype=np.float32),
        sr=sample_rate,
        hop_length=256,
    )
    right = librosa.onset.onset_strength(
        y=np.asarray(rendered, dtype=np.float32),
        sr=sample_rate,
        hop_length=256,
    )
    length = min(left.size, right.size)
    if length < 3:
        return 1.0
    left = np.asarray(left[:length], dtype=np.float64)
    right = np.asarray(right[:length], dtype=np.float64)
    if float(np.std(left)) <= 1e-9 or float(np.std(right)) <= 1e-9:
        return 1.0
    return float(np.clip(np.corrcoef(left, right)[0, 1], -1.0, 1.0))


def _high_frequency_ratio(audio: np.ndarray, sample_rate: int) -> float:
    source = np.asarray(audio, dtype=np.float64)
    if source.size < 8:
        return 0.0
    window = np.hanning(source.size)
    spectrum = np.abs(np.fft.rfft(source * window)) ** 2
    frequencies = np.fft.rfftfreq(source.size, d=1.0 / sample_rate)
    total = float(np.sum(spectrum))
    if total <= 1e-18:
        return 0.0
    threshold = min(8000.0, sample_rate * 0.35)
    return float(np.sum(spectrum[frequencies >= threshold]) / total)


def _safe_slug(value: float) -> str:
    prefix = "p" if value >= 0 else "m"
    return f"{prefix}{abs(value):g}".replace(".", "_")


def run() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    samples = [
        sample
        for sample in manifest["samples"]
        if sample.get("role") == "short_vocal_phrase"
    ]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    results: list[MatrixResult] = []
    old_engine = os.environ.get("HAKYKING_PITCH_ENGINE")
    try:
        for sample in samples:
            sample_id = str(sample["id"])
            source, sample_rate = AudioReader.load_mono(str(sample["path"]))
            source_midi = _median_midi(source, sample_rate)
            if source_midi is None:
                raise RuntimeError(f"{sample_id}: no source pitch")
            source_rms = _rms(source)
            source_hf = _high_frequency_ratio(source, sample_rate)
            sample_dir = ARTIFACTS / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            for engine in ENGINES:
                os.environ["HAKYKING_PITCH_ENGINE"] = engine
                for steps in STEPS:
                    started = time.perf_counter()
                    rendered = process_blob(source, sample_rate, n_steps=steps)
                    elapsed = time.perf_counter() - started
                    rendered = np.asarray(rendered, dtype=np.float32)
                    finite = bool(rendered.size and np.all(np.isfinite(rendered)))
                    rendered_midi = (
                        _median_midi(
                            rendered,
                            sample_rate,
                            expected_midi=source_midi + steps,
                        )
                        if finite
                        else None
                    )
                    pitch_error = (
                        abs(rendered_midi - (source_midi + steps))
                        if rendered_midi is not None
                        else float("inf")
                    )
                    duration_error_ms = abs(rendered.size - source.size) * 1000.0 / sample_rate
                    rendered_rms = _rms(rendered)
                    hf_delta = _db_ratio(
                        _high_frequency_ratio(rendered, sample_rate),
                        source_hf,
                    )
                    peak = float(np.max(np.abs(rendered))) if rendered.size else 0.0
                    clip_fraction = (
                        float(np.mean(np.abs(rendered) >= 1.0)) if rendered.size else 0.0
                    )
                    output_path = sample_dir / f"{engine}_{_safe_slug(steps)}.wav"
                    listening = prepare_playback_audio(rendered, sample_rate)
                    sf.write(str(output_path), listening, sample_rate, subtype="PCM_16")
                    result = MatrixResult(
                        sample_id=sample_id,
                        engine=engine,
                        steps=steps,
                        elapsed_seconds=elapsed,
                        pitch_error_semitones=float(pitch_error),
                        duration_error_ms=float(duration_error_ms),
                        rms_delta_db=_db_ratio(rendered_rms, source_rms),
                        onset_correlation=_onset_correlation(source, rendered, sample_rate),
                        high_frequency_delta_db=hf_delta,
                        peak=peak,
                        clip_fraction=clip_fraction,
                        finite=finite,
                        output_path=str(output_path),
                    )
                    results.append(result)
                    print(
                        f"{sample_id} {engine} {steps:+g}: "
                        f"pitch={pitch_error:.2f}st rms={result.rms_delta_db:+.2f}dB "
                        f"onset={result.onset_correlation:.2f} time={elapsed:.2f}s",
                        flush=True,
                    )
    finally:
        if old_engine is None:
            os.environ.pop("HAKYKING_PITCH_ENGINE", None)
        else:
            os.environ["HAKYKING_PITCH_ENGINE"] = old_engine

    report_path = ROOT / "qa_artifacts" / "pitch_quality_matrix_latest.md"
    json_path = ROOT / "qa_artifacts" / "pitch_quality_matrix_latest.json"
    json_path.write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Hakyking Pitch Quality Matrix",
        "",
        "Lower pitch error and level drift are better; higher onset correlation is better.",
        "The WAV files are passed through the same final playback limiter used by the app.",
        "",
        "| Sample | Engine | Shift | Pitch error | RMS delta | Onset corr. | HF delta | Time |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| `{result.sample_id}` | `{result.engine}` | {result.steps:+g} st | "
            f"{result.pitch_error_semitones:.2f} st | {result.rms_delta_db:+.2f} dB | "
            f"{result.onset_correlation:.2f} | {result.high_frequency_delta_db:+.2f} dB | "
            f"{result.elapsed_seconds:.2f}s |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    failures = [
        result
        for result in results
        if (
            not result.finite
            or result.duration_error_ms > 25.0
            or result.pitch_error_semitones > 1.5
        )
    ]
    print(f"report={report_path}")
    print(f"matrix_rows={len(results)} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
