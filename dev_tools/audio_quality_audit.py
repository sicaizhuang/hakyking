from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "qa_artifacts"
RUBBERBAND_ROOT = ROOT / "tools" / "rubberband"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for candidate in RUBBERBAND_ROOT.rglob("rubberband.exe"):
    os.environ["PATH"] = str(candidate.parent) + os.pathsep + os.environ.get("PATH", "")
    break

from hakyking.audio.audio_engine import (  # noqa: E402
    calculate_time_rate,
    process_advanced_blob,
    process_blob,
    render_slice_from_file,
    transient_protected_time_stretch,
)
from hakyking.audio.playback import prepare_playback_audio  # noqa: E402
from hakyking.models.audio_slice import AudioSlice  # noqa: E402


@dataclass(frozen=True)
class AuditResult:
    name: str
    status: str
    detail: str
    elapsed: float


class AudioQualityAudit:
    def __init__(self) -> None:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        self.sample_rate = 44100
        self.source_path = ARTIFACTS / "quality_source_vowel.wav"
        self.source_audio = self._write_source()
        self.results: list[AuditResult] = []

    def run(self) -> int:
        tests = [
            self.test_pitch_shift_hits_target,
            self.test_time_stretch_hits_duration,
            self.test_transient_protected_time_stretch,
            self.test_playback_edges_are_faded,
            self.test_gain_preserves_relative_level_and_master_is_safe,
            self.test_advanced_formant_flatten_is_stable,
            self.test_silence_and_tiny_inputs_are_safe,
        ]
        for test in tests:
            self._run_one(test)
        self._write_report()
        fail_count = sum(1 for result in self.results if result.status == "FAIL")
        pass_count = sum(1 for result in self.results if result.status == "PASS")
        print(f"audio_quality_audit PASS={pass_count} FAIL={fail_count}")
        print(f"report={ARTIFACTS / 'audio_quality_audit_latest.md'}")
        return 1 if fail_count else 0

    def test_pitch_shift_hits_target(self) -> str:
        audio_slice = AudioSlice(
            source_path=str(self.source_path),
            index=0,
            start_time=0.0,
            end_time=1.0,
            midi_note=60,
            f0_hz=261.625565,
        )
        result = render_slice_from_file(
            audio_slice,
            target_midi_note=72,
            target_duration=1.0,
            cache_key="quality_pitch",
        )
        midi = _estimate_median_midi(result.audio, result.sample_rate)
        assert midi is not None, "pitch estimator found no voiced frames"
        error = abs(midi - 72.0)
        assert error <= 1.25, f"target C5, estimated MIDI={midi:.2f}, error={error:.2f}"
        _assert_audio_safe(result.audio)
        sf.write(str(ARTIFACTS / "quality_pitch_up.wav"), result.audio, result.sample_rate)
        return f"target=72.00 estimated={midi:.2f} error={error:.2f}"

    def test_time_stretch_hits_duration(self) -> str:
        target_duration = 1.45
        rate = calculate_time_rate(1.0, target_duration)
        rendered = process_blob(self.source_audio, self.sample_rate, rate=rate)
        duration = len(rendered) / self.sample_rate
        error = abs(duration - target_duration)
        assert error <= 0.08, f"target={target_duration:.2f}s actual={duration:.2f}s"
        _assert_audio_safe(rendered)
        sf.write(str(ARTIFACTS / "quality_time_stretch.wav"), rendered, self.sample_rate)
        return f"target={target_duration:.2f}s actual={duration:.2f}s error={error:.3f}s"

    def test_transient_protected_time_stretch(self) -> str:
        source = self.source_audio.copy()
        transient_length = int(self.sample_rate * 0.025)
        source[:transient_length] += np.random.default_rng(7).normal(
            0.0,
            0.18,
            transient_length,
        ).astype(np.float32)
        stretched = transient_protected_time_stretch(source, self.sample_rate, 1.5)
        contracted = transient_protected_time_stretch(source, self.sample_rate, 0.1)
        assert len(stretched) == round(len(source) * 1.5)
        assert len(contracted) == round(len(source) * 0.1)
        protected_head = int(self.sample_rate * 0.05)
        head_error = float(np.max(np.abs(stretched[:protected_head] - source[:protected_head])))
        assert head_error <= 1e-7, f"protected transient changed: {head_error:.7f}"
        _assert_audio_safe(stretched)
        _assert_audio_safe(contracted)
        return (
            f"stretch={len(stretched)} shrink={len(contracted)} "
            f"protected_head_error={head_error:.7f}"
        )

    def test_playback_edges_are_faded(self) -> str:
        prepared = prepare_playback_audio(self.source_audio, self.sample_rate, fade_ms=8.0)
        head_peak = float(np.max(np.abs(prepared[:128])))
        tail_peak = float(np.max(np.abs(prepared[-128:])))
        assert head_peak < 0.04, f"head edge too hot: {head_peak:.4f}"
        assert tail_peak < 0.04, f"tail edge too hot: {tail_peak:.4f}"
        click_score = _edge_click_score(prepared)
        assert click_score < 0.08, f"edge click score too high: {click_score:.4f}"
        return f"head={head_peak:.4f} tail={tail_peak:.4f} click_score={click_score:.4f}"

    def test_gain_preserves_relative_level_and_master_is_safe(self) -> str:
        source = self.source_audio * 1.8
        baseline = process_blob(source, self.sample_rate, gain_db=0.0)
        rendered = process_blob(source, self.sample_rate, gain_db=18.0)
        _assert_audio_finite(baseline)
        _assert_audio_finite(rendered)
        baseline_rms = float(np.sqrt(np.mean(np.square(baseline, dtype=np.float64))))
        rendered_rms = float(np.sqrt(np.mean(np.square(rendered, dtype=np.float64))))
        ratio = rendered_rms / max(1e-12, baseline_rms)
        expected_ratio = 10.0 ** (18.0 / 20.0)
        assert abs(ratio - expected_ratio) <= 0.02, (
            f"gain ratio={ratio:.4f}, expected={expected_ratio:.4f}"
        )
        assert float(np.max(np.abs(rendered))) > 1.0, "slice gain was normalized too early"
        prepared = prepare_playback_audio(rendered, self.sample_rate)
        _assert_audio_safe(prepared)
        return (
            f"internal_gain_ratio={ratio:.3f} "
            f"internal_peak={float(np.max(np.abs(rendered))):.3f} "
            f"master_peak={float(np.max(np.abs(prepared))):.3f}"
        )

    def test_advanced_formant_flatten_is_stable(self) -> str:
        vibrato_audio = self._synth_vibrato_vowel(seconds=1.2)
        rendered = process_advanced_blob(
            vibrato_audio,
            self.sample_rate,
            n_steps=4.0,
            flatten_amount=0.85,
            formant_shift=4.0,
        )
        _assert_audio_safe(rendered)
        duration = len(rendered) / self.sample_rate
        assert 1.0 <= duration <= 1.35, f"unexpected advanced duration={duration:.2f}s"
        sf.write(str(ARTIFACTS / "quality_advanced.wav"), rendered, self.sample_rate)
        return f"duration={duration:.2f}s peak={float(np.max(np.abs(rendered))):.3f}"

    def test_silence_and_tiny_inputs_are_safe(self) -> str:
        silence = np.zeros(2048, dtype=np.float32)
        tiny = np.ones(24, dtype=np.float32) * 0.2
        rendered_silence = process_blob(silence, self.sample_rate, n_steps=5.0)
        rendered_tiny = process_blob(tiny, self.sample_rate, n_steps=5.0)
        _assert_audio_safe(rendered_silence)
        _assert_audio_safe(rendered_tiny)
        assert rendered_silence.shape[0] > 0
        assert rendered_tiny.shape[0] == tiny.shape[0]
        return f"silence={rendered_silence.shape[0]} samples tiny={rendered_tiny.shape[0]} samples"

    def _run_one(self, test) -> None:  # noqa: ANN001
        start = time.perf_counter()
        name = test.__name__.replace("test_", "")
        try:
            detail = test()
        except Exception as exc:  # noqa: BLE001 - audit reports failures
            self.results.append(AuditResult(name, "FAIL", repr(exc), time.perf_counter() - start))
            print(f"[FAIL] {name}: {exc!r}", flush=True)
        else:
            self.results.append(AuditResult(name, "PASS", detail, time.perf_counter() - start))
            print(f"[PASS] {name}: {detail}", flush=True)

    def _write_source(self) -> np.ndarray:
        audio = self._synth_vowel(seconds=1.0)
        sf.write(str(self.source_path), audio, self.sample_rate)
        return audio

    def _synth_vowel(self, seconds: float) -> np.ndarray:
        t = np.arange(int(self.sample_rate * seconds), dtype=np.float32) / self.sample_rate
        f0 = 261.625565
        audio = (
            0.45 * np.sin(2.0 * math.pi * f0 * t)
            + 0.22 * np.sin(2.0 * math.pi * f0 * 2.0 * t)
            + 0.12 * np.sin(2.0 * math.pi * f0 * 3.0 * t)
        )
        envelope = np.sin(np.linspace(0.0, math.pi, audio.size, dtype=np.float32))
        return np.asarray(audio * envelope * 0.65, dtype=np.float32)

    def _synth_vibrato_vowel(self, seconds: float) -> np.ndarray:
        t = np.arange(int(self.sample_rate * seconds), dtype=np.float32) / self.sample_rate
        base = 220.0
        vibrato = 2.0 ** (0.55 * np.sin(2.0 * math.pi * 5.2 * t) / 12.0)
        phase = 2.0 * math.pi * np.cumsum(base * vibrato) / self.sample_rate
        audio = 0.5 * np.sin(phase) + 0.18 * np.sin(phase * 2.0)
        envelope = np.sin(np.linspace(0.0, math.pi, audio.size, dtype=np.float32))
        return np.asarray(audio * envelope * 0.6, dtype=np.float32)

    def _write_report(self) -> None:
        report_path = ARTIFACTS / "audio_quality_audit_latest.md"
        pass_count = sum(1 for result in self.results if result.status == "PASS")
        fail_count = sum(1 for result in self.results if result.status == "FAIL")
        lines = [
            "# Hakyking Audio Quality Audit",
            "",
            f"- PASS: {pass_count}",
            f"- FAIL: {fail_count}",
            "",
            "| Test | Status | Time | Detail |",
            "| --- | --- | ---: | --- |",
        ]
        for result in self.results:
            detail = result.detail.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{result.name}` | {result.status} | {result.elapsed:.2f}s | {detail} |")
        lines.extend(
            [
                "",
                "## Rendered Listening Artifacts",
                "",
                "- `qa_artifacts/quality_pitch_up.wav`",
                "- `qa_artifacts/quality_time_stretch.wav`",
                "- `qa_artifacts/quality_advanced.wav`",
            ]
        )
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _estimate_median_midi(audio: np.ndarray, sample_rate: int) -> float | None:
    import librosa

    source = np.asarray(audio, dtype=np.float32)
    if source.size < 512:
        return None
    f0 = librosa.yin(
        source,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
    )
    valid = np.isfinite(f0) & (f0 > 0)
    if int(np.sum(valid)) < 3:
        return None
    return float(np.median(librosa.hz_to_midi(f0[valid])))


def _assert_audio_safe(audio: np.ndarray) -> None:
    _assert_audio_finite(audio)
    source = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(source)))
    assert peak <= 1.0001, f"audio clips: peak={peak:.4f}"


def _assert_audio_finite(audio: np.ndarray) -> None:
    source = np.asarray(audio, dtype=np.float32)
    assert source.size > 0, "empty audio"
    assert np.all(np.isfinite(source)), "audio contains NaN or Inf"


def _edge_click_score(audio: np.ndarray) -> float:
    source = np.asarray(audio, dtype=np.float32)
    if source.size < 4:
        return 0.0
    head = float(np.max(np.abs(np.diff(source[: min(512, source.size)]))))
    tail = float(np.max(np.abs(np.diff(source[-min(512, source.size) :]))))
    return max(head, tail)


def _integrated_loudness(audio: np.ndarray, sample_rate: int) -> float:
    import pyloudnorm as pyln

    source = np.asarray(audio, dtype=np.float32)
    min_len = int(sample_rate * 0.45)
    if source.size < min_len:
        source = np.pad(source, (0, min_len - source.size))
    meter = pyln.Meter(sample_rate)
    value = float(meter.integrated_loudness(source))
    return value if math.isfinite(value) else -120.0


if __name__ == "__main__":
    raise SystemExit(AudioQualityAudit().run())
