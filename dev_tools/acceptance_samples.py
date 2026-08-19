from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hakyking.audio.reader import AudioReader  # noqa: E402
from hakyking.audio.slicer import parse_audio_slices  # noqa: E402


DEFAULT_MANIFEST = PROJECT_ROOT / "qa" / "acceptance_samples.json"


class AcceptanceFailure(RuntimeError):
    pass


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("samples"), list):
        raise AcceptanceFailure(f"Invalid acceptance manifest: {path}")
    return payload


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceFailure(message)


def _stream_preview(path: Path, seconds: float | None) -> tuple[np.ndarray, int]:
    with sf.SoundFile(str(path)) as source:
        frame_count = len(source)
        if seconds is not None:
            frame_count = min(frame_count, max(1, int(round(seconds * source.samplerate))))
        audio = source.read(frame_count, dtype="float32", always_2d=True)
        sample_rate = int(source.samplerate)
    mono = np.mean(np.asarray(audio, dtype=np.float32), axis=1, dtype=np.float32)
    return np.ascontiguousarray(mono), sample_rate


def _median_midi(audio: np.ndarray, sample_rate: int) -> float:
    import librosa

    f0 = librosa.yin(
        audio,
        fmin=60.0,
        fmax=min(1000.0, sample_rate * 0.45),
        sr=sample_rate,
        frame_length=2048,
        hop_length=256,
    )
    valid = np.asarray(f0, dtype=np.float64)
    valid = valid[np.isfinite(valid) & (valid > 0.0)]
    _assert(valid.size > 0, "pitch scan returned no finite F0 frames")
    midi = 69.0 + 12.0 * np.log2(valid / 440.0)
    return float(np.median(midi))


def validate_sample(sample: dict[str, object]) -> list[str]:
    sample_id = str(sample.get("id", "unknown"))
    path = Path(str(sample["path"]))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    expected = dict(sample["expected"])
    _assert(path.is_file(), f"{sample_id}: source is missing: {path}")

    info = AudioReader.read_info(str(path))
    _assert(info.sample_rate == int(expected["sample_rate"]), f"{sample_id}: sample rate changed")
    _assert(info.channels == int(expected["channels"]), f"{sample_id}: channel count changed")
    duration_tolerance = float(expected["duration_tolerance_seconds"])
    _assert(
        abs(info.duration - float(expected["duration_seconds"])) <= duration_tolerance,
        f"{sample_id}: duration changed to {info.duration:.6f}s",
    )

    stream_seconds = expected.get("stream_scan_seconds")
    audio, sample_rate = _stream_preview(
        path,
        None if stream_seconds is None else float(stream_seconds),
    )
    _assert(audio.size > 0, f"{sample_id}: preview buffer is empty")
    _assert(bool(np.all(np.isfinite(audio))), f"{sample_id}: preview buffer contains NaN/Inf")
    rms = float(math.sqrt(float(np.mean(audio.astype(np.float64) ** 2))))
    _assert(rms >= float(expected["preview_rms_min"]), f"{sample_id}: preview is unexpectedly silent")

    messages = [
        f"metadata {info.sample_rate} Hz / {info.duration:.3f}s / {info.channels} ch",
        f"preview finite, RMS={rms:.5f}",
    ]

    baseline_count = expected.get("parser_baseline_slice_count")
    if baseline_count is not None:
        slices = parse_audio_slices(str(path))
        _assert(len(slices) == int(baseline_count), f"{sample_id}: slice baseline changed to {len(slices)}")
        _assert(bool(slices), f"{sample_id}: no slices")
        _assert(abs(slices[0].start_time) <= 1e-6, f"{sample_id}: slices do not start at zero")
        _assert(
            abs(slices[-1].end_time - info.duration) <= duration_tolerance,
            f"{sample_id}: slices do not cover the source tail",
        )
        for left, right in zip(slices, slices[1:], strict=False):
            _assert(
                abs(left.end_time - right.start_time) <= 1e-6,
                f"{sample_id}: slices contain a gap or overlap",
            )
        midi_min, midi_max = (int(value) for value in expected["midi_range"])
        voiced = [entry.midi_note for entry in slices if entry.midi_note is not None]
        _assert(bool(voiced), f"{sample_id}: no voiced slice pitch")
        _assert(all(midi_min <= value <= midi_max for value in voiced), f"{sample_id}: pitch left expected range")
        target = int(expected["semantic_syllable_target"])
        messages.append(
            f"slice baseline={len(slices)}, semantic target={target}, MIDI={min(voiced)}..{max(voiced)}"
        )
    else:
        median_midi = _median_midi(audio, sample_rate)
        midi_min, midi_max = (float(value) for value in expected["median_midi_range"])
        _assert(midi_min <= median_midi <= midi_max, f"{sample_id}: median MIDI changed to {median_midi:.2f}")
        messages.append(f"streamed pitch median MIDI={median_midi:.2f}")
    return messages


def run(manifest_path: Path = DEFAULT_MANIFEST) -> int:
    manifest = load_manifest(manifest_path)
    failures: list[str] = []
    for raw_sample in manifest["samples"]:
        sample = dict(raw_sample)
        sample_id = str(sample.get("id", "unknown"))
        source_path = Path(str(sample.get("path", "")))
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        if not source_path.is_file():
            print(f"SKIP {sample_id}: optional local fixture is missing: {source_path}")
            continue
        try:
            messages = validate_sample(sample)
        except Exception as exc:
            failures.append(f"{sample_id}: {exc}")
            print(f"FAIL {sample_id}: {exc}")
            continue
        print(f"PASS {sample_id}")
        for message in messages:
            print(f"  {message}")
    if failures:
        print(f"\n{len(failures)} acceptance sample(s) failed.")
        return 1
    print("\nAll three fixed acceptance samples passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Hakyking's three frozen acceptance samples.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    return run(args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
