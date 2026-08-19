from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hakyking.audio.audio_engine import process_blob  # noqa: E402
from hakyking.audio.reader import AudioReader  # noqa: E402
from hakyking.audio.slicer import parse_audio_slices  # noqa: E402
from hakyking.runtime import configure_external_tool_paths  # noqa: E402


AUDIO_EXTENSIONS = AudioReader.supported_extensions
DEFAULT_INPUT = ROOT / "tests" / "fixtures" / "acceptance"
DEFAULT_OUTPUT = ROOT / "qa_artifacts" / "algorithm_lab"


@dataclass(frozen=True)
class SliceRegion:
    start: float
    end: float
    f0_hz: float | None
    midi: float | None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class AlgorithmResult:
    name: str
    regions: list[SliceRegion]
    elapsed: float
    error: str = ""


@dataclass(frozen=True)
class ScoreResult:
    matched: int
    predicted_boundaries: int
    reference_boundaries: int
    mean_error_ms: float | None
    score: float | None


def main() -> int:
    configure_external_tool_paths()
    args = _parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = _load_annotations(Path(args.annotations).resolve() if args.annotations else None, input_dir)
    files = _scan_audio_files(input_dir, max_files=args.max_files)
    if not files:
        print(f"No supported audio/video files found: {input_dir}")
        return 1

    report_items = []
    start = time.perf_counter()
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path}")
        try:
            report_items.append(
                analyze_file(
                    path=path,
                    output_dir=output_dir,
                    max_duration=args.max_duration,
                    annotations=annotations,
                    pitch_steps=list(args.pitch_steps),
                )
            )
        except Exception as exc:  # noqa: BLE001 - lab should continue across bad files
            report_items.append(
                {
                    "path": str(path),
                    "title": path.name,
                    "error": repr(exc),
                    "algorithms": [],
                    "pitch_tests": [],
                    "plot": "",
                }
            )
            print(f"  failed: {exc!r}")

    report_path = output_dir / "index.html"
    report_path.write_text(
        _render_html_report(
            input_dir=input_dir,
            output_dir=output_dir,
            items=report_items,
            elapsed=time.perf_counter() - start,
        ),
        encoding="utf-8",
    )
    print(f"report={report_path}")
    return 0


def analyze_file(
    path: Path,
    output_dir: Path,
    max_duration: float,
    annotations: dict[str, list[tuple[float, float, str]]],
    pitch_steps: list[float],
) -> dict[str, object]:
    audio, sample_rate = AudioReader.load_mono(str(path))
    audio = _finite_mono(audio)
    original_duration = len(audio) / sample_rate if sample_rate > 0 else 0.0
    if max_duration > 0 and original_duration > max_duration:
        audio = audio[: int(round(max_duration * sample_rate))]
    duration = len(audio) / sample_rate if sample_rate > 0 else 0.0

    file_dir = output_dir / _safe_name(path.stem)
    if file_dir.exists():
        shutil.rmtree(file_dir)
    file_dir.mkdir(parents=True, exist_ok=True)

    clipped_path = file_dir / "source_clip.wav"
    sf.write(str(clipped_path), audio, sample_rate)

    f0_times, f0_hz = _extract_f0_curve(audio, sample_rate)
    algorithms = [
        _run_algorithm("hakyking_current", lambda: _algo_hakyking_current(clipped_path)),
        _run_algorithm("librosa_split", lambda: _algo_librosa_split(audio, sample_rate, str(clipped_path))),
        _run_algorithm("onset_grid", lambda: _algo_onset_grid(audio, sample_rate, str(clipped_path))),
        _run_algorithm("energy_valley", lambda: _algo_energy_valley(audio, sample_rate, str(clipped_path))),
        _run_algorithm("hybrid_vote", lambda: _algo_hybrid_vote(audio, sample_rate, str(clipped_path))),
    ]

    reference = _reference_for(path, annotations)
    scores = {
        result.name: _score_regions(result.regions, reference)
        for result in algorithms
        if not result.error
    }

    slice_previews: dict[str, list[dict[str, str]]] = {}
    for result in algorithms:
        if result.error:
            continue
        slice_previews[result.name] = _write_slice_wavs(
            result,
            audio,
            sample_rate,
            file_dir / result.name,
            report_root=output_dir,
        )

    plot_path = file_dir / "analysis.png"
    _plot_analysis(
        audio=audio,
        sample_rate=sample_rate,
        f0_times=f0_times,
        f0_hz=f0_hz,
        algorithms=algorithms,
        reference=reference,
        output_path=plot_path,
        title=path.name,
    )

    pitch_tests = _write_pitch_tests(audio, sample_rate, file_dir, pitch_steps, report_root=output_dir)

    return {
        "path": str(path),
        "title": path.name,
        "duration": duration,
        "original_duration": original_duration,
        "sample_rate": sample_rate,
        "source_clip": str(clipped_path.relative_to(output_dir)).replace("\\", "/"),
        "plot": str(plot_path.relative_to(output_dir)).replace("\\", "/"),
        "algorithms": algorithms,
        "scores": scores,
        "reference": reference,
        "slice_previews": slice_previews,
        "pitch_tests": pitch_tests,
        "error": "",
    }


def _run_algorithm(name: str, callback) -> AlgorithmResult:  # noqa: ANN001
    started = time.perf_counter()
    try:
        regions = callback()
    except Exception as exc:  # noqa: BLE001 - result captures failures
        return AlgorithmResult(name=name, regions=[], elapsed=time.perf_counter() - started, error=repr(exc))
    return AlgorithmResult(name=name, regions=regions, elapsed=time.perf_counter() - started)


def _algo_hakyking_current(path: Path) -> list[SliceRegion]:
    return [
        SliceRegion(
            start=float(audio_slice.start_time),
            end=float(audio_slice.end_time),
            f0_hz=audio_slice.f0_hz,
            midi=_hz_to_midi(audio_slice.f0_hz),
        )
        for audio_slice in parse_audio_slices(str(path))
    ]


def _algo_librosa_split(audio: np.ndarray, sample_rate: int, source_path: str) -> list[SliceRegion]:
    import librosa

    intervals = librosa.effects.split(audio, top_db=35)
    return _regions_from_intervals(intervals, audio, sample_rate, source_path)


def _algo_onset_grid(audio: np.ndarray, sample_rate: int, source_path: str) -> list[SliceRegion]:
    import librosa

    if audio.size == 0:
        return []
    hop_length = max(64, min(256, int(sample_rate * 0.006)))
    onset_env = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        aggregate=np.median,
    )
    frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=hop_length,
        units="frames",
        backtrack=True,
        delta=0.05,
        wait=max(1, int(round(0.06 * sample_rate / hop_length))),
    )
    points = [0]
    for frame in frames:
        sample = int(librosa.frames_to_samples(int(frame), hop_length=hop_length))
        if 0 < sample < audio.size:
            points.append(sample)
    points.append(audio.size)
    intervals = _samples_to_intervals(_dedupe_samples(points, int(0.055 * sample_rate)), audio.size)
    return _regions_from_intervals(intervals, audio, sample_rate, source_path)


def _algo_energy_valley(audio: np.ndarray, sample_rate: int, source_path: str) -> list[SliceRegion]:
    if audio.size == 0:
        return []
    duration = audio.size / sample_rate
    target_count = max(1, min(10, int(round(duration / 0.22))))
    if target_count <= 1:
        return [SliceRegion(0.0, duration, *_estimate_region_pitch(audio, sample_rate))]

    points = [0]
    for index in range(1, target_count):
        sample = int(round(audio.size * index / target_count))
        points.append(_refine_to_energy_valley(audio, sample, sample_rate))
    points.append(audio.size)
    intervals = _samples_to_intervals(_dedupe_samples(points, int(0.06 * sample_rate)), audio.size)
    return _regions_from_intervals(intervals, audio, sample_rate, source_path)


def _algo_hybrid_vote(audio: np.ndarray, sample_rate: int, source_path: str) -> list[SliceRegion]:
    split_regions = _algo_librosa_split(audio, sample_rate, source_path)
    onset_regions = _algo_onset_grid(audio, sample_rate, source_path)
    valley_regions = _algo_energy_valley(audio, sample_rate, source_path)
    candidates = []
    for regions in (split_regions, onset_regions, valley_regions):
        candidates.extend(_internal_boundaries(regions))

    merged = _cluster_boundaries(candidates, tolerance=0.045)
    points = [0, *[int(round(time_value * sample_rate)) for time_value in merged], audio.size]
    intervals = _samples_to_intervals(_dedupe_samples(points, int(0.065 * sample_rate)), audio.size)
    return _regions_from_intervals(intervals, audio, sample_rate, source_path)


def _regions_from_intervals(
    intervals: np.ndarray | list[tuple[int, int]],
    audio: np.ndarray,
    sample_rate: int,
    _source_path: str,
) -> list[SliceRegion]:
    regions: list[SliceRegion] = []
    min_samples = max(1, int(round(0.025 * sample_rate)))
    for start_sample, end_sample in intervals:
        start = max(0, int(start_sample))
        end = min(audio.size, int(end_sample))
        if end - start < min_samples:
            continue
        segment = audio[start:end]
        f0_hz, midi = _estimate_region_pitch(segment, sample_rate)
        regions.append(
            SliceRegion(
                start=start / sample_rate,
                end=end / sample_rate,
                f0_hz=f0_hz,
                midi=midi,
            )
        )
    return regions


def _estimate_region_pitch(segment: np.ndarray, sample_rate: int) -> tuple[float | None, float | None]:
    if segment.size < max(512, int(0.025 * sample_rate)):
        return None, None
    try:
        import librosa

        f0 = librosa.yin(
            np.asarray(segment, dtype=np.float32),
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
        )
    except Exception:
        return None, None
    finite = f0[np.isfinite(f0)]
    finite = finite[(finite > 30.0) & (finite < 4000.0)]
    if finite.size == 0:
        return None, None
    f0_hz = float(np.median(finite))
    return f0_hz, _hz_to_midi(f0_hz)


def _extract_f0_curve(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    if audio.size < 512:
        return np.array([]), np.array([])
    try:
        import librosa

        hop_length = max(128, min(512, int(sample_rate * 0.01)))
        f0 = librosa.yin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
            hop_length=hop_length,
        )
        times = librosa.frames_to_time(np.arange(f0.size), sr=sample_rate, hop_length=hop_length)
    except Exception:
        return np.array([]), np.array([])
    f0 = np.asarray(f0, dtype=np.float64)
    f0[~np.isfinite(f0)] = np.nan
    return np.asarray(times, dtype=np.float64), f0


def _plot_analysis(
    audio: np.ndarray,
    sample_rate: int,
    f0_times: np.ndarray,
    f0_hz: np.ndarray,
    algorithms: list[AlgorithmResult],
    reference: list[tuple[float, float, str]],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    time_axis = np.arange(audio.size, dtype=np.float64) / sample_rate
    figure_height = 2.6 + max(1, len(algorithms)) * 0.68
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, figure_height),
        gridspec_kw={"height_ratios": [2.0, max(1.5, len(algorithms) * 0.42)]},
        sharex=True,
    )
    waveform_axis, boundary_axis = axes
    waveform_axis.plot(time_axis, audio, color="#59d7ff", linewidth=0.8)
    waveform_axis.set_title(title)
    waveform_axis.set_ylabel("wave")
    waveform_axis.grid(True, alpha=0.16)

    if f0_times.size and f0_hz.size:
        pitch_axis = waveform_axis.twinx()
        pitch_axis.plot(f0_times, _hz_to_midi_array(f0_hz), color="#ff9f2f", linewidth=1.0, alpha=0.85)
        pitch_axis.set_ylabel("MIDI F0")

    if reference:
        for start, end, _label in reference:
            waveform_axis.axvspan(start, end, color="#ffffff", alpha=0.04)
            waveform_axis.axvline(start, color="#ffffff", alpha=0.28, linewidth=1.0)
        waveform_axis.axvline(reference[-1][1], color="#ffffff", alpha=0.28, linewidth=1.0)

    row_labels = []
    for row, result in enumerate(algorithms):
        row_y = len(algorithms) - row
        row_labels.append(result.name)
        if result.error:
            boundary_axis.text(0.01, row_y, result.error, color="#ff7b7b", fontsize=8)
            continue
        for region in result.regions:
            boundary_axis.broken_barh(
                [(region.start, max(0.001, region.duration))],
                (row_y - 0.36, 0.72),
                facecolors="#4cc9f0",
                edgecolors="#c8f7ff",
                alpha=0.72,
            )
            if region.duration > 0.08:
                label = "" if region.midi is None else f"{region.midi:.0f}"
                boundary_axis.text(region.start + 0.01, row_y - 0.1, label, fontsize=7, color="#10131a")

    boundary_axis.set_yticks(range(1, len(algorithms) + 1))
    boundary_axis.set_yticklabels(list(reversed(row_labels)))
    boundary_axis.set_xlabel("seconds")
    boundary_axis.grid(True, axis="x", alpha=0.18)
    boundary_axis.set_ylim(0.35, len(algorithms) + 0.65)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _pitch_label(region: SliceRegion) -> str:
    if region.midi is None:
        return "MIDI --"
    return f"MIDI {region.midi:.1f}"


def _write_slice_wavs(
    result: AlgorithmResult,
    audio: np.ndarray,
    sample_rate: int,
    output_dir: Path,
    report_root: Path,
    preview_limit: int = 12,
) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    previews: list[dict[str, str]] = []
    for index, region in enumerate(result.regions, start=1):
        start_sample = max(0, min(audio.size, int(round(region.start * sample_rate))))
        end_sample = max(start_sample, min(audio.size, int(round(region.end * sample_rate))))
        segment = audio[start_sample:end_sample]
        if segment.size == 0:
            continue
        wav_path = output_dir / f"slice_{index:02d}.wav"
        sf.write(str(wav_path), segment, sample_rate)
        if len(previews) < preview_limit:
            previews.append(
                {
                    "name": f"{index:02d}  {region.duration:.2f}s  {_pitch_label(region)}",
                    "path": _relative_web_path(wav_path, report_root),
                }
            )
    return previews


def _write_pitch_tests(
    audio: np.ndarray,
    sample_rate: int,
    output_dir: Path,
    pitch_steps: list[float],
    report_root: Path,
) -> list[dict[str, str]]:
    pitch_dir = output_dir / "pitch_tests"
    pitch_dir.mkdir(parents=True, exist_ok=True)
    tests: list[dict[str, str]] = []
    if audio.size == 0:
        return tests

    source = audio[: min(audio.size, int(round(2.0 * sample_rate)))]
    source_path = pitch_dir / "source.wav"
    sf.write(str(source_path), source, sample_rate)
    tests.append({"name": "source", "path": _relative_web_path(source_path, report_root)})

    for steps in pitch_steps:
        step_label = _step_label(steps)
        step_slug = _step_slug(steps)

        try:
            rendered = process_blob(source, sample_rate, n_steps=steps)
            rendered = _safe_peak(rendered)
            wav_path = pitch_dir / f"hakyking_{step_slug}.wav"
            sf.write(str(wav_path), rendered, sample_rate)
            tests.append({"name": f"hakyking {step_label}", "path": _relative_web_path(wav_path, report_root)})
        except Exception as exc:  # noqa: BLE001
            tests.append({"name": f"hakyking {step_label} failed", "path": "", "error": repr(exc)})

        try:
            import pyrubberband as pyrb

            shifted = pyrb.pitch_shift(source, sample_rate, n_steps=steps)
            shifted = _safe_peak(shifted)
            wav_path = pitch_dir / f"rubberband_{step_slug}.wav"
            sf.write(str(wav_path), shifted, sample_rate)
            tests.append({"name": f"rubberband {step_label}", "path": _relative_web_path(wav_path, report_root)})
        except Exception as exc:  # noqa: BLE001
            tests.append({"name": f"rubberband {step_label} failed", "path": "", "error": repr(exc)})

        try:
            import librosa

            shifted = librosa.effects.pitch_shift(source, sr=sample_rate, n_steps=steps)
            shifted = _safe_peak(shifted)
            wav_path = pitch_dir / f"librosa_{step_slug}.wav"
            sf.write(str(wav_path), shifted, sample_rate)
            tests.append({"name": f"librosa {step_label}", "path": _relative_web_path(wav_path, report_root)})
        except Exception as exc:  # noqa: BLE001
            tests.append({"name": f"librosa {step_label} failed", "path": "", "error": repr(exc)})

    return tests


def _step_label(steps: float) -> str:
    return f"{steps:+g} st"


def _step_slug(steps: float) -> str:
    sign = "plus" if steps >= 0 else "minus"
    value = f"{abs(steps):g}".replace(".", "p")
    return f"{sign}_{value}"


def _relative_web_path(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _render_html_report(
    input_dir: Path,
    output_dir: Path,
    items: list[dict[str, object]],
    elapsed: float,
) -> str:
    rows = []
    for item in items:
        rows.append(_render_file_section(item))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Hakyking Algorithm Lab</title>
<style>
body {{ margin: 0; background: #17181c; color: #d8dce4; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }}
main {{ max-width: 1320px; margin: 0 auto; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 26px; }}
h2 {{ margin: 32px 0 12px; font-size: 20px; color: #ffffff; }}
.meta {{ color: #9aa4b2; margin-bottom: 22px; }}
.card {{ background: #22252b; border: 1px solid #343944; border-radius: 8px; padding: 16px; margin: 18px 0; }}
.plot {{ width: 100%; border-radius: 6px; border: 1px solid #343944; background: #101116; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
th, td {{ border-bottom: 1px solid #363b45; padding: 8px 10px; text-align: left; font-size: 13px; }}
th {{ color: #ffffff; background: #2a2e36; }}
a {{ color: #76d7ff; text-decoration: none; }}
audio {{ width: 260px; height: 30px; }}
.bad {{ color: #ff8b8b; }}
.pill {{ display: inline-block; padding: 2px 7px; border-radius: 999px; background: #303642; color: #b9c6d8; font-size: 12px; margin-right: 6px; }}
.preview-block {{ margin: 10px 0 14px; padding: 10px; background: #1b1d22; border-radius: 6px; }}
.preview-title {{ font-weight: 600; color: #f2f6ff; margin-bottom: 8px; }}
.preview-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 8px 14px; }}
.preview-item {{ display: flex; align-items: center; gap: 8px; min-width: 0; }}
.preview-item audio {{ flex: 1; min-width: 180px; }}
.muted {{ color: #8e98a8; }}
</style>
</head>
<body>
<main>
<h1>Hakyking Algorithm Lab</h1>
<div class="meta">
素材目录：{html.escape(str(input_dir))}<br>
输出目录：{html.escape(str(output_dir))}<br>
文件数：{len(items)} · 耗时：{elapsed:.1f}s
</div>
{''.join(rows)}
</main>
</body>
</html>
"""


def _render_file_section(item: dict[str, object]) -> str:
    title = html.escape(str(item.get("title", "")))
    path = html.escape(str(item.get("path", "")))
    error = str(item.get("error", ""))
    if error:
        return f"<section class='card'><h2>{title}</h2><div class='meta'>{path}</div><p class='bad'>{html.escape(error)}</p></section>"

    plot = html.escape(str(item.get("plot", "")))
    source_clip = html.escape(str(item.get("source_clip", "")))
    duration = float(item.get("duration", 0.0) or 0.0)
    sample_rate = int(item.get("sample_rate", 0) or 0)
    algorithms = item.get("algorithms", [])
    scores = item.get("scores", {})
    pitch_tests = item.get("pitch_tests", [])
    slice_previews = item.get("slice_previews", {})

    algorithm_rows = []
    if isinstance(algorithms, list):
        for result in algorithms:
            if not isinstance(result, AlgorithmResult):
                continue
            score = scores.get(result.name) if isinstance(scores, dict) else None
            algorithm_rows.append(
                "<tr>"
                f"<td>{html.escape(result.name)}</td>"
                f"<td>{len(result.regions)}</td>"
                f"<td>{result.elapsed:.2f}s</td>"
                f"<td>{_score_text(score)}</td>"
                f"<td class='bad'>{html.escape(result.error)}</td>"
                "</tr>"
            )

    pitch_links = []
    if isinstance(pitch_tests, list):
        for test in pitch_tests:
            if not isinstance(test, dict):
                continue
            name = html.escape(str(test.get("name", "")))
            test_path = str(test.get("path", ""))
            if test_path:
                pitch_links.append(
                    f"<div><span class='pill'>{name}</span><audio controls src='{html.escape(test_path)}'></audio></div>"
                )
            else:
                pitch_links.append(
                    f"<div><span class='pill bad'>{name}</span>{html.escape(str(test.get('error', '')))}</div>"
                )

    slice_preview_html = _render_slice_previews(slice_previews)

    return f"""
<section class="card">
<h2>{title}</h2>
<div class="meta">{path}<br>{duration:.2f}s · {sample_rate} Hz · source <a href="{source_clip}">wav</a></div>
<img class="plot" src="{plot}" alt="{title} analysis">
<table>
<thead><tr><th>算法</th><th>切片数</th><th>耗时</th><th>人工标注评分</th><th>错误</th></tr></thead>
<tbody>{''.join(algorithm_rows)}</tbody>
</table>
<h3>\u5207\u7247\u8bd5\u542c</h3>
{slice_preview_html}
<h3>变调试听</h3>
{''.join(pitch_links)}
</section>
"""


def _render_slice_previews(slice_previews: object) -> str:
    if not isinstance(slice_previews, dict) or not slice_previews:
        return "<p class='muted'>no slice previews</p>"

    sections: list[str] = []
    for algorithm_name, previews in slice_previews.items():
        if not isinstance(previews, list) or not previews:
            continue
        items = []
        for preview in previews:
            if not isinstance(preview, dict):
                continue
            name = html.escape(str(preview.get("name", "")))
            wav_path = html.escape(str(preview.get("path", "")))
            if not wav_path:
                continue
            items.append(
                "<div class='preview-item'>"
                f"<span class='pill'>{name}</span>"
                f"<audio controls src='{wav_path}'></audio>"
                "</div>"
            )
        if items:
            sections.append(
                "<div class='preview-block'>"
                f"<div class='preview-title'>{html.escape(str(algorithm_name))}</div>"
                f"<div class='preview-grid'>{''.join(items)}</div>"
                "</div>"
            )
    return "".join(sections) if sections else "<p class='muted'>no slice previews</p>"


def _score_text(score: object) -> str:
    if not isinstance(score, ScoreResult) or score.score is None:
        return "无人工标注"
    error = "--" if score.mean_error_ms is None else f"{score.mean_error_ms:.1f} ms"
    return f"{score.score:.2f} · {score.matched}/{max(score.predicted_boundaries, score.reference_boundaries)} · error {error}"


def _score_regions(
    predicted: list[SliceRegion],
    reference: list[tuple[float, float, str]],
    tolerance: float = 0.030,
) -> ScoreResult:
    reference_boundaries = [end for _start, end, _text in reference[:-1]]
    predicted_boundaries = _internal_boundaries(predicted)
    if not reference_boundaries:
        return ScoreResult(0, len(predicted_boundaries), 0, None, None)

    used = set()
    errors = []
    for predicted_boundary in predicted_boundaries:
        best_index = None
        best_error = tolerance
        for index, reference_boundary in enumerate(reference_boundaries):
            if index in used:
                continue
            error = abs(predicted_boundary - reference_boundary)
            if error <= best_error:
                best_index = index
                best_error = error
        if best_index is not None:
            used.add(best_index)
            errors.append(best_error)

    denominator = max(len(reference_boundaries), len(predicted_boundaries), 1)
    return ScoreResult(
        matched=len(errors),
        predicted_boundaries=len(predicted_boundaries),
        reference_boundaries=len(reference_boundaries),
        mean_error_ms=(float(np.mean(errors) * 1000.0) if errors else None),
        score=len(errors) / denominator,
    )


def _load_annotations(
    explicit_path: Path | None,
    input_dir: Path,
) -> dict[str, list[tuple[float, float, str]]]:
    candidates = [explicit_path] if explicit_path else [input_dir / "annotations.json", ROOT / "test_materials" / "annotations.json"]
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        return _parse_annotations_payload(payload)
    return {}


def _parse_annotations_payload(payload: object) -> dict[str, list[tuple[float, float, str]]]:
    result: dict[str, list[tuple[float, float, str]]] = {}
    items = payload.get("files", payload) if isinstance(payload, dict) else payload
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            file_name = str(item.get("file", ""))
            syllables = item.get("syllables", [])
            if file_name:
                result[file_name] = _parse_syllables(syllables)
    elif isinstance(items, dict):
        for file_name, syllables in items.items():
            result[str(file_name)] = _parse_syllables(syllables)
    return result


def _parse_syllables(payload: object) -> list[tuple[float, float, str]]:
    regions: list[tuple[float, float, str]] = []
    if not isinstance(payload, list):
        return regions
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
        except Exception:
            continue
        if end <= start:
            continue
        regions.append((start, end, str(item.get("text", ""))))
    return sorted(regions, key=lambda value: (value[0], value[1]))


def _reference_for(path: Path, annotations: dict[str, list[tuple[float, float, str]]]) -> list[tuple[float, float, str]]:
    return annotations.get(path.name, annotations.get(str(path), []))


def _scan_audio_files(input_dir: Path, max_files: int) -> list[Path]:
    files = sorted(
        [
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        ],
        key=lambda value: str(value).lower(),
    )
    return files[:max_files] if max_files > 0 else files


def _finite_mono(audio: np.ndarray) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float32)
    if source.ndim > 1:
        source = np.mean(source, axis=-1, dtype=np.float32)
    source = source.copy()
    source[~np.isfinite(source)] = 0.0
    return np.ascontiguousarray(source, dtype=np.float32)


def _safe_peak(audio: np.ndarray) -> np.ndarray:
    source = _finite_mono(audio)
    peak = float(np.max(np.abs(source))) if source.size else 0.0
    if peak > 0.98:
        source = source * (0.98 / peak)
    return np.asarray(source, dtype=np.float32)


def _hz_to_midi(f0_hz: float | None) -> float | None:
    if f0_hz is None or f0_hz <= 0 or not math.isfinite(f0_hz):
        return None
    return 69.0 + 12.0 * math.log2(f0_hz / 440.0)


def _hz_to_midi_array(f0_hz: np.ndarray) -> np.ndarray:
    result = np.full_like(f0_hz, np.nan, dtype=np.float64)
    mask = np.isfinite(f0_hz) & (f0_hz > 0)
    result[mask] = 69.0 + 12.0 * np.log2(f0_hz[mask] / 440.0)
    return result


def _internal_boundaries(regions: list[SliceRegion]) -> list[float]:
    if len(regions) <= 1:
        return []
    return [float(region.end) for region in regions[:-1]]


def _samples_to_intervals(points: list[int], audio_size: int) -> list[tuple[int, int]]:
    points = sorted(set(max(0, min(audio_size, int(point))) for point in points))
    return [(left, right) for left, right in zip(points, points[1:], strict=False) if right > left]


def _dedupe_samples(points: list[int], min_gap: int) -> list[int]:
    if not points:
        return []
    points = sorted(int(point) for point in points)
    deduped = [points[0]]
    for point in points[1:]:
        if point - deduped[-1] < min_gap:
            deduped[-1] = int(round((deduped[-1] + point) / 2))
        else:
            deduped.append(point)
    return deduped


def _cluster_boundaries(boundaries: list[float], tolerance: float) -> list[float]:
    values = sorted(value for value in boundaries if value > 0)
    if not values:
        return []
    clusters = [[values[0]]]
    for value in values[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(np.median(cluster)) for cluster in clusters if len(cluster) >= 2]


def _refine_to_energy_valley(audio: np.ndarray, sample: int, sample_rate: int) -> int:
    before = int(round(sample_rate * 0.035))
    after = int(round(sample_rate * 0.020))
    start = max(0, sample - before)
    end = min(audio.size, sample + after)
    if end - start < 8:
        return max(0, min(audio.size, sample))
    envelope = np.abs(audio[start:end]).astype(np.float64)
    window = max(3, int(round(sample_rate * 0.004)))
    kernel = np.ones(window, dtype=np.float64) / window
    smooth = np.convolve(envelope, kernel, mode="same")
    return start + int(np.argmin(smooth))


def _safe_name(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in name).strip("._")
    return safe[:80] or "audio"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Hakyking slicing and pitch algorithms.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Material folder to scan.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output report folder.")
    parser.add_argument("--annotations", default="", help="Optional annotations.json path.")
    parser.add_argument("--max-files", type=int, default=12, help="Maximum files to analyze. 0 means all.")
    parser.add_argument("--max-duration", type=float, default=8.0, help="Seconds to analyze per file. 0 means full.")
    parser.add_argument(
        "--pitch-steps",
        type=float,
        nargs="+",
        default=[5.0, 12.0, -5.0],
        help="Pitch shift amounts for comparison renders.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
