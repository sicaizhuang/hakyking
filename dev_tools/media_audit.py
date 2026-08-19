from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hakyking.audio.reader import AudioReader  # noqa: E402


@dataclass(frozen=True)
class ProbeResult:
    path: Path
    status: str
    detail: str
    elapsed: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit real media compatibility for Hakyking.")
    parser.add_argument("root", help="Folder to scan")
    parser.add_argument("--per-ext", type=int, default=5, help="Samples to probe per extension")
    parser.add_argument(
        "--report",
        default=str(ROOT / "qa_artifacts" / "media_audit_latest.md"),
        help="Markdown report path",
    )
    args = parser.parse_args()

    root = Path(args.root)
    report_path = Path(args.report)
    if not root.exists():
        raise SystemExit(f"Folder does not exist: {root}")

    files = list(_iter_files(root))
    extension_counts = Counter(path.suffix.lower() or "<none>" for path in files)
    samples = _sample_supported_files(files, max(1, args.per_ext))
    results = [_probe(path) for path in samples]
    _write_report(root, extension_counts, results, report_path)
    _print_summary(root, extension_counts, results, report_path)
    return 1 if any(result.status == "FAIL" for result in results) else 0


def _iter_files(root: Path):
    for path in root.rglob("*"):
        try:
            if path.is_file():
                yield path
        except OSError:
            continue


def _sample_supported_files(files: list[Path], per_ext: int) -> list[Path]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        suffix = path.suffix.lower()
        if suffix in AudioReader.supported_extensions:
            grouped[suffix].append(path)
    samples: list[Path] = []
    for suffix in sorted(grouped):
        samples.extend(sorted(grouped[suffix], key=lambda item: (item.stat().st_size, str(item)))[:per_ext])
    return samples


def _probe(path: Path) -> ProbeResult:
    start = time.perf_counter()
    try:
        if path.suffix.lower() in AudioReader.video_extensions and not AudioReader.has_audio_stream(str(path)):
            return ProbeResult(path, "SKIP", "video has no audio stream", time.perf_counter() - start)
        info = AudioReader.read_info(str(path))
    except Exception as exc:  # noqa: BLE001 - audit should report every media failure
        return ProbeResult(path, "FAIL", repr(exc), time.perf_counter() - start)
    detail = (
        f"{info.sample_rate} Hz, {info.channels} ch, "
        f"{info.duration:.2f}s, {path.stat().st_size / (1024 * 1024):.2f} MiB"
    )
    return ProbeResult(path, "PASS", detail, time.perf_counter() - start)


def _write_report(
    root: Path,
    extension_counts: Counter[str],
    results: list[ProbeResult],
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pass_count = sum(1 for result in results if result.status == "PASS")
    fail_count = sum(1 for result in results if result.status == "FAIL")
    skip_count = sum(1 for result in results if result.status == "SKIP")
    lines = [
        "# Hakyking Real Media Audit",
        "",
        f"- Root: `{root}`",
        f"- Supported sample probes: PASS={pass_count} FAIL={fail_count} SKIP={skip_count}",
        "",
        "## Extension Counts",
        "",
        "| Extension | Count |",
        "| --- | ---: |",
    ]
    for suffix, count in extension_counts.most_common():
        lines.append(f"| `{suffix}` | {count} |")
    lines.extend(
        [
            "",
            "## Sample Probe Results",
            "",
            "| Status | Time | File | Detail |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for result in results:
        detail = result.detail.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {result.status} | {result.elapsed:.2f}s | `{result.path}` | {detail} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(
    root: Path,
    extension_counts: Counter[str],
    results: list[ProbeResult],
    report_path: Path,
) -> None:
    pass_count = sum(1 for result in results if result.status == "PASS")
    fail_count = sum(1 for result in results if result.status == "FAIL")
    skip_count = sum(1 for result in results if result.status == "SKIP")
    supported_count = sum(
        count
        for suffix, count in extension_counts.items()
        if suffix in AudioReader.supported_extensions
    )
    print(f"root={root}")
    print(f"files={sum(extension_counts.values())} supported={supported_count}")
    print(f"sample_probe PASS={pass_count} FAIL={fail_count} SKIP={skip_count}")
    print(f"report={report_path}")


if __name__ == "__main__":
    raise SystemExit(main())
