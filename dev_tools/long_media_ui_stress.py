from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hakyking.controllers.main_controller import MainController  # noqa: E402
from hakyking.models.audio_slice import AudioSlice  # noqa: E402
from hakyking.models.project import ProjectModel  # noqa: E402
from hakyking.qt import QApplication, QEventLoop, QTimer, Qt  # noqa: E402
from hakyking.views.main_window import MainWindow  # noqa: E402


def _default_long_sample() -> str:
    manifest = json.loads(
        (ROOT / "qa" / "acceptance_samples.json").read_text(encoding="utf-8")
    )
    for sample in manifest["samples"]:
        if sample.get("role") == "long_vocal_reference":
            return str(sample["path"])
    raise RuntimeError("No long_vocal_reference in acceptance manifest")


def _pump_until(app: QApplication, predicate, timeout: float) -> bool:  # noqa: ANN001
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        app.processEvents(QEventLoop.AllEvents, 20)
        if predicate():
            return True
        time.sleep(0.002)
    return bool(predicate())


def _material_block_count(window: MainWindow) -> int:
    return sum(
        isinstance(item.data(Qt.UserRole), AudioSlice)
        for item in window.material_browser.slice_list.scene().items()
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exercise Hakyking's long-media controller and batched Qt pipeline."
    )
    parser.add_argument("path", nargs="?", default=_default_long_sample())
    parser.add_argument(
        "--report",
        default=str(ROOT / "qa_artifacts" / "long_media_ui_stress_latest.md"),
    )
    args = parser.parse_args()

    source = str(Path(args.path).expanduser().resolve(strict=True))
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    controller = MainController(ProjectModel(), window)
    controller.autosave_path = ROOT / "qa_artifacts" / "long_media_ui_stress_autosave.haky"
    controller.initialize()
    window.resize(1536, 912)
    window.show()
    app.processEvents()

    tick_times: list[float] = []
    tick_timer = QTimer()
    tick_timer.setInterval(10)
    tick_timer.timeout.connect(lambda: tick_times.append(time.perf_counter()))
    tick_timer.start()

    try:
        window.material_browser.load_material_into_preview(source)
        parse_started = time.perf_counter()
        controller._start_parse_worker(source, display_result=True, preparse=False)
        if not _pump_until(app, lambda: source not in controller._parse_inflight, 15.0):
            raise RuntimeError("long-media parse did not finish within 15 seconds")
        parse_seconds = time.perf_counter() - parse_started
        slices = controller._slice_cache.get(source, [])
        if not slices:
            raise RuntimeError("long-media parse returned no slices")

        render_started = time.perf_counter()
        if not _pump_until(
            app,
            lambda: _material_block_count(window) == len(slices),
            5.0,
        ):
            raise RuntimeError("source editor did not finish batched slice rendering")
        material_render_seconds = time.perf_counter() - render_started

        undo_before = window.undo_stack.count()
        insert_started = time.perf_counter()
        controller._on_audio_file_dropped_as_slice(source, 0.0, 220.0, 1)
        if not _pump_until(
            app,
            lambda: len(window.workspace.slice_items()) == len(slices),
            5.0,
        ):
            raise RuntimeError("workspace did not finish batched slice insertion")
        workspace_insert_seconds = time.perf_counter() - insert_started
        if window.undo_stack.count() != undo_before + 1:
            raise RuntimeError("batched workspace insertion was not one undo command")

        window.workspace.set_pitch_curve_edit_mode(True)
        controller._on_pitch_curve_view_changed(True)
        visible_items = window.workspace.visible_slice_items()
        if not visible_items:
            raise RuntimeError("workspace reported no visible long-media slices")
        refinement_started = time.perf_counter()
        _pump_until(
            app,
            lambda: any(item.base_audio_cache is not None for item in visible_items),
            8.0,
        )
        refinement_seconds = time.perf_counter() - refinement_started
        refined_visible = sum(item.base_audio_cache is not None for item in visible_items)
        queued_waveforms = len(controller._waveform_queue) + len(
            controller._waveform_active_keys
        )
        if queued_waveforms > len(visible_items) + 2:
            raise RuntimeError(
                f"waveform queue escaped viewport bounds: {queued_waveforms} > {len(visible_items) + 2}"
            )

        max_tick_gap = max(
            (right - left for left, right in zip(tick_times, tick_times[1:])),
            default=0.0,
        )
        if len(tick_times) < 5:
            raise RuntimeError("Qt event loop did not remain responsive")
        if max_tick_gap > 0.75:
            raise RuntimeError(f"Qt event loop stalled for {max_tick_gap:.3f}s")

        lines = [
            "# Hakyking Long Media UI Stress",
            "",
            f"- Source: `{source}`",
            f"- Slices: {len(slices)}",
            f"- Controller parse/cache: {parse_seconds:.3f}s",
            f"- Batched source-editor render: {material_render_seconds:.3f}s",
            f"- Batched workspace insert: {workspace_insert_seconds:.3f}s",
            f"- Visible slices: {len(visible_items)}",
            f"- Visible slices refined during window: {refined_visible}",
            f"- Refinement observation window: {refinement_seconds:.3f}s",
            f"- Pending/active waveform jobs: {queued_waveforms}",
            f"- Event-loop ticks: {len(tick_times)}",
            f"- Maximum tick gap: {max_tick_gap:.3f}s",
            f"- Undo commands added by full import: {window.undo_stack.count() - undo_before}",
            "",
        ]
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return 0
    finally:
        tick_timer.stop()
        controller.shutdown()
        window.close()
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
