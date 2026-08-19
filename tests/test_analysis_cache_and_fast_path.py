from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from hakyking.audio.analysis_cache import load_cached_slices, store_cached_slices
from hakyking.audio.slicer import parse_audio_slices
from hakyking.audio.vocal_analysis import (
    PitchTrack,
    cache_source_pitch_track,
    cached_source_pitch_track,
    clear_source_pitch_cache,
)
from hakyking.audio.waveform import build_waveform_result
from hakyking.controllers.audio_worker import ParseWorker
from hakyking.models.audio_slice import AudioSlice, copy_audio_slice


class AnalysisCacheAndFastPathTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_source_pitch_cache()

    def tearDown(self) -> None:
        clear_source_pitch_cache()

    def test_cache_round_trip_restores_slices_and_source_pitch_track(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            sf.write(source, self._sine_audio(1.0), 16000)
            slices = [AudioSlice(str(source), 0, 0.0, 1.0, 57, 220.0, 0.7, "test")]
            track = PitchTrack(
                times=np.linspace(0.0, 1.0, 51),
                f0_hz=np.full(51, 220.0, dtype=np.float32),
                confidence=np.full(51, 0.7, dtype=np.float32),
                backend="test",
            )
            cache_source_pitch_track(str(source), track)
            store_cached_slices(str(source), slices, cache_dir=root / "cache")

            clear_source_pitch_cache()
            restored = load_cached_slices(str(source), cache_dir=root / "cache")

            self.assertEqual(restored, slices)
            restored_track = cached_source_pitch_track(str(source))
            self.assertIsNotNone(restored_track)
            self.assertEqual(restored_track.backend, "test")
            np.testing.assert_allclose(restored_track.f0_hz, track.f0_hz)

    def test_cache_invalidates_when_source_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            sf.write(source, self._sine_audio(0.25), 16000)
            slices = [AudioSlice(str(source), 0, 0.0, 0.25, 57, 220.0)]
            store_cached_slices(str(source), slices, cache_dir=root / "cache")
            source.write_bytes(source.read_bytes() + b"changed")
            self.assertIsNone(
                load_cached_slices(str(source), cache_dir=root / "cache")
            )

    def test_waveform_reuses_source_pitch_track(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            sf.write(source, self._sine_audio(0.5), 16000)
            audio_slice = AudioSlice(str(source), 0, 0.0, 0.5, 57, 220.0)
            cache_source_pitch_track(
                str(source),
                PitchTrack(
                    times=np.linspace(0.0, 0.5, 26),
                    f0_hz=np.full(26, 220.0, dtype=np.float32),
                    confidence=np.full(26, 0.7, dtype=np.float32),
                    backend="test",
                ),
            )
            with patch(
                "hakyking.audio.waveform.compute_pitch_contour",
                side_effect=AssertionError("pitch should come from the source cache"),
            ):
                result = build_waveform_result("cached", audio_slice, max_points=32)
            self.assertGreater(result.envelope.size, 0)
            self.assertGreater(result.pitch_contour.size, 0)

    def test_visible_refinement_can_bypass_fast_source_pitch_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            sf.write(source, self._sine_audio(0.5), 16000)
            audio_slice = AudioSlice(
                str(source), 0, 0.0, 0.5, 57, 220.0, analysis_backend="librosa_yin_fast"
            )
            with patch(
                "hakyking.audio.waveform.compute_pitch_contour",
                return_value=np.asarray([57.0, 57.1, 56.9], dtype=np.float32),
            ) as precise:
                result = build_waveform_result(
                    "precise",
                    audio_slice,
                    max_points=32,
                    prefer_cached_pitch=False,
                )
            precise.assert_called_once()
            np.testing.assert_allclose(result.pitch_contour, [57.0, 57.1, 56.9])

    def test_parse_worker_honors_cancel_before_start(self) -> None:
        worker = ParseWorker("cancelled.wav")
        finished: list[object] = []
        completed: list[bool] = []
        worker.finished.connect(lambda *_args: finished.append(True))
        worker.completed.connect(lambda: completed.append(True))
        worker.cancel()
        worker.run()
        self.assertEqual(finished, [])
        self.assertEqual(completed, [True])

    def test_short_parse_does_not_repeat_pitch_detection_per_slice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            audio = np.concatenate(
                [
                    self._sine_audio(0.28),
                    np.zeros(1600, dtype=np.float32),
                    self._sine_audio(0.28, frequency=247.0),
                ]
            )
            sf.write(source, audio, 16000)
            calls = 0

            def annotate(path, _audio, _sample_rate, slices):
                nonlocal calls
                calls += 1
                return [
                    copy_audio_slice(
                        item,
                        midi_note=57,
                        f0_hz=220.0,
                        pitch_confidence=0.7,
                        analysis_backend="test",
                    )
                    for item in slices
                ]

            with (
                patch(
                    "hakyking.audio.vocal_analysis.annotate_slices_with_vocal_analysis",
                    side_effect=annotate,
                ),
                patch(
                    "hakyking.audio.slicer._estimate_average_f0",
                    side_effect=AssertionError("per-slice F0 regression"),
                ),
            ):
                slices = parse_audio_slices(str(source))
            self.assertGreaterEqual(len(slices), 2)
            self.assertEqual(calls, 1)

    def test_long_media_uses_fast_pitch_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "long.wav"
            sf.write(source, self._sine_audio(1.25, sample_rate=44100), 44100)
            with patch(
                "hakyking.audio.slicer.LONG_AUDIO_FAST_THRESHOLD_SECONDS",
                1.0,
            ):
                slices = parse_audio_slices(str(source))
            self.assertTrue(slices)
            self.assertAlmostEqual(slices[-1].end_time, 1.25, delta=0.01)
            self.assertEqual(
                {item.analysis_backend for item in slices},
                {"librosa_yin_fast"},
            )

    def test_parse_worker_uses_valid_disk_cache(self) -> None:
        cached = [AudioSlice("cached.wav", 0, 0.0, 0.5, 57, 220.0)]
        result: dict[str, object] = {}
        worker = ParseWorker("cached.wav")
        worker.finished.connect(
            lambda path, slices: result.update({"path": path, "slices": list(slices)})
        )
        worker.failed.connect(lambda message: result.update({"error": message}))
        with (
            patch(
                "hakyking.controllers.audio_worker.load_cached_slices",
                return_value=cached,
            ),
            patch(
                "hakyking.controllers.audio_worker.parse_audio_slices",
                side_effect=AssertionError("cache hit must not re-run analysis"),
            ),
        ):
            worker.run()
        self.assertNotIn("error", result)
        self.assertEqual(result.get("path"), "cached.wav")
        self.assertEqual(result.get("slices"), cached)

    @staticmethod
    def _sine_audio(
        duration: float,
        frequency: float = 220.0,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        times = np.arange(int(round(duration * sample_rate))) / sample_rate
        return (0.2 * np.sin(2.0 * np.pi * frequency * times)).astype(np.float32)


if __name__ == "__main__":
    unittest.main()
