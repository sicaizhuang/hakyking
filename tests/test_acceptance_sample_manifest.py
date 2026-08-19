from __future__ import annotations

import unittest
from pathlib import Path

from dev_tools.acceptance_samples import DEFAULT_MANIFEST, load_manifest
from hakyking.audio.reader import AudioReader


class AcceptanceSampleManifestTests(unittest.TestCase):
    def test_manifest_has_exactly_the_three_frozen_samples(self) -> None:
        manifest = load_manifest(DEFAULT_MANIFEST)
        samples = list(manifest["samples"])
        self.assertEqual(
            [sample["id"] for sample in samples],
            ["haya_kunalu", "hakimi_x3", "sarilang_long_vocal"],
        )

    def test_sources_exist_and_metadata_matches(self) -> None:
        manifest = load_manifest(DEFAULT_MANIFEST)
        for sample in manifest["samples"]:
            source_path = Path(str(sample["path"]))
            if not source_path.is_absolute():
                source_path = Path(__file__).resolve().parents[1] / source_path
            if not source_path.is_file():
                self.skipTest(
                    "Acceptance audio is optional and is not bundled with the public repository"
                )
            expected = sample["expected"]
            info = AudioReader.read_info(sample["path"])
            self.assertEqual(info.sample_rate, expected["sample_rate"])
            self.assertEqual(info.channels, expected["channels"])
            self.assertAlmostEqual(
                info.duration,
                expected["duration_seconds"],
                delta=expected["duration_tolerance_seconds"],
            )


if __name__ == "__main__":
    unittest.main()
