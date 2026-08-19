from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hakyking.models.project import ProjectModel
from hakyking.project_manager import PROJECT_FORMAT, PROJECT_VERSION, ProjectManager
from hakyking.qt import QApplication
from hakyking.views.workspace import WorkspaceView


class ProjectReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.workspace = WorkspaceView()
        self.project = ProjectModel(title="Reliability")
        self.project.bootstrap_default_tracks()
        self.manager = ProjectManager()

    def tearDown(self) -> None:
        self.workspace.close()

    def test_atomic_save_keeps_previous_backup_and_recovers_from_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "project.haky"
            self.manager.save(path, self.project, self.workspace)
            self.project.title = "Second"
            self.manager.save(path, self.project, self.workspace)

            backup_path = self.manager.backup_path(path)
            self.assertTrue(backup_path.is_file())
            path.write_text("{broken", encoding="utf-8")

            loaded = self.manager.load(path)
            self.assertEqual(loaded.project.title, "Reliability")
            self.assertEqual(loaded.recovered_from, str(backup_path))

    def test_future_project_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future.haky"
            path.write_text(
                json.dumps(
                    {
                        "format": PROJECT_FORMAT,
                        "version": PROJECT_VERSION + 1,
                        "project": {},
                        "tracks": [],
                        "slices": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsupported .haky project version"):
                self.manager.load(path)


if __name__ == "__main__":
    unittest.main()
