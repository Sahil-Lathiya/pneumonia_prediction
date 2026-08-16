import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Pnuemonia_detection_project.ipynb"
RESULTS = ROOT / "results" / "reported_metrics.json"


class RepositoryIntegrityTests(unittest.TestCase):
    def test_reported_metrics_have_expected_provenance(self):
        payload = json.loads(RESULTS.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["status"], "archived_notebook_output_not_reproduced_in_ci"
        )
        self.assertEqual(payload["test_images"], 624)
        self.assertEqual(
            [item["model"] for item in payload["models"]],
            ["ResNet50", "MobileNetV2", "Custom CNN"],
        )
        self.assertEqual(
            [item["accuracy_percent"] for item in payload["models"]],
            [78.04, 90.54, 83.49],
        )

    def test_notebook_is_portable_and_cleared(self):
        payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        serialized = json.dumps(payload).lower()
        self.assertNotIn("d:\\ai & ml", serialized)
        self.assertTrue(all(not cell.get("outputs") for cell in payload["cells"]))
        self.assertTrue(
            all(
                cell.get("execution_count") is None
                for cell in payload["cells"]
                if cell["cell_type"] == "code"
            )
        )

    def test_config_import_has_no_filesystem_side_effects(self):
        with TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "data"
            env = os.environ.copy()
            env["PNEUMONIA_DATASET_DIR"] = str(dataset_path)
            completed = subprocess.run(
                [sys.executable, "-c", "import config; print(config.DATASET_DIR)"],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(completed.stdout.strip(), str(dataset_path.resolve()))
            self.assertFalse((ROOT / "saved_models").exists())
            self.assertFalse((ROOT / "report_charts").exists())

    def test_readme_rejects_unsupported_performance_claims(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        for unsupported in (
            "95.64%",
            "93.33%",
            "production-ready",
            "eu ai act compliant",
        ):
            self.assertNotIn(unsupported, readme)
        self.assertIn("recorded outputs from one archived local run", readme)
        self.assertIn("not a medical device", readme)


if __name__ == "__main__":
    unittest.main()
