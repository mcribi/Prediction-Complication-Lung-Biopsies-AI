import os
import tempfile
import unittest
from pathlib import Path

from collect_two_output_results import collect_config, discover_configs


class CollectorScopingTests(unittest.TestCase):
    def test_discovery_excludes_historical_errors_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_error = root / "old_run" / "error.txt"
            new_error = root / "new_run" / "error.txt"
            old_error.parent.mkdir(parents=True)
            new_error.parent.mkdir(parents=True)
            old_error.write_text("old", encoding="utf-8")
            new_error.write_text("new", encoding="utf-8")
            os.utime(old_error, (100.0, 100.0))
            os.utime(new_error, (300.0, 300.0))

            observed = discover_configs(root, min_mtime_epoch=200.0)

            self.assertEqual(observed, [new_error.parent])

    def test_collection_rejects_required_artifacts_from_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            config.mkdir()
            required = [
                config / "fold_summary.csv",
                config / "oof_metrics_fixed_0_5.json",
                config / "oof_metrics_tuned.json",
                config / "oof_predictions.csv",
            ]
            for path in required:
                path.write_text("placeholder", encoding="utf-8")
                os.utime(path, (100.0, 100.0))

            with self.assertRaisesRegex(ValueError, "mixes artifacts"):
                collect_config(config, min_mtime_epoch=200.0)


if __name__ == "__main__":
    unittest.main()
