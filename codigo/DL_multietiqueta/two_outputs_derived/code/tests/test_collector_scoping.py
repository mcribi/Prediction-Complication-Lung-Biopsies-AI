import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import collect_two_output_results as collector
from collect_two_output_results import collect_config, discover_configs


class CollectorScopingTests(unittest.TestCase):
    def test_threshold_validation_tolerates_serialized_tie(self):
        probabilities = np.array([[0.65, 0.20]])
        thresholds = np.array([[0.65, 0.50]])
        stored_predictions = np.array([[0, 0]])

        tolerated = collector.validate_predictions_against_thresholds(
            probabilities,
            thresholds,
            stored_predictions,
            name="tuned",
        )

        self.assertEqual(tolerated, 1)

    def test_threshold_validation_rejects_non_tie_mismatch(self):
        probabilities = np.array([[0.80, 0.20]])
        thresholds = np.array([[0.65, 0.50]])
        stored_predictions = np.array([[0, 0]])

        with self.assertRaisesRegex(
            ValueError,
            "outside serialization tolerance",
        ):
            collector.validate_predictions_against_thresholds(
                probabilities,
                thresholds,
                stored_predictions,
                name="tuned",
            )

    def test_missing_artifacts_are_nonfatal_when_valid_rows_exist(self):
        errors = [
            {
                "config_dir": "/tmp/incomplete",
                "error_type": "FileNotFoundError",
                "error": "Incomplete configuration",
            }
        ]

        self.assertFalse(collector.collection_should_fail(210, errors))

    def test_integrity_error_remains_fatal(self):
        errors = [
            {
                "config_dir": "/tmp/inconsistent",
                "error_type": "ValueError",
                "error": "Metric mismatch",
            }
        ]

        self.assertTrue(collector.collection_should_fail(210, errors))

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
