import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aggregate_radiomics_subgroups import collect_completed


class TestAggregation(unittest.TestCase):
    def test_aggregation_rejects_missing_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = pd.DataFrame(
                [
                    {"task_id": 0, "rule_id": "r1", "model_name": "m1", "output_dir": str(root / "t0")},
                    {"task_id": 1, "rule_id": "r2", "model_name": "m1", "output_dir": str(root / "t1")},
                ]
            )
            (root / "t0").mkdir()
            (root / "t0" / "COMPLETED.json").write_text(json.dumps({"status": "ok", "num_folds_completed": 5, "all_folds_present": True, "n_subgroup": 10}))
            (root / "t0" / "oof_metrics.json").write_text(json.dumps({"general_tuned": {"two_outputs_f1_micro": 0.4}, "specialized_tuned": {"two_outputs_f1_micro": 0.5}}))
            with self.assertRaisesRegex(ValueError, "Incomplete tasks"):
                collect_completed(manifest)

    def test_aggregation_returns_one_row_per_complete_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for task_id in range(2):
                output = root / f"t{task_id}"
                output.mkdir()
                (output / "COMPLETED.json").write_text(json.dumps({"status": "ok", "num_folds_completed": 5, "all_folds_present": True, "n_subgroup": 10 + task_id}))
                (output / "oof_metrics.json").write_text(json.dumps({"general_tuned": {"two_outputs_f1_micro": 0.4}, "specialized_tuned": {"two_outputs_f1_micro": 0.5 + task_id}}))
                rows.append({"task_id": task_id, "rule_id": f"r{task_id}", "model_name": "m1", "output_dir": str(output)})
            summary = collect_completed(pd.DataFrame(rows))
            self.assertEqual(len(summary), 2)
            self.assertIn("delta_tuned_two_outputs_f1_micro", summary.columns)
            self.assertAlmostEqual(summary.iloc[0]["delta_tuned_two_outputs_f1_micro"], 0.1)


if __name__ == "__main__":
    unittest.main()
