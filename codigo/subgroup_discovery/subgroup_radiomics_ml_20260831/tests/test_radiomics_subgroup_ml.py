import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from radiomics_subgroup_ml import (
    RULES,
    apply_rule,
    choose_threshold,
    derive_no_complication,
    run_experiment,
    save_result,
    validate_outer_splits,
)


class TestRadiomicsSubgroupML(unittest.TestCase):
    def test_all_six_rules_are_registered(self):
        self.assertEqual(
            set(RULES),
            {
                "hemorrhage_small_nodule",
                "pneumothorax_shallow_small_nodule",
                "no_comp_no_lung_pathology",
                "no_comp_large_nodule",
                "hemorrhage_age_over_60",
                "pneumothorax_tobacco_any",
            },
        )

    def test_numeric_rule_boundaries_and_missing_values(self):
        frame = pd.DataFrame(
            {
                "Edad": [60.0, 60.01, np.nan, 75.0],
                "tamano_nodulo_mm": [31.05, 42.50, 42.80, np.nan],
                "profundidad_centroidal_pleura_mm": [10.21, 10.21, 10.22, 1.0],
                "Sin_patología_pulmonar": [0, 1, 1, np.nan],
                "Tabac_any": [0, 1, np.nan, 1],
            }
        )
        self.assertEqual(apply_rule(frame, "hemorrhage_small_nodule").tolist(), [True, False, False, False])
        self.assertEqual(apply_rule(frame, "pneumothorax_shallow_small_nodule").tolist(), [True, True, False, False])
        self.assertEqual(apply_rule(frame, "no_comp_no_lung_pathology").tolist(), [False, True, True, False])
        self.assertEqual(apply_rule(frame, "no_comp_large_nodule").tolist(), [False, False, True, False])
        self.assertEqual(apply_rule(frame, "hemorrhage_age_over_60").tolist(), [False, True, False, True])
        self.assertEqual(apply_rule(frame, "pneumothorax_tobacco_any").tolist(), [False, True, False, True])

    def test_no_complication_is_derived_from_two_outputs(self):
        pred = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        self.assertEqual(derive_no_complication(pred).tolist(), [1, 0, 0, 0])

    def test_threshold_tie_prefers_closest_to_half_then_lower(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        threshold, _ = choose_threshold(y, scores, [0.4, 0.5, 0.6])
        self.assertEqual(threshold, 0.5)

    def test_outer_splits_require_five_folds_and_one_test_assignment_per_patient(self):
        rows = []
        ids = [f"P{i}" for i in range(10)]
        for fold in range(1, 6):
            test = {ids[2 * (fold - 1)], ids[2 * (fold - 1) + 1]}
            for patient_id in ids:
                rows.append({"fold": fold, "split": "test" if patient_id in test else "train", "patient_id": patient_id})
        validated = validate_outer_splits(pd.DataFrame(rows), ids)
        self.assertEqual(sorted(validated), [1, 2, 3, 4, 5])
        self.assertEqual(sum(len(v["test"]) for v in validated.values()), len(ids))

    def test_outer_splits_reject_duplicate_test_assignment(self):
        rows = []
        ids = [f"P{i}" for i in range(10)]
        for fold in range(1, 6):
            test = {ids[2 * (fold - 1)], ids[2 * (fold - 1) + 1]}
            if fold == 2:
                test.add("P0")
            for patient_id in ids:
                rows.append({"fold": fold, "split": "test" if patient_id in test else "train", "patient_id": patient_id})
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_outer_splits(pd.DataFrame(rows), ids)

    def test_end_to_end_experiment_returns_complete_matched_oof(self):
        rng = np.random.RandomState(7)
        ids = [f"P{i:02d}" for i in range(50)]
        data = pd.DataFrame(
            {
                "patient_id": ids,
                "f1": rng.normal(size=50),
                "f2": rng.normal(size=50),
                "f3": rng.normal(size=50),
                "Edad": [65 if i % 2 else 55 for i in range(50)],
                "Tabac_any": [i % 2 for i in range(50)],
                "Sin_patología_pulmonar": [(i + 1) % 2 for i in range(50)],
                "tamano_nodulo_mm": [20 + (i % 20) for i in range(50)],
                "profundidad_centroidal_pleura_mm": [5 + (i % 10) for i in range(50)],
                "Hemorragia": [i % 3 == 0 for i in range(50)],
                "Neumotórax": [i % 4 == 0 for i in range(50)],
            }
        )
        rows = []
        for fold in range(1, 6):
            test = set(ids[10 * (fold - 1) : 10 * fold])
            for patient_id in ids:
                rows.append({"fold": fold, "split": "test" if patient_id in test else "train", "patient_id": patient_id})
        result = run_experiment(
            data=data,
            feature_columns=["f1", "f2", "f3"],
            outer_splits=validate_outer_splits(pd.DataFrame(rows), ids),
            rule_id="hemorrhage_age_over_60",
            model_name="dummy",
            seed=7,
        )
        predictions = result["predictions"]
        expected_ids = set(data.loc[data["Edad"] > 60, "patient_id"])
        self.assertEqual(set(predictions["patient_id"]), expected_ids)
        self.assertEqual(len(predictions), len(expected_ids))
        self.assertEqual(sorted(predictions["fold"].unique()), [1, 2, 3, 4, 5])
        self.assertEqual(
            set(result["metrics"]),
            {"general_fixed", "general_tuned", "specialized_fixed", "specialized_tuned"},
        )
        self.assertEqual(len(result["fold_metrics"]), 5 * 4)

    def test_completed_marker_requires_all_five_folds(self):
        minimal = {
            "rule_id": "hemorrhage_age_over_60",
            "model_name": "dummy",
            "n_subgroup": 10,
            "folds_completed": [1],
            "predictions": pd.DataFrame([{"patient_id": "P1", "fold": 1}]),
            "thresholds": pd.DataFrame([{"fold": 1}]),
            "fold_metrics": pd.DataFrame([{"fold": 1}]),
            "metrics": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "five folds"):
                save_result(minimal, Path(directory), {"test": True})
            minimal["folds_completed"] = [1, 2, 3, 4, 5]
            save_result(minimal, Path(directory), {"test": True})
            self.assertTrue((Path(directory) / "COMPLETED.json").exists())
            completion = __import__("json").loads((Path(directory) / "COMPLETED.json").read_text())
            self.assertTrue(completion["all_folds_present"])


if __name__ == "__main__":
    unittest.main()
