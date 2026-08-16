import math
import unittest

import numpy as np
import pandas as pd

from pu_nested_grid import (
    binary_metrics,
    choose_threshold,
    clinical_feature_filter,
    estimate_c_from_oof,
    paired_seed,
    validate_outer_folds,
)


class TestPUNestedGrid(unittest.TestCase):
    def test_clinical_filter_excludes_sex_and_retains_age(self):
        columns = ["patient_id", "Edad", "Sexo_binaria", "tamano_nodulo_mm", "rad_feature"]
        selected = clinical_feature_filter(columns)
        self.assertIn("Edad", selected)
        self.assertNotIn("Sexo_binaria", selected)
        self.assertNotIn("patient_id", selected)

    def test_feature_filter_excludes_image_metadata(self):
        columns = ["patient_id", "image_path", "image_size", "image_spacing", "Edad", "lung_original_firstorder_Mean"]
        selected = clinical_feature_filter(columns)
        self.assertEqual(selected, ["Edad", "lung_original_firstorder_Mean"])

    def test_binary_metrics_include_consistent_gmean_and_confusion_counts(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        y_score = np.array([0.9, 0.4, 0.7, 0.1])
        result = binary_metrics(y_true, y_pred, y_score)
        self.assertEqual((result["tp"], result["fn"], result["fp"], result["tn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(result["sensitivity"], 0.5)
        self.assertAlmostEqual(result["specificity"], 0.5)
        self.assertAlmostEqual(result["gmean"], 0.5)
        self.assertIn("mcc", result)
        self.assertIn("average_precision", result)
        self.assertIn("brier_score", result)

    def test_threshold_is_selected_only_from_supplied_inner_oof_values(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([0.1, 0.4, 0.6, 0.9])
        threshold, metrics = choose_threshold(y, score)
        self.assertGreater(threshold, 0.4)
        self.assertLessEqual(threshold, 0.6)
        self.assertAlmostEqual(metrics["gmean"], 1.0)

    def test_c_estimate_uses_only_oof_positive_scores(self):
        y = np.array([1, 0, 1, 0])
        score = np.array([0.4, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(estimate_c_from_oof(y, score), 0.6)

    def test_paired_seed_does_not_depend_on_formulation(self):
        a = paired_seed(base_seed=42, outer_fold=3, model_family="random_forest", candidate_index=2)
        b = paired_seed(base_seed=42, outer_fold=3, model_family="random_forest", candidate_index=2)
        self.assertEqual(a, b)

    def test_outer_fold_validator_requires_five_disjoint_complete_tests(self):
        ids = [f"p{i}" for i in range(10)]
        rows = []
        for fold in range(1, 6):
            test = set(ids[2 * (fold - 1):2 * fold])
            for pid in ids:
                rows.append({"fold": fold, "split": "test" if pid in test else "train", "patient_id": pid})
        folds = pd.DataFrame(rows)
        validate_outer_folds(folds, expected_ids=set(ids), n_splits=5)

        broken = folds[~((folds["fold"] == 5) & (folds["split"] == "test"))]
        with self.assertRaises(ValueError):
            validate_outer_folds(broken, expected_ids=set(ids), n_splits=5)


if __name__ == "__main__":
    unittest.main()
