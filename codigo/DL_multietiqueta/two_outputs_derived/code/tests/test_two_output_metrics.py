import unittest

import numpy as np

from two_output_metrics import (
    DEFAULT_THRESHOLD,
    THRESHOLD_GRID,
    apply_thresholds,
    compute_metrics_from_predictions,
    compute_two_output_and_derived_metrics,
    derive_no_complication,
    select_per_label_thresholds,
)


class TwoOutputMetricsTests(unittest.TestCase):
    def test_rejects_non_binary_predictions(self):
        targets = np.array([[0, 1], [1, 0]])
        probabilities = np.array([[0.2, 0.8], [0.7, 0.3]])
        predictions = np.array([[0.2, 1.0], [1.0, 0.0]])
        with self.assertRaises(ValueError):
            compute_metrics_from_predictions(targets, probabilities, predictions)

    def test_rejects_probabilities_outside_unit_interval(self):
        probabilities = np.array([[0.2, 1.1], [0.7, 0.3]])
        with self.assertRaises(ValueError):
            apply_thresholds(probabilities, [0.5, 0.5])

    def setUp(self):
        self.targets = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 0]])
        self.probs = np.asarray(
            [
                [0.20, 0.10],
                [0.45, 0.15],
                [0.25, 0.40],
                [0.80, 0.75],
                [0.35, 0.20],
                [0.60, 0.30],
            ]
        )

    def test_derived_state_is_one_only_for_zero_zero(self):
        observed = derive_no_complication(self.targets)
        np.testing.assert_array_equal(observed, np.asarray([1, 0, 0, 0, 1, 0]))

    def test_thresholds_are_applied_independently(self):
        observed = apply_thresholds(self.probs, [0.40, 0.35])
        np.testing.assert_array_equal(
            observed,
            np.asarray([[0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 0]]),
        )

    def test_threshold_selection_uses_grid_and_marks_two_selected_rows(self):
        selection = select_per_label_thresholds(self.targets, self.probs, THRESHOLD_GRID)
        self.assertEqual(selection.thresholds.shape, (2,))
        self.assertTrue(np.isin(selection.thresholds, THRESHOLD_GRID).all())
        self.assertEqual(sum(row["selected"] for row in selection.curve_rows), 2)

    def test_ties_prefer_threshold_closest_to_point_five(self):
        targets = np.asarray([[0, 0], [1, 1]])
        probs = np.asarray([[0.1, 0.1], [0.9, 0.9]])
        selection = select_per_label_thresholds(targets, probs, [0.3, 0.5, 0.7])
        np.testing.assert_allclose(selection.thresholds, [DEFAULT_THRESHOLD, DEFAULT_THRESHOLD])

    def test_metrics_include_two_outputs_and_derived_state(self):
        metrics, predictions, derived = compute_two_output_and_derived_metrics(
            self.targets,
            self.probs,
            [0.40, 0.35],
        )
        self.assertEqual(predictions.shape, self.targets.shape)
        np.testing.assert_array_equal(derived, derive_no_complication(predictions))
        self.assertEqual(metrics["two_outputs_f1_micro"], 1.0)
        self.assertEqual(metrics["clinical_three_states_f1_micro"], 1.0)
        self.assertEqual(metrics["f1_Sin_complicacion"], 1.0)
        for key in [
            "specificity_Hemorragia",
            "roc_auc_Neumotórax",
            "average_precision_Hemorragia",
            "clinical_three_states_hamming_loss",
            "two_outputs_subset_accuracy",
        ]:
            self.assertIn(key, metrics)

    def test_rejects_three_output_matrix(self):
        with self.assertRaises(ValueError):
            derive_no_complication(np.zeros((4, 3)))


if __name__ == "__main__":
    unittest.main()
