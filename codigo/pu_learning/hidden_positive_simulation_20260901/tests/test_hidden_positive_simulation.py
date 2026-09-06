import unittest

import pandas as pd

from hidden_positive_simulation import select_hidden_positive_ids


class HiddenPositiveSimulationTests(unittest.TestCase):
    def test_select_hidden_positive_ids_is_deterministic_and_uses_only_positives(self):
        frame = pd.DataFrame(
            {
                "patient_id": [f"P{i:02d}" for i in range(20)],
                "target": [1] * 10 + [0] * 10,
            }
        )

        first = select_hidden_positive_ids(frame, "target", 0.30, 17)
        second = select_hidden_positive_ids(frame, "target", 0.30, 17)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        positives = set(frame.loc[frame["target"] == 1, "patient_id"])
        self.assertTrue(set(first).issubset(positives))

    def test_apply_hidden_labels_preserves_ground_truth_and_changes_only_selected_rows(self):
        from hidden_positive_simulation import apply_hidden_labels

        frame = pd.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3", "P4"],
                "target": [1, 1, 0, 0],
            }
        )

        altered = apply_hidden_labels(frame, "target", ["P2"])

        self.assertEqual(altered["y_true"].tolist(), [1, 1, 0, 0])
        self.assertEqual(altered["y_observed"].tolist(), [1, 0, 0, 0])
        self.assertEqual(altered.loc[altered["patient_id"] == "P2", "is_hidden_positive"].item(), 1)
        self.assertEqual(altered["is_hidden_positive"].sum(), 1)

    def test_build_task_rows_covers_reference_and_all_seeded_levels(self):
        from hidden_positive_simulation import build_task_rows

        rows = build_task_rows(
            targets=["Hemorragia", "Neumotórax"],
            fractions=[0.0, 0.1, 0.2, 0.3, 0.4],
            concealment_seeds=list(range(1001, 1011)),
        )

        self.assertEqual(len(rows), 82)
        self.assertEqual([row["task_id"] for row in rows], list(range(82)))
        for target in ["Hemorragia", "Neumotórax"]:
            zero = [row for row in rows if row["target"] == target and row["hide_fraction"] == 0.0]
            self.assertEqual(len(zero), 1)
            self.assertEqual(zero[0]["concealment_seed"], 0)
            for fraction in [0.1, 0.2, 0.3, 0.4]:
                seeded = [row for row in rows if row["target"] == target and row["hide_fraction"] == fraction]
                self.assertEqual(len(seeded), 10)

    def test_validate_hidden_ids_rejects_test_patients_and_nonpositives(self):
        from hidden_positive_simulation import validate_hidden_ids

        train = pd.DataFrame(
            {"patient_id": ["P1", "P2", "P3"], "target": [1, 1, 0]}
        )
        test_ids = {"P4", "P5"}

        validate_hidden_ids(train, "target", ["P2"], test_ids)
        with self.assertRaises(ValueError):
            validate_hidden_ids(train, "target", ["P4"], test_ids)
        with self.assertRaises(ValueError):
            validate_hidden_ids(train, "target", ["P3"], test_ids)


if __name__ == "__main__":
    unittest.main()
