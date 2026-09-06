import unittest

from gradcam_all_oof_four_slices import select_representative_slices, validate_oof_records


class AllOofGradcamTests(unittest.TestCase):
    def test_selects_four_highest_nodule_area_slices_in_spatial_order(self):
        masses = [0, 1, 8, 3, 10, 6, 2, 0]
        self.assertEqual(select_representative_slices(masses, 4), [2, 3, 4, 5])

    def test_returns_four_unique_slices_when_nodule_spans_fewer_slices(self):
        masses = [0, 0, 4, 0, 0, 0]
        selected = select_representative_slices(masses, 4)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(set(selected)), 4)
        self.assertIn(2, selected)
        self.assertTrue(all(0 <= index < len(masses) for index in selected))

    def test_validates_complete_oof_with_five_folds_and_unique_patients(self):
        records = []
        for fold in range(1, 6):
            for patient in range(3):
                records.append({"patient_id": f"P{fold}_{patient}", "fold": f"fold_{fold}"})
        report = validate_oof_records(records, expected_patients=15)
        self.assertEqual(report["patients"], 15)
        self.assertEqual(report["folds"], ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"])

    def test_rejects_duplicate_oof_patient(self):
        records = [
            {"patient_id": "P1", "fold": "fold_1"},
            {"patient_id": "P1", "fold": "fold_2"},
        ]
        with self.assertRaises(ValueError):
            validate_oof_records(records, expected_patients=2)


if __name__ == "__main__":
    unittest.main()
