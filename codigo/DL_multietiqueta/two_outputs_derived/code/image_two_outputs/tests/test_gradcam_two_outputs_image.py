import csv
import json
import tempfile
import unittest
from pathlib import Path

from gradcam_two_outputs_image import (
    choose_slice_index,
    discover_complete_configs,
    select_oof_cases,
    validate_probability_match,
)


def write_csv(path: Path, rows):
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_complete_config(root: Path, run_name: str, config_name: str, f1: float) -> Path:
    run_dir = root / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "final_report.json").write_text(json.dumps({"failed_configs": 0}), encoding="utf-8")
    (run_dir / "setup.json").write_text(json.dumps({"model_name": "resnet18"}), encoding="utf-8")
    config_dir = run_dir / config_name
    config_dir.mkdir()
    (config_dir / "oof_metrics_tuned.json").write_text(
        json.dumps({"two_outputs_f1_micro": f1}), encoding="utf-8"
    )
    write_csv(config_dir / "oof_predictions.csv", [{"patient_id": "P1"}])
    for fold in range(1, 6):
        fold_dir = config_dir / f"fold_{fold}"
        fold_dir.mkdir()
        (fold_dir / "best_model.pt").write_bytes(b"checkpoint")
        write_csv(
            fold_dir / "test_predictions_best_epoch.csv",
            [{"patient_id": f"P{fold}"}],
        )
    return config_dir


class GradcamDiscoveryTests(unittest.TestCase):
    def test_slice_selection_prioritizes_nodule_area_over_global_cam(self):
        selected = choose_slice_index(
            cam_scores=[100.0, 90.0, 10.0, 20.0],
            nodule_masses=[0.0, 0.0, 5.0, 12.0],
        )
        self.assertEqual(selected, 3)

    def test_probability_match_tolerates_cross_gpu_noise_but_rejects_mismatch(self):
        error = validate_probability_match([0.500273245, 0.25], [0.5, 0.25])
        self.assertAlmostEqual(error, 0.000273245)
        with self.assertRaises(RuntimeError):
            validate_probability_match([0.503, 0.25], [0.5, 0.25])

    def test_discover_complete_configs_excludes_active_and_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = make_complete_config(
                root,
                "resnet18_finished",
                "resize_small__ct_lung_nodule__bs4",
                0.61,
            )
            active = make_complete_config(
                root,
                "resnet34_active",
                "resize_small__ct_lung_nodule__bs4",
                0.99,
            )
            (active.parent / "final_report.json").unlink()
            incomplete = make_complete_config(
                root,
                "resnet10_incomplete",
                "resize_small__ct_lung_nodule__bs4",
                0.80,
            )
            (incomplete / "fold_5" / "test_predictions_best_epoch.csv").unlink()

            found = discover_complete_configs(root)

            self.assertEqual([item.config_dir for item in found], [complete])
            self.assertEqual(found[0].model_name, "resnet18")
            self.assertEqual(found[0].tuned_f1_micro, 0.61)

    def test_select_oof_cases_uses_only_modeled_outputs_and_is_deterministic(self):
        records = [
            {"patient_id": "P1", "fold": "fold_1", "true_Hemorragia": 1, "pred_tuned_Hemorragia": 1, "prob_Hemorragia": 0.91, "true_Neumotórax": 0, "pred_tuned_Neumotórax": 0, "prob_Neumotórax": 0.11},
            {"patient_id": "P2", "fold": "fold_1", "true_Hemorragia": 1, "pred_tuned_Hemorragia": 0, "prob_Hemorragia": 0.39, "true_Neumotórax": 0, "pred_tuned_Neumotórax": 0, "prob_Neumotórax": 0.20},
            {"patient_id": "P3", "fold": "fold_1", "true_Hemorragia": 0, "pred_tuned_Hemorragia": 1, "prob_Hemorragia": 0.88, "true_Neumotórax": 1, "pred_tuned_Neumotórax": 1, "prob_Neumotórax": 0.93},
            {"patient_id": "P4", "fold": "fold_1", "true_Hemorragia": 0, "pred_tuned_Hemorragia": 0, "prob_Hemorragia": 0.10, "true_Neumotórax": 1, "pred_tuned_Neumotórax": 0, "prob_Neumotórax": 0.42},
            {"patient_id": "P5", "fold": "fold_1", "true_Hemorragia": 0, "pred_tuned_Hemorragia": 0, "prob_Hemorragia": 0.20, "true_Neumotórax": 0, "pred_tuned_Neumotórax": 1, "prob_Neumotórax": 0.89},
            {"patient_id": "P6", "fold": "fold_1", "true_Hemorragia": 0, "pred_tuned_Hemorragia": 0, "prob_Hemorragia": 0.30, "true_Neumotórax": 0, "pred_tuned_Neumotórax": 0, "prob_Neumotórax": 0.10},
        ]

        selected = select_oof_cases(records, cases_per_state=1)

        self.assertEqual(
            [(x.label, x.state, x.patient_id) for x in selected],
            [
                ("Hemorragia", "TP", "P1"),
                ("Hemorragia", "FN", "P2"),
                ("Neumotórax", "TP", "P3"),
                ("Neumotórax", "FN", "P4"),
            ],
        )
        self.assertEqual({x.label for x in selected}, {"Hemorragia", "Neumotórax"})


if __name__ == "__main__":
    unittest.main()
