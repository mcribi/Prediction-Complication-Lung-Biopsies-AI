from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from multimodal_common import (
    CLINICAL_FEATURES,
    AlignedMultimodalDataset,
    FusionClassifier,
    fit_clinical_transform,
    load_and_validate_split,
    select_best_complete_candidate,
    transform_clinical,
    validate_binary_labels,
    validate_outer_test_partition,
)


class DummyImageEncoder(nn.Module):
    output_dim = 4

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image[:, :4]


class MultimodalCommonTests(unittest.TestCase):
    def test_select_best_complete_candidate_uses_oof_test_f1_micro(self) -> None:
        rows = [
            {
                "config_id": "incomplete_high",
                "status": "ok",
                "num_folds_completed": "4",
                "all_folds_present": "False",
                "oof_f1_micro": "0.99",
                "oof_f1_macro": "0.99",
            },
            {
                "config_id": "complete_lower",
                "status": "ok",
                "num_folds_completed": "5",
                "all_folds_present": "True",
                "oof_f1_micro": "0.55",
                "oof_f1_macro": "0.52",
            },
            {
                "config_id": "complete_best",
                "status": "ok",
                "num_folds_completed": "5",
                "all_folds_present": "True",
                "oof_f1_micro": "0.56",
                "oof_f1_macro": "0.49",
            },
        ]
        selected = select_best_complete_candidate(rows)
        self.assertEqual(selected["config_id"], "complete_best")

    def test_select_best_complete_candidate_uses_macro_as_tie_break(self) -> None:
        rows = [
            {
                "config_id": "a",
                "status": "ok",
                "num_folds_completed": 5,
                "all_folds_present": True,
                "oof_f1_micro": 0.56,
                "oof_f1_macro": 0.48,
            },
            {
                "config_id": "b",
                "status": "ok",
                "num_folds_completed": 5,
                "all_folds_present": True,
                "oof_f1_micro": 0.56,
                "oof_f1_macro": 0.50,
            },
        ]
        self.assertEqual(select_best_complete_candidate(rows)["config_id"], "b")

    def test_select_best_complete_candidate_rejects_non_finite_metrics(self) -> None:
        rows = [
            {
                "config_id": "nan_candidate",
                "status": "ok",
                "num_folds_completed": 5,
                "all_folds_present": True,
                "oof_f1_micro": float("nan"),
                "oof_f1_macro": 0.99,
            },
            {
                "config_id": "finite_candidate",
                "status": "ok",
                "num_folds_completed": 5,
                "all_folds_present": True,
                "oof_f1_micro": 0.50,
                "oof_f1_macro": 0.45,
            },
        ]
        self.assertEqual(
            select_best_complete_candidate(rows)["config_id"],
            "finite_candidate",
        )

    def test_clinical_transform_is_fit_only_on_train_inner(self) -> None:
        clinical = pd.DataFrame(
            {
                "patient_id": ["p1", "p2", "p3"],
                "Edad": [10.0, 20.0, 100.0],
                **{
                    name: [0, 1, 0]
                    for name in CLINICAL_FEATURES
                    if name != "Edad"
                },
            }
        ).set_index("patient_id")
        params = fit_clinical_transform(clinical, ["p1", "p2"])
        transformed = transform_clinical(clinical, params)
        self.assertEqual(params, {"age_mean": 15.0, "age_std": 5.0})
        self.assertAlmostEqual(transformed.loc["p1", "Edad"], -1.0)
        self.assertAlmostEqual(transformed.loc["p2", "Edad"], 1.0)
        self.assertAlmostEqual(transformed.loc["p3", "Edad"], 17.0)
        self.assertEqual(transformed.loc["p2", "Sexo_binaria"], 1.0)

    def test_clinical_transform_rejects_missing_features(self) -> None:
        clinical = pd.DataFrame({"Edad": [50.0]}, index=["p1"])
        with self.assertRaisesRegex(ValueError, "Missing clinical features"):
            fit_clinical_transform(clinical, ["p1"])

    def test_load_and_validate_split_preserves_disjoint_subsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split_assignments.csv"
            pd.DataFrame(
                {
                    "split": ["train_inner", "train_inner", "val_inner", "test_outer"],
                    "patient_id": ["p1", "p2", "p3", "p4"],
                }
            ).to_csv(path, index=False)
            split = load_and_validate_split(
                path,
                expected_ids={"p1", "p2", "p3", "p4"},
            )
        self.assertEqual(split["train_inner"], ["p1", "p2"])
        self.assertEqual(split["val_inner"], ["p3"])
        self.assertEqual(split["test_outer"], ["p4"])

    def test_load_and_validate_split_rejects_duplicate_patient(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split_assignments.csv"
            pd.DataFrame(
                {
                    "split": ["train_inner", "test_outer"],
                    "patient_id": ["p1", "p1"],
                }
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "more than one split"):
                load_and_validate_split(path, expected_ids={"p1"})

    def test_outer_test_partition_requires_each_patient_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            patient_ids = ["p1", "p2", "p3", "p4", "p5"]
            for fold, test_id in enumerate(patient_ids, start=1):
                path = Path(directory) / f"fold_{fold}.csv"
                remaining = [pid for pid in patient_ids if pid != test_id]
                pd.DataFrame(
                    {
                        "split": ["train_inner"] * 3 + ["val_inner", "test_outer"],
                        "patient_id": [*remaining[:3], remaining[3], test_id],
                    }
                ).to_csv(path, index=False)
                paths.append(path)
            validate_outer_test_partition(paths, set(patient_ids))

    def test_outer_test_partition_rejects_repeated_test_patient(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            patient_ids = ["p1", "p2", "p3", "p4", "p5"]
            for fold in range(1, 6):
                path = Path(directory) / f"fold_{fold}.csv"
                test_id = "p1"
                remaining = [pid for pid in patient_ids if pid != test_id]
                pd.DataFrame(
                    {
                        "split": ["train_inner"] * 3 + ["val_inner", "test_outer"],
                        "patient_id": [*remaining[:3], remaining[3], test_id],
                    }
                ).to_csv(path, index=False)
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "exactly once"):
                validate_outer_test_partition(paths, set(patient_ids))

    def test_binary_label_validation_rejects_invalid_values(self) -> None:
        labels = pd.DataFrame(
            {
                "patient_id": ["p1", "p2"],
                "Hemorragia": [0, 2],
                "Neumotórax": [0, 1],
                "Sin_complicacion": [1, 0],
            }
        )
        with self.assertRaisesRegex(ValueError, "binary"):
            validate_binary_labels(labels)

    def test_binary_label_validation_rejects_duplicate_ids(self) -> None:
        labels = pd.DataFrame(
            {
                "patient_id": ["p1", "p1"],
                "Hemorragia": [0, 1],
                "Neumotórax": [0, 0],
                "Sin_complicacion": [1, 0],
            }
        )
        with self.assertRaisesRegex(ValueError, "Duplicated"):
            validate_binary_labels(labels)

    def test_fusion_classifier_outputs_three_logits(self) -> None:
        for mode in ["image_only", "clinical_only", "multimodal"]:
            with self.subTest(mode=mode):
                model = FusionClassifier(
                    mode=mode,
                    image_encoder=DummyImageEncoder(),
                    image_dim=4,
                    clinical_dim=len(CLINICAL_FEATURES),
                    image_projection_dim=6,
                    clinical_projection_dim=3,
                    hidden_dim=5,
                    out_dim=3,
                    dropout=0.0,
                )
                image = torch.arange(8, dtype=torch.float32).reshape(2, 4)
                clinical = torch.ones((2, len(CLINICAL_FEATURES)), dtype=torch.float32)
                logits = model(
                    image if mode != "clinical_only" else None,
                    clinical if mode != "image_only" else None,
                )
                self.assertEqual(tuple(logits.shape), (2, 3))
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_aligned_dataset_pairs_modalities_by_patient(self) -> None:
        labels = pd.DataFrame(
            {
                "patient_id": ["p1", "p2"],
                "Hemorragia": [1, 0],
                "Neumotórax": [0, 1],
                "Sin_complicacion": [0, 0],
            }
        )
        clinical = pd.DataFrame(
            {
                feature: [float(index), float(index + 10)]
                for index, feature in enumerate(CLINICAL_FEATURES)
            },
            index=["p1", "p2"],
        )

        class DummyVolumeDataset:
            patient_ids = ["p2", "p1"]

            def __getitem__(self, index: int):
                pid = self.patient_ids[index]
                value = 2.0 if pid == "p2" else 1.0
                return torch.tensor([value]), torch.zeros(3), pid

            def __len__(self):
                return len(self.patient_ids)

        dataset = AlignedMultimodalDataset(
            labels_frame=labels,
            clinical_frame=clinical,
            patient_ids=["p2", "p1"],
            mode="multimodal",
            volume_dataset=DummyVolumeDataset(),
        )
        image, clinical_x, target, pid = dataset[0]
        self.assertEqual(pid, "p2")
        self.assertEqual(image.item(), 2.0)
        self.assertEqual(clinical_x[0].item(), clinical.loc["p2", "Edad"])
        self.assertEqual(target.tolist(), [0.0, 1.0, 0.0])

    def test_clinical_only_dataset_does_not_need_volume_dataset(self) -> None:
        labels = pd.DataFrame(
            {
                "patient_id": ["p1"],
                "Hemorragia": [0],
                "Neumotórax": [0],
                "Sin_complicacion": [1],
            }
        )
        clinical = pd.DataFrame(
            {feature: [0.0] for feature in CLINICAL_FEATURES},
            index=["p1"],
        )
        dataset = AlignedMultimodalDataset(
            labels_frame=labels,
            clinical_frame=clinical,
            patient_ids=["p1"],
            mode="clinical_only",
            volume_dataset=None,
        )
        image, clinical_x, target, pid = dataset[0]
        self.assertEqual(pid, "p1")
        self.assertEqual(image.numel(), 0)
        self.assertEqual(tuple(clinical_x.shape), (len(CLINICAL_FEATURES),))
        self.assertEqual(target.tolist(), [0.0, 0.0, 1.0])

    def test_multimodal_classifier_requires_both_inputs(self) -> None:
        model = FusionClassifier(
            mode="multimodal",
            image_encoder=DummyImageEncoder(),
            image_dim=4,
            clinical_dim=len(CLINICAL_FEATURES),
            image_projection_dim=6,
            clinical_projection_dim=3,
            hidden_dim=5,
            out_dim=3,
            dropout=0.0,
        )
        with self.assertRaisesRegex(ValueError, "image input"):
            model(None, torch.ones((1, len(CLINICAL_FEATURES))))
        with self.assertRaisesRegex(ValueError, "clinical input"):
            model(torch.ones((1, 4)), None)


if __name__ == "__main__":
    unittest.main()
