import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import two_point_five_d_core as core


class TestSliceExtraction(unittest.TestCase):
    def test_nodule_centroid_uses_mask_coordinates(self):
        mask = np.zeros((7, 9, 11), dtype=np.float32)
        mask[2:5, 3:6, 4:9] = 1
        self.assertEqual(core.nodule_centroid(mask), (3, 4, 6))

    def test_nodule_centroid_falls_back_to_volume_center(self):
        mask = np.zeros((6, 8, 10), dtype=np.float32)
        self.assertEqual(core.nodule_centroid(mask), (3, 4, 5))

    def test_axial_stack_clips_boundary_indices(self):
        volume = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
        stack = core.extract_axial_stack(volume, center=(0, 1, 1), num_slices=5)
        self.assertEqual(stack.shape, (5, 3, 2))
        np.testing.assert_array_equal(stack[0], volume[0])
        np.testing.assert_array_equal(stack[1], volume[0])
        np.testing.assert_array_equal(stack[2], volume[0])
        np.testing.assert_array_equal(stack[3], volume[1])
        np.testing.assert_array_equal(stack[4], volume[2])

    def test_triplanar_returns_three_square_views(self):
        volume = np.arange(5 ** 3, dtype=np.float32).reshape(5, 5, 5)
        views = core.extract_triplanar(volume, center=(1, 2, 3))
        self.assertEqual(views.shape, (3, 5, 5))
        np.testing.assert_array_equal(views[0], volume[1, :, :])
        np.testing.assert_array_equal(views[1], volume[:, 2, :])
        np.testing.assert_array_equal(views[2], volume[:, :, 3])


class TestTwoPointFiveDDataset(unittest.TestCase):
    def test_dataset_builds_axial_input_and_multilabel_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "masks_nodule").mkdir()
            volume = np.arange(7 * 5 * 5, dtype=np.float32).reshape(7, 5, 5)
            volume /= volume.max()
            mask = np.zeros_like(volume)
            mask[2:5, 1:4, 1:4] = 1
            np.save(root / "images" / "p1.npy", volume)
            np.save(root / "masks_nodule" / "p1.npy", mask)
            labels = pd.DataFrame(
                {
                    "patient_id": ["p1"],
                    "Hemorragia": [1],
                    "Neumotórax": [0],
                    "Sin_complicacion": [0],
                }
            )
            dataset = core.TwoPointFiveDDataset(
                labels_frame=labels,
                patient_ids=["p1"],
                data_root=root,
                representation="axial5",
            )
            x, y, patient_id = dataset[0]
            self.assertEqual(patient_id, "p1")
            self.assertEqual(tuple(x.shape), (5, 5, 5))
            self.assertTrue(torch.equal(y, torch.tensor([1.0, 0.0, 0.0])))

    def test_dataset_builds_triplanar_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "masks_nodule").mkdir()
            volume = np.ones((5, 5, 5), dtype=np.float32)
            mask = np.zeros_like(volume)
            mask[2, 2, 2] = 1
            np.save(root / "images" / "p1.npy", volume)
            np.save(root / "masks_nodule" / "p1.npy", mask)
            labels = pd.DataFrame(
                {
                    "patient_id": ["p1"],
                    "Hemorragia": [0],
                    "Neumotórax": [1],
                    "Sin_complicacion": [0],
                }
            )
            dataset = core.TwoPointFiveDDataset(labels, ["p1"], root, "triplanar")
            x, _, _ = dataset[0]
            self.assertEqual(tuple(x.shape), (3, 5, 5))


class TestModel(unittest.TestCase):
    def test_resnet18_2d_produces_three_logits(self):
        model = core.build_resnet18_2d(in_channels=5, out_channels=3, dropout=0.3)
        output = model(torch.zeros((2, 5, 64, 64), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (2, 3))


class TestSplitIntegrity(unittest.TestCase):
    def test_validate_split_assignments_accepts_disjoint_complete_split(self):
        frame = pd.DataFrame(
            {
                "split": ["train_inner", "train_inner", "val_inner", "test_outer"],
                "patient_id": ["a", "b", "c", "d"],
            }
        )
        counts = core.validate_split_assignments(frame, expected_ids={"a", "b", "c", "d"})
        self.assertEqual(counts, {"train_inner": 2, "val_inner": 1, "test_outer": 1})

    def test_validate_split_assignments_rejects_overlap(self):
        frame = pd.DataFrame(
            {
                "split": ["train_inner", "val_inner", "test_outer"],
                "patient_id": ["a", "a", "b"],
            }
        )
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            core.validate_split_assignments(frame, expected_ids={"a", "b"})

    def test_validate_oof_requires_five_folds_and_unique_patients(self):
        frame = pd.DataFrame(
            {
                "fold": ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"],
                "patient_id": ["a", "b", "c", "d", "e"],
            }
        )
        core.validate_oof(frame, expected_ids={"a", "b", "c", "d", "e"}, n_folds=5)
        duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            core.validate_oof(duplicated, expected_ids={"a", "b", "c", "d", "e"}, n_folds=5)


if __name__ == "__main__":
    unittest.main()
