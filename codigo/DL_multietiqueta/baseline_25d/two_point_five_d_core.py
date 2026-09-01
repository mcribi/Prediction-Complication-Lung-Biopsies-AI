from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.networks.nets import resnet18
from torch.utils.data import Dataset

SPLIT_NAMES = ("train_inner", "val_inner", "test_outer")
TARGET_LABELS = ("Hemorragia", "Neumotórax", "Sin_complicacion")


def build_resnet18_2d(in_channels: int, out_channels: int, dropout: float):
    model = resnet18(
        spatial_dims=2,
        n_input_channels=int(in_channels),
        num_classes=int(out_channels),
    )
    model.fc = nn.Sequential(nn.Dropout(float(dropout)), model.fc)
    return model


def nodule_centroid(mask: np.ndarray) -> tuple[int, int, int]:
    """Return the rounded voxel centroid of a 3D nodule mask.

    Empty masks fall back to the geometric centre of the volume.
    """
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, received shape {mask.shape}")
    coordinates = np.argwhere(mask > 0)
    if coordinates.size == 0:
        return tuple(int(size // 2) for size in mask.shape)
    centre = np.rint(coordinates.mean(axis=0)).astype(int)
    return tuple(int(value) for value in centre)


def _clipped_indices(center: int, size: int, count: int) -> list[int]:
    if count <= 0 or count % 2 == 0:
        raise ValueError("The number of slices must be a positive odd integer")
    radius = count // 2
    return [int(np.clip(center + offset, 0, size - 1)) for offset in range(-radius, radius + 1)]


def extract_axial_stack(
    volume: np.ndarray,
    center: tuple[int, int, int],
    num_slices: int = 5,
) -> np.ndarray:
    """Extract consecutive slices along axis 0 as image channels."""
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, received shape {volume.shape}")
    indices = _clipped_indices(int(center[0]), volume.shape[0], int(num_slices))
    return np.stack([volume[index, :, :] for index in indices], axis=0).astype(np.float32)


def extract_triplanar(
    volume: np.ndarray,
    center: tuple[int, int, int],
) -> np.ndarray:
    """Extract axial, coronal and sagittal slices through a voxel centre."""
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, received shape {volume.shape}")
    z, y, x = (int(value) for value in center)
    z = int(np.clip(z, 0, volume.shape[0] - 1))
    y = int(np.clip(y, 0, volume.shape[1] - 1))
    x = int(np.clip(x, 0, volume.shape[2] - 1))
    views = [volume[z, :, :], volume[:, y, :], volume[:, :, x]]
    shapes = {view.shape for view in views}
    if len(shapes) != 1:
        raise ValueError(
            "Triplanar channel stacking requires a cubic volume; "
            f"received view shapes {sorted(shapes)}"
        )
    return np.stack(views, axis=0).astype(np.float32)


class TwoPointFiveDDataset(Dataset):
    def __init__(
        self,
        labels_frame: pd.DataFrame,
        patient_ids: Iterable[str],
        data_root,
        representation: str,
    ) -> None:
        if representation not in {"axial5", "triplanar"}:
            raise ValueError(f"Unsupported representation: {representation}")
        self.labels = labels_frame.copy()
        self.labels["patient_id"] = self.labels["patient_id"].astype(str)
        self.labels = self.labels.set_index("patient_id")
        self.patient_ids = [str(value) for value in patient_ids]
        self.data_root = pd.io.common.stringify_path(data_root)
        self.representation = representation

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int):
        from pathlib import Path

        patient_id = self.patient_ids[index]
        root = Path(self.data_root)
        volume = np.load(root / "images" / f"{patient_id}.npy").astype(np.float32)
        mask = np.load(root / "masks_nodule" / f"{patient_id}.npy").astype(np.float32)
        if volume.ndim == 4:
            volume = volume[0] if volume.shape[0] <= 4 else volume[..., 0]
        if mask.ndim == 4:
            mask = mask[0] if mask.shape[0] <= 4 else mask[..., 0]
        centre = nodule_centroid(mask)
        if self.representation == "axial5":
            image = extract_axial_stack(volume, centre, num_slices=5)
        else:
            image = extract_triplanar(volume, centre)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = np.clip(image, 0.0, 1.0).astype(np.float32)
        target = self.labels.loc[patient_id, list(TARGET_LABELS)].to_numpy(dtype=np.float32)
        return (
            torch.from_numpy(image),
            torch.from_numpy(target),
            patient_id,
        )


def validate_split_assignments(
    frame: pd.DataFrame,
    expected_ids: Iterable[str],
) -> dict[str, int]:
    required = {"split", "patient_id"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing split columns: {sorted(missing_columns)}")
    clean = frame.loc[:, ["split", "patient_id"]].copy()
    clean["patient_id"] = clean["patient_id"].astype(str)
    invalid_splits = set(clean["split"]) - set(SPLIT_NAMES)
    if invalid_splits:
        raise ValueError(f"Unexpected split names: {sorted(invalid_splits)}")
    duplicated = clean[clean["patient_id"].duplicated(keep=False)]
    if not duplicated.empty:
        raise ValueError("At least one patient appears in multiple splits")
    observed_ids = set(clean["patient_id"])
    expected_ids = {str(value) for value in expected_ids}
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise ValueError(f"Split cohort mismatch: missing={missing}, extra={extra}")
    counts = clean["split"].value_counts().to_dict()
    return {name: int(counts.get(name, 0)) for name in SPLIT_NAMES}


def validate_oof(
    frame: pd.DataFrame,
    expected_ids: Iterable[str],
    n_folds: int = 5,
) -> None:
    required = {"fold", "patient_id"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing OOF columns: {sorted(missing_columns)}")
    patient_ids = frame["patient_id"].astype(str)
    if patient_ids.duplicated().any():
        raise ValueError("OOF predictions contain duplicate patients")
    expected_ids = {str(value) for value in expected_ids}
    observed_ids = set(patient_ids)
    if observed_ids != expected_ids:
        raise ValueError("OOF predictions do not cover the expected cohort")
    folds = set(frame["fold"].astype(str))
    expected_folds = {f"fold_{index}" for index in range(1, n_folds + 1)}
    if folds != expected_folds:
        raise ValueError(f"Incomplete OOF folds: observed={sorted(folds)}")
