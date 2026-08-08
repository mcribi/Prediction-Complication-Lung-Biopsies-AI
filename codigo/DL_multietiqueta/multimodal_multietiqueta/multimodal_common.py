#!/usr/bin/env python3
"""Shared components for the retrospective multimodal experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

TARGET_LABELS = ["Hemorragia", "Neumotórax", "Sin_complicacion"]
MODES = ("image_only", "clinical_only", "multimodal")
CLINICAL_FEATURES = [
    "Edad",
    "Sexo_binaria",
    "Tabac_any",
    "CardioRisk",
    "Enfisema_any",
    "Dislipemia_any",
    "Hipertensión_pulmonar",
    "DM_any",
    "Obesidad",
    "Fibrosis_any",
    "AOS",
    "Sin_factor_de_riesgo",
    "Sin_patología_pulmonar",
]


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def select_best_complete_candidate(rows: Iterable[Mapping[str, object]]) -> dict:
    """Select the complete candidate with the best outer-test OOF F1."""
    complete = []
    for row in rows:
        status_ok = str(row.get("status", "")).strip().lower() == "ok"
        folds = int(float(row.get("num_folds_completed", 0) or 0))
        all_folds = _as_bool(row.get("all_folds_present", False))
        try:
            micro = float(row.get("oof_f1_micro", float("nan")))
            macro = float(row.get("oof_f1_macro", float("-inf")))
        except (TypeError, ValueError):
            continue
        if status_ok and folds == 5 and all_folds and np.isfinite(micro):
            candidate = dict(row)
            candidate["_selection_micro"] = micro
            candidate["_selection_macro"] = macro if np.isfinite(macro) else float("-inf")
            complete.append(candidate)
    if not complete:
        raise ValueError("No complete candidate with five outer folds was found")
    selected = max(
        complete,
        key=lambda row: (
            row["_selection_micro"],
            row["_selection_macro"],
            str(row.get("config_id", "")),
        ),
    )
    selected.pop("_selection_micro")
    selected.pop("_selection_macro")
    return selected


def validate_clinical_frame(clinical_frame: pd.DataFrame) -> None:
    missing = sorted(set(CLINICAL_FEATURES) - set(clinical_frame.columns))
    if missing:
        raise ValueError(f"Missing clinical features: {missing}")
    if clinical_frame.index.has_duplicates:
        raise ValueError("Duplicated patient identifiers in clinical table")
    if clinical_frame[CLINICAL_FEATURES].isna().any().any():
        raise ValueError("Missing values in clinical features")


def fit_clinical_transform(
    clinical_frame: pd.DataFrame,
    train_inner_ids: Sequence[str],
) -> dict[str, float]:
    """Fit age standardization using train_inner only."""
    validate_clinical_frame(clinical_frame)
    missing_ids = sorted(set(map(str, train_inner_ids)) - set(clinical_frame.index.astype(str)))
    if missing_ids:
        raise ValueError(f"Clinical data missing train patients: {missing_ids}")
    ages = clinical_frame.loc[list(train_inner_ids), "Edad"].astype(float).to_numpy()
    age_mean = float(np.mean(ages))
    age_std = float(np.std(ages, ddof=0))
    if not np.isfinite(age_std) or age_std <= 0.0:
        age_std = 1.0
    return {"age_mean": age_mean, "age_std": age_std}


def transform_clinical(
    clinical_frame: pd.DataFrame,
    params: Mapping[str, float],
) -> pd.DataFrame:
    validate_clinical_frame(clinical_frame)
    transformed = clinical_frame[CLINICAL_FEATURES].astype(np.float32).copy()
    transformed["Edad"] = (
        transformed["Edad"].astype(float) - float(params["age_mean"])
    ) / float(params["age_std"])
    values = transformed.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite values after clinical transformation")
    return transformed.astype(np.float32)


def load_and_validate_split(
    split_path: Path,
    expected_ids: set[str],
) -> dict[str, list[str]]:
    """Load one original nested-CV split and reject leakage or drift."""
    frame = pd.read_csv(split_path, dtype={"patient_id": str})
    required = {"split", "patient_id"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing split columns: {missing_columns}")
    allowed = {"train_inner", "val_inner", "test_outer"}
    unexpected = sorted(set(frame["split"]) - allowed)
    if unexpected:
        raise ValueError(f"Unexpected split names: {unexpected}")
    if frame["patient_id"].duplicated().any():
        raise ValueError("A patient appears in more than one split")
    observed_ids = set(frame["patient_id"].astype(str))
    if observed_ids != set(map(str, expected_ids)):
        missing = sorted(set(map(str, expected_ids)) - observed_ids)
        extra = sorted(observed_ids - set(map(str, expected_ids)))
        raise ValueError(f"Split cohort mismatch; missing={missing}, extra={extra}")
    result = {
        name: frame.loc[frame["split"] == name, "patient_id"].astype(str).tolist()
        for name in ("train_inner", "val_inner", "test_outer")
    }
    if any(not values for values in result.values()):
        raise ValueError("Every split must contain at least one patient")
    return result


def validate_binary_labels(labels_frame: pd.DataFrame) -> None:
    required = {"patient_id", *TARGET_LABELS}
    missing = sorted(required - set(labels_frame.columns))
    if missing:
        raise ValueError(f"Missing label columns: {missing}")
    if labels_frame["patient_id"].isna().any():
        raise ValueError("Missing patient identifiers in label table")
    if labels_frame["patient_id"].duplicated().any():
        raise ValueError("Duplicated patient identifiers in label table")
    values = labels_frame[TARGET_LABELS].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Labels contain missing or non-finite values")
    invalid = sorted(set(np.unique(values.to_numpy(dtype=float))) - {0.0, 1.0})
    if invalid:
        raise ValueError(f"Labels must be binary; invalid values: {invalid}")


def validate_outer_test_partition(
    split_paths: Sequence[Path],
    expected_ids: set[str],
) -> None:
    if len(split_paths) != 5:
        raise ValueError(f"Expected five outer split files, found {len(split_paths)}")
    test_ids = []
    for path in split_paths:
        split = load_and_validate_split(path, expected_ids)
        test_ids.extend(split["test_outer"])
    counts = pd.Series(test_ids, dtype=str).value_counts()
    observed = set(test_ids)
    if observed != set(map(str, expected_ids)) or not counts.eq(1).all():
        raise ValueError("Every patient must appear exactly once across the five test_outer folds")


class AlignedMultimodalDataset(Dataset):
    """Align image, clinical and target tensors by patient identifier."""

    def __init__(
        self,
        labels_frame: pd.DataFrame,
        clinical_frame: pd.DataFrame,
        patient_ids: Sequence[str],
        mode: str,
        volume_dataset=None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        validate_clinical_frame(clinical_frame)
        self.mode = mode
        self.patient_ids = list(map(str, patient_ids))
        self.labels = labels_frame.copy()
        self.labels["patient_id"] = self.labels["patient_id"].astype(str)
        self.labels = self.labels.set_index("patient_id")
        self.clinical = clinical_frame.copy()
        self.clinical.index = self.clinical.index.astype(str)
        self.volume_dataset = volume_dataset
        missing_labels = sorted(set(self.patient_ids) - set(self.labels.index))
        missing_clinical = sorted(set(self.patient_ids) - set(self.clinical.index))
        if missing_labels or missing_clinical:
            raise ValueError(
                f"Unmatched patients; labels={missing_labels}, clinical={missing_clinical}"
            )
        if mode != "clinical_only":
            if volume_dataset is None:
                raise ValueError("A volume dataset is required for image modes")
            volume_ids = list(map(str, getattr(volume_dataset, "patient_ids", [])))
            if volume_ids != self.patient_ids:
                raise ValueError("Volume dataset patient order does not match requested IDs")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int):
        pid = self.patient_ids[index]
        if self.mode == "clinical_only":
            image = torch.empty(0, dtype=torch.float32)
        else:
            image, _, volume_pid = self.volume_dataset[index]
            if str(volume_pid) != pid:
                raise RuntimeError("Volume and clinical patient identifiers are misaligned")
        clinical = torch.tensor(
            self.clinical.loc[pid, CLINICAL_FEATURES].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        target = torch.tensor(
            self.labels.loc[pid, TARGET_LABELS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        return image, clinical, target, pid


class FusionClassifier(nn.Module):
    """Shared fusion head with zero-filled missing modality blocks."""

    def __init__(
        self,
        mode: str,
        image_encoder: nn.Module,
        image_dim: int,
        clinical_dim: int,
        image_projection_dim: int = 256,
        clinical_projection_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.image_encoder = image_encoder
        self.image_projection_dim = int(image_projection_dim)
        self.clinical_projection_dim = int(clinical_projection_dim)
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, image_projection_dim),
            nn.LayerNorm(image_projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, clinical_projection_dim),
            nn.LayerNorm(clinical_projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(image_projection_dim + clinical_projection_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        image: torch.Tensor | None,
        clinical: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.mode in {"image_only", "multimodal"} and image is None:
            raise ValueError("An image input is required for this mode")
        if self.mode in {"clinical_only", "multimodal"} and clinical is None:
            raise ValueError("A clinical input is required for this mode")
        reference = image if image is not None else clinical
        if reference is None:
            raise ValueError("At least one modality is required")
        batch_size = int(reference.shape[0])
        device = reference.device
        dtype = reference.dtype
        if self.mode == "clinical_only":
            image_embedding = torch.zeros(
                (batch_size, self.image_projection_dim),
                device=device,
                dtype=dtype,
            )
        else:
            image_embedding = self.image_projection(self.image_encoder(image))
        if self.mode == "image_only":
            clinical_embedding = torch.zeros(
                (batch_size, self.clinical_projection_dim),
                device=device,
                dtype=dtype,
            )
        else:
            clinical_embedding = self.clinical_encoder(clinical)
        fused = torch.cat([image_embedding, clinical_embedding], dim=1)
        return self.fusion_head(fused)
