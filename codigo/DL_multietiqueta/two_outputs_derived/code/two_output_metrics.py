from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_LABELS = ["Hemorragia", "Neumotórax"]
DERIVED_LABEL = "Sin_complicacion"
DEFAULT_THRESHOLD = 0.5
THRESHOLD_GRID = np.round(np.arange(0.10, 0.901, 0.05), 2)


@dataclass(frozen=True)
class ThresholdSelection:
    thresholds: np.ndarray
    curve_rows: list[dict]


def _as_binary_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2); found {matrix.shape}")
    if name in {"targets", "predictions"}:
        unique = set(np.unique(matrix).tolist())
        if not unique.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"{name} must be binary; found {sorted(unique)}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    if name == "probabilities" and np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("probabilities must be within [0, 1]")
    return matrix


def derive_no_complication(binary_two_outputs: np.ndarray) -> np.ndarray:
    matrix = _as_binary_matrix(binary_two_outputs, "predictions")
    return ((matrix[:, 0] == 0) & (matrix[:, 1] == 0)).astype(int)


def apply_thresholds(probabilities: np.ndarray, thresholds: Iterable[float]) -> np.ndarray:
    probs = _as_binary_matrix(probabilities, "probabilities").astype(float)
    threshold_array = np.asarray(list(thresholds), dtype=float)
    if threshold_array.shape != (2,):
        raise ValueError(f"thresholds must have shape (2,); found {threshold_array.shape}")
    if not np.isfinite(threshold_array).all() or np.any((threshold_array < 0) | (threshold_array > 1)):
        raise ValueError("thresholds must be finite and within [0, 1]")
    return (probs >= threshold_array.reshape(1, 2)).astype(int)


def select_per_label_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    grid: Iterable[float] = THRESHOLD_GRID,
) -> ThresholdSelection:
    y_true = _as_binary_matrix(targets, "targets").astype(int)
    y_prob = _as_binary_matrix(probabilities, "probabilities").astype(float)
    candidates = np.asarray(list(grid), dtype=float)
    if candidates.ndim != 1 or candidates.size == 0:
        raise ValueError("threshold grid must be a non-empty vector")
    if not np.isfinite(candidates).all() or np.any((candidates < 0) | (candidates > 1)):
        raise ValueError("threshold grid must be finite and within [0, 1]")

    selected = []
    curve_rows = []
    for label_index, label in enumerate(TARGET_LABELS):
        label_rows = []
        for threshold in candidates:
            pred = (y_prob[:, label_index] >= threshold).astype(int)
            score = float(f1_score(y_true[:, label_index], pred, zero_division=0))
            row = {
                "label": label,
                "label_index": label_index,
                "threshold": float(threshold),
                "f1": score,
                "precision": float(precision_score(y_true[:, label_index], pred, zero_division=0)),
                "recall": float(recall_score(y_true[:, label_index], pred, zero_division=0)),
                "distance_to_0_5": float(abs(threshold - DEFAULT_THRESHOLD)),
            }
            label_rows.append(row)
            curve_rows.append(row.copy())
        best = sorted(
            label_rows,
            key=lambda row: (-row["f1"], row["distance_to_0_5"], row["threshold"]),
        )[0]
        selected.append(best["threshold"])
        for row in curve_rows:
            if row["label_index"] == label_index:
                row["selected"] = bool(np.isclose(row["threshold"], best["threshold"]))
    return ThresholdSelection(np.asarray(selected, dtype=float), curve_rows)


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return (
        float(roc_auc_score(y_true, scores))
        if np.unique(y_true).size == 2 and np.isfinite(scores).all()
        else float("nan")
    )


def _safe_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    return (
        float(average_precision_score(y_true, scores))
        if np.any(y_true == 1) and np.isfinite(scores).all()
        else float("nan")
    )


def _per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, labels: list[str]) -> dict:
    metrics: dict[str, float | int] = {}
    for index, label in enumerate(labels):
        true = y_true[:, index].astype(int)
        pred = y_pred[:, index].astype(int)
        score = y_prob[:, index].astype(float)
        tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
        metrics.update(
            {
                f"precision_{label}": float(precision_score(true, pred, zero_division=0)),
                f"recall_{label}": float(recall_score(true, pred, zero_division=0)),
                f"accuracy_{label}": float(accuracy_score(true, pred)),
                f"balanced_accuracy_{label}": float(balanced_accuracy_score(true, pred)),
                f"sensitivity_{label}": float(recall_score(true, pred, zero_division=0)),
                f"specificity_{label}": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
                f"f1_{label}": float(f1_score(true, pred, zero_division=0)),
                f"roc_auc_{label}": _safe_auc(true, score),
                f"average_precision_{label}": _safe_ap(true, score),
                f"tn_{label}": int(tn),
                f"fp_{label}": int(fp),
                f"fn_{label}": int(fn),
                f"tp_{label}": int(tp),
                f"support_{label}": int(true.sum()),
            }
        )
    return metrics


def _aggregate_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    result = {
        f"{prefix}f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        f"{prefix}f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"{prefix}precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        f"{prefix}precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"{prefix}recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        f"{prefix}recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"{prefix}jaccard_micro": float(jaccard_score(y_true, y_pred, average="micro", zero_division=0)),
        f"{prefix}jaccard_macro": float(jaccard_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}jaccard_weighted": float(jaccard_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"{prefix}subset_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}hamming_loss": float(hamming_loss(y_true, y_pred)),
    }
    try:
        result[f"{prefix}f1_samples"] = float(f1_score(y_true, y_pred, average="samples", zero_division=0))
        result[f"{prefix}jaccard_samples"] = float(jaccard_score(y_true, y_pred, average="samples", zero_division=0))
    except ValueError:
        result[f"{prefix}f1_samples"] = float("nan")
        result[f"{prefix}jaccard_samples"] = float("nan")
    return result


def compute_metrics_from_predictions(
    targets: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> tuple[dict, np.ndarray]:
    y_true_two = _as_binary_matrix(targets, "targets").astype(int)
    y_prob_two = _as_binary_matrix(probabilities, "probabilities").astype(float)
    y_pred_two = _as_binary_matrix(predictions, "predictions").astype(int)
    true_derived = derive_no_complication(y_true_two)
    pred_derived = derive_no_complication(y_pred_two)

    y_true_clinical = np.column_stack([y_true_two, true_derived])
    y_pred_clinical = np.column_stack([y_pred_two, pred_derived])
    # There is no independent logit or calibrated probability for the derived state.
    # Threshold-independent score metrics are therefore intentionally undefined.
    derived_score = np.full(len(y_prob_two), np.nan, dtype=float)
    y_prob_clinical = np.column_stack([y_prob_two, derived_score])

    metrics: dict = {"n_samples": int(len(y_true_two))}
    metrics.update(_aggregate_metrics(y_true_two, y_pred_two, "two_outputs_"))
    metrics.update(_per_label_metrics(y_true_two, y_pred_two, y_prob_two, TARGET_LABELS))
    metrics.update(_aggregate_metrics(y_true_clinical, y_pred_clinical, "clinical_three_states_"))
    metrics.update(
        _per_label_metrics(
            y_true_clinical,
            y_pred_clinical,
            y_prob_clinical,
            [*TARGET_LABELS, DERIVED_LABEL],
        )
    )
    return metrics, pred_derived


def compute_two_output_and_derived_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Iterable[float],
) -> tuple[dict, np.ndarray, np.ndarray]:
    threshold_array = np.asarray(list(thresholds), dtype=float)
    y_pred_two = apply_thresholds(probabilities, threshold_array)
    metrics, pred_derived = compute_metrics_from_predictions(
        targets,
        probabilities,
        y_pred_two,
    )
    metrics["threshold_Hemorragia"] = float(threshold_array[0])
    metrics["threshold_Neumotorax"] = float(threshold_array[1])
    return metrics, y_pred_two, pred_derived
