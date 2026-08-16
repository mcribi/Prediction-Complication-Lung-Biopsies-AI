from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_auc(metric, y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) != 2 or not np.isfinite(y_score).all():
        return None
    return float(metric(y_true, y_score))


def binary_metrics(y_true, y_pred, y_score) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    return {
        "n": int(len(y_true)),
        "support": int(y_true.sum()),
        "predicted_positive": int(y_pred.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "recall": float(sensitivity),
        "specificity": float(specificity),
        "npv": float(npv),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "gmean": float(math.sqrt(sensitivity * specificity)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": safe_auc(roc_auc_score, y_true, y_score),
        "average_precision": safe_auc(average_precision_score, y_true, y_score),
        "brier_score": float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0))),
    }


def choose_threshold(y_true, y_score) -> Tuple[float, Dict[str, float]]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    values = np.unique(y_score[np.isfinite(y_score)])
    if not len(values):
        pred = np.zeros_like(y_true)
        return 0.5, binary_metrics(y_true, pred, np.zeros_like(y_score))
    if len(values) > 201:
        values = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 201)))
    candidates = np.unique(np.r_[0.0, values, 1.0])
    ranked = []
    for threshold in candidates:
        metrics = binary_metrics(y_true, y_score >= threshold, y_score)
        ranked.append((
            metrics["gmean"],
            metrics["mcc"],
            metrics["f1"],
            -abs(float(threshold) - 0.5),
            float(threshold),
            metrics,
        ))
    best = max(ranked, key=lambda row: row[:5])
    return best[4], best[5]


def estimate_c(observed_labels, observed_probabilities, minimum: float = 0.05) -> float:
    labels = np.asarray(observed_labels, dtype=int)
    probs = np.asarray(observed_probabilities, dtype=float)
    positive_probs = probs[labels == 1]
    if not len(positive_probs):
        raise ValueError("Cannot estimate c without observed positives")
    return float(np.clip(np.mean(positive_probs), minimum, 1.0))


def estimate_class_prior(observed_labels, c_value: float, minimum: float = 1e-3, maximum: float = 0.99) -> float:
    labels = np.asarray(observed_labels, dtype=int)
    if not 0.0 < c_value <= 1.0:
        raise ValueError("c_value must be in (0, 1]")
    prior = float(labels.mean()) / float(c_value)
    return float(np.clip(prior, minimum, maximum))


def elkan_noto_correct(probabilities, c_value: float) -> np.ndarray:
    if not 0.0 < c_value <= 1.0:
        raise ValueError("c_value must be in (0, 1]")
    return np.clip(np.asarray(probabilities, dtype=float) / c_value, 0.0, 1.0)


def nnpu_loss(
    logits: torch.Tensor,
    observed_labels: torch.Tensor,
    class_prior: float,
    beta: float = 0.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    logits = logits.reshape(-1)
    labels = observed_labels.reshape(-1)
    positive = logits[labels > 0.5]
    unlabeled = logits[labels <= 0.5]
    if positive.numel() == 0 or unlabeled.numel() == 0:
        raise ValueError("Each nnPU batch must contain positive and unlabeled examples")
    positive_risk = class_prior * F.softplus(-positive).mean()
    negative_risk = F.softplus(unlabeled).mean() - class_prior * F.softplus(positive).mean()
    if negative_risk < -beta:
        return -gamma * negative_risk
    return positive_risk + negative_risk



def load_outer_fold(folds_path: Path, fold: int) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(folds_path)
    required = {"fold", "split", "patient_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"Fold file lacks columns: {sorted(required - set(df.columns))}")
    part = df[df["fold"] == int(fold)].copy()
    train_ids = part.loc[part["split"] == "train", "patient_id"].astype(str).tolist()
    test_ids = part.loc[part["split"] == "test", "patient_id"].astype(str).tolist()
    if not train_ids or not test_ids:
        raise ValueError(f"Fold {fold} has an empty train or test partition")
    if set(train_ids) & set(test_ids):
        raise ValueError(f"Fold {fold} has patient overlap")
    return train_ids, test_ids


def make_inner_split(
    labels_df: pd.DataFrame,
    outer_train_ids: List[str],
    target: str,
    seed: int,
    val_ratio: float = 0.2,
) -> Tuple[List[str], List[str]]:
    frame = labels_df.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame = frame.set_index("patient_id").loc[list(map(str, outer_train_ids))].reset_index()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(frame["patient_id"], frame[target]))
    train_ids = frame.iloc[train_idx]["patient_id"].astype(str).tolist()
    val_ids = frame.iloc[val_idx]["patient_id"].astype(str).tolist()
    if set(train_ids) & set(val_ids):
        raise RuntimeError("Inner train and validation overlap")
    if set(train_ids) | set(val_ids) != set(map(str, outer_train_ids)):
        raise RuntimeError("Inner split does not cover outer training patients")
    return train_ids, val_ids
