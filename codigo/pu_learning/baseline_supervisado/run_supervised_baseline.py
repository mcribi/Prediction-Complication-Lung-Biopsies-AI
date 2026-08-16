#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    coverage_error,
    f1_score,
    hamming_loss,
    jaccard_score,
    label_ranking_average_precision_score,
    label_ranking_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    zero_one_loss,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC

TARGET_COLS_DEFAULT = ["Hemorragia", "Neumotórax", "Sin_complicación"]
METADATA_COLS = {"patient_id", "mask_kind", "image_path", "image_size", "image_spacing"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run one supervised multilabel baseline configuration.")
    parser.add_argument("config_index", type=int)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--folds_csv", type=str, default=None, help="Optional existing split_assignments.csv to reuse.")
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--job_suffix", type=str, default=None)
    return parser.parse_args()


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_configs(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        configs = json.load(f)
    if not isinstance(configs, list):
        raise ValueError("config_file must contain a list")
    return configs


def config_output_dir(output_root: Path, config_index: int, config: Dict[str, Any], suffix: Optional[str]) -> Path:
    name = config.get("config_name", f"config_{config_index:06d}")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)[:180]
    if suffix:
        safe = f"{safe}__{suffix}"
    return output_root / safe


def normalize_ids(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def load_and_align(config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    id_col = config.get("id_col", "patient_id")
    target_cols = list(config.get("target_cols", TARGET_COLS_DEFAULT))
    features_csv = Path(config["features_csv"]).expanduser().resolve()
    labels_csv = Path(config["labels_csv"]).expanduser().resolve()

    features = pd.read_csv(features_csv)
    labels = pd.read_csv(labels_csv)
    if id_col not in features.columns or id_col not in labels.columns:
        raise ValueError(f"{id_col} must exist in features and labels")
    missing_targets = [c for c in target_cols if c not in labels.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    features[id_col] = normalize_ids(features[id_col])
    labels[id_col] = normalize_ids(labels[id_col])

    cohort_ids_csv = config.get("cohort_ids_csv")
    if cohort_ids_csv:
        cohort = pd.read_csv(Path(cohort_ids_csv).expanduser().resolve())
        cohort_ids = set(normalize_ids(cohort[id_col]))
        features = features[features[id_col].isin(cohort_ids)].copy()
        labels = labels[labels[id_col].isin(cohort_ids)].copy()

    labels = labels[[id_col] + target_cols].copy()
    merged = features.merge(labels, on=id_col, how="inner", validate="one_to_one")
    for col in target_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.dropna(subset=target_cols).copy()
    for col in target_cols:
        merged[col] = merged[col].astype(int)

    feature_cols = []
    for col in merged.columns:
        if col in set(target_cols) | METADATA_COLS:
            continue
        if col.endswith("_image_path") or col.endswith("_image_size") or col.endswith("_image_spacing"):
            continue
        num = pd.to_numeric(merged[col], errors="coerce")
        if num.notna().any():
            merged[col] = num
            feature_cols.append(col)
    if not feature_cols:
        raise ValueError("No numeric feature columns available")

    merged = merged.sort_values(id_col).reset_index(drop=True)
    report = {
        "features_csv": str(features_csv),
        "labels_csv": str(labels_csv),
        "feature_set": config.get("feature_set"),
        "n_rows_features_after_cohort_filter": int(len(features)),
        "n_rows_labels_after_cohort_filter": int(len(labels)),
        "n_rows_merged": int(len(merged)),
        "n_patients_merged": int(merged[id_col].nunique()),
        "feature_count": int(len(feature_cols)),
        "feature_columns_preview": feature_cols[:30],
        "target_support": {c: int(merged[c].sum()) for c in target_cols},
        "labelset_counts": merged[target_cols].astype(str).agg("".join, axis=1).value_counts().to_dict(),
    }
    return merged[[id_col] + feature_cols + target_cols], feature_cols, report


def build_labelset_strata(y: np.ndarray) -> np.ndarray:
    labels = np.array(["|".join(map(str, row.astype(int).tolist())) for row in y])
    values, counts = np.unique(labels, return_counts=True)
    count_map = dict(zip(values, counts))
    return np.array([label if count_map[label] >= 2 else "RARE_LABELSET" for label in labels])


def make_splits(y: np.ndarray, config: Dict[str, Any], patient_ids: np.ndarray, folds_csv: Optional[Path]) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    n_splits = int(config.get("n_splits", 5))
    random_state = int(config.get("random_state", 42))
    shuffle = bool(config.get("shuffle_folds", True))
    strategy = config.get("split_strategy", "labelset_stratified")

    if folds_csv is not None:
        ref = pd.read_csv(folds_csv)
        needed = {"fold", "split", "patient_id"}
        if not needed.issubset(ref.columns):
            raise ValueError(f"folds_csv must contain {needed}")
        id_to_idx = {str(pid): i for i, pid in enumerate(patient_ids.astype(str))}
        splits = []
        for fold in sorted(ref["fold"].unique()):
            fold_df = ref[ref["fold"] == fold]
            test_ids = set(fold_df.loc[fold_df["split"] == "test", "patient_id"].astype(str))
            train_ids = set(fold_df.loc[fold_df["split"] == "train", "patient_id"].astype(str))
            if not test_ids:
                continue
            usable_test = [id_to_idx[pid] for pid in test_ids if pid in id_to_idx]
            usable_train = [id_to_idx[pid] for pid in train_ids if pid in id_to_idx]
            if not usable_train:
                usable_train = [i for i, pid in enumerate(patient_ids.astype(str)) if pid not in test_ids]
            splits.append((np.array(usable_train, dtype=int), np.array(usable_test, dtype=int)))
        return splits, {"resolved_splitter": "reference_split_assignments", "reference_folds_csv": str(folds_csv), "n_reference_folds_used": len(splits)}

    if strategy == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return list(splitter.split(np.zeros(len(y)))), {"resolved_splitter": "KFold", "stratification_basis": None}

    strata = build_labelset_strata(y)
    values, counts = np.unique(strata, return_counts=True)
    min_count = int(counts.min()) if len(counts) else 0
    if len(values) >= 2 and min_count >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return list(splitter.split(np.zeros(len(y)), strata)), {"resolved_splitter": "StratifiedKFold", "stratification_basis": "labelset_signature", "min_stratum_count": min_count}
    if strategy == "labelset_stratified":
        raise ValueError(f"labelset_stratified requested but min stratum count={min_count} < n_splits={n_splits}")
    splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return list(splitter.split(np.zeros(len(y)))), {"resolved_splitter": "KFold", "stratification_basis": "fallback_from_auto", "min_stratum_count": min_count}


def build_base_classifier(name: str, params: Dict[str, Any]):
    params = dict(params)
    if name == "LogisticRegression":
        params.setdefault("max_iter", 3000)
        return LogisticRegression(**params)
    if name == "LinearSVM":
        return LinearSVC(**params)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(**params)
    if name == "RandomForest":
        return RandomForestClassifier(**params)
    if name == "GradientBoosting":
        return GradientBoostingClassifier(**params)
    if name == "MLP":
        return MLPClassifier(**params)
    raise ValueError(f"Unsupported classifier: {name}")


def classifier_params(config: Dict[str, Any]) -> Dict[str, Any]:
    reserved = {
        "config_name", "feature_set", "features_csv", "labels_csv", "cohort_ids_csv", "id_col", "target_cols",
        "n_splits", "random_state", "shuffle_folds", "split_strategy", "classifier", "multilabel_strategy",
        "scaler", "dimred",
    }
    return {k: v for k, v in config.items() if k not in reserved}


def build_pipeline(config: Dict[str, Any]) -> Pipeline:
    random_state = int(config.get("random_state", 42))
    classifier_name = config["classifier"]
    params = classifier_params(config)
    params.setdefault("random_state", random_state)
    dimred = config.get("dimred", "none")
    scaler = config.get("scaler", "none")
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if dimred == "varth_0.01":
        steps.append(("vt", VarianceThreshold(threshold=0.01)))
    elif dimred in {"pca_95", "pca_99"}:
        steps.append(("scaler_pca", StandardScaler()))
        steps.append(("pca", PCA(n_components=0.95 if dimred == "pca_95" else 0.99, svd_solver="full", random_state=random_state)))
    elif dimred not in {None, "none", "None"}:
        raise ValueError(f"Unsupported dimred: {dimred}")
    if dimred not in {"pca_95", "pca_99"}:
        if scaler == "StandardScaler":
            steps.append(("scaler", StandardScaler()))
        elif scaler == "MinMaxScaler":
            steps.append(("scaler", MinMaxScaler()))
        elif scaler in {None, "none", "None"}:
            pass
        else:
            raise ValueError(f"Unsupported scaler: {scaler}")
    base = build_base_classifier(classifier_name, params)
    strategy = config.get("multilabel_strategy", "one_vs_rest")
    if strategy == "one_vs_rest":
        clf = OneVsRestClassifier(base)
    elif strategy == "multi_output":
        clf = MultiOutputClassifier(base)
    elif strategy == "classifier_chain":
        clf = ClassifierChain(base, random_state=random_state)
    else:
        raise ValueError(f"Unsupported multilabel_strategy: {strategy}")
    steps.append(("classifier", clf))
    return Pipeline(steps)


def get_scores(model: Pipeline, x: np.ndarray) -> Optional[np.ndarray]:
    clf = model.named_steps["classifier"]
    try:
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(x)
            if isinstance(p, list):
                return np.vstack([np.asarray(a)[:, 1] if np.asarray(a).ndim == 2 and np.asarray(a).shape[1] > 1 else np.asarray(a).ravel() for a in p]).T
            arr = np.asarray(p)
            if arr.ndim == 3 and arr.shape[2] > 1:
                return arr[:, :, 1]
            return arr
        if hasattr(clf, "decision_function"):
            return np.asarray(clf.decision_function(x))
    except Exception:
        return None
    return None


def safe_metric(fn, *args, **kwargs):
    try:
        value = fn(*args, **kwargs)
        if isinstance(value, np.ndarray):
            if value.size != 1:
                return None
            value = value.item()
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    except Exception:
        return None


def is_probability(scores: Optional[np.ndarray]) -> bool:
    if scores is None:
        return False
    scores = np.asarray(scores)
    return scores.size > 0 and np.isfinite(scores).all() and bool(((scores >= 0) & (scores <= 1)).all())


def compute_metrics(y_true, y_pred, y_score):
    metrics = {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "zero_one_loss": float(zero_one_loss(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "hamming_score": float(1.0 - hamming_loss(y_true, y_pred)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_samples": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "jaccard_micro": float(jaccard_score(y_true, y_pred, average="micro", zero_division=0)),
        "jaccard_macro": float(jaccard_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_score is not None:
        metrics["roc_auc_macro"] = safe_metric(roc_auc_score, y_true, y_score, average="macro")
        metrics["roc_auc_micro"] = safe_metric(roc_auc_score, y_true, y_score, average="micro")
        metrics["label_ranking_average_precision"] = safe_metric(label_ranking_average_precision_score, y_true, y_score)
        metrics["label_ranking_loss"] = safe_metric(label_ranking_loss, y_true, y_score)
        metrics["coverage_error"] = safe_metric(coverage_error, y_true, y_score)
        if is_probability(y_score):
            metrics["average_precision_macro"] = safe_metric(average_precision_score, y_true, y_score, average="macro")
            metrics["average_precision_micro"] = safe_metric(average_precision_score, y_true, y_score, average="micro")
        else:
            metrics["average_precision_macro"] = None
            metrics["average_precision_micro"] = None
    else:
        for key in ["roc_auc_macro", "roc_auc_micro", "label_ranking_average_precision", "label_ranking_loss", "coverage_error", "average_precision_macro", "average_precision_micro"]:
            metrics[key] = None
    return metrics


def label_metrics(y_true, y_pred, y_score, target_cols, fold, scope):
    precision, recall, f1_values, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    rows = []
    prob = is_probability(y_score)
    for i, label in enumerate(target_cols):
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1]).ravel()
        row = {
            "scope": scope,
            "fold": int(fold),
            "label": label,
            "balanced_accuracy": safe_metric(balanced_accuracy_score, y_true[:, i], y_pred[:, i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1_values[i]),
            "support": int(support[i]),
            "prevalence": int(y_true[:, i].sum()),
            "predicted_positives": int(y_pred[:, i].sum()),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "specificity": None if tn + fp == 0 else float(tn / (tn + fp)),
            "mcc": safe_metric(matthews_corrcoef, y_true[:, i], y_pred[:, i]),
            "cohen_kappa": safe_metric(cohen_kappa_score, y_true[:, i], y_pred[:, i]),
        }
        if y_score is not None and y_score.ndim == 2 and i < y_score.shape[1]:
            row["roc_auc"] = safe_metric(roc_auc_score, y_true[:, i], y_score[:, i])
            row["average_precision"] = safe_metric(average_precision_score, y_true[:, i], y_score[:, i]) if prob else None
            row["brier_score"] = safe_metric(brier_score_loss, y_true[:, i], y_score[:, i]) if prob else None
        else:
            row["roc_auc"] = None
            row["average_precision"] = None
            row["brier_score"] = None
        rows.append(row)
    return rows


def main():
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    config_file = Path(args.config_file).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    configs = load_configs(config_file)
    config = configs[args.config_index]
    out_dir = config_output_dir(output_root, args.config_index, config, args.job_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged, feature_cols, coverage = load_and_align(config)
    id_col = config.get("id_col", "patient_id")
    target_cols = list(config.get("target_cols", TARGET_COLS_DEFAULT))
    x = merged[feature_cols].values
    y = merged[target_cols].values.astype(int)
    patient_ids = merged[id_col].astype(str).values
    folds_csv = Path(args.folds_csv).expanduser().resolve() if args.folds_csv else None
    splits, split_meta = make_splits(y, config, patient_ids, folds_csv)

    run_context = {
        "config_index": int(args.config_index),
        "config_name": config.get("config_name"),
        "config_file": str(config_file),
        "output_dir": str(out_dir),
        "target_cols": target_cols,
        "save_models": bool(args.save_models),
        "save_predictions": bool(args.save_predictions),
        "dry_run": bool(args.dry_run),
        **config,
        **coverage,
        **split_meta,
    }
    save_json(out_dir / "config_used.json", run_context)

    split_rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        for idx in train_idx:
            split_rows.append({"fold": fold_idx, "split": "train", "patient_id": patient_ids[idx]})
        for idx in test_idx:
            split_rows.append({"fold": fold_idx, "split": "test", "patient_id": patient_ids[idx]})
    save_dataframe(out_dir / "split_assignments.csv", pd.DataFrame(split_rows))

    if args.dry_run:
        summary = {**run_context, "status": "dry_run_ok", "n_folds_prepared": len(splits)}
        save_json(out_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    pipeline = build_pipeline(config)
    models_dir = out_dir / "models"
    fold_rows = []
    label_rows = []
    pred_rows = []
    oof_true = []
    oof_pred = []
    oof_score = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        model = clone(pipeline)
        model.fit(x[train_idx], y[train_idx])
        y_pred = np.asarray(model.predict(x[test_idx]))
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        y_score = get_scores(model, x[test_idx])
        if y_score is not None:
            y_score = np.asarray(y_score)
            if y_score.ndim == 1:
                y_score = y_score.reshape(-1, 1)
        metrics = compute_metrics(y[test_idx], y_pred, y_score)
        fold_rows.append({"fold": fold_idx, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)), **metrics})
        label_rows.extend(label_metrics(y[test_idx], y_pred, y_score, target_cols, fold_idx, "fold"))
        oof_true.append(y[test_idx])
        oof_pred.append(y_pred)
        if y_score is not None:
            oof_score.append(y_score)
        if args.save_predictions:
            for r, pid in enumerate(patient_ids[test_idx]):
                row = {"fold": fold_idx, "patient_id": pid}
                for j, label in enumerate(target_cols):
                    row[f"true__{label}"] = int(y[test_idx][r, j])
                    row[f"pred__{label}"] = int(y_pred[r, j])
                    if y_score is not None and y_score.ndim == 2 and j < y_score.shape[1]:
                        row[f"score__{label}"] = float(y_score[r, j])
                pred_rows.append(row)
        if args.save_models:
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, models_dir / f"model_fold{fold_idx}.pkl")

    fold_df = pd.DataFrame(fold_rows)
    label_df = pd.DataFrame(label_rows)
    save_dataframe(out_dir / "fold_metrics.csv", fold_df)
    save_dataframe(out_dir / "label_metrics.csv", label_df)
    if pred_rows:
        save_dataframe(out_dir / "predictions.csv", pd.DataFrame(pred_rows))

    y_true_all = np.vstack(oof_true)
    y_pred_all = np.vstack(oof_pred)
    y_score_all = np.vstack(oof_score) if len(oof_score) == len(oof_true) else None
    oof_metrics = compute_metrics(y_true_all, y_pred_all, y_score_all)
    save_json(out_dir / "oof_metrics.json", oof_metrics)
    save_dataframe(out_dir / "oof_label_metrics.csv", pd.DataFrame(label_metrics(y_true_all, y_pred_all, y_score_all, target_cols, 0, "oof_all")))

    summary = {
        **run_context,
        **{f"mean_{c}": None if pd.to_numeric(fold_df[c], errors="coerce").dropna().empty else float(pd.to_numeric(fold_df[c], errors="coerce").mean()) for c in fold_df.columns if c not in {"fold", "n_train", "n_test"}},
        **{f"oof_{k}": v for k, v in oof_metrics.items()},
        "n_folds_executed": int(len(fold_df)),
        "status": "completed",
        "completed_at": datetime.now().isoformat(),
    }
    save_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
