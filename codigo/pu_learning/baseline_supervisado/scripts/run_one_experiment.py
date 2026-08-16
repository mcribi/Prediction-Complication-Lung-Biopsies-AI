#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import VarianceThreshold

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

NON_FEATURE_COLS = {"patient_id", "mask_kind", "image_path", "image_size", "image_spacing", "Hemorragia", "Neumotórax", "Sin_complicación", "Derrame_pleural", "Complicacion_binaria"}


def parse_args():
    p = argparse.ArgumentParser(description="Run one expanded binary supervised/PU experiment from manifest.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def read_manifest_row(path: str, task_id: int) -> Dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["task_id"]) == task_id:
                return row
    raise ValueError(f"Task id {task_id} not found in {path}")


def safe_name(text: str) -> str:
    text = text.replace("ó", "o").replace("á", "a").replace("é", "e").replace("í", "i").replace("ú", "u").replace("ñ", "n")
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")


def params_short_hash(params: Dict) -> str:
    s = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def get_feature_columns(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        if c in NON_FEATURE_COLS or c.endswith("_image_path") or c.endswith("_image_size") or c.endswith("_image_spacing"):
            continue
        cols.append(c)
    return cols


def load_data(row):
    xdf = pd.read_csv(row["features_csv"])
    ydf = pd.read_csv(row["labels_csv"])
    cdf = pd.read_csv(row["cohort_ids_csv"])
    xdf["patient_id"] = xdf["patient_id"].astype(str)
    ydf["patient_id"] = ydf["patient_id"].astype(str)
    cdf["patient_id"] = cdf["patient_id"].astype(str)
    ids = set(cdf["patient_id"])
    df = xdf.merge(ydf[["patient_id", "Hemorragia", "Neumotórax", "Sin_complicación"]], on="patient_id", how="inner")
    df = df[df["patient_id"].isin(ids)].copy()
    df = df.sort_values("patient_id").reset_index(drop=True)
    feature_cols = get_feature_columns(df)
    return df, feature_cols


def build_base_model(model_name: str, params: Dict, seed: int):
    params = dict(params)
    if model_name == "logistic_regression":
        return LogisticRegression(C=params["C"], penalty=params.get("penalty", "l2"), class_weight=params.get("class_weight"), solver="liblinear", max_iter=3000, random_state=seed)
    if model_name == "linear_svm":
        return LinearSVC(C=params["C"], class_weight=params.get("class_weight"), max_iter=12000, dual="auto", random_state=seed)
    if model_name == "rbf_svm":
        return SVC(C=params["C"], gamma=params["gamma"], class_weight=params.get("class_weight"), probability=False, random_state=seed)
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params.get("max_depth"), min_samples_leaf=params.get("min_samples_leaf", 1), class_weight=params.get("class_weight"), max_features="sqrt", n_jobs=1, random_state=seed)
    if model_name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=params["n_estimators"], max_depth=params.get("max_depth"), min_samples_leaf=params.get("min_samples_leaf", 1), class_weight=params.get("class_weight"), max_features="sqrt", n_jobs=1, random_state=seed)
    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is not installed")
        return LGBMClassifier(n_estimators=params["n_estimators"], learning_rate=params["learning_rate"], max_depth=params["max_depth"], subsample=params["subsample"], colsample_bytree=params["colsample_bytree"], objective="binary", class_weight="balanced", n_jobs=1, random_state=seed, verbose=-1)
    raise ValueError(f"Unsupported model: {model_name}")


def build_pipeline(model_name: str, params: Dict, seed: int):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("vt", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("model", build_base_model(model_name, params, seed)),
    ])


def scores_from_model(model, x):
    clf = model.named_steps["model"]
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(model[:-1].transform(x))
        if isinstance(p, list):
            p = p[0]
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[1] > 1:
            return p[:, 1]
        return p.ravel()
    if hasattr(clf, "decision_function"):
        return np.asarray(clf.decision_function(model[:-1].transform(x))).ravel()
    return None


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def decision_to_label(scores):
    scores = np.asarray(scores)
    if scores.size == 0:
        return np.array([], dtype=int)
    if np.nanmin(scores) >= 0.0 and np.nanmax(scores) <= 1.0:
        return (scores >= 0.5).astype(int)
    return (scores >= 0.0).astype(int)


def build_train_subset(train_df, target, formulation):
    if formulation == "supervised_all_negatives":
        work = train_df.copy()
        y = work[target].astype(int).to_numpy()
        return work, y, {"n_train_used": int(len(work)), "n_positive_train": int(y.sum()), "negative_definition": "all_non_target_patients"}
    if formulation == "supervised_clean_negatives":
        mask = (train_df[target].astype(int) == 1) | (train_df["Sin_complicación"].astype(int) == 1)
        work = train_df[mask].copy()
        y = work[target].astype(int).to_numpy()
        return work, y, {"n_train_used": int(len(work)), "n_positive_train": int(y.sum()), "negative_definition": "sin_complicacion_only"}
    if formulation == "pu_learning":
        work = train_df.copy()
        y = work[target].astype(int).to_numpy()
        return work, y, {"n_train_used": int(len(work)), "n_positive_train": int(y.sum()), "unlabeled_train": int((y == 0).sum()), "pu_method": "non_negative_unlabeled_as_negative_with_elkan_noto_score_adjustment"}
    raise ValueError(f"Unsupported formulation: {formulation}")


def fit_and_predict(train_df, test_df, feature_cols, target, formulation, model_name, params, seed):
    train_used, y_train, meta = build_train_subset(train_df, target, formulation)
    x_train = train_used[feature_cols].apply(pd.to_numeric, errors="coerce")
    x_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if len(np.unique(y_train)) < 2:
        const = int(y_train[0]) if len(y_train) else 0
        y_pred = np.full(len(test_df), const, dtype=int)
        y_score = np.full(len(test_df), float(const), dtype=float)
        meta["constant_model"] = True
        return y_pred, y_score, meta
    model = build_pipeline(model_name, params, seed)
    model.fit(x_train, y_train)
    scores = scores_from_model(model, x_test)
    if scores is None:
        y_pred = model.predict(x_test)
        y_score = y_pred.astype(float)
    else:
        y_score = np.asarray(scores, dtype=float)
        if formulation == "pu_learning":
            train_scores = scores_from_model(model, x_train)
            if train_scores is not None:
                ts = np.asarray(train_scores, dtype=float)
                if ts.min() < 0 or ts.max() > 1:
                    ts = sigmoid(ts)
                    y_score = sigmoid(y_score)
                c = float(np.clip(np.mean(ts[y_train == 1]), 1e-3, 1.0)) if np.any(y_train == 1) else 1.0
                y_score = np.clip(y_score / c, 0.0, 1.0)
                meta["pu_c_estimate"] = c
        y_pred = decision_to_label(y_score)
    meta["constant_model"] = False
    return y_pred.astype(int), np.asarray(y_score, dtype=float), meta


def fold_metrics(y_true, y_pred, y_score):
    out = {
        "n_test": int(len(y_true)),
        "support": int(np.sum(y_true)),
        "predicted_positive": int(np.sum(y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    try:
        if len(np.unique(y_true)) == 2 and y_score is not None and np.isfinite(y_score).all():
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
            out["average_precision"] = float(average_precision_score(y_true, y_score))
        else:
            out["roc_auc"] = None
            out["average_precision"] = None
    except Exception:
        out["roc_auc"] = None
        out["average_precision"] = None
    return out


def main():
    args = parse_args()
    row = read_manifest_row(args.manifest, args.task_id)
    params = json.loads(row["params_json"])
    seed = int(row.get("seed", 42)) + int(row["task_id"])
    task_tag = f"task_{int(row['task_id']):04d}__{safe_name(row['formulation'])}__{safe_name(row['feature_set'])}__{safe_name(row['target'])}__{safe_name(row['model'])}__{params_short_hash(params)}"
    output_root = Path(os.path.expandvars(row["output_root"])).expanduser()
    out_dir = output_root / task_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat()
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({**row, "params": params, "started_at": started}, f, indent=2, ensure_ascii=False)
    df, feature_cols = load_data(row)
    folds = pd.read_csv(row["folds_csv"])
    if args.dry_run:
        print(json.dumps({"task_id": int(row["task_id"]), "status": "dry_run_ok", "n_patients": len(df), "n_features": len(feature_cols), "n_folds": int(folds["fold"].nunique())}, indent=2))
        return
    pred_rows = []
    metric_rows = []
    fold_meta_rows = []
    t0 = time.time()
    for fold in sorted(folds["fold"].unique()):
        train_ids = set(folds[(folds["fold"] == fold) & (folds["split"] == "train")]["patient_id"].astype(str))
        test_ids = set(folds[(folds["fold"] == fold) & (folds["split"] == "test")]["patient_id"].astype(str))
        train_df = df[df["patient_id"].isin(train_ids)].copy()
        test_df = df[df["patient_id"].isin(test_ids)].copy()
        y_true = test_df[row["target"]].astype(int).to_numpy()
        y_pred, y_score, meta = fit_and_predict(train_df, test_df, feature_cols, row["target"], row["formulation"], row["model"], params, seed + int(fold))
        fm = fold_metrics(y_true, y_pred, y_score)
        metric_rows.append({"fold": int(fold), **fm})
        fold_meta_rows.append({"fold": int(fold), "n_train": int(len(train_df)), "n_test": int(len(test_df)), **meta})
        for pid, yt, yp, ys in zip(test_df["patient_id"].astype(str), y_true, y_pred, y_score):
            pred_rows.append({"fold": int(fold), "patient_id": pid, "y_true": int(yt), "y_pred": int(yp), "y_score": float(ys) if np.isfinite(ys) else None})
    pred_df = pd.DataFrame(pred_rows)
    metrics_df = pd.DataFrame(metric_rows)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(fold_meta_rows).to_csv(out_dir / "fold_metadata.csv", index=False)
    y_true = pred_df["y_true"].to_numpy()
    y_pred = pred_df["y_pred"].to_numpy()
    y_score = pred_df["y_score"].to_numpy(dtype=float)
    summary = fold_metrics(y_true, y_pred, y_score)
    summary.update({
        "task_id": int(row["task_id"]),
        "status": "completed",
        "formulation": row["formulation"],
        "feature_set": row["feature_set"],
        "target": row["target"],
        "model": row["model"],
        "params": params,
        "n_features": int(len(feature_cols)),
        "n_patients": int(len(df)),
        "n_folds_executed": int(metrics_df["fold"].nunique()),
        "elapsed_seconds": round(time.time() - t0, 3),
        "completed_at": datetime.now().isoformat(),
    })
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
