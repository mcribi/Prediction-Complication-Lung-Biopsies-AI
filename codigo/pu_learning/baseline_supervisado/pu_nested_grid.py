#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

TARGETS = ["Hemorragia", "Neumotórax"]
FORMULATIONS = [
    "supervised_all_negatives",
    "supervised_clean_negatives",
    "pu_naive",
    "pu_elkan_noto_oof",
    "pu_bagging",
]
EXCLUDED_FEATURES = {"patient_id", "Sexo_binaria", "image_path", "image_size", "image_spacing"}


def clinical_feature_filter(columns):
    return [c for c in columns if c not in EXCLUDED_FEATURES and not c.endswith("_image_path") and not c.endswith("_image_size") and not c.endswith("_image_spacing")]


def paired_seed(base_seed, outer_fold, model_family, candidate_index, inner_fold=0):
    text = f"{base_seed}|{outer_fold}|{inner_fold}|{model_family}|{candidate_index}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def safe_auc(metric, y_true, y_score):
    try:
        if len(np.unique(y_true)) != 2 or not np.isfinite(y_score).all():
            return None
        return float(metric(y_true, y_score))
    except Exception:
        return None


def binary_metrics(y_true, y_pred, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    return {
        "n": int(len(y_true)), "support": int(y_true.sum()),
        "predicted_positive": int(y_pred.sum()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity), "recall": float(sensitivity),
        "specificity": float(specificity), "npv": float(npv),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "gmean": float(math.sqrt(sensitivity * specificity)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": safe_auc(roc_auc_score, y_true, y_score),
        "average_precision": safe_auc(average_precision_score, y_true, y_score),
        "brier_score": float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0))),
    }


def choose_threshold(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    values = np.unique(y_score[np.isfinite(y_score)])
    if not len(values):
        return 0.5, binary_metrics(y_true, np.zeros_like(y_true), np.zeros_like(y_score))
    if len(values) > 201:
        values = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 201)))
    candidates = np.unique(np.r_[0.0, values, 1.0])
    ranked = []
    for threshold in candidates:
        metrics = binary_metrics(y_true, (y_score >= threshold).astype(int), y_score)
        ranked.append((metrics["gmean"], metrics["mcc"], metrics["f1"], -abs(threshold - 0.5), float(threshold), metrics))
    best = max(ranked, key=lambda x: x[:5])
    return best[4], best[5]


def estimate_c_from_oof(y_observed, oof_scores):
    y_observed = np.asarray(y_observed, dtype=int)
    scores = np.asarray(oof_scores, dtype=float)
    positive = scores[(y_observed == 1) & np.isfinite(scores)]
    if not len(positive):
        return 1.0
    return float(np.clip(np.mean(positive), 1e-3, 1.0))


def validate_outer_folds(folds, expected_ids, n_splits=5):
    required = {"fold", "split", "patient_id"}
    if not required.issubset(folds.columns):
        raise ValueError(f"Missing fold columns: {required - set(folds.columns)}")
    folds = folds.copy()
    folds["patient_id"] = folds["patient_id"].astype(str)
    if set(folds["fold"].unique()) != set(range(1, n_splits + 1)):
        raise ValueError("Outer folds are not exactly 1..n_splits")
    seen = []
    expected_ids = set(map(str, expected_ids))
    for fold in range(1, n_splits + 1):
        tr = set(folds[(folds.fold == fold) & (folds.split == "train")].patient_id)
        te = set(folds[(folds.fold == fold) & (folds.split == "test")].patient_id)
        if tr & te or tr | te != expected_ids or not te:
            raise ValueError(f"Invalid train/test partition in fold {fold}")
        seen.extend(te)
    if len(seen) != len(expected_ids) or set(seen) != expected_ids:
        raise ValueError("External test folds do not cover each patient exactly once")


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def probability_scores(model, x):
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(x))
        return p[:, 1] if p.ndim == 2 else p.ravel()
    if hasattr(model, "decision_function"):
        return sigmoid(model.decision_function(x))
    return np.asarray(model.predict(x), dtype=float)


def candidate_grid():
    rows = []
    def add(family, **params): rows.append({"family": family, "params": params})
    for c in [0.01, 0.1, 1.0, 10.0]:
        for cw in [None, "balanced"]: add("logistic_regression", C=c, class_weight=cw)
    for c in [0.01, 0.1, 1.0, 10.0]:
        for cw in [None, "balanced"]: add("linear_svm", C=c, class_weight=cw)
    for c in [0.1, 1.0, 10.0]:
        for gamma in ["scale", "auto"]: add("rbf_svm", C=c, gamma=gamma, class_weight="balanced")
    for n in [300, 700]:
        for depth in [None, 5, 10]:
            for leaf in [1, 3]: add("random_forest", n_estimators=n, max_depth=depth, min_samples_leaf=leaf, class_weight="balanced_subsample")
            for leaf in [1, 3]: add("extra_trees", n_estimators=n, max_depth=depth, min_samples_leaf=leaf, class_weight="balanced_subsample")
    for lr in [0.03, 0.1]:
        for leaves in [7, 15]: add("lightgbm", n_estimators=300, learning_rate=lr, num_leaves=leaves, max_depth=-1)
        for depth in [2, 3]: add("xgboost", n_estimators=300, learning_rate=lr, max_depth=depth)
    for lr in [0.03, 0.1]:
        for leaf in [10, 20]: add("hist_gradient_boosting", learning_rate=lr, max_iter=200, max_leaf_nodes=15, min_samples_leaf=leaf)
    for hidden in [(32,), (64,), (64, 32)]:
        for alpha in [0.0001, 0.01]: add("mlp", hidden_layer_sizes=hidden, alpha=alpha)
    return rows


def build_estimator(candidate, seed):
    family, p = candidate["family"], dict(candidate["params"])
    if family == "logistic_regression":
        clf = LogisticRegression(solver="liblinear", max_iter=5000, random_state=seed, **p)
    elif family == "linear_svm":
        clf = LinearSVC(max_iter=20000, dual="auto", random_state=seed, **p)
    elif family == "rbf_svm":
        clf = SVC(probability=False, random_state=seed, **p)
    elif family == "random_forest":
        clf = RandomForestClassifier(max_features="sqrt", n_jobs=1, random_state=seed, **p)
    elif family == "extra_trees":
        clf = ExtraTreesClassifier(max_features="sqrt", n_jobs=1, random_state=seed, **p)
    elif family == "lightgbm":
        if LGBMClassifier is None: raise RuntimeError("lightgbm is unavailable")
        clf = LGBMClassifier(objective="binary", class_weight="balanced", n_jobs=1, random_state=seed, verbose=-1, **p)
    elif family == "xgboost":
        if XGBClassifier is None: raise RuntimeError("xgboost is unavailable")
        clf = XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_jobs=1, random_state=seed, **p)
    elif family == "hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(random_state=seed, **p)
    elif family == "mlp":
        clf = MLPClassifier(max_iter=1000, early_stopping=True, validation_fraction=0.15, random_state=seed, **p)
    else:
        raise ValueError(f"Unknown family {family}")
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("vt", VarianceThreshold(0.0)), ("scaler", StandardScaler()), ("model", clf)])


def training_subset(df, target, formulation):
    if formulation == "supervised_clean_negatives":
        return df[(df[target] == 1) | (df["Sin_complicación"] == 1)].copy()
    return df.copy()


def fit_standard(train, test, features, target, formulation, candidate, seed):
    used = training_subset(train, target, formulation)
    model = build_estimator(candidate, seed)
    model.fit(used[features], used[target].astype(int))
    return probability_scores(model, test[features]), model, {"n_train_used": len(used)}


def fit_bagging(train, test, features, target, candidate, seed, n_bags=15, unlabeled_ratio=1.0):
    rng = np.random.RandomState(seed)
    positives = train[train[target] == 1]
    unlabeled = train[train[target] == 0]
    if not len(positives) or not len(unlabeled):
        return fit_standard(train, test, features, target, "pu_naive", candidate, seed)
    scores = []
    n_u = min(len(unlabeled), max(1, int(round(len(positives) * unlabeled_ratio))))
    for bag in range(n_bags):
        sampled = unlabeled.iloc[rng.choice(len(unlabeled), size=n_u, replace=len(unlabeled) < n_u)]
        used = pd.concat([positives, sampled], ignore_index=True)
        model = build_estimator(candidate, seed + bag)
        model.fit(used[features], used[target].astype(int))
        scores.append(probability_scores(model, test[features]))
    return np.mean(scores, axis=0), None, {"n_bags": n_bags, "unlabeled_per_bag": n_u}


def fit_predict(train, test, features, target, formulation, candidate, seed):
    if formulation == "pu_bagging":
        return fit_bagging(train, test, features, target, candidate, seed)
    return fit_standard(train, test, features, target, formulation, candidate, seed)


def adjust_elkan(scores, c):
    return np.clip(np.asarray(scores, dtype=float) / float(c), 0.0, 1.0)


def evaluate_candidate_inner(df, folds, outer_fold, features, target, formulation, candidate, candidate_index, base_seed):
    inner_folds = [f for f in sorted(folds.fold.unique()) if f != outer_fold]
    rows = []
    for inner_fold in inner_folds:
        val_ids = set(folds[(folds.fold == inner_fold) & (folds.split == "test")].patient_id.astype(str))
        inner_pool_ids = set(folds[(folds.fold != outer_fold) & (folds.split == "test")].patient_id.astype(str))
        train = df[df.patient_id.astype(str).isin(inner_pool_ids - val_ids)]
        val = df[df.patient_id.astype(str).isin(val_ids)]
        seed = paired_seed(base_seed, outer_fold, candidate["family"], candidate_index, inner_fold)
        score, _, _ = fit_predict(train, val, features, target, formulation, candidate, seed)
        for pid, y, s in zip(val.patient_id.astype(str), val[target].astype(int), score):
            rows.append({"inner_fold": inner_fold, "patient_id": pid, "y_true": int(y), "raw_score": float(s)})
    pred = pd.DataFrame(rows)
    c = 1.0
    if formulation == "pu_elkan_noto_oof":
        c = estimate_c_from_oof(pred.y_true, pred.raw_score)
    pred["y_score"] = adjust_elkan(pred.raw_score, c) if formulation == "pu_elkan_noto_oof" else pred.raw_score
    threshold, threshold_metrics = choose_threshold(pred.y_true, pred.y_score)
    ap = safe_auc(average_precision_score, pred.y_true.to_numpy(), pred.y_score.to_numpy())
    return pred, {"average_precision": -1.0 if ap is None else ap, "gmean": threshold_metrics["gmean"], "mcc": threshold_metrics["mcc"], "threshold": threshold, "c_estimate": c}


def select_candidate(df, folds, outer_fold, features, target, formulation, candidates, base_seed):
    summaries = []
    for idx, candidate in enumerate(candidates):
        pred, score = evaluate_candidate_inner(df, folds, outer_fold, features, target, formulation, candidate, idx, base_seed)
        summaries.append({"candidate_index": idx, "family": candidate["family"], "params_json": json.dumps(candidate["params"], sort_keys=True), **score})
    table = pd.DataFrame(summaries).sort_values(["average_precision", "gmean", "mcc", "candidate_index"], ascending=[False, False, False, True])
    best = table.iloc[0]
    return int(best.candidate_index), table


def load_task(manifest_path, task_id):
    manifest = pd.read_csv(manifest_path)
    row = manifest[manifest.task_id == task_id]
    if len(row) != 1: raise ValueError(f"Task {task_id} not found uniquely")
    return row.iloc[0].to_dict()


def run_task(row, smoke=False):
    out = Path(os.path.expandvars(row["output_root"])) / f"task_{int(row['task_id']):02d}__{row['target']}__{row['feature_set']}__{row['formulation']}"
    out.mkdir(parents=True, exist_ok=True)
    x = pd.read_csv(row["features_csv"]); y = pd.read_csv(row["labels_csv"]); ids = set(pd.read_csv(row["cohort_ids_csv"]).patient_id.astype(str))
    x["patient_id"] = x.patient_id.astype(str); y["patient_id"] = y.patient_id.astype(str)
    df = x.merge(y[["patient_id", "Hemorragia", "Neumotórax", "Sin_complicación"]], on="patient_id", how="inner")
    df = df[df.patient_id.isin(ids)].sort_values("patient_id").reset_index(drop=True)
    features = clinical_feature_filter([c for c in x.columns if c != "patient_id"])
    if "Sexo_binaria" in features or "Edad" not in features and row["feature_set"] != "radiomics":
        raise ValueError("Clinical feature policy violated")
    folds = pd.read_csv(row["folds_csv"]); folds["patient_id"] = folds.patient_id.astype(str)
    validate_outer_folds(folds, set(df.patient_id), 5)
    candidates = candidate_grid()
    if smoke: candidates = candidates[:2]
    config = {**row, "features": features, "excluded_features": sorted(EXCLUDED_FEATURES), "n_patients": len(df), "n_features": len(features), "n_candidates": len(candidates), "selection_metric": "average_precision", "tie_breakers": ["gmean", "mcc", "candidate_index"], "threshold_rule": "maximize_gmean_then_mcc_then_f1", "created_at": datetime.now().isoformat()}
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    outer_predictions, fold_summaries = [], []
    t0 = time.time()
    outer_values = [1] if smoke else list(range(1, 6))
    for outer_fold in outer_values:
        test_ids = set(folds[(folds.fold == outer_fold) & (folds.split == "test")].patient_id)
        train_ids = set(df.patient_id) - test_ids
        train, test = df[df.patient_id.isin(train_ids)], df[df.patient_id.isin(test_ids)]
        best_idx, inner_table = select_candidate(df, folds, outer_fold, features, row["target"], row["formulation"], candidates, int(row["seed"]))
        inner_table.to_csv(out / f"outer_fold_{outer_fold}_inner_grid.csv", index=False)
        candidate = candidates[best_idx]
        seed = paired_seed(int(row["seed"]), outer_fold, candidate["family"], best_idx)
        raw_score, _, meta = fit_predict(train, test, features, row["target"], row["formulation"], candidate, seed)
        chosen = inner_table[inner_table.candidate_index == best_idx].iloc[0]
        c = float(chosen.c_estimate)
        score = adjust_elkan(raw_score, c) if row["formulation"] == "pu_elkan_noto_oof" else raw_score
        threshold = float(chosen.threshold)
        pred = (score >= threshold).astype(int)
        metrics = binary_metrics(test[row["target"]].astype(int), pred, score)
        fold_summaries.append({"outer_fold": outer_fold, "candidate_index": best_idx, "family": candidate["family"], "params_json": json.dumps(candidate["params"], sort_keys=True), "threshold": threshold, "c_estimate": c, **meta, **metrics})
        for pid, yt, yp, ys, rs in zip(test.patient_id, test[row["target"]].astype(int), pred, score, raw_score):
            outer_predictions.append({"outer_fold": outer_fold, "patient_id": pid, "y_true": int(yt), "y_pred": int(yp), "y_score": float(ys), "raw_score": float(rs), "threshold": threshold})
    pred_df = pd.DataFrame(outer_predictions); fold_df = pd.DataFrame(fold_summaries)
    pred_df.to_csv(out / "outer_oof_predictions.csv", index=False); fold_df.to_csv(out / "outer_fold_metrics.csv", index=False)
    overall = binary_metrics(pred_df.y_true, pred_df.y_pred, pred_df.y_score)
    summary = {"task_id": int(row["task_id"]), "status": "smoke_completed" if smoke else "completed", "target": row["target"], "feature_set": row["feature_set"], "formulation": row["formulation"], "n_outer_folds": int(fold_df.outer_fold.nunique()), "elapsed_seconds": round(time.time() - t0, 3), **overall}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--task-id", type=int, required=True); p.add_argument("--smoke", action="store_true")
    args = p.parse_args(); run_task(load_task(args.manifest, args.task_id), smoke=args.smoke)


if __name__ == "__main__":
    main()
