#!/usr/bin/env python3

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
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hidden_positive_simulation import (
    apply_hidden_labels,
    select_hidden_positive_ids,
    validate_hidden_ids,
)

METHODS = ["PN", "Bagging_PU"]
C_VALUES = [0.01, 0.1, 1.0, 10.0]
N_BAGS = 15
MODEL_SEED = 1729
TARGET_COLUMNS = {"Hemorragia", "Neumotórax", "Sin_complicación", "Sin_complicacion"}
EXCLUDED_FEATURES = {
    "patient_id",
    "Sexo_binaria",
    "image_path",
    "image_size",
    "image_spacing",
    "y_true",
    "y_observed",
    "is_hidden_positive",
}


def stable_seed(*parts):
    text = "|".join(map(str, parts))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)


def probability_scores(model, frame):
    scores = np.asarray(model.predict_proba(frame))
    return scores[:, 1] if scores.ndim == 2 else scores.ravel()


def build_model(c_value, seed):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(0.0)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def safe_rank_metric(metric, y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) != 2 or not np.isfinite(y_score).all():
        return None
    return float(metric(y_true, y_score))


def binary_metrics(y_true, y_pred, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
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
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "gmean": float(math.sqrt(sensitivity * specificity)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": safe_rank_metric(roc_auc_score, y_true, y_score),
        "average_precision": safe_rank_metric(average_precision_score, y_true, y_score),
        "brier_score": float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0))),
    }


def choose_threshold(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    finite = np.unique(y_score[np.isfinite(y_score)])
    if len(finite) > 201:
        finite = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 201)))
    candidates = np.unique(np.r_[0.0, finite, 1.0])
    ranked = []
    for threshold in candidates:
        metrics = binary_metrics(y_true, (y_score >= threshold).astype(int), y_score)
        ranked.append(
            (
                metrics["gmean"],
                metrics["mcc"],
                metrics["f1"],
                -abs(float(threshold) - 0.5),
                float(threshold),
            )
        )
    return max(ranked)[4]


def fit_predict(method, train, evaluation, features, c_value, seed, n_bags=N_BAGS):
    if method == "PN":
        model = build_model(c_value, seed)
        model.fit(train[features], train["y_observed"].astype(int))
        return probability_scores(model, evaluation[features])
    if method != "Bagging_PU":
        raise ValueError("Unknown method: %s" % method)
    positives = train[train["y_observed"] == 1]
    unlabeled = train[train["y_observed"] == 0]
    if len(positives) < 2 or len(unlabeled) < 2:
        raise ValueError("Insufficient observed positives or unlabeled rows for Bagging PU")
    rng = np.random.RandomState(int(seed))
    n_unlabeled = min(len(unlabeled), len(positives))
    scores = []
    for bag_index in range(int(n_bags)):
        sampled_index = rng.choice(len(unlabeled), size=n_unlabeled, replace=False)
        sampled = unlabeled.iloc[sampled_index]
        used = pd.concat([positives, sampled], ignore_index=True)
        model = build_model(c_value, int(seed) + bag_index)
        model.fit(used[features], used["y_observed"].astype(int))
        scores.append(probability_scores(model, evaluation[features]))
    return np.mean(np.vstack(scores), axis=0)


def inner_fold_ids(folds, outer_fold):
    values = []
    for inner_fold in sorted(folds["fold"].unique()):
        if int(inner_fold) == int(outer_fold):
            continue
        ids = set(
            folds.loc[
                (folds["fold"] == inner_fold) & (folds["split"] == "test"),
                "patient_id",
            ].astype(str)
        )
        values.append((int(inner_fold), ids))
    if len(values) != 4:
        raise ValueError("Expected four inner folds")
    return values


def inner_oof_scores(frame, folds, outer_fold, outer_train_ids, features, method, c_value):
    rows = []
    for inner_fold, validation_ids in inner_fold_ids(folds, outer_fold):
        training_ids = set(outer_train_ids) - set(validation_ids)
        train = frame[frame["patient_id"].isin(training_ids)]
        validation = frame[frame["patient_id"].isin(validation_ids)]
        seed = stable_seed(MODEL_SEED, frame.attrs["target"], frame.attrs["feature_set"], outer_fold, inner_fold, method, c_value)
        scores = fit_predict(method, train, validation, features, c_value, seed)
        for patient_id, y_observed, score in zip(
            validation["patient_id"], validation["y_observed"], scores
        ):
            rows.append(
                {
                    "inner_fold": inner_fold,
                    "patient_id": str(patient_id),
                    "y_observed": int(y_observed),
                    "y_score": float(score),
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != len(outer_train_ids) or result["patient_id"].nunique() != len(outer_train_ids):
        raise ValueError("Inner OOF predictions do not cover outer training exactly once")
    return result


def choose_shared_c(frame, folds, outer_fold, outer_train_ids, features):
    rows = []
    tables = {}
    for c_index, c_value in enumerate(C_VALUES):
        predictions = inner_oof_scores(
            frame, folds, outer_fold, outer_train_ids, features, "PN", c_value
        )
        score = safe_rank_metric(
            average_precision_score,
            predictions["y_observed"],
            predictions["y_score"],
        )
        rows.append(
            {
                "c_index": c_index,
                "c_value": c_value,
                "observed_average_precision": -1.0 if score is None else score,
            }
        )
        tables[c_value] = predictions
    ranking = pd.DataFrame(rows).sort_values(
        ["observed_average_precision", "c_index"], ascending=[False, True]
    )
    best_c = float(ranking.iloc[0]["c_value"])
    return best_c, ranking, tables[best_c]


def load_inputs(row):
    labels = pd.read_csv(row["labels_csv"])
    cohort_ids = set(pd.read_csv(row["cohort_ids_csv"])["patient_id"].astype(str))
    labels["patient_id"] = labels["patient_id"].astype(str)
    labels = labels[labels["patient_id"].isin(cohort_ids)].copy()
    folds = pd.read_csv(row["folds_csv"])
    folds["patient_id"] = folds["patient_id"].astype(str)
    feature_frames = {}
    feature_columns = {}
    paths = json.loads(row["feature_paths_json"])
    for feature_set, feature_path in paths.items():
        source = pd.read_csv(feature_path)
        source["patient_id"] = source["patient_id"].astype(str)
        source = source[source["patient_id"].isin(cohort_ids)].copy()
        features = [
            column
            for column in source.columns
            if column not in EXCLUDED_FEATURES
            and column not in TARGET_COLUMNS
            and not column.endswith("_image_path")
            and not column.endswith("_image_size")
            and not column.endswith("_image_spacing")
        ]
        frame = source[["patient_id"] + features].merge(
            labels[["patient_id", row["target"]]], on="patient_id", how="inner"
        )
        frame = frame.sort_values("patient_id").reset_index(drop=True)
        if set(frame["patient_id"]) != cohort_ids:
            raise ValueError("Feature set %s does not cover the shared cohort" % feature_set)
        if "Sexo_binaria" in features:
            raise ValueError("Sexo_binaria must remain excluded")
        feature_frames[feature_set] = frame
        feature_columns[feature_set] = features
    if len(labels) != len(cohort_ids) or labels["patient_id"].nunique() != len(cohort_ids):
        raise ValueError("Labels do not cover the shared cohort exactly once")
    return labels, folds, feature_frames, feature_columns, cohort_ids


def task_row(manifest_path, task_id):
    manifest = pd.read_csv(manifest_path)
    selected = manifest[manifest["task_id"] == int(task_id)]
    if len(selected) != 1:
        raise ValueError("Task ID is not unique in manifest")
    return selected.iloc[0].to_dict(), len(manifest)


def run_task(row, manifest_size):
    output_root = Path(os.path.expandvars(row["output_root"]))
    out = output_root / ("task_%03d" % int(row["task_id"]))
    out.mkdir(parents=True, exist_ok=True)
    labels, folds, feature_frames, feature_columns, cohort_ids = load_inputs(row)
    target = row["target"]
    hide_fraction = float(row["hide_fraction"])
    concealment_seed = int(row["concealment_seed"])
    config = {
        "task_id": int(row["task_id"]),
        "manifest_size": int(manifest_size),
        "target": target,
        "hide_fraction": hide_fraction,
        "concealment_seed": concealment_seed,
        "methods": METHODS,
        "c_values": C_VALUES,
        "n_bags": N_BAGS,
        "model_seed": MODEL_SEED,
        "selection_labels": "observed_labels_inside_outer_training",
        "evaluation_labels": "unaltered_outer_test_ground_truth",
        "created_at": datetime.now().isoformat(),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    mask_rows = []
    prediction_rows = []
    fold_metric_rows = []
    selection_rows = []
    start = time.time()
    for outer_fold in range(1, 6):
        test_ids = set(
            folds.loc[
                (folds["fold"] == outer_fold) & (folds["split"] == "test"),
                "patient_id",
            ].astype(str)
        )
        train_ids = set(cohort_ids) - test_ids
        train_truth = labels[labels["patient_id"].isin(train_ids)][
            ["patient_id", target]
        ].copy()
        mask_seed = stable_seed("concealment", concealment_seed, target, outer_fold)
        hidden_ids = select_hidden_positive_ids(
            train_truth, target, hide_fraction, mask_seed
        )
        validate_hidden_ids(train_truth, target, hidden_ids, test_ids)
        n_train_positive = int(train_truth[target].sum())
        effective_fraction = len(hidden_ids) / n_train_positive
        for patient_id in hidden_ids:
            mask_rows.append(
                {
                    "outer_fold": outer_fold,
                    "patient_id": patient_id,
                    "target": target,
                    "nominal_fraction": hide_fraction,
                    "effective_fraction": effective_fraction,
                    "concealment_seed": concealment_seed,
                    "mask_seed": mask_seed,
                    "n_train_positive": n_train_positive,
                    "n_hidden": len(hidden_ids),
                }
            )
        for feature_set, source_frame in feature_frames.items():
            frame = apply_hidden_labels(source_frame, target, hidden_ids)
            frame.attrs["target"] = target
            frame.attrs["feature_set"] = feature_set
            features = feature_columns[feature_set]
            best_c, c_ranking, pn_inner = choose_shared_c(
                frame, folds, outer_fold, train_ids, features
            )
            for _, item in c_ranking.iterrows():
                selection_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "feature_set": feature_set,
                        "record_type": "c_selection",
                        "method": "PN",
                        "c_value": float(item["c_value"]),
                        "selected": int(float(item["c_value"]) == best_c),
                        "observed_average_precision": float(
                            item["observed_average_precision"]
                        ),
                        "threshold": np.nan,
                    }
                )
            outer_train = frame[frame["patient_id"].isin(train_ids)]
            outer_test = frame[frame["patient_id"].isin(test_ids)]
            for method in METHODS:
                if method == "PN":
                    inner_predictions = pn_inner
                else:
                    inner_predictions = inner_oof_scores(
                        frame,
                        folds,
                        outer_fold,
                        train_ids,
                        features,
                        method,
                        best_c,
                    )
                threshold = choose_threshold(
                    inner_predictions["y_observed"], inner_predictions["y_score"]
                )
                selection_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "feature_set": feature_set,
                        "record_type": "threshold_selection",
                        "method": method,
                        "c_value": best_c,
                        "selected": 1,
                        "observed_average_precision": safe_rank_metric(
                            average_precision_score,
                            inner_predictions["y_observed"],
                            inner_predictions["y_score"],
                        ),
                        "threshold": threshold,
                    }
                )
                final_seed = stable_seed(
                    MODEL_SEED, target, feature_set, outer_fold, method, best_c
                )
                test_scores = fit_predict(
                    method,
                    outer_train,
                    outer_test,
                    features,
                    best_c,
                    final_seed,
                )
                test_predictions = (test_scores >= threshold).astype(int)
                metrics = binary_metrics(
                    outer_test["y_true"], test_predictions, test_scores
                )
                fold_metric_rows.append(
                    {
                        "task_id": int(row["task_id"]),
                        "target": target,
                        "hide_fraction": hide_fraction,
                        "concealment_seed": concealment_seed,
                        "outer_fold": outer_fold,
                        "feature_set": feature_set,
                        "method": method,
                        "c_value": best_c,
                        "threshold": threshold,
                        "n_train_positive_true": n_train_positive,
                        "n_hidden": len(hidden_ids),
                        "effective_fraction": effective_fraction,
                        "n_train_positive_observed": int(outer_train["y_observed"].sum()),
                        **metrics,
                    }
                )
                for patient_id, y_true, y_score, y_pred in zip(
                    outer_test["patient_id"],
                    outer_test["y_true"],
                    test_scores,
                    test_predictions,
                ):
                    prediction_rows.append(
                        {
                            "task_id": int(row["task_id"]),
                            "target": target,
                            "hide_fraction": hide_fraction,
                            "concealment_seed": concealment_seed,
                            "outer_fold": outer_fold,
                            "feature_set": feature_set,
                            "method": method,
                            "patient_id": str(patient_id),
                            "y_true": int(y_true),
                            "y_score": float(y_score),
                            "y_pred": int(y_pred),
                        }
                    )
    masks = pd.DataFrame(mask_rows)
    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    selections = pd.DataFrame(selection_rows)
    masks.to_csv(out / "hidden_masks.csv", index=False)
    predictions.to_csv(out / "outer_oof_predictions.csv", index=False)
    fold_metrics.to_csv(out / "outer_fold_metrics.csv", index=False)
    selections.to_csv(out / "inner_selection.csv", index=False)
    oof_rows = []
    for (feature_set, method), group in predictions.groupby(
        ["feature_set", "method"], sort=True
    ):
        if group["outer_fold"].nunique() != 5:
            raise ValueError("Missing outer folds in OOF predictions")
        if len(group) != len(cohort_ids) or group["patient_id"].nunique() != len(
            cohort_ids
        ):
            raise ValueError("OOF predictions do not cover each patient exactly once")
        metrics = binary_metrics(group["y_true"], group["y_pred"], group["y_score"])
        oof_rows.append(
            {
                "task_id": int(row["task_id"]),
                "target": target,
                "hide_fraction": hide_fraction,
                "concealment_seed": concealment_seed,
                "feature_set": feature_set,
                "method": method,
                "n_outer_folds": 5,
                **metrics,
            }
        )
    oof_metrics = pd.DataFrame(oof_rows)
    oof_metrics.to_csv(out / "oof_metrics.csv", index=False)
    if len(oof_metrics) != 6 or len(fold_metrics) != 30:
        raise ValueError("Unexpected method-feature-fold coverage")
    expected_hidden_rows = int(fold_metrics.groupby("outer_fold")["n_hidden"].first().sum())
    if len(masks) != expected_hidden_rows:
        raise ValueError("Hidden mask artifact is incomplete")
    summary = {
        "status": "completed",
        "task_id": int(row["task_id"]),
        "target": target,
        "hide_fraction": hide_fraction,
        "concealment_seed": concealment_seed,
        "n_outer_folds": 5,
        "n_feature_sets": 3,
        "n_methods": 2,
        "n_oof_rows": int(len(predictions)),
        "n_oof_metric_rows": int(len(oof_metrics)),
        "n_hidden_mask_rows": int(len(masks)),
        "ground_truth_untouched": True,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    row, manifest_size = task_row(args.manifest, args.task_id)
    run_task(row, manifest_size)


if __name__ == "__main__":
    main()
