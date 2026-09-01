#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RULES = {
    "hemorrhage_small_nodule": {"description": "tamano_nodulo_mm <= 31.05", "target": "Hemorragia"},
    "pneumothorax_shallow_small_nodule": {
        "description": "profundidad_centroidal_pleura_mm <= 10.21 and tamano_nodulo_mm <= 42.80",
        "target": "Neumotórax",
    },
    "no_comp_no_lung_pathology": {"description": "Sin_patología_pulmonar == 1", "target": "Sin_complicación"},
    "no_comp_large_nodule": {"description": "tamano_nodulo_mm > 42.50", "target": "Sin_complicación"},
    "hemorrhage_age_over_60": {"description": "Edad > 60", "target": "Hemorragia"},
    "pneumothorax_tobacco_any": {"description": "Tabac_any == 1", "target": "Neumotórax"},
}

MODEL_NAMES = ["dummy", "logistic", "linear_svm", "gaussian_nb", "random_forest", "extra_trees", "lightgbm"]
TARGETS = ["Hemorragia", "Neumotórax"]
THRESHOLD_CANDIDATES = [round(x, 2) for x in np.arange(0.10, 0.901, 0.05)]


def apply_rule(frame, rule_id):
    if rule_id not in RULES:
        raise KeyError("Unknown rule_id: {}".format(rule_id))
    if rule_id == "hemorrhage_small_nodule":
        mask = frame["tamano_nodulo_mm"] <= 31.05
    elif rule_id == "pneumothorax_shallow_small_nodule":
        mask = (frame["profundidad_centroidal_pleura_mm"] <= 10.21) & (frame["tamano_nodulo_mm"] <= 42.80)
    elif rule_id == "no_comp_no_lung_pathology":
        mask = frame["Sin_patología_pulmonar"] == 1
    elif rule_id == "no_comp_large_nodule":
        mask = frame["tamano_nodulo_mm"] > 42.50
    elif rule_id == "hemorrhage_age_over_60":
        mask = frame["Edad"] > 60
    else:
        mask = frame["Tabac_any"] == 1
    return mask.fillna(False).astype(bool)


def derive_no_complication(predictions):
    predictions = np.asarray(predictions, dtype=int)
    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("Expected an n x 2 prediction matrix")
    return (predictions.sum(axis=1) == 0).astype(int)


def choose_threshold(y_true, scores, candidates=THRESHOLD_CANDIDATES):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    ranked = []
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        value = f1_score(y_true, pred, zero_division=0)
        ranked.append((float(value), -abs(float(threshold) - 0.5), -float(threshold), float(threshold)))
    best = max(ranked)
    return best[3], best[0]


def validate_outer_splits(assignments, patient_ids):
    required = {"fold", "split", "patient_id"}
    if not required.issubset(assignments.columns):
        raise ValueError("Outer assignments require fold, split and patient_id")
    frame = assignments.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    expected_ids = set(map(str, patient_ids))
    folds = sorted(frame["fold"].unique().tolist())
    if folds != [1, 2, 3, 4, 5]:
        raise ValueError("Expected exactly folds 1..5")
    result = {}
    test_counts = {patient_id: 0 for patient_id in expected_ids}
    for fold in folds:
        part = frame[frame["fold"] == fold]
        train = set(part.loc[part["split"] == "train", "patient_id"])
        test = set(part.loc[part["split"] == "test", "patient_id"])
        if train & test:
            raise ValueError("Train/test overlap in fold {}".format(fold))
        if train | test != expected_ids:
            raise ValueError("Fold {} does not contain the full cohort".format(fold))
        for patient_id in test:
            test_counts[patient_id] += 1
        result[int(fold)] = {"train": sorted(train), "test": sorted(test)}
    if any(value != 1 for value in test_counts.values()):
        raise ValueError("Every patient must occur in outer test exactly once")
    return result


def build_estimator(model_name, seed=1337, n_features=None):
    if model_name == "dummy":
        model = DummyClassifier(strategy="prior")
    elif model_name == "logistic":
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, solver="liblinear", random_state=seed)
    elif model_name == "linear_svm":
        model = SVC(C=1.0, kernel="linear", probability=True, class_weight="balanced", random_state=seed)
    elif model_name == "gaussian_nb":
        model = GaussianNB()
    elif model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=1, random_state=seed)
    elif model_name == "extra_trees":
        model = ExtraTreesClassifier(n_estimators=300, max_depth=5, min_samples_leaf=3, class_weight="balanced", n_jobs=1, random_state=seed)
    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=200, max_depth=3, num_leaves=15, learning_rate=0.03, class_weight="balanced", subsample=0.8, colsample_bytree=0.8, n_jobs=1, random_state=seed, verbosity=-1)
    else:
        raise KeyError("Unknown model_name: {}".format(model_name))
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.01)),
            ("select", SelectKBest(score_func=f_classif, k=min(20, n_features) if n_features else 20)),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def positive_scores(estimator, x):
    probabilities = estimator.predict_proba(x)
    classes = list(estimator.named_steps["model"].classes_)
    if 1 not in classes:
        return np.zeros(len(x), dtype=float)
    return np.asarray(probabilities[:, classes.index(1)], dtype=float)


def constant_scores(value, n):
    return np.full(n, float(value), dtype=float)


def fit_scores(estimator, x_train, y_train, x_test):
    y_train = np.asarray(y_train, dtype=int)
    unique = np.unique(y_train)
    if len(unique) < 2:
        return constant_scores(unique[0], len(x_test)), None
    fitted = clone(estimator).fit(x_train, y_train)
    return positive_scores(fitted, x_test), fitted


def inner_oof_threshold(estimator, x, y, seed):
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    minimum = int(counts.min())
    if minimum < 2:
        return 0.5, math.nan, 0
    n_splits = min(4, minimum)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = np.zeros(len(y), dtype=float)
    for train_index, val_index in splitter.split(x, y):
        fold_scores, _ = fit_scores(estimator, x.iloc[train_index], y[train_index], x.iloc[val_index])
        scores[val_index] = fold_scores
    threshold, value = choose_threshold(y, scores)
    return threshold, value, n_splits


def safe_auc(y_true, scores):
    return float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else math.nan


def safe_average_precision(y_true, scores):
    return float(average_precision_score(y_true, scores)) if np.sum(y_true) > 0 else math.nan


def compute_metrics(y_true, predictions, scores):
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    scores = np.asarray(scores, dtype=float)
    metrics = {
        "n_samples": int(len(y_true)),
        "two_outputs_f1_micro": float(f1_score(y_true, predictions, average="micro", zero_division=0)),
        "two_outputs_f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "two_outputs_precision_micro": float(precision_score(y_true, predictions, average="micro", zero_division=0)),
        "two_outputs_recall_micro": float(recall_score(y_true, predictions, average="micro", zero_division=0)),
        "two_outputs_jaccard_micro": float(jaccard_score(y_true, predictions, average="micro", zero_division=0)),
        "two_outputs_subset_accuracy": float(accuracy_score(y_true, predictions)),
        "two_outputs_hamming_loss": float(hamming_loss(y_true, predictions)),
    }
    for index, label in enumerate(TARGETS):
        truth = y_true[:, index]
        pred = predictions[:, index]
        metrics["f1_{}".format(label)] = float(f1_score(truth, pred, zero_division=0))
        metrics["precision_{}".format(label)] = float(precision_score(truth, pred, zero_division=0))
        metrics["recall_{}".format(label)] = float(recall_score(truth, pred, zero_division=0))
        metrics["balanced_accuracy_{}".format(label)] = float(balanced_accuracy_score(truth, pred)) if len(np.unique(truth)) == 2 else math.nan
        metrics["roc_auc_{}".format(label)] = safe_auc(truth, scores[:, index])
        metrics["average_precision_{}".format(label)] = safe_average_precision(truth, scores[:, index])
        metrics["support_{}".format(label)] = int(truth.sum())
    true_no = derive_no_complication(y_true)
    pred_no = derive_no_complication(predictions)
    no_score = 1.0 - np.maximum(scores[:, 0], scores[:, 1])
    metrics["f1_Sin_complicación"] = float(f1_score(true_no, pred_no, zero_division=0))
    metrics["precision_Sin_complicación"] = float(precision_score(true_no, pred_no, zero_division=0))
    metrics["recall_Sin_complicación"] = float(recall_score(true_no, pred_no, zero_division=0))
    metrics["roc_auc_Sin_complicación"] = safe_auc(true_no, no_score)
    metrics["average_precision_Sin_complicación"] = safe_average_precision(true_no, no_score)
    metrics["support_Sin_complicación"] = int(true_no.sum())
    return metrics


def run_experiment(data, feature_columns, outer_splits, rule_id, model_name, seed=1337, max_folds=None):
    frame = data.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame = frame.set_index("patient_id", drop=False)
    member_ids = set(frame.index[apply_rule(frame, rule_id)])
    if not member_ids:
        raise ValueError("Rule {} has no members".format(rule_id))
    estimator = build_estimator(model_name, seed=seed, n_features=len(feature_columns))
    prediction_rows = []
    threshold_rows = []
    fold_metric_rows = []
    selected_folds = sorted(outer_splits)
    if max_folds is not None:
        selected_folds = selected_folds[: int(max_folds)]
    for fold in selected_folds:
        train_all = [patient_id for patient_id in outer_splits[fold]["train"] if patient_id in frame.index]
        test_subgroup = [patient_id for patient_id in outer_splits[fold]["test"] if patient_id in member_ids]
        train_subgroup = [patient_id for patient_id in train_all if patient_id in member_ids]
        if not test_subgroup:
            raise ValueError("Rule {} has no external test patients in fold {}".format(rule_id, fold))
        source_outputs = {}
        for source, train_ids in [("general", train_all), ("specialized", train_subgroup)]:
            if len(train_ids) < 4:
                raise ValueError("Insufficient {} training patients in fold {}".format(source, fold))
            x_train = frame.loc[train_ids, feature_columns].reset_index(drop=True)
            x_test = frame.loc[test_subgroup, feature_columns].reset_index(drop=True)
            scores = np.zeros((len(test_subgroup), 2), dtype=float)
            thresholds = []
            for label_index, label in enumerate(TARGETS):
                y_train = frame.loc[train_ids, label].astype(int).to_numpy()
                threshold, inner_f1, inner_splits = inner_oof_threshold(
                    estimator, x_train, y_train, seed + 100 * fold + label_index
                )
                label_scores, _ = fit_scores(estimator, x_train, y_train, x_test)
                scores[:, label_index] = label_scores
                thresholds.append(threshold)
                threshold_rows.append(
                    {
                        "fold": fold,
                        "source": source,
                        "label": label,
                        "threshold": threshold,
                        "inner_oof_f1": inner_f1,
                        "inner_splits": inner_splits,
                        "n_train": len(train_ids),
                        "positive_train": int(y_train.sum()),
                    }
                )
            source_outputs[source] = {
                "scores": scores,
                "fixed": (scores >= 0.5).astype(int),
                "tuned": (scores >= np.asarray(thresholds)[None, :]).astype(int),
            }
        y_test = frame.loc[test_subgroup, TARGETS].astype(int).to_numpy()
        for row_index, patient_id in enumerate(test_subgroup):
            row = {
                "patient_id": patient_id,
                "fold": fold,
                "y_Hemorragia": int(y_test[row_index, 0]),
                "y_Neumotórax": int(y_test[row_index, 1]),
                "y_Sin_complicación": int(derive_no_complication(y_test[[row_index]])[0]),
            }
            for source in ["general", "specialized"]:
                for label_index, label in enumerate(TARGETS):
                    row["{}_score_{}".format(source, label)] = float(source_outputs[source]["scores"][row_index, label_index])
                    row["{}_fixed_pred_{}".format(source, label)] = int(source_outputs[source]["fixed"][row_index, label_index])
                    row["{}_tuned_pred_{}".format(source, label)] = int(source_outputs[source]["tuned"][row_index, label_index])
                row["{}_fixed_pred_Sin_complicación".format(source)] = int(derive_no_complication(source_outputs[source]["fixed"][[row_index]])[0])
                row["{}_tuned_pred_Sin_complicación".format(source)] = int(derive_no_complication(source_outputs[source]["tuned"][[row_index]])[0])
            prediction_rows.append(row)
        for source in ["general", "specialized"]:
            for decision in ["fixed", "tuned"]:
                values = compute_metrics(y_test, source_outputs[source][decision], source_outputs[source]["scores"])
                fold_metric_rows.append({"fold": fold, "source": source, "decision": decision, **values})
    predictions = pd.DataFrame(prediction_rows).sort_values(["fold", "patient_id"]).reset_index(drop=True)
    metrics = {}
    y_oof = predictions[["y_Hemorragia", "y_Neumotórax"]].to_numpy(dtype=int)
    for source in ["general", "specialized"]:
        scores = predictions[["{}_score_Hemorragia".format(source), "{}_score_Neumotórax".format(source)]].to_numpy(dtype=float)
        for decision in ["fixed", "tuned"]:
            pred = predictions[["{}_{}_pred_Hemorragia".format(source, decision), "{}_{}_pred_Neumotórax".format(source, decision)]].to_numpy(dtype=int)
            metrics["{}_{}".format(source, decision)] = compute_metrics(y_oof, pred, scores)
    return {
        "rule_id": rule_id,
        "model_name": model_name,
        "n_subgroup": int(len(member_ids)),
        "folds_completed": selected_folds,
        "predictions": predictions,
        "thresholds": pd.DataFrame(threshold_rows),
        "fold_metrics": pd.DataFrame(fold_metric_rows),
        "metrics": metrics,
    }


def save_result(result, output_dir, setup):
    output_dir = Path(output_dir)
    folds = sorted(map(int, result["folds_completed"]))
    if folds != [1, 2, 3, 4, 5]:
        raise ValueError("Cannot mark a run complete without all five folds")
    output_dir.mkdir(parents=True, exist_ok=True)
    result["predictions"].to_csv(output_dir / "oof_predictions.csv", index=False)
    result["thresholds"].to_csv(output_dir / "selected_thresholds.csv", index=False)
    result["fold_metrics"].to_csv(output_dir / "fold_metrics.csv", index=False)
    (output_dir / "oof_metrics.json").write_text(json.dumps(result["metrics"], indent=2, allow_nan=True), encoding="utf-8")
    (output_dir / "setup.json").write_text(json.dumps(setup, indent=2, ensure_ascii=False), encoding="utf-8")
    completion = {
        "status": "ok",
        "rule_id": result["rule_id"],
        "model_name": result["model_name"],
        "n_subgroup": int(result["n_subgroup"]),
        "num_folds_completed": 5,
        "all_folds_present": True,
    }
    (output_dir / "COMPLETED.json").write_text(json.dumps(completion, indent=2, ensure_ascii=False), encoding="utf-8")
    return completion


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(features_csv, labels_csv, clinical_csv, geometry_csv, feature_columns_csv, outer_splits_csv):
    features = pd.read_csv(features_csv)
    labels = pd.read_csv(labels_csv)[["patient_id"] + TARGETS]
    clinical_columns = ["patient_id", "Edad", "Tabac_any", "Sin_patología_pulmonar"]
    clinical = pd.read_csv(clinical_csv)[clinical_columns]
    geometry_columns = ["patient_id", "tamano_nodulo_mm", "profundidad_centroidal_pleura_mm"]
    geometry = pd.read_csv(geometry_csv)[geometry_columns]
    feature_columns = pd.read_csv(feature_columns_csv)["feature"].astype(str).tolist()
    for name, frame in [("features", features), ("labels", labels), ("clinical", clinical), ("geometry", geometry)]:
        frame["patient_id"] = frame["patient_id"].astype(str)
        if frame["patient_id"].duplicated().any():
            raise ValueError("Duplicate patient_id in {}".format(name))
    missing_columns = sorted(set(feature_columns) - set(features.columns))
    if missing_columns:
        raise ValueError("Missing radiomic columns: {}".format(missing_columns[:5]))
    data = features[["patient_id"] + feature_columns].merge(labels, on="patient_id", how="inner")
    data = data.merge(clinical, on="patient_id", how="left").merge(geometry, on="patient_id", how="left")
    assignments = pd.read_csv(outer_splits_csv)
    assignments["patient_id"] = assignments["patient_id"].astype(str)
    cohort_ids = sorted(assignments["patient_id"].unique().tolist())
    data = data[data["patient_id"].isin(cohort_ids)].copy()
    if set(data["patient_id"]) != set(cohort_ids):
        missing = sorted(set(cohort_ids) - set(data["patient_id"]))
        raise ValueError("Missing cohort patients after merge: {}".format(missing))
    if data[TARGETS].isna().any().any():
        raise ValueError("Missing target labels")
    outer_splits = validate_outer_splits(assignments, cohort_ids)
    return data, feature_columns, outer_splits


def write_partial_result(result, output_dir, setup):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result["predictions"].to_csv(output_dir / "oof_predictions.csv", index=False)
    result["thresholds"].to_csv(output_dir / "selected_thresholds.csv", index=False)
    result["fold_metrics"].to_csv(output_dir / "fold_metrics.csv", index=False)
    (output_dir / "oof_metrics.json").write_text(json.dumps(result["metrics"], indent=2, allow_nan=True), encoding="utf-8")
    (output_dir / "setup.json").write_text(json.dumps(setup, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "SMOKE.json").write_text(json.dumps({"status": "ok", "folds_completed": result["folds_completed"]}, indent=2), encoding="utf-8")


def main():
    defaults = {
        "features": "/home/mcribilles/tfm/radiomica_multietiqueta/radiomica_grids/multimask_clinical_grid_massive_fix2__2026-06-03__12-53-45/features/radiomics__basic__vessels.csv",
        "labels": "/home/mcribilles/tfm/radiomica_multietiqueta/radiomica_grids/multimask_clinical_grid_massive_fix2__2026-06-03__12-53-45/labels/labels_main_drop_dpleural_only.csv",
        "clinical": "/home/mcribilles/tfm/subgroup_discovery/data/clinical_data_multimodal.csv",
        "geometry": "/home/mcribilles/tfm/subgroup_discovery/data/nodule_geometry_210.csv",
        "feature_columns": "/home/mcribilles/tfm/radiomic_results_two_outputs_derived_thresholds_20260827/R11__two_outputs__derived_no_complication__nested_thresholds/feature_columns.csv",
        "outer_splits": "/home/mcribilles/tfm/radiomic_results_two_outputs_derived_thresholds_20260827/R11__two_outputs__derived_no_complication__nested_thresholds/outer_split_assignments.csv",
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule-id", choices=sorted(RULES))
    parser.add_argument("--model-name", choices=MODEL_NAMES)
    parser.add_argument("--output-dir")
    parser.add_argument("--manifest")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--features-csv", default=defaults["features"])
    parser.add_argument("--labels-csv", default=defaults["labels"])
    parser.add_argument("--clinical-csv", default=defaults["clinical"])
    parser.add_argument("--geometry-csv", default=defaults["geometry"])
    parser.add_argument("--feature-columns-csv", default=defaults["feature_columns"])
    parser.add_argument("--outer-splits-csv", default=defaults["outer_splits"])
    args = parser.parse_args()
    if args.manifest:
        manifest = pd.read_csv(args.manifest)
        task_id = args.task_id
        if task_id is None:
            task_id = int(__import__("os").environ["SLURM_ARRAY_TASK_ID"])
        matches = manifest[manifest["task_id"] == task_id]
        if len(matches) != 1:
            raise ValueError("Expected one manifest row for task_id {}".format(task_id))
        row = matches.iloc[0]
        args.rule_id = str(row["rule_id"])
        args.model_name = str(row["model_name"])
        args.output_dir = str(row["output_dir"])
    data, feature_columns, outer_splits = load_inputs(
        args.features_csv,
        args.labels_csv,
        args.clinical_csv,
        args.geometry_csv,
        args.feature_columns_csv,
        args.outer_splits_csv,
    )
    if args.preflight:
        report = {rule_id: {"n": int(apply_rule(data, rule_id).sum()), "target": RULES[rule_id]["target"]} for rule_id in RULES}
        print(json.dumps({"cohort": len(data), "n_features": len(feature_columns), "rules": report}, indent=2, ensure_ascii=False))
        return
    if not args.rule_id or not args.model_name or not args.output_dir:
        parser.error("rule-id, model-name and output-dir are required")
    result = run_experiment(data, feature_columns, outer_splits, args.rule_id, args.model_name, seed=args.seed, max_folds=args.max_folds)
    input_paths = [args.features_csv, args.labels_csv, args.clinical_csv, args.geometry_csv, args.feature_columns_csv, args.outer_splits_csv]
    setup = {
        "rule_id": args.rule_id,
        "rule": RULES[args.rule_id],
        "model_name": args.model_name,
        "model_family": "independent_binary_relevance",
        "modelled_labels": TARGETS,
        "derived_label": "Sin_complicación = (0, 0)",
        "radiomics_reference": "R11 basic vessels",
        "n_features_before_fold_local_selection": len(feature_columns),
        "pipeline": ["median_imputation", "variance_threshold_0.01", "SelectKBest_f_classif_k20", "StandardScaler", args.model_name],
        "outer_folds": "reused exactly from R11",
        "inner_threshold_selection": "up to 4-fold stratified OOF inside each outer training set; per-label max F1",
        "threshold_candidates": THRESHOLD_CANDIDATES,
        "seed": args.seed,
        "n_cohort": len(data),
        "n_subgroup": result["n_subgroup"],
        "input_sha256": {str(path): file_sha256(path) for path in input_paths},
        "python": platform.python_version(),
        "sklearn": __import__("sklearn").__version__,
        "exploratory_posthoc": True,
    }
    if args.max_folds is None:
        completion = save_result(result, args.output_dir, setup)
        print(json.dumps(completion, ensure_ascii=False))
    else:
        write_partial_result(result, args.output_dir, setup)
        print(json.dumps({"status": "smoke_ok", "folds": result["folds_completed"]}))


if __name__ == "__main__":
    main()
