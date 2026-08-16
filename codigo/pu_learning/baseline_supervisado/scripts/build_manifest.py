#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import itertools
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

BASE = Path.home() / "tfm" / "pu_learning" / "baseline_supervisado"
RAD_GRID = Path.home() / "tfm" / "radiomica_multietiqueta" / "radiomica_grids" / "multimask_clinical_grid_imbalance_small__2026-06-06__17-58-35"
LABELS_CSV = RAD_GRID / "labels" / "labels_main_drop_dpleural_only.csv"
FEATURES_DIR = RAD_GRID / "features"
TARGETS = ["Hemorragia", "Neumotórax"]
FORMULATIONS = ["supervised_all_negatives", "supervised_clean_negatives", "pu_learning"]
FEATURE_SETS = {
    "clinical_geometry": FEATURES_DIR / "clinical_only__clin_sd_geom.csv",
    "radiomics": FEATURES_DIR / "radiomics__basic__lung__nodule__vessels.csv",
    "clinical_geometry_radiomics": FEATURES_DIR / "radiomics__basic__lung__nodule__vessels__clin_sd_geom.csv",
}


def parse_args():
    p = argparse.ArgumentParser(description="Build expanded manifest for binary supervised and PU baselines.")
    p.add_argument("--out_dir", default=str(BASE / "manifests" / "expanded_grid_20260811_1205"))
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include_lightgbm", action="store_true")
    return p.parse_args()


def feature_ids(path):
    return set(pd.read_csv(path, usecols=["patient_id"])["patient_id"].astype(str))


def build_model_grid(include_lightgbm):
    rows = []
    for C, cw in itertools.product([0.01, 0.1, 1.0, 10.0, 100.0], ["balanced", None]):
        rows.append(("logistic_regression", {"C": C, "penalty": "l2", "class_weight": cw}))
    for C, cw in itertools.product([0.01, 0.1, 1.0, 10.0, 100.0], ["balanced", None]):
        rows.append(("linear_svm", {"C": C, "class_weight": cw}))
    for C, gamma in itertools.product([0.1, 1.0, 10.0, 100.0], ["scale", "auto"]):
        rows.append(("rbf_svm", {"C": C, "gamma": gamma, "class_weight": "balanced"}))
    for n, depth, leaf, cw in itertools.product([300, 600, 1000], [None, 3, 5, 10], [1, 2, 5], ["balanced", "balanced_subsample"]):
        rows.append(("random_forest", {"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf, "class_weight": cw}))
    for n, depth, leaf, cw in itertools.product([300, 600, 1000], [None, 3, 5, 10], [1, 2, 5], ["balanced", "balanced_subsample"]):
        rows.append(("extra_trees", {"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf, "class_weight": cw}))
    if include_lightgbm:
        for n, lr, depth, subsample, colsample in itertools.product([100, 300], [0.03, 0.1], [2, 3], [0.8, 1.0], [0.8, 1.0]):
            rows.append(("lightgbm", {"n_estimators": n, "learning_rate": lr, "max_depth": depth, "subsample": subsample, "colsample_bytree": colsample}))
    return rows


def make_folds(labels, common_ids, targets, n_splits, seed):
    df = labels[labels["patient_id"].astype(str).isin(common_ids)].copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df = df.sort_values("patient_id").reset_index(drop=True)
    strata = df[targets].astype(int).astype(str).agg("".join, axis=1)
    counts = strata.value_counts()
    if counts.min() < n_splits:
        raise ValueError(f"Minimum labelset stratum count {counts.min()} is lower than n_splits={n_splits}")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(df[["patient_id"]], strata), start=1):
        for idx in tr:
            rows.append({"fold": fold, "split": "train", "patient_id": df.loc[idx, "patient_id"]})
        for idx in te:
            rows.append({"fold": fold, "split": "test", "patient_id": df.loc[idx, "patient_id"]})
    return pd.DataFrame(rows), df, counts.to_dict()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS_CSV)
    common_ids = set(labels["patient_id"].astype(str))
    feature_info = []
    for name, path in FEATURE_SETS.items():
        ids = feature_ids(path)
        common_ids &= ids
        feature_info.append({"feature_set": name, "features_csv": str(path), "n_patients_raw": len(ids)})
    folds, cohort_df, strata_counts = make_folds(labels, common_ids, TARGETS + ["Sin_complicación"], args.n_splits, args.seed)
    folds_path = out_dir / "folds_labelset_stratified_s42.csv"
    folds.to_csv(folds_path, index=False)
    cohort_path = out_dir / "shared_cohort_patient_ids.csv"
    cohort_df[["patient_id"]].to_csv(cohort_path, index=False)
    model_grid = build_model_grid(args.include_lightgbm)
    results_root = BASE / "results" / "expanded_grid_20260811_1205_${SLURM_ARRAY_JOB_ID}"
    manifest_path = out_dir / "expanded_grid_manifest.csv"
    task_id = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "formulation", "feature_set", "target", "model", "params_json", "features_csv", "labels_csv", "folds_csv", "cohort_ids_csv", "output_root", "seed"])
        writer.writeheader()
        for formulation, target, (feature_set, features_csv), (model, params) in itertools.product(FORMULATIONS, TARGETS, FEATURE_SETS.items(), model_grid):
            writer.writerow({
                "task_id": task_id,
                "formulation": formulation,
                "feature_set": feature_set,
                "target": target,
                "model": model,
                "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
                "features_csv": str(features_csv),
                "labels_csv": str(LABELS_CSV),
                "folds_csv": str(folds_path),
                "cohort_ids_csv": str(cohort_path),
                "output_root": str(results_root),
                "seed": args.seed,
            })
            task_id += 1
    summary = {
        "created_at": datetime.now().isoformat(),
        "manifest_csv": str(manifest_path),
        "folds_csv": str(folds_path),
        "cohort_ids_csv": str(cohort_path),
        "labels_csv": str(LABELS_CSV),
        "targets": TARGETS,
        "formulations": FORMULATIONS,
        "feature_sets": feature_info,
        "n_common_patients": len(cohort_df),
        "n_model_configs": len(model_grid),
        "n_tasks": task_id,
        "include_lightgbm": bool(args.include_lightgbm),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "labelset_counts_for_folds": strata_counts,
    }
    with (out_dir / "manifest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
