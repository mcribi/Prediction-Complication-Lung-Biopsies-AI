#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_BASE = Path("/mnt/homeGPU/mcribilles/tfm/codigo")
DEFAULT_RADIOMICS_GRID = DEFAULT_BASE / "radiomica_multietiqueta" / "radiomica_grids" / "multimask_clinical_grid_imbalance_small__2026-06-06__17-58-35"
DEFAULT_LABELS = DEFAULT_RADIOMICS_GRID / "labels" / "labels_main_drop_dpleural_only.csv"
DEFAULT_FEATURES_DIR = DEFAULT_RADIOMICS_GRID / "features"
TARGET_COLS = ["Hemorragia", "Neumotórax", "Sin_complicación"]
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Build extended supervised baseline configs for PU-learning comparison.")
    parser.add_argument("--out_root", type=str, default=str(Path("/mnt/homeGPU/mcribilles/tfm/codigo/pu_learning/baseline_supervisado/baseline_grid_extended")))
    parser.add_argument("--labels_csv", type=str, default=str(DEFAULT_LABELS))
    parser.add_argument("--clinical_geometry_csv", type=str, default=str(DEFAULT_FEATURES_DIR / "clinical_only__clin_sd_geom.csv"))
    parser.add_argument("--radiomics_csv", type=str, default=str(DEFAULT_FEATURES_DIR / "radiomics__basic__lung__nodule__vessels.csv"))
    parser.add_argument("--clinical_geometry_radiomics_csv", type=str, default=str(DEFAULT_FEATURES_DIR / "radiomics__basic__lung__nodule__vessels__clin_sd_geom.csv"))
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=SEED)
    return parser.parse_args()


def read_ids(path: Path):
    df = pd.read_csv(path, usecols=["patient_id"])
    return set(df["patient_id"].astype(str))


def count_features(path: Path, target_cols):
    df = pd.read_csv(path, nrows=5)
    drop = {"patient_id", "mask_kind", "image_path", "image_size", "image_spacing"} | set(target_cols)
    return len([c for c in df.columns if c not in drop and not c.endswith("_image_path") and not c.endswith("_image_size") and not c.endswith("_image_spacing")])


def add_template(templates, name, classifier, multilabel_strategy, scaler, dimred, params):
    templates.append({
        "template_name": name,
        "classifier": classifier,
        "multilabel_strategy": multilabel_strategy,
        "scaler": scaler,
        "dimred": dimred,
        "params": dict(params),
    })


def build_templates(random_state):
    templates = []

    for c in [0.1, 1.0, 10.0]:
        tag = str(c).replace(".", "p")
        add_template(
            templates,
            f"logreg_ovr_std_varth_c{tag}_bal",
            "LogisticRegression",
            "one_vs_rest",
            "StandardScaler",
            "varth_0.01",
            {"C": c, "class_weight": "balanced", "solver": "liblinear", "max_iter": 3000, "random_state": random_state},
        )

    add_template(
        templates,
        "logreg_ovr_pca95_c1_bal",
        "LogisticRegression",
        "one_vs_rest",
        "none",
        "pca_95",
        {"C": 1.0, "class_weight": "balanced", "solver": "liblinear", "max_iter": 3000, "random_state": random_state},
    )

    for c in [0.1, 1.0, 10.0]:
        tag = str(c).replace(".", "p")
        add_template(
            templates,
            f"svm_linear_ovr_std_varth_c{tag}_bal",
            "LinearSVM",
            "one_vs_rest",
            "StandardScaler",
            "varth_0.01",
            {"C": c, "class_weight": "balanced", "max_iter": 5000, "random_state": random_state},
        )

    for max_depth in [None, 4, 8]:
        dtag = "none" if max_depth is None else str(max_depth)
        add_template(
            templates,
            f"extratrees_ovr_none_none_d{dtag}_sqrt_balsub",
            "ExtraTrees",
            "one_vs_rest",
            "none",
            "none",
            {"n_estimators": 300, "max_features": "sqrt", "max_depth": max_depth, "class_weight": "balanced_subsample", "n_jobs": 1, "random_state": random_state},
        )
    add_template(
        templates,
        "extratrees_ovr_none_none_d8_log2_balsub",
        "ExtraTrees",
        "one_vs_rest",
        "none",
        "none",
        {"n_estimators": 300, "max_features": "log2", "max_depth": 8, "class_weight": "balanced_subsample", "n_jobs": 1, "random_state": random_state},
    )

    for max_depth in [None, 4, 8]:
        dtag = "none" if max_depth is None else str(max_depth)
        add_template(
            templates,
            f"rf_ovr_none_none_d{dtag}_sqrt_bal",
            "RandomForest",
            "one_vs_rest",
            "none",
            "none",
            {"n_estimators": 300, "max_features": "sqrt", "max_depth": max_depth, "class_weight": "balanced", "n_jobs": 1, "random_state": random_state},
        )
    add_template(
        templates,
        "rf_ovr_none_none_d8_log2_bal",
        "RandomForest",
        "one_vs_rest",
        "none",
        "none",
        {"n_estimators": 300, "max_features": "log2", "max_depth": 8, "class_weight": "balanced", "n_jobs": 1, "random_state": random_state},
    )

    for n_estimators, learning_rate, max_depth in [(100, 0.05, 2), (100, 0.10, 2), (200, 0.05, 2), (100, 0.05, 3)]:
        lr_tag = str(learning_rate).replace(".", "p")
        add_template(
            templates,
            f"gbt_chain_none_varth_n{n_estimators}_lr{lr_tag}_d{max_depth}",
            "GradientBoosting",
            "classifier_chain",
            "none",
            "varth_0.01",
            {"n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth, "random_state": random_state},
        )

    for hidden_layer_sizes, alpha in [([32], 0.001), ([64], 0.001), ([32], 0.01), ([64], 0.01)]:
        htag = "x".join(str(x) for x in hidden_layer_sizes)
        atag = str(alpha).replace(".", "p")
        add_template(
            templates,
            f"mlp_ovr_std_varth_h{htag}_a{atag}",
            "MLP",
            "one_vs_rest",
            "StandardScaler",
            "varth_0.01",
            {"hidden_layer_sizes": hidden_layer_sizes, "alpha": alpha, "max_iter": 500, "early_stopping": True, "random_state": random_state},
        )

    return templates


def main():
    args = parse_args()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    labels_csv = Path(args.labels_csv).expanduser().resolve()
    feature_sets = [
        ("clinical_geometry", Path(args.clinical_geometry_csv).expanduser().resolve()),
        ("radiomics", Path(args.radiomics_csv).expanduser().resolve()),
        ("clinical_geometry_radiomics", Path(args.clinical_geometry_radiomics_csv).expanduser().resolve()),
    ]

    missing = [str(p) for _, p in feature_sets if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing feature CSVs: " + "; ".join(missing))
    if not labels_csv.exists():
        raise FileNotFoundError(f"Missing labels CSV: {labels_csv}")

    labels = pd.read_csv(labels_csv)
    missing_targets = [c for c in TARGET_COLS if c not in labels.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in labels CSV: {missing_targets}")

    common_ids = set(labels["patient_id"].astype(str))
    feature_info = []
    for feature_set, path in feature_sets:
        ids = read_ids(path)
        common_ids &= ids
        feature_info.append({
            "feature_set": feature_set,
            "features_csv": str(path),
            "n_patients_raw": len(ids),
            "n_features_preview": count_features(path, TARGET_COLS),
        })

    common_ids_sorted = sorted(common_ids)
    if len(common_ids_sorted) < 50:
        raise ValueError(f"Unexpectedly small common cohort: {len(common_ids_sorted)} patients")

    cohort_path = out_root / "shared_cohort_patient_ids.csv"
    pd.DataFrame({"patient_id": common_ids_sorted}).to_csv(cohort_path, index=False)

    templates = build_templates(args.random_state)
    configs = []
    for feature_set, path in feature_sets:
        for template in templates:
            cfg = {
                "config_name": f"{feature_set}__{template['template_name']}__labelset_stratified__s{args.random_state}",
                "feature_set": feature_set,
                "features_csv": str(path),
                "labels_csv": str(labels_csv),
                "cohort_ids_csv": str(cohort_path),
                "id_col": "patient_id",
                "target_cols": TARGET_COLS,
                "n_splits": args.n_splits,
                "random_state": args.random_state,
                "shuffle_folds": True,
                "split_strategy": "labelset_stratified",
                "classifier": template["classifier"],
                "multilabel_strategy": template["multilabel_strategy"],
                "scaler": template["scaler"],
                "dimred": template["dimred"],
                **template["params"],
            }
            configs.append(cfg)

    configs_path = out_root / "valid_configurations_supervised_baseline_extended.json"
    with configs_path.open("w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)

    summary = {
        "created_at": datetime.now().isoformat(),
        "out_root": str(out_root),
        "labels_csv": str(labels_csv),
        "target_cols": TARGET_COLS,
        "n_common_patients": len(common_ids_sorted),
        "n_configs": len(configs),
        "n_templates": len(templates),
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "split_strategy": "labelset_stratified",
        "feature_sets": feature_info,
        "configs_file": str(configs_path),
        "cohort_ids_file": str(cohort_path),
        "design_note": "Extended but bounded grid: small linear C sweep, bounded tree ensembles, shallow gradient boosting, and small MLPs. No very deep trees with many estimators or large neural networks.",
    }
    summary_path = out_root / "grid_summary_extended.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
