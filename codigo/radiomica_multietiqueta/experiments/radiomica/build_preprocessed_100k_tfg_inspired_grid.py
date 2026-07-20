import argparse
import csv
import json
from collections import Counter
from itertools import product
from pathlib import Path

TARGET_COLS_MAIN = ["Hemorragia", "Neumotórax"]
DEFAULT_RESULTS_ROOT = Path(
    "/home/mcribilles/tfm/codigo_nuevo/radiomica_preprocessed_batches/"
    "radiomics_preprocessed_allmasks__2026-06-08__19-45-43/results"
)
DEFAULT_LABELS_CSV = Path("/home/mcribilles/tfm/clinical_data/210pacientes/clinical_data_limpios.csv")
DEFAULT_PREPROC_NAMES = [
    "resize_cube128",
    "resize_cube128_hu_m300_1400",
    "resize_cube128_hu_m600_1500",
    "resize_cube64",
    "resize_cube64_hu_m300_1400",
    "resize_cube64_hu_m600_1500",
    "resize_medium",
    "resize_medium_hu_m300_1400",
    "resize_medium_hu_m600_1500",
]
DEFAULT_FEATURE_MODES = ["basic", "extended"]
DEFAULT_MASK_VARIANTS = [
    "lung",
    "available_merge",
    "nodule",
    "vessels",
    "trachea_bronchia",
    "lung+nodule",
    "nodule+vessels",
    "nodule+trachea_bronchia",
    "lung+nodule+vessels",
    "lung+nodule+trachea_bronchia",
]
EXPECTED_FEATURE_VARIANT_COUNT = 12
FULL_TEMPLATE_TARGET = 833
BONUS_CONFIG_TARGET = 40
FINAL_CONFIG_TARGET = 100000
TFG_VALID_CONFIG_COUNT = 48960
TFG_SOURCE_FILES = {
    "search_space_json": "/root/.hermes/cache/documents/doc_428303e802ba_search_space.json",
    "valid_configurations_json": "/root/.hermes/cache/documents/doc_6976be91e8a8_valid_configurations.json",
}
TFG_SEARCH_SPACE = {
    "Scaler": ["StandardScaler", "MinMaxScaler", None],
    "UseMasks": ["lung", "both_merge"],
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
    },
    "SVM": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    },
    "LogisticRegression": {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs", "saga"],
    },
    "GradientBoosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [None, 5, 10],
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [None, 5, 10],
    },
    "LightGBM": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1],
        "num_leaves": [31, 50],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
    },
    "DecisionTree": {
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a 100k preprocessed multilabel grid inspired by the historical TFG search space."
    )
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--results_root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--labels_csv", type=str, default=str(DEFAULT_LABELS_CSV))
    parser.add_argument("--count_only", action="store_true")
    return parser.parse_args()


def sanitize_name_token(value):
    if value is None:
        return "none"
    return str(value).replace("+", "__").replace(":", "__").replace(".", "p")


def serialize_value(value):
    if value is None:
        return "none"
    if isinstance(value, float):
        text = f"{value}".rstrip("0").rstrip(".") if "." in f"{value}" else f"{value}"
        return text.replace("-", "m").replace(".", "p")
    return str(value).replace("-", "m").replace(".", "p")


def csv_has_header_like_content(path: Path) -> bool:
    try:
        if not path.exists() or not path.is_file():
            return False
        if path.stat().st_size <= 64:
            return False
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if not header:
            return False
        header_set = {str(col).strip() for col in header}
        return "patient_id" in header_set and "mask_kind" in header_set
    except Exception:
        return False


def build_feature_manifest(results_root: Path):
    manifest = []
    for preproc_name in DEFAULT_PREPROC_NAMES:
        for feature_mode in DEFAULT_FEATURE_MODES:
            path = results_root / preproc_name / f"radiomics_preprocessed_{preproc_name}_{feature_mode}_all_per_mask.csv"
            readable = csv_has_header_like_content(path)
            manifest.append(
                {
                    "preproc_name": preproc_name,
                    "feature_mode": feature_mode,
                    "features_csv": str(path),
                    "features_csv_exists": path.exists(),
                    "features_csv_readable": readable,
                }
            )
    return manifest


def build_param_grid(space_dict):
    keys = list(space_dict.keys())
    values = [space_dict[k] for k in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def normalize_params_for_runner(classifier_name, params):
    out = {}
    for key, value in params.items():
        if classifier_name == "XGBoost" and key == "max_depth" and value is None:
            continue
        out[key] = value
    return out


def template_name(prefix, scaler, multilabel_strategy, dimred, split_strategy, random_state, params):
    parts = [
        prefix,
        f"scaler_{sanitize_name_token(scaler)}",
        f"ml_{sanitize_name_token(multilabel_strategy)}",
        f"dimred_{sanitize_name_token(dimred)}",
        f"split_{sanitize_name_token(split_strategy)}",
        f"seed_{serialize_value(random_state)}",
    ]
    for key in sorted(params.keys()):
        parts.append(f"{key}_{serialize_value(params[key])}")
    return "__".join(parts)


def build_historical_templates():
    templates = []
    strategies = ["one_vs_rest", "classifier_chain"]
    family_order = [
        "RandomForest",
        "SVM",
        "LogisticRegression",
        "GradientBoosting",
        "XGBoost",
        "LightGBM",
        "KNN",
        "DecisionTree",
    ]
    for family in family_order:
        for params in build_param_grid(TFG_SEARCH_SPACE[family]):
            clean_params = normalize_params_for_runner(family, params)
            for scaler in TFG_SEARCH_SPACE["Scaler"]:
                for multilabel_strategy in strategies:
                    templates.append(
                        {
                            "name": template_name(
                                prefix=f"tfg_{family.lower()}",
                                scaler=scaler,
                                multilabel_strategy=multilabel_strategy,
                                dimred="none",
                                split_strategy="auto",
                                random_state=42,
                                params=clean_params,
                            ),
                            "classifier": family,
                            "scaler": scaler,
                            "dimred": "none",
                            "multilabel_strategy": multilabel_strategy,
                            "split_strategy": "auto",
                            "random_state": 42,
                            "params": dict(clean_params),
                            "template_group": "tfg_historical",
                        }
                    )
    return templates


def build_extratrees_templates():
    templates = []
    for params in build_param_grid(TFG_SEARCH_SPACE["RandomForest"]):
        clean_params = dict(params)
        for scaler in TFG_SEARCH_SPACE["Scaler"]:
            for multilabel_strategy in ["one_vs_rest", "classifier_chain"]:
                templates.append(
                    {
                        "name": template_name(
                            prefix="extra_extratrees",
                            scaler=scaler,
                            multilabel_strategy=multilabel_strategy,
                            dimred="none",
                            split_strategy="auto",
                            random_state=42,
                            params=clean_params,
                        ),
                        "classifier": "ExtraTrees",
                        "scaler": scaler,
                        "dimred": "none",
                        "multilabel_strategy": multilabel_strategy,
                        "split_strategy": "auto",
                        "random_state": 42,
                        "params": dict(clean_params),
                        "template_group": "extra_extratrees_from_tfg_rf_pattern",
                    }
                )
    return templates


def build_multi_output_templates():
    templates = []
    family_to_space = {
        "RandomForest": TFG_SEARCH_SPACE["RandomForest"],
        "LightGBM": TFG_SEARCH_SPACE["LightGBM"],
        "KNN": TFG_SEARCH_SPACE["KNN"],
        "DecisionTree": TFG_SEARCH_SPACE["DecisionTree"],
    }
    for family, space in family_to_space.items():
        for params in build_param_grid(space):
            clean_params = normalize_params_for_runner(family, params)
            for scaler in TFG_SEARCH_SPACE["Scaler"]:
                templates.append(
                    {
                        "name": template_name(
                            prefix=f"multioutput_{family.lower()}",
                            scaler=scaler,
                            multilabel_strategy="multi_output",
                            dimred="none",
                            split_strategy="auto",
                            random_state=42,
                            params=clean_params,
                        ),
                        "classifier": family,
                        "scaler": scaler,
                        "dimred": "none",
                        "multilabel_strategy": "multi_output",
                        "split_strategy": "auto",
                        "random_state": 42,
                        "params": dict(clean_params),
                        "template_group": "multi_output_extension",
                    }
                )
    return templates


def curated_spec(name, classifier, scaler, dimred, multilabel_strategy, split_strategy, random_state, params):
    return {
        "name": name,
        "classifier": classifier,
        "scaler": scaler,
        "dimred": dimred,
        "multilabel_strategy": multilabel_strategy,
        "split_strategy": split_strategy,
        "random_state": random_state,
        "params": params,
        "template_group": "curated_full_cross",
    }


def build_curated_full_templates():
    return [
        curated_spec("curated_randomforest_chain_bal_n200_sqrt", "RandomForest", None, "none", "classifier_chain", "auto", 42, {"n_estimators": 200, "max_features": "sqrt", "class_weight": "balanced", "n_jobs": 1}),
        curated_spec("curated_randomforest_ovr_bal_n200_sqrt", "RandomForest", None, "none", "one_vs_rest", "auto", 42, {"n_estimators": 200, "max_features": "sqrt", "class_weight": "balanced", "n_jobs": 1}),
        curated_spec("curated_randomforest_chain_bal_n400_sqrt", "RandomForest", None, "none", "classifier_chain", "auto", 42, {"n_estimators": 400, "max_features": "sqrt", "class_weight": "balanced", "n_jobs": 1}),
        curated_spec("curated_randomforest_ovr_bal_n400_sqrt", "RandomForest", None, "none", "one_vs_rest", "auto", 42, {"n_estimators": 400, "max_features": "sqrt", "class_weight": "balanced", "n_jobs": 1}),
        curated_spec("curated_decisiontree_ovr_bal_d7", "DecisionTree", None, "none", "one_vs_rest", "auto", 42, {"max_depth": 7, "class_weight": "balanced"}),
        curated_spec("curated_decisiontree_ovr_bal_d9", "DecisionTree", None, "none", "one_vs_rest", "auto", 42, {"max_depth": 9, "class_weight": "balanced"}),
        curated_spec("curated_decisiontree_chain_bal_d7", "DecisionTree", None, "none", "classifier_chain", "auto", 42, {"max_depth": 7, "class_weight": "balanced"}),
        curated_spec("curated_decisiontree_ovr_bal_dnone_msl3", "DecisionTree", None, "none", "one_vs_rest", "auto", 42, {"max_depth": None, "min_samples_leaf": 3, "class_weight": "balanced"}),
        curated_spec("curated_lightgbm_chain_bal_varth001_n120_d3_lr01_l15", "LightGBM", None, "varth_0.01", "classifier_chain", "auto", 42, {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.1, "num_leaves": 15, "subsample": 0.8, "colsample_bytree": 0.8, "class_weight": "balanced", "n_jobs": 1, "verbose": -1}),
        curated_spec("curated_lightgbm_ovr_bal_varth001_n200_d3_lr005_l31", "LightGBM", None, "varth_0.01", "one_vs_rest", "auto", 42, {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8, "class_weight": "balanced", "n_jobs": 1, "verbose": -1}),
        curated_spec("curated_xgboost_chain_varth001_n120_d3_lr01", "XGBoost", None, "varth_0.01", "classifier_chain", "auto", 42, {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": 1}),
        curated_spec("curated_xgboost_ovr_varth001_n200_d3_lr005", "XGBoost", None, "varth_0.01", "one_vs_rest", "auto", 42, {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": 1}),
        curated_spec("curated_logreg_ovr_bal_pca90_c05", "LogisticRegression", "StandardScaler", "pca_90", "one_vs_rest", "auto", 42, {"C": 0.5, "class_weight": "balanced", "solver": "liblinear", "max_iter": 3000}),
        curated_spec("curated_logreg_chain_bal_pca95_c1", "LogisticRegression", "StandardScaler", "pca_95", "classifier_chain", "auto", 42, {"C": 1.0, "class_weight": "balanced", "solver": "liblinear", "max_iter": 3000}),
        curated_spec("curated_logreg_ovr_bal_varth001_c2", "LogisticRegression", "StandardScaler", "varth_0.01", "one_vs_rest", "auto", 42, {"C": 2.0, "class_weight": "balanced", "solver": "liblinear", "max_iter": 3000}),
        curated_spec("curated_svm_linear_ovr_bal_varth001_c05", "SVM", "StandardScaler", "varth_0.01", "one_vs_rest", "auto", 42, {"kernel": "linear", "C": 0.5, "class_weight": "balanced", "max_iter": 5000}),
        curated_spec("curated_svm_linear_chain_bal_pca90_c1", "SVM", "StandardScaler", "pca_90", "classifier_chain", "auto", 42, {"kernel": "linear", "C": 1.0, "class_weight": "balanced", "max_iter": 5000}),
        curated_spec("curated_svm_rbf_ovr_pca95_c1_gscale", "SVM", "StandardScaler", "pca_95", "one_vs_rest", "auto", 42, {"kernel": "rbf", "C": 1.0, "gamma": "scale", "max_iter": 5000}),
        curated_spec("curated_extratrees_ovr_bal_n400_sqrt", "ExtraTrees", None, "none", "one_vs_rest", "auto", 42, {"n_estimators": 400, "max_features": "sqrt", "class_weight": "balanced", "n_jobs": 1}),
        curated_spec("curated_extratrees_chain_bal_n400_sqrt", "ExtraTrees", None, "none", "classifier_chain", "auto", 42, {"n_estimators": 400, "max_features": "sqrt", "class_weight": "balanced", "n_jobs": 1}),
        curated_spec("curated_gradientboosting_ovr_varth001_n120_d3_lr005", "GradientBoosting", None, "varth_0.01", "one_vs_rest", "auto", 42, {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.05}),
        curated_spec("curated_gradientboosting_chain_varth001_n200_d2_lr01", "GradientBoosting", None, "varth_0.01", "classifier_chain", "auto", 42, {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.1}),
        curated_spec("curated_knn_ovr_std_pca90_k7_dist", "KNN", "StandardScaler", "pca_90", "one_vs_rest", "auto", 42, {"n_neighbors": 7, "weights": "distance"}),
    ]


def build_bonus_templates():
    return [
        {
            "name": "bonus_randomforest_chain_bal_n200_sqrt_seed1337",
            "classifier": "RandomForest",
            "scaler": None,
            "dimred": "none",
            "multilabel_strategy": "classifier_chain",
            "split_strategy": "kfold",
            "random_state": 1337,
            "params": {"n_estimators": 200, "max_features": "sqrt", "class_weight": "balanced", "n_jobs": 1},
            "template_group": "bonus_targeted",
        },
        {
            "name": "bonus_decisiontree_ovr_bal_d7_seed1337",
            "classifier": "DecisionTree",
            "scaler": None,
            "dimred": "none",
            "multilabel_strategy": "one_vs_rest",
            "split_strategy": "kfold",
            "random_state": 1337,
            "params": {"max_depth": 7, "class_weight": "balanced"},
            "template_group": "bonus_targeted",
        },
        {
            "name": "bonus_lightgbm_chain_bal_varth001_seed1337",
            "classifier": "LightGBM",
            "scaler": None,
            "dimred": "varth_0.01",
            "multilabel_strategy": "classifier_chain",
            "split_strategy": "kfold",
            "random_state": 1337,
            "params": {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.1, "num_leaves": 15, "subsample": 0.8, "colsample_bytree": 0.8, "class_weight": "balanced", "n_jobs": 1, "verbose": -1},
            "template_group": "bonus_targeted",
        },
        {
            "name": "bonus_xgboost_chain_varth001_seed2026",
            "classifier": "XGBoost",
            "scaler": None,
            "dimred": "varth_0.01",
            "multilabel_strategy": "classifier_chain",
            "split_strategy": "kfold",
            "random_state": 2026,
            "params": {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": 1},
            "template_group": "bonus_targeted",
        },
        {
            "name": "bonus_extratrees_chain_bal_n400_sqrt_seed2026",
            "classifier": "ExtraTrees",
            "scaler": None,
            "dimred": "none",
            "multilabel_strategy": "classifier_chain",
            "split_strategy": "kfold",
            "random_state": 2026,
            "params": {"n_estimators": 400, "max_features": "sqrt", "class_weight": "balanced", "n_jobs": 1},
            "template_group": "bonus_targeted",
        },
    ]


def build_full_templates():
    templates = []
    templates.extend(build_historical_templates())
    templates.extend(build_extratrees_templates())
    templates.extend(build_multi_output_templates())
    templates.extend(build_curated_full_templates())
    names = [tpl["name"] for tpl in templates]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicated template names: {duplicates[:10]}")
    if len(templates) != FULL_TEMPLATE_TARGET:
        raise ValueError(f"Expected {FULL_TEMPLATE_TARGET} full templates, got {len(templates)}")
    return templates


def make_config(entry, mask_variant, template):
    config_name = (
        f"tfg100k__preproc_{entry['preproc_name']}__{entry['feature_mode']}"
        f"__mask_{sanitize_name_token(mask_variant)}__{template['name']}"
    )
    cfg = {
        "config_name": config_name,
        "features_csv": entry["features_csv"],
        "labels_csv": str(DEFAULT_LABELS_CSV),
        "id_col": "patient_id",
        "target_cols": list(TARGET_COLS_MAIN),
        "label_mode": "derive_no_complication",
        "use_masks": mask_variant,
        "n_splits": 5,
        "random_state": template["random_state"],
        "shuffle_folds": True,
        "split_strategy": template["split_strategy"],
        "scaler": template["scaler"],
        "classifier": template["classifier"],
        "dimred": template["dimred"],
        "multilabel_strategy": template["multilabel_strategy"],
        "save_models": False,
        "save_predictions": False,
    }
    cfg.update(template["params"])
    return cfg


def build_bonus_configs(readable_entries):
    shortlist = [
        ("resize_cube128", "basic"),
        ("resize_cube128_hu_m600_1500", "basic"),
        ("resize_cube64", "basic"),
        ("resize_cube64_hu_m300_1400", "basic"),
    ]
    shortlist_masks = ["nodule", "available_merge"]
    bonus_templates = build_bonus_templates()
    entry_map = {(e["preproc_name"], e["feature_mode"]): e for e in readable_entries}
    configs = []
    for preproc_name, feature_mode in shortlist:
        entry = entry_map[(preproc_name, feature_mode)]
        for mask_variant in shortlist_masks:
            for template in bonus_templates:
                configs.append(make_config(entry=entry, mask_variant=mask_variant, template=template))
    if len(configs) != BONUS_CONFIG_TARGET:
        raise ValueError(f"Expected {BONUS_CONFIG_TARGET} bonus configs, got {len(configs)}")
    return configs


def summarize(feature_manifest, configs):
    return {
        "design_name": "preprocessed_tfg_inspired_100k_main_only",
        "goal": (
            "Build a 100k main-endpoint multilabel sweep over readable per-mask preprocessed radiomics, "
            "starting from the historical TFG search space and expanding with current promising masks and models."
        ),
        "tfg_source_files": dict(TFG_SOURCE_FILES),
        "tfg_valid_config_count": TFG_VALID_CONFIG_COUNT,
        "target_cols": list(TARGET_COLS_MAIN),
        "label_mode": "derive_no_complication",
        "mask_variants": list(DEFAULT_MASK_VARIANTS),
        "feature_manifest": feature_manifest,
        "feature_variant_count": len(feature_manifest),
        "feature_variant_count_readable": sum(int(e["features_csv_readable"]) for e in feature_manifest),
        "full_template_target": FULL_TEMPLATE_TARGET,
        "bonus_config_target": BONUS_CONFIG_TARGET,
        "final_config_target": FINAL_CONFIG_TARGET,
        "template_group_counts": dict(Counter(cfg.get("template_group", "unknown") for cfg in configs)),
        "classifier_counts": dict(Counter(cfg["classifier"] for cfg in configs)),
        "use_masks_counts": dict(Counter(cfg["use_masks"] for cfg in configs)),
        "feature_mode_counts": dict(Counter("basic" if "_basic_" in cfg["features_csv"] else "extended" for cfg in configs)),
        "config_count": len(configs),
    }


def write_outputs(out_root: Path, summary, configs, full_templates):
    out_root.mkdir(parents=True, exist_ok=True)
    configs_dir = out_root / "configs"
    summaries_dir = out_root / "summaries"
    configs_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    config_path = configs_dir / "valid_configurations_preprocessed_tfg_inspired_100k_main.json"
    summary_path = summaries_dir / "preprocessed_tfg_inspired_100k_summary.json"
    templates_path = summaries_dir / "preprocessed_tfg_inspired_full_templates.json"

    config_path.write_text(json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    templates_path.write_text(json.dumps(full_templates, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path, summary_path, templates_path


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    results_root = Path(args.results_root)

    feature_manifest = build_feature_manifest(results_root=results_root)
    readable_entries = [e for e in feature_manifest if e["features_csv_readable"]]
    if len(readable_entries) != EXPECTED_FEATURE_VARIANT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_VARIANT_COUNT} readable feature variants, got {len(readable_entries)}"
        )

    full_templates = build_full_templates()
    configs = []
    for entry in readable_entries:
        for mask_variant in DEFAULT_MASK_VARIANTS:
            for template in full_templates:
                cfg = make_config(entry=entry, mask_variant=mask_variant, template=template)
                cfg["labels_csv"] = args.labels_csv
                configs.append(cfg)

    bonus_configs = build_bonus_configs(readable_entries=readable_entries)
    for cfg in bonus_configs:
        cfg["labels_csv"] = args.labels_csv
    configs.extend(bonus_configs)

    names = [cfg["config_name"] for cfg in configs]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicated config names detected: {duplicates[:10]}")
    if len(configs) != FINAL_CONFIG_TARGET:
        raise ValueError(f"Expected {FINAL_CONFIG_TARGET} configs, got {len(configs)}")

    summary = summarize(feature_manifest=feature_manifest, configs=configs)

    if args.count_only:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    config_path, summary_path, templates_path = write_outputs(
        out_root=out_root,
        summary=summary,
        configs=configs,
        full_templates=full_templates,
    )
    print(f"config_path: {config_path}")
    print(f"summary_path: {summary_path}")
    print(f"templates_path: {templates_path}")
    print(f"config_count: {len(configs)}")


if __name__ == "__main__":
    main()
