import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

TARGET_COLS_MAIN = ["Hemorragia", "Neumotórax"]
DEFAULT_BATCH_ROOT = Path(
    "/home/mcribilles/tfm/codigo_nuevo/radiomica_preprocessed_batches/"
    "radiomics_preprocessed_allmasks__2026-06-08__19-45-43"
)
DEFAULT_RESULTS_ROOT = DEFAULT_BATCH_ROOT / "results"
DEFAULT_LABELS_CSV = Path("/home/mcribilles/tfm/clinical_data/210pacientes/clinical_data_limpios.csv")
DEFAULT_BASE_PREPROC_NAMES = [
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
DEFAULT_USE_MASKS_VARIANTS = [
    "nodule",
    "lung+nodule",
    "nodule+vessels",
    "available_merge",
]
DEFAULT_RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a bounded radiomics classification grid over preprocessed variants."
    )
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--batch_root", type=str, default=str(DEFAULT_BATCH_ROOT))
    parser.add_argument("--results_root", type=str, default=None)
    parser.add_argument("--labels_csv", type=str, default=str(DEFAULT_LABELS_CSV))
    parser.add_argument(
        "--preproc_names",
        nargs="*",
        default=None,
        help=(
            "Explicit preprocessed variants to include. If omitted, the builder uses the base list "
            "and auto-adds extracted multiwindowing_separadas variants when present under results_root."
        ),
    )
    parser.add_argument(
        "--feature_modes",
        nargs="*",
        default=list(DEFAULT_FEATURE_MODES),
        choices=["basic", "extended"],
    )
    parser.add_argument(
        "--use_masks_variants",
        nargs="*",
        default=list(DEFAULT_USE_MASKS_VARIANTS),
        help=(
            "Mask variants to include in the grid. available_merge is the closest supported mode "
            "to a no-single-mask setup when mask_kind is present in the features CSV."
        ),
    )
    parser.add_argument(
        "--include_multiwindowing_if_present",
        dest="include_multiwindowing_if_present",
        action="store_true",
        default=True,
        help="Auto-add extracted *_multiwindowing_separadas variants found under results_root.",
    )
    parser.add_argument(
        "--no_include_multiwindowing_if_present",
        dest="include_multiwindowing_if_present",
        action="store_false",
        help="Disable auto-add for extracted *_multiwindowing_separadas variants.",
    )
    parser.add_argument("--count_only", action="store_true")
    return parser.parse_args()


def dedupe_preserve_order(values):
    out = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def sanitize_name_token(value: str) -> str:
    return str(value).replace("+", "__").replace(":", "__")


def add_template(
    templates,
    name,
    classifier,
    multilabel_strategy,
    scaler,
    dimred,
    split_strategy,
    params,
    family,
):
    templates.append(
        {
            "name": name,
            "classifier": classifier,
            "multilabel_strategy": multilabel_strategy,
            "scaler": scaler,
            "dimred": dimred,
            "split_strategy": split_strategy,
            "params": dict(params),
            "family": family,
        }
    )


def build_templates():
    templates = []
    split_strategy = "auto"
    seed = DEFAULT_RANDOM_STATE

    add_template(
        templates,
        name="logreg__ovr__std__pca_90__C0p5__bal",
        classifier="LogisticRegression",
        multilabel_strategy="one_vs_rest",
        scaler="StandardScaler",
        dimred="pca_90",
        split_strategy=split_strategy,
        params={
            "C": 0.5,
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 3000,
        },
        family="LogisticRegression",
    )
    add_template(
        templates,
        name="logreg__ovr__std__pca_95__C1__bal",
        classifier="LogisticRegression",
        multilabel_strategy="one_vs_rest",
        scaler="StandardScaler",
        dimred="pca_95",
        split_strategy=split_strategy,
        params={
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 3000,
        },
        family="LogisticRegression",
    )

    add_template(
        templates,
        name="svm_lin__ovr__std__varth_0p01__C0p5__bal",
        classifier="SVM",
        multilabel_strategy="one_vs_rest",
        scaler="StandardScaler",
        dimred="varth_0.01",
        split_strategy=split_strategy,
        params={
            "kernel": "linear",
            "C": 0.5,
            "class_weight": "balanced",
            "max_iter": 5000,
        },
        family="SVM_linear",
    )
    add_template(
        templates,
        name="svm_lin__ovr__std__pca_99__C1__bal",
        classifier="SVM",
        multilabel_strategy="one_vs_rest",
        scaler="StandardScaler",
        dimred="pca_99",
        split_strategy=split_strategy,
        params={
            "kernel": "linear",
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 5000,
        },
        family="SVM_linear",
    )

    add_template(
        templates,
        name="knn__multi__minmax__pca_95__k7__wdistance",
        classifier="KNN",
        multilabel_strategy="multi_output",
        scaler="MinMaxScaler",
        dimred="pca_95",
        split_strategy=split_strategy,
        params={
            "n_neighbors": 7,
            "weights": "distance",
        },
        family="KNN",
    )

    add_template(
        templates,
        name="randomforest__ovr__none__none__n200__mfsqrt__bal",
        classifier="RandomForest",
        multilabel_strategy="one_vs_rest",
        scaler="none",
        dimred="none",
        split_strategy=split_strategy,
        params={
            "n_estimators": 200,
            "max_features": "sqrt",
            "max_depth": None,
            "class_weight": "balanced",
            "n_jobs": 1,
            "random_state": seed,
        },
        family="RandomForest",
    )

    add_template(
        templates,
        name="decisiontree__ovr__none__none__d5__bal",
        classifier="DecisionTree",
        multilabel_strategy="one_vs_rest",
        scaler="none",
        dimred="none",
        split_strategy=split_strategy,
        params={
            "max_depth": 5,
            "class_weight": "balanced",
            "random_state": seed,
        },
        family="DecisionTree",
    )

    add_template(
        templates,
        name="lightgbm__ovr__none__varth_0p01__n120__d3__lr0p1__l15",
        classifier="LightGBM",
        multilabel_strategy="one_vs_rest",
        scaler="none",
        dimred="varth_0.01",
        split_strategy=split_strategy,
        params={
            "n_estimators": 120,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "num_leaves": 15,
            "class_weight": "balanced",
            "verbose": -1,
            "n_jobs": 1,
            "random_state": seed,
        },
        family="LightGBM",
    )

    add_template(
        templates,
        name="xgboost__ovr__none__varth_0p01__n120__d3__lr0p1",
        classifier="XGBoost",
        multilabel_strategy="one_vs_rest",
        scaler="none",
        dimred="varth_0.01",
        split_strategy=split_strategy,
        params={
            "n_estimators": 120,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": 1,
            "random_state": seed,
        },
        family="XGBoost",
    )

    add_template(
        templates,
        name="gradboost__ovr__none__none__n100__d2__lr0p05",
        classifier="GradientBoosting",
        multilabel_strategy="one_vs_rest",
        scaler="none",
        dimred="none",
        split_strategy=split_strategy,
        params={
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 2,
            "subsample": 0.8,
            "random_state": seed,
        },
        family="GradientBoosting",
    )

    names = [template["name"] for template in templates]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicated template names detected: {duplicates[:10]}")
    return templates


def discover_available_preproc_modes(results_root: Path, feature_modes):
    availability = {}
    if not results_root.exists():
        return availability

    for variant_dir in sorted(results_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        available_modes = []
        for feature_mode in feature_modes:
            features_csv = (
                variant_dir / f"radiomics_preprocessed_{variant_dir.name}_{feature_mode}_all_raw.csv"
            )
            if features_csv.exists():
                available_modes.append(feature_mode)
        if available_modes:
            availability[variant_dir.name] = available_modes
    return availability


def resolve_preproc_names(results_root: Path, requested_preproc_names, feature_modes, include_multiwindowing_if_present):
    discovered_modes = discover_available_preproc_modes(results_root=results_root, feature_modes=feature_modes)

    if requested_preproc_names is None:
        preproc_names = list(DEFAULT_BASE_PREPROC_NAMES)
    else:
        preproc_names = list(requested_preproc_names)

    auto_added_multiwindowing = []
    if include_multiwindowing_if_present:
        auto_added_multiwindowing = [
            name
            for name in sorted(discovered_modes)
            if "multiwindowing_separadas" in name
        ]
        preproc_names.extend(auto_added_multiwindowing)

    preproc_names = dedupe_preserve_order(preproc_names)
    return preproc_names, discovered_modes, auto_added_multiwindowing


def build_feature_manifest(results_root: Path, preproc_names, feature_modes, discovered_modes):
    manifest = []
    for preproc_name in preproc_names:
        variant_dir = results_root / preproc_name
        available_modes = set(discovered_modes.get(preproc_name, []))
        for feature_mode in feature_modes:
            features_csv = (
                variant_dir / f"radiomics_preprocessed_{preproc_name}_{feature_mode}_all_raw.csv"
            )
            manifest.append(
                {
                    "preproc_name": preproc_name,
                    "feature_mode": feature_mode,
                    "features_csv": str(features_csv),
                    "features_csv_exists": feature_mode in available_modes,
                }
            )
    return manifest


def build_configs(feature_manifest, templates, labels_csv: str, use_masks_variants):
    configs = []
    for entry in feature_manifest:
        if not entry.get("features_csv_exists", False):
            continue
        for use_masks in use_masks_variants:
            mask_tag = sanitize_name_token(use_masks)
            for template in templates:
                config = {
                    "config_name": (
                        f"preproc__{entry['preproc_name']}__{entry['feature_mode']}"
                        f"__mask_{mask_tag}"
                        f"__main_hn_derive_no_comp__{template['name']}"
                    ),
                    "features_csv": entry["features_csv"],
                    "labels_csv": labels_csv,
                    "id_col": "patient_id",
                    "target_cols": list(TARGET_COLS_MAIN),
                    "label_mode": "derive_no_complication",
                    "use_masks": use_masks,
                    "n_splits": 5,
                    "random_state": DEFAULT_RANDOM_STATE,
                    "shuffle_folds": True,
                    "split_strategy": template["split_strategy"],
                    "scaler": template["scaler"],
                    "classifier": template["classifier"],
                    "dimred": template["dimred"],
                    "multilabel_strategy": template["multilabel_strategy"],
                    "save_models": False,
                    "save_predictions": False,
                }
                config.update(template["params"])
                configs.append(config)
    names = [cfg["config_name"] for cfg in configs]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicated config names detected: {duplicates[:10]}")
    return configs


def summarize(feature_manifest, templates, configs, args, results_root, discovered_modes, auto_added_multiwindowing):
    by_model = defaultdict(int)
    by_dimred = defaultdict(int)
    by_scaler = defaultdict(int)
    by_mode = defaultdict(int)
    by_preproc = defaultdict(int)
    by_masks = defaultdict(int)
    existing_feature_variants = 0

    for cfg in configs:
        by_model[cfg["classifier"]] += 1
        by_dimred[cfg["dimred"]] += 1
        by_scaler[cfg["scaler"]] += 1
        by_masks[cfg["use_masks"]] += 1

    for entry in feature_manifest:
        by_mode[entry["feature_mode"]] += 1
        by_preproc[entry["preproc_name"]] += 1
        if entry["features_csv_exists"]:
            existing_feature_variants += 1

    return {
        "design_name": "preprocessed_radiomics_bounded_grid_v2",
        "goal": (
            "Compare extracted preprocessed radiomics variants, auto-adding available "
            "multiwindowing_separadas families, with a bounded multilabel classification grid."
        ),
        "batch_root": str(Path(args.batch_root)),
        "results_root": str(results_root),
        "results_root_exists": results_root.exists(),
        "labels_csv": args.labels_csv,
        "target_cols": list(TARGET_COLS_MAIN),
        "label_mode": "derive_no_complication",
        "use_masks_variants": list(args.use_masks_variants),
        "mask_variant_note": (
            "available_merge is used as the closest supported proxy to a no-single-mask setup "
            "when mask_kind is present in the features CSV."
        ),
        "include_multiwindowing_if_present": bool(args.include_multiwindowing_if_present),
        "auto_added_multiwindowing_preproc_names": list(auto_added_multiwindowing),
        "included_preproc_names": list(dedupe_preserve_order(args.preproc_names or DEFAULT_BASE_PREPROC_NAMES + auto_added_multiwindowing)),
        "feature_modes": list(args.feature_modes),
        "feature_variant_count": len(feature_manifest),
        "feature_variant_count_existing_on_disk": existing_feature_variants,
        "template_count": len(templates),
        "config_count": len(configs),
        "config_count_by_model": dict(sorted(by_model.items())),
        "config_count_by_dimred": dict(sorted(by_dimred.items())),
        "config_count_by_scaler": dict(sorted(by_scaler.items())),
        "config_count_by_use_masks": dict(sorted(by_masks.items())),
        "feature_variant_count_by_mode": dict(sorted(by_mode.items())),
        "feature_variant_count_by_preproc": dict(sorted(by_preproc.items())),
        "discovered_preproc_modes": discovered_modes,
        "templates": templates,
        "feature_manifest": feature_manifest,
    }


def write_outputs(out_root: Path, summary, configs):
    out_root.mkdir(parents=True, exist_ok=True)
    configs_dir = out_root / "configs"
    summaries_dir = out_root / "summaries"
    configs_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    config_path = configs_dir / "valid_configurations_preprocessed_bounded_main.json"
    summary_path = summaries_dir / "preprocessed_bounded_grid_summary.json"

    config_path.write_text(json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path, summary_path


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    results_root = Path(args.results_root) if args.results_root else Path(args.batch_root) / "results"

    preproc_names, discovered_modes, auto_added_multiwindowing = resolve_preproc_names(
        results_root=results_root,
        requested_preproc_names=args.preproc_names,
        feature_modes=args.feature_modes,
        include_multiwindowing_if_present=args.include_multiwindowing_if_present,
    )
    args.preproc_names = preproc_names

    templates = build_templates()
    feature_manifest = build_feature_manifest(
        results_root=results_root,
        preproc_names=preproc_names,
        feature_modes=args.feature_modes,
        discovered_modes=discovered_modes,
    )
    configs = build_configs(
        feature_manifest=feature_manifest,
        templates=templates,
        labels_csv=args.labels_csv,
        use_masks_variants=args.use_masks_variants,
    )
    summary = summarize(
        feature_manifest=feature_manifest,
        templates=templates,
        configs=configs,
        args=args,
        results_root=results_root,
        discovered_modes=discovered_modes,
        auto_added_multiwindowing=auto_added_multiwindowing,
    )

    if args.count_only:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    config_path, summary_path = write_outputs(out_root=out_root, summary=summary, configs=configs)
    print(f"config_path: {config_path}")
    print(f"summary_path: {summary_path}")
    print(f"config_count: {len(configs)}")


if __name__ == "__main__":
    main()
