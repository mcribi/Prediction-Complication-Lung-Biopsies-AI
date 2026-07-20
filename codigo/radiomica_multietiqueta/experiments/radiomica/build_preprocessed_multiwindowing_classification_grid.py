import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

TARGET_COLS_MAIN = ["Hemorragia", "Neumotórax"]
NO_COMPLICATION_COL = "Sin_complicación"
DEFAULT_RESULTS_ROOT = Path(
    "/home/mcribilles/tfm/codigo_nuevo/radiomica_preprocessed_batches/"
    "radiomics_preprocessed_multiwindowing_only_fixmerge__2026-07-03__14-45-58/results"
)
DEFAULT_LABELS_FULL_CSV = Path("/home/mcribilles/tfm/clinical_data/210pacientes/clinical_data_limpios.csv")
DEFAULT_PREPROC_NAMES = [
    "resize_cube128_multiwindowing_separadas",
    "resize_cube64_multiwindowing_separadas",
]
DEFAULT_FEATURE_MODES = ["basic", "extended"]
DEFAULT_USE_MASKS_VARIANTS = [
    "nodule",
    "lung+nodule",
    "nodule+vessels",
    "available_merge",
]
DEFAULT_RANDOM_STATE = 42
TRIVIAL_EMPTY_BYTES = 64


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a bounded multilabel classification grid for validated multiwindowing radiomics features."
    )
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--results_root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--labels_full_csv", type=str, default=str(DEFAULT_LABELS_FULL_CSV))
    parser.add_argument("--preproc_names", nargs="*", default=list(DEFAULT_PREPROC_NAMES))
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


def csv_is_structurally_readable(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= TRIVIAL_EMPTY_BYTES:
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if not header or "patient_id" not in header:
                return False
            next(reader)
        return True
    except StopIteration:
        return False
    except Exception:
        return False


def build_labels_main(labels_full_csv: Path):
    df = pd.read_csv(labels_full_csv, sep=None, engine="python", encoding="utf-8-sig")
    for col in ["Derrame_pleural", "Hemorragia", "Neumotórax", NO_COMPLICATION_COL]:
        if col not in df.columns:
            raise ValueError(f"labels_full_csv is missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    only_dpleural = (
        (df["Derrame_pleural"] == 1)
        & (df["Hemorragia"] == 0)
        & (df["Neumotórax"] == 0)
    )
    excluded_ids = df.loc[only_dpleural, "patient_id"].astype(str).tolist()
    return df.loc[~only_dpleural].copy(), excluded_ids


def add_template(templates, name, classifier, multilabel_strategy, scaler, dimred, params, family):
    templates.append(
        {
            "name": name,
            "classifier": classifier,
            "multilabel_strategy": multilabel_strategy,
            "scaler": scaler,
            "dimred": dimred,
            "split_strategy": "auto",
            "params": dict(params),
            "family": family,
        }
    )


def build_templates():
    templates = []
    seed = DEFAULT_RANDOM_STATE
    add_template(
        templates,
        name="logreg__ovr__std__pca_90__C0p5__bal",
        classifier="LogisticRegression",
        multilabel_strategy="one_vs_rest",
        scaler="StandardScaler",
        dimred="pca_90",
        params={"C": 0.5, "class_weight": "balanced", "solver": "liblinear", "max_iter": 3000},
        family="LogisticRegression",
    )
    add_template(
        templates,
        name="logreg__ovr__std__pca_95__C1__bal",
        classifier="LogisticRegression",
        multilabel_strategy="one_vs_rest",
        scaler="StandardScaler",
        dimred="pca_95",
        params={"C": 1.0, "class_weight": "balanced", "solver": "liblinear", "max_iter": 3000},
        family="LogisticRegression",
    )
    add_template(
        templates,
        name="svm_lin__ovr__std__varth_0p01__C0p5__bal",
        classifier="SVM",
        multilabel_strategy="one_vs_rest",
        scaler="StandardScaler",
        dimred="varth_0.01",
        params={"kernel": "linear", "C": 0.5, "class_weight": "balanced", "max_iter": 5000},
        family="SVM_linear",
    )
    add_template(
        templates,
        name="randomforest__ovr__none__none__n200__mfsqrt__bal",
        classifier="RandomForest",
        multilabel_strategy="one_vs_rest",
        scaler="none",
        dimred="none",
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
        name="extratrees__ovr__none__none__n200__mfsqrt__bal",
        classifier="ExtraTrees",
        multilabel_strategy="one_vs_rest",
        scaler="none",
        dimred="none",
        params={
            "n_estimators": 200,
            "max_features": "sqrt",
            "max_depth": None,
            "class_weight": "balanced",
            "n_jobs": 1,
            "random_state": seed,
        },
        family="ExtraTrees",
    )
    add_template(
        templates,
        name="gradboost__ovr__none__none__n100__d2__lr0p05",
        classifier="GradientBoosting",
        multilabel_strategy="one_vs_rest",
        scaler="none",
        dimred="none",
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


def build_feature_manifest(results_root: Path, preproc_names, feature_modes):
    manifest = []
    for preproc_name in preproc_names:
        variant_dir = results_root / preproc_name
        for feature_mode in feature_modes:
            features_csv = variant_dir / f"radiomics_preprocessed_{preproc_name}_{feature_mode}_all_per_mask.csv"
            readable = csv_is_structurally_readable(features_csv)
            manifest.append(
                {
                    "preproc_name": preproc_name,
                    "feature_mode": feature_mode,
                    "features_csv": str(features_csv),
                    "features_csv_exists": features_csv.exists(),
                    "features_csv_readable": readable,
                }
            )
    return manifest


def build_configs(feature_manifest, templates, labels_csv: str, use_masks_variants):
    configs = []
    for entry in feature_manifest:
        if not entry.get("features_csv_readable", False):
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


def summarize(feature_manifest, templates, configs, args, labels_main_csv: Path, excluded_ids):
    by_model = defaultdict(int)
    by_masks = defaultdict(int)
    by_mode = defaultdict(int)
    by_preproc = defaultdict(int)
    readable_feature_variants = 0

    for cfg in configs:
        by_model[cfg["classifier"]] += 1
        by_masks[cfg["use_masks"]] += 1

    for entry in feature_manifest:
        by_mode[entry["feature_mode"]] += 1
        by_preproc[entry["preproc_name"]] += 1
        if entry["features_csv_readable"]:
            readable_feature_variants += 1

    return {
        "design_name": "preprocessed_multiwindowing_bounded_main_grid",
        "goal": "Run a bounded multilabel radiomics classification sweep over the validated multiwindowing per-mask features.",
        "results_root": str(Path(args.results_root)),
        "labels_full_csv": args.labels_full_csv,
        "labels_main_csv": str(labels_main_csv),
        "excluded_only_dpleural_patient_ids_for_main": excluded_ids,
        "excluded_only_dpleural_count_for_main": len(excluded_ids),
        "target_cols": list(TARGET_COLS_MAIN),
        "label_mode": "derive_no_complication",
        "use_masks_variants": list(args.use_masks_variants),
        "included_preproc_names": list(dedupe_preserve_order(args.preproc_names)),
        "feature_modes": list(args.feature_modes),
        "feature_variant_count": len(feature_manifest),
        "feature_variant_count_readable_on_disk": readable_feature_variants,
        "template_count": len(templates),
        "config_count": len(configs),
        "config_count_by_model": dict(sorted(by_model.items())),
        "config_count_by_use_masks": dict(sorted(by_masks.items())),
        "feature_variant_count_by_mode": dict(sorted(by_mode.items())),
        "feature_variant_count_by_preproc": dict(sorted(by_preproc.items())),
        "templates": templates,
        "feature_manifest": feature_manifest,
    }


def write_outputs(out_root: Path, summary, configs, labels_main_df: pd.DataFrame):
    out_root.mkdir(parents=True, exist_ok=True)
    configs_dir = out_root / "configs"
    labels_dir = out_root / "labels"
    summaries_dir = out_root / "summaries"
    configs_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    config_path = configs_dir / "valid_configurations_preprocessed_multiwindowing_main.json"
    labels_main_path = labels_dir / "labels_main_drop_dpleural_only.csv"
    summary_path = summaries_dir / "preprocessed_multiwindowing_grid_summary.json"

    config_path.write_text(json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8")
    labels_main_df.to_csv(labels_main_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path, labels_main_path, summary_path


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    results_root = Path(args.results_root)
    preproc_names = dedupe_preserve_order(args.preproc_names)

    templates = build_templates()
    labels_main_df, excluded_ids = build_labels_main(Path(args.labels_full_csv))
    labels_main_path = out_root / "labels" / "labels_main_drop_dpleural_only.csv"
    feature_manifest = build_feature_manifest(
        results_root=results_root,
        preproc_names=preproc_names,
        feature_modes=args.feature_modes,
    )
    configs = build_configs(
        feature_manifest=feature_manifest,
        templates=templates,
        labels_csv=str(labels_main_path),
        use_masks_variants=args.use_masks_variants,
    )
    summary = summarize(
        feature_manifest=feature_manifest,
        templates=templates,
        configs=configs,
        args=args,
        labels_main_csv=labels_main_path,
        excluded_ids=excluded_ids,
    )

    if args.count_only:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    config_path, labels_path, summary_path = write_outputs(
        out_root=out_root,
        summary=summary,
        configs=configs,
        labels_main_df=labels_main_df,
    )
    print(f"config_path: {config_path}")
    print(f"labels_main_path: {labels_path}")
    print(f"summary_path: {summary_path}")
    print(f"config_count: {len(configs)}")


if __name__ == "__main__":
    main()
