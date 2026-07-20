import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

FEATURE_VARIANT_WHITELIST = [
    "radiomics__basic__lung__nodule__vessels__clin_full",
    "radiomics__basic__lung__nodule__vessels__clin_sd",
    "radiomics__basic__vessels",
    "radiomics__extended__lung__nodule__vessels__clin_sd_geom",
    "radiomics__basic__nodule__vessels__clin_sd_geom",
]

TARGET_COLS = ["Hemorragia", "Neumotórax"]
SEED_MAIN = 42
SEED_ALT = 1337


def parse_args():
    parser = argparse.ArgumentParser(description="Build a small imbalance-aware multimask clinical grid.")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument(
        "--basic_clean_csv",
        type=str,
        default="/home/mcribilles/tfm/codigo_nuevo/radiomica_runs/extract_radiomics_all_merged_final__2026-06-03__10-57-23/base_clean_for_grid/final_basic_all_clean.csv",
    )
    parser.add_argument(
        "--extended_clean_csv",
        type=str,
        default="/home/mcribilles/tfm/codigo_nuevo/radiomica_runs/extract_radiomics_all_merged_final__2026-06-03__10-57-23/base_clean_for_grid/final_extended_all_clean.csv",
    )
    parser.add_argument(
        "--labels_full_csv",
        type=str,
        default="/home/mcribilles/tfm/clinical_data/210pacientes/clinical_data_limpios.csv",
    )
    parser.add_argument(
        "--clinical_full_csv",
        type=str,
        default="/home/mcribilles/tfm/clinical_data/210pacientes/clinical_data_limpios.csv",
    )
    parser.add_argument(
        "--clinical_sd_csv",
        type=str,
        default="/home/mcribilles/tfm/clinical_data/210pacientes/clinical_data_limpios_SD.csv",
    )
    parser.add_argument(
        "--clinical_sd_geom_csv",
        type=str,
        default="/home/mcribilles/tfm/clinical_data/210pacientes/clinical_data_SD_con_geometria_210.csv",
    )
    parser.add_argument("--count_only", action="store_true")
    return parser.parse_args()


def fmt_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:g}".replace("-", "m").replace(".", "p")
    return str(value).replace("-", "m").replace(".", "p")


def add_template(
    templates,
    name,
    classifier,
    multilabel_strategy,
    scaler,
    dimred,
    split_strategy,
    seed,
    params,
    family,
    imbalance_mode,
):
    templates.append(
        {
            "name": name,
            "classifier": classifier,
            "multilabel_strategy": multilabel_strategy,
            "scaler": scaler,
            "dimred": dimred,
            "split_strategy": split_strategy,
            "seed": seed,
            "params": dict(params),
            "include_in_exploratory": False,
            "family": family,
            "imbalance_mode": imbalance_mode,
        }
    )


def compute_label_ratios(labels_main_csv: Path):
    df = pd.read_csv(labels_main_csv)
    ratios = {}
    for col in TARGET_COLS:
        series = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        pos = int(series.sum())
        neg = int(len(series) - pos)
        ratios[col] = {
            "pos": pos,
            "neg": neg,
            "scale_pos_weight": round(neg / max(pos, 1), 6),
        }
    derived_no_comp = ((pd.to_numeric(df["Hemorragia"], errors="coerce").fillna(0).astype(int) == 0) & (pd.to_numeric(df["Neumotórax"], errors="coerce").fillna(0).astype(int) == 0)).astype(int)
    ratios["Sin_complicacion_derivada"] = {
        "pos": int(derived_no_comp.sum()),
        "neg": int(len(derived_no_comp) - int(derived_no_comp.sum())),
    }
    ratios["n_rows"] = int(len(df))
    return ratios


def build_templates(scale_pos_weight_h, scale_pos_weight_n):
    templates = []
    split_strategy = "labelset_stratified"

    add_template(
        templates=templates,
        name="gbt__chain__none__varth_0p01__n200__lr0p1__d3__baseline__labelset_stratified__s42",
        classifier="GradientBoosting",
        multilabel_strategy="classifier_chain",
        scaler="none",
        dimred="varth_0.01",
        split_strategy=split_strategy,
        seed=SEED_MAIN,
        params={
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": SEED_MAIN,
        },
        family="GradientBoosting",
        imbalance_mode="baseline_top1",
    )

    add_template(
        templates=templates,
        name="gbt__chain__none__varth_0p01__n400__lr0p03__d3__baseline__labelset_stratified__s42",
        classifier="GradientBoosting",
        multilabel_strategy="classifier_chain",
        scaler="none",
        dimred="varth_0.01",
        split_strategy=split_strategy,
        seed=SEED_MAIN,
        params={
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 3,
            "random_state": SEED_MAIN,
        },
        family="GradientBoosting",
        imbalance_mode="baseline_top2",
    )

    add_template(
        templates=templates,
        name="xgboost__chain__none__varth_0p01__n700__d6__lr0p03__baseline__labelset_stratified__s1337",
        classifier="XGBoost",
        multilabel_strategy="classifier_chain",
        scaler="none",
        dimred="varth_0.01",
        split_strategy=split_strategy,
        seed=SEED_ALT,
        params={
            "n_estimators": 700,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": 1,
            "random_state": SEED_ALT,
        },
        family="XGBoost",
        imbalance_mode="baseline_top4",
    )

    add_template(
        templates=templates,
        name="lightgbm__chain__none__varth_0p01__n300__d3__lr0p1__baseline__labelset_stratified__s42",
        classifier="LightGBM",
        multilabel_strategy="classifier_chain",
        scaler="none",
        dimred="varth_0.01",
        split_strategy=split_strategy,
        seed=SEED_MAIN,
        params={
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "num_leaves": 15,
            "verbose": -1,
            "n_jobs": 1,
            "random_state": SEED_MAIN,
        },
        family="LightGBM",
        imbalance_mode="baseline_top10_family",
    )

    add_template(
        templates=templates,
        name="logreg__ovr__std__varth_0p01__C1__bal__labelset_stratified__s42",
        classifier="LogisticRegression",
        multilabel_strategy="one_vs_rest",
        scaler="StandardScaler",
        dimred="varth_0.01",
        split_strategy=split_strategy,
        seed=SEED_MAIN,
        params={
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 3000,
        },
        family="LogisticRegression",
        imbalance_mode="class_weight_balanced",
    )

    add_template(
        templates=templates,
        name="svm_lin__ovr__std__varth_0p01__C1__bal__labelset_stratified__s42",
        classifier="SVM",
        multilabel_strategy="one_vs_rest",
        scaler="StandardScaler",
        dimred="varth_0.01",
        split_strategy=split_strategy,
        seed=SEED_MAIN,
        params={
            "kernel": "linear",
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 5000,
        },
        family="SVM_linear",
        imbalance_mode="class_weight_balanced",
    )

    add_template(
        templates=templates,
        name="extratrees__ovr__none__none__n500__mfsqrt__dnone__balsub__labelset_stratified__s42",
        classifier="ExtraTrees",
        multilabel_strategy="one_vs_rest",
        scaler="none",
        dimred="none",
        split_strategy=split_strategy,
        seed=SEED_MAIN,
        params={
            "n_estimators": 500,
            "max_features": "sqrt",
            "max_depth": None,
            "class_weight": "balanced_subsample",
            "n_jobs": 1,
            "random_state": SEED_MAIN,
        },
        family="ExtraTrees",
        imbalance_mode="class_weight_balanced_subsample",
    )

    add_template(
        templates=templates,
        name="lightgbm__ovr__none__varth_0p01__n300__d3__lr0p1__bal__labelset_stratified__s42",
        classifier="LightGBM",
        multilabel_strategy="one_vs_rest",
        scaler="none",
        dimred="varth_0.01",
        split_strategy=split_strategy,
        seed=SEED_MAIN,
        params={
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "num_leaves": 15,
            "class_weight": "balanced",
            "verbose": -1,
            "n_jobs": 1,
            "random_state": SEED_MAIN,
        },
        family="LightGBM",
        imbalance_mode="class_weight_balanced",
    )

    add_template(
        templates=templates,
        name=f"xgboost__ovr__none__varth_0p01__n300__d3__lr0p1__spwh{fmt_value(scale_pos_weight_h)}__labelset_stratified__s42",
        classifier="XGBoost",
        multilabel_strategy="one_vs_rest",
        scaler="none",
        dimred="varth_0.01",
        split_strategy=split_strategy,
        seed=SEED_MAIN,
        params={
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": float(scale_pos_weight_h),
            "max_delta_step": 1,
            "n_jobs": 1,
            "random_state": SEED_MAIN,
        },
        family="XGBoost",
        imbalance_mode="scale_pos_weight_hemorragia",
    )

    add_template(
        templates=templates,
        name=f"xgboost__ovr__none__varth_0p01__n300__d3__lr0p1__spwn{fmt_value(scale_pos_weight_n)}__labelset_stratified__s42",
        classifier="XGBoost",
        multilabel_strategy="one_vs_rest",
        scaler="none",
        dimred="varth_0.01",
        split_strategy=split_strategy,
        seed=SEED_MAIN,
        params={
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": float(scale_pos_weight_n),
            "max_delta_step": 1,
            "n_jobs": 1,
            "random_state": SEED_MAIN,
        },
        family="XGBoost",
        imbalance_mode="scale_pos_weight_neumotorax",
    )

    names = [template["name"] for template in templates]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicated template names detected: {duplicates[:10]}")
    return templates


def summarize_templates(templates):
    by_family = defaultdict(int)
    by_imbalance = defaultdict(int)
    for template in templates:
        by_family[template["family"]] += 1
        by_imbalance[template["imbalance_mode"]] += 1
    return {
        "template_count": len(templates),
        "template_count_by_family": dict(sorted(by_family.items())),
        "template_count_by_imbalance_mode": dict(sorted(by_imbalance.items())),
    }


def build_main_configs(manifest, templates):
    configs = []
    for entry in manifest:
        for tmpl in templates:
            config = {
                "config_name": f"{entry['variant_name']}__main_hn_derive_no_comp__{tmpl['name']}",
                "features_csv": entry["features_csv"],
                "labels_csv": entry["labels_csv_main"],
                "id_col": "patient_id",
                "target_cols": list(TARGET_COLS),
                "label_mode": "derive_no_complication",
                "use_masks": entry["mask_combo_label"],
                "n_splits": 5,
                "random_state": tmpl["seed"],
                "shuffle_folds": True,
                "split_strategy": tmpl["split_strategy"],
                "scaler": tmpl["scaler"],
                "classifier": tmpl["classifier"],
                "dimred": tmpl["dimred"],
                "multilabel_strategy": tmpl["multilabel_strategy"],
                "save_models": False,
                "save_predictions": False,
            }
            config.update(tmpl["params"])
            configs.append(config)
    names = [cfg["config_name"] for cfg in configs]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicated config names detected: {duplicates[:10]}")
    return configs


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    from build_multimask_clinical_grid import build_feature_sets

    manifest, labels_main_csv, _, excluded_ids = build_feature_sets(args, out_root)
    label_ratios = compute_label_ratios(labels_main_csv)
    templates = build_templates(
        scale_pos_weight_h=label_ratios["Hemorragia"]["scale_pos_weight"],
        scale_pos_weight_n=label_ratios["Neumotórax"]["scale_pos_weight"],
    )
    template_summary = summarize_templates(templates)

    if args.count_only:
        summary = {
            "mode": "count_only",
            "feature_variant_whitelist_count": len(FEATURE_VARIANT_WHITELIST),
            "feature_variant_whitelist": FEATURE_VARIANT_WHITELIST,
            **template_summary,
            "expected_main_config_count": len(FEATURE_VARIANT_WHITELIST) * len(templates),
            "endpoint": "main_hn_derive_no_comp",
            "target_cols": list(TARGET_COLS),
            "label_mode": "derive_no_complication",
            "label_ratios": label_ratios,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    selected_manifest = [entry for entry in manifest if entry["variant_name"] in FEATURE_VARIANT_WHITELIST]
    selected_names = {entry["variant_name"] for entry in selected_manifest}
    missing_variants = [name for name in FEATURE_VARIANT_WHITELIST if name not in selected_names]
    if missing_variants:
        raise ValueError(f"Missing expected feature variants: {missing_variants}")

    main_configs = build_main_configs(selected_manifest, templates)

    configs_root = out_root / "configs"
    configs_root.mkdir(parents=True, exist_ok=True)
    main_json = configs_root / "valid_configurations_multimask_clinical_main.json"
    main_json.write_text(json.dumps(main_configs, indent=2, ensure_ascii=False))

    summaries_root = out_root / "summaries"
    summaries_root.mkdir(parents=True, exist_ok=True)
    (summaries_root / "template_summary.json").write_text(json.dumps(template_summary, indent=2, ensure_ascii=False))

    summary = {
        "out_root": str(out_root),
        "feature_variant_count_selected": len(selected_manifest),
        "feature_variant_names_selected": [entry["variant_name"] for entry in selected_manifest],
        "main_config_count": len(main_configs),
        "labels_main_csv": str(labels_main_csv),
        "excluded_only_dpleural_patient_ids_for_main": excluded_ids,
        "label_ratios": label_ratios,
        **template_summary,
        "target_cols": list(TARGET_COLS),
        "label_mode": "derive_no_complication",
        "design_note": "Small grid centered on top historical feature combinations plus native imbalance-aware variants.",
    }
    (summaries_root / "grid_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
