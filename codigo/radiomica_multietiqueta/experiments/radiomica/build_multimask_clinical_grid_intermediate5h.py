import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

FEATURE_VARIANT_WHITELIST = [
    "clinical_only__clin_sd_geom",
    "radiomics__basic__nodule",
    "radiomics__basic__nodule__clin_sd_geom",
    "radiomics__extended__nodule__clin_sd_geom",
    "radiomics__basic__lung__nodule__clin_sd_geom",
    "radiomics__extended__lung__nodule__clin_sd_geom",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build intermediate multimask clinical grid with bounded runtime.")
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
        }
    )



def build_intermediate_templates():
    templates = []
    seed = 42

    for split_strategy in ["kfold", "labelset_stratified"]:
        for c_value, class_weight, weight_tag in [
            (0.1, None, "unw"),
            (1.0, None, "unw"),
            (1.0, "balanced", "bal"),
            (3.0, "balanced", "bal"),
        ]:
            add_template(
                templates=templates,
                name=f"logreg__ovr__std__none__C{str(c_value).replace('.', 'p')}__{weight_tag}__{split_strategy}__s{seed}",
                classifier="LogisticRegression",
                multilabel_strategy="one_vs_rest",
                scaler="StandardScaler",
                dimred="none",
                split_strategy=split_strategy,
                seed=seed,
                params={
                    "C": c_value,
                    "class_weight": class_weight,
                    "solver": "liblinear",
                    "max_iter": 3000,
                },
                family="LogisticRegression",
            )

    for split_strategy in ["kfold", "labelset_stratified"]:
        for c_value in [0.1, 1.0, 3.0]:
            add_template(
                templates=templates,
                name=f"svm_lin__ovr__std__varth_0p01__C{str(c_value).replace('.', 'p')}__{split_strategy}__s{seed}",
                classifier="SVM",
                multilabel_strategy="one_vs_rest",
                scaler="StandardScaler",
                dimred="varth_0.01",
                split_strategy=split_strategy,
                seed=seed,
                params={
                    "kernel": "linear",
                    "C": c_value,
                    "max_iter": 5000,
                },
                family="SVM_linear",
            )

    for split_strategy in ["kfold", "labelset_stratified"]:
        for n_neighbors, weights in [(5, "uniform"), (5, "distance"), (11, "distance")]:
            add_template(
                templates=templates,
                name=f"knn__multi__std__pca_95__k{n_neighbors}__w{weights}__{split_strategy}__s{seed}",
                classifier="KNN",
                multilabel_strategy="multi_output",
                scaler="StandardScaler",
                dimred="pca_95",
                split_strategy=split_strategy,
                seed=seed,
                params={
                    "n_neighbors": n_neighbors,
                    "weights": weights,
                },
                family="KNN",
            )

    for family_name in ["RandomForest", "ExtraTrees"]:
        for split_strategy in ["kfold", "labelset_stratified"]:
            for n_estimators, max_features in [(300, "sqrt"), (300, "log2"), (500, "sqrt")]:
                add_template(
                    templates=templates,
                    name=(
                        f"{family_name.lower()}__ovr__none__none"
                        f"__n{n_estimators}__mf{max_features}__{split_strategy}__s{seed}"
                    ),
                    classifier=family_name,
                    multilabel_strategy="one_vs_rest",
                    scaler="none",
                    dimred="none",
                    split_strategy=split_strategy,
                    seed=seed,
                    params={
                        "n_estimators": n_estimators,
                        "max_features": max_features,
                        "max_depth": None,
                        "class_weight": "balanced",
                        "n_jobs": 1,
                        "random_state": seed,
                    },
                    family=family_name,
                )

    for family_name in ["XGBoost", "LightGBM"]:
        for split_strategy in ["kfold", "labelset_stratified"]:
            for n_estimators, learning_rate in [(120, 0.1), (200, 0.05)]:
                params = {
                    "n_estimators": n_estimators,
                    "max_depth": 3,
                    "learning_rate": learning_rate,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "n_jobs": 1,
                    "random_state": seed,
                }
                if family_name == "LightGBM":
                    params["num_leaves"] = 15
                    params["verbose"] = -1
                add_template(
                    templates=templates,
                    name=(
                        f"{family_name.lower()}__ovr__none__varth_0p01"
                        f"__n{n_estimators}__d3__lr{str(learning_rate).replace('.', 'p')}__{split_strategy}__s{seed}"
                    ),
                    classifier=family_name,
                    multilabel_strategy="one_vs_rest",
                    scaler="none",
                    dimred="varth_0.01",
                    split_strategy=split_strategy,
                    seed=seed,
                    params=params,
                    family=family_name,
                )

    names = [template["name"] for template in templates]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicated template names detected: {duplicates[:10]}")
    return templates



def summarize_templates(templates):
    by_family = defaultdict(int)
    by_split = defaultdict(int)
    for template in templates:
        by_family[template["family"]] += 1
        by_split[template["split_strategy"]] += 1
    return {
        "template_count": len(templates),
        "template_count_by_family": dict(sorted(by_family.items())),
        "template_count_by_split": dict(sorted(by_split.items())),
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
                "target_cols": ["Hemorragia", "Neumotórax"],
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

    templates = build_intermediate_templates()
    template_summary = summarize_templates(templates)

    if args.count_only:
        summary = {
            "mode": "count_only",
            "feature_variant_whitelist_count": len(FEATURE_VARIANT_WHITELIST),
            "feature_variant_whitelist": FEATURE_VARIANT_WHITELIST,
            **template_summary,
            "expected_main_config_count": len(FEATURE_VARIANT_WHITELIST) * len(templates),
            "endpoint": "main_hn_derive_no_comp",
            "target_cols": ["Hemorragia", "Neumotórax"],
            "label_mode": "derive_no_complication",
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    from build_multimask_clinical_grid import build_feature_sets

    manifest, labels_main_csv, _, excluded_ids = build_feature_sets(args, out_root)
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
        **template_summary,
        "target_cols": ["Hemorragia", "Neumotórax"],
        "label_mode": "derive_no_complication",
        "runtime_design_note": "Intermediate grid intended to stay below about 5h with cheap model families and conservative launch settings. This is a planning target, not a strict guarantee.",
    }
    (summaries_root / "grid_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
