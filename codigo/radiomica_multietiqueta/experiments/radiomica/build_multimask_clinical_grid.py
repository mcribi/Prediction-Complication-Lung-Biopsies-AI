import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BASE_META_COLS = ["patient_id", "image_path", "image_size", "image_spacing"]
MASK_ORDER = ["lung", "nodule", "trachea_bronchia", "vessels"]
TARGET_COLS_FULL = ["Derrame_pleural", "Hemorragia", "Neumotórax", "Sin_complicación"]
CLINICAL_DROP_COLS = set(["patient_id", "Complicacion_binaria"] + TARGET_COLS_FULL)

DEFAULT_BASIC_CLEAN = Path(
    "/home/mcribilles/tfm/codigo_nuevo/radiomica_runs/extract_radiomics_all_merged_final__2026-06-03__10-57-23/base_clean_for_grid/final_basic_all_clean.csv"
)
DEFAULT_EXTENDED_CLEAN = Path(
    "/home/mcribilles/tfm/codigo_nuevo/radiomica_runs/extract_radiomics_all_merged_final__2026-06-03__10-57-23/base_clean_for_grid/final_extended_all_clean.csv"
)
DEFAULT_LABELS_FULL = Path("/home/mcribilles/tfm/clinical_data/210pacientes/clinical_data_limpios.csv")
DEFAULT_CLINICAL_SOURCES = {
    "clin_full": Path("/home/mcribilles/tfm/clinical_data/210pacientes/clinical_data_limpios.csv"),
    "clin_sd": Path("/home/mcribilles/tfm/clinical_data/210pacientes/clinical_data_limpios_SD.csv"),
    "clin_sd_geom": Path("/home/mcribilles/tfm/clinical_data/210pacientes/clinical_data_SD_con_geometria_210.csv"),
}

MAIN_SEEDS = [42]
EXPLORATORY_EXTRA_SEEDS = [1337]


def parse_args():
    parser = argparse.ArgumentParser(description="Build multimask + clinical radiomics feature sets and JSON grids.")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--basic_clean_csv", type=str, default=str(DEFAULT_BASIC_CLEAN))
    parser.add_argument("--extended_clean_csv", type=str, default=str(DEFAULT_EXTENDED_CLEAN))
    parser.add_argument("--labels_full_csv", type=str, default=str(DEFAULT_LABELS_FULL))
    parser.add_argument("--clinical_full_csv", type=str, default=str(DEFAULT_CLINICAL_SOURCES["clin_full"]))
    parser.add_argument("--clinical_sd_csv", type=str, default=str(DEFAULT_CLINICAL_SOURCES["clin_sd"]))
    parser.add_argument("--clinical_sd_geom_csv", type=str, default=str(DEFAULT_CLINICAL_SOURCES["clin_sd_geom"]))
    return parser.parse_args()


def slugify_mask_combo(mask_combo):
    return "__".join(mask_combo)


def label_for_mask_combo(mask_combo):
    return "+".join(mask_combo)


def iter_mask_combos():
    for r in range(1, len(MASK_ORDER) + 1):
        for combo in itertools.combinations(MASK_ORDER, r):
            yield list(combo)


def get_mask_cols(df, mask_combo):
    prefixes = tuple(f"{mask}_" for mask in mask_combo)
    return [c for c in df.columns if c.startswith(prefixes)]


def load_clinical_features(path: Path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in CLINICAL_DROP_COLS]
    return df[["patient_id"] + feature_cols].copy(), feature_cols


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_main_labels(labels_full_df: pd.DataFrame):
    df = labels_full_df.copy()
    for c in TARGET_COLS_FULL:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    only_dpleural = (
        (df["Derrame_pleural"] == 1)
        & (df["Hemorragia"] == 0)
        & (df["Neumotórax"] == 0)
    )
    return df.loc[~only_dpleural].copy(), df.loc[only_dpleural, "patient_id"].astype(str).tolist()


def make_feature_entry(
    variant_name,
    feature_mode,
    mask_combo,
    clinical_variant,
    csv_path,
    n_rows,
    n_cols,
    n_patients,
    labels_main_csv,
    labels_expl_csv,
):
    return {
        "variant_name": variant_name,
        "feature_mode": feature_mode,
        "mask_combo": mask_combo,
        "mask_combo_label": "clinical_only" if not mask_combo else label_for_mask_combo(mask_combo),
        "clinical_variant": clinical_variant,
        "features_csv": str(csv_path),
        "labels_csv_main": str(labels_main_csv),
        "labels_csv_exploratory": str(labels_expl_csv),
        "rows": int(n_rows),
        "cols": int(n_cols),
        "patients": int(n_patients),
    }


def build_feature_sets(args, out_root: Path):
    features_root = out_root / "features"
    labels_root = out_root / "labels"
    summary_root = out_root / "summaries"
    features_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)

    labels_full_df = pd.read_csv(args.labels_full_csv)
    labels_main_df, excluded_ids = build_main_labels(labels_full_df)
    labels_main_csv = labels_root / "labels_main_drop_dpleural_only.csv"
    labels_expl_csv = labels_root / "labels_exploratory_all_labels.csv"
    write_csv(labels_main_df, labels_main_csv)
    write_csv(labels_full_df, labels_expl_csv)

    clinical_sources = {
        "clin_full": Path(args.clinical_full_csv),
        "clin_sd": Path(args.clinical_sd_csv),
        "clin_sd_geom": Path(args.clinical_sd_geom_csv),
    }
    clinical_feature_frames = {}
    clinical_feature_cols = {}
    for key, path in clinical_sources.items():
        df_clin, cols = load_clinical_features(path)
        clinical_feature_frames[key] = df_clin
        clinical_feature_cols[key] = cols

    mode_to_path = {
        "basic": Path(args.basic_clean_csv),
        "extended": Path(args.extended_clean_csv),
    }

    manifest = []

    for feature_mode, csv_path in mode_to_path.items():
        df_mode = pd.read_csv(csv_path)
        base_df = df_mode[[c for c in BASE_META_COLS if c in df_mode.columns]].copy()
        for mask_combo in iter_mask_combos():
            mask_cols = get_mask_cols(df_mode, mask_combo)
            variant_slug = slugify_mask_combo(mask_combo)
            radiomics_df = pd.concat([base_df, df_mode[mask_cols].copy()], axis=1)

            out_csv = features_root / f"radiomics__{feature_mode}__{variant_slug}.csv"
            write_csv(radiomics_df, out_csv)
            manifest.append(
                make_feature_entry(
                    variant_name=f"radiomics__{feature_mode}__{variant_slug}",
                    feature_mode=feature_mode,
                    mask_combo=mask_combo,
                    clinical_variant="none",
                    csv_path=out_csv,
                    n_rows=len(radiomics_df),
                    n_cols=radiomics_df.shape[1],
                    n_patients=radiomics_df["patient_id"].nunique(),
                    labels_main_csv=labels_main_csv,
                    labels_expl_csv=labels_expl_csv,
                )
            )

            for clin_key, clin_df in clinical_feature_frames.items():
                merged = radiomics_df.merge(clin_df, on="patient_id", how="left")
                out_csv = features_root / f"radiomics__{feature_mode}__{variant_slug}__{clin_key}.csv"
                write_csv(merged, out_csv)
                manifest.append(
                    make_feature_entry(
                        variant_name=f"radiomics__{feature_mode}__{variant_slug}__{clin_key}",
                        feature_mode=feature_mode,
                        mask_combo=mask_combo,
                        clinical_variant=clin_key,
                        csv_path=out_csv,
                        n_rows=len(merged),
                        n_cols=merged.shape[1],
                        n_patients=merged["patient_id"].nunique(),
                        labels_main_csv=labels_main_csv,
                        labels_expl_csv=labels_expl_csv,
                    )
                )

    for clin_key, clin_df in clinical_feature_frames.items():
        out_csv = features_root / f"clinical_only__{clin_key}.csv"
        write_csv(clin_df, out_csv)
        manifest.append(
            make_feature_entry(
                variant_name=f"clinical_only__{clin_key}",
                feature_mode="clinical_only",
                mask_combo=[],
                clinical_variant=clin_key,
                csv_path=out_csv,
                n_rows=len(clin_df),
                n_cols=clin_df.shape[1],
                n_patients=clin_df["patient_id"].nunique(),
                labels_main_csv=labels_main_csv,
                labels_expl_csv=labels_expl_csv,
            )
        )

    feature_manifest = {
        "feature_variant_count": len(manifest),
        "radiomics_variant_count": sum(1 for row in manifest if row["feature_mode"] != "clinical_only"),
        "clinical_only_variant_count": sum(1 for row in manifest if row["feature_mode"] == "clinical_only"),
        "excluded_only_dpleural_patient_ids_for_main": excluded_ids,
        "clinical_feature_counts": {k: len(v) for k, v in clinical_feature_cols.items()},
        "variants": manifest,
    }
    (summary_root / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2, ensure_ascii=False))
    return manifest, labels_main_csv, labels_expl_csv, excluded_ids


def fmt_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, float):
        text = f"{value:g}"
        return text.replace("-", "m").replace(".", "p")
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
    include_in_exploratory,
    family,
):
    template = {
        "name": name,
        "classifier": classifier,
        "multilabel_strategy": multilabel_strategy,
        "scaler": scaler,
        "dimred": dimred,
        "split_strategy": split_strategy,
        "seed": seed,
        "params": dict(params),
        "include_in_exploratory": include_in_exploratory,
        "family": family,
    }
    templates.append(template)


def build_template_library():
    templates = []

    all_seeds = MAIN_SEEDS + EXPLORATORY_EXTRA_SEEDS

    for seed in MAIN_SEEDS:
        for split_strategy in ["kfold", "labelset_stratified"]:
            for dimred in ["none", "varth_0.01", "pca_90", "pca_95"]:
                for multilabel_strategy in ["one_vs_rest", "classifier_chain"]:
                    for c_value in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
                        for class_weight in [None, "balanced"]:
                            suffix = "unw" if class_weight is None else "bal"
                            name = (
                                f"logreg__{multilabel_strategy}__std__{dimred}"
                                f"__C{fmt_value(c_value)}__{suffix}__{split_strategy}__s{seed}"
                            )
                            add_template(
                                templates=templates,
                                name=name,
                                classifier="LogisticRegression",
                                multilabel_strategy=multilabel_strategy,
                                scaler="StandardScaler",
                                dimred=dimred,
                                split_strategy=split_strategy,
                                seed=seed,
                                params={
                                    "C": c_value,
                                    "class_weight": class_weight,
                                    "solver": "liblinear",
                                    "max_iter": 3000,
                                },
                                include_in_exploratory=True,
                                family="LogisticRegression",
                            )

    for seed in MAIN_SEEDS:
        for split_strategy in ["kfold", "labelset_stratified"]:
            for dimred in ["none", "varth_0.01", "pca_95"]:
                for multilabel_strategy in ["one_vs_rest", "classifier_chain"]:
                    for c_value in [0.1, 0.3, 1.0, 3.0, 10.0]:
                        name = (
                            f"svm_lin__{multilabel_strategy}__std__{dimred}"
                            f"__C{fmt_value(c_value)}__{split_strategy}__s{seed}"
                        )
                        add_template(
                            templates=templates,
                            name=name,
                            classifier="SVM",
                            multilabel_strategy=multilabel_strategy,
                            scaler="StandardScaler",
                            dimred=dimred,
                            split_strategy=split_strategy,
                            seed=seed,
                            params={"kernel": "linear", "C": c_value, "max_iter": 5000},
                            include_in_exploratory=True,
                            family="SVM_linear",
                        )

    for seed in MAIN_SEEDS:
        for split_strategy in ["kfold", "labelset_stratified"]:
            for dimred in ["pca_90", "pca_95"]:
                for c_value in [0.3, 1.0, 3.0, 10.0]:
                    for gamma in ["scale", "auto"]:
                        name = (
                            f"svm_rbf__ovr__std__{dimred}__C{fmt_value(c_value)}"
                            f"__g{fmt_value(gamma)}__{split_strategy}__s{seed}"
                        )
                        add_template(
                            templates=templates,
                            name=name,
                            classifier="SVM",
                            multilabel_strategy="one_vs_rest",
                            scaler="StandardScaler",
                            dimred=dimred,
                            split_strategy=split_strategy,
                            seed=seed,
                            params={"kernel": "rbf", "C": c_value, "gamma": gamma, "max_iter": 5000},
                            include_in_exploratory=True,
                            family="SVM_rbf",
                        )

    for seed in MAIN_SEEDS:
        for split_strategy in ["kfold", "labelset_stratified"]:
            for dimred in ["none", "pca_95"]:
                for n_neighbors in [3, 5, 9, 15]:
                    for weights in ["uniform", "distance"]:
                        name = (
                            f"knn__multi__std__{dimred}__k{n_neighbors}"
                            f"__w{weights}__{split_strategy}__s{seed}"
                        )
                        add_template(
                            templates=templates,
                            name=name,
                            classifier="KNN",
                            multilabel_strategy="multi_output",
                            scaler="StandardScaler",
                            dimred=dimred,
                            split_strategy=split_strategy,
                            seed=seed,
                            params={"n_neighbors": n_neighbors, "weights": weights},
                            include_in_exploratory=True,
                            family="KNN",
                        )

    for family_name, classifier_name in [("RandomForest", "RandomForest"), ("ExtraTrees", "ExtraTrees")]:
        for seed in all_seeds:
            include_in_exploratory = seed in EXPLORATORY_EXTRA_SEEDS
            for split_strategy in ["kfold", "labelset_stratified"]:
                for multilabel_strategy in ["one_vs_rest", "multi_output", "classifier_chain"]:
                    for n_estimators in [300, 700, 1200]:
                        for max_features in ["sqrt", "log2"]:
                            for max_depth in [None, 16]:
                                name = (
                                    f"{family_name.lower()}__{multilabel_strategy}__none__none"
                                    f"__n{n_estimators}__mf{fmt_value(max_features)}"
                                    f"__d{fmt_value(max_depth)}__{split_strategy}__s{seed}"
                                )
                                add_template(
                                    templates=templates,
                                    name=name,
                                    classifier=classifier_name,
                                    multilabel_strategy=multilabel_strategy,
                                    scaler="none",
                                    dimred="none",
                                    split_strategy=split_strategy,
                                    seed=seed,
                                    params={
                                        "n_estimators": n_estimators,
                                        "max_features": max_features,
                                        "max_depth": max_depth,
                                        "class_weight": "balanced",
                                        "n_jobs": 1,
                                        "random_state": seed,
                                    },
                                    include_in_exploratory=include_in_exploratory,
                                    family=family_name,
                                )

    for seed in MAIN_SEEDS:
        for split_strategy in ["kfold", "labelset_stratified"]:
            for multilabel_strategy in ["one_vs_rest", "classifier_chain"]:
                for max_depth in [None, 8, 16]:
                    for min_samples_leaf in [1, 2]:
                        name = (
                            f"dt__{multilabel_strategy}__none__none__d{fmt_value(max_depth)}"
                            f"__leaf{min_samples_leaf}__{split_strategy}__s{seed}"
                        )
                        add_template(
                            templates=templates,
                            name=name,
                            classifier="DecisionTree",
                            multilabel_strategy=multilabel_strategy,
                            scaler="none",
                            dimred="none",
                            split_strategy=split_strategy,
                            seed=seed,
                            params={
                                "max_depth": max_depth,
                                "min_samples_leaf": min_samples_leaf,
                                "class_weight": "balanced",
                                "random_state": seed,
                            },
                            include_in_exploratory=True,
                            family="DecisionTree",
                        )

    for seed in MAIN_SEEDS:
        for split_strategy in ["kfold", "labelset_stratified"]:
            for dimred in ["none", "varth_0.01"]:
                for multilabel_strategy in ["one_vs_rest", "classifier_chain"]:
                    for n_estimators in [200, 400]:
                        for learning_rate in [0.03, 0.1]:
                            for max_depth in [2, 3]:
                                name = (
                                    f"gbt__{multilabel_strategy}__none__{dimred}__n{n_estimators}"
                                    f"__lr{fmt_value(learning_rate)}__d{max_depth}__{split_strategy}__s{seed}"
                                )
                                add_template(
                                    templates=templates,
                                    name=name,
                                    classifier="GradientBoosting",
                                    multilabel_strategy=multilabel_strategy,
                                    scaler="none",
                                    dimred=dimred,
                                    split_strategy=split_strategy,
                                    seed=seed,
                                    params={
                                        "n_estimators": n_estimators,
                                        "learning_rate": learning_rate,
                                        "max_depth": max_depth,
                                        "random_state": seed,
                                    },
                                    include_in_exploratory=True,
                                    family="GradientBoosting",
                                )

    for seed in MAIN_SEEDS:
        for split_strategy in ["kfold", "labelset_stratified"]:
            for n_estimators in [100, 300, 500]:
                for learning_rate in [0.03, 0.1, 0.3]:
                    name = (
                        f"adaboost__ovr__none__none__n{n_estimators}"
                        f"__lr{fmt_value(learning_rate)}__{split_strategy}__s{seed}"
                    )
                    add_template(
                        templates=templates,
                        name=name,
                        classifier="AdaBoost",
                        multilabel_strategy="one_vs_rest",
                        scaler="none",
                        dimred="none",
                        split_strategy=split_strategy,
                        seed=seed,
                        params={
                            "n_estimators": n_estimators,
                            "learning_rate": learning_rate,
                            "random_state": seed,
                        },
                        include_in_exploratory=True,
                        family="AdaBoost",
                    )

    for family_name, classifier_name in [("XGBoost", "XGBoost"), ("LightGBM", "LightGBM")]:
        for seed in all_seeds:
            include_in_exploratory = seed in EXPLORATORY_EXTRA_SEEDS
            for split_strategy in ["kfold", "labelset_stratified"]:
                for dimred in ["none", "varth_0.01"]:
                    for multilabel_strategy in ["one_vs_rest", "classifier_chain"]:
                        for n_estimators in [300, 700]:
                            for max_depth in [3, 6]:
                                for learning_rate in [0.03, 0.1]:
                                    name = (
                                        f"{family_name.lower()}__{multilabel_strategy}__none__{dimred}"
                                        f"__n{n_estimators}__d{max_depth}__lr{fmt_value(learning_rate)}"
                                        f"__{split_strategy}__s{seed}"
                                    )
                                    params = {
                                        "n_estimators": n_estimators,
                                        "max_depth": max_depth,
                                        "learning_rate": learning_rate,
                                        "subsample": 0.8,
                                        "colsample_bytree": 0.8,
                                        "n_jobs": 1,
                                        "random_state": seed,
                                    }
                                    if family_name == "LightGBM":
                                        params["num_leaves"] = 31 if max_depth == 6 else 15
                                        params["verbose"] = -1
                                    add_template(
                                        templates=templates,
                                        name=name,
                                        classifier=classifier_name,
                                        multilabel_strategy=multilabel_strategy,
                                        scaler="none",
                                        dimred=dimred,
                                        split_strategy=split_strategy,
                                        seed=seed,
                                        params=params,
                                        include_in_exploratory=include_in_exploratory,
                                        family=family_name,
                                    )

    for seed in all_seeds:
        include_in_exploratory = seed in EXPLORATORY_EXTRA_SEEDS
        for split_strategy in ["kfold", "labelset_stratified"]:
            for dimred in ["none", "varth_0.01"]:
                for n_estimators in [300, 700]:
                    for depth in [4, 8]:
                        for learning_rate in [0.03, 0.1]:
                            name = (
                                f"catboost__ovr__none__{dimred}__n{n_estimators}"
                                f"__d{depth}__lr{fmt_value(learning_rate)}__{split_strategy}__s{seed}"
                            )
                            add_template(
                                templates=templates,
                                name=name,
                                classifier="CatBoost",
                                multilabel_strategy="one_vs_rest",
                                scaler="none",
                                dimred=dimred,
                                split_strategy=split_strategy,
                                seed=seed,
                                params={
                                    "iterations": n_estimators,
                                    "depth": depth,
                                    "learning_rate": learning_rate,
                                    "verbose": False,
                                    "random_seed": seed,
                                },
                                include_in_exploratory=include_in_exploratory,
                                family="CatBoost",
                            )

    hidden_layers = {
        "h128": (128,),
        "h256x128": (256, 128),
        "h512x256": (512, 256),
    }
    for seed in MAIN_SEEDS:
        for split_strategy in ["kfold", "labelset_stratified"]:
            for dimred in ["varth_0.01", "pca_95"]:
                for hidden_name, hidden_tuple in hidden_layers.items():
                    for alpha in [1e-4, 1e-3, 1e-2]:
                        name = (
                            f"mlp__ovr__std__{dimred}__{hidden_name}"
                            f"__a{fmt_value(alpha)}__{split_strategy}__s{seed}"
                        )
                        add_template(
                            templates=templates,
                            name=name,
                            classifier="MLP",
                            multilabel_strategy="one_vs_rest",
                            scaler="StandardScaler",
                            dimred=dimred,
                            split_strategy=split_strategy,
                            seed=seed,
                            params={
                                "hidden_layer_sizes": list(hidden_tuple),
                                "alpha": alpha,
                                "learning_rate_init": 0.001,
                                "max_iter": 500,
                                "early_stopping": True,
                                "random_state": seed,
                            },
                            include_in_exploratory=True,
                            family="MLP",
                        )

    for seed in all_seeds:
        include_in_exploratory = seed in EXPLORATORY_EXTRA_SEEDS
        for split_strategy in ["kfold", "labelset_stratified"]:
            for multilabel_strategy in ["one_vs_rest", "multi_output"]:
                name = f"tabpfn__{multilabel_strategy}__none__none__{split_strategy}__s{seed}"
                add_template(
                    templates=templates,
                    name=name,
                    classifier="TabPFN",
                    multilabel_strategy=multilabel_strategy,
                    scaler="none",
                    dimred="none",
                    split_strategy=split_strategy,
                    seed=seed,
                    params={"device": "cpu", "N_ensemble_configurations": 32, "seed": seed},
                    include_in_exploratory=include_in_exploratory,
                    family="TabPFN",
                )

    names = [template["name"] for template in templates]
    dup_names = [name for name, count in Counter(names).items() if count > 1]
    if dup_names:
        raise ValueError(f"Duplicated template names detected, examples: {dup_names[:10]}")

    return templates


def build_configs(manifest, endpoint_name, labels_csv_key, target_cols, label_mode, templates):
    configs = []
    for entry in manifest:
        labels_csv = entry[labels_csv_key]
        use_masks_label = entry["mask_combo_label"]
        variant_tag = entry["variant_name"].replace("+", "plus")
        for tmpl in templates:
            config = {
                "config_name": f"{variant_tag}__{endpoint_name}__{tmpl['name']}",
                "features_csv": entry["features_csv"],
                "labels_csv": labels_csv,
                "id_col": "patient_id",
                "target_cols": list(target_cols),
                "label_mode": label_mode,
                "use_masks": use_masks_label,
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
    dup_names = [name for name, count in Counter(names).items() if count > 1]
    if dup_names:
        raise ValueError(f"Duplicated config names detected, examples: {dup_names[:10]}")
    return configs


def summarize_templates(templates):
    families = defaultdict(int)
    seeds = defaultdict(int)
    exploratory_families = defaultdict(int)
    for template in templates:
        families[template["family"]] += 1
        seeds[str(template["seed"])] += 1
        if template["include_in_exploratory"]:
            exploratory_families[template["family"]] += 1
    return {
        "template_count_main": len(templates),
        "template_count_exploratory": sum(1 for template in templates if template["include_in_exploratory"]),
        "template_count_by_family_main": dict(sorted(families.items())),
        "template_count_by_family_exploratory": dict(sorted(exploratory_families.items())),
        "template_count_by_seed_main": dict(sorted(seeds.items())),
    }


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest, labels_main_csv, labels_expl_csv, excluded_ids = build_feature_sets(args, out_root)
    templates = [t for t in build_template_library() if t["family"] != "TabPFN"]
    exploratory_templates = [t for t in templates if t["include_in_exploratory"]]

    main_configs = build_configs(
        manifest=manifest,
        endpoint_name="main_hn_derive_no_comp",
        labels_csv_key="labels_csv_main",
        target_cols=["Hemorragia", "Neumotórax"],
        label_mode="derive_no_complication",
        templates=templates,
    )
    exploratory_configs = build_configs(
        manifest=manifest,
        endpoint_name="exploratory_all_labels",
        labels_csv_key="labels_csv_exploratory",
        target_cols=["Derrame_pleural", "Hemorragia", "Neumotórax", "Sin_complicación"],
        label_mode="include_no_complication",
        templates=[t for t in exploratory_templates if t["split_strategy"] != "labelset_stratified"],
    )

    configs_root = out_root / "configs"
    configs_root.mkdir(parents=True, exist_ok=True)
    main_json = configs_root / "valid_configurations_multimask_clinical_main.json"
    exploratory_json = configs_root / "valid_configurations_multimask_clinical_exploratory.json"
    main_json.write_text(json.dumps(main_configs, indent=2, ensure_ascii=False))
    exploratory_json.write_text(json.dumps(exploratory_configs, indent=2, ensure_ascii=False))

    template_summary = summarize_templates(templates)
    (out_root / "summaries" / "template_summary.json").write_text(
        json.dumps(template_summary, indent=2, ensure_ascii=False)
    )

    summary = {
        "out_root": str(out_root),
        "feature_variant_count": len(manifest),
        "main_config_count": len(main_configs),
        "exploratory_config_count": len(exploratory_configs),
        "labels_main_csv": str(labels_main_csv),
        "labels_exploratory_csv": str(labels_expl_csv),
        "excluded_only_dpleural_patient_ids_for_main": excluded_ids,
        "template_count_main": len(templates),
        "template_count_exploratory": len(exploratory_templates),
        "feature_modes": sorted({row["feature_mode"] for row in manifest}),
        "clinical_variants": sorted({row["clinical_variant"] for row in manifest}),
        "template_count_by_family_main": template_summary["template_count_by_family_main"],
        "template_count_by_family_exploratory": template_summary["template_count_by_family_exploratory"],
        "template_count_by_seed_main": template_summary["template_count_by_seed_main"],
    }
    (out_root / "summaries" / "grid_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
