#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone

LABELS = ["Hemorragia", "Neumotórax"]
OUTCOMES = ["TP", "TN", "FP", "FN"]


def parse_args():
    parser = argparse.ArgumentParser(description="Verified OOF SHAP analysis for radiomics R07")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-runner", type=Path, required=True)
    parser.add_argument("--two-runner", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-display", type=int, default=20)
    parser.add_argument("--waterfall-display", type=int, default=12)
    return parser.parse_args()


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_diagnostic_feature(name: str) -> bool:
    value = str(name).lower()
    return bool(re.search(r"(^|_)diagnostics?_|(^|_)diag_", value))


def clean_feature_name(name: str) -> str:
    value = str(name)
    value = value.replace("original_", "orig_")
    value = value.replace("chain_pred_Hemorragia", "prediccion previa de hemorragia")
    return value


def positive_class_shap(explainer, x_aug: np.ndarray):
    raw = explainer.shap_values(x_aug, check_additivity=True)
    expected = explainer.expected_value
    if isinstance(raw, list):
        values = np.asarray(raw[1], dtype=float)
        base = float(np.asarray(expected).reshape(-1)[1])
    else:
        raw = np.asarray(raw)
        if raw.ndim == 3 and raw.shape[-1] == 2:
            values = np.asarray(raw[:, :, 1], dtype=float)
            base = float(np.asarray(expected).reshape(-1)[1])
        elif raw.ndim == 2:
            values = np.asarray(raw, dtype=float)
            base = float(np.asarray(expected).reshape(-1)[0])
        else:
            raise ValueError(f"Unsupported SHAP output shape: {raw.shape}")
    return values, base


def outcome_name(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    return "FN"


def transform_r07_with_alignment(model, x_test, feature_names):
    expected_steps = [name for name, _ in model.steps]
    if expected_steps != ["imputer", "vt", "classifier"]:
        raise ValueError(f"Unexpected R07 pipeline steps: {expected_steps}")
    imputed = model.named_steps["imputer"].transform(x_test)
    selector = model.named_steps["vt"]
    support = np.asarray(selector.get_support(), dtype=bool)
    if support.shape != (len(feature_names),):
        raise ValueError("VarianceThreshold support does not match original feature columns")
    transformed = selector.transform(imputed)
    retained_names = np.asarray(feature_names, dtype=str)[support].tolist()
    if transformed.shape[1] != len(retained_names):
        raise ValueError("Transformed matrix and retained feature names are not aligned")
    return imputed, transformed, support, retained_names


def save_global_plots(values, data, feature_names, label, out_dir, max_display):
    safe = label.replace("ó", "o")
    plt.figure()
    shap.summary_plot(values, data, feature_names=feature_names, max_display=max_display,
                      show=False, plot_size=(11, 8.5))
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xlabel("Valor SHAP (impacto en la salida del modelo)")
    ax.set_title(f"Distribucion global de valores SHAP OOF - {label}", pad=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"shap_beeswarm_{safe}.png", dpi=250, bbox_inches="tight")
    fig.savefig(out_dir / f"shap_beeswarm_{safe}.pdf", bbox_inches="tight")
    plt.close(fig)

    importance = np.mean(np.abs(values), axis=0)
    order = np.argsort(importance)[::-1][:max_display][::-1]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.barh(np.asarray(feature_names)[order], importance[order], color="#2878B5")
    ax.set_xlabel("Media del valor SHAP absoluto")
    ax.set_title(f"Importancia global SHAP OOF - {label}", pad=12)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / f"shap_importance_{safe}.png", dpi=250, bbox_inches="tight")
    fig.savefig(out_dir / f"shap_importance_{safe}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_waterfall(values, base, data, feature_names, title, output_png, max_display):
    explanation = shap.Explanation(values=np.asarray(values, dtype=float),
                                   base_values=float(base),
                                   data=np.asarray(data, dtype=float),
                                   feature_names=list(feature_names))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(11, 6.5)
    plt.title(title, fontsize=10, pad=12)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def save_gallery(paths, label, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5))
    for ax, outcome in zip(axes.ravel(), OUTCOMES):
        path = paths.get(outcome)
        if path is None:
            ax.text(0.5, 0.5, f"Sin casos {outcome}", ha="center", va="center")
        else:
            ax.imshow(plt.imread(path))
        ax.axis("off")
        ax.set_title(outcome, fontsize=12, fontweight="bold")
    fig.suptitle(f"Explicaciones SHAP locales representativas - {label}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97), pad=1.8)
    safe = label.replace("ó", "o")
    fig.savefig(output_dir / f"shap_local_representative_{safe}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"shap_local_representative_{safe}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    for name in ["global", "individual", "tables", "models"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    started = time.time()
    metadata = json.loads((result_dir / "run_metadata.json").read_text(encoding="utf-8"))
    context = dict(metadata["resolved_context"])
    context["features_csv"] = str(args.features_csv.resolve())
    context["labels_csv"] = str(args.labels_csv.resolve())
    base = import_module(args.base_runner.resolve(), "radiomics_base_runner_shap_r07")
    two = import_module(args.two_runner.resolve(), "radiomics_two_runner_shap_r07")

    x, y, patient_ids, feature_cols, data_report = two.build_data(base, context)
    saved_features = pd.read_csv(result_dir / "feature_columns.csv")["feature"].astype(str).tolist()
    if feature_cols != saved_features:
        raise ValueError("Feature columns differ from the final R07 run")
    diagnostics = [name for name in feature_cols if is_diagnostic_feature(name)]
    if diagnostics:
        raise ValueError(f"Diagnostic/procedural columns detected before model fitting: {diagnostics}")
    expected_n = int(metadata["n_oof_patients"])
    if len(patient_ids) != expected_n or len(feature_cols) != 368:
        raise ValueError(f"Unexpected data shape: patients={len(patient_ids)}, features={len(feature_cols)}")

    splits = pd.read_csv(result_dir / "outer_split_assignments.csv")
    splits["patient_id"] = two.canonical_ids(splits["patient_id"].tolist())
    oof_saved = pd.read_csv(result_dir / "oof_predictions.csv")
    oof_saved["patient_id"] = two.canonical_ids(oof_saved["patient_id"].tolist())
    pipeline_template = two.build_pipeline_from_context(base, context)

    folds = sorted(splits["fold"].unique().tolist())
    if folds != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected five folds, found {folds}")
    if args.max_folds is not None:
        folds = folds[:args.max_folds]

    collected = {label: {key: [] for key in ["values", "data", "base", "patient_id", "fold", "true", "pred", "score"]} for label in LABELS}
    validation_rows = []
    max_probability_error = 0.0
    max_additivity_error = 0.0
    retained_by_fold = {}

    for fold in folds:
        test_ids = set(splits.loc[(splits["fold"] == fold) & (splits["split"] == "test"), "patient_id"])
        test_idx = np.asarray([idx for idx, pid in enumerate(patient_ids) if pid in test_ids], dtype=int)
        train_idx = np.asarray([idx for idx, pid in enumerate(patient_ids) if pid not in test_ids], dtype=int)
        if len(test_idx) != len(test_ids):
            raise ValueError(f"Fold {fold}: test IDs do not map one-to-one")

        model = clone(pipeline_template)
        model.fit(x[train_idx], y[train_idx])
        scores = two.validate_probability_scores(two.get_probability_scores(model, x[test_idx]), len(test_idx))
        pred = np.asarray(model.predict(x[test_idx]), dtype=int)

        saved = oof_saved[oof_saved["fold"] == fold].set_index("patient_id")
        order = [str(patient_ids[idx]) for idx in test_idx]
        saved = saved.loc[order]
        saved_scores = saved[["score__Hemorragia", "score__Neumotórax"]].to_numpy(float)
        saved_pred = saved[["pred_default__Hemorragia", "pred_default__Neumotórax"]].to_numpy(int)
        saved_true = saved[["true__Hemorragia", "true__Neumotórax"]].to_numpy(int)
        probability_error = float(np.max(np.abs(scores - saved_scores)))
        max_probability_error = max(max_probability_error, probability_error)
        if probability_error > 1e-12:
            raise ValueError(f"Fold {fold}: probability mismatch {probability_error}")
        if not np.array_equal(pred, saved_pred) or not np.array_equal(y[test_idx], saved_true):
            raise ValueError(f"Fold {fold}: predictions or labels differ from saved OOF")

        imputed, transformed, support, retained_names = transform_r07_with_alignment(model, x[test_idx], feature_cols)
        if any(is_diagnostic_feature(name) for name in retained_names):
            raise ValueError(f"Fold {fold}: diagnostic columns survived preprocessing")
        retained_by_fold[int(fold)] = retained_names
        chain = model.named_steps["classifier"]
        if list(chain.order_) != [0, 1]:
            raise ValueError(f"Unexpected classifier-chain order: {chain.order_}")
        chain_pred = np.zeros((len(test_idx), len(LABELS)), dtype=int)

        for chain_idx, wrapped in enumerate(chain.estimators_):
            label_idx = int(chain.order_[chain_idx])
            label = LABELS[label_idx]
            x_aug = np.hstack([transformed, chain_pred[:, :chain_idx]])
            names_aug = list(feature_cols) + [f"chain_pred_{LABELS[int(i)]}" for i in chain.order_[:chain_idx]]
            data_aug_full = np.hstack([imputed, chain_pred[:, :chain_idx]])
            estimator = wrapped.estimator_
            estimator_prob = estimator.predict_proba(x_aug)[:, 1]
            if np.max(np.abs(estimator_prob - scores[:, label_idx])) > 1e-12:
                raise ValueError(f"Fold {fold} {label}: chain probability mismatch")
            explainer = shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent")
            shap_selected, base_value = positive_class_shap(explainer, x_aug)
            reconstructed = base_value + shap_selected.sum(axis=1)
            model_output = np.asarray(estimator.decision_function(x_aug), dtype=float).reshape(-1)
            additivity_error = float(np.max(np.abs(reconstructed - model_output)))
            max_additivity_error = max(max_additivity_error, additivity_error)
            if additivity_error > 1e-8:
                raise ValueError(f"Fold {fold} {label}: SHAP additivity mismatch {additivity_error}")

            shap_full = np.zeros((len(test_idx), len(names_aug)), dtype=float)
            shap_full[:, np.flatnonzero(support)] = shap_selected[:, :len(retained_names)]
            if chain_idx:
                shap_full[:, len(feature_cols):] = shap_selected[:, len(retained_names):]

            block = collected[label]
            block["values"].append(shap_full)
            block["data"].append(data_aug_full)
            block["base"].extend([base_value] * len(test_idx))
            block["patient_id"].extend(order)
            block["fold"].extend([int(fold)] * len(test_idx))
            block["true"].extend(y[test_idx, label_idx].astype(int).tolist())
            block["pred"].extend(pred[:, label_idx].astype(int).tolist())
            block["score"].extend(scores[:, label_idx].astype(float).tolist())
            block["feature_names"] = names_aug
            chain_pred[:, chain_idx] = estimator.predict(x_aug).astype(int)
            validation_rows.append({"fold": int(fold), "label": label,
                                    "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
                                    "n_features_retained": int(len(retained_names)),
                                    "probability_max_abs_error": probability_error,
                                    "shap_additivity_max_abs_error": additivity_error})
        joblib.dump(model, output_dir / "models" / f"R07_outer_fold_{fold}.joblib")

    representative_rows = []
    classification_rows = []
    for label in LABELS:
        block = collected[label]
        values = np.vstack(block["values"])
        data = np.vstack(block["data"])
        bases = np.asarray(block["base"], dtype=float)
        pids = np.asarray(block["patient_id"], dtype=str)
        fold_arr = np.asarray(block["fold"], dtype=int)
        y_true = np.asarray(block["true"], dtype=int)
        y_pred = np.asarray(block["pred"], dtype=int)
        scores = np.asarray(block["score"], dtype=float)
        feature_names = list(block["feature_names"])
        if any(is_diagnostic_feature(name) for name in feature_names):
            raise ValueError(f"Diagnostic columns detected in final {label} SHAP archive")
        display_names = [clean_feature_name(name) for name in feature_names]
        outcomes = np.asarray([outcome_name(t, p) for t, p in zip(y_true, y_pred)])

        safe = label.replace("ó", "o")
        np.savez_compressed(output_dir / "tables" / f"oof_shap_{safe}.npz",
                            shap_values=values, feature_values=data, base_values=bases,
                            patient_id=pids, fold=fold_arr, y_true=y_true, y_pred=y_pred,
                            score=scores, outcome=outcomes,
                            feature_names=np.asarray(feature_names, dtype=str))
        importance = pd.DataFrame({"feature": feature_names,
                                   "display_feature": display_names,
                                   "mean_abs_shap": np.mean(np.abs(values), axis=0),
                                   "mean_shap": np.mean(values, axis=0)}).sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(output_dir / "tables" / f"global_importance_{safe}.csv", index=False)
        save_global_plots(values, data, display_names, label, output_dir / "global", args.max_display)

        for i in range(len(pids)):
            classification_rows.append({"patient_id": pids[i], "fold": int(fold_arr[i]), "label": label,
                                        "y_true": int(y_true[i]), "y_pred": int(y_pred[i]),
                                        "score": float(scores[i]), "outcome": outcomes[i],
                                        "base_value": float(bases[i])})
        paths = {}
        for outcome in OUTCOMES:
            subset = np.flatnonzero(outcomes == outcome)
            if not len(subset):
                continue
            median = float(np.median(scores[subset]))
            chosen = int(subset[np.argmin(np.abs(scores[subset] - median))])
            representative_rows.append({"label": label, "outcome": outcome,
                                        "patient_id": pids[chosen], "fold": int(fold_arr[chosen]),
                                        "score": float(scores[chosen]),
                                        "selection_rule": "closest_to_outcome_median_score"})
            output_png = output_dir / "individual" / safe / f"{outcome}_representative.png"
            title = f"{label} | fold {fold_arr[chosen]} | real={y_true[chosen]} | prediccion={y_pred[chosen]} | p={scores[chosen]:.3f}"
            save_waterfall(values[chosen], bases[chosen], data[chosen], display_names,
                           title, output_png, args.waterfall_display)
            paths[outcome] = output_png
        save_gallery(paths, label, output_dir / "global")

    class_df = pd.DataFrame(classification_rows)
    class_df.to_csv(output_dir / "tables" / "oof_patient_classification.csv", index=False)
    pd.DataFrame(representative_rows).to_csv(output_dir / "tables" / "representative_cases.csv", index=False)
    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(output_dir / "tables" / "validation_by_fold.csv", index=False)
    counts = class_df.groupby(["label", "outcome"]).size().reset_index(name="n")
    counts.to_csv(output_dir / "tables" / "outcome_counts.csv", index=False)

    report = {
        "status": "completed", "analysis": "Verified OOF SHAP for radiomics R07",
        "modelled_labels": LABELS, "folds_completed": [int(f) for f in folds],
        "all_five_outer_folds_present": folds == [1, 2, 3, 4, 5],
        "n_patients": int(len(patient_ids)), "n_input_features": int(len(feature_cols)),
        "diagnostic_feature_count": 0,
        "diagnostic_feature_guard": "passed",
        "features_retained_by_fold": {str(k): len(v) for k, v in retained_by_fold.items()},
        "max_probability_reproduction_error": max_probability_error,
        "max_shap_additivity_error": max_additivity_error,
        "base_runner_sha256": sha256_file(args.base_runner.resolve()),
        "two_runner_sha256": sha256_file(args.two_runner.resolve()),
        "features_csv_sha256": sha256_file(args.features_csv.resolve()),
        "labels_csv_sha256": sha256_file(args.labels_csv.resolve()),
        "source_metadata_sha256": sha256_file(result_dir / "run_metadata.json"),
        "versions": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                     "shap": shap.__version__, "sklearn": __import__("sklearn").__version__,
                     "matplotlib": matplotlib.__version__},
        "duration_seconds": time.time() - started,
        "data_report": data_report,
        "outcome_counts": counts.to_dict(orient="records")
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
