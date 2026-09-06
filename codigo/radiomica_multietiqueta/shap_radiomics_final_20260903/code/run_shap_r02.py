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
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone

LABELS = ["Hemorragia", "Neumotórax"]
OUTCOMES = ["TP", "TN", "FP", "FN"]


def parse_args():
    parser = argparse.ArgumentParser(description="OOF SHAP analysis for final radiomics R02")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--skip-local", action="store_true")
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


def clean_feature_name(name: str) -> str:
    value = str(name)
    if value.startswith("nodule_"):
        value = value[len("nodule_"):]
    value = value.replace("diagnostics_", "diag_")
    value = value.replace("original_", "orig_")
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


def save_global_plots(values, data, feature_names, label, out_dir, max_display):
    safe = label.replace("ó", "o")
    plt.figure()
    shap.summary_plot(
        values,
        data,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=(10, 8),
    )
    plt.title(f"SHAP global OOF - {label}")
    plt.tight_layout()
    plt.savefig(out_dir / f"beeswarm_{safe}.png", dpi=220, bbox_inches="tight")
    plt.savefig(out_dir / f"beeswarm_{safe}.pdf", bbox_inches="tight")
    plt.close()

    importance = np.mean(np.abs(values), axis=0)
    order = np.argsort(importance)[::-1][:max_display][::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(np.asarray(feature_names)[order], importance[order], color="#2878B5")
    ax.set_xlabel("Media de |valor SHAP|")
    ax.set_title(f"Importancia global OOF - {label}")
    fig.tight_layout()
    fig.savefig(out_dir / f"importance_{safe}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"importance_{safe}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_one_waterfall(values, base, data, feature_names, title, output_png, max_display):
    explanation = shap.Explanation(
        values=np.asarray(values, dtype=float),
        base_values=float(base),
        data=np.asarray(data, dtype=float),
        feature_names=list(feature_names),
    )
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    plt.title(title, fontsize=10)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_gallery(image_paths, output_base: Path, panels_per_page: int = 4):
    if not image_paths:
        return 0
    pages = 0
    output_base.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_base.with_suffix(".pdf")) as pdf:
        for start in range(0, len(image_paths), panels_per_page):
            subset = image_paths[start:start + panels_per_page]
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.ravel()
            for ax, path in zip(axes, subset):
                ax.imshow(plt.imread(path))
                ax.axis("off")
            for ax in axes[len(subset):]:
                ax.axis("off")
            fig.tight_layout()
            page = start // panels_per_page + 1
            fig.savefig(output_base.parent / f"{output_base.name}_page_{page:02d}.png", dpi=180, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages += 1
    return pages


def main():
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "global").mkdir()
    (output_dir / "local").mkdir()
    (output_dir / "mosaics").mkdir()
    (output_dir / "models").mkdir()
    (output_dir / "tables").mkdir()

    started = time.time()
    metadata = json.loads((result_dir / "run_metadata.json").read_text(encoding="utf-8"))
    context = metadata["resolved_context"]
    base_path = Path(metadata["base_runner_path"])
    two_path = base_path.parent / "two_outputs_derived" / "run_radiomics_two_outputs_thresholds.py"
    base = import_module(base_path, "radiomics_base_runner_shap")
    two = import_module(two_path, "radiomics_two_runner_shap")

    x, y, patient_ids, feature_cols, data_report = two.build_data(base, context)
    saved_features = pd.read_csv(result_dir / "feature_columns.csv")["feature"].astype(str).tolist()
    if feature_cols != saved_features:
        raise ValueError("Feature columns differ from the final R02 run")
    if len(patient_ids) != 195 or len(feature_cols) != 120:
        raise ValueError(f"Unexpected data shape: patients={len(patient_ids)}, features={len(feature_cols)}")

    splits = pd.read_csv(result_dir / "outer_split_assignments.csv")
    splits["patient_id"] = two.canonical_ids(splits["patient_id"].tolist())
    oof_saved = pd.read_csv(result_dir / "oof_predictions.csv")
    oof_saved["patient_id"] = two.canonical_ids(oof_saved["patient_id"].tolist())
    thresholds = pd.read_csv(result_dir / "selected_thresholds.csv")
    pipeline_template = two.build_pipeline_from_context(base, context)

    folds = sorted(splits["fold"].unique().tolist())
    if folds != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected five folds, found {folds}")
    if args.max_folds is not None:
        folds = folds[:args.max_folds]

    collected = {
        label: {"values": [], "data": [], "base": [], "patient_id": [], "fold": [], "true": [], "pred": [], "score": []}
        for label in LABELS
    }
    validation_rows = []
    max_probability_error = 0.0
    max_additivity_error = 0.0

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

        transformed = x[test_idx]
        for _name, transformer in model.steps[:-1]:
            transformed = transformer.transform(transformed)
        chain = model.named_steps["classifier"]
        if list(chain.order_) != [0, 1]:
            raise ValueError(f"Unexpected classifier-chain order: {chain.order_}")
        chain_pred = np.zeros((len(test_idx), len(LABELS)), dtype=int)

        for chain_idx, wrapped in enumerate(chain.estimators_):
            label_idx = int(chain.order_[chain_idx])
            label = LABELS[label_idx]
            x_aug = np.hstack([transformed, chain_pred[:, :chain_idx]])
            names_aug = list(feature_cols) + [f"chain_pred_{LABELS[int(i)]}" for i in chain.order_[:chain_idx]]
            estimator = wrapped.estimator_
            estimator_prob = estimator.predict_proba(x_aug)[:, 1]
            if np.max(np.abs(estimator_prob - scores[:, label_idx])) > 1e-12:
                raise ValueError(f"Fold {fold} {label}: chain probability mismatch")
            explainer = shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent")
            shap_values, base_value = positive_class_shap(explainer, x_aug)
            reconstructed = base_value + shap_values.sum(axis=1)
            additivity_error = float(np.max(np.abs(reconstructed - estimator_prob)))
            max_additivity_error = max(max_additivity_error, additivity_error)
            if additivity_error > 1e-8:
                raise ValueError(f"Fold {fold} {label}: SHAP additivity mismatch {additivity_error}")

            block = collected[label]
            block["values"].append(shap_values)
            block["data"].append(x_aug)
            block["base"].extend([base_value] * len(test_idx))
            block["patient_id"].extend(order)
            block["fold"].extend([int(fold)] * len(test_idx))
            block["true"].extend(y[test_idx, label_idx].astype(int).tolist())
            block["pred"].extend(pred[:, label_idx].astype(int).tolist())
            block["score"].extend(scores[:, label_idx].astype(float).tolist())
            block["feature_names"] = names_aug

            chain_pred[:, chain_idx] = estimator.predict(x_aug).astype(int)
            validation_rows.append({
                "fold": int(fold),
                "label": label,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "probability_max_abs_error": probability_error,
                "shap_additivity_max_abs_error": additivity_error,
            })

        joblib.dump(model, output_dir / "models" / f"R02_outer_fold_{fold}.joblib")

    all_local_rows = []
    representative_rows = []
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
        display_names = [clean_feature_name(name) for name in feature_names]
        outcomes = np.asarray([outcome_name(t, p) for t, p in zip(y_true, y_pred)])

        np.savez_compressed(
            output_dir / "tables" / f"oof_shap_{label.replace('ó','o')}.npz",
            shap_values=values,
            feature_values=data,
            base_values=bases,
            patient_id=pids,
            fold=fold_arr,
            y_true=y_true,
            y_pred=y_pred,
            score=scores,
            outcome=outcomes,
            feature_names=np.asarray(feature_names, dtype=str),
        )
        importance = pd.DataFrame({
            "feature": feature_names,
            "display_feature": display_names,
            "mean_abs_shap": np.mean(np.abs(values), axis=0),
            "mean_shap": np.mean(values, axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(output_dir / "tables" / f"global_importance_{label.replace('ó','o')}.csv", index=False)
        save_global_plots(values, data, display_names, label, output_dir / "global", args.max_display)

        local_rows = pd.DataFrame({
            "patient_id": pids,
            "fold": fold_arr,
            "label": label,
            "y_true": y_true,
            "y_pred": y_pred,
            "score": scores,
            "outcome": outcomes,
            "base_value": bases,
        })
        all_local_rows.append(local_rows)

        if not args.skip_local:
            paths_by_outcome = {name: [] for name in OUTCOMES}
            for idx in range(len(pids)):
                outcome = outcomes[idx]
                safe_pid = re.sub(r"[^A-Za-z0-9_.-]", "_", pids[idx])
                output_png = output_dir / "local" / label.replace("ó", "o") / outcome / f"{safe_pid}.png"
                title = f"{label} | {pids[idx]} | fold {fold_arr[idx]} | real={y_true[idx]} pred={y_pred[idx]} p={scores[idx]:.3f}"
                save_one_waterfall(values[idx], bases[idx], data[idx], display_names, title, output_png, args.waterfall_display)
                paths_by_outcome[outcome].append(output_png)

            selected_paths = []
            for outcome in OUTCOMES:
                subset_idx = np.flatnonzero(outcomes == outcome)
                if len(subset_idx):
                    median = float(np.median(scores[subset_idx]))
                    chosen = int(subset_idx[np.argmin(np.abs(scores[subset_idx] - median))])
                    selected_paths.append(paths_by_outcome[outcome][list(subset_idx).index(chosen)])
                    representative_rows.append({
                        "label": label,
                        "outcome": outcome,
                        "patient_id": pids[chosen],
                        "fold": int(fold_arr[chosen]),
                        "score": float(scores[chosen]),
                        "selection_rule": "closest_to_outcome_median_score",
                    })
                safe_label = label.replace("ó", "o")
                save_gallery(paths_by_outcome[outcome], output_dir / "mosaics" / f"waterfalls_{safe_label}_{outcome}")
            save_gallery(selected_paths, output_dir / "mosaics" / f"representative_all_outcomes_{label.replace('ó','o')}")

    local_df = pd.concat(all_local_rows, ignore_index=True)
    local_df.to_csv(output_dir / "tables" / "oof_patient_classification.csv", index=False)
    pd.DataFrame(representative_rows).to_csv(output_dir / "tables" / "representative_cases.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(output_dir / "tables" / "validation_by_fold.csv", index=False)
    counts = local_df.groupby(["label", "outcome"]).size().reset_index(name="n")
    counts.to_csv(output_dir / "tables" / "outcome_counts.csv", index=False)

    report = {
        "status": "completed",
        "analysis": "OOF SHAP for final radiomics R02",
        "result_dir": str(result_dir),
        "output_dir": str(output_dir),
        "decision_for_outcome_groups": "default estimator prediction",
        "modelled_labels": LABELS,
        "folds_completed": [int(fold) for fold in folds],
        "all_five_outer_folds_present": folds == [1, 2, 3, 4, 5],
        "n_patients": int(len(patient_ids)),
        "n_features": int(len(feature_cols)),
        "classifier": context["classifier"],
        "multilabel_strategy": context["multilabel_strategy"],
        "classifier_params": context["classifier_params"],
        "max_probability_reproduction_error": max_probability_error,
        "max_shap_additivity_error": max_additivity_error,
        "base_runner_sha256": sha256_file(base_path),
        "two_outputs_runner_sha256": sha256_file(two_path),
        "source_metadata_sha256": sha256_file(result_dir / "run_metadata.json"),
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "shap": shap.__version__,
            "sklearn": __import__("sklearn").__version__,
            "matplotlib": matplotlib.__version__,
        },
        "duration_seconds": time.time() - started,
        "data_report": data_report,
        "outcome_counts": counts.to_dict(orient="records"),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
