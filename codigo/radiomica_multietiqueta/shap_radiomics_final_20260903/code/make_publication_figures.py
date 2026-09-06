#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

LABELS = ["Hemorragia", "Neumotórax"]
OUTCOMES = ["TP", "TN", "FP", "FN"]


def parse_args():
    parser = argparse.ArgumentParser(description="Create publication-ready SHAP figures from verified OOF arrays")
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-display", type=int, default=20)
    parser.add_argument("--waterfall-display", type=int, default=12)
    return parser.parse_args()


def safe_label(label: str) -> str:
    return label.replace("ó", "o")


def clean_feature_name(name: str) -> str:
    value = str(name)
    if value.startswith("nodule_"):
        value = value[len("nodule_"):]
    value = value.replace("diagnostics_", "diag_")
    value = value.replace("original_", "orig_")
    value = value.replace("chain_pred_Hemorragia", "predicción previa de hemorragia")
    return value


def save_beeswarm(values, data, names, label, out_dir, max_display):
    plt.figure()
    shap.summary_plot(
        values,
        data,
        feature_names=names,
        max_display=max_display,
        show=False,
        plot_size=(11, 8.5),
    )
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xlabel("Valor SHAP (impacto en la salida del modelo)")
    ax.set_title(f"Distribución global de valores SHAP OOF — {label}", pad=12)
    if len(fig.axes) > 1:
        colorbar_ax = fig.axes[-1]
        colorbar_ax.set_ylabel("Valor de la característica")
        ticks = colorbar_ax.get_yticks()
        if len(ticks) >= 2:
            colorbar_ax.set_yticklabels(["Bajo", "Alto"])
    fig.tight_layout()
    stem = out_dir / f"shap_beeswarm_{safe_label(label)}"
    fig.savefig(stem.with_suffix(".png"), dpi=250, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def save_importance(values, names, label, out_dir, max_display):
    importance = np.mean(np.abs(values), axis=0)
    order = np.argsort(importance)[::-1][:max_display][::-1]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.barh(np.asarray(names)[order], importance[order], color="#2878B5")
    ax.set_xlabel("Media del valor SHAP absoluto")
    ax.set_title(f"Importancia global SHAP OOF — {label}", pad=12)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    stem = out_dir / f"shap_importance_{safe_label(label)}"
    fig.savefig(stem.with_suffix(".png"), dpi=250, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def save_waterfall(values, base, data, names, title, output_png, max_display):
    explanation = shap.Explanation(
        values=np.asarray(values, dtype=float),
        base_values=float(base),
        data=np.asarray(data, dtype=float),
        feature_names=list(names),
    )
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(11, 6.5)
    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    span = max(xmax - xmin, 1e-6)
    ax.set_xlim(xmin - 0.03 * span, xmax + 0.13 * span)
    ax.set_title(title, fontsize=10, pad=12)
    fig.tight_layout()
    fig.savefig(output_png, dpi=220, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def save_gallery(paths, label, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5))
    axes = axes.ravel()
    for ax, outcome in zip(axes, OUTCOMES):
        path = paths.get(outcome)
        if path is None:
            ax.text(0.5, 0.5, f"Sin casos {outcome}", ha="center", va="center")
        else:
            ax.imshow(plt.imread(path))
        ax.axis("off")
        ax.set_title(outcome, fontsize=12, fontweight="bold", pad=4)
    fig.suptitle(f"Explicaciones SHAP locales representativas — {label}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97), pad=1.8)
    stem = out_dir / f"shap_local_representative_{safe_label(label)}"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main():
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    representatives = pd.read_csv(analysis_dir / "tables" / "representative_cases.csv")

    for label in LABELS:
        archive = np.load(analysis_dir / "tables" / f"oof_shap_{safe_label(label)}.npz")
        values = archive["shap_values"]
        data = archive["feature_values"]
        bases = archive["base_values"]
        pids = archive["patient_id"].astype(str)
        folds = archive["fold"].astype(int)
        y_true = archive["y_true"].astype(int)
        y_pred = archive["y_pred"].astype(int)
        scores = archive["score"].astype(float)
        names = [clean_feature_name(name) for name in archive["feature_names"].astype(str)]

        save_beeswarm(values, data, names, label, output_dir, args.max_display)
        save_importance(values, names, label, output_dir, args.max_display)

        paths = {}
        selected = representatives[representatives["label"] == label]
        for _, row in selected.iterrows():
            outcome = str(row["outcome"])
            match = np.flatnonzero((pids == str(row["patient_id"])) & (folds == int(row["fold"])))
            if len(match) != 1:
                raise ValueError(f"Representative case does not map once: {label} {row['patient_id']}")
            idx = int(match[0])
            safe_pid = re.sub(r"[^A-Za-z0-9_.-]", "_", pids[idx])
            output_png = output_dir / "individual" / safe_label(label) / f"{outcome}_{safe_pid}.png"
            output_png.parent.mkdir(parents=True, exist_ok=True)
            title = f"{label} | ID {pids[idx]} | fold {folds[idx]} | real={y_true[idx]} | predicción={y_pred[idx]} | p={scores[idx]:.3f}"
            save_waterfall(values[idx], bases[idx], data[idx], names, title, output_png, args.waterfall_display)
            paths[outcome] = output_png
        save_gallery(paths, label, output_dir)

    generated = sorted(path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file())
    (output_dir / "FILES.txt").write_text("\n".join(generated) + "\n", encoding="utf-8")
    print(f"Generated {len(generated)} files in {output_dir}")


if __name__ == "__main__":
    main()
