#!/usr/bin/env python3

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = [
    "average_precision",
    "roc_auc",
    "sensitivity",
    "specificity",
    "f1",
    "precision",
    "balanced_accuracy",
    "gmean",
    "mcc",
    "brier_score",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root)
    manifest = pd.read_csv(run_root / "manifest" / "manifest.csv")
    aggregate_root = run_root / "aggregate"
    aggregate_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    oof_tables = []
    errors = []
    for row in manifest.itertuples(index=False):
        task_dir = run_root / "tasks" / ("task_%03d" % int(row.task_id))
        summary_path = task_dir / "summary.json"
        metrics_path = task_dir / "oof_metrics.csv"
        if not summary_path.exists() or not metrics_path.exists():
            errors.append({"task_id": int(row.task_id), "error": "missing summary or metrics"})
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "completed" or summary.get("n_outer_folds") != 5:
            errors.append({"task_id": int(row.task_id), "error": "task not complete"})
            continue
        metrics = pd.read_csv(metrics_path)
        if len(metrics) != 6 or not (metrics["n_outer_folds"] == 5).all():
            errors.append({"task_id": int(row.task_id), "error": "invalid OOF coverage"})
            continue
        summaries.append(summary)
        oof_tables.append(metrics)

    if errors or len(summaries) != len(manifest):
        report = {
            "status": "incomplete",
            "expected_tasks": int(len(manifest)),
            "completed_tasks": int(len(summaries)),
            "errors": errors,
        }
        (aggregate_root / "completeness.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
        raise SystemExit(json.dumps(report, ensure_ascii=False))

    all_metrics = pd.concat(oof_tables, ignore_index=True)
    all_metrics.to_csv(aggregate_root / "all_task_oof_metrics.csv", index=False)
    grouping = ["target", "feature_set", "hide_fraction", "method"]
    rows = []
    for keys, group in all_metrics.groupby(grouping, sort=True):
        item = dict(zip(grouping, keys))
        item["n_seeds"] = int(group["concealment_seed"].nunique())
        for metric in METRICS:
            item[metric + "_mean"] = float(group[metric].mean())
            item[metric + "_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(item)
    summary_table = pd.DataFrame(rows)
    summary_table.to_csv(aggregate_root / "summary_mean_std.csv", index=False)

    key = ["task_id", "target", "feature_set", "hide_fraction", "concealment_seed"]
    pn = all_metrics[all_metrics["method"] == "PN"].set_index(key)
    pu = all_metrics[all_metrics["method"] == "Bagging_PU"].set_index(key)
    if set(pn.index) != set(pu.index):
        raise ValueError("PN and Bagging PU tasks are not paired")
    paired = pn.reset_index()[key].copy()
    for metric in METRICS:
        paired["pn_" + metric] = pn.loc[paired.set_index(key).index, metric].to_numpy()
        paired["pu_" + metric] = pu.loc[paired.set_index(key).index, metric].to_numpy()
        paired["delta_" + metric] = paired["pu_" + metric] - paired["pn_" + metric]
    paired.to_csv(aggregate_root / "paired_task_deltas.csv", index=False)

    delta_rows = []
    delta_grouping = ["target", "feature_set", "hide_fraction"]
    for keys, group in paired.groupby(delta_grouping, sort=True):
        item = dict(zip(delta_grouping, keys))
        item["n_seeds"] = int(group["concealment_seed"].nunique())
        for metric in METRICS:
            values = group["delta_" + metric]
            item["delta_" + metric + "_mean"] = float(values.mean())
            item["delta_" + metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            if metric != "brier_score":
                item["pu_wins_" + metric] = int((values > 0).sum())
            else:
                item["pu_wins_" + metric] = int((values < 0).sum())
        delta_rows.append(item)
    pd.DataFrame(delta_rows).to_csv(
        aggregate_root / "paired_delta_summary.csv", index=False
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for target in sorted(summary_table["target"].unique()):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
            for axis, feature_set in zip(
                axes, sorted(summary_table["feature_set"].unique())
            ):
                subset = summary_table[
                    (summary_table["target"] == target)
                    & (summary_table["feature_set"] == feature_set)
                ]
                for method in ["PN", "Bagging_PU"]:
                    current = subset[subset["method"] == method].sort_values(
                        "hide_fraction"
                    )
                    axis.errorbar(
                        current["hide_fraction"] * 100,
                        current["average_precision_mean"],
                        yerr=current["average_precision_std"],
                        marker="o",
                        capsize=3,
                        label=method,
                    )
                axis.set_title(feature_set)
                axis.set_xlabel("Hidden positives (%)")
                axis.grid(alpha=0.3)
            axes[0].set_ylabel("OOF average precision")
            axes[-1].legend()
            fig.suptitle(target)
            fig.tight_layout()
            fig.savefig(aggregate_root / ("average_precision_%s.png" % target), dpi=180)
            plt.close(fig)
    except Exception as error:
        (aggregate_root / "plot_warning.txt").write_text(str(error))

    report = {
        "status": "completed",
        "created_at": datetime.now().isoformat(),
        "expected_tasks": int(len(manifest)),
        "completed_tasks": int(len(summaries)),
        "all_tasks_have_five_outer_folds": True,
        "paired_methods": True,
        "ground_truth_evaluation": "unaltered outer-test labels",
        "errors": [],
    }
    (aggregate_root / "completeness.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
