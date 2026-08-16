#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate expanded PU grid results.")
    p.add_argument("--results-root", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.results_root).expanduser().resolve()
    rows = []
    failed = []
    for p in sorted(root.glob("task_*/summary.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
            row = {k: d.get(k) for k in ["task_id", "status", "formulation", "feature_set", "target", "model", "n_patients", "n_features", "n_folds_executed", "support", "predicted_positive", "f1", "precision", "recall", "balanced_accuracy", "roc_auc", "average_precision", "elapsed_seconds", "completed_at"]}
            row["params_json"] = json.dumps(d.get("params", {}), ensure_ascii=False, sort_keys=True)
            row["result_dir"] = str(p.parent)
            rows.append(row)
        except Exception as e:
            failed.append({"path": str(p), "error": str(e)})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["f1", "average_precision", "roc_auc"], ascending=[False, False, False])
        df.to_csv(root / "aggregate_metrics.csv", index=False)
        best = df.groupby(["formulation", "feature_set", "target"], dropna=False).head(5)
        best.to_csv(root / "top5_by_problem.csv", index=False)
    summary = {"results_root": str(root), "n_summary_files": len(rows), "n_failed_reads": len(failed)}
    with (root / "aggregate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if len(df):
        print(df.head(20)[["formulation", "feature_set", "target", "model", "f1", "average_precision", "roc_auc", "elapsed_seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
