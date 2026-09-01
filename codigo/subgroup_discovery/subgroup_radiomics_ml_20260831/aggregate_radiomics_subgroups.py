#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def collect_completed(manifest):
    incomplete = []
    rows = []
    for _, task in manifest.sort_values("task_id").iterrows():
        output = Path(task["output_dir"])
        completion_path = output / "COMPLETED.json"
        metrics_path = output / "oof_metrics.json"
        if not completion_path.exists() or not metrics_path.exists():
            incomplete.append(int(task["task_id"]))
            continue
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") != "ok" or completion.get("num_folds_completed") != 5 or completion.get("all_folds_present") is not True:
            incomplete.append(int(task["task_id"]))
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "task_id": int(task["task_id"]),
            "rule_id": str(task["rule_id"]),
            "model_name": str(task["model_name"]),
            "n_subgroup": int(completion["n_subgroup"]),
            "num_folds_completed": 5,
            "all_folds_present": True,
        }
        for variant, values in metrics.items():
            for metric, value in values.items():
                row["{}_{}".format(variant, metric)] = value
        for decision in ["fixed", "tuned"]:
            general = metrics.get("general_{}".format(decision), {})
            specialized = metrics.get("specialized_{}".format(decision), {})
            for metric in sorted(set(general) & set(specialized)):
                try:
                    row["delta_{}_{}".format(decision, metric)] = float(specialized[metric]) - float(general[metric])
                except (TypeError, ValueError):
                    pass
        rows.append(row)
    if incomplete:
        raise ValueError("Incomplete tasks: {}".format(incomplete))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    summary = collect_completed(manifest)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary_all_tasks.csv", index=False)
    primary = "delta_tuned_two_outputs_f1_micro"
    ranking = summary.sort_values(primary, ascending=False).reset_index(drop=True)
    ranking.to_csv(output / "ranking_by_delta_tuned_f1_micro.csv", index=False)
    report = {
        "status": "ok",
        "expected_tasks": int(len(manifest)),
        "completed_tasks": int(len(summary)),
        "all_tasks_complete": len(summary) == len(manifest),
        "all_folds_verified": bool(summary["all_folds_present"].all()),
        "models": sorted(summary["model_name"].unique().tolist()),
        "rules": sorted(summary["rule_id"].unique().tolist()),
    }
    (output / "final_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
