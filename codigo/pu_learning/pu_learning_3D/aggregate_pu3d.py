#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pu3d_core import binary_metrics

FORMULATIONS = ["supervised", "elkan_noto", "nnpu"]
TARGETS = ["Hemorragia", "Neumotórax"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--labels-csv",
        default=str(
            Path(__file__).resolve().parent
            / "manifest"
            / "labels_205.csv"
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    labels_path = Path(args.labels_csv)

    labels = pd.read_csv(labels_path)
    labels["patient_id"] = labels["patient_id"].astype(str)

    if labels["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient IDs in labels")

    expected_patient_ids = set(labels["patient_id"])
    expected_n = len(expected_patient_ids)

    rows = []
    predictions = []
    errors = []
    complete_tasks = 0

    for target in TARGETS:
        for fold in range(1, 6):
            fold_dir = run_dir / f"{target}__fold_{fold}"
            summary_path = fold_dir / "summary.json"

            if not summary_path.exists():
                errors.append(f"missing {summary_path}")
                continue

            summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )

            if summary.get("status") != "ok":
                errors.append(f"incomplete {summary_path}")
                continue

            task_complete = True

            for formulation in FORMULATIONS:
                pred_path = (
                    fold_dir
                    / f"predictions_{formulation}.csv"
                )

                if not pred_path.exists():
                    errors.append(f"missing {pred_path}")
                    task_complete = False
                    continue

                frame = pd.read_csv(pred_path)
                frame["patient_id"] = (
                    frame["patient_id"].astype(str)
                )

                if len(frame) != summary["n_test_outer"]:
                    errors.append(
                        f"wrong row count {pred_path}: "
                        f"rows={len(frame)} "
                        f"expected={summary['n_test_outer']}"
                    )
                    task_complete = False

                if frame["patient_id"].duplicated().any():
                    errors.append(
                        f"duplicate patients {pred_path}"
                    )
                    task_complete = False

                predictions.append(frame)

                row = {
                    "target": target,
                    "outer_fold": fold,
                    "formulation": formulation,
                }
                row.update(
                    summary["metrics"][formulation]
                )
                row["c_value"] = summary["c_value"]
                row["class_prior"] = summary["class_prior"]
                rows.append(row)

            if task_complete:
                complete_tasks += 1

    pd.DataFrame(rows).to_csv(
        run_dir / "fold_metrics.csv",
        index=False,
    )

    if predictions:
        all_predictions = pd.concat(
            predictions,
            ignore_index=True,
        )
        all_predictions.to_csv(
            run_dir / "all_outer_predictions.csv",
            index=False,
        )

        oof_rows = []

        for (
            target,
            formulation,
        ), frame in all_predictions.groupby(
            ["target", "formulation"]
        ):
            frame = frame.copy()
            frame["patient_id"] = (
                frame["patient_id"].astype(str)
            )

            actual_ids = set(frame["patient_id"])

            if len(frame) != expected_n:
                errors.append(
                    f"incomplete OOF {target} "
                    f"{formulation}: rows={len(frame)} "
                    f"expected={expected_n}"
                                    )
                continue

            if frame["patient_id"].nunique() != expected_n:
                errors.append(
                    f"duplicate OOF patients "
                    f"{target} {formulation}"
                )
                continue

            if actual_ids != expected_patient_ids:
                missing = sorted(
                    expected_patient_ids - actual_ids
                )
                unexpected = sorted(
                    actual_ids - expected_patient_ids
                )
                errors.append(
                    f"wrong OOF cohort {target} "
                    f"{formulation}: "
                    f"missing={missing}, "
                    f"unexpected={unexpected}"
                )
                continue

            metric = binary_metrics(
                frame["y_true"].values,
                frame["y_pred"].values,
                frame["y_score"].values,
            )
            metric.update({
                "target": target,
                "formulation": formulation,
            })
            oof_rows.append(metric)

        pd.DataFrame(oof_rows).to_csv(
            run_dir / "oof_metrics.csv",
            index=False,
        )

    report = {
        "expected_tasks": 10,
        "complete_tasks": complete_tasks,
        "expected_patients": expected_n,
        "errors": errors,
    }

    (
        run_dir / "aggregate_report.json"
    ).write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
