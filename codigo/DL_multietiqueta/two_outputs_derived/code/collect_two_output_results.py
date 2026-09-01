#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from two_output_metrics import (
    DEFAULT_THRESHOLD,
    TARGET_LABELS,
    compute_metrics_from_predictions,
    derive_no_complication,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten(prefix: str, payload: dict) -> dict:
    return {f"{prefix}{key}": value for key, value in payload.items()}


def validate_metric_payload(observed: dict, recalculated: dict, name: str) -> None:
    if set(observed) != set(recalculated):
        missing = sorted(set(recalculated) - set(observed))
        extra = sorted(set(observed) - set(recalculated))
        raise ValueError(f"Metric keys differ for {name}: missing={missing}, extra={extra}")
    for key, expected in recalculated.items():
        actual = observed[key]
        if not np.isclose(float(actual), float(expected), rtol=1e-10, atol=1e-12, equal_nan=True):
            raise ValueError(
                f"Metric mismatch for {name}.{key}: stored={actual}, recalculated={expected}"
            )


def collect_config(config_dir: Path, min_mtime_epoch: float | None = None) -> dict:
    fold_summary_path = config_dir / "fold_summary.csv"
    oof_fixed_path = config_dir / "oof_metrics_fixed_0_5.json"
    oof_tuned_path = config_dir / "oof_metrics_tuned.json"
    oof_predictions_path = config_dir / "oof_predictions.csv"
    required = [fold_summary_path, oof_fixed_path, oof_tuned_path, oof_predictions_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete configuration {config_dir}: {missing}")
    if min_mtime_epoch is not None:
        stale = [str(path) for path in required if path.stat().st_mtime < min_mtime_epoch]
        if stale:
            raise ValueError(f"Configuration mixes artifacts from before the current launch: {stale}")

    folds = pd.read_csv(fold_summary_path)
    if len(folds) != 5:
        raise ValueError(f"Expected five folds in {fold_summary_path}, found {len(folds)}")
    if "fold" not in folds.columns:
        raise ValueError(f"Missing fold column in {fold_summary_path}")
    if set(folds["fold"].astype(int)) != {1, 2, 3, 4, 5}:
        raise ValueError(f"Unexpected fold identifiers in {fold_summary_path}")

    fixed_oof = read_json(oof_fixed_path)
    tuned_oof = read_json(oof_tuned_path)
    oof = pd.read_csv(oof_predictions_path, dtype={"patient_id": str})
    if oof["patient_id"].duplicated().any():
        raise ValueError(f"Duplicated OOF patients in {oof_predictions_path}")
    if "fold" not in oof.columns:
        raise ValueError(f"Missing fold column in {oof_predictions_path}")
    normalized_folds = {
        int(str(value).replace("fold_", "")) for value in oof["fold"].unique()
    }
    if normalized_folds != {1, 2, 3, 4, 5}:
        raise ValueError(f"Unexpected OOF folds in {oof_predictions_path}")
    if int(fixed_oof.get("n_samples", -1)) != len(oof) or int(tuned_oof.get("n_samples", -1)) != len(oof):
        raise ValueError(f"OOF metric sample count does not match predictions in {config_dir}")

    required_columns = {"patient_id", "fold"}
    for label in TARGET_LABELS:
        required_columns.update(
            {
                f"true_{label}",
                f"prob_{label}",
                f"pred_fixed_{label}",
                f"pred_tuned_{label}",
                f"threshold_tuned_{label}",
            }
        )
    required_columns.update(
        {
            "true_Sin_complicacion",
            "pred_fixed_Sin_complicacion",
            "pred_tuned_Sin_complicacion",
        }
    )
    missing_columns = sorted(required_columns - set(oof.columns))
    if missing_columns:
        raise ValueError(f"Missing OOF columns in {oof_predictions_path}: {missing_columns}")

    targets = oof[[f"true_{label}" for label in TARGET_LABELS]].to_numpy()
    probabilities = oof[[f"prob_{label}" for label in TARGET_LABELS]].to_numpy()
    fixed_predictions = oof[[f"pred_fixed_{label}" for label in TARGET_LABELS]].to_numpy()
    tuned_predictions = oof[[f"pred_tuned_{label}" for label in TARGET_LABELS]].to_numpy()
    tuned_thresholds = oof[[f"threshold_tuned_{label}" for label in TARGET_LABELS]].to_numpy()

    expected_fixed = (probabilities >= DEFAULT_THRESHOLD).astype(int)
    expected_tuned = (probabilities >= tuned_thresholds).astype(int)
    if not np.array_equal(fixed_predictions, expected_fixed):
        raise ValueError(f"Fixed predictions are inconsistent with probabilities in {config_dir}")
    if not np.array_equal(tuned_predictions, expected_tuned):
        raise ValueError(f"Tuned predictions are inconsistent with thresholds in {config_dir}")
    if np.any((tuned_thresholds < 0) | (tuned_thresholds > 1)):
        raise ValueError(f"Tuned thresholds outside [0, 1] in {config_dir}")

    derived_checks = {
        "true_Sin_complicacion": derive_no_complication(targets),
        "pred_fixed_Sin_complicacion": derive_no_complication(fixed_predictions),
        "pred_tuned_Sin_complicacion": derive_no_complication(tuned_predictions),
    }
    for column, expected in derived_checks.items():
        if not np.array_equal(oof[column].to_numpy().astype(int), expected):
            raise ValueError(f"Derived state {column} is inconsistent in {config_dir}")

    recalculated_fixed, _ = compute_metrics_from_predictions(targets, probabilities, fixed_predictions)
    recalculated_tuned, _ = compute_metrics_from_predictions(targets, probabilities, tuned_predictions)
    validate_metric_payload(fixed_oof, recalculated_fixed, "fixed_oof")
    validate_metric_payload(tuned_oof, recalculated_tuned, "tuned_oof")

    threshold_summary_columns = {
        "Hemorragia": "threshold_Hemorragia",
        "Neumotórax": "threshold_Neumotorax",
    }
    normalized_oof_folds = oof["fold"].astype(str).str.replace("fold_", "", regex=False).astype(int)
    for label, summary_column in threshold_summary_columns.items():
        if summary_column not in folds.columns:
            raise ValueError(f"Missing {summary_column} in {fold_summary_path}")
        prediction_column = f"threshold_tuned_{label}"
        for fold in range(1, 6):
            values = oof.loc[normalized_oof_folds == fold, prediction_column].unique()
            if len(values) != 1:
                raise ValueError(f"Fold {fold} has multiple {prediction_column} values")
            stored = float(folds.loc[folds["fold"].astype(int) == fold, summary_column].iloc[0])
            if not np.isclose(float(values[0]), stored, rtol=0.0, atol=1e-12):
                raise ValueError(f"Threshold mismatch for {label}, fold {fold}")

    row = {
        "config_dir": str(config_dir),
        "num_folds_completed": int(len(folds)),
        "all_folds_present": True,
    }
    row.update(flatten("fixed_oof_", fixed_oof))
    row.update(flatten("tuned_oof_", tuned_oof))
    metric_columns = [
        column for column in folds.columns
        if column.startswith(("fixed_", "tuned_"))
        and pd.api.types.is_numeric_dtype(folds[column])
    ]
    for column in metric_columns:
        row[f"mean_{column}"] = float(folds[column].mean())
        row[f"std_ddof1_{column}"] = float(folds[column].std(ddof=1))
    for column in ["threshold_Hemorragia", "threshold_Neumotorax"]:
        if column in folds:
            row[f"mean_{column}"] = float(folds[column].mean())
            row[f"std_ddof1_{column}"] = float(folds[column].std(ddof=1))
    return row


def discover_configs(root: Path, min_mtime_epoch: float | None = None) -> list[Path]:
    candidates: set[Path] = set()
    for filename in (
        "fold_summary.csv",
        "oof_predictions.csv",
        "oof_metrics_fixed_0_5.json",
        "oof_metrics_tuned.json",
        "error.txt",
    ):
        for path in root.rglob(filename):
            if min_mtime_epoch is not None and path.stat().st_mtime < min_mtime_epoch:
                continue
            config_dir = path.parent
            if config_dir.name.startswith("fold_"):
                config_dir = config_dir.parent
            candidates.add(config_dir)
    return sorted(candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-mtime-epoch", type=float)
    args = parser.parse_args()

    configs = discover_configs(args.root, args.min_mtime_epoch)
    if not configs:
        raise FileNotFoundError(f"No result configurations found under {args.root}")
    rows = []
    errors = []
    for config_dir in configs:
        try:
            rows.append(collect_config(config_dir, args.min_mtime_epoch))
        except Exception as exc:
            errors.append({"config_dir": str(config_dir), "error": str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    error_path = args.output.with_name(f"{args.output.stem}_incomplete.csv")
    pd.DataFrame(errors).to_csv(error_path, index=False)
    report = {
        "root": str(args.root),
        "min_mtime_epoch": args.min_mtime_epoch,
        "complete_configurations": len(rows),
        "incomplete_configurations": len(errors),
        "table": str(args.output),
        "incomplete_table": str(error_path),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors or not rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
