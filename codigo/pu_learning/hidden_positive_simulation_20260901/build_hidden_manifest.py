#!/usr/bin/env python3

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from hidden_positive_simulation import build_task_rows

TARGETS = ["Hemorragia", "Neumotórax"]
FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4]
CONCEALMENT_SEEDS = list(range(1001, 1011))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--folds-csv", required=True)
    parser.add_argument("--cohort-ids-csv", required=True)
    parser.add_argument("--clinical-geometry-csv", required=True)
    parser.add_argument("--radiomics-csv", required=True)
    parser.add_argument("--combined-csv", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    manifest_dir = run_root / "manifest"
    tasks_root = run_root / "tasks"
    logs_root = run_root / "slurm_outputs"
    aggregate_root = run_root / "aggregate"
    for path in [manifest_dir, tasks_root, logs_root, aggregate_root]:
        path.mkdir(parents=True, exist_ok=True)

    feature_paths = {
        "clinical_geometry": str(Path(args.clinical_geometry_csv).resolve()),
        "radiomics": str(Path(args.radiomics_csv).resolve()),
        "clinical_geometry_radiomics": str(Path(args.combined_csv).resolve()),
    }
    rows = build_task_rows(TARGETS, FRACTIONS, CONCEALMENT_SEEDS)
    for row in rows:
        row.update(
            {
                "feature_paths_json": json.dumps(feature_paths, sort_keys=True),
                "labels_csv": str(Path(args.labels_csv).resolve()),
                "folds_csv": str(Path(args.folds_csv).resolve()),
                "cohort_ids_csv": str(Path(args.cohort_ids_csv).resolve()),
                "output_root": str(tasks_root.resolve()),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_path = manifest_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    inputs = {
        "labels_csv": args.labels_csv,
        "folds_csv": args.folds_csv,
        "cohort_ids_csv": args.cohort_ids_csv,
        **feature_paths,
    }
    metadata = {
        "created_at": datetime.now().isoformat(),
        "objective": "Compare PN and Bagging PU under controlled hidden-positive training labels",
        "hypothesis": "Bagging PU degrades less than PN as hidden-positive fraction increases",
        "targets": TARGETS,
        "feature_sets": list(feature_paths),
        "hide_fractions": FRACTIONS,
        "concealment_seeds": CONCEALMENT_SEEDS,
        "n_tasks": len(rows),
        "n_outer_folds": 5,
        "methods": ["PN", "Bagging_PU"],
        "base_classifier": "LogisticRegression",
        "shared_c_grid": [0.01, 0.1, 1.0, 10.0],
        "selection_metric": "observed-label inner-OOF average precision",
        "threshold_rule": "observed-label inner-OOF max G-mean, then MCC, F1, closeness to 0.5",
        "evaluation": "unaltered outer-test ground truth",
        "input_sha256": {key: sha256(value) for key, value in inputs.items()},
        "input_paths": inputs,
        "limitations": "Artificial hiding tests algorithmic robustness and does not prove real unrecorded complications",
    }
    (manifest_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    print(json.dumps({"manifest": str(manifest_path), "n_tasks": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
