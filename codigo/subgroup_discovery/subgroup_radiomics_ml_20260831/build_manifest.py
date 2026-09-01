#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd

from radiomics_subgroup_ml import MODEL_NAMES, RULES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    task_id = 0
    for rule_id in RULES:
        for model_name in MODEL_NAMES:
            rows.append(
                {
                    "task_id": task_id,
                    "rule_id": rule_id,
                    "model_name": model_name,
                    "output_dir": str(root / "tasks" / "task_{:03d}__{}__{}".format(task_id, rule_id, model_name)),
                }
            )
            task_id += 1
    manifest = pd.DataFrame(rows)
    manifest.to_csv(root / "manifest.csv", index=False)
    metadata = {
        "status": "ready",
        "expected_tasks": len(manifest),
        "n_rules": len(RULES),
        "n_models": len(MODEL_NAMES),
        "models": MODEL_NAMES,
        "rules": RULES,
        "primary_metric": "delta_tuned_two_outputs_f1_micro",
        "hypothesis": "Subgroup homogeneity may compensate for reduced training size for simple radiomic classifiers.",
        "exploratory_posthoc": True,
    }
    (root / "manifest_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
