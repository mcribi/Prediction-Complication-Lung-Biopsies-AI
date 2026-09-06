from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MODELED_OUTPUTS = {"Hemorragia", "Neumotórax"}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify(output_dir: Path, expected_configs: int = 5, expected_cases: int = 20) -> dict:
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_rows(output_dir / "gradcam_manifest.csv")
    selection = json.loads(
        (output_dir / "selected_configs_and_cases.json").read_text(encoding="utf-8")
    )
    errors = []
    if summary.get("modeled_outputs") != ["Hemorragia", "Neumotórax"]:
        errors.append("Unexpected modeled outputs")
    if summary.get("derived_state_has_independent_gradcam") is not False:
        errors.append("Derived state was marked as an independent Grad-CAM target")
    if len(selection) != expected_configs:
        errors.append(f"Expected {expected_configs} configs, found {len(selection)}")
    if len(rows) != expected_cases:
        errors.append(f"Expected {expected_cases} cases, found {len(rows)}")

    max_probability_error = 0.0
    for row in rows:
        if row["selected_for_label"] not in MODELED_OUTPUTS:
            errors.append(f"Invalid selected output: {row['selected_for_label']}")
        config_dir = Path(row["config_dir"])
        fold_dir = config_dir / row["fold"]
        patient_id = row["patient_id"]
        if not (fold_dir / "best_model.pt").exists():
            errors.append(f"Missing checkpoint: {fold_dir}")
        fold_rows = read_rows(fold_dir / "test_predictions_best_epoch.csv")
        fold_ids = {item["patient_id"] for item in fold_rows}
        if patient_id not in fold_ids:
            errors.append(f"Patient {patient_id} is not in {fold_dir.name} test_outer")
        for artifact_key in ("png_path", "npz_path"):
            artifact = Path(row[artifact_key])
            if not artifact.exists() or artifact.stat().st_size == 0:
                errors.append(f"Missing or empty artifact: {artifact}")
        max_probability_error = max(
            max_probability_error,
            float(row["probability_check_max_abs_error"]),
        )

    if max_probability_error > 1e-3:
        errors.append(f"Probability error exceeds tolerance: {max_probability_error}")
    report = {
        "status": "ok" if not errors else "failed",
        "output_dir": str(output_dir),
        "configs": len(selection),
        "cases": len(rows),
        "max_probability_abs_error": max_probability_error,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-configs", type=int, default=5)
    parser.add_argument("--expected-cases", type=int, default=20)
    args = parser.parse_args()
    verify(args.output_dir, args.expected_configs, args.expected_cases)


if __name__ == "__main__":
    main()
