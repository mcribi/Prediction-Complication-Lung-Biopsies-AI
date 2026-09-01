from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from two_point_five_d_core import TARGET_LABELS, validate_oof

REPRESENTATIONS = ("axial5", "triplanar")


def aggregate(run_dir: Path) -> dict:
    reference_oof = Path(
        "/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/val_externa_interna/runs/"
        "resnet18_nestedcv_focused_20260717_105720_resnet18_focused/"
        "resize_cube128__nodule_only_masked_ct__bs8/oof_predictions.csv"
    )
    expected_ids = set(pd.read_csv(reference_oof)["patient_id"].astype(str))
    report = {"run_dir": str(run_dir), "expected_patients": len(expected_ids), "representations": {}, "errors": []}
    from pathlib import Path as _Path
    import sys

    phase_a = _Path(__file__).resolve().parents[1] / "una_validacion_solo"
    if str(phase_a) not in sys.path:
        sys.path.insert(0, str(phase_a))
    import phase_a_serial_common as base

    for representation in REPRESENTATIONS:
        prediction_frames = []
        fold_summaries = []
        for fold in range(1, 6):
            fold_dir = run_dir / representation / f"fold_{fold}"
            prediction_path = fold_dir / "test_predictions_best_epoch.csv"
            metrics_path = fold_dir / "fold_metrics.json"
            if not prediction_path.exists() or not metrics_path.exists():
                report["errors"].append(f"Missing artifacts for {representation} fold {fold}")
                continue
            prediction_frames.append(pd.read_csv(prediction_path))
            fold_summaries.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        if len(prediction_frames) != 5:
            continue
        oof = pd.concat(prediction_frames, ignore_index=True)
        validate_oof(oof, expected_ids=expected_ids, n_folds=5)
        y_true = oof[[f"true_{label}" for label in TARGET_LABELS]].to_numpy()
        y_prob = oof[[f"prob_{label}" for label in TARGET_LABELS]].to_numpy()
        metrics = {key: float(value) for key, value in base.compute_metrics(y_true, y_prob, threshold=0.5).items()}
        oof.to_csv(run_dir / representation / "oof_predictions.csv", index=False)
        pd.DataFrame(fold_summaries).to_csv(run_dir / representation / "fold_summary.csv", index=False)
        (run_dir / representation / "oof_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report["representations"][representation] = {
            "folds": 5,
            "patients": int(oof["patient_id"].nunique()),
            "metrics": metrics,
            "mean_peak_alloc_gb": float(pd.DataFrame(fold_summaries)["peak_alloc_gb"].mean()),
            "mean_epochs": float(pd.DataFrame(fold_summaries)["num_epochs_ran"].mean()),
        }
    if set(report["representations"]) != set(REPRESENTATIONS):
        report["errors"].append("Not all representations were aggregated")
    (run_dir / "aggregate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if report["errors"]:
        raise RuntimeError("; ".join(report["errors"]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
