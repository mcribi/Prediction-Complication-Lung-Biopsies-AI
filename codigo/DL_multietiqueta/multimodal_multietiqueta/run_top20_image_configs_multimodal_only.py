#!/usr/bin/env python3
"""Select the best image-only configurations and run only the multimodal branch.

This script is intentionally retrospective: it ranks completed image-only nested-CV
configs by outer-test OOF F1 micro and reuses their splits/configuration for a
multimodal-only experiment. It does not run image-only or clinical-only controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
DL_ROOT = MODULE_DIR.parent
IMAGE_RUNS_ROOT = DL_ROOT / "val_externa_interna" / "runs"
MULTIMODAL_RUNS_ROOT = MODULE_DIR / "runs"
MANIFEST_ROOT = MODULE_DIR / "manifests_top20_multimodal_only"
SELECTION_CSV = MODULE_DIR / "top20_image_configs_for_multimodal_only.csv"
RUNNER = MODULE_DIR / "run_multimodal_experiments.py"
CLINICAL_PREP = MODULE_DIR / "prepare_clinical_multimodal.py"
UNIT_TEST = MODULE_DIR / "tests" / "test_multimodal_common.py"
VOLUME_HELPER = DL_ROOT / "una_validacion_solo" / "phase_a_serial_common.py"
DEFAULT_LIMIT = 20
DEFAULT_METRIC = "oof_f1_micro"
SUPPORTED_MODELS = {"resnet10", "resnet18", "resnet34", "seresnet50", "densenet121"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the top image-only configs as multimodal-only experiments."
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--start-rank", type=int, default=1)
    parser.add_argument("--end-rank", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-now", action="store_true", help="Run sequentially in the current allocation.")
    parser.add_argument("--submit", action="store_true", help="Create and submit one sbatch job per pending config.")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--partition", default="dgx2,dgx")
    parser.add_argument("--time", default="4-00:00:00")
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def safe_name(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def load_input_channels() -> dict[str, list[str]]:
    import importlib.util

    if not VOLUME_HELPER.exists():
        raise FileNotFoundError(f"Volume helper not found: {VOLUME_HELPER}")
    spec = importlib.util.spec_from_file_location("phase_a_serial_common", VOLUME_HELPER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {VOLUME_HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = getattr(module, "PRIMARY_INPUTS_ALL", None)
    if inputs is None:
        inputs = getattr(module, "PRIMARY_INPUTS", [])
    mapping = {}
    for item in inputs:
        mapping[str(item["name"])] = list(item["channels"])
    return mapping


def source_config_dir(run_root: Path, row: pd.Series) -> Path:
    return run_root / f"{row['preprocessing']}__{row['input_name']}__bs{int(row['batch_size'])}"


def load_completed_candidates(metric: str) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(IMAGE_RUNS_ROOT.glob("*/summary_all_configs.csv")):
        run_root = summary_path.parent
        frame = pd.read_csv(summary_path)
        required = {
            "model_name",
            "preprocessing",
            "input_name",
            "batch_size",
            "status",
            "num_folds_completed",
            "all_folds_present",
            metric,
            "oof_f1_macro",
            "mean_test_f1_micro",
            "mean_val_f1_micro",
            "mean_in_channels",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing columns in {summary_path}: {missing}")
        for _, row in frame.iterrows():
            model_name = str(row["model_name"])
            if model_name not in SUPPORTED_MODELS:
                continue
            status_ok = str(row["status"]).strip().lower() == "ok"
            folds_value = row["num_folds_completed"]
            folds_ok = pd.notna(folds_value) and int(float(folds_value)) == 5
            all_folds = as_bool(row["all_folds_present"])
            score = float(row[metric]) if pd.notna(row[metric]) else float("nan")
            if not (status_ok and folds_ok and all_folds and np.isfinite(score)):
                continue
            config_dir = source_config_dir(run_root, row)
            if not config_dir.exists():
                raise FileNotFoundError(f"Expected source config directory not found: {config_dir}")
            split_files = [config_dir / f"fold_{fold}" / "split_assignments.csv" for fold in range(1, 6)]
            missing_splits = [str(path) for path in split_files if not path.exists()]
            if missing_splits:
                raise FileNotFoundError(f"Missing split files for {config_dir}: {missing_splits}")
            item = row.to_dict()
            item["source_run_root"] = str(run_root)
            item["source_config_dir"] = str(config_dir)
            rows.append(item)
    if not rows:
        raise ValueError("No complete image-only candidates were found")
    result = pd.DataFrame(rows)
    result[metric] = pd.to_numeric(result[metric], errors="coerce")
    result["oof_f1_macro"] = pd.to_numeric(result["oof_f1_macro"], errors="coerce").fillna(-np.inf)
    result["mean_test_f1_micro"] = pd.to_numeric(result["mean_test_f1_micro"], errors="coerce").fillna(-np.inf)
    result = result.sort_values(
        [metric, "oof_f1_macro", "mean_test_f1_micro", "model_name", "preprocessing", "input_name"],
        ascending=[False, False, False, True, True, True],
    ).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def find_existing_multimodal_runs() -> dict[str, str]:
    existing = {}
    for config_path in MULTIMODAL_RUNS_ROOT.glob("*/run_configuration.json"):
        run_root = config_path.parent
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            source = payload["configuration"]["selected_configuration"]["source_config_dir"]
        except Exception:
            continue
        complete = run_root / "multimodal" / "oof_metrics.json"
        fold_preds = [
            run_root / "multimodal" / f"fold_{fold}" / "test_predictions_best_epoch.csv"
            for fold in range(1, 6)
        ]
        if complete.exists() and all(path.exists() for path in fold_preds):
            existing[str(Path(source))] = str(run_root)
    return existing


def build_manifest(row: pd.Series, channels_by_input: dict[str, list[str]], metric: str) -> dict:
    input_name = str(row["input_name"])
    if input_name not in channels_by_input:
        raise KeyError(f"Input {input_name} not found in {VOLUME_HELPER}")
    return {
        "selection_scope": "top complete image-only 3D configurations from val_externa_interna/runs",
        "selection_metric": f"{metric} on test_outer",
        "selection_direction": "maximize",
        "selection_rank": int(row["rank"]),
        "selection_status": "retrospective_exploratory",
        "selection_warning": "The outer-test metric was used for configuration selection. New multimodal outer-test results are therefore exploratory and not an unbiased confirmatory estimate.",
        "selected_model": str(row["model_name"]),
        "selected_preprocessing": str(row["preprocessing"]),
        "selected_input_name": input_name,
        "selected_channels": channels_by_input[input_name],
        "expected_input_channels": int(round(float(row["mean_in_channels"]))),
        "batch_size": int(row["batch_size"]),
        "grad_accum_steps": int(row.get("grad_accum_steps", 1) if pd.notna(row.get("grad_accum_steps", 1)) else 1),
        "effective_batch_size": int(row.get("effective_batch_size", row["batch_size"]) if pd.notna(row.get("effective_batch_size", row["batch_size"])) else row["batch_size"]),
        "source_oof_f1_micro": float(row["oof_f1_micro"]),
        "source_oof_f1_macro": float(row["oof_f1_macro"]),
        "source_mean_val_f1_micro": float(row["mean_val_f1_micro"]),
        "source_mean_test_f1_micro": float(row["mean_test_f1_micro"]),
        "source_num_folds_completed": int(float(row["num_folds_completed"])),
        "source_cohort_size": 204,
        "volume_helper_path": str(VOLUME_HELPER),
        "volume_helper_sha256": file_sha256(VOLUME_HELPER),
        "source_run_root": str(row["source_run_root"]),
        "source_config_dir": str(row["source_config_dir"]),
    }


def run_name_for(row: pd.Series) -> str:
    return "top{rank:02d}_{model}_{preproc}__{input_name}__bs{bs}_multimodal_only".format(
        rank=int(row["rank"]),
        model=safe_name(row["model_name"]),
        preproc=safe_name(row["preprocessing"]),
        input_name=safe_name(row["input_name"]),
        bs=int(row["batch_size"]),
    )


def write_sbatch(run_name: str, manifest_path: Path, args: argparse.Namespace) -> Path:
    sbatch_dir = MODULE_DIR / "slurm_top20_multimodal_only"
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    script_path = sbatch_dir / f"submit_{safe_name(run_name)}.sbatch"
    script = f"""#!/bin/bash
#SBATCH -J mm_top20
#SBATCH -p {args.partition}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem={args.mem}
#SBATCH --time={args.time}
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.err

set -euo pipefail
set +u
source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc
set -u
cd {MODULE_DIR}
python {CLINICAL_PREP}
PYTHONPATH=. python {UNIT_TEST} -v
python {RUNNER} \
  --manifest {manifest_path} \
  --run-name {run_name} \
  --modes multimodal \
  --max-epochs {args.max_epochs} \
  --patience {args.patience} \
  --num-workers {args.num_workers}
"""
    script_path.write_text(script, encoding="utf-8")
    return script_path


def run_one(run_name: str, manifest_path: Path, args: argparse.Namespace) -> None:
    subprocess.run([sys.executable, str(CLINICAL_PREP)], check=True, cwd=MODULE_DIR)
    subprocess.run([sys.executable, str(UNIT_TEST), "-v"], check=True, cwd=MODULE_DIR)
    command = [
        sys.executable,
        str(RUNNER),
        "--manifest",
        str(manifest_path),
        "--run-name",
        run_name,
        "--modes",
        "multimodal",
        "--max-epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--num-workers",
        str(args.num_workers),
    ]
    subprocess.run(command, check=True, cwd=MODULE_DIR)


def main() -> None:
    args = parse_args()
    if args.run_now and args.submit:
        raise ValueError("Use either --run-now or --submit, not both")
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.start_rank <= 0:
        raise ValueError("--start-rank must be positive")

    channels_by_input = load_input_channels()
    ranked = load_completed_candidates(args.metric).head(args.limit).copy()
    existing = find_existing_multimodal_runs()
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

    plan_rows = []
    for _, row in ranked.iterrows():
        manifest = build_manifest(row, channels_by_input, args.metric)
        run_name = run_name_for(row)
        manifest_path = MANIFEST_ROOT / f"top{int(row['rank']):02d}_{safe_name(row['model_name'])}_{safe_name(row['preprocessing'])}__{safe_name(row['input_name'])}__bs{int(row['batch_size'])}.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        existing_run = existing.get(str(Path(row["source_config_dir"])), "")
        plan_rows.append(
            {
                "rank": int(row["rank"]),
                "model_name": row["model_name"],
                "preprocessing": row["preprocessing"],
                "input_name": row["input_name"],
                "batch_size": int(row["batch_size"]),
                "expected_input_channels": manifest["expected_input_channels"],
                "source_oof_f1_micro": manifest["source_oof_f1_micro"],
                "source_oof_f1_macro": manifest["source_oof_f1_macro"],
                "source_config_dir": row["source_config_dir"],
                "manifest_path": str(manifest_path),
                "run_name": run_name,
                "planned_run_root": str(MULTIMODAL_RUNS_ROOT / run_name),
                "existing_complete_multimodal_run": existing_run,
            }
        )

    plan = pd.DataFrame(plan_rows)
    plan.to_csv(SELECTION_CSV, index=False)
    print(f"Selection written to {SELECTION_CSV}")
    print(plan[["rank", "model_name", "preprocessing", "input_name", "batch_size", "source_oof_f1_micro", "existing_complete_multimodal_run"]].to_string(index=False))

    end_rank = args.end_rank if args.end_rank is not None else args.limit
    selected = plan[(plan["rank"] >= args.start_rank) & (plan["rank"] <= end_rank)].copy()
    if args.skip_existing:
        selected = selected[selected["existing_complete_multimodal_run"].astype(str).str.len() == 0]

    if not args.run_now and not args.submit:
        print("Dry run only. Use --run-now inside a GPU allocation or --submit to launch jobs.")
        return

    for _, item in selected.iterrows():
        run_name = str(item["run_name"])
        manifest_path = Path(item["manifest_path"])
        if args.submit:
            script_path = write_sbatch(run_name, manifest_path, args)
            subprocess.run(["sbatch", str(script_path)], check=True)
        else:
            run_one(run_name, manifest_path, args)


if __name__ == "__main__":
    main()
