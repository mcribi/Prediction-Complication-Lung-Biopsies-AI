import argparse
import json
import random
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.networks.nets import resnet10, resnet34, seresnet50

import phase_a_serial_common as common

DEFAULT_MODELS = ["resnet10", "resnet34", "seresnet50"]
DEFAULT_TOP_K = 6
OUTPUT_ROOT = common.PROJECT_ROOT / "codigo" / "DL_multietiqueta" / "phase_b_top_resnet_backbones"
REFERENCE_RUN_BASE = common.OUTPUT_ROOT
ALLOWED_PREPROC_PREFIXES = ("resize_cube64", "resize_cube128")
MIN_BATCH_SIZE = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Compare top ResNet18 configs against other 3D backbones.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--reference-run-root", type=str, default="")
    parser.add_argument("--min-batch-size", type=int, default=MIN_BATCH_SIZE)
    parser.add_argument("--strict-batch", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model_override(model_name: str, in_channels: int, out_channels: int):
    if model_name == "resnet10":
        model = resnet10(spatial_dims=3, n_input_channels=in_channels, num_classes=out_channels)
        model.fc = nn.Sequential(nn.Dropout(common.DROPOUT), model.fc)
        return model
    if model_name == "resnet34":
        model = resnet34(spatial_dims=3, n_input_channels=in_channels, num_classes=out_channels)
        model.fc = nn.Sequential(nn.Dropout(common.DROPOUT), model.fc)
        return model
    if model_name == "seresnet50":
        model = seresnet50(spatial_dims=3, in_channels=in_channels, num_classes=out_channels)
        model.last_linear = nn.Sequential(nn.Dropout(common.DROPOUT), model.last_linear)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def load_reference_run(reference_run_root: str):
    if reference_run_root:
        run_root = Path(reference_run_root)
    else:
        candidates = sorted(REFERENCE_RUN_BASE.glob("resnet18_phase_a_*"))
        if not candidates:
            raise FileNotFoundError("No resnet18_phase_a_* runs found")
        run_root = candidates[-1]
    summary_path = run_root / "summary_all_configs.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    summary = pd.read_csv(summary_path)
    return run_root, summary


def select_top_reference_configs(summary: pd.DataFrame, top_k: int):
    df = summary.copy()
    df = df[df["status"] == "ok"].copy()
    df = df[df["preprocessing"].astype(str).str.startswith(ALLOWED_PREPROC_PREFIXES)].copy()
    df = df.sort_values(["oof_f1_micro", "mean_f1_micro", "oof_f1_macro"], ascending=False)
    selected = df.head(top_k).copy()
    if selected.empty:
        raise RuntimeError("No reference configs selected from ResNet18 summary")
    return selected


def get_preproc_lookup(preprocessings):
    return {row["name"]: row for row in preprocessings}


def get_input_lookup(inputs):
    return {row["name"]: row for row in inputs}


def is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "oom" in msg or "cuda error: out of memory" in msg


def run_attempt(model_name, labels_df, preproc_name, input_cfg, batch_size, split_indices, config_dir, device):
    fold_rows = []
    oof_parts = []
    attempt_dir = config_dir / f"attempt_bs{batch_size}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(split_indices, start=1):
        fold_dir = attempt_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_ids = labels_df.iloc[train_idx]["patient_id"].tolist()
        val_ids = labels_df.iloc[val_idx]["patient_id"].tolist()
        fold_summary = common.train_one_fold(
            model_name=model_name,
            labels_df=labels_df,
            preproc_name=preproc_name,
            input_cfg=input_cfg,
            batch_size=batch_size,
            train_ids=train_ids,
            val_ids=val_ids,
            fold_dir=fold_dir,
            device=device,
        )
        fold_summary["fold"] = fold_idx
        fold_rows.append(fold_summary)
        pred_path = fold_dir / "val_predictions_best_epoch.csv"
        if pred_path.exists():
            oof_parts.append(pd.read_csv(pred_path))

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(attempt_dir / "fold_summary.csv", index=False)

    row = {
        "status": "ok",
        "actual_batch_size": int(batch_size),
        "attempt_dir": str(attempt_dir),
        "num_folds_completed": int(len(fold_df)),
        "all_folds_present": bool(len(fold_df) == common.N_FOLDS),
        "mean_f1_micro": float(fold_df["best_val_f1_micro"].mean()),
        "std_f1_micro": float(fold_df["best_val_f1_micro"].std(ddof=0)),
        "mean_peak_alloc_gb": float(fold_df["peak_alloc_gb"].mean()),
        "max_peak_alloc_gb": float(fold_df["peak_alloc_gb"].max()),
        "max_peak_reserved_gb": float(fold_df["peak_reserved_gb"].max()),
        "mean_epochs": float(fold_df["num_epochs_ran"].mean()),
        "mean_in_channels": float(fold_df["in_channels"].mean()),
    }
    if oof_parts:
        oof_df = pd.concat(oof_parts, ignore_index=True)
        oof_df.to_csv(attempt_dir / "oof_predictions.csv", index=False)
        y_true = oof_df[[f"true_{c}" for c in common.TARGET_LABELS]].values
        y_prob = oof_df[[f"prob_{c}" for c in common.TARGET_LABELS]].values
        oof_metrics = common.compute_metrics(y_true, y_prob, threshold=common.VAL_THRESHOLD)
        (attempt_dir / "oof_metrics.json").write_text(json.dumps(oof_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        row["oof_f1_micro"] = float(oof_metrics["f1_micro"])
        row["oof_f1_macro"] = float(oof_metrics["f1_macro"])
    return row


def run_config_with_fallback(model_name, labels_df, preproc_name, input_cfg, requested_batch_size, split_indices, config_dir, device, min_batch_size, strict_batch=False):
    current_bs = int(requested_batch_size)
    attempts = []
    while current_bs >= min_batch_size:
        try:
            row = run_attempt(
                model_name=model_name,
                labels_df=labels_df,
                preproc_name=preproc_name,
                input_cfg=input_cfg,
                batch_size=current_bs,
                split_indices=split_indices,
                config_dir=config_dir,
                device=device,
            )
            row["requested_batch_size"] = int(requested_batch_size)
            row["batch_fallback_used"] = bool(current_bs != requested_batch_size)
            row["fallback_attempts"] = json.dumps(attempts, ensure_ascii=False)
            return row
        except RuntimeError as exc:
            err = {
                "batch_size": int(current_bs),
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            }
            attempts.append(err)
            (config_dir / f"error_bs{current_bs}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            torch.cuda.empty_cache()
            if strict_batch or not is_oom_error(exc) or current_bs // 2 < min_batch_size:
                return {
                    "status": "oom_or_runtime_error",
                    "requested_batch_size": int(requested_batch_size),
                    "actual_batch_size": None,
                    "batch_fallback_used": False,
                    "fallback_attempts": json.dumps(attempts, ensure_ascii=False),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                    "num_folds_completed": 0,
                    "all_folds_present": False,
                }
            current_bs = max(min_batch_size, current_bs // 2)
        except Exception as exc:
            (config_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            return {
                "status": "failed",
                "requested_batch_size": int(requested_batch_size),
                "actual_batch_size": None,
                "batch_fallback_used": False,
                "fallback_attempts": json.dumps(attempts, ensure_ascii=False),
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
                "num_folds_completed": 0,
                "all_folds_present": False,
            }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this job")

    set_seed(common.SEED)
    device = torch.device("cuda")
    common.build_model = build_model_override

    reference_run_root, reference_summary = load_reference_run(args.reference_run_root)
    selected_configs = select_top_reference_configs(reference_summary, args.top_k)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = OUTPUT_ROOT / f"backbone_compare_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    labels_df = common.build_multilabel_dataframe(common.CLINICAL_CSV)
    preprocessings = common.discover_preprocessings()
    preproc_lookup = get_preproc_lookup(preprocessings)
    input_lookup = get_input_lookup(common.PRIMARY_INPUTS)

    selected_preprocessings = [preproc_lookup[name] for name in selected_configs["preprocessing"].unique()]
    selected_inputs = [input_lookup[name] for name in selected_configs["input_name"].unique()]
    common_ids = common.collect_common_ids(labels_df, selected_preprocessings, selected_inputs)
    labels_df = labels_df[labels_df["patient_id"].isin(common_ids)].copy().reset_index(drop=True)

    splitter = common.StratifiedKFold(n_splits=common.N_FOLDS, shuffle=True, random_state=common.SEED)
    split_indices = list(splitter.split(labels_df["patient_id"].tolist(), labels_df["labelset_key"].tolist()))

    setup = {
        "timestamp": timestamp,
        "reference_run_root": str(reference_run_root),
        "selected_reference_configs": selected_configs[["preprocessing", "input_name", "batch_size", "oof_f1_micro", "oof_f1_macro"]].to_dict(orient="records"),
        "models": list(args.models),
        "top_k": int(args.top_k),
        "min_batch_size": int(args.min_batch_size),
        "strict_batch": bool(args.strict_batch),
        "seed": common.SEED,
        "n_folds": common.N_FOLDS,
        "max_epochs": common.MAX_EPOCHS,
        "patience": common.PATIENCE,
        "lr": common.LEARNING_RATE,
        "dropout": common.DROPOUT,
        "weight_decay": common.WEIGHT_DECAY,
        "target_labels": common.TARGET_LABELS,
        "cohort_size": int(len(labels_df)),
        "gpu_name": torch.cuda.get_device_name(0),
    }
    (run_root / "setup.json").write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding="utf-8")

    all_rows = []
    for model_name in args.models:
        model_dir = run_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        for config in selected_configs.to_dict(orient="records"):
            preproc_name = config["preprocessing"]
            input_name = config["input_name"]
            input_cfg = deepcopy(input_lookup[input_name])
            requested_batch_size = int(config["batch_size"])
            config_name = f"{preproc_name}__{input_name}__refbs{requested_batch_size}"
            config_dir = model_dir / config_name
            config_dir.mkdir(parents=True, exist_ok=True)

            print(f"START model={model_name} config={config_name}")
            row = {
                "model_name": model_name,
                "preprocessing": preproc_name,
                "input_name": input_name,
                "reference_batch_size": requested_batch_size,
                "reference_oof_f1_micro": float(config.get("oof_f1_micro", np.nan)),
                "reference_oof_f1_macro": float(config.get("oof_f1_macro", np.nan)),
                "status": "pending",
                "error_type": None,
                "error_message": None,
            }
            result = run_config_with_fallback(
                model_name=model_name,
                labels_df=labels_df,
                preproc_name=preproc_name,
                input_cfg=input_cfg,
                requested_batch_size=requested_batch_size,
                split_indices=split_indices,
                config_dir=config_dir,
                device=device,
                min_batch_size=args.min_batch_size,
                strict_batch=args.strict_batch,
            )
            row.update(result)
            all_rows.append(row)
            pd.DataFrame(all_rows).to_csv(run_root / "summary_all_configs.csv", index=False)

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(run_root / "summary_all_configs.csv", index=False)
    summary_ok = summary_df[summary_df["status"] == "ok"].copy()
    if not summary_ok.empty:
        best_by_model = (
            summary_ok.sort_values(["model_name", "oof_f1_micro", "mean_f1_micro", "oof_f1_macro"], ascending=[True, False, False, False])
            .groupby("model_name", as_index=False)
            .head(1)
        )
        best_by_model.to_csv(run_root / "best_by_model.csv", index=False)

    final_report = {
        "run_root": str(run_root),
        "reference_run_root": str(reference_run_root),
        "models": list(args.models),
        "total_jobs": int(len(summary_df)),
        "ok_jobs": int((summary_df["status"] == "ok").sum()),
        "failed_jobs": int((summary_df["status"] != "ok").sum()),
    }
    (run_root / "final_report.json").write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final_report, ensure_ascii=False))


if __name__ == "__main__":
    main()
