import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a top-config resampling follow-up grid from completed baseline results.")
    parser.add_argument("--source_results_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--sampler_name", type=str, required=True, choices=["random_over_sampler", "smote"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--baseline_tag", type=str, default="__baseline__")
    return parser.parse_args()


def normalize_dir_name(name: str) -> str:
    for token in ["__slurm_", "__smoke_"]:
        if token in name:
            return name.split(token)[0]
    return name


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def config_from_used(config_used: dict, sampler_name: str) -> dict:
    config = {
        "config_name": f"{config_used['config_name']}__{sampler_name}__labelwise",
        "features_csv": config_used["features_csv"],
        "labels_csv": config_used["labels_csv"],
        "mapping_csv": config_used.get("mapping_csv"),
        "id_col": config_used.get("id_col", "patient_id"),
        "target_cols": list(config_used["target_cols_requested"]),
        "label_mode": config_used["label_mode"],
        "use_masks": config_used["use_masks"],
        "n_splits": int(config_used.get("n_splits", 5)),
        "random_state": int(config_used.get("random_state", 42)),
        "shuffle_folds": bool(config_used.get("shuffle_folds", True)),
        "scaler": config_used.get("scaler"),
        "classifier": config_used["classifier"],
        "dimred": config_used.get("dimred", "none"),
        "multilabel_strategy": config_used.get("multilabel_strategy", "one_vs_rest"),
        "split_strategy": config_used.get("split_strategy", "auto"),
        "preproc_names": config_used.get("preproc_names"),
        "save_models": False,
        "save_predictions": False,
        "sampler_name": sampler_name,
        "sampler_kwargs": {},
    }
    config.update(config_used.get("classifier_params", {}))
    return config


def main():
    args = parse_args()
    source_root = Path(args.source_results_root)
    out_root = Path(args.out_root)
    configs_root = out_root / "configs"
    summaries_root = out_root / "summaries"
    configs_root.mkdir(parents=True, exist_ok=True)
    summaries_root.mkdir(parents=True, exist_ok=True)

    candidates = []
    for run_dir in sorted(source_root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        config_used_path = run_dir / "config_used.json"
        fold_metrics_path = run_dir / "fold_metrics.csv"
        label_metrics_path = run_dir / "label_metrics.csv"
        if not (summary_path.exists() and config_used_path.exists() and fold_metrics_path.exists() and label_metrics_path.exists()):
            continue
        summary = load_json(summary_path)
        config_used = load_json(config_used_path)
        config_name = config_used.get("config_name", normalize_dir_name(run_dir.name))
        if args.baseline_tag not in config_name:
            continue
        if summary.get("status") != "completed":
            continue
        if int(summary.get("n_folds_executed", 0)) != int(config_used.get("n_splits", 5)):
            continue
        f1_micro = summary.get("mean_f1_micro")
        if f1_micro is None:
            continue
        target_cols = list(summary.get("target_cols_final") or config_used.get("target_cols_requested") or [])
        if not target_cols:
            continue
        candidates.append(
            {
                "run_dir": str(run_dir),
                "logical_name": normalize_dir_name(run_dir.name),
                "config_name": config_name,
                "summary": summary,
                "config_used": config_used,
                "mean_f1_micro": float(f1_micro),
                "mean_f1_macro": summary.get("mean_f1_macro"),
                "mean_subset_accuracy": summary.get("mean_subset_accuracy"),
                "mean_hamming_loss": summary.get("mean_hamming_loss"),
                "n_labels": len(target_cols),
                "target_cols_final": target_cols,
            }
        )

    candidates.sort(
        key=lambda row: (
            row["mean_f1_micro"],
            row["mean_f1_macro"] if row["mean_f1_macro"] is not None else -1.0,
            row["mean_subset_accuracy"] if row["mean_subset_accuracy"] is not None else -1.0,
        ),
        reverse=True,
    )
    selected = candidates[: args.top_k]
    if not selected:
        raise SystemExit("No completed baseline candidates found")

    configs = [config_from_used(row["config_used"], args.sampler_name) for row in selected]
    config_path = configs_root / f"top_{args.top_k}_{args.sampler_name}.json"
    config_path.write_text(json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_payload = {
        "source_results_root": str(source_root),
        "sampler_name": args.sampler_name,
        "top_k": args.top_k,
        "baseline_tag": args.baseline_tag,
        "selected_baselines": [
            {
                "rank": idx + 1,
                "config_name": row["config_name"],
                "run_dir": row["run_dir"],
                "mean_f1_micro": row["mean_f1_micro"],
                "mean_f1_macro": row["mean_f1_macro"],
                "mean_subset_accuracy": row["mean_subset_accuracy"],
                "mean_hamming_loss": row["mean_hamming_loss"],
                "target_cols_final": row["target_cols_final"],
            }
            for idx, row in enumerate(selected)
        ],
        "generated_config_names": [cfg["config_name"] for cfg in configs],
        "generated_config_count": len(configs),
    }
    (summaries_root / f"top_{args.top_k}_{args.sampler_name}_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
