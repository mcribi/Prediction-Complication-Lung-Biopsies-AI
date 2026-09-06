from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from gradcam_two_outputs_image import (
    MODELED_OUTPUTS,
    _compute_gradcams,
    _load_state_dict,
    _target_layer,
    discover_complete_configs,
    parse_config_name,
    read_csv_records,
    validate_probability_match,
)

DEFAULT_RUNS_ROOT = Path(
    "/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/"
    "two_outputs_derived/image_only/runs"
)
DEFAULT_CODE_DIR = Path(
    "/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/"
    "two_outputs_derived/code/image_two_outputs"
)


def select_representative_slices(
    nodule_masses: Sequence[float],
    number: int = 4,
) -> list[int]:
    if number < 1:
        raise ValueError("The number of slices must be positive")
    if len(nodule_masses) < number:
        raise ValueError("The volume has fewer slices than requested")
    ranked = sorted(
        range(len(nodule_masses)),
        key=lambda index: (-float(nodule_masses[index]), index),
    )
    selected = ranked[:number]
    return sorted(selected)


def validate_oof_records(
    records: Iterable[Mapping],
    expected_patients: int = 204,
) -> dict:
    rows = [dict(row) for row in records]
    patient_ids = [str(row["patient_id"]) for row in rows]
    if len(rows) != expected_patients:
        raise ValueError(f"Expected {expected_patients} OOF rows, found {len(rows)}")
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("A patient appears more than once in OOF predictions")
    folds = sorted({str(row["fold"]) for row in rows})
    expected_folds = [f"fold_{index}" for index in range(1, 6)]
    if folds != expected_folds:
        raise ValueError(f"Expected folds {expected_folds}, found {folds}")
    return {"patients": len(rows), "folds": folds}


def _load_runtime(code_dir: Path):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    import dl_nested_cv_two_outputs as pipeline

    return pipeline, pipeline.base


def _mask_volume(base, preprocessing: str, mask_name: str, patient_id: str):
    import numpy as np

    path = base.PREPROC_ROOT / preprocessing / "npy" / mask_name / f"{patient_id}.npy"
    if not path.exists():
        return None
    mask = np.load(path)
    if mask.ndim == 4:
        mask = base.preprocess_channels(mask)[0]
    return (mask > 0.5).astype(np.float32)


def _actual_ct_channel(x, input_cfg, base_image_channel_count: int):
    import numpy as np

    ct_channels = []
    cursor = 0
    for channel in input_cfg["channels"]:
        if channel in {"ct_lung", "ct_nodule"}:
            ct_channels.extend(
                x[cursor + offset].detach().cpu().numpy()
                for offset in range(base_image_channel_count)
            )
            cursor += base_image_channel_count
        else:
            cursor += 1
    if not ct_channels:
        raise RuntimeError("The input has no CT intensity channel")
    return max(ct_channels, key=lambda volume: float(np.std(volume))).astype(np.float32)


def _activation_fraction(cam, mask) -> float:
    import numpy as np

    total = float(np.sum(cam))
    if total <= 1e-12 or mask is None:
        return 0.0
    return float(np.sum(cam * mask) / total)


def _render_four_slices(
    output_path: Path,
    ct_volume,
    cams,
    nodule_mask,
    slice_indices: Sequence[int],
    title: str,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import numpy as np

    figure, axes = plt.subplots(4, 3, figsize=(11, 13), constrained_layout=True)
    for row_index, slice_index in enumerate(slice_indices):
        background = ct_volume[slice_index]
        axes[row_index, 0].imshow(background, cmap="gray", origin="lower")
        axes[row_index, 0].set_ylabel(f"Corte {slice_index}")
        for column_index, cam in enumerate(cams, start=1):
            axes[row_index, column_index].imshow(background, cmap="gray", origin="lower")
            cam_slice = cam[slice_index]
            alpha = np.where(cam_slice >= 0.10, 0.15 + 0.50 * cam_slice, 0.0)
            axes[row_index, column_index].imshow(
                cam_slice,
                cmap="jet",
                vmin=0.0,
                vmax=1.0,
                alpha=alpha,
                origin="lower",
            )
        if nodule_mask is not None and float(nodule_mask[slice_index].max()) > 0:
            for axis in axes[row_index]:
                axis.contour(
                    nodule_mask[slice_index],
                    levels=[0.5],
                    colors="white",
                    linewidths=0.8,
                    origin="lower",
                )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])

    axes[0, 0].set_title("TC usada por el modelo")
    axes[0, 1].set_title("Grad-CAM Hemorragia")
    axes[0, 2].set_title("Grad-CAM Neumotorax")
    color_scale = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap="jet")
    figure.colorbar(
        color_scale,
        ax=axes,
        fraction=0.018,
        pad=0.015,
        label="Contribucion relativa",
    )
    figure.suptitle(title, fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    figure.savefig(temporary, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(output_path)


def _verify_fold_membership(config_dir: Path, records: list[dict]) -> None:
    by_fold = defaultdict(set)
    for row in records:
        by_fold[str(row["fold"])].add(str(row["patient_id"]))
    for fold in [f"fold_{index}" for index in range(1, 6)]:
        fold_rows = read_csv_records(config_dir / fold / "test_predictions_best_epoch.csv")
        fold_ids = {str(row["patient_id"]) for row in fold_rows}
        if fold_ids != by_fold[fold]:
            raise RuntimeError(f"OOF/test_outer mismatch in {config_dir} {fold}")


def _manifest_fieldnames() -> list[str]:
    return [
        "config_rank",
        "model_name",
        "config_dir",
        "tuned_oof_f1_micro",
        "patient_id",
        "fold",
        "slice_indices",
        "true_Hemorragia",
        "pred_tuned_Hemorragia",
        "prob_Hemorragia",
        "threshold_Hemorragia",
        "true_Neumotorax",
        "pred_tuned_Neumotorax",
        "prob_Neumotorax",
        "threshold_Neumotorax",
        "cam_Hemorragia_inside_lung_fraction",
        "cam_Hemorragia_inside_nodule_fraction",
        "cam_Neumotorax_inside_lung_fraction",
        "cam_Neumotorax_inside_nodule_fraction",
        "probability_check_max_abs_error",
        "png_path",
    ]


def generate_all_oof(
    runs_root: Path,
    code_dir: Path,
    output_dir: Path,
    top_k_configs: int = 5,
    slices_per_patient: int = 4,
    expected_patients: int = 204,
    max_patients_per_config: int | None = None,
) -> dict:
    import numpy as np
    import pandas as pd
    import torch

    pipeline, base = _load_runtime(code_dir)
    configs = discover_complete_configs(runs_root)[:top_k_configs]
    if len(configs) != top_k_configs:
        raise RuntimeError(f"Expected {top_k_configs} complete configs, found {len(configs)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    labels_df = base.build_multilabel_dataframe(base.CLINICAL_CSV)
    input_lookup = {row["name"]: row for row in pipeline.resolve_input_profiles("all")}

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "gradcam_manifest.csv"
    existing_rows = read_csv_records(manifest_path) if manifest_path.exists() else []
    completed_keys = {
        (row["config_dir"], row["patient_id"])
        for row in existing_rows
        if Path(row["png_path"]).exists()
    }
    fieldnames = _manifest_fieldnames()
    manifest_handle = manifest_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(manifest_handle, fieldnames=fieldnames)
    if manifest_path.stat().st_size == 0:
        writer.writeheader()
        manifest_handle.flush()

    selected_configs = []
    generated = 0
    skipped = 0
    try:
        for rank, item in enumerate(configs, start=1):
            preprocessing, input_name, _batch_size = parse_config_name(item.config_dir.name)
            input_cfg = input_lookup[input_name]
            records = read_csv_records(item.config_dir / "oof_predictions.csv")
            coverage = validate_oof_records(records, expected_patients=expected_patients)
            _verify_fold_membership(item.config_dir, records)
            selected_configs.append(
                {
                    "rank": rank,
                    "model_name": item.model_name,
                    "config_dir": str(item.config_dir),
                    "tuned_oof_f1_micro": item.tuned_f1_micro,
                    "coverage": coverage,
                }
            )
            by_fold = defaultdict(list)
            for row in records:
                by_fold[str(row["fold"])].append(row)
            if max_patients_per_config is not None:
                limited = sorted(records, key=lambda row: str(row["patient_id"]))[
                    :max_patients_per_config
                ]
                allowed = {str(row["patient_id"]) for row in limited}
                by_fold = defaultdict(
                    list,
                    {
                        fold: [row for row in fold_rows if str(row["patient_id"]) in allowed]
                        for fold, fold_rows in by_fold.items()
                    },
                )

            for fold in [f"fold_{index}" for index in range(1, 6)]:
                fold_rows = sorted(by_fold[fold], key=lambda row: str(row["patient_id"]))
                if not fold_rows:
                    continue
                patient_ids = [str(row["patient_id"]) for row in fold_rows]
                dataset = base.MultiInputVolumeDataset(
                    labels_df,
                    patient_ids,
                    preprocessing,
                    input_cfg,
                )
                sample_x, _sample_y, _sample_id = dataset[0]
                model = pipeline.build_model(
                    item.model_name,
                    int(sample_x.shape[0]),
                    len(MODELED_OUTPUTS),
                )
                state_dict = _load_state_dict(item.config_dir / fold / "best_model.pt", torch)
                model.load_state_dict(state_dict)
                model = model.to(device).eval()
                target_layer = _target_layer(item.model_name, model)

                for row_index, row in enumerate(fold_rows):
                    patient_id = str(row["patient_id"])
                    key = (str(item.config_dir), patient_id)
                    if key in completed_keys:
                        skipped += 1
                        continue
                    dataset_index = patient_ids.index(patient_id)
                    x, _y, returned_id = dataset[dataset_index]
                    if str(returned_id) != patient_id:
                        raise RuntimeError("Dataset returned an unexpected patient")
                    probabilities, cams = _compute_gradcams(
                        model,
                        target_layer,
                        x.unsqueeze(0).to(device),
                        torch,
                    )
                    expected = [float(row[f"prob_{label}"]) for label in MODELED_OUTPUTS]
                    probability_error = validate_probability_match(probabilities, expected)

                    raw_image = np.load(
                        base.PREPROC_ROOT
                        / preprocessing
                        / "npy"
                        / "images"
                        / f"{patient_id}.npy"
                    )
                    base_image_channels = len(base.preprocess_channels(raw_image))
                    ct_volume = _actual_ct_channel(x, input_cfg, base_image_channels)
                    nodule_mask = _mask_volume(
                        base, preprocessing, "masks_nodule", patient_id
                    )
                    lung_mask = _mask_volume(base, preprocessing, "masks_lung", patient_id)
                    if nodule_mask is None:
                        raise RuntimeError(f"Missing nodule mask for {patient_id}")
                    nodule_masses = nodule_mask.sum(axis=(1, 2)).tolist()
                    slice_indices = select_representative_slices(
                        nodule_masses,
                        slices_per_patient,
                    )

                    config_slug = f"rank_{rank:02d}_{item.model_name}_{item.config_dir.name}"
                    png_path = output_dir / config_slug / fold / f"{patient_id}.png"
                    title = (
                        f"Paciente {patient_id} | {fold} | contorno blanco: nodulo\n"
                        f"H: real={int(float(row['true_Hemorragia']))}, "
                        f"pred={int(float(row['pred_tuned_Hemorragia']))}, "
                        f"p={float(row['prob_Hemorragia']):.3f}, "
                        f"u={float(row['threshold_tuned_Hemorragia']):.2f} | "
                        f"N: real={int(float(row['true_Neumotórax']))}, "
                        f"pred={int(float(row['pred_tuned_Neumotórax']))}, "
                        f"p={float(row['prob_Neumotórax']):.3f}, "
                        f"u={float(row['threshold_tuned_Neumotórax']):.2f}"
                    )
                    _render_four_slices(
                        png_path,
                        ct_volume,
                        cams,
                        nodule_mask,
                        slice_indices,
                        title,
                    )
                    manifest_row = {
                        "config_rank": rank,
                        "model_name": item.model_name,
                        "config_dir": str(item.config_dir),
                        "tuned_oof_f1_micro": item.tuned_f1_micro,
                        "patient_id": patient_id,
                        "fold": fold,
                        "slice_indices": json.dumps(slice_indices),
                        "true_Hemorragia": int(float(row["true_Hemorragia"])),
                        "pred_tuned_Hemorragia": int(float(row["pred_tuned_Hemorragia"])),
                        "prob_Hemorragia": float(row["prob_Hemorragia"]),
                        "threshold_Hemorragia": float(row["threshold_tuned_Hemorragia"]),
                        "true_Neumotorax": int(float(row["true_Neumotórax"])),
                        "pred_tuned_Neumotorax": int(float(row["pred_tuned_Neumotórax"])),
                        "prob_Neumotorax": float(row["prob_Neumotórax"]),
                        "threshold_Neumotorax": float(row["threshold_tuned_Neumotórax"]),
                        "cam_Hemorragia_inside_lung_fraction": _activation_fraction(cams[0], lung_mask),
                        "cam_Hemorragia_inside_nodule_fraction": _activation_fraction(cams[0], nodule_mask),
                        "cam_Neumotorax_inside_lung_fraction": _activation_fraction(cams[1], lung_mask),
                        "cam_Neumotorax_inside_nodule_fraction": _activation_fraction(cams[1], nodule_mask),
                        "probability_check_max_abs_error": probability_error,
                        "png_path": str(png_path),
                    }
                    writer.writerow(manifest_row)
                    manifest_handle.flush()
                    generated += 1
                    print(
                        f"generated rank={rank} fold={fold} patient={patient_id} "
                        f"count={generated}",
                        flush=True,
                    )
                    del x
                    torch.cuda.empty_cache()
                del model
                torch.cuda.empty_cache()
    finally:
        manifest_handle.close()

    all_rows = read_csv_records(manifest_path)
    summary = {
        "status": "ok",
        "created_at": datetime.now().isoformat(),
        "configs": len(configs),
        "expected_patients_per_config": expected_patients,
        "total_manifest_rows": len(all_rows),
        "generated_this_run": generated,
        "skipped_existing": skipped,
        "slices_per_patient": slices_per_patient,
        "modeled_outputs": list(MODELED_OUTPUTS),
        "derived_state_has_independent_gradcam": False,
        "saved_npz": False,
        "selected_configs": selected_configs,
        "manifest": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--code-dir", type=Path, default=DEFAULT_CODE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k-configs", type=int, default=5)
    parser.add_argument("--slices-per-patient", type=int, default=4)
    parser.add_argument("--expected-patients", type=int, default=204)
    parser.add_argument("--max-patients-per-config", type=int, default=None)
    args = parser.parse_args()
    generate_all_oof(
        runs_root=args.runs_root,
        code_dir=args.code_dir,
        output_dir=args.output_dir,
        top_k_configs=args.top_k_configs,
        slices_per_patient=args.slices_per_patient,
        expected_patients=args.expected_patients,
        max_patients_per_config=args.max_patients_per_config,
    )


if __name__ == "__main__":
    main()
