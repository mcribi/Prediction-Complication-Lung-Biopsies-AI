from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

MODELED_OUTPUTS = ("Hemorragia", "Neumotórax")
DEFAULT_RUNS_ROOT = Path(
    "/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/"
    "two_outputs_derived/image_only/runs"
)
DEFAULT_CODE_DIR = Path(
    "/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/"
    "two_outputs_derived/code/image_two_outputs"
)


@dataclass(frozen=True)
class CompletedConfig:
    model_name: str
    run_dir: Path
    config_dir: Path
    tuned_f1_micro: float


@dataclass(frozen=True)
class SelectedCase:
    label: str
    state: str
    patient_id: str
    fold: str
    probability: float


def _as_int(value) -> int:
    return int(float(value))


def discover_complete_configs(runs_root: Path) -> list[CompletedConfig]:
    runs_root = Path(runs_root)
    found: list[CompletedConfig] = []
    for metrics_path in runs_root.glob("*/*/oof_metrics_tuned.json"):
        config_dir = metrics_path.parent
        run_dir = config_dir.parent
        if not (run_dir / "final_report.json").exists():
            continue
        setup_path = run_dir / "setup.json"
        oof_path = config_dir / "oof_predictions.csv"
        if not setup_path.exists() or not oof_path.exists():
            continue
        complete = True
        for fold_index in range(1, 6):
            fold_dir = config_dir / f"fold_{fold_index}"
            required = [
                fold_dir / "best_model.pt",
                fold_dir / "test_predictions_best_epoch.csv",
            ]
            if not all(path.exists() for path in required):
                complete = False
                break
        if not complete:
            continue
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        found.append(
            CompletedConfig(
                model_name=str(setup["model_name"]),
                run_dir=run_dir,
                config_dir=config_dir,
                tuned_f1_micro=float(metrics["two_outputs_f1_micro"]),
            )
        )
    return sorted(found, key=lambda item: (-item.tuned_f1_micro, str(item.config_dir)))


def select_oof_cases(
    records: Iterable[Mapping],
    cases_per_state: int = 1,
) -> list[SelectedCase]:
    rows = [dict(row) for row in records]
    selected: list[SelectedCase] = []
    for label in MODELED_OUTPUTS:
        true_key = f"true_{label}"
        pred_key = f"pred_tuned_{label}"
        prob_key = f"prob_{label}"
        for state, true_value, pred_value in (("TP", 1, 1), ("FN", 1, 0)):
            candidates = [
                row
                for row in rows
                if _as_int(row[true_key]) == true_value
                and _as_int(row[pred_key]) == pred_value
            ]
            candidates.sort(
                key=lambda row: (-float(row[prob_key]), str(row["patient_id"]))
            )
            for row in candidates[:cases_per_state]:
                selected.append(
                    SelectedCase(
                        label=label,
                        state=state,
                        patient_id=str(row["patient_id"]),
                        fold=str(row["fold"]),
                        probability=float(row[prob_key]),
                    )
                )
    return selected


def read_csv_records(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_config_name(config_name: str) -> tuple[str, str, int]:
    preprocessing, input_name, batch_text = config_name.rsplit("__", 2)
    match = re.fullmatch(r"bs(\d+)", batch_text)
    if match is None:
        raise ValueError(f"Invalid config name: {config_name}")
    return preprocessing, input_name, int(match.group(1))


def validate_probability_match(
    actual: Sequence[float],
    expected: Sequence[float],
    tolerance: float = 1e-3,
) -> float:
    if len(actual) != len(expected):
        raise RuntimeError("Prediction vectors have different lengths")
    maximum_error = max(
        abs(float(actual_value) - float(expected_value))
        for actual_value, expected_value in zip(actual, expected)
    )
    if maximum_error > tolerance:
        raise RuntimeError(
            f"Prediction mismatch: max_abs_error={maximum_error:.6g}, "
            f"tolerance={tolerance:.6g}"
        )
    return maximum_error


def choose_slice_index(
    cam_scores: Sequence[float],
    nodule_masses: Sequence[float] | None = None,
) -> int:
    if not cam_scores:
        raise ValueError("At least one slice score is required")
    if nodule_masses is not None:
        if len(nodule_masses) != len(cam_scores):
            raise ValueError("CAM scores and nodule masses have different lengths")
        if max(float(value) for value in nodule_masses) > 0:
            return max(
                range(len(nodule_masses)),
                key=lambda index: float(nodule_masses[index]),
            )
    if max(float(value) for value in cam_scores) > 1e-12:
        return max(range(len(cam_scores)), key=lambda index: float(cam_scores[index]))
    return len(cam_scores) // 2


def _load_runtime(code_dir: Path):
    code_dir = Path(code_dir)
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    import dl_nested_cv_two_outputs as pipeline

    return pipeline, pipeline.base


def _target_layer(model_name: str, model):
    if model_name in {"resnet10", "resnet18", "resnet34"}:
        return model.layer4[-1]
    if model_name == "densenet121":
        return model.features.denseblock4
    if model_name == "seresnet50":
        return model.layer4[-1]
    raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")


def _load_state_dict(path: Path, torch_module):
    try:
        return torch_module.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch_module.load(path, map_location="cpu")


def _compute_gradcams(model, target_layer, x, torch_module):
    import torch.nn.functional as functional

    captured = []

    def forward_hook(_module, _inputs, output):
        captured.append(output)

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        if len(captured) != 1:
            raise RuntimeError(f"Expected one target activation, found {len(captured)}")
        activations = captured[0]
        cams = []
        for class_index in range(len(MODELED_OUTPUTS)):
            gradients = torch_module.autograd.grad(
                logits[0, class_index],
                activations,
                retain_graph=(class_index < len(MODELED_OUTPUTS) - 1),
            )[0]
            weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
            cam = torch_module.relu((weights * activations).sum(dim=1, keepdim=True))
            cam = functional.interpolate(
                cam,
                size=tuple(x.shape[2:]),
                mode="trilinear",
                align_corners=False,
            )[0, 0]
            cam = cam.detach().float().cpu().numpy()
            maximum = float(cam.max())
            if maximum > 1e-12:
                cam = cam / maximum
            else:
                cam.fill(0.0)
            cams.append(cam)
        probabilities = torch_module.sigmoid(logits).detach().float().cpu().numpy()[0]
        return probabilities, cams
    finally:
        handle.remove()


def _display_volume(base, preprocessing: str, patient_id: str):
    import numpy as np

    image_path = base.PREPROC_ROOT / preprocessing / "npy" / "images" / f"{patient_id}.npy"
    channels = base.preprocess_channels(np.load(image_path))
    channel_index = int(np.argmax([float(channel.std()) for channel in channels]))
    volume = channels[channel_index].astype(np.float32)
    low, high = np.percentile(volume, [1, 99])
    if high > low:
        volume = np.clip((volume - low) / (high - low), 0.0, 1.0)
    else:
        volume = np.zeros_like(volume)
    return volume, image_path, channel_index


def _select_slice(cams: Sequence, base, preprocessing: str, patient_id: str) -> int:
    import numpy as np

    combined = np.maximum(cams[0], cams[1])
    per_slice = combined.sum(axis=(1, 2))
    nodule_path = (
        base.PREPROC_ROOT
        / preprocessing
        / "npy"
        / "masks_nodule"
        / f"{patient_id}.npy"
    )
    if nodule_path.exists():
        mask = np.load(nodule_path)
        if mask.ndim == 4:
            mask = base.preprocess_channels(mask)[0]
        mask_mass = mask.sum(axis=(1, 2))
        return choose_slice_index(per_slice.tolist(), mask_mass.tolist())
    return choose_slice_index(per_slice.tolist())


def _render_case(
    output_path: Path,
    display_volume,
    cams,
    slice_index: int,
    title: str,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import numpy as np

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    background = display_volume[slice_index]
    axes[0].imshow(background, cmap="gray", origin="lower")
    axes[0].set_title("TC preprocesada")
    for axis, cam, label in zip(axes[1:], cams, MODELED_OUTPUTS):
        axis.imshow(background, cmap="gray", origin="lower")
        cam_slice = cam[slice_index]
        alpha = np.where(cam_slice >= 0.10, 0.15 + 0.50 * cam_slice, 0.0)
        axis.imshow(
            cam_slice,
            cmap="jet",
            vmin=0.0,
            vmax=1.0,
            alpha=alpha,
            origin="lower",
        )
        axis.set_title(f"Grad-CAM {label}")
    for axis in axes:
        axis.axis("off")
    color_scale = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap="jet")
    figure.colorbar(
        color_scale,
        ax=axes,
        fraction=0.025,
        pad=0.02,
        label="Contribucion relativa",
    )
    figure.suptitle(title, fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def generate_gradcams(
    runs_root: Path,
    code_dir: Path,
    output_dir: Path,
    top_k_configs: int,
    cases_per_state: int,
    max_cases_per_config: int | None = None,
) -> dict:
    import numpy as np
    import pandas as pd
    import torch

    pipeline, base = _load_runtime(code_dir)
    completed = discover_complete_configs(runs_root)[:top_k_configs]
    if not completed:
        raise RuntimeError("No completed image-only configurations were found")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for 3D Grad-CAM generation")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    labels_df = base.build_multilabel_dataframe(base.CLINICAL_CSV)
    manifest_rows = []
    selected_configs = []

    input_lookup = {row["name"]: row for row in pipeline.resolve_input_profiles("all")}
    for rank, item in enumerate(completed, start=1):
        preprocessing, input_name, _batch_size = parse_config_name(item.config_dir.name)
        input_cfg = input_lookup[input_name]
        records = read_csv_records(item.config_dir / "oof_predictions.csv")
        selected = select_oof_cases(records, cases_per_state=cases_per_state)
        if max_cases_per_config is not None:
            selected = selected[:max_cases_per_config]
        selected_configs.append(
            {
                "rank": rank,
                "model_name": item.model_name,
                "config_dir": str(item.config_dir),
                "tuned_oof_f1_micro": item.tuned_f1_micro,
                "selected_cases": [asdict(case) for case in selected],
            }
        )
        config_output = output_dir / f"rank_{rank:02d}_{item.model_name}_{item.config_dir.name}"
        oof_by_patient = {str(row["patient_id"]): row for row in records}

        for case_index, case in enumerate(selected, start=1):
            row = oof_by_patient[case.patient_id]
            fold_dir = item.config_dir / case.fold
            if not fold_dir.exists():
                raise FileNotFoundError(f"Fold directory not found: {fold_dir}")
            dataset = base.MultiInputVolumeDataset(
                labels_df,
                [case.patient_id],
                preprocessing,
                input_cfg,
            )
            x, _y, returned_id = dataset[0]
            if str(returned_id) != case.patient_id:
                raise RuntimeError("Dataset returned an unexpected patient")
            model = pipeline.build_model(item.model_name, int(x.shape[0]), len(MODELED_OUTPUTS))
            state_dict = _load_state_dict(fold_dir / "best_model.pt", torch)
            model.load_state_dict(state_dict)
            model = model.to(device).eval()
            probabilities, cams = _compute_gradcams(
                model,
                _target_layer(item.model_name, model),
                x.unsqueeze(0).to(device),
                torch,
            )
            expected = np.asarray(
                [float(row[f"prob_{label}"]) for label in MODELED_OUTPUTS],
                dtype=np.float64,
            )
            try:
                max_probability_error = validate_probability_match(probabilities, expected)
            except RuntimeError as error:
                raise RuntimeError(f"{error} for patient {case.patient_id}") from error
            display_volume, source_image, display_channel = _display_volume(
                base,
                preprocessing,
                case.patient_id,
            )
            slice_index = _select_slice(cams, base, preprocessing, case.patient_id)
            stem = (
                f"case_{case_index:02d}_{case.patient_id}_"
                f"selected_{case.label}_{case.state}"
            )
            png_path = config_output / f"{stem}.png"
            npz_path = config_output / f"{stem}.npz"
            title = (
                f"Paciente {case.patient_id} | {case.fold} | seleccionado por "
                f"{case.label} {case.state} | corte eje 0: {slice_index}\n"
                f"H: real={_as_int(row['true_Hemorragia'])}, "
                f"pred={_as_int(row['pred_tuned_Hemorragia'])}, "
                f"p={float(row['prob_Hemorragia']):.3f}, "
                f"u={float(row['threshold_tuned_Hemorragia']):.2f} | "
                f"N: real={_as_int(row['true_Neumotórax'])}, "
                f"pred={_as_int(row['pred_tuned_Neumotórax'])}, "
                f"p={float(row['prob_Neumotórax']):.3f}, "
                f"u={float(row['threshold_tuned_Neumotórax']):.2f}"
            )
            _render_case(png_path, display_volume, cams, slice_index, title)
            np.savez_compressed(
                npz_path,
                gradcam_Hemorragia=cams[0],
                gradcam_Neumotorax=cams[1],
                selected_slice=np.asarray(slice_index),
            )
            manifest_rows.append(
                {
                    "config_rank": rank,
                    "model_name": item.model_name,
                    "config_dir": str(item.config_dir),
                    "tuned_oof_f1_micro": item.tuned_f1_micro,
                    "selected_for_label": case.label,
                    "selection_state": case.state,
                    "patient_id": case.patient_id,
                    "fold": case.fold,
                    "true_Hemorragia": _as_int(row["true_Hemorragia"]),
                    "pred_tuned_Hemorragia": _as_int(row["pred_tuned_Hemorragia"]),
                    "prob_Hemorragia": float(row["prob_Hemorragia"]),
                    "threshold_Hemorragia": float(row["threshold_tuned_Hemorragia"]),
                    "true_Neumotorax": _as_int(row["true_Neumotórax"]),
                    "pred_tuned_Neumotorax": _as_int(row["pred_tuned_Neumotórax"]),
                    "prob_Neumotorax": float(row["prob_Neumotórax"]),
                    "threshold_Neumotorax": float(row["threshold_tuned_Neumotórax"]),
                    "true_Sin_complicacion": _as_int(row["true_Sin_complicacion"]),
                    "pred_tuned_Sin_complicacion": _as_int(row["pred_tuned_Sin_complicacion"]),
                    "selected_slice_axis0": slice_index,
                    "display_source": str(source_image),
                    "display_channel": display_channel,
                    "cam_Hemorragia_sum": float(cams[0].sum()),
                    "cam_Neumotorax_sum": float(cams[1].sum()),
                    "cam_Hemorragia_null": bool(float(cams[0].sum()) <= 1e-12),
                    "cam_Neumotorax_null": bool(float(cams[1].sum()) <= 1e-12),
                    "probability_check_max_abs_error": max_probability_error,
                    "png_path": str(png_path),
                    "npz_path": str(npz_path),
                }
            )
            del model, x
            torch.cuda.empty_cache()

    manifest_path = output_dir / "gradcam_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    selection_path = output_dir / "selected_configs_and_cases.json"
    selection_path.write_text(
        json.dumps(selected_configs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "status": "ok",
        "created_at": datetime.now().isoformat(),
        "runs_root": str(runs_root),
        "top_k_configs": len(completed),
        "generated_cases": len(manifest_rows),
        "modeled_outputs": list(MODELED_OUTPUTS),
        "derived_state_has_independent_gradcam": False,
        "manifest": str(manifest_path),
        "selection": str(selection_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--code-dir", type=Path, default=DEFAULT_CODE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k-configs", type=int, default=5)
    parser.add_argument("--cases-per-state", type=int, default=1)
    parser.add_argument("--max-cases-per-config", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_gradcams(
        runs_root=args.runs_root,
        code_dir=args.code_dir,
        output_dir=args.output_dir,
        top_k_configs=args.top_k_configs,
        cases_per_state=args.cases_per_state,
        max_cases_per_config=args.max_cases_per_config,
    )


if __name__ == "__main__":
    main()
