import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except Exception:
    Image = None
    ImageDraw = None
    PIL_AVAILABLE = False

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dl_nested_cv_common as common

TOP_K = 5
CASES_PER_LABEL = 2
TARGET_LABELS = list(common.TARGET_LABELS)
LABEL_SAFE = {
    "Hemorragia": "hemorragia",
    "Neumotórax": "neumotorax",
    "Sin_complicacion": "sin_complicacion",
}
DEFAULT_RUNS_ROOT = Path("/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/val_externa_interna/runs")
OUTPUT_ROOT = Path("/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/val_externa_interna/posthoc_gradcam_outputs_test_external")
SOURCE_CT_ROOT = Path("/mnt/homeGPU/mcribilles/tfm/segmentation/segmentaciones_lung_and_nodules_vessels_bronquia")
MAX_IMAGE_WIDTH = 1200
CAM_DISPLAY_THRESHOLD = 0.20


class ActivationStore:
    def __init__(self, layer: nn.Module):
        self.activations = None
        self.gradients = None
        self.fwd_handle = layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def normalize_slice(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    positive = arr[arr > 0]
    if positive.size >= 16:
        lo = float(np.quantile(positive, 0.01))
        hi = float(np.quantile(positive, 0.995))
    else:
        lo = float(arr.min())
        hi = float(arr.max())
    if hi <= lo + 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo)


def normalize_ct_slice(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=-1000.0, posinf=400.0, neginf=-1000.0).astype(np.float32)
    if float(arr.min()) < -100.0:
        lo, hi = -1000.0, 400.0
    else:
        finite = arr[np.isfinite(arr)]
        lo, hi = (float(np.quantile(finite, 0.01)), float(np.quantile(finite, 0.995)))
    if hi <= lo + 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def colorize_heatmap(cam2d: np.ndarray) -> np.ndarray:
    x = np.clip(np.nan_to_num(cam2d, nan=0.0), 0.0, 1.0).astype(np.float32)
    r = np.clip(4.0 * x - 1.5, 0.0, 1.0)
    g = np.clip(4.0 * x - 0.5, 0.0, 1.0)
    b = np.clip(1.5 - 4.0 * x, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def mask_outline(mask2d: np.ndarray) -> np.ndarray:
    m = (mask2d > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(m, dtype=bool)
    inner = np.zeros_like(m, dtype=np.uint8)
    inner[1:-1, 1:-1] = (
        m[1:-1, 1:-1]
        & m[:-2, 1:-1]
        & m[2:, 1:-1]
        & m[1:-1, :-2]
        & m[1:-1, 2:]
    )
    return (m > 0) & (inner == 0)


def overlay_slice(
    base2d: np.ndarray,
    cam2d: np.ndarray,
    ref_mask2d: np.ndarray | None = None,
    alpha: float = 0.55,
    cam_threshold: float = CAM_DISPLAY_THRESHOLD,
) -> np.ndarray:
    gray = normalize_ct_slice(base2d)
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    heat_rgb = colorize_heatmap(cam2d)
    cam_global = np.clip(np.nan_to_num(cam2d, nan=0.0), 0.0, 1.0).astype(np.float32)
    visible = cam_global >= cam_threshold
    strength = np.where(
        visible,
        (cam_global - cam_threshold) / max(1e-6, 1.0 - cam_threshold),
        0.0,
    )[..., None]
    opacity = alpha * (0.35 + 0.65 * strength) * visible[..., None]
    mixed = gray_rgb * (1.0 - opacity) + heat_rgb * opacity
    if ref_mask2d is not None:
        outline = mask_outline(ref_mask2d)
        mixed[outline] = np.array([0.0, 1.0, 0.35], dtype=np.float32)
    return np.clip(mixed * 255.0, 0.0, 255.0).astype(np.uint8)


def resize_if_needed_array(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    if w <= MAX_IMAGE_WIDTH:
        return arr
    ratio = MAX_IMAGE_WIDTH / float(w)
    new_h = max(1, int(round(h * ratio)))
    ys = np.linspace(0, h - 1, new_h).astype(int)
    xs = np.linspace(0, w - 1, MAX_IMAGE_WIDTH).astype(int)
    return arr[ys][:, xs]


def save_rgb_image(arr: np.ndarray, path: Path) -> Path:
    arr = resize_if_needed_array(arr)
    if PIL_AVAILABLE:
        Image.fromarray(arr).save(path)
        return path
    ppm_path = path.with_suffix(".ppm")
    with ppm_path.open("wb") as f:
        header = f"P6\n{arr.shape[1]} {arr.shape[0]}\n255\n".encode("ascii")
        f.write(header)
        f.write(arr.astype(np.uint8).tobytes())
    return ppm_path


def load_display_volume(preproc_name: str, patient_id: str) -> np.ndarray:
    image_path = common.base.PREPROC_ROOT / preproc_name / "npy" / "images" / f"{patient_id}.npy"
    img = np.load(image_path).astype(np.float32)
    img_channels = common.base.preprocess_channels(img)
    volume = np.mean(img_channels, axis=0)
    return np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_source_ct(patient_id: str) -> np.ndarray | None:
    path = SOURCE_CT_ROOT / f"{patient_id}_seg" / f"{patient_id}.nii.gz"
    if not path.exists():
        return None
    try:
        import nibabel as nib
    except ImportError:
        return None
    source_xyz = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    if source_xyz.ndim != 3:
        raise ValueError(f"Expected 3-D source CT at {path}, got {source_xyz.shape}")
    return np.transpose(source_xyz, (2, 0, 1))


def resize_spatial_array(arr: np.ndarray, shape: Tuple[int, int, int], nearest: bool = False) -> np.ndarray:
    tensor = torch.from_numpy(arr.astype(np.float32))[None, None]
    mode = "nearest" if nearest else "trilinear"
    kwargs = {} if nearest else {"align_corners": True}
    resized = F.interpolate(tensor, size=shape, mode=mode, **kwargs)
    return resized[0, 0].numpy().astype(np.float32)


def project_to_source_grid(
    cam: np.ndarray,
    source_ct: np.ndarray,
    ref_mask: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray | None]:
    source_shape = tuple(int(v) for v in source_ct.shape)
    source_cam = resize_spatial_array(cam, source_shape, nearest=False)
    source_mask = None
    if ref_mask is not None:
        source_mask = resize_spatial_array(ref_mask, source_shape, nearest=True)
    return source_cam, source_mask


def load_reference_mask(preproc_name: str, input_name: str, patient_id: str) -> np.ndarray | None:
    input_lookup = {row["name"]: row for row in common.base.PRIMARY_INPUTS}
    input_cfg = input_lookup[input_name]
    needed = common.base.required_masks_for_input(input_cfg)
    if not needed:
        return None
    base_dir = common.base.PREPROC_ROOT / preproc_name / "npy"
    masks = []
    for mask_key in needed:
        mask_path = base_dir / common.base.MASK_DEPENDENCIES[mask_key] / f"{patient_id}.npy"
        if not mask_path.exists():
            continue
        arr = np.load(mask_path).astype(np.float32)
        if arr.ndim == 4:
            if arr.shape[0] <= 4:
                arr = arr[0]
            elif arr.shape[-1] <= 4:
                arr = arr[..., 0]
        masks.append((arr > 0).astype(np.float32))
    if not masks:
        return None
    merged = np.maximum.reduce(masks)
    return np.nan_to_num(merged, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_case_tensor(labels_df: pd.DataFrame, patient_id: str, preproc_name: str, input_name: str) -> torch.Tensor:
    input_lookup = {row["name"]: row for row in common.base.PRIMARY_INPUTS}
    input_cfg = input_lookup[input_name]
    ds = common.base.MultiInputVolumeDataset(labels_df, [patient_id], preproc_name, input_cfg)
    x, _, _ = ds[0]
    return x.unsqueeze(0)


def get_last_conv3d(model: nn.Module) -> Tuple[str, nn.Module]:
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            candidates.append((name, module))
    if not candidates:
        raise RuntimeError("No Conv3d layer found for Grad-CAM")
    return candidates[-1]


def compute_gradcam(model: nn.Module, target_layer: nn.Module, x: torch.Tensor, label_idx: int, device: torch.device) -> np.ndarray:
    model.eval()
    store = ActivationStore(target_layer)
    try:
        xb = x.to(device)
        model.zero_grad(set_to_none=True)
        logits = model(xb)
        score = logits[:, label_idx].sum()
        score.backward()
        activations = store.activations
        gradients = store.gradients
        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")
        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = F.interpolate(cam, size=tuple(int(v) for v in x.shape[2:]), mode="trilinear", align_corners=True)
        cam = cam - cam.amin(dim=(2, 3, 4), keepdim=True)
        denom = cam.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-8)
        cam = cam / denom
        cam_np = cam[0, 0].detach().cpu().numpy().astype(np.float32)
        cam_np = normalize_slice(cam_np)
        return cam_np
    finally:
        store.close()


def extract_plane_slices(volume: np.ndarray, cam: np.ndarray, plane: str, index: int) -> Tuple[np.ndarray, np.ndarray]:
    if plane == "axial":
        return volume[index], cam[index]
    if plane == "coronal":
        return volume[:, index, :], cam[:, index, :]
    if plane == "sagittal":
        return volume[:, :, index], cam[:, :, index]
    raise ValueError(f"Unsupported plane: {plane}")


def extract_plane_mask(ref_mask: np.ndarray | None, plane: str, index: int) -> np.ndarray | None:
    if ref_mask is None:
        return None
    if plane == "axial":
        return ref_mask[index]
    if plane == "coronal":
        return ref_mask[:, index, :]
    if plane == "sagittal":
        return ref_mask[:, :, index]
    raise ValueError(f"Unsupported plane: {plane}")


def render_slice_panel(base2d: np.ndarray, cam2d: np.ndarray, title: str, ref_mask2d: np.ndarray | None = None) -> np.ndarray:
    rgb = overlay_slice(base2d, cam2d, ref_mask2d=ref_mask2d)
    text_bar = np.zeros((28, rgb.shape[1], 3), dtype=np.uint8)
    if PIL_AVAILABLE:
        text_img = Image.fromarray(text_bar)
        draw = ImageDraw.Draw(text_img)
        draw.text((8, 6), title, fill=(255, 255, 255))
        text_bar = np.array(text_img)
    return np.concatenate([text_bar, rgb], axis=0)


def build_contact_sheet_array(
    volume: np.ndarray,
    cam: np.ndarray,
    header_lines: List[str],
    top_n_per_plane: int = 3,
    ref_mask: np.ndarray | None = None,
) -> np.ndarray:
    validate_aligned_shapes(volume, cam, ref_mask)
    rows = []
    if PIL_AVAILABLE:
        header = np.zeros((28 * max(1, len(header_lines)), 1200, 3), dtype=np.uint8)
        header_img = Image.fromarray(header)
        draw = ImageDraw.Draw(header_img)
        for idx, line in enumerate(header_lines):
            draw.text((8, 6 + 28 * idx), line, fill=(255, 255, 255))
        rows.append(np.array(header_img))
    for plane, axis in [("axial", 0), ("coronal", 1), ("sagittal", 2)]:
        cam_scores = cam.max(axis=tuple(i for i in range(3) if i != axis))
        best_indices = np.argsort(cam_scores)[::-1][:top_n_per_plane]
        panels = []
        for rank, idx in enumerate(best_indices, start=1):
            base2d, cam2d = extract_plane_slices(volume, cam, plane, int(idx))
            ref2d = extract_plane_mask(ref_mask, plane, int(idx))
            title = f"{plane} slice={int(idx)} rank={rank}"
            panels.append(render_slice_panel(base2d, cam2d, title, ref_mask2d=ref2d))
        rows.append(np.concatenate(panels, axis=1))
    width = max(arr.shape[1] for arr in rows)
    padded = []
    for arr in rows:
        if arr.shape[1] < width:
            pad = np.zeros((arr.shape[0], width - arr.shape[1], 3), dtype=np.uint8)
            arr = np.concatenate([arr, pad], axis=1)
        padded.append(arr)
    return np.concatenate(padded, axis=0)


def save_plane_cut_series(
    volume: np.ndarray,
    cam: np.ndarray,
    out_dir: Path,
    stem: str,
    ref_mask: np.ndarray | None = None,
    top_n_per_plane: int = 3,
) -> List[str]:
    validate_aligned_shapes(volume, cam, ref_mask)
    saved_paths = []
    for plane, axis in [("axial", 0), ("coronal", 1), ("sagittal", 2)]:
        plane_dir = out_dir / plane
        plane_dir.mkdir(parents=True, exist_ok=True)
        cam_scores = cam.max(axis=tuple(i for i in range(3) if i != axis))
        best_indices = np.argsort(cam_scores)[::-1][:top_n_per_plane]
        for rank, idx in enumerate(best_indices, start=1):
            base2d, cam2d = extract_plane_slices(volume, cam, plane, int(idx))
            ref2d = extract_plane_mask(ref_mask, plane, int(idx))
            rgb = overlay_slice(base2d, cam2d, ref_mask2d=ref2d)
            path = save_rgb_image(rgb, plane_dir / f"{stem}__{plane}_rank{rank}_slice{int(idx)}.png")
            saved_paths.append(str(path))
    return saved_paths


def validate_aligned_shapes(
    volume: np.ndarray,
    cam: np.ndarray,
    ref_mask: np.ndarray | None = None,
) -> None:
    if volume.ndim != 3 or cam.ndim != 3:
        raise ValueError(f"Expected 3-D volume and CAM, got {volume.shape=} {cam.shape=}")
    if volume.shape != cam.shape:
        raise ValueError(f"CT/CAM grid mismatch: CT={volume.shape}, CAM={cam.shape}")
    if ref_mask is not None and ref_mask.shape != volume.shape:
        raise ValueError(f"CT/mask grid mismatch: CT={volume.shape}, mask={ref_mask.shape}")


def cam_overlap_metrics(cam: np.ndarray, ref_mask: np.ndarray | None) -> Dict[str, float | None]:
    if ref_mask is None:
        return {
            "cam_mass_inside_reference": None,
            "visible_cam_inside_reference": None,
        }
    mask = ref_mask > 0
    total_mass = float(cam.sum())
    visible = cam >= CAM_DISPLAY_THRESHOLD
    return {
        "cam_mass_inside_reference": float(cam[mask].sum() / total_mass) if total_mass > 0 else None,
        "visible_cam_inside_reference": float(visible[mask].mean()) if mask.any() else None,
    }


def label_priority_columns(label: str) -> Tuple[str, str, str]:
    return f"true_{label}", f"pred_{label}", f"prob_{label}"


def select_cases_for_label(pred_df: pd.DataFrame, label: str) -> List[Dict]:
    true_col, pred_col, prob_col = label_priority_columns(label)
    work = pred_df.copy()
    work["multi_positive"] = ((work.get("true_Hemorragia", 0) == 1) & (work.get("true_Neumotórax", 0) == 1)).astype(int)

    tp = work[(work[true_col] == 1) & (work[pred_col] == 1)].copy()
    fn = work[(work[true_col] == 1) & (work[pred_col] == 0)].copy()
    tp = tp.sort_values(["multi_positive", prob_col], ascending=[False, False])
    fn = fn.sort_values(["multi_positive", prob_col], ascending=[False, True])

    selected = []
    for subset, case_kind in [(tp, "tp"), (fn, "fn")]:
        if not subset.empty:
            row = subset.iloc[0]
            selected.append(
                {
                    "label": label,
                    "case_kind": case_kind,
                    "patient_id": str(row["patient_id"]),
                    "probability": float(row[prob_col]),
                    "true_label": int(row[true_col]),
                    "pred_label": int(row[pred_col]),
                    "multi_positive": bool(row["multi_positive"]),
                    "fold": str(row["fold"]),
                }
            )

    if len(selected) < CASES_PER_LABEL:
        combined = pd.concat([tp, fn], axis=0)
        seen_ids = {item["patient_id"] for item in selected}
        for _, row in combined.iterrows():
            pid = str(row["patient_id"])
            if pid in seen_ids:
                continue
            case_kind = "tp" if int(row[pred_col]) == 1 else "fn"
            selected.append(
                {
                    "label": label,
                    "case_kind": case_kind,
                    "patient_id": pid,
                    "probability": float(row[prob_col]),
                    "true_label": int(row[true_col]),
                    "pred_label": int(row[pred_col]),
                    "multi_positive": bool(row["multi_positive"]),
                    "fold": str(row["fold"]),
                }
            )
            seen_ids.add(pid)
            if len(selected) >= CASES_PER_LABEL:
                break
    return selected[:CASES_PER_LABEL]


def load_labels_df_for_run(run_root: Path) -> pd.DataFrame:
    setup = json.loads((run_root / "setup.json").read_text(encoding="utf-8"))
    labels_df = common.base.build_multilabel_dataframe(common.base.CLINICAL_CSV)
    preprocessings = common.base.discover_preprocessings()
    inputs = setup.get("inputs", common.base.PRIMARY_INPUTS)
    common_ids = common.base.collect_common_ids(labels_df, preprocessings, inputs)
    labels_df = labels_df[labels_df["patient_id"].isin(common_ids)].copy().reset_index(drop=True)
    expected = int(setup.get("cohort_size", len(labels_df)))
    if len(labels_df) != expected:
        print(f"warning: cohort_size actual={len(labels_df)} expected={expected}")
    return labels_df


def summarize_run(model_name: str, run_root: Path) -> Dict:
    summary_df = pd.read_csv(run_root / "summary_all_configs.csv")
    setup = json.loads((run_root / "setup.json").read_text(encoding="utf-8"))
    ok = summary_df[summary_df["status"] == "ok"].copy()
    top = ok.sort_values(["oof_f1_micro", "mean_test_f1_micro", "mean_val_f1_micro"], ascending=False).head(TOP_K).copy()
    expected_total = len(setup.get("inputs", [])) * sum(len(item.get("candidate_batch_sizes", [])) for item in setup.get("preprocessings", []))
    return {
        "model_name": model_name,
        "run_root": str(run_root),
        "expected_total_configs": int(expected_total),
        "completed_ok_configs": int(len(ok)),
        "top_df": top,
    }


def build_model_for_config(model_name: str, sample_x: torch.Tensor) -> nn.Module:
    return common.build_model(model_name, int(sample_x.shape[1]), len(TARGET_LABELS))


def get_fold_model(model_cache: Dict[str, Tuple[nn.Module, nn.Module]], model_name: str, fold_dir: Path, sample_x: torch.Tensor, device: torch.device):
    key = str(fold_dir)
    if key in model_cache:
        return model_cache[key]
    model = build_model_for_config(model_name, sample_x)
    state = torch.load(fold_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    _, target_layer = get_last_conv3d(model)
    model_cache[key] = (model, target_layer)
    return model_cache[key]


def process_config(model_name: str, run_root: Path, labels_df: pd.DataFrame, row: pd.Series, model_output_dir: Path, device: torch.device) -> Dict:
    cfg_name = f"{row['preprocessing']}__{row['input_name']}__bs{int(row['batch_size'])}"
    cfg_dir = run_root / cfg_name
    fold_df = pd.read_csv(cfg_dir / "fold_summary.csv")
    if len(fold_df) != common.N_OUTER_FOLDS:
        raise RuntimeError(f"Config {cfg_name} does not have all folds: {len(fold_df)}")

    test_df = pd.read_csv(cfg_dir / "oof_predictions.csv")
    selected_cases = []
    for label in TARGET_LABELS:
        selected_cases.extend(select_cases_for_label(test_df, label))

    out_dir = model_output_dir / cfg_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_cache: Dict[str, Tuple[nn.Module, nn.Module]] = {}
    case_rows = []

    for case in selected_cases:
        pid = case["patient_id"]
        label = case["label"]
        fold_name = str(case["fold"])
        fold_num = int(str(fold_name).split("_")[-1]) if "_" in str(fold_name) else int(fold_name)
        fold_dir = cfg_dir / f"fold_{fold_num}"
        label_idx = TARGET_LABELS.index(label)
        x = load_case_tensor(labels_df, pid, str(row["preprocessing"]), str(row["input_name"]))
        model, target_layer = get_fold_model(model_cache, model_name, fold_dir, x, device)
        cam = compute_gradcam(model, target_layer, x, label_idx, device)
        volume = load_display_volume(str(row["preprocessing"]), pid)
        ref_mask = load_reference_mask(str(row["preprocessing"]), str(row["input_name"]), pid)
        validate_aligned_shapes(volume, cam, ref_mask)
        overlap = cam_overlap_metrics(cam, ref_mask)

        sub_dir = out_dir / LABEL_SAFE[label]
        sub_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{pid}__{LABEL_SAFE[label]}__{case['case_kind']}__{fold_name}"
        np.save(sub_dir / f"{stem}__cam.npy", cam)
        np.save(sub_dir / f"{stem}__volume.npy", volume)
        if ref_mask is not None:
            np.save(sub_dir / f"{stem}__refmask.npy", ref_mask)

        cam_sum = float(cam.sum())
        header = [
            f"model={model_name} cfg={cfg_name}",
            f"fold={fold_name} patient={pid} label={label} case={case['case_kind']} prob={case['probability']:.4f}",
            f"true={case['true_label']} pred={case['pred_label']} multi_positive={case['multi_positive']} cam_sum={cam_sum:.2f}",
            f"display=aligned_preprocessed_ct threshold={CAM_DISPLAY_THRESHOLD:.2f} mask_outline=green",
        ]
        image_arr = build_contact_sheet_array(volume, cam, header, ref_mask=ref_mask)
        image_path = save_rgb_image(image_arr, sub_dir / f"{stem}.png")
        cut_paths = save_plane_cut_series(volume, cam, sub_dir, stem, ref_mask=ref_mask)

        source_ct = load_source_ct(pid)
        source_image_path = ""
        source_cut_paths: List[str] = []
        source_cam_path = ""
        if source_ct is not None:
            source_cam, source_mask = project_to_source_grid(cam, source_ct, ref_mask)
            validate_aligned_shapes(source_ct, source_cam, source_mask)
            source_cam_file = sub_dir / f"{stem}__cam_source_grid.npy"
            np.save(source_cam_file, source_cam)
            source_cam_path = str(source_cam_file)
            source_header = [
                f"model={model_name} cfg={cfg_name}",
                f"fold={fold_name} patient={pid} label={label} case={case['case_kind']} prob={case['probability']:.4f}",
                f"display=source_ct_native_grid threshold={CAM_DISPLAY_THRESHOLD:.2f}",
                "cam=inverse_resize_to_source mask_outline=green",
                "alignment=transpose_inverse_plus_resize_back_no_manual_rotation",
            ]
            source_image = build_contact_sheet_array(source_ct, source_cam, source_header, ref_mask=source_mask)
            source_image_path = str(save_rgb_image(source_image, sub_dir / f"{stem}__source_ct.png"))
            source_cut_paths = save_plane_cut_series(
                source_ct,
                source_cam,
                sub_dir / "source_ct",
                f"{stem}__source_ct",
                ref_mask=source_mask,
            )

        case_rows.append(
            {
                "model_name": model_name,
                "config_name": cfg_name,
                "patient_id": pid,
                "label": label,
                "case_kind": case["case_kind"],
                "fold": fold_name,
                "probability": case["probability"],
                "true_label": case["true_label"],
                "pred_label": case["pred_label"],
                "multi_positive": case["multi_positive"],
                "image_path": str(image_path),
                "n_cut_pngs": len(cut_paths),
                "cut_png_example": cut_paths[0] if cut_paths else "",
                "source_ct_image_path": source_image_path,
                "n_source_ct_cut_pngs": len(source_cut_paths),
                "source_ct_cut_png_example": source_cut_paths[0] if source_cut_paths else "",
                "cam_sum": cam_sum,
                **overlap,
                "cam_path": str(sub_dir / f"{stem}__cam.npy"),
                "source_grid_cam_path": source_cam_path,
                "volume_path": str(sub_dir / f"{stem}__volume.npy"),
                "refmask_path": str(sub_dir / f"{stem}__refmask.npy") if ref_mask is not None else "",
            }
        )

    pd.DataFrame(case_rows).to_csv(out_dir / "selected_cases.csv", index=False)
    config_report = {
        "model_name": model_name,
        "config_name": cfg_name,
        "preprocessing": row["preprocessing"],
        "input_name": row["input_name"],
        "batch_size": int(row["batch_size"]),
        "oof_f1_micro": float(row["oof_f1_micro"]),
        "oof_f1_macro": float(row["oof_f1_macro"]),
        "mean_test_f1_micro": float(row["mean_test_f1_micro"]),
        "mean_val_f1_micro": float(row["mean_val_f1_micro"]),
        "cases_file": str(out_dir / "selected_cases.csv"),
    }
    (out_dir / "config_report.json").write_text(json.dumps(config_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_report


def discover_run_roots(runs_root: Path, models: List[str]) -> Dict[str, Path]:
    run_roots = {}
    for model_name in models:
        candidates = sorted(runs_root.glob(f"{model_name}_nestedcv_*"))
        if not candidates:
            continue
        valid = [p for p in candidates if (p / "summary_all_configs.csv").exists() and (p / "setup.json").exists()]
        if not valid:
            continue
        run_roots[model_name] = valid[-1]
    return run_roots


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--models", nargs="*", default=list(common.MODEL_NAMES))
    return parser


def main():
    args = build_parser().parse_args()
    runs_root = Path(args.runs_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final = {
        "output_root": str(output_root),
        "device": str(device),
        "models": [],
    }

    run_roots = discover_run_roots(runs_root, list(args.models))
    for model_name, run_root in run_roots.items():
        labels_df = load_labels_df_for_run(run_root)
        summary = summarize_run(model_name, run_root)
        model_dir = output_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        model_report = {
            "model_name": model_name,
            "run_root": summary["run_root"],
            "expected_total_configs": summary["expected_total_configs"],
            "completed_ok_configs": summary["completed_ok_configs"],
            "top_configs": [],
        }
        top_df = summary["top_df"]
        top_df.to_csv(model_dir / "top5_summary.csv", index=False)

        for _, row in top_df.iterrows():
            report = process_config(model_name, run_root, labels_df, row, model_dir, device)
            model_report["top_configs"].append(report)

        (model_dir / "model_report.json").write_text(json.dumps(model_report, ensure_ascii=False, indent=2), encoding="utf-8")
        final["models"].append(model_report)

    (output_root / "final_report.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False))


if __name__ == "__main__":
    main()
