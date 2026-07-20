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

import phase_a_serial_common as common

RUN_ROOTS = {
    "resnet18": Path("/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/phase_a_single_job_runs/resnet18_phase_a_20260710_095215"),
    "densenet121": Path("/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/phase_a_single_job_runs/densenet121_phase_a_20260710_095215"),
}
TOP_K = 5
CASES_PER_LABEL = 2
TARGET_LABELS = list(common.TARGET_LABELS)
LABEL_SAFE = {
    "Hemorragia": "hemorragia",
    "Neumotórax": "neumotorax",
    "Sin_complicacion": "sin_complicacion",
}
OUTPUT_ROOT = Path("/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/posthoc_gradcam_outputs_aligned_preprocessed_ct")
SOURCE_CT_ROOT = Path("/mnt/homeGPU/mcribilles/tfm/segmentation/segmentaciones_lung_and_nodules_vessels_bronquia")
MAX_IMAGE_WIDTH = 1200
# The last convolutional map is very coarse (typically 4^3 for a 128^3
# input). Trilinear interpolation therefore creates tiny positive tails over
# much of the volume. Do not display those tails as model evidence.
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
    """Display CT consistently; use a lung window when values are in HU."""
    arr = np.nan_to_num(arr, nan=-1000.0, posinf=400.0, neginf=-1000.0).astype(np.float32)
    if float(arr.min()) < -100.0:  # unnormalised CT in Hounsfield units
        lo, hi = -1000.0, 400.0
    else:
        finite = arr[np.isfinite(arr)]
        lo, hi = (float(np.quantile(finite, 0.01)), float(np.quantile(finite, 0.995)))
    if hi <= lo + 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def colorize_heatmap(cam2d: np.ndarray) -> np.ndarray:
    """Colour an already globally-normalised CAM without slice renormalisation."""
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
    # Opacity encodes strength and is exactly zero below the global threshold.
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
    """Load the CT on the exact preprocessed voxel grid seen by the model.

    This is deliberately not called an 'original CT': the preprocessing used
    TransposeD and ResizeD and the .npy file has no physical-space affine.
    """
    image_path = common.PREPROC_ROOT / preproc_name / "npy" / "images" / f"{patient_id}.npy"
    img = np.load(image_path).astype(np.float32)
    img_channels = common.preprocess_channels(img)
    volume = np.mean(img_channels, axis=0)
    return np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_source_ct(patient_id: str) -> np.ndarray | None:
    """Load the true pre-resize CT and return it in the model's (Z, X, Y) order."""
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
    # Exact inverse of preprocessing TransposeD(indices=(0, 3, 1, 2)).
    return np.transpose(source_xyz, (2, 0, 1))


def resize_spatial_array(arr: np.ndarray, shape: Tuple[int, int, int], nearest: bool = False) -> np.ndarray:
    """Resize a (Z, X, Y) array while keeping the same axis convention."""
    tensor = torch.from_numpy(arr.astype(np.float32))[None, None]
    mode = "nearest" if nearest else "trilinear"
    # The forward preprocessing used ResizeD(..., align_corners=True).
    kwargs = {} if nearest else {"align_corners": True}
    resized = F.interpolate(tensor, size=shape, mode=mode, **kwargs)
    return resized[0, 0].numpy().astype(np.float32)


def project_to_source_grid(
    cam: np.ndarray,
    source_ct: np.ndarray,
    ref_mask: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Undo the model-grid resize; source CT is already transposed to (Z, X, Y)."""
    source_shape = tuple(int(v) for v in source_ct.shape)
    source_cam = resize_spatial_array(cam, source_shape, nearest=False)
    source_mask = None
    if ref_mask is not None:
        source_mask = resize_spatial_array(ref_mask, source_shape, nearest=True)
    return source_cam, source_mask


def load_reference_mask(preproc_name: str, input_name: str, patient_id: str) -> np.ndarray | None:
    input_lookup = {row["name"]: row for row in common.PRIMARY_INPUTS}
    input_cfg = input_lookup[input_name]
    needed = common.required_masks_for_input(input_cfg)
    if not needed:
        return None
    base_dir = common.PREPROC_ROOT / preproc_name / "npy"
    masks = []
    for mask_key in needed:
        mask_path = base_dir / common.MASK_DEPENDENCIES[mask_key] / f"{patient_id}.npy"
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
    union_mask = np.clip(np.sum(masks, axis=0), 0.0, 1.0)
    return union_mask.astype(np.float32)


def choose_slice_indices(volume: np.ndarray, cam: np.ndarray, axis: int, top_n: int = 3, focus_mask: np.ndarray | None = None) -> List[int]:
    reduce_axes = tuple(i for i in range(3) if i != axis)
    cam_scores = cam.sum(axis=reduce_axes)
    vol_scores = np.abs(volume).sum(axis=reduce_axes)
    valid = vol_scores > 1e-6
    valid_idx = np.where(valid)[0]
    focus_scores = None if focus_mask is None else focus_mask.sum(axis=reduce_axes)
    if valid_idx.size == 0:
        dim = volume.shape[axis]
        mid = dim // 2
        return sorted(set([max(0, mid - 1), mid, min(dim - 1, mid + 1)]))[:top_n]
    if float(cam_scores.max()) > 1e-8 and float(cam_scores[valid].sum()) > 1e-8:
        ranked = valid_idx[np.argsort(cam_scores[valid_idx])[::-1]]
        selected = []
        min_gap = max(1, volume.shape[axis] // 12)
        for idx in ranked:
            if all(abs(int(idx) - s) >= min_gap for s in selected):
                selected.append(int(idx))
            if len(selected) >= top_n:
                break
        if selected:
            return sorted(selected)
    if focus_scores is not None and float(focus_scores.max()) > 1e-8:
        focus_valid = valid_idx[focus_scores[valid_idx] > 0]
        if focus_valid.size > 0:
            ranked = focus_valid[np.argsort(focus_scores[focus_valid])[::-1]]
            selected = []
            min_gap = max(1, volume.shape[axis] // 12)
            for idx in ranked:
                if all(abs(int(idx) - s) >= min_gap for s in selected):
                    selected.append(int(idx))
                if len(selected) >= top_n:
                    break
            if selected:
                return sorted(selected)
    if valid_idx.size >= top_n:
        positions = np.linspace(0, valid_idx.size - 1, top_n).round().astype(int)
        return sorted({int(valid_idx[pos]) for pos in positions})
    return sorted(int(x) for x in valid_idx)


def extract_plane_slice(volume: np.ndarray, cam: np.ndarray, axis: int, index: int) -> Tuple[np.ndarray, np.ndarray]:
    if axis == 0:
        return volume[index, :, :], cam[index, :, :]
    if axis == 1:
        return volume[:, index, :], cam[:, index, :]
    return volume[:, :, index], cam[:, :, index]


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
    # Preprocessing applies TransposeD(indices=(0, 3, 1, 2)); spatial order is
    # consequently (Z, X, Y), not the usual array convention (Z, Y, X).
    plane_specs = [(0, "axial", "z"), (1, "sagittal", "x"), (2, "coronal", "y")]
    rendered = []
    max_w = 0
    total_h = 0
    for axis, plane_name, axis_label in plane_specs:
        indices = choose_slice_indices(volume, cam, axis=axis, top_n=top_n_per_plane, focus_mask=ref_mask)
        for rank, idx in enumerate(indices, start=1):
            base2d, cam2d = extract_plane_slice(volume, cam, axis, idx)
            mask2d = None if ref_mask is None else extract_plane_slice(ref_mask, ref_mask, axis, idx)[0]
            merged = render_slice_panel(base2d, cam2d, f"{plane_name} {rank}/{len(indices)} {axis_label}={idx}", ref_mask2d=mask2d)
            rendered.append(merged)
            max_w = max(max_w, merged.shape[1])
            total_h += merged.shape[0]
    header_h = 24 * max(1, len(header_lines)) + 12
    canvas = np.zeros((total_h + header_h, max_w, 3), dtype=np.uint8)
    if PIL_AVAILABLE:
        header_img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(header_img)
        y = 6
        for line in header_lines:
            draw.text((8, y), line, fill=(255, 255, 255))
            y += 24
        canvas = np.array(header_img)
    offset_y = header_h
    for img in rendered:
        canvas[offset_y:offset_y + img.shape[0], :img.shape[1], :] = img
        offset_y += img.shape[0]
    return resize_if_needed_array(canvas)


def save_plane_cut_series(volume: np.ndarray, cam: np.ndarray, out_dir: Path, stem: str, ref_mask: np.ndarray | None = None) -> List[str]:
    validate_aligned_shapes(volume, cam, ref_mask)
    saved_paths = []
    plane_specs = [(0, "axial", "z"), (1, "sagittal", "x"), (2, "coronal", "y")]
    for axis, plane_name, axis_label in plane_specs:
        indices = choose_slice_indices(volume, cam, axis=axis, top_n=3, focus_mask=ref_mask)
        plane_dir = out_dir / f"cuts_{plane_name}"
        plane_dir.mkdir(parents=True, exist_ok=True)
        for rank, idx in enumerate(indices, start=1):
            base2d, cam2d = extract_plane_slice(volume, cam, axis, idx)
            mask2d = None if ref_mask is None else extract_plane_slice(ref_mask, ref_mask, axis, idx)[0]
            panel = render_slice_panel(base2d, cam2d, f"{plane_name} {rank}/{len(indices)} {axis_label}={idx}", ref_mask2d=mask2d)
            saved = save_rgb_image(panel, plane_dir / f"{stem}__{plane_name}_{rank:02d}_{axis_label}{idx}.png")
            saved_paths.append(str(saved))
    return saved_paths


def get_last_conv3d(model: nn.Module) -> Tuple[str, nn.Module]:
    last_name = None
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            last_name = name
            last_module = module
    if last_module is None:
        raise RuntimeError("No Conv3d layer found for Grad-CAM")
    return last_name, last_module


def validate_aligned_shapes(
    volume: np.ndarray,
    cam: np.ndarray,
    ref_mask: np.ndarray | None = None,
) -> None:
    """Fail loudly instead of silently overlaying arrays on different grids."""
    if volume.ndim != 3 or cam.ndim != 3:
        raise ValueError(f"Expected 3-D volume and CAM, got {volume.shape=} {cam.shape=}")
    if volume.shape != cam.shape:
        raise ValueError(f"CT/CAM grid mismatch: CT={volume.shape}, CAM={cam.shape}")
    if ref_mask is not None and ref_mask.shape != volume.shape:
        raise ValueError(f"CT/mask grid mismatch: CT={volume.shape}, mask={ref_mask.shape}")


def cam_overlap_metrics(cam: np.ndarray, ref_mask: np.ndarray | None) -> Dict[str, float | None]:
    """Quantify localisation without pretending Grad-CAM is a segmentation."""
    if ref_mask is None:
        return {
            "cam_mass_inside_reference": None,
            "visible_cam_inside_reference": None,
        }
    mask = ref_mask > 0
    total_mass = float(cam.sum())
    visible = cam >= CAM_DISPLAY_THRESHOLD
    return {
        "cam_mass_inside_reference": (
            float(cam[mask].sum()) / total_mass if total_mass > 1e-8 else 0.0
        ),
        "visible_cam_inside_reference": (
            float((visible & mask).sum()) / float(visible.sum()) if visible.any() else 0.0
        ),
    }


def load_case_tensor(labels_df: pd.DataFrame, patient_id: str, preproc_name: str, input_name: str) -> torch.Tensor:
    input_lookup = {row["name"]: row for row in common.PRIMARY_INPUTS}
    ds = common.MultiInputVolumeDataset(labels_df, [patient_id], preproc_name, input_lookup[input_name])
    x, _, _ = ds[0]
    return x


def compute_gradcam(model: nn.Module, layer: nn.Module, x: torch.Tensor, target_idx: int, device: torch.device) -> np.ndarray:
    model.eval()
    store = ActivationStore(layer)
    xb = x.unsqueeze(0).to(device)
    model.zero_grad(set_to_none=True)
    logits = model(xb)
    score = logits[0, target_idx]
    score.backward()
    activations = store.activations
    gradients = store.gradients
    store.close()
    weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=tuple(x.shape[1:]), mode="trilinear", align_corners=False)
    cam_np = cam[0, 0].detach().cpu().numpy().astype(np.float32)
    # One global normalisation for the complete 3-D map. Rendering must never
    # renormalise each slice, otherwise weak interpolation tails look strong.
    cam_np = normalize_slice(cam_np)
    return cam_np


def label_priority_columns(label: str) -> Tuple[str, str, str]:
    return f"true_{label}", f"pred_{label}", f"prob_{label}"


def select_cases_for_label(val_df: pd.DataFrame, label: str) -> List[Dict]:
    true_col, pred_col, prob_col = label_priority_columns(label)
    work = val_df.copy()
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
                }
            )
            seen_ids.add(pid)
            if len(selected) >= CASES_PER_LABEL:
                break
    return selected[:CASES_PER_LABEL]


def load_labels_df_for_run(run_root: Path) -> pd.DataFrame:
    setup = json.loads((run_root / "setup.json").read_text(encoding="utf-8"))
    labels_df = common.build_multilabel_dataframe(common.CLINICAL_CSV)
    preprocessings = common.discover_preprocessings()
    common_ids = common.collect_common_ids(labels_df, preprocessings, common.PRIMARY_INPUTS)
    labels_df = labels_df[labels_df["patient_id"].isin(common_ids)].copy().reset_index(drop=True)
    expected = int(setup.get("cohort_size", len(labels_df)))
    if len(labels_df) != expected:
        print(f"warning: cohort_size actual={len(labels_df)} expected={expected}")
    return labels_df


def summarize_run(model_name: str, run_root: Path) -> Dict:
    summary_df = pd.read_csv(run_root / "summary_all_configs.csv")
    setup = json.loads((run_root / "setup.json").read_text(encoding="utf-8"))
    ok = summary_df[summary_df["status"] == "ok"].copy()
    top = ok.sort_values(["oof_f1_micro", "mean_f1_micro", "oof_f1_macro"], ascending=False).head(TOP_K).copy()
    expected_total = len(setup.get("inputs", [])) * sum(len(item.get("candidate_batch_sizes", [])) for item in setup.get("preprocessings", []))
    return {
        "model_name": model_name,
        "run_root": str(run_root),
        "expected_total_configs": int(expected_total),
        "completed_ok_configs": int(len(ok)),
        "top_df": top,
    }


def build_model_for_config(model_name: str, sample_x: torch.Tensor) -> nn.Module:
    return common.build_model(model_name, int(sample_x.shape[0]), len(TARGET_LABELS))


def process_config(model_name: str, run_root: Path, labels_df: pd.DataFrame, row: pd.Series, model_output_dir: Path, device: torch.device) -> Dict:
    cfg_name = f"{row['preprocessing']}__{row['input_name']}__bs{int(row['batch_size'])}"
    cfg_dir = run_root / cfg_name
    fold_df = pd.read_csv(cfg_dir / "fold_summary.csv")
    if len(fold_df) != common.N_FOLDS:
        raise RuntimeError(f"Config {cfg_name} does not have all folds: {len(fold_df)}")

    best_fold_row = fold_df.sort_values(["best_val_f1_micro", "best_epoch"], ascending=[False, False]).iloc[0]
    best_fold = int(best_fold_row["fold"])
    fold_dir = cfg_dir / f"fold_{best_fold}"
    val_df = pd.read_csv(fold_dir / "val_predictions_best_epoch.csv")

    selected_cases = []
    for label in TARGET_LABELS:
        selected_cases.extend(select_cases_for_label(val_df, label))

    sample_x = load_case_tensor(labels_df, selected_cases[0]["patient_id"], str(row["preprocessing"]), str(row["input_name"]))
    model = build_model_for_config(model_name, sample_x)
    state = torch.load(fold_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    _, target_layer = get_last_conv3d(model)

    out_dir = model_output_dir / cfg_name / f"best_fold_{best_fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    case_rows = []
    for case in selected_cases:
        pid = case["patient_id"]
        label = case["label"]
        label_idx = TARGET_LABELS.index(label)
        x = load_case_tensor(labels_df, pid, str(row["preprocessing"]), str(row["input_name"]))
        cam = compute_gradcam(model, target_layer, x, label_idx, device)
        volume = load_display_volume(str(row["preprocessing"]), pid)
        ref_mask = load_reference_mask(str(row["preprocessing"]), str(row["input_name"]), pid)
        validate_aligned_shapes(volume, cam, ref_mask)
        overlap = cam_overlap_metrics(cam, ref_mask)

        sub_dir = out_dir / LABEL_SAFE[label]
        sub_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{pid}__{LABEL_SAFE[label]}__{case['case_kind']}"
        np.save(sub_dir / f"{stem}__cam.npy", cam)
        np.save(sub_dir / f"{stem}__volume.npy", volume)
        if ref_mask is not None:
            np.save(sub_dir / f"{stem}__refmask.npy", ref_mask)

        cam_sum = float(cam.sum())

        header = [
            f"model={model_name} cfg={cfg_name}",
            f"fold={best_fold} patient={pid} label={label} case={case['case_kind']} prob={case['probability']:.4f}",
            f"multi_positive={case['multi_positive']} true={case['true_label']} pred={case['pred_label']} cam_sum={cam_sum:.2f}",
            f"display=aligned_preprocessed_ct threshold={CAM_DISPLAY_THRESHOLD:.2f} mask_outline=green",
        ]
        image_arr = build_contact_sheet_array(volume, cam, header, ref_mask=ref_mask)
        image_path = save_rgb_image(image_arr, sub_dir / f"{stem}.png")
        cut_paths = save_plane_cut_series(volume, cam, sub_dir, stem, ref_mask=ref_mask)

        # Also place the CAM back on the true pre-resize CT voxel grid. This
        # avoids relying on the stale affine stored in the resized NIfTI files.
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
                f"patient={pid} label={label} case={case['case_kind']} prob={case['probability']:.4f}",
                f"display=source_ct_native_grid threshold={CAM_DISPLAY_THRESHOLD:.2f}",
                "cam=inverse_resize_to_source mask_outline=green",
            ]
            source_image = build_contact_sheet_array(
                source_ct, source_cam, source_header, ref_mask=source_mask
            )
            source_image_path = str(
                save_rgb_image(source_image, sub_dir / f"{stem}__source_ct.png")
            )
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
                "best_fold": best_fold,
                "patient_id": pid,
                "label": label,
                "case_kind": case["case_kind"],
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
        "mean_f1_micro": float(row["mean_f1_micro"]),
        "best_fold": best_fold,
        "best_fold_f1_micro": float(best_fold_row["best_val_f1_micro"]),
        "best_fold_best_epoch": int(best_fold_row["best_epoch"]),
        "cases_file": str(out_dir / "selected_cases.csv"),
    }
    (out_dir / "config_report.json").write_text(json.dumps(config_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_report


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final = {
        "output_root": str(OUTPUT_ROOT),
        "device": str(device),
        "models": [],
    }

    for model_name, run_root in RUN_ROOTS.items():
        labels_df = load_labels_df_for_run(run_root)
        summary = summarize_run(model_name, run_root)
        model_dir = OUTPUT_ROOT / model_name
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

    (OUTPUT_ROOT / "final_report.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False))


if __name__ == "__main__":
    main()
