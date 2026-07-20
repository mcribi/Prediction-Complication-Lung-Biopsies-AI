import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

DEFAULT_PREPROC_ROOT = Path("/home/mcribilles/tfm/volumenes_preprocesados")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results_preprocessed"
DEFAULT_SEED = 42
DEFAULT_N_SPLITS = 5
MASK_DIRNAME_MAP = {
    "lung": "masks_lung",
    "nodule": "masks_nodule",
    "vessels": "masks_vessels",
    "trachea_bronchia": "masks_trachea_bronchia",
}
BASE_META_COLS = [
    "patient_id",
    "preproc_name",
    "image_path",
    "image_size",
    "image_spacing",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Radiomics extraction for preprocessed NIfTI volumes.")
    parser.add_argument("--preproc_root", type=str, default=str(DEFAULT_PREPROC_ROOT))
    parser.add_argument("--preproc_dir", type=str, default=None)
    parser.add_argument("--preproc_name", type=str, default=None)
    parser.add_argument("--which_masks", type=str, default="all")
    parser.add_argument("--feature_mode", choices=["basic", "extended", "both"], default="both")
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--bin_width", type=float, default=25.0)
    parser.add_argument("--resample_spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--no_resample", action="store_true")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.add_argument("--remove_outliers", type=float, default=3.0)
    parser.add_argument("--interpolator", type=str, default="sitkBSpline")
    parser.add_argument("--expected_total", type=int, default=210)
    parser.add_argument("--limit_patients", type=int, default=None)
    parser.add_argument("--patient_ids_file", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--out_prefix", type=str, default="radiomics_preprocessed")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n_splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--variance_threshold", type=float, default=0.01)
    parser.add_argument("--pca_thresholds", type=float, nargs="*", default=[0.99, 0.95, 0.90])
    parser.add_argument("--skip_reduction", action="store_true")
    parser.add_argument("--save_raw_per_mask", action="store_true")
    return parser.parse_args()


def list_nifti_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    for pattern in ("*.nii.gz", "*.nii"):
        files.extend(folder.glob(pattern))
    return sorted(files)


def strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def build_path_map(folder: Path) -> Dict[str, Path]:
    path_map: Dict[str, Path] = {}
    for path in list_nifti_files(folder):
        path_map[strip_nii_suffix(path.name)] = path
    return path_map


def split_image_channels(image: sitk.Image, reference_image: sitk.Image):
    dimension = image.GetDimension()
    if dimension == 3:
        return [(None, image)]
    if dimension != 4:
        raise ValueError(f"Unsupported image dimension for radiomics extraction: {dimension}")

    image_array = sitk.GetArrayFromImage(image)
    if image_array.ndim != 4:
        raise ValueError(f"Unexpected 4D array shape for image splitting: {image_array.shape}")

    n_channels = int(image_array.shape[-1])
    if n_channels <= 0:
        raise ValueError(f"Invalid 4D image with zero channels: array_shape={image_array.shape}")

    channels = []
    for channel_idx in range(n_channels):
        channel_array = image_array[..., channel_idx]
        channel_image = sitk.GetImageFromArray(channel_array)
        channel_image.CopyInformation(reference_image)
        channels.append((f"ch{channel_idx}", channel_image))
    return channels


def resolve_preproc_dir(args) -> Path:
    if args.preproc_dir is not None:
        return Path(args.preproc_dir)
    if args.preproc_name is not None:
        return Path(args.preproc_root) / args.preproc_name
    raise ValueError("Either --preproc_dir or --preproc_name is required")


def infer_preproc_name(preproc_dir: Path, explicit_name: Optional[str]) -> str:
    return explicit_name if explicit_name else preproc_dir.name


def parse_mask_kinds(which_masks: str) -> List[str]:
    which = which_masks.strip()
    if which == "all":
        return list(MASK_DIRNAME_MAP.keys())
    if "+" in which:
        parts = [x.strip() for x in which.split("+") if x.strip()]
    elif "," in which:
        parts = [x.strip() for x in which.split(",") if x.strip()]
    else:
        parts = [which]
    invalid = [x for x in parts if x not in MASK_DIRNAME_MAP]
    if invalid:
        raise ValueError(f"Unsupported mask kinds: {invalid}")
    return parts


def get_feature_modes(feature_mode: str) -> List[str]:
    if feature_mode == "both":
        return ["basic", "extended"]
    return [feature_mode]


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def create_extractor(args, feature_mode: str):
    params = {
        "binWidth": args.bin_width,
        "resampledPixelSpacing": None if args.no_resample else list(args.resample_spacing),
        "interpolator": args.interpolator,
        "enableCExtensions": True,
        "normalize": args.normalize,
        "removeOutliers": args.remove_outliers,
        "verbose": False,
    }

    if feature_mode == "basic":
        extractor = featureextractor.RadiomicsFeatureExtractor(**params)
        extractor.enableImageTypeByName("Original")
    elif feature_mode == "extended":
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.settings.update(params)
        extractor.disableAllImageTypes()
        image_types = {
            "Original": {},
            "Wavelet": {},
            "LoG": {"sigma": [1.0, 2.0, 3.0]},
            "Square": {},
            "SquareRoot": {},
            "Exponential": {},
            "Logarithm": {},
            "Gradient": {},
            "LBP2D": {},
        }
        if has_module("trimesh"):
            image_types["LBP3D"] = {}
        extractor.enableImageTypes(**image_types)
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    return extractor, params


def load_patient_filter(patient_ids_file: Optional[str]) -> Optional[set]:
    if not patient_ids_file:
        return None
    ids = []
    with open(patient_ids_file, "r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value:
                ids.append(value)
    return set(ids)


def get_tasks(preproc_dir: Path, preproc_name: str, mask_kinds: List[str], patient_ids_filter: Optional[set], limit_patients: Optional[int]):
    nifti_root = preproc_dir / "nifti"
    images_dir = nifti_root / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"images dir not found: {images_dir}")

    image_map = build_path_map(images_dir)
    if patient_ids_filter is not None:
        image_map = {k: v for k, v in image_map.items() if k in patient_ids_filter}

    if limit_patients is not None:
        selected_ids = sorted(image_map)[:limit_patients]
        image_map = {k: image_map[k] for k in selected_ids}

    mask_maps = {}
    for mask_kind in mask_kinds:
        mask_dir = nifti_root / MASK_DIRNAME_MAP[mask_kind]
        if not mask_dir.exists():
            raise FileNotFoundError(f"mask dir not found for {mask_kind}: {mask_dir}")
        current_map = build_path_map(mask_dir)
        if patient_ids_filter is not None:
            current_map = {k: v for k, v in current_map.items() if k in patient_ids_filter}
        mask_maps[mask_kind] = current_map

    image_ids = set(image_map)
    per_mask_present_counts = {mask_kind: len(set(mask_maps[mask_kind]) & image_ids) for mask_kind in mask_kinds}
    per_mask_missing_ids = {
        mask_kind: sorted(image_ids - set(mask_maps[mask_kind]))
        for mask_kind in mask_kinds
    }

    tasks = []
    missing_masks = []
    for patient_id in sorted(image_map):
        image_path = image_map[patient_id]
        for mask_kind in mask_kinds:
            mask_path = mask_maps[mask_kind].get(patient_id)
            if mask_path is None:
                missing_masks.append(
                    {
                        "patient_id": patient_id,
                        "preproc_name": preproc_name,
                        "mask_kind": mask_kind,
                        "missing_path": str((nifti_root / MASK_DIRNAME_MAP[mask_kind]) / f"{patient_id}.nii.gz"),
                    }
                )
                continue
            tasks.append(
                {
                    "patient_id": patient_id,
                    "preproc_name": preproc_name,
                    "mask_kind": mask_kind,
                    "image_path": image_path,
                    "mask_path": mask_path,
                }
            )

    summary = {
        "preproc_dir": str(preproc_dir),
        "preproc_name": preproc_name,
        "n_images": len(image_map),
        "selected_patient_ids": sorted(image_map),
        "n_selected_patients": len(image_map),
        "per_mask_present_counts": per_mask_present_counts,
        "per_mask_missing_counts": {k: len(v) for k, v in per_mask_missing_ids.items()},
        "per_mask_missing_ids": per_mask_missing_ids,
        "missing_masks": missing_masks,
        "selected_tasks": len(tasks),
    }
    return tasks, summary


def extract_one(extractor, patient_id: str, preproc_name: str, mask_kind: str, image_path: Path, mask_path: Path, label: int):
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    row = {
        "patient_id": patient_id,
        "preproc_name": preproc_name,
        "mask_kind": mask_kind,
        "image_path": str(image_path),
        "image_size": "x".join(map(str, image.GetSize())),
        "image_spacing": "x".join(map(str, image.GetSpacing())),
    }
    for channel_prefix, current_image in split_image_channels(image, mask):
        features = extractor.execute(current_image, mask, label=label)
        for key, value in features.items():
            if not isinstance(key, str):
                continue
            feature_prefix = f"{mask_kind}_{key}" if channel_prefix is None else f"{mask_kind}_{channel_prefix}_{key}"
            row[feature_prefix] = value
    return row


def combine_rows_by_patient(rows, mask_kinds: List[str]):
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if len(mask_kinds) == 1:
        df = df.drop(columns=["mask_kind"], errors="ignore")
        feature_cols = sorted([c for c in df.columns if c not in BASE_META_COLS])
        return df[BASE_META_COLS + feature_cols].sort_values("patient_id").reset_index(drop=True)

    combined = []
    skip_cols = {"patient_id", "preproc_name", "mask_kind", "image_path", "image_size", "image_spacing"}
    for patient_id, group in df.groupby("patient_id", sort=True):
        base_row = {
            "patient_id": patient_id,
            "preproc_name": group["preproc_name"].iloc[0],
            "image_path": group["image_path"].iloc[0],
            "image_size": group["image_size"].iloc[0],
            "image_spacing": group["image_spacing"].iloc[0],
        }
        for _, row in group.iterrows():
            for col, val in row.items():
                if col in skip_cols:
                    continue
                if pd.isna(val):
                    continue
                base_row[col] = val
        combined.append(base_row)

    out = pd.DataFrame(combined)
    feature_cols = sorted([c for c in out.columns if c not in BASE_META_COLS])
    return out[BASE_META_COLS + feature_cols].sort_values("patient_id").reset_index(drop=True)


def save_raw_outputs(df_raw: pd.DataFrame, errors_df: pd.DataFrame, out_dir: Path, run_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / f"{run_name}_raw.csv"
    errors_csv = out_dir / f"{run_name}_errors.csv"
    df_raw.to_csv(raw_csv, index=False)
    errors_df.to_csv(errors_csv, index=False)
    return raw_csv, errors_csv


def remove_diagnostics(df: pd.DataFrame):
    diag_cols = [c for c in df.columns if "diagnostics" in c.lower()]
    cleaned = df.drop(columns=diag_cols, errors="ignore").copy()
    return cleaned, diag_cols


def remove_perfect_correlations(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in BASE_META_COLS]
    if not feature_cols:
        return df.copy(), []
    corr = df[feature_cols].corr(numeric_only=True)
    to_drop = set()
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.notna(val) and abs(val) == 1.0:
                to_drop.add(cols[j])
    out = df.drop(columns=sorted(to_drop), errors="ignore").copy()
    return out, sorted(to_drop)


def save_clean_dataset(df_clean: pd.DataFrame, out_dir: Path, run_name: str):
    clean_csv = out_dir / f"{run_name}_clean.csv"
    df_clean.to_csv(clean_csv, index=False)
    return clean_csv


def save_split_ids(kf, patient_ids, split_root: Path):
    split_root.mkdir(parents=True, exist_ok=True)
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(patient_ids), start=1):
        fold_dir = split_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_ids = pd.DataFrame({"patient_id": patient_ids.iloc[train_idx].tolist()})
        test_ids = pd.DataFrame({"patient_id": patient_ids.iloc[test_idx].tolist()})
        train_ids.to_csv(fold_dir / "train_ids.csv", index=False)
        test_ids.to_csv(fold_dir / "test_ids.csv", index=False)


def resolve_n_splits(n_samples: int, requested_n_splits: int):
    if n_samples < 2:
        return None
    return min(requested_n_splits, n_samples)


def save_fold_feature_sets(df_data: pd.DataFrame, kf, split_root: Path):
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df_data), start=1):
        fold_dir = split_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        df_data.iloc[train_idx].to_csv(fold_dir / "train.csv", index=False)
        df_data.iloc[test_idx].to_csv(fold_dir / "test.csv", index=False)


def run_pca_variants(df_clean: pd.DataFrame, pca_thresholds, split_root: Path, seed: int, n_splits: int):
    results = []
    feature_cols = [c for c in df_clean.columns if c not in BASE_META_COLS]
    if not feature_cols:
        return results
    X = df_clean[feature_cols]
    patient_meta = df_clean[BASE_META_COLS]
    for threshold in pca_thresholds:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=threshold, svd_solver="full")
        X_pca = pca.fit_transform(X_scaled)
        pca_cols = [f"PCA_{i + 1}" for i in range(X_pca.shape[1])]
        df_pca = pd.concat([patient_meta.reset_index(drop=True), pd.DataFrame(X_pca, columns=pca_cols)], axis=1)
        variant_name = f"pca_{int(round(threshold * 100))}"
        variant_dir = split_root / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        csv_path = variant_dir / f"{variant_name}.csv"
        df_pca.to_csv(csv_path, index=False)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        save_fold_feature_sets(df_pca, kf, variant_dir / "splits")
        results.append({
            "variant": variant_name,
            "csv": str(csv_path),
            "n_components": int(X_pca.shape[1]),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        })
    return results


def run_variance_threshold(df_clean: pd.DataFrame, variance_threshold: float, split_root: Path, seed: int, n_splits: int):
    feature_cols = [c for c in df_clean.columns if c not in BASE_META_COLS]
    if not feature_cols:
        return None
    X = df_clean[feature_cols]
    patient_meta = df_clean[BASE_META_COLS]
    selector = VarianceThreshold(threshold=variance_threshold)
    X_sel = selector.fit_transform(X)
    selected_cols = X.columns[selector.get_support()].tolist()
    df_sel = pd.concat([patient_meta.reset_index(drop=True), pd.DataFrame(X_sel, columns=selected_cols)], axis=1)
    variant_name = f"varthresh_{str(variance_threshold).replace('.', '_')}"
    variant_dir = split_root / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    csv_path = variant_dir / f"{variant_name}.csv"
    df_sel.to_csv(csv_path, index=False)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    save_fold_feature_sets(df_sel, kf, variant_dir / "splits")
    return {
        "variant": variant_name,
        "csv": str(csv_path),
        "n_selected_features": len(selected_cols),
    }


def run_clean_and_reduction(df_raw: pd.DataFrame, out_dir: Path, run_name: str, args):
    reduction_root = out_dir / f"{run_name}_reduction"
    reduction_root.mkdir(parents=True, exist_ok=True)
    df_nodiag, removed_diag_cols = remove_diagnostics(df_raw)
    df_clean, perfect_corr_drop = remove_perfect_correlations(df_nodiag)
    clean_csv = save_clean_dataset(df_clean, reduction_root, run_name)
    patient_ids = df_clean["patient_id"].reset_index(drop=True)
    effective_n_splits = resolve_n_splits(len(patient_ids), args.n_splits)
    ids_root = reduction_root / f"shared_splits_seed_{args.seed}"
    base_clean_dir = reduction_root / "base_clean"
    base_clean_dir.mkdir(parents=True, exist_ok=True)
    base_clean_csv = base_clean_dir / "base_clean.csv"
    df_clean.to_csv(base_clean_csv, index=False)
    pca_results = []
    var_result = None
    if effective_n_splits is not None:
        kf = KFold(n_splits=effective_n_splits, shuffle=True, random_state=args.seed)
        save_split_ids(kf, patient_ids, ids_root)
        base_kf = KFold(n_splits=effective_n_splits, shuffle=True, random_state=args.seed)
        save_fold_feature_sets(df_clean, base_kf, base_clean_dir / "splits")
        pca_results = run_pca_variants(df_clean=df_clean, pca_thresholds=args.pca_thresholds, split_root=reduction_root, seed=args.seed, n_splits=effective_n_splits)
        var_result = run_variance_threshold(df_clean=df_clean, variance_threshold=args.variance_threshold, split_root=reduction_root, seed=args.seed, n_splits=effective_n_splits)
    return {
        "reduction_root": str(reduction_root),
        "clean_csv": str(clean_csv),
        "base_clean_csv": str(base_clean_csv),
        "shared_splits_dir": str(ids_root),
        "requested_n_splits": int(args.n_splits),
        "effective_n_splits": None if effective_n_splits is None else int(effective_n_splits),
        "removed_diagnostics_count": len(removed_diag_cols),
        "removed_perfect_corr_count": len(perfect_corr_drop),
        "n_rows_clean": int(df_clean.shape[0]),
        "n_cols_clean": int(df_clean.shape[1]),
        "n_feature_cols_clean": int(max(df_clean.shape[1] - len(BASE_META_COLS), 0)),
        "pca_results": pca_results,
        "variance_threshold_result": var_result,
    }


def run_extraction_for_mode(tasks, args, feature_mode: str, preproc_name: str, mask_kinds: List[str], out_dir: Path):
    extractor, extractor_params = create_extractor(args, feature_mode)
    rows = []
    errors = []
    for task in tqdm(tasks, desc=f"extracting {preproc_name} {feature_mode}"):
        try:
            row = extract_one(
                extractor=extractor,
                patient_id=task["patient_id"],
                preproc_name=task["preproc_name"],
                mask_kind=task["mask_kind"],
                image_path=task["image_path"],
                mask_path=task["mask_path"],
                label=args.label,
            )
            rows.append(row)
        except Exception as exc:
            errors.append({
                "patient_id": task["patient_id"],
                "preproc_name": task["preproc_name"],
                "mask_kind": task["mask_kind"],
                "image_path": str(task["image_path"]),
                "mask_path": str(task["mask_path"]),
                "error": repr(exc),
            })
    df_per_mask = pd.DataFrame(rows)
    df_raw = combine_rows_by_patient(rows, mask_kinds)
    errors_df = pd.DataFrame(errors)
    run_name = f"{args.out_prefix}_{preproc_name}_{feature_mode}_{args.which_masks}"
    raw_csv, errors_csv = save_raw_outputs(df_raw, errors_df, out_dir, run_name)
    reduction_summary = None
    if not args.skip_reduction:
        reduction_summary = run_clean_and_reduction(df_raw, out_dir, run_name, args)
    mode_summary = {
        "feature_mode": feature_mode,
        "run_name": run_name,
        "raw_csv": str(raw_csv),
        "errors_csv": str(errors_csv),
        "n_rows_raw": int(df_raw.shape[0]) if not df_raw.empty else 0,
        "n_cols_raw": int(df_raw.shape[1]) if not df_raw.empty else 0,
        "n_feature_cols_raw": int(max(df_raw.shape[1] - len(BASE_META_COLS), 0)) if not df_raw.empty else 0,
        "n_errors": int(errors_df.shape[0]),
        "extractor_params": extractor_params,
        "reduction": reduction_summary,
    }
    if args.save_raw_per_mask and not df_per_mask.empty:
        per_mask_csv = out_dir / f"{run_name}_per_mask.csv"
        df_per_mask.to_csv(per_mask_csv, index=False)
        mode_summary["raw_per_mask_csv"] = str(per_mask_csv)
    return mode_summary


def main():
    args = parse_args()
    preproc_dir = resolve_preproc_dir(args)
    preproc_name = infer_preproc_name(preproc_dir, args.preproc_name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not preproc_dir.exists():
        raise FileNotFoundError(f"preproc_dir not found: {preproc_dir}")
    mask_kinds = parse_mask_kinds(args.which_masks)
    feature_modes = get_feature_modes(args.feature_mode)
    patient_ids_filter = load_patient_filter(args.patient_ids_file)
    tasks, task_summary = get_tasks(preproc_dir, preproc_name, mask_kinds, patient_ids_filter, args.limit_patients)
    summary = {
        "preproc_dir": str(preproc_dir),
        "preproc_name": preproc_name,
        "which_masks": args.which_masks,
        "mask_kinds_resolved": mask_kinds,
        "feature_mode": args.feature_mode,
        "feature_modes_run": feature_modes,
        "seed": args.seed,
        "n_splits": args.n_splits,
        "expected_total": args.expected_total,
        "skip_reduction": bool(args.skip_reduction),
        "patient_ids_file": args.patient_ids_file,
        "task_summary": task_summary,
        "runs": [],
    }
    for feature_mode in feature_modes:
        summary["runs"].append(run_extraction_for_mode(tasks, args, feature_mode, preproc_name, mask_kinds, out_dir))
    summary_path = out_dir / f"{args.out_prefix}_{preproc_name}_{args.feature_mode}_{args.which_masks}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Run finished")
    print(f"summary_json: {summary_path}")
    print(f"preproc_name: {preproc_name}")
    print(f"selected_patients: {summary['task_summary']['n_selected_patients']}")
    print(f"selected_tasks: {summary['task_summary']['selected_tasks']}")
    for run in summary["runs"]:
        print(f"mode: {run['feature_mode']} raw_rows: {run['n_rows_raw']} raw_features: {run['n_feature_cols_raw']} errors: {run['n_errors']}")


if __name__ == "__main__":
    main()
