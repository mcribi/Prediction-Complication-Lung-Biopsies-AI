import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


DEFAULT_IMAGES_DIR = "/mnt/homeGPU/mcribilles/TFG/volumenes/nifti_convertidos_anonimizados"
DEFAULT_SEG_ROOT = "/mnt/homeGPU/mcribilles/tfm/segmentation/segmentaciones_lung_and_nodules_vessels_bronquia"
DEFAULT_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = DEFAULT_SCRIPT_DIR / "results"
DEFAULT_SEED = 42
DEFAULT_N_SPLITS = 5
KNOWN_MASK_FILENAME_TO_KIND = {
    "lung.nii.gz": "lung",
    "lung_nodules.nii.gz": "nodule",
    "lung_vessels.nii.gz": "vessels",
    "lung_trachea_bronchia.nii.gz": "trachea_bronchia",
}
BASE_META_COLS = [
    "patient_id",
    "image_path",
    "image_size",
    "image_spacing",
]
PER_MASK_META_COLS = BASE_META_COLS + ["mask_kind", "mask_path"]


def parse_args():
    parser = argparse.ArgumentParser(description="Radiomics extraction and dimensionality reduction for 210 clinical patients.")
    parser.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--seg_root", type=str, default=DEFAULT_SEG_ROOT)
    parser.add_argument(
        "--which_masks",
        type=str,
        default="all",
        help="Mask selection. Supported values: single mask (nodule, lung, vessels), both, all, available, or combinations like lung+nodule+vessels.",
    )
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--out_prefix", type=str, default="radiomics_original")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n_splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--variance_threshold", type=float, default=0.01)
    parser.add_argument("--pca_thresholds", type=float, nargs="*", default=[0.99, 0.95, 0.90])
    parser.add_argument("--skip_reduction", action="store_true")
    parser.add_argument("--save_raw_per_mask", action="store_true")
    return parser.parse_args()


def list_nifti_files(folder: Path):
    files = []
    for pattern in ("*.nii.gz", "*.nii"):
        files.extend(folder.glob(pattern))
    return sorted(files)


def strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def build_image_map(images_dir: Path):
    image_map = {}
    for path in list_nifti_files(images_dir):
        patient_id = strip_nii_suffix(path.name)
        image_map[patient_id] = path
    return image_map


def build_seg_dir_map(seg_root: Path):
    seg_map = {}
    for path in sorted(seg_root.glob("*_seg")):
        if path.is_dir():
            patient_id = path.name[:-4]
            seg_map[patient_id] = path
    return seg_map


def canonical_mask_kind(patient_id: str, mask_path: Path):
    filename = mask_path.name
    if filename in KNOWN_MASK_FILENAME_TO_KIND:
        return KNOWN_MASK_FILENAME_TO_KIND[filename]

    stem = strip_nii_suffix(filename)
    if stem == patient_id:
        return None

    normalized = stem.lower().replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def discover_masks_for_patient(patient_id: str, seg_dir: Path):
    mask_map = {}
    duplicate_masks = []
    ignored_files = []
    for path in list_nifti_files(seg_dir):
        mask_kind = canonical_mask_kind(patient_id, path)
        if mask_kind is None:
            ignored_files.append(str(path))
            continue
        if mask_kind in mask_map:
            duplicate_masks.append(
                {
                    "patient_id": patient_id,
                    "mask_kind": mask_kind,
                    "existing_path": str(mask_map[mask_kind]),
                    "duplicate_path": str(path),
                }
            )
            continue
        mask_map[mask_kind] = path
    return mask_map, duplicate_masks, ignored_files


def resolve_requested_mask_kinds(which_masks: str, available_mask_kinds):
    request = str(which_masks).strip()
    if not request:
        raise ValueError("which_masks cannot be empty")
    if request in {"all", "available"}:
        return list(available_mask_kinds)
    if request == "both":
        return ["nodule", "lung"]

    tokens = [token.strip() for token in request.replace(",", "+").split("+") if token.strip()]
    if not tokens:
        raise ValueError(f"Invalid which_masks value: {which_masks}")
    return tokens


def get_feature_modes(feature_mode: str):
    if feature_mode == "both":
        return ["basic", "extended"]
    return [feature_mode]


def has_module(module_name: str):
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


def get_tasks(images_dir: Path, seg_root: Path, which_masks: str, limit=None):
    image_map = build_image_map(images_dir)
    seg_map = build_seg_dir_map(seg_root)

    common_ids = sorted(set(image_map) & set(seg_map))
    image_only = sorted(set(image_map) - set(seg_map))
    seg_only = sorted(set(seg_map) - set(image_map))

    patient_masks = {}
    available_mask_counts = {}
    duplicate_masks = []
    ignored_seg_files = []
    for patient_id in common_ids:
        mask_map, dup_entries, ignored_files = discover_masks_for_patient(patient_id, seg_map[patient_id])
        patient_masks[patient_id] = mask_map
        duplicate_masks.extend(dup_entries)
        ignored_seg_files.extend(ignored_files)
        for mask_kind in mask_map:
            available_mask_counts[mask_kind] = available_mask_counts.get(mask_kind, 0) + 1

    available_mask_kinds = sorted(available_mask_counts)
    selected_mask_kinds = resolve_requested_mask_kinds(which_masks, available_mask_kinds)

    tasks = []
    missing_masks = []
    for patient_id in common_ids:
        image_path = image_map[patient_id]
        mask_map = patient_masks.get(patient_id, {})
        for mask_kind in selected_mask_kinds:
            mask_path = mask_map.get(mask_kind)
            if mask_path is None:
                missing_masks.append(
                    {
                        "patient_id": patient_id,
                        "mask_kind": mask_kind,
                        "missing_path": str(seg_map[patient_id] / f"{mask_kind}.nii.gz"),
                    }
                )
                continue
            tasks.append(
                {
                    "patient_id": patient_id,
                    "mask_kind": mask_kind,
                    "image_path": image_path,
                    "mask_path": mask_path,
                }
            )

    if limit is not None:
        tasks = tasks[:limit]

    summary = {
        "n_images": len(image_map),
        "n_seg_dirs": len(seg_map),
        "n_common_patients": len(common_ids),
        "image_only_ids": image_only,
        "seg_only_ids": seg_only,
        "available_mask_kinds": available_mask_kinds,
        "available_mask_counts": available_mask_counts,
        "selected_mask_kinds": selected_mask_kinds,
        "missing_masks": missing_masks,
        "duplicate_masks": duplicate_masks,
        "ignored_seg_files": ignored_seg_files,
        "selected_tasks": len(tasks),
    }
    return tasks, summary


def extract_one(extractor, patient_id: str, mask_kind: str, image_path: Path, mask_path: Path, label: int):
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    features = extractor.execute(image, mask, label=label)

    row = {
        "patient_id": patient_id,
        "mask_kind": mask_kind,
        "mask_path": str(mask_path),
        "image_path": str(image_path),
        "image_size": "x".join(map(str, image.GetSize())),
        "image_spacing": "x".join(map(str, image.GetSpacing())),
    }

    for key, value in features.items():
        if isinstance(key, str):
            row[str(key)] = value
    return row


def combine_rows_by_patient(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    combined = []
    meta_skip = set(PER_MASK_META_COLS)
    for patient_id, group in df.groupby("patient_id", sort=True):
        base_row = {
            "patient_id": patient_id,
            "image_path": group["image_path"].iloc[0],
            "image_size": group["image_size"].iloc[0],
            "image_spacing": group["image_spacing"].iloc[0],
        }
        for _, row in group.iterrows():
            mask_kind = row["mask_kind"]
            for col, val in row.items():
                if col in meta_skip:
                    continue
                base_row[f"{mask_kind}_{col}"] = val
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
        df_pca = pd.concat(
            [patient_meta.reset_index(drop=True), pd.DataFrame(X_pca, columns=pca_cols)],
            axis=1,
        )

        variant_name = f"pca_{int(round(threshold * 100))}"
        variant_dir = split_root / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        csv_path = variant_dir / f"{variant_name}.csv"
        df_pca.to_csv(csv_path, index=False)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        save_fold_feature_sets(df_pca, kf, variant_dir / "splits")

        results.append(
            {
                "variant": variant_name,
                "csv": str(csv_path),
                "n_components": int(X_pca.shape[1]),
                "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            }
        )
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

    df_sel = pd.concat(
        [patient_meta.reset_index(drop=True), pd.DataFrame(X_sel, columns=selected_cols)],
        axis=1,
    )

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

        pca_results = run_pca_variants(
            df_clean=df_clean,
            pca_thresholds=args.pca_thresholds,
            split_root=reduction_root,
            seed=args.seed,
            n_splits=effective_n_splits,
        )
        var_result = run_variance_threshold(
            df_clean=df_clean,
            variance_threshold=args.variance_threshold,
            split_root=reduction_root,
            seed=args.seed,
            n_splits=effective_n_splits,
        )

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


def run_extraction_for_mode(tasks, args, feature_mode: str, out_dir: Path):
    extractor, extractor_params = create_extractor(args, feature_mode)

    rows = []
    errors = []
    for task in tqdm(tasks, desc=f"extracting {feature_mode}"):
        try:
            row = extract_one(
                extractor=extractor,
                patient_id=task["patient_id"],
                mask_kind=task["mask_kind"],
                image_path=task["image_path"],
                mask_path=task["mask_path"],
                label=args.label,
            )
            rows.append(row)
        except Exception as exc:
            errors.append(
                {
                    "patient_id": task["patient_id"],
                    "mask_kind": task["mask_kind"],
                    "image_path": str(task["image_path"]),
                    "mask_path": str(task["mask_path"]),
                    "error": repr(exc),
                }
            )

    df_per_mask = pd.DataFrame(rows)
    if not df_per_mask.empty:
        per_mask_feature_cols = sorted([c for c in df_per_mask.columns if c not in PER_MASK_META_COLS])
        df_per_mask = df_per_mask[PER_MASK_META_COLS + per_mask_feature_cols]
    df_raw = combine_rows_by_patient(rows)
    errors_df = pd.DataFrame(errors)

    safe_mask_name = str(args.which_masks).replace(",", "+").replace("/", "_")
    run_name = f"{args.out_prefix}_{feature_mode}_{safe_mask_name}"
    raw_csv, errors_csv = save_raw_outputs(df_raw, errors_df, out_dir, run_name)

    per_mask_csv = out_dir / f"{run_name}_per_mask.csv"
    if not df_per_mask.empty:
        df_per_mask.to_csv(per_mask_csv, index=False)

    reduction_summary = None
    if not args.skip_reduction:
        reduction_summary = run_clean_and_reduction(df_raw, out_dir, run_name, args)

    mode_summary = {
        "feature_mode": feature_mode,
        "run_name": run_name,
        "raw_csv": str(raw_csv),
        "errors_csv": str(errors_csv),
        "raw_per_mask_csv": None if df_per_mask.empty else str(per_mask_csv),
        "n_rows_raw": int(df_raw.shape[0]) if not df_raw.empty else 0,
        "n_cols_raw": int(df_raw.shape[1]) if not df_raw.empty else 0,
        "n_feature_cols_raw": int(max(df_raw.shape[1] - len(BASE_META_COLS), 0)) if not df_raw.empty else 0,
        "n_rows_per_mask": int(df_per_mask.shape[0]) if not df_per_mask.empty else 0,
        "n_cols_per_mask": int(df_per_mask.shape[1]) if not df_per_mask.empty else 0,
        "n_errors": int(errors_df.shape[0]),
        "extractor_params": extractor_params,
        "reduction": reduction_summary,
    }

    return mode_summary


def main():
    args = parse_args()

    images_dir = Path(args.images_dir)
    seg_root = Path(args.seg_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    if not seg_root.exists():
        raise FileNotFoundError(f"seg_root not found: {seg_root}")

    feature_modes = get_feature_modes(args.feature_mode)
    tasks, task_summary = get_tasks(images_dir, seg_root, which_masks=args.which_masks, limit=args.limit)

    summary = {
        "images_dir": str(images_dir),
        "seg_root": str(seg_root),
        "which_masks": args.which_masks,
        "feature_mode": args.feature_mode,
        "feature_modes_run": feature_modes,
        "seed": args.seed,
        "n_splits": args.n_splits,
        "expected_total": args.expected_total,
        "task_summary": task_summary,
        "runs": [],
    }

    for feature_mode in feature_modes:
        mode_summary = run_extraction_for_mode(tasks, args, feature_mode, out_dir)
        summary["runs"].append(mode_summary)

    summary_path = out_dir / f"{args.out_prefix}_{args.feature_mode}_{args.which_masks}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Run finished")
    print(f"summary_json: {summary_path}")
    print(f"common_patients: {summary['task_summary']['n_common_patients']}")
    print(f"expected_total: {summary['expected_total']}")
    for run in summary["runs"]:
        print(f"mode: {run['feature_mode']} raw_rows: {run['n_rows_raw']} raw_features: {run['n_feature_cols_raw']} errors: {run['n_errors']}")
        reduction = run.get("reduction")
        if reduction:
            print(f"  clean_features: {reduction['n_feature_cols_clean']} shared_splits_dir: {reduction['shared_splits_dir']}")


if __name__ == "__main__":
    main()
