from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import SimpleITK as sitk

INDEX_CSV = Path('/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/pretrained_finetuning/rescue_20260728_pretraining_strategy/03_luna_lidc_domain_pretraining/luna16_patch_index/luna16_patch_index_pos1_neg3_seed17.csv')
OUT_JSON = INDEX_CSV.parent / 'luna16_patch_read_smoke.json'
PATCH_SIZE = 64
HU_MIN = -1000.0
HU_MAX = 400.0


def crop_patch_zyx(volume: np.ndarray, center_xyz, patch_size: int = PATCH_SIZE) -> np.ndarray:
    cx, cy, cz = [int(round(float(v))) for v in center_xyz]
    center_zyx = np.array([cz, cy, cx], dtype=int)
    shape = np.array(volume.shape, dtype=int)
    half = patch_size // 2
    start = center_zyx - half
    end = start + patch_size

    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, shape)
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    patch = np.full((patch_size, patch_size, patch_size), HU_MIN, dtype=np.float32)
    patch[dst_start[0]:dst_end[0], dst_start[1]:dst_end[1], dst_start[2]:dst_end[2]] = volume[src_start[0]:src_end[0], src_start[1]:src_end[1], src_start[2]:src_end[2]]
    return patch


def normalize_hu(patch: np.ndarray) -> np.ndarray:
    patch = np.clip(patch.astype(np.float32), HU_MIN, HU_MAX)
    return (patch - HU_MIN) / (HU_MAX - HU_MIN)


def read_one(row: pd.Series) -> dict:
    img = sitk.ReadImage(str(row['mhd_path']))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # z, y, x
    patch = crop_patch_zyx(arr, (row['voxel_x'], row['voxel_y'], row['voxel_z']))
    patch_norm = normalize_hu(patch)
    return {
        'sample_id': row['sample_id'],
        'label': int(row['label']),
        'seriesuid': row['seriesuid'],
        'volume_shape_zyx': list(arr.shape),
        'patch_shape_zyx': list(patch.shape),
        'patch_min_hu': float(np.min(patch)),
        'patch_max_hu': float(np.max(patch)),
        'patch_mean_hu': float(np.mean(patch)),
        'patch_norm_min': float(np.min(patch_norm)),
        'patch_norm_max': float(np.max(patch_norm)),
        'patch_norm_mean': float(np.mean(patch_norm)),
        'finite': bool(np.isfinite(patch_norm).all()),
    }


def main() -> int:
    df = pd.read_csv(INDEX_CSV)
    samples = [df[df['label'] == 1].iloc[0], df[df['label'] == 0].iloc[0]]
    results = [read_one(row) for row in samples]
    report = {'index_csv': str(INDEX_CSV), 'patch_size': PATCH_SIZE, 'results': results}
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
