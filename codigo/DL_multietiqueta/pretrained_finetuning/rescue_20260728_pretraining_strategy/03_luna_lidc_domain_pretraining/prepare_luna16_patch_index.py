from __future__ import annotations

from pathlib import Path
import json
import math
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd

LUNA_ROOT = Path('/mnt/homeGPU/mcribilles/tfm/external_data/LUNA16')
EXTRACTED_ROOT = LUNA_ROOT / 'extracted'
OUT_DIR = Path('/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/pretrained_finetuning/rescue_20260728_pretraining_strategy/03_luna_lidc_domain_pretraining/luna16_patch_index')
PATCH_SIZE_VOX = 64
NEG_PER_POS = 3
MIN_NEG_DISTANCE_MM = 15.0
SEED = 17
VAL_SERIES_FRACTION = 0.2


def parse_mhd(path: Path) -> Dict[str, str]:
    meta = {}
    for line in path.read_text(errors='replace').splitlines():
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        meta[key.strip()] = value.strip()
    return meta


def parse_float_triplet(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.split()], dtype=float)


def parse_int_triplet(text: str) -> np.ndarray:
    return np.array([int(x) for x in text.split()], dtype=int)


def build_series_map() -> Dict[str, Dict[str, object]]:
    series_map = {}
    for mhd_path in sorted(EXTRACTED_ROOT.glob('subset*/*.mhd')):
        seriesuid = mhd_path.stem
        subset_match = re.search(r'subset\d+', str(mhd_path))
        subset = subset_match.group(0) if subset_match else ''
        meta = parse_mhd(mhd_path)
        raw_path = mhd_path.parent / meta.get('ElementDataFile', '')
        spacing = parse_float_triplet(meta['ElementSpacing'])
        origin = parse_float_triplet(meta['Offset'])
        direction = np.array([float(x) for x in meta.get('TransformMatrix', '1 0 0 0 1 0 0 0 1').split()], dtype=float).reshape(3, 3)
        dim_size = parse_int_triplet(meta['DimSize'])
        seg_path = EXTRACTED_ROOT / 'seg-lungs-LUNA16' / f'{seriesuid}.mhd'
        series_map[seriesuid] = {
            'seriesuid': seriesuid,
            'subset': subset,
            'mhd_path': str(mhd_path),
            'raw_path': str(raw_path),
            'seg_mhd_path': str(seg_path) if seg_path.exists() else '',
            'spacing_x': spacing[0],
            'spacing_y': spacing[1],
            'spacing_z': spacing[2],
            'origin_x': origin[0],
            'origin_y': origin[1],
            'origin_z': origin[2],
            'dim_x': dim_size[0],
            'dim_y': dim_size[1],
            'dim_z': dim_size[2],
            'direction': direction,
        }
    return series_map


def world_to_voxel(row: pd.Series, info: Dict[str, object]) -> Tuple[float, float, float]:
    world = np.array([row['coordX'], row['coordY'], row['coordZ']], dtype=float)
    origin = np.array([info['origin_x'], info['origin_y'], info['origin_z']], dtype=float)
    spacing = np.array([info['spacing_x'], info['spacing_y'], info['spacing_z']], dtype=float)
    direction = info['direction']
    continuous_index = np.linalg.inv(direction).dot(world - origin) / spacing
    return tuple(float(x) for x in continuous_index)


def add_geometry(df: pd.DataFrame, series_map: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        info = series_map.get(str(row['seriesuid']))
        if info is None:
            continue
        vx, vy, vz = world_to_voxel(row, info)
        rec = row.to_dict()
        rec.update({k: v for k, v in info.items() if k != 'direction'})
        rec.update({'voxel_x': vx, 'voxel_y': vy, 'voxel_z': vz})
        rec['center_inside_volume'] = bool(0 <= vx < info['dim_x'] and 0 <= vy < info['dim_y'] and 0 <= vz < info['dim_z'])
        records.append(rec)
    return pd.DataFrame(records)


def filter_negatives_far_from_annotations(candidates_neg: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    ann_by_series = {
        sid: group[['coordX', 'coordY', 'coordZ']].to_numpy(dtype=float)
        for sid, group in annotations.groupby('seriesuid')
    }
    keep = []
    for _, row in candidates_neg.iterrows():
        ann = ann_by_series.get(row['seriesuid'])
        if ann is None or len(ann) == 0:
            keep.append(True)
            continue
        point = np.array([row['coordX'], row['coordY'], row['coordZ']], dtype=float)
        min_dist = np.sqrt(((ann - point) ** 2).sum(axis=1)).min()
        keep.append(bool(min_dist >= MIN_NEG_DISTANCE_MM))
    return candidates_neg.loc[keep].copy()


def assign_series_split(series_ids: pd.Series) -> Dict[str, str]:
    unique = np.array(sorted(series_ids.astype(str).unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * VAL_SERIES_FRACTION)))
    val = set(unique[:n_val])
    return {sid: ('val' if sid in val else 'train') for sid in unique}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    series_map = build_series_map()
    annotations = pd.read_csv(LUNA_ROOT / 'annotations.csv')
    candidates_path = EXTRACTED_ROOT / 'candidates_V2.csv'
    if not candidates_path.exists():
        candidates_path = LUNA_ROOT / 'candidates.csv'
    candidates = pd.read_csv(candidates_path)

    annotations = annotations[annotations['seriesuid'].astype(str).isin(series_map)].copy()
    positives = annotations.copy()
    positives['label'] = 1
    positives['source'] = 'annotations'

    candidates_neg = candidates[candidates['class'] == 0].copy()
    candidates_neg = candidates_neg[candidates_neg['seriesuid'].astype(str).isin(series_map)].copy()
    candidates_neg = filter_negatives_far_from_annotations(candidates_neg, annotations)
    n_neg = min(len(candidates_neg), len(positives) * NEG_PER_POS)
    negatives = candidates_neg.sample(n=n_neg, random_state=SEED).copy()
    negatives['diameter_mm'] = np.nan
    negatives['label'] = 0
    negatives['source'] = 'candidates_v2_class0' if candidates_path.name == 'candidates_V2.csv' else 'candidates_class0'

    index = pd.concat([
        positives[['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'label', 'source']],
        negatives[['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'label', 'source']],
    ], ignore_index=True)
    index = add_geometry(index, series_map)
    index = index[index['center_inside_volume']].copy().reset_index(drop=True)
    split_map = assign_series_split(index['seriesuid'])
    index['split'] = index['seriesuid'].astype(str).map(split_map)
    index['patch_size_vox'] = PATCH_SIZE_VOX
    index['sample_id'] = [f'luna16_{i:06d}' for i in range(len(index))]

    # Stable column order.
    first_cols = [
        'sample_id', 'split', 'label', 'source', 'seriesuid', 'subset',
        'coordX', 'coordY', 'coordZ', 'voxel_x', 'voxel_y', 'voxel_z',
        'diameter_mm', 'patch_size_vox', 'mhd_path', 'raw_path', 'seg_mhd_path',
        'dim_x', 'dim_y', 'dim_z', 'spacing_x', 'spacing_y', 'spacing_z',
    ]
    other_cols = [c for c in index.columns if c not in first_cols]
    index = index[first_cols + other_cols]

    out_csv = OUT_DIR / 'luna16_patch_index_pos1_neg3_seed17.csv'
    index.to_csv(out_csv, index=False)

    summary = {
        'luna_root': str(LUNA_ROOT),
        'out_csv': str(out_csv),
        'num_series_with_scans': len(series_map),
        'num_annotations_available': int(len(annotations)),
        'num_positive_samples': int((index['label'] == 1).sum()),
        'num_negative_samples': int((index['label'] == 0).sum()),
        'num_total_samples': int(len(index)),
        'neg_per_pos_target': NEG_PER_POS,
        'min_neg_distance_mm': MIN_NEG_DISTANCE_MM,
        'patch_size_vox': PATCH_SIZE_VOX,
        'seed': SEED,
        'split_counts': index.groupby(['split', 'label']).size().unstack(fill_value=0).to_dict(),
        'series_split_counts': pd.Series(split_map).value_counts().to_dict(),
        'candidates_source': str(candidates_path),
    }
    out_json = OUT_DIR / 'luna16_patch_index_summary.json'
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
