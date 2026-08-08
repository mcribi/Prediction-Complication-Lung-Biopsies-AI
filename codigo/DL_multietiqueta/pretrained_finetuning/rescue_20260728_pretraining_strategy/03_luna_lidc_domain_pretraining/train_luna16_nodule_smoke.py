from __future__ import annotations

import argparse
import json
import random
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

DEFAULT_INDEX_CSV = Path('/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/pretrained_finetuning/rescue_20260728_pretraining_strategy/03_luna_lidc_domain_pretraining/luna16_patch_index/luna16_patch_index_pos1_neg3_seed17.csv')
DEFAULT_OUTPUT_ROOT = Path('/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/pretrained_finetuning/runs/luna16_nodule_pretraining_smoke')
TARGET_PATCH_SIZE = 64
HU_MIN = -1000.0
HU_MAX = 400.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def crop_patch_zyx(volume: np.ndarray, center_xyz: Tuple[float, float, float], patch_size: int) -> np.ndarray:
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
    patch = (patch - HU_MIN) / (HU_MAX - HU_MIN)
    return patch.astype(np.float32)


class SeriesCache:
    def __init__(self, max_items: int = 2) -> None:
        self.max_items = max_items
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: str) -> np.ndarray:
        path = str(path)
        if path in self.cache:
            arr = self.cache.pop(path)
            self.cache[path] = arr
            return arr
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        self.cache[path] = arr
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return arr


class Luna16PatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, patch_size: int = TARGET_PATCH_SIZE, cache_items: int = 2) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.patch_size = patch_size
        self.cache = SeriesCache(max_items=cache_items)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        volume = self.cache.get(row['mhd_path'])
        patch = crop_patch_zyx(volume, (row['voxel_x'], row['voxel_y'], row['voxel_z']), self.patch_size)
        patch = normalize_hu(patch)
        if not np.isfinite(patch).all():
            raise RuntimeError(f'Non finite patch in sample {row["sample_id"]}')
        x = torch.from_numpy(patch[None, ...])
        y = torch.tensor(float(row['label']), dtype=torch.float32)
        return x, y, str(row['sample_id'])


class Tiny3DCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x).flatten(1)
        return self.head(z).squeeze(1)


def balanced_subset(df: pd.DataFrame, split: str, max_samples: int, seed: int) -> pd.DataFrame:
    part = df[df['split'] == split].copy()
    if part.empty:
        raise RuntimeError(f'No rows for split={split}')
    if max_samples <= 0 or len(part) <= max_samples:
        return part.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    half = max_samples // 2
    pos = part[part['label'] == 1].sample(n=min(half, int((part['label'] == 1).sum())), random_state=seed)
    neg_n = max_samples - len(pos)
    neg = part[part['label'] == 0].sample(n=min(neg_n, int((part['label'] == 0).sum())), random_state=seed + 1)
    out = pd.concat([pos, neg], ignore_index=True)
    if len(out) < max_samples:
        remain = part.drop(out.index, errors='ignore')
        extra = part[~part['sample_id'].isin(out['sample_id'])].sample(n=min(max_samples - len(out), len(part) - len(out)), random_state=seed + 2)
        out = pd.concat([out, extra], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)


def validate_index(df: pd.DataFrame) -> Dict:
    required = ['sample_id', 'split', 'label', 'seriesuid', 'voxel_x', 'voxel_y', 'voxel_z', 'mhd_path', 'raw_path', 'center_inside_volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f'Missing index columns: {missing}')
    missing_paths = []
    for col in ['mhd_path', 'raw_path']:
        bad = [p for p in df[col].dropna().unique()[:200] if not Path(str(p)).exists()]
        missing_paths.extend(bad)
    if missing_paths:
        raise RuntimeError(f'Missing image paths, first examples: {missing_paths[:5]}')
    return {
        'num_rows': int(len(df)),
        'num_series': int(df['seriesuid'].nunique()),
        'split_label_counts': {str(k): int(v) for k, v in df.groupby(['split', 'label']).size().items()},
        'center_inside_volume_counts': {str(k): int(v) for k, v in df['center_inside_volume'].value_counts(dropna=False).items()},
    }


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> Dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total = 0
    correct = 0
    for x, y, _sample_id in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
        probs = torch.sigmoid(logits.detach())
        pred = (probs >= 0.5).float()
        correct += int((pred.cpu() == y.cpu()).sum().item())
        total += int(y.numel())
        total_loss += float(loss.detach().cpu()) * int(y.numel())
    return {'loss': total_loss / max(total, 1), 'accuracy': correct / max(total, 1), 'num_samples': total}


def main() -> int:
    parser = argparse.ArgumentParser(description='LUNA16 patch dataset smoke training for nodule/non-nodule proxy task.')
    parser.add_argument('--index-csv', type=Path, default=DEFAULT_INDEX_CSV)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--max-train-samples', type=int, default=64)
    parser.add_argument('--max-val-samples', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--run-tag', default='')
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) or args.device == 'cuda' else 'cpu')

    df = pd.read_csv(args.index_csv)
    index_report = validate_index(df)
    train_df = balanced_subset(df, 'train', args.max_train_samples, args.seed)
    val_df = balanced_subset(df, 'val', args.max_val_samples, args.seed + 100)

    run_name = datetime.now().strftime('%Y%m%d_%H%M%S') + (f'_{args.run_tag}' if args.run_tag else '')
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(run_dir / 'train_subset.csv', index=False)
    val_df.to_csv(run_dir / 'val_subset.csv', index=False)

    train_loader = DataLoader(Luna16PatchDataset(train_df), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(Luna16PatchDataset(val_df), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    model = Tiny3DCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        row = {'epoch': epoch, **{f'train_{k}': v for k, v in train_metrics.items()}, **{f'val_{k}': v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    pd.DataFrame(history).to_csv(run_dir / 'history.csv', index=False)
    torch.save({'model_state_dict': model.state_dict(), 'args': vars(args)}, run_dir / 'tiny3dcnn_smoke_checkpoint.pt')
    report = {
        'status': 'ok',
        'purpose': 'dataset_dataloader_forward_backward_smoke',
        'index_csv': str(args.index_csv),
        'run_dir': str(run_dir),
        'device': str(device),
        'cuda_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'elapsed_seconds': round(time.time() - start, 3),
        'index_report': index_report,
        'train_subset': {'num_samples': int(len(train_df)), 'label_counts': {str(k): int(v) for k, v in train_df['label'].value_counts().items()}},
        'val_subset': {'num_samples': int(len(val_df)), 'label_counts': {str(k): int(v) for k, v in val_df['label'].value_counts().items()}},
        'last_metrics': history[-1] if history else None,
        'outputs': ['train_subset.csv', 'val_subset.csv', 'history.csv', 'tiny3dcnn_smoke_checkpoint.pt', 'smoke_report.json'],
    }
    (run_dir / 'smoke_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
