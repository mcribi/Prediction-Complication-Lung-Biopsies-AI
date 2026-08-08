from __future__ import annotations

import argparse
import csv
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
from monai.networks.nets import resnet18, resnet34
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

DEFAULT_INDEX_CSV = Path('/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/pretrained_finetuning/rescue_20260728_pretraining_strategy/03_luna_lidc_domain_pretraining/luna16_patch_index/luna16_patch_index_pos1_neg3_seed17.csv')
DEFAULT_OUTPUT_ROOT = Path('/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/pretrained_finetuning/runs/luna16_nodule_pretraining_real')
TARGET_PATCH_SIZE = 64
HU_MIN = -1000.0
HU_MAX = 400.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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
            raise RuntimeError(f'Non finite patch in sample {row[sample_id]}')
        x = torch.from_numpy(patch[None, ...])
        y = torch.tensor(float(row['label']), dtype=torch.float32)
        return x, y, str(row['sample_id'])


def balanced_subset(df: pd.DataFrame, split: str, max_samples: int, seed: int) -> pd.DataFrame:
    part = df[df['split'] == split].copy()
    if part.empty:
        raise RuntimeError(f'No rows for split={split}')
    if max_samples <= 0 or len(part) <= max_samples:
        return part.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    half = max_samples // 2
    pos_pool = part[part['label'] == 1]
    neg_pool = part[part['label'] == 0]
    pos = pos_pool.sample(n=min(half, len(pos_pool)), random_state=seed)
    neg_n = max_samples - len(pos)
    neg = neg_pool.sample(n=min(neg_n, len(neg_pool)), random_state=seed + 1)
    out = pd.concat([pos, neg], ignore_index=True)
    if len(out) < max_samples:
        extra_pool = part[~part['sample_id'].isin(out['sample_id'])]
        extra = extra_pool.sample(n=min(max_samples - len(out), len(extra_pool)), random_state=seed + 2)
        out = pd.concat([out, extra], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)


def validate_index(df: pd.DataFrame) -> Dict:
    required = ['sample_id', 'split', 'label', 'seriesuid', 'voxel_x', 'voxel_y', 'voxel_z', 'mhd_path', 'raw_path', 'center_inside_volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f'Missing index columns: {missing}')
    missing_paths = []
    for col in ['mhd_path', 'raw_path']:
        for p in df[col].dropna().unique()[:500]:
            if not Path(str(p)).exists():
                missing_paths.append(str(p))
    if missing_paths:
        raise RuntimeError(f'Missing image paths, first examples: {missing_paths[:5]}')
    return {
        'num_rows': int(len(df)),
        'num_series': int(df['seriesuid'].nunique()),
        'split_label_counts': {str(k): int(v) for k, v in df.groupby(['split', 'label']).size().items()},
        'center_inside_volume_counts': {str(k): int(v) for k, v in df['center_inside_volume'].value_counts(dropna=False).items()},
    }


def make_model(architecture: str) -> nn.Module:
    if architecture == 'resnet18':
        model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=1, shortcut_type='A')
    elif architecture == 'resnet34':
        model = resnet34(spatial_dims=3, n_input_channels=1, num_classes=1, shortcut_type='A')
    else:
        raise ValueError(f'Unsupported architecture: {architecture}')
    return model


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        out['roc_auc'] = float(roc_auc_score(y_true, y_prob))
    else:
        out['roc_auc'] = float('nan')
    return out


def run_epoch(model, loader, criterion, optimizer, device, scaler, use_amp: bool, train: bool) -> Dict[str, float]:
    model.train(train)
    losses, probs, targets = [], [], []
    for x, y, _sample_id in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with autocast(enabled=(use_amp and device.type == 'cuda')):
                logits = model(x).squeeze(1)
                loss = criterion(logits, y)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        losses.append(float(loss.detach().cpu()) * int(y.numel()))
        probs.append(torch.sigmoid(logits.detach()).cpu().numpy())
        targets.append(y.detach().cpu().numpy())
    y_prob = np.concatenate(probs, axis=0)
    y_true = np.concatenate(targets, axis=0).astype(int)
    metrics = binary_metrics(y_true, y_prob)
    metrics['loss'] = float(np.sum(losses) / max(1, len(y_true)))
    metrics['num_samples'] = int(len(y_true))
    return metrics


def save_checkpoint(path: Path, model: nn.Module, optimizer, epoch: int, metrics: Dict, args, setup: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': int(epoch),
        'architecture': args.architecture,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': metrics,
        'setup': setup,
    }, path)


def main() -> int:
    parser = argparse.ArgumentParser(description='Bounded LUNA16 supervised nodule pretraining for TFM transfer experiments.')
    parser.add_argument('--index-csv', type=Path, default=DEFAULT_INDEX_CSV)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--architecture', choices=['resnet18', 'resnet34'], default='resnet34')
    parser.add_argument('--max-train-samples', type=int, default=2048, help='0 means all train rows from the prepared index')
    parser.add_argument('--max-val-samples', type=int, default=512, help='0 means all val rows from the prepared index')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--cache-items', type=int, default=2)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--run-tag', default='bounded_real')
    parser.add_argument('--no-amp', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) or args.device == 'cuda' else 'cpu')
    use_amp = (not args.no_amp) and device.type == 'cuda'

    df = pd.read_csv(args.index_csv)
    index_report = validate_index(df)
    train_df = balanced_subset(df, 'train', args.max_train_samples, args.seed)
    val_df = balanced_subset(df, 'val', args.max_val_samples, args.seed + 100)

    run_name = datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{args.architecture}_{args.run_tag}'
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    train_df.to_csv(run_dir / 'train_subset.csv', index=False)
    val_df.to_csv(run_dir / 'val_subset.csv', index=False)

    train_loader = DataLoader(Luna16PatchDataset(train_df, cache_items=args.cache_items), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(Luna16PatchDataset(val_df, cache_items=args.cache_items), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    model = make_model(args.architecture).to(device)
    pos = max(1, int((train_df['label'] == 1).sum()))
    neg = max(1, int((train_df['label'] == 0).sum()))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = GradScaler(enabled=use_amp)

    setup = {
        'status': 'running',
        'purpose': 'bounded_luna16_supervised_nodule_pretraining_for_tfm_transfer',
        'index_csv': str(args.index_csv),
        'run_dir': str(run_dir),
        'architecture': args.architecture,
        'device': str(device),
        'cuda_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'seed': args.seed,
        'max_train_samples': args.max_train_samples,
        'max_val_samples': args.max_val_samples,
        'epochs': args.epochs,
        'patience': args.patience,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'pos_weight': float(pos_weight.detach().cpu().item()),
        'use_amp': bool(use_amp),
        'index_report': index_report,
        'train_subset': {'num_samples': int(len(train_df)), 'label_counts': {str(k): int(v) for k, v in train_df['label'].value_counts().items()}},
        'val_subset': {'num_samples': int(len(val_df)), 'label_counts': {str(k): int(v) for k, v in val_df['label'].value_counts().items()}},
        'transfer_note': 'Load model_state_dict into MONAI resnet34/resnet18 and skip fc for TFM multilabel fine tuning. First conv is one-channel and can be adapted by existing loader.',
    }
    (run_dir / 'setup.json').write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding='utf-8')

    history_path = run_dir / 'history.csv'
    best_score = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []
    start = time.time()
    with history_path.open('w', newline='') as f:
        writer = None
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp, train=True)
            val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, scaler, use_amp, train=False)
            scheduler.step(val_metrics['f1'])
            row = {'epoch': epoch}
            for prefix, metrics in [('train', train_metrics), ('val', val_metrics)]:
                for k, v in metrics.items():
                    row[f'{prefix}_{k}'] = v
            row['lr'] = float(optimizer.param_groups[0]['lr'])
            history.append(row)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)
            score = float(val_metrics['f1'])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(run_dir / 'best_luna16_pretrain_checkpoint.pt', model, optimizer, epoch, val_metrics, args, setup)
                torch.save(model.state_dict(), run_dir / 'best_encoder_state_dict.pt')
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(json.dumps({'event': 'early_stopping', 'epoch': epoch, 'best_epoch': best_epoch, 'best_val_f1': best_score}, ensure_ascii=False), flush=True)
                    break

    report = dict(setup)
    report.update({
        'status': 'ok',
        'elapsed_seconds': round(time.time() - start, 2),
        'epochs_ran': int(len(history)),
        'best_epoch': int(best_epoch),
        'best_val_f1': float(best_score),
        'best_val_metrics': {k.replace('val_', ''): v for k, v in history[best_epoch - 1].items() if k.startswith('val_')} if best_epoch > 0 else {},
        'last_metrics': history[-1] if history else {},
        'outputs': ['train_subset.csv', 'val_subset.csv', 'history.csv', 'setup.json', 'best_luna16_pretrain_checkpoint.pt', 'best_encoder_state_dict.pt', 'pretrain_report.json'],
    })
    (run_dir / 'pretrain_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
