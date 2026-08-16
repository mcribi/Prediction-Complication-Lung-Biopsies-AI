#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.networks.nets import resnet18
from sklearn.metrics import average_precision_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
DL_COMMON = Path('/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/una_validacion_solo')
for path in [HERE, DL_COMMON]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import phase_a_serial_common as dl
from pu3d_core import (
    binary_metrics,
    choose_threshold,
    elkan_noto_correct,
    estimate_c,
    estimate_class_prior,
    load_outer_fold,
    make_inner_split,
    nnpu_loss,
    set_seed,
)

PREPROCESSING = 'resize_cube64'
INPUT_NAME = 'ct_lung_nodule'
INPUT_CFG = {'name': INPUT_NAME, 'channels': ['ct_lung', 'lung', 'nodule']}
MODEL_NAME = 'resnet18'
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
PATIENCE = 10
NUM_WORKERS = 4
USE_AMP = True


class BinaryVolumeDataset(dl.MultiInputVolumeDataset):
    def __init__(self, labels_frame, patient_ids, target):
        frame = labels_frame.copy()
        frame['binary_target'] = frame[target].astype(np.float32)
        original = dl.TARGET_LABELS
        dl.TARGET_LABELS = ['binary_target']
        try:
            super().__init__(frame, patient_ids, PREPROCESSING, INPUT_CFG)
        finally:
            dl.TARGET_LABELS = original
        self.target = target

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        img = np.load(self.image_dir / f'{pid}.npy')
        img_channels = dl.preprocess_channels(img)
        masks = {key: self._load_mask(key, pid) for key in dl.required_masks_for_input(self.input_cfg)}
        channels = []
        for channel in self.input_cfg['channels']:
            if channel == 'ct_lung':
                channels.extend([image_channel * masks['lung'] for image_channel in img_channels])
            elif channel == 'ct_nodule':
                channels.extend([image_channel * masks['nodule'] for image_channel in img_channels])
            else:
                channels.append(masks[channel])
        x = np.stack(channels, axis=0).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, 0.0, 1.0)
        y = float(self.df.loc[pid, self.target])
        return torch.tensor(x), torch.tensor(y, dtype=torch.float32), pid


def build_model(in_channels: int) -> nn.Module:
    model = resnet18(spatial_dims=3, n_input_channels=in_channels, num_classes=1)
    model.fc = nn.Sequential(nn.Dropout(DROPOUT), model.fc)
    return model


def make_loader(dataset, batch_size=BATCH_SIZE, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=True)


def predict(model, loader, device):
    model.eval()
    probabilities, labels, patient_ids = [], [], []
    with torch.no_grad():
        for xb, yb, pids in loader:
            xb = xb.to(device, non_blocking=True)
            with autocast(enabled=(USE_AMP and device.type == 'cuda')):
                logits = model(xb).reshape(-1)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(yb.numpy().reshape(-1))
            patient_ids.extend(map(str, pids))
    return np.concatenate(probabilities), np.concatenate(labels).astype(int), patient_ids


def safe_ap(labels, probabilities):
    return float(average_precision_score(labels, probabilities)) if len(np.unique(labels)) == 2 else -np.inf


def train_supervised(train_ds, val_ds, device, max_epochs, seed):
    set_seed(seed)
    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds)
    sample, _, _ = train_ds[0]
    model = build_model(int(sample.shape[0])).to(device)
    labels = np.array([float(train_ds.df.loc[pid, train_ds.target]) for pid in train_ds.patient_ids])
    positives = labels.sum()
    pos_weight = torch.tensor([(len(labels) - positives) / max(positives, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = GradScaler(enabled=(USE_AMP and device.type == 'cuda'))
    best_state, best_ap, best_epoch, patience_count = None, -np.inf, None, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(USE_AMP and device.type == 'cuda')):
                logits = model(xb).reshape(-1)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        val_prob, val_y, _ = predict(model, val_loader, device)
        val_ap = safe_ap(val_y, val_prob)
        scheduler.step(val_ap)
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'val_average_precision': val_ap})
        print(f'supervised epoch={epoch:02d} loss={np.mean(losses):.5f} val_ap={val_ap:.5f}', flush=True)
        if val_ap > best_ap + 1e-4:
            best_state, best_ap, best_epoch = deepcopy(model.state_dict()), val_ap, epoch
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError('No supervised checkpoint selected')
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), int(best_epoch), float(best_ap)


def build_pu_loaders(train_ds):
    labels = np.array([float(train_ds.df.loc[pid, train_ds.target]) for pid in train_ds.patient_ids])
    positive_idx = np.flatnonzero(labels == 1).tolist()
    unlabeled_idx = np.flatnonzero(labels == 0).tolist()
    if not positive_idx or not unlabeled_idx:
        raise ValueError('nnPU requires positive and unlabeled training samples')
    positive_loader = make_loader(Subset(train_ds, positive_idx), batch_size=min(BATCH_SIZE // 2, len(positive_idx)), shuffle=True)
    unlabeled_loader = make_loader(Subset(train_ds, unlabeled_idx), batch_size=min(BATCH_SIZE, len(unlabeled_idx)), shuffle=True)
    return positive_loader, unlabeled_loader


def train_nnpu(train_ds, val_ds, device, max_epochs, seed, class_prior):
    set_seed(seed)
    positive_loader, unlabeled_loader = build_pu_loaders(train_ds)
    val_loader = make_loader(val_ds)
    sample, _, _ = train_ds[0]
    model = build_model(int(sample.shape[0])).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = GradScaler(enabled=(USE_AMP and device.type == 'cuda'))
    best_state, best_ap, best_epoch, patience_count = None, -np.inf, None, 0
    history = []
    steps = max(len(positive_loader), len(unlabeled_loader))
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        positive_iter, unlabeled_iter = cycle(positive_loader), cycle(unlabeled_loader)
        for _ in range(steps):
            xp, _, _ = next(positive_iter)
            xu, _, _ = next(unlabeled_iter)
            x = torch.cat([xp, xu], dim=0).to(device, non_blocking=True)
            observed = torch.cat([torch.ones(len(xp)), torch.zeros(len(xu))]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(USE_AMP and device.type == 'cuda')):
                logits = model(x).reshape(-1)
                loss = nnpu_loss(logits, observed, class_prior)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        val_prob, val_y, _ = predict(model, val_loader, device)
        val_ap = safe_ap(val_y, val_prob)
        scheduler.step(val_ap)
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'val_average_precision': val_ap})
        print(f'nnpu epoch={epoch:02d} loss={np.mean(losses):.5f} val_ap={val_ap:.5f}', flush=True)
        if val_ap > best_ap + 1e-4:
            best_state, best_ap, best_epoch = deepcopy(model.state_dict()), val_ap, epoch
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError('No nnPU checkpoint selected')
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), int(best_epoch), float(best_ap)


def save_predictions(path, fold, target, formulation, patient_ids, y_true, scores, threshold):
    pd.DataFrame({
        'outer_fold': fold,
        'patient_id': patient_ids,
        'target': target,
        'formulation': formulation,
        'y_true': y_true.astype(int),
        'y_score': scores,
        'threshold': float(threshold),
        'y_pred': (scores >= threshold).astype(int),
    }).to_csv(path, index=False)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    device = torch.device('cuda')
    labels = pd.read_csv(args.labels_csv)
    labels['patient_id'] = labels['patient_id'].astype(str)
    if labels['patient_id'].duplicated().any():
        raise ValueError('Duplicate patient IDs in labels')
    train_outer, test_outer = load_outer_fold(Path(args.folds_csv), args.fold)
    cohort = set(labels['patient_id'])
    if not (set(train_outer) | set(test_outer)).issubset(cohort):
        raise ValueError('Fold patients missing from labels')
    train_inner, val_inner = make_inner_split(labels, train_outer, args.target, args.seed + args.fold, args.val_ratio)
    if set(test_outer) & (set(train_inner) | set(val_inner)):
        raise RuntimeError('External test leakage')

    out = Path(args.output_dir) / f'{args.target}__fold_{args.fold}'
    out.mkdir(parents=True, exist_ok=True)
    train_ds = BinaryVolumeDataset(labels, train_inner, args.target)
    val_ds = BinaryVolumeDataset(labels, val_inner, args.target)
    test_ds = BinaryVolumeDataset(labels, test_outer, args.target)
    val_loader, test_loader = make_loader(val_ds), make_loader(test_ds)

    split_rows = ([{'split': 'train_inner', 'patient_id': p} for p in train_inner]
                  + [{'split': 'val_inner', 'patient_id': p} for p in val_inner]
                  + [{'split': 'test_outer', 'patient_id': p} for p in test_outer])
    pd.DataFrame(split_rows).to_csv(out / 'split_assignments.csv', index=False)

    supervised, history, sup_epoch, sup_ap = train_supervised(train_ds, val_ds, device, args.max_epochs, args.seed + 100 * args.fold)
    history.to_csv(out / 'history_supervised.csv', index=False)
    torch.save(supervised.state_dict(), out / 'best_supervised.pt')
    val_raw, val_y, _ = predict(supervised, val_loader, device)
    test_raw, test_y, test_ids = predict(supervised, test_loader, device)
    c_value = estimate_c(val_y, val_raw)
    class_prior = estimate_class_prior(labels.set_index('patient_id').loc[train_outer, args.target].values, c_value)

    sup_threshold, _ = choose_threshold(val_y, val_raw)
    sup_metrics = binary_metrics(test_y, test_raw >= sup_threshold, test_raw)
    save_predictions(out / 'predictions_supervised.csv', args.fold, args.target, 'supervised', test_ids, test_y, test_raw, sup_threshold)

    val_en = elkan_noto_correct(val_raw, c_value)
    test_en = elkan_noto_correct(test_raw, c_value)
    en_threshold, _ = choose_threshold(val_y, val_en)
    en_metrics = binary_metrics(test_y, test_en >= en_threshold, test_en)
    save_predictions(out / 'predictions_elkan_noto.csv', args.fold, args.target, 'elkan_noto', test_ids, test_y, test_en, en_threshold)

    del supervised
    torch.cuda.empty_cache()
    nnpu, nn_history, nn_epoch, nn_ap = train_nnpu(train_ds, val_ds, device, args.max_epochs, args.seed + 1000 + 100 * args.fold, class_prior)
    nn_history.to_csv(out / 'history_nnpu.csv', index=False)
    torch.save(nnpu.state_dict(), out / 'best_nnpu.pt')
    val_nn, val_nn_y, _ = predict(nnpu, val_loader, device)
    test_nn, test_nn_y, test_nn_ids = predict(nnpu, test_loader, device)
    nn_threshold, _ = choose_threshold(val_nn_y, val_nn)
    nn_metrics = binary_metrics(test_nn_y, test_nn >= nn_threshold, test_nn)
    save_predictions(out / 'predictions_nnpu.csv', args.fold, args.target, 'nnpu', test_nn_ids, test_nn_y, test_nn, nn_threshold)

    summary = {
        'status': 'ok', 'target': args.target, 'outer_fold': args.fold,
        'n_train_inner': len(train_inner), 'n_val_inner': len(val_inner), 'n_test_outer': len(test_outer),
        'observed_positive_train_outer': int(labels.set_index('patient_id').loc[train_outer, args.target].sum()),
        'c_value': c_value, 'class_prior': class_prior,
        'supervised_best_epoch': sup_epoch, 'supervised_best_val_ap': sup_ap, 'supervised_threshold': sup_threshold,
        'nnpu_best_epoch': nn_epoch, 'nnpu_best_val_ap': nn_ap, 'nnpu_threshold': nn_threshold,
        'elkan_noto_threshold': en_threshold,
        'metrics': {'supervised': sup_metrics, 'elkan_noto': en_metrics, 'nnpu': nn_metrics},
        'model': MODEL_NAME, 'preprocessing': PREPROCESSING, 'input': INPUT_NAME, 'batch_size': BATCH_SIZE,
        'seed': args.seed, 'max_epochs': args.max_epochs, 'gpu': torch.cuda.get_device_name(0),
    }
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['Hemorragia', 'Neumotórax'], required=True)
    parser.add_argument('--fold', type=int, choices=range(1, 6), required=True)
    parser.add_argument('--labels-csv', required=True)
    parser.add_argument('--folds-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
