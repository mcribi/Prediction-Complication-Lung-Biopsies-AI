from __future__ import annotations

import argparse
import json
import gc
import random
import sys
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import resnet18, resnet34
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
DL_ROOT = SCRIPT_DIR.parents[2]
VAL_DIR = DL_ROOT / 'val_externa_interna'
PHASE_A_DIR = DL_ROOT / 'una_validacion_solo'
for extra_path in [SCRIPT_DIR, DL_ROOT, VAL_DIR, PHASE_A_DIR]:
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import phase_a_serial_common as base
import dl_nested_cv_common as nested

PROJECT_ROOT = base.PROJECT_ROOT
OUTPUT_ROOT = PROJECT_ROOT / 'codigo' / 'DL_multietiqueta' / 'pretrained_finetuning' / 'runs'
WEIGHTS_ROOT = PROJECT_ROOT / 'pretrained_weights'
TARGET_LABELS = list(base.TARGET_LABELS)
SEED = base.SEED
N_OUTER_FOLDS = 5
INNER_VAL_RATIO = 0.2
MAX_EPOCHS = base.MAX_EPOCHS
PATIENCE = base.PATIENCE
DROPOUT = base.DROPOUT
WEIGHT_DECAY = base.WEIGHT_DECAY
NUM_WORKERS = base.NUM_WORKERS
VAL_THRESHOLD = base.VAL_THRESHOLD
USE_AMP = base.USE_AMP
ENCODER_LR = 3e-5
HEAD_LR = 1e-4

PILOT_CONFIGS = [
    {
        'config_id': 'resnet34_resize_small_hu_m300_1400__ct_lung_nodule_vessels__bs4',
        'model_name': 'resnet34',
        'preprocessing': 'resize_small_hu_m300_1400',
        'input_name': 'ct_lung_nodule_vessels',
        'batch_size': 4,
        'scratch_oof_f1_micro': 0.5465116279069767,
        'scratch_oof_f1_macro': 0.532794713754234,
    },
    {
        'config_id': 'resnet34_resize_small_multiwindowing_separadas__nodule_only_masked_ct__bs4',
        'model_name': 'resnet34',
        'preprocessing': 'resize_small_multiwindowing_separadas',
        'input_name': 'nodule_only_masked_ct',
        'batch_size': 4,
        'scratch_oof_f1_micro': 0.5338645418326693,
        'scratch_oof_f1_macro': 0.5209313691235112,
    },
]
FREEZE_EPOCHS = 5
THRESHOLD_GRID = [round(x, 2) for x in np.linspace(0.1, 0.9, 17)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def adapt_first_conv_weight(weight: torch.Tensor, in_channels: int) -> torch.Tensor:
    if weight.ndim != 5:
        return weight
    old_channels = int(weight.shape[1])
    if old_channels == in_channels:
        return weight
    if old_channels == 1:
        return weight.repeat(1, in_channels, 1, 1, 1) / float(in_channels)
    if old_channels > in_channels:
        return weight[:, :in_channels].contiguous()
    reps = int(np.ceil(in_channels / old_channels))
    expanded = weight.repeat(1, reps, 1, 1, 1)[:, :in_channels].contiguous()
    return expanded * (float(old_channels) / float(in_channels))


def unwrap_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ['state_dict', 'model_state_dict', 'model', 'net']:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError('Unsupported checkpoint format')


def clean_key(key: str) -> str:
    for prefix in ['module.', 'model.', 'net.']:
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def load_matching_weights(model: nn.Module, weight_path: Path, first_conv_keys: List[str], skip_prefixes: Tuple[str, ...]) -> Dict:
    checkpoint = torch.load(weight_path, map_location='cpu')
    source = unwrap_state_dict(checkpoint)
    target = model.state_dict()
    adapted = {}
    skipped_shape = []
    skipped_name = []
    loaded = []
    adapted_first_conv = []

    for raw_key, value in source.items():
        key = clean_key(raw_key)
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped_name.append(key)
            continue
        if key not in target:
            skipped_name.append(key)
            continue
        tensor = value.detach().cpu() if torch.is_tensor(value) else torch.tensor(value)
        if key in first_conv_keys and tensor.shape != target[key].shape:
            tensor = adapt_first_conv_weight(tensor, int(target[key].shape[1]))
            adapted_first_conv.append({'key': key, 'source_shape': list(value.shape), 'target_shape': list(target[key].shape)})
        if tensor.shape == target[key].shape:
            adapted[key] = tensor
            loaded.append(key)
        else:
            skipped_shape.append({'key': key, 'source_shape': list(tensor.shape), 'target_shape': list(target[key].shape)})

    missing, unexpected = model.load_state_dict(adapted, strict=False)
    report = {
        'weight_path': str(weight_path),
        'source_keys': int(len(source)),
        'loaded_keys': int(len(loaded)),
        'loaded_ratio_vs_source': float(len(loaded) / max(1, len(source))),
        'adapted_first_conv': adapted_first_conv,
        'skipped_by_name_count': int(len(skipped_name)),
        'skipped_by_shape': skipped_shape[:100],
        'missing_after_partial_load_count': int(len(missing)),
        'unexpected_after_partial_load_count': int(len(unexpected)),
        'missing_after_partial_load_sample': list(missing)[:100],
        'unexpected_after_partial_load_sample': list(unexpected)[:100],
    }
    return report


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError(f'expected 5D input, got {input.dim()}D')

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act='relu'):
        super().__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)
        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError(f'Unsupported activation: {act}')

    def forward(self, x):
        return self.activation(self.bn1(self.conv1(x)))


def make_nconv(in_channel, depth, act='relu'):
    layer1 = LUConv(in_channel, 32 * (2 ** depth), act)
    layer2 = LUConv(32 * (2 ** depth), 32 * (2 ** depth) * 2, act)
    return nn.Sequential(layer1, layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channel, depth, act='relu'):
        super().__init__()
        self.ops = make_nconv(in_channel, depth, act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
        else:
            out = self.maxpool(self.ops(x))
        return out


class GenesisEncoderClassifier(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down_tr64 = DownTransition(in_channels, 0, 'relu')
        self.down_tr128 = DownTransition(64, 1, 'relu')
        self.down_tr256 = DownTransition(128, 2, 'relu')
        self.down_tr512 = DownTransition(256, 3, 'relu')
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(512, out_channels))

    def forward(self, x):
        x = self.down_tr64(x)
        x = self.down_tr128(x)
        x = self.down_tr256(x)
        x = self.down_tr512(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)



class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        loss = targets * torch.log(xs_pos.clamp(min=self.eps)) + (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        pt = xs_pos * targets + xs_neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        loss = loss * torch.pow(1.0 - pt, gamma)
        return -loss.mean()

def build_pretrained_model(pretraining: str, in_channels: int, out_channels: int, report_dir: Path) -> nn.Module:
    if pretraining == 'scratch_asl':
        model = nested.build_model('resnet34', in_channels, out_channels)
        report = {'pretraining': pretraining, 'model': 'resnet34', 'in_channels': int(in_channels), 'out_channels': int(out_channels), 'weights': 'scratch'}
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / 'pretrained_loading_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return model
    if pretraining == 'medicalnet':
        model = resnet34(spatial_dims=3, n_input_channels=in_channels, num_classes=out_channels, shortcut_type='A')
        model.fc = nn.Sequential(nn.Dropout(DROPOUT), model.fc)
        weight_path = WEIGHTS_ROOT / 'medicalnet' / 'resnet_34_23dataset.pth'
        if not weight_path.exists():
            raise FileNotFoundError(f'MedicalNet ResNet34 weights not found: {weight_path}')
        report = load_matching_weights(model, weight_path, ['conv1.weight'], skip_prefixes=('fc',))
    elif pretraining == 'genesis_chest_ct':
        model = GenesisEncoderClassifier(in_channels=in_channels, out_channels=out_channels)
        weight_path = WEIGHTS_ROOT / 'models_genesis' / 'Genesis_Chest_CT.pt'
        report = load_matching_weights(model, weight_path, ['down_tr64.ops.0.conv1.weight'], skip_prefixes=('up_tr', 'out_tr'))
    else:
        raise ValueError(f'Unsupported pretraining: {pretraining}')
    report['pretraining'] = pretraining
    report['in_channels'] = int(in_channels)
    report['out_channels'] = int(out_channels)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'pretrained_loading_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return model


def get_head_and_encoder_params(model: nn.Module, pretraining: str):
    if pretraining == 'scratch_asl':
        return list(model.parameters()), []
    if pretraining == 'medicalnet':
        head_params = list(model.fc.parameters())
    else:
        head_params = list(model.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    encoder_params = [p for p in model.parameters() if id(p) not in head_ids]
    return encoder_params, head_params


def set_encoder_trainable(model: nn.Module, pretraining: str, trainable: bool) -> None:
    if pretraining == 'scratch_asl':
        for p in model.parameters():
            p.requires_grad = True
        return
    encoder_params, _ = get_head_and_encoder_params(model, pretraining)
    for p in encoder_params:
        p.requires_grad = trainable


def make_optimizer(model: nn.Module, pretraining: str):
    if pretraining == 'scratch_asl':
        return AdamW(model.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    encoder_params, head_params = get_head_and_encoder_params(model, pretraining)
    return AdamW([
        {'params': encoder_params, 'lr': ENCODER_LR},
        {'params': head_params, 'lr': HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)


def compute_metrics_with_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> Dict[str, float]:
    # Reuse the project metric function by binarizing probabilities through per-label shifted thresholds.
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(int)
    shifted_prob = y_pred.astype(float)
    return base.compute_metrics(y_true, shifted_prob, threshold=0.5)


def tune_thresholds_on_val(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    best_thresholds = np.array([VAL_THRESHOLD] * len(TARGET_LABELS), dtype=float)
    best_metrics = base.compute_metrics(y_true, y_prob, threshold=VAL_THRESHOLD)
    best_key = (best_metrics['f1_micro'], best_metrics['f1_macro'])
    grid = THRESHOLD_GRID
    for t0 in grid:
        for t1 in grid:
            for t2 in grid:
                th = np.array([t0, t1, t2], dtype=float)
                metrics = compute_metrics_with_thresholds(y_true, y_prob, th)
                key = (metrics['f1_micro'], metrics['f1_macro'])
                if key > best_key:
                    best_key = key
                    best_metrics = metrics
                    best_thresholds = th
    return best_thresholds, best_metrics


def safe_outer_split(labels_df: pd.DataFrame):
    splitter = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
    return list(splitter.split(labels_df['patient_id'].tolist(), labels_df['labelset_key'].tolist()))


def safe_inner_split(train_df: pd.DataFrame, fold_seed: int):
    labels = train_df['labelset_key'].astype(str).tolist()
    counts = pd.Series(labels).value_counts()
    train_ids = train_df['patient_id'].astype(str).tolist()
    if len(counts) >= 2 and int(counts.min()) >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=INNER_VAL_RATIO, random_state=fold_seed)
        inner_train_idx, inner_val_idx = next(splitter.split(train_ids, labels))
        strategy = 'StratifiedShuffleSplit'
    else:
        rng = np.random.default_rng(fold_seed)
        indices = np.arange(len(train_df))
        rng.shuffle(indices)
        val_size = max(1, int(round(len(indices) * INNER_VAL_RATIO)))
        val_size = min(val_size, len(indices) - 1)
        inner_val_idx = np.sort(indices[:val_size])
        inner_train_idx = np.sort(indices[val_size:])
        strategy = 'RandomFallback'
    inner_train_ids = train_df.iloc[inner_train_idx]['patient_id'].astype(str).tolist()
    inner_val_ids = train_df.iloc[inner_val_idx]['patient_id'].astype(str).tolist()
    return inner_train_ids, inner_val_ids, strategy


def evaluate_loader(model, loader, criterion, device):
    losses, probs, targets, patient_ids = [], [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb, pids in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with autocast(enabled=(USE_AMP and device.type == 'cuda')):
                logits = model(xb)
                loss = criterion(logits, yb)
            losses.append(loss.item())
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(yb.cpu().numpy())
            patient_ids.extend(pids)
    loss_value = float(np.mean(losses)) if losses else np.nan
    return loss_value, np.concatenate(probs, axis=0), np.concatenate(targets, axis=0), patient_ids, base.compute_metrics(np.concatenate(targets, axis=0), np.concatenate(probs, axis=0), threshold=VAL_THRESHOLD)


def save_prediction_df(out_path: Path, fold_name: str, patient_ids: List[str], prob_matrix: np.ndarray, target_matrix: np.ndarray, thresholds: np.ndarray | None = None) -> None:
    pred_df = pd.DataFrame(prob_matrix, columns=[f'prob_{c}' for c in TARGET_LABELS])
    pred_df.insert(0, 'patient_id', patient_ids)
    pred_df.insert(1, 'fold', fold_name)
    if thresholds is None:
        thresholds = np.array([VAL_THRESHOLD] * len(TARGET_LABELS), dtype=float)
    for idx, label in enumerate(TARGET_LABELS):
        pred_df[f'true_{label}'] = target_matrix[:, idx].astype(int)
        pred_df[f'threshold_{label}'] = float(thresholds[idx])
        pred_df[f'pred_{label}'] = (prob_matrix[:, idx] >= thresholds[idx]).astype(int)
    pred_df.to_csv(out_path, index=False)


def train_one_outer_fold(pretraining, labels_df, preproc_name, input_cfg, batch_size, train_outer_ids, test_outer_ids, fold_dir, device, fold_seed):
    outer_train_df = labels_df[labels_df['patient_id'].isin(train_outer_ids)].copy().reset_index(drop=True)
    inner_train_ids, inner_val_ids, inner_strategy = safe_inner_split(outer_train_df, fold_seed=fold_seed)

    train_ds = base.MultiInputVolumeDataset(labels_df, inner_train_ids, preproc_name, input_cfg)
    val_ds = base.MultiInputVolumeDataset(labels_df, inner_val_ids, preproc_name, input_cfg)
    test_ds = base.MultiInputVolumeDataset(labels_df, test_outer_ids, preproc_name, input_cfg)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    sample_x, _, _ = train_ds[0]
    in_channels = int(sample_x.shape[0])
    grad_accum_steps = max(1, int(nested.grad_accum_steps_from_shape(sample_x.shape, batch_size)))
    model = build_pretrained_model(pretraining, in_channels, len(TARGET_LABELS), fold_dir).to(device)
    set_encoder_trainable(model, pretraining, False)
    freeze_unfreeze_events = [{'epoch': 1, 'encoder_trainable': False, 'note': 'encoder frozen; train classifier head only'}]
    pos_weight = base.compute_pos_weight(labels_df, inner_train_ids).to(device)
    criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05) if pretraining == 'scratch_asl' else nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = make_optimizer(model, pretraining)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler(enabled=(USE_AMP and device.type == 'cuda'))

    best_state, best_epoch = None, None
    best_score = -np.inf
    patience_counter = 0
    history = []
    best_val_probs = best_val_targets = None
    best_val_ids = None
    best_thresholds = np.array([VAL_THRESHOLD] * len(TARGET_LABELS), dtype=float)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == FREEZE_EPOCHS + 1:
            set_encoder_trainable(model, pretraining, True)
            freeze_unfreeze_events.append({'epoch': epoch, 'encoder_trainable': True, 'note': 'encoder unfrozen for fine tuning'})
        model.train()
        train_losses, train_probs, train_targets = [], [], []
        optimizer.zero_grad(set_to_none=True)
        for step_idx, (xb, yb, _) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with autocast(enabled=(USE_AMP and device.type == 'cuda')):
                logits = model(xb)
                loss = criterion(logits, yb)
                loss_for_backward = loss / grad_accum_steps
            scaler.scale(loss_for_backward).backward()
            should_step = (step_idx % grad_accum_steps == 0) or (step_idx == len(train_loader))
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append(loss.item())
            train_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            train_targets.append(yb.detach().cpu().numpy())

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        train_probs = np.concatenate(train_probs, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)
        train_metrics = base.compute_metrics(train_targets, train_probs, threshold=VAL_THRESHOLD)
        val_loss, val_probs, val_targets, val_ids_local, val_metrics_default = evaluate_loader(model, val_loader, criterion, device)
        tuned_thresholds, val_metrics = tune_thresholds_on_val(val_targets, val_probs)
        scheduler.step(val_loss)

        history_row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'encoder_lr': optimizer.param_groups[0]['lr'],
            'head_lr': optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]['lr'],
            'train_f1_micro': train_metrics['f1_micro'],
            'train_f1_macro': train_metrics['f1_macro'],
            'val_f1_micro_default_threshold': val_metrics_default['f1_micro'],
            'val_f1_macro_default_threshold': val_metrics_default['f1_macro'],
            'val_f1_micro': val_metrics['f1_micro'],
            'val_f1_macro': val_metrics['f1_macro'],
            'threshold_Hemorragia': float(tuned_thresholds[0]),
            'threshold_Neumotórax': float(tuned_thresholds[1]),
            'threshold_Sin_complicacion': float(tuned_thresholds[2]),
            'val_subset_accuracy': val_metrics['subset_accuracy'],
            'val_hamming_loss': val_metrics['hamming_loss'],
            'peak_alloc_gb': torch.cuda.max_memory_allocated() / (1024 ** 3) if device.type == 'cuda' else 0.0,
            'peak_reserved_gb': torch.cuda.max_memory_reserved() / (1024 ** 3) if device.type == 'cuda' else 0.0,
        }
        history.append(history_row)

        if val_metrics['f1_micro'] > best_score + 1e-4:
            best_score = val_metrics['f1_micro']
            best_epoch = epoch
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
            best_val_probs = val_probs.copy()
            best_val_targets = val_targets.copy()
            best_val_ids = list(val_ids_local)
            best_thresholds = tuned_thresholds.copy()
        else:
            patience_counter += 1

        print(f'[{pretraining} | {preproc_name} | {input_cfg["name"]} | bs={batch_size}] epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1_micro={val_metrics["f1_micro"]:.4f} val_f1_macro={val_metrics["f1_macro"]:.4f}')
        if patience_counter >= PATIENCE:
            print(f'Early stopping at epoch {epoch}')
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(fold_dir / 'history.csv', index=False)
    split_rows = [{'split': 'train_inner', 'patient_id': pid} for pid in inner_train_ids]
    split_rows += [{'split': 'val_inner', 'patient_id': pid} for pid in inner_val_ids]
    split_rows += [{'split': 'test_outer', 'patient_id': pid} for pid in test_outer_ids]
    pd.DataFrame(split_rows).to_csv(fold_dir / 'split_assignments.csv', index=False)

    if best_state is None:
        raise RuntimeError('No best state was selected during training')
    torch.save(best_state, fold_dir / 'best_model.pt')
    save_prediction_df(fold_dir / 'val_predictions_best_epoch.csv', fold_dir.name, best_val_ids, best_val_probs, best_val_targets, best_thresholds)
    (fold_dir / 'best_thresholds.json').write_text(json.dumps({label: float(best_thresholds[i]) for i, label in enumerate(TARGET_LABELS)}, ensure_ascii=False, indent=2), encoding='utf-8')
    (fold_dir / 'freeze_unfreeze_events.json').write_text(json.dumps(freeze_unfreeze_events, ensure_ascii=False, indent=2), encoding='utf-8')
    model.load_state_dict(best_state)
    test_loss, test_probs, test_targets, test_ids, test_metrics_default = evaluate_loader(model, test_loader, criterion, device)
    test_metrics = compute_metrics_with_thresholds(test_targets, test_probs, best_thresholds)
    save_prediction_df(fold_dir / 'test_predictions_best_epoch.csv', fold_dir.name, test_ids, test_probs, test_targets, best_thresholds)

    return {
        'best_epoch': int(best_epoch) if best_epoch is not None else None,
        'best_val_f1_micro': float(best_score) if best_epoch is not None else None,
        'test_f1_micro': float(test_metrics['f1_micro']),
        'test_f1_macro': float(test_metrics['f1_macro']),
        'test_f1_micro_default_threshold': float(test_metrics_default['f1_micro']),
        'test_f1_macro_default_threshold': float(test_metrics_default['f1_macro']),
        'threshold_Hemorragia': float(best_thresholds[0]),
        'threshold_Neumotórax': float(best_thresholds[1]),
        'threshold_Sin_complicacion': float(best_thresholds[2]),
        'test_subset_accuracy': float(test_metrics['subset_accuracy']),
        'test_hamming_loss': float(test_metrics['hamming_loss']),
        'test_loss': float(test_loss),
        'peak_alloc_gb': float(history_df['peak_alloc_gb'].max()) if not history_df.empty else None,
        'peak_reserved_gb': float(history_df['peak_reserved_gb'].max()) if not history_df.empty else None,
        'num_epochs_ran': int(len(history_df)),
        'in_channels': int(sample_x.shape[0]),
        'grad_accum_steps': int(grad_accum_steps),
        'effective_batch_size': int(batch_size * grad_accum_steps),
        'n_train_inner': int(len(inner_train_ids)),
        'n_val_inner': int(len(inner_val_ids)),
        'n_test_outer': int(len(test_outer_ids)),
        'inner_split_strategy': inner_strategy,
    }


def run_pilot(pretraining: str, run_tag: str = '', config_indices: List[int] | None = None, batch_size_override: int | None = None) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available in this job')
    set_seed(SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    labels_df_all = base.build_multilabel_dataframe(base.CLINICAL_CSV)
    device = torch.device('cuda')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = f'_{run_tag}' if run_tag else ''
    run_root = OUTPUT_ROOT / f'{pretraining}_freeze_unfreeze_threshold_tuning_{timestamp}{suffix}'
    run_root.mkdir(parents=True, exist_ok=True)

    selected_configs = [dict(cfg) for cfg in PILOT_CONFIGS]
    if config_indices is not None:
        selected_configs = [selected_configs[i] for i in config_indices]
    if batch_size_override is not None:
        for cfg in selected_configs:
            cfg['batch_size'] = int(batch_size_override)
            cfg['config_id'] = f"{cfg['model_name']}_{cfg['preprocessing']}__{cfg['input_name']}__bs{cfg['batch_size']}"

    setup = {
        'timestamp': timestamp,
        'pretraining': pretraining,
        'pilot_scope': 'rescue_line2_scratch_asymmetric_loss_with_val_inner_threshold_tuning',
        'configs': selected_configs,
        'batch_size_override': batch_size_override,
        'outer_cv_strategy': 'StratifiedKFold_labelset_key',
        'n_outer_folds': N_OUTER_FOLDS,
        'inner_val_ratio': INNER_VAL_RATIO,
        'threshold_tuning': 'per-label thresholds tuned on val_inner each epoch by F1 micro with F1 macro tie-breaker; test_outer uses thresholds from selected best val_inner epoch',
        'freeze_unfreeze': {'freeze_epochs': FREEZE_EPOCHS, 'initial': 'encoder frozen', 'after': 'full fine tuning'},
        'seed': SEED,
        'max_epochs': MAX_EPOCHS,
        'patience': PATIENCE,
        'encoder_lr': ENCODER_LR,
        'head_lr': HEAD_LR,
        'dropout': DROPOUT,
        'weight_decay': WEIGHT_DECAY,
        'target_labels': TARGET_LABELS,
        'gpu_name': torch.cuda.get_device_name(0),
        'methodology_note': 'Best epoch and thresholds are selected only on val_inner; final fold metrics are computed on test_outer.',
    }
    (run_root / 'setup.json').write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding='utf-8')

    summary_rows = []
    for cfg in selected_configs:
        preprocessings = [row for row in base.discover_preprocessings() if row['name'] == cfg['preprocessing']]
        if len(preprocessings) != 1:
            raise RuntimeError(f"Preprocessing not found: {cfg['preprocessing']}")
        input_cfgs = [icfg for icfg in base.PRIMARY_INPUTS if icfg['name'] == cfg['input_name']]
        if len(input_cfgs) != 1:
            raise RuntimeError(f"Input not found: {cfg['input_name']}")
        input_cfg = input_cfgs[0]
        common_ids = base.collect_common_ids(labels_df_all, preprocessings, [input_cfg])
        labels_df = labels_df_all[labels_df_all['patient_id'].isin(common_ids)].copy().reset_index(drop=True)
        split_indices = safe_outer_split(labels_df)
        config_name = f"{cfg['preprocessing']}__{cfg['input_name']}__bs{cfg['batch_size']}"
        config_dir = run_root / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        row = {
            'pretraining': pretraining,
            'model_family': 'resnet34_scratch_asl' if pretraining == 'scratch_asl' else ('resnet34' if pretraining == 'medicalnet' else 'genesis_encoder_classifier'),
            'source_scratch_model': cfg['model_name'],
            'preprocessing': cfg['preprocessing'],
            'input_name': cfg['input_name'],
            'batch_size': cfg['batch_size'],
            'scratch_oof_f1_micro': cfg['scratch_oof_f1_micro'],
            'scratch_oof_f1_macro': cfg['scratch_oof_f1_macro'],
            'cohort_size': int(len(labels_df)),
            'status': 'ok',
            'error_type': None,
            'error_message': None,
        }
        fold_rows = []
        oof_parts = []
        try:
            for fold_idx, (train_idx, test_idx) in enumerate(split_indices, start=1):
                fold_dir = config_dir / f'fold_{fold_idx}'
                fold_dir.mkdir(parents=True, exist_ok=True)
                train_outer_ids = labels_df.iloc[train_idx]['patient_id'].astype(str).tolist()
                test_outer_ids = labels_df.iloc[test_idx]['patient_id'].astype(str).tolist()
                fold_summary = train_one_outer_fold(pretraining, labels_df, cfg['preprocessing'], input_cfg, int(cfg['batch_size']), train_outer_ids, test_outer_ids, fold_dir, device, SEED + fold_idx)
                fold_summary['fold'] = fold_idx
                fold_rows.append(fold_summary)
                pred_path = fold_dir / 'test_predictions_best_epoch.csv'
                if pred_path.exists():
                    oof_parts.append(pd.read_csv(pred_path))
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
            fold_df = pd.DataFrame(fold_rows)
            fold_df.to_csv(config_dir / 'fold_summary.csv', index=False)
            if oof_parts:
                oof_df = pd.concat(oof_parts, ignore_index=True)
                oof_df.to_csv(config_dir / 'oof_predictions.csv', index=False)
                y_true = oof_df[[f'true_{c}' for c in TARGET_LABELS]].values
                y_prob = oof_df[[f'prob_{c}' for c in TARGET_LABELS]].values
                th = oof_df[[f'threshold_{c}' for c in TARGET_LABELS]].values
                y_pred = (y_prob >= th).astype(int)
                oof_metrics = base.compute_metrics(y_true, y_pred.astype(float), threshold=0.5)
                (config_dir / 'oof_metrics.json').write_text(json.dumps(oof_metrics, ensure_ascii=False, indent=2), encoding='utf-8')
                row['oof_f1_micro'] = float(oof_metrics['f1_micro'])
                row['oof_f1_macro'] = float(oof_metrics['f1_macro'])
                row['delta_oof_f1_micro_vs_scratch'] = float(oof_metrics['f1_micro'] - cfg['scratch_oof_f1_micro'])
                row['delta_oof_f1_macro_vs_scratch'] = float(oof_metrics['f1_macro'] - cfg['scratch_oof_f1_macro'])
                for key, value in oof_metrics.items():
                    if key.startswith('f1_'):
                        row[f'oof_{key}'] = float(value)
            row['mean_val_f1_micro'] = float(fold_df['best_val_f1_micro'].mean())
            row['std_val_f1_micro'] = float(fold_df['best_val_f1_micro'].std(ddof=0))
            row['mean_test_f1_micro'] = float(fold_df['test_f1_micro'].mean())
            row['std_test_f1_micro'] = float(fold_df['test_f1_micro'].std(ddof=0))
            row['mean_test_f1_macro'] = float(fold_df['test_f1_macro'].mean())
            row['mean_peak_alloc_gb'] = float(fold_df['peak_alloc_gb'].mean())
            row['max_peak_alloc_gb'] = float(fold_df['peak_alloc_gb'].max())
            row['max_peak_reserved_gb'] = float(fold_df['peak_reserved_gb'].max())
            row['mean_epochs'] = float(fold_df['num_epochs_ran'].mean())
            row['mean_in_channels'] = float(fold_df['in_channels'].mean())
            row['num_folds_completed'] = int(len(fold_df))
            row['all_folds_present'] = bool(len(fold_df) == N_OUTER_FOLDS)
        except RuntimeError as e:
            row['status'] = 'oom_or_runtime_error'
            row['error_type'] = type(e).__name__
            row['error_message'] = str(e)[:1000]
            (config_dir / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            row['status'] = 'failed'
            row['error_type'] = type(e).__name__
            row['error_message'] = str(e)[:1000]
            (config_dir / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(run_root / 'summary_all_configs.csv', index=False)

    summary_df = pd.DataFrame(summary_rows)
    final_report = {
        'run_root': str(run_root),
        'pretraining': pretraining,
        'status_counts': summary_df['status'].value_counts().to_dict() if not summary_df.empty else {},
        'num_configs': int(len(summary_df)),
        'best_oof_f1_micro': float(summary_df['oof_f1_micro'].max()) if 'oof_f1_micro' in summary_df else None,
        'best_oof_f1_macro': float(summary_df['oof_f1_macro'].max()) if 'oof_f1_macro' in summary_df else None,
    }
    (run_root / 'final_report.json').write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(final_report, ensure_ascii=False, indent=2))
    return run_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretraining', choices=['scratch_asl'], required=True)
    parser.add_argument('--run-tag', default='')
    parser.add_argument('--config-index', type=int, action='append', default=None, help='0-based config index; repeat to run a subset')
    parser.add_argument('--batch-size-override', type=int, default=None, help='Override batch size for selected configs')
    args = parser.parse_args()
    run_pilot(args.pretraining, run_tag=args.run_tag, config_indices=args.config_index, batch_size_override=args.batch_size_override)


if __name__ == '__main__':
    main()
