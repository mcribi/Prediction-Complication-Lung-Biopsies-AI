from __future__ import annotations

import argparse
import hashlib
import json
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
DL_ROOT = Path('/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta')
VAL_DIR = DL_ROOT / 'val_externa_interna'
PHASE_A_DIR = DL_ROOT / 'una_validacion_solo'
for extra_path in [SCRIPT_DIR, DL_ROOT, VAL_DIR, PHASE_A_DIR]:
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import phase_a_serial_common as base
import dl_nested_cv_common as nested
from two_output_metrics import (
    DEFAULT_THRESHOLD,
    TARGET_LABELS,
    THRESHOLD_GRID,
    compute_metrics_from_predictions,
    compute_two_output_and_derived_metrics,
    derive_no_complication,
    select_per_label_thresholds,
)

EXPECTED_PHASE_A_SHA256 = '34276e0503b9c29b541205f5eba4471d7456bb07edbd5525d06506f5b5eaa25a'
EXPECTED_NESTED_SHA256 = '60f108e77cd321d59029b5fd8ef59eaaca1c660b2f8dafe4378ba8ba8d63b610'
for module, expected in [(base, EXPECTED_PHASE_A_SHA256), (nested, EXPECTED_NESTED_SHA256)]:
    observed = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if observed != expected:
        raise RuntimeError(f'{Path(module.__file__).name} changed: expected {expected}, found {observed}')

PROJECT_ROOT = base.PROJECT_ROOT
OUTPUT_ROOT = PROJECT_ROOT / 'codigo' / 'DL_multietiqueta' / 'two_outputs_derived' / 'pretrained_finetuning' / 'runs'
WEIGHTS_ROOT = PROJECT_ROOT / 'pretrained_weights'
LUNA_RUN_ROOT = DL_ROOT / 'pretrained_finetuning' / 'runs' / 'luna16_nodule_pretraining_real'
PINNED_LUNA_CHECKPOINT = LUNA_RUN_ROOT / '20260803_100315_resnet34_bounded_real' / 'best_encoder_state_dict.pt'
EXPECTED_LUNA_SHA256 = '648c3ee1e4d34f83f0ce5ec49c5657724a4de42b32782720fd83393ffdb04778'
base.TARGET_LABELS = list(TARGET_LABELS)
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

CONFIG_PROFILES = {
    'phase2': [
        {
            'config_id': 'resnet34_resize_small_hu_m300_1400__ct_lung_nodule_vessels__bs4',
            'model_name': 'resnet34',
            'preprocessing': 'resize_small_hu_m300_1400',
            'input_name': 'ct_lung_nodule_vessels',
            'batch_size': 4,
        },
        {
            'config_id': 'resnet34_resize_small_multiwindowing_separadas__nodule_only_masked_ct__bs4',
            'model_name': 'resnet34',
            'preprocessing': 'resize_small_multiwindowing_separadas',
            'input_name': 'nodule_only_masked_ct',
            'batch_size': 4,
        },
    ],
    'ct_simple': [
        {
            'config_id': 'resnet34_resize_small_hu_m300_1400__nodule_only_masked_ct__bs2',
            'model_name': 'resnet34',
            'preprocessing': 'resize_small_hu_m300_1400',
            'input_name': 'nodule_only_masked_ct',
            'batch_size': 2,
        },
        {
            'config_id': 'resnet34_resize_small_hu_m300_1400__lung_only_masked_ct__bs2',
            'model_name': 'resnet34',
            'preprocessing': 'resize_small_hu_m300_1400',
            'input_name': 'lung_only_masked_ct',
            'batch_size': 2,
        },
    ],
    'luna16': [
        {
            'config_id': 'resnet34_resize_small_hu_m300_1400__nodule_only_masked_ct__bs4',
            'model_name': 'resnet34',
            'preprocessing': 'resize_small_hu_m300_1400',
            'input_name': 'nodule_only_masked_ct',
            'batch_size': 4,
        },
        {
            'config_id': 'resnet34_resize_small_hu_m300_1400__lung_only_masked_ct__bs4',
            'model_name': 'resnet34',
            'preprocessing': 'resize_small_hu_m300_1400',
            'input_name': 'lung_only_masked_ct',
            'batch_size': 4,
        },
    ],
}
FREEZE_EPOCHS = 5
LUNA_CHECKPOINT: Path | None = None


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
    if len(loaded) < 10:
        raise RuntimeError(
            f'Insufficient pretrained transfer from {weight_path}: only {len(loaded)} tensors loaded'
        )
    missing_critical = [key for key in first_conv_keys if key not in loaded]
    if missing_critical:
        raise RuntimeError(
            f'Critical pretrained tensors were not loaded from {weight_path}: {missing_critical}'
        )
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
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )


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


def build_pretrained_model(pretraining: str, in_channels: int, out_channels: int, report_dir: Path) -> nn.Module:
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
    elif pretraining == 'luna16_resnet34':
        if LUNA_CHECKPOINT is None or not LUNA_CHECKPOINT.exists():
            raise FileNotFoundError(f'LUNA16 checkpoint not found: {LUNA_CHECKPOINT}')
        model = resnet34(spatial_dims=3, n_input_channels=in_channels, num_classes=out_channels, shortcut_type='A')
        model.fc = nn.Sequential(nn.Dropout(DROPOUT), model.fc)
        weight_path = LUNA_CHECKPOINT
        report = load_matching_weights(model, weight_path, ['conv1.weight'], skip_prefixes=('fc',))
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
    if pretraining in {'medicalnet', 'luna16_resnet34'}:
        head_params = list(model.fc.parameters())
    else:
        head_params = list(model.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    encoder_params = [p for p in model.parameters() if id(p) not in head_ids]
    return encoder_params, head_params


def set_encoder_trainable(model: nn.Module, pretraining: str, trainable: bool) -> None:
    encoder_params, _ = get_head_and_encoder_params(model, pretraining)
    for p in encoder_params:
        p.requires_grad = trainable


def set_frozen_encoder_batchnorm_eval(model: nn.Module, pretraining: str) -> None:
    head_prefix = 'fc' if pretraining in {'medicalnet', 'luna16_resnet34'} else 'classifier'
    for name, module in model.named_modules():
        if not name.startswith(head_prefix) and isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def make_optimizer(model: nn.Module, pretraining: str):
    encoder_params, head_params = get_head_and_encoder_params(model, pretraining)
    return AdamW([
        {'params': encoder_params, 'lr': ENCODER_LR},
        {'params': head_params, 'lr': HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)


def compute_metrics_with_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[str, float]:
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(int)
    metrics, _ = compute_metrics_from_predictions(y_true, y_prob, y_pred)
    return metrics


def tune_thresholds_on_val(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float], List[Dict]]:
    selection = select_per_label_thresholds(y_true, y_prob, THRESHOLD_GRID)
    metrics = compute_metrics_with_thresholds(y_true, y_prob, selection.thresholds)
    return selection.thresholds, metrics, selection.curve_rows


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


def save_prediction_df(
    out_path: Path,
    fold_name: str,
    patient_ids: List[str],
    prob_matrix: np.ndarray,
    target_matrix: np.ndarray,
    tuned_thresholds: np.ndarray,
) -> None:
    pred_df = pd.DataFrame(prob_matrix, columns=[f'prob_{c}' for c in TARGET_LABELS])
    pred_df.insert(0, 'patient_id', patient_ids)
    pred_df.insert(1, 'fold', fold_name)
    for idx, label in enumerate(TARGET_LABELS):
        pred_df[f'true_{label}'] = target_matrix[:, idx].astype(int)
        pred_df[f'pred_fixed_{label}'] = (prob_matrix[:, idx] >= DEFAULT_THRESHOLD).astype(int)
        pred_df[f'pred_tuned_{label}'] = (prob_matrix[:, idx] >= tuned_thresholds[idx]).astype(int)
        pred_df[f'threshold_tuned_{label}'] = float(tuned_thresholds[idx])
    pred_df['true_Sin_complicacion'] = derive_no_complication(target_matrix.astype(int))
    pred_df['pred_fixed_Sin_complicacion'] = derive_no_complication(
        pred_df[[f'pred_fixed_{label}' for label in TARGET_LABELS]].to_numpy()
    )
    pred_df['pred_tuned_Sin_complicacion'] = derive_no_complication(
        pred_df[[f'pred_tuned_{label}' for label in TARGET_LABELS]].to_numpy()
    )
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
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = make_optimizer(model, pretraining)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler(enabled=(USE_AMP and device.type == 'cuda'))

    best_state, best_epoch = None, None
    best_score = -np.inf
    patience_counter = 0
    history = []
    best_val_probs = best_val_targets = None
    best_val_ids = None


    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == FREEZE_EPOCHS + 1:
            set_encoder_trainable(model, pretraining, True)
            freeze_unfreeze_events.append({'epoch': epoch, 'encoder_trainable': True, 'note': 'encoder unfrozen for fine tuning'})
        model.train()
        if epoch <= FREEZE_EPOCHS:
            set_frozen_encoder_batchnorm_eval(model, pretraining)
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
        scheduler.step(val_loss)

        history_row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'encoder_lr': optimizer.param_groups[0]['lr'],
            'head_lr': optimizer.param_groups[1]['lr'],
            'train_f1_micro': train_metrics['f1_micro'],
            'train_f1_macro': train_metrics['f1_macro'],
            'val_f1_micro_default_threshold': val_metrics_default['f1_micro'],
            'val_f1_macro_default_threshold': val_metrics_default['f1_macro'],
            'val_f1_micro': val_metrics_default['f1_micro'],
            'val_f1_macro': val_metrics_default['f1_macro'],
            'val_subset_accuracy': val_metrics_default['subset_accuracy'],
            'val_hamming_loss': val_metrics_default['hamming_loss'],
            'peak_alloc_gb': torch.cuda.max_memory_allocated() / (1024 ** 3) if device.type == 'cuda' else 0.0,
            'peak_reserved_gb': torch.cuda.max_memory_reserved() / (1024 ** 3) if device.type == 'cuda' else 0.0,
        }
        history.append(history_row)

        if val_metrics_default['f1_micro'] > best_score + 1e-4:
            best_score = val_metrics_default['f1_micro']
            best_epoch = epoch
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
            best_val_probs = val_probs.copy()
            best_val_targets = val_targets.copy()
            best_val_ids = list(val_ids_local)
        else:
            patience_counter += 1

        print(f'[{pretraining} | {preproc_name} | {input_cfg["name"]} | bs={batch_size}] epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1_micro={val_metrics_default["f1_micro"]:.4f} val_f1_macro={val_metrics_default["f1_macro"]:.4f}')
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
    best_thresholds, _, best_threshold_curve = tune_thresholds_on_val(
        best_val_targets, best_val_probs
    )
    pd.DataFrame(best_threshold_curve).to_csv(fold_dir / 'threshold_search_curves.csv', index=False)
    pd.DataFrame([
        {'label': label, 'selected_threshold': float(best_thresholds[index])}
        for index, label in enumerate(TARGET_LABELS)
    ]).to_csv(fold_dir / 'selected_thresholds.csv', index=False)
    save_prediction_df(fold_dir / 'val_predictions_best_epoch.csv', fold_dir.name, best_val_ids, best_val_probs, best_val_targets, best_thresholds)
    (fold_dir / 'freeze_unfreeze_events.json').write_text(json.dumps(freeze_unfreeze_events, ensure_ascii=False, indent=2), encoding='utf-8')
    model.load_state_dict(best_state)
    test_loss, test_probs, test_targets, test_ids, _ = evaluate_loader(model, test_loader, criterion, device)
    save_prediction_df(fold_dir / 'test_predictions_best_epoch.csv', fold_dir.name, test_ids, test_probs, test_targets, best_thresholds)
    fixed_metrics, _, _ = compute_two_output_and_derived_metrics(
        test_targets, test_probs, [DEFAULT_THRESHOLD, DEFAULT_THRESHOLD]
    )
    tuned_metrics, _, _ = compute_two_output_and_derived_metrics(
        test_targets, test_probs, best_thresholds
    )
    (fold_dir / 'test_metrics_fixed_0_5.json').write_text(
        json.dumps(fixed_metrics, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    (fold_dir / 'test_metrics_tuned.json').write_text(
        json.dumps(tuned_metrics, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    summary = {
        'best_epoch': int(best_epoch) if best_epoch is not None else None,
        'best_val_f1_micro_fixed_0_5': float(best_score) if best_epoch is not None else None,
        'threshold_Hemorragia': float(best_thresholds[0]),
        'threshold_Neumotorax': float(best_thresholds[1]),
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
    summary.update({f'fixed_{key}': value for key, value in fixed_metrics.items()})
    summary.update({f'tuned_{key}': value for key, value in tuned_metrics.items()})
    return summary


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def find_latest_luna_checkpoint() -> Path:
    if not PINNED_LUNA_CHECKPOINT.exists():
        raise FileNotFoundError(f'Pinned LUNA16 checkpoint not found: {PINNED_LUNA_CHECKPOINT}')
    observed = file_sha256(PINNED_LUNA_CHECKPOINT)
    if observed != EXPECTED_LUNA_SHA256:
        raise RuntimeError(
            f'Pinned LUNA16 checkpoint changed: expected {EXPECTED_LUNA_SHA256}, found {observed}'
        )
    return PINNED_LUNA_CHECKPOINT


def run_pilot(pretraining: str, profile: str, run_tag: str = '', config_indices: List[int] | None = None, luna_checkpoint: Path | None = None) -> Path:
    global LUNA_CHECKPOINT
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available in this job')
    if profile not in CONFIG_PROFILES:
        raise ValueError(f'Unsupported profile: {profile}')
    if pretraining == 'luna16_resnet34':
        if profile != 'luna16':
            raise ValueError('luna16_resnet34 requires profile luna16')
        selected_luna_checkpoint = luna_checkpoint if luna_checkpoint is not None else find_latest_luna_checkpoint()
        if selected_luna_checkpoint.resolve() != PINNED_LUNA_CHECKPOINT.resolve():
            raise ValueError(
                f'LUNA16 must use the pinned checkpoint {PINNED_LUNA_CHECKPOINT}, found {selected_luna_checkpoint}'
            )
        LUNA_CHECKPOINT = find_latest_luna_checkpoint()
    elif profile == 'luna16':
        raise ValueError('profile luna16 requires pretraining luna16_resnet34')

    set_seed(SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    clinical_source_df = pd.read_csv(base.CLINICAL_CSV, dtype={'patient_id': str})
    clinical_source_size = int(len(clinical_source_df))
    if clinical_source_size != 210:
        raise ValueError(f'Expected 210 rows in the clinical source, found {clinical_source_size}')
    labels_df_all = base.build_multilabel_dataframe(base.CLINICAL_CSV)
    target_eligible_size = int(len(labels_df_all))
    excluded_off_target = set(clinical_source_df['patient_id']) - set(labels_df_all['patient_id'])
    if target_eligible_size != 209 or excluded_off_target != {'27HASD'}:
        raise ValueError(
            f'Unexpected target-eligible cohort: size={target_eligible_size}, excluded={sorted(excluded_off_target)}'
        )
    device = torch.device('cuda')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = f'_{run_tag}' if run_tag else ''
    run_root = OUTPUT_ROOT / f'{pretraining}_{profile}_freeze_unfreeze_{timestamp}{suffix}'
    run_root.mkdir(parents=True, exist_ok=True)
    all_configs = [dict(config) for config in CONFIG_PROFILES[profile]]
    selected_configs = all_configs if config_indices is None else [all_configs[index] for index in config_indices]

    setup = {
        'timestamp': timestamp, 'pretraining': pretraining, 'profile': profile,
        'configs': selected_configs, 'outer_cv_strategy': 'StratifiedKFold_labelset_key',
        'n_outer_folds': N_OUTER_FOLDS, 'inner_val_ratio': INNER_VAL_RATIO,
        'threshold_tuning': 'independent per-label thresholds selected on val_inner predictions from the best fixed-0.5 checkpoint',
        'freeze_unfreeze': {'freeze_epochs': FREEZE_EPOCHS, 'initial': 'encoder frozen', 'after': 'full fine tuning'},
        'seed': SEED, 'max_epochs': MAX_EPOCHS, 'patience': PATIENCE,
        'encoder_lr': ENCODER_LR, 'head_lr': HEAD_LR, 'dropout': DROPOUT,
        'weight_decay': WEIGHT_DECAY, 'target_labels': TARGET_LABELS,
        'derived_state': 'Sin_complicacion = (Hemorragia == 0) and (Neumotorax == 0)',
        'fixed_threshold_reference': DEFAULT_THRESHOLD,
        'threshold_grid': [float(value) for value in THRESHOLD_GRID],
        'threshold_selection_split': 'val_inner_only', 'clinical_source_size': clinical_source_size,
        'target_eligible_cohort_size': target_eligible_size,
        'off_target_only_excluded_patients': sorted(excluded_off_target),
        'luna_checkpoint': str(LUNA_CHECKPOINT) if LUNA_CHECKPOINT is not None else None,
        'helper_hashes': {'phase_a_serial_common.py': EXPECTED_PHASE_A_SHA256, 'dl_nested_cv_common.py': EXPECTED_NESTED_SHA256},
        'gpu_name': torch.cuda.get_device_name(0),
        'methodology_note': 'Checkpoint and thresholds are selected only on val_inner; test_outer is evaluated once.',
    }
    (run_root / 'setup.json').write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding='utf-8')

    summary_rows = []
    for cfg in selected_configs:
        preprocessings = [row for row in base.discover_preprocessings() if row['name'] == cfg['preprocessing']]
        if len(preprocessings) != 1:
            raise RuntimeError(f"Preprocessing not found: {cfg['preprocessing']}")
        input_cfgs = [item for item in base.PRIMARY_INPUTS if item['name'] == cfg['input_name']]
        if len(input_cfgs) != 1:
            raise RuntimeError(f"Input not found: {cfg['input_name']}")
        input_cfg = input_cfgs[0]
        common_ids = base.collect_common_ids(labels_df_all, preprocessings, [input_cfg])
        labels_df = labels_df_all[labels_df_all['patient_id'].isin(common_ids)].copy().reset_index(drop=True)
        if len(labels_df) != 204:
            raise ValueError(
                f"Expected 204 effective patients for {cfg['config_id']}, found {len(labels_df)}"
            )
        split_indices = safe_outer_split(labels_df)
        config_name = f"{cfg['preprocessing']}__{cfg['input_name']}__bs{cfg['batch_size']}"
        config_dir = run_root / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        row = {
            'pretraining': pretraining, 'profile': profile,
            'model_family': 'genesis_encoder_classifier' if pretraining == 'genesis_chest_ct' else 'resnet34',
            'preprocessing': cfg['preprocessing'], 'input_name': cfg['input_name'],
            'batch_size': cfg['batch_size'], 'cohort_size': int(len(labels_df)),
            'status': 'ok', 'error_type': None, 'error_message': None,
            'all_folds_present': False,
        }
        fold_rows, oof_parts = [], []
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
                    oof_parts.append(pd.read_csv(pred_path, dtype={'patient_id': str}))
            fold_df = pd.DataFrame(fold_rows)
            fold_df.to_csv(config_dir / 'fold_summary.csv', index=False)
            if len(fold_rows) != N_OUTER_FOLDS or len(oof_parts) != N_OUTER_FOLDS:
                raise ValueError('Cannot aggregate without five completed folds and five prediction files')
            if set(fold_df['fold'].astype(int)) != set(range(1, N_OUTER_FOLDS + 1)):
                raise ValueError('fold_summary does not contain exactly folds 1-5')
            oof_df = pd.concat(oof_parts, ignore_index=True)
            expected_folds = {f'fold_{index}' for index in range(1, N_OUTER_FOLDS + 1)}
            if set(oof_df['fold'].astype(str)) != expected_folds:
                raise ValueError('OOF predictions do not contain exactly folds 1-5')
            if oof_df['patient_id'].duplicated().any():
                raise ValueError('Duplicated patients in OOF predictions')
            if set(oof_df['patient_id'].astype(str)) != set(labels_df['patient_id'].astype(str)):
                raise ValueError('Incomplete OOF imaging cohort')
            oof_df.to_csv(config_dir / 'oof_predictions.csv', index=False)
            y_true = oof_df[[f'true_{label}' for label in TARGET_LABELS]].values
            y_prob = oof_df[[f'prob_{label}' for label in TARGET_LABELS]].values
            fixed_pred = oof_df[[f'pred_fixed_{label}' for label in TARGET_LABELS]].values
            tuned_pred = oof_df[[f'pred_tuned_{label}' for label in TARGET_LABELS]].values
            fixed_oof_metrics, _ = compute_metrics_from_predictions(y_true, y_prob, fixed_pred)
            tuned_oof_metrics, _ = compute_metrics_from_predictions(y_true, y_prob, tuned_pred)
            (config_dir / 'oof_metrics_fixed_0_5.json').write_text(json.dumps(fixed_oof_metrics, ensure_ascii=False, indent=2), encoding='utf-8')
            (config_dir / 'oof_metrics_tuned.json').write_text(json.dumps(tuned_oof_metrics, ensure_ascii=False, indent=2), encoding='utf-8')
            row.update({f'fixed_oof_{key}': value for key, value in fixed_oof_metrics.items()})
            row.update({f'tuned_oof_{key}': value for key, value in tuned_oof_metrics.items()})
            metric_columns = [column for column in fold_df.columns if column.startswith(('fixed_', 'tuned_')) and pd.api.types.is_numeric_dtype(fold_df[column])]
            for column in metric_columns:
                row[f'mean_{column}'] = float(fold_df[column].mean())
                row[f'std_ddof1_{column}'] = float(fold_df[column].std(ddof=1))
            row['mean_val_f1_micro_fixed_0_5'] = float(fold_df['best_val_f1_micro_fixed_0_5'].mean())
            row['std_ddof1_val_f1_micro_fixed_0_5'] = float(fold_df['best_val_f1_micro_fixed_0_5'].std(ddof=1))
            row['mean_peak_alloc_gb'] = float(fold_df['peak_alloc_gb'].mean())
            row['max_peak_alloc_gb'] = float(fold_df['peak_alloc_gb'].max())
            row['max_peak_reserved_gb'] = float(fold_df['peak_reserved_gb'].max())
            row['mean_epochs'] = float(fold_df['num_epochs_ran'].mean())
            row['mean_in_channels'] = float(fold_df['in_channels'].mean())
            row['num_folds_completed'] = int(len(fold_df))
            row['all_folds_present'] = True
        except RuntimeError as error:
            row['status'] = 'oom_or_runtime_error'
            row['error_type'] = type(error).__name__
            row['error_message'] = str(error)[:1000]
            (config_dir / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
            torch.cuda.empty_cache()
        except Exception as error:
            row['status'] = 'failed'
            row['error_type'] = type(error).__name__
            row['error_message'] = str(error)[:1000]
            (config_dir / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(run_root / 'summary_all_configs.csv', index=False)

    summary_df = pd.DataFrame(summary_rows)
    all_complete = bool(len(summary_df) == len(selected_configs) and (summary_df['status'] == 'ok').all() and summary_df['all_folds_present'].fillna(False).all())
    final_report = {
        'run_root': str(run_root), 'pretraining': pretraining, 'profile': profile,
        'status_counts': summary_df['status'].value_counts().to_dict(),
        'num_configs': int(len(summary_df)), 'all_complete': all_complete,
    }
    (run_root / 'final_report.json').write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(final_report, ensure_ascii=False, indent=2))
    if not all_complete:
        raise RuntimeError(f'Pretrained freeze job is incomplete; see {run_root}')
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretraining', choices=['medicalnet', 'genesis_chest_ct', 'luna16_resnet34'], required=True)
    parser.add_argument('--profile', choices=sorted(CONFIG_PROFILES), required=True)
    parser.add_argument('--run-tag', default='')
    parser.add_argument('--config-index', type=int, action='append', default=None)
    parser.add_argument('--luna-checkpoint', type=Path, default=None)
    args = parser.parse_args()
    run_pilot(args.pretraining, profile=args.profile, run_tag=args.run_tag, config_indices=args.config_index, luna_checkpoint=args.luna_checkpoint)


if __name__ == '__main__':
    main()
