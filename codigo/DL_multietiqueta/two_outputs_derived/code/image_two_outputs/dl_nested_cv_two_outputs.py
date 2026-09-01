from __future__ import annotations

import argparse
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
from monai.networks.nets import DenseNet121, resnet10, resnet18, resnet34, seresnet50
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

MODULE_DIR = Path(__file__).resolve().parent
DL_ROOT = Path('/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta')
PHASE_A_DIR = DL_ROOT / 'una_validacion_solo'
for extra_path in [MODULE_DIR.parent, DL_ROOT, PHASE_A_DIR]:
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import phase_a_serial_common as base

from two_output_metrics import (
    DEFAULT_THRESHOLD,
    TARGET_LABELS,
    THRESHOLD_GRID,
    compute_metrics_from_predictions,
    compute_two_output_and_derived_metrics,
    derive_no_complication,
    select_per_label_thresholds,
)

PROJECT_ROOT = base.PROJECT_ROOT
OUTPUT_ROOT = PROJECT_ROOT / 'codigo' / 'DL_multietiqueta' / 'two_outputs_derived' / 'image_only' / 'runs'
SEED = base.SEED
N_OUTER_FOLDS = 5
INNER_VAL_RATIO = 0.2
MAX_EPOCHS = base.MAX_EPOCHS
PATIENCE = base.PATIENCE
LEARNING_RATE = base.LEARNING_RATE
DROPOUT = base.DROPOUT
WEIGHT_DECAY = base.WEIGHT_DECAY
NUM_WORKERS = base.NUM_WORKERS
VAL_THRESHOLD = base.VAL_THRESHOLD
USE_AMP = base.USE_AMP
base.TARGET_LABELS = list(TARGET_LABELS)
GRADCAM_TOP_K = base.GRADCAM_TOP_K

PRIMARY_INPUTS_ALL = list(base.PRIMARY_INPUTS)
PRIMARY_INPUTS_FOCUSED = [cfg for cfg in PRIMARY_INPUTS_ALL if cfg['name'] in {'nodule_only_masked_ct', 'ct_lung_nodule', 'ct_lung_nodule_vessels'}]
PRIMARY_INPUTS_DENSENET_LIGHT = [cfg for cfg in PRIMARY_INPUTS_ALL if cfg['name'] in {'nodule_only_masked_ct', 'ct_lung_nodule'}]

MODEL_NAMES = ['resnet18', 'densenet121', 'resnet10', 'resnet34', 'seresnet50']


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_name: str, in_channels: int, out_channels: int):
    if model_name == 'resnet18':
        model = resnet18(spatial_dims=3, n_input_channels=in_channels, num_classes=out_channels)
        model.fc = nn.Sequential(nn.Dropout(DROPOUT), model.fc)
        return model
    if model_name == 'densenet121':
        return DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=DROPOUT,
        )
    if model_name == 'resnet10':
        model = resnet10(spatial_dims=3, n_input_channels=in_channels, num_classes=out_channels)
        model.fc = nn.Sequential(nn.Dropout(DROPOUT), model.fc)
        return model
    if model_name == 'resnet34':
        model = resnet34(spatial_dims=3, n_input_channels=in_channels, num_classes=out_channels)
        model.fc = nn.Sequential(nn.Dropout(DROPOUT), model.fc)
        return model
    if model_name == 'seresnet50':
        model = seresnet50(spatial_dims=3, in_channels=in_channels, num_classes=out_channels)
        model.last_linear = nn.Sequential(nn.Dropout(DROPOUT), model.last_linear)
        return model
    raise ValueError(f'Unsupported model: {model_name}')


def candidate_batch_sizes_from_shape(shape: Tuple[int, ...]) -> List[int]:
    spatial = base.spatial_shape(shape)
    mapping = {
        (64, 64, 64): [16],
        (128, 128, 128): [8],
        (128, 256, 256): [4],
        (256, 512, 512): [1],
    }
    return list(mapping.get(spatial, []))


def grad_accum_steps_from_shape(shape: Tuple[int, ...], batch_size: int) -> int:
    spatial = base.spatial_shape(shape)
    virtual_batch_size = {
        (256, 512, 512): 4,
    }.get(spatial)
    if virtual_batch_size is None:
        return 1
    return max(1, int(virtual_batch_size) // int(batch_size))


def resolve_input_profiles(profile_name: str) -> List[Dict]:
    if profile_name == 'all':
        return list(PRIMARY_INPUTS_ALL)
    if profile_name == 'focused':
        return list(PRIMARY_INPUTS_FOCUSED)
    if profile_name == 'densenet_light':
        return list(PRIMARY_INPUTS_DENSENET_LIGHT)
    raise ValueError(f'Unsupported input profile: {profile_name}')


def default_input_profile_for_model(model_name: str) -> str:
    if model_name == 'densenet121':
        return 'densenet_light'
    return 'focused'


def safe_outer_split(labels_df: pd.DataFrame):
    splitter = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
    return list(splitter.split(labels_df['patient_id'].tolist(), labels_df['labelset_key'].tolist()))


def safe_inner_split(train_df: pd.DataFrame, fold_seed: int) -> Tuple[List[str], List[str], str]:
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
    losses = []
    probs = []
    targets = []
    patient_ids = []
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
    prob_matrix = np.concatenate(probs, axis=0)
    target_matrix = np.concatenate(targets, axis=0)
    metrics = base.compute_metrics(target_matrix, prob_matrix, threshold=VAL_THRESHOLD)
    return loss_value, prob_matrix, target_matrix, patient_ids, metrics


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
    true_two = target_matrix.astype(int)
    pred_fixed_two = pred_df[[f'pred_fixed_{c}' for c in TARGET_LABELS]].to_numpy()
    pred_tuned_two = pred_df[[f'pred_tuned_{c}' for c in TARGET_LABELS]].to_numpy()
    pred_df['true_Sin_complicacion'] = derive_no_complication(true_two)
    pred_df['pred_fixed_Sin_complicacion'] = derive_no_complication(pred_fixed_two)
    pred_df['pred_tuned_Sin_complicacion'] = derive_no_complication(pred_tuned_two)
    pred_df.to_csv(out_path, index=False)


def train_one_outer_fold(model_name, labels_df, preproc_name, input_cfg, batch_size, train_outer_ids, test_outer_ids, fold_dir, device, fold_seed):
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
    grad_accum_steps = max(1, int(grad_accum_steps_from_shape(sample_x.shape, batch_size)))
    model = build_model(model_name, in_channels, len(TARGET_LABELS)).to(device)
    pos_weight = base.compute_pos_weight(labels_df, inner_train_ids).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler(enabled=(USE_AMP and device.type == 'cuda'))

    best_state = None
    best_score = -np.inf
    best_epoch = None
    patience_counter = 0
    history = []
    best_val_probs = None
    best_val_targets = None
    best_val_ids = None

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        train_probs = []
        train_targets = []
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

        val_loss, val_probs, val_targets, val_ids_local, val_metrics = evaluate_loader(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history_row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'train_f1_micro': train_metrics['f1_micro'],
            'train_f1_macro': train_metrics['f1_macro'],
            'val_f1_micro': val_metrics['f1_micro'],
            'val_f1_macro': val_metrics['f1_macro'],
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
        else:
            patience_counter += 1

        print(
            f'[{model_name} | {preproc_name} | {input_cfg["name"]} | bs={batch_size}] '
            f'epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} '
            f'val_f1_micro={val_metrics["f1_micro"]:.4f} val_f1_macro={val_metrics["f1_macro"]:.4f}'
        )

        if patience_counter >= PATIENCE:
            print(f'Early stopping at epoch {epoch}')
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(fold_dir / 'history.csv', index=False)

    split_rows = []
    for pid in inner_train_ids:
        split_rows.append({'split': 'train_inner', 'patient_id': pid})
    for pid in inner_val_ids:
        split_rows.append({'split': 'val_inner', 'patient_id': pid})
    for pid in test_outer_ids:
        split_rows.append({'split': 'test_outer', 'patient_id': pid})
    pd.DataFrame(split_rows).to_csv(fold_dir / 'split_assignments.csv', index=False)

    if best_state is None:
        raise RuntimeError('No best state was selected during training')

    torch.save(best_state, fold_dir / 'best_model.pt')
    threshold_selection = select_per_label_thresholds(best_val_targets, best_val_probs, THRESHOLD_GRID)
    tuned_thresholds = threshold_selection.thresholds
    pd.DataFrame(threshold_selection.curve_rows).to_csv(
        fold_dir / 'threshold_search_curves.csv', index=False
    )
    pd.DataFrame([
        {'label': label, 'selected_threshold': float(tuned_thresholds[index])}
        for index, label in enumerate(TARGET_LABELS)
    ]).to_csv(fold_dir / 'selected_thresholds.csv', index=False)
    save_prediction_df(
        fold_dir / 'val_predictions_best_epoch.csv',
        fold_dir.name,
        best_val_ids,
        best_val_probs,
        best_val_targets,
        tuned_thresholds,
    )

    model.load_state_dict(best_state)
    test_loss, test_probs, test_targets, test_ids, _ = evaluate_loader(model, test_loader, criterion, device)
    save_prediction_df(
        fold_dir / 'test_predictions_best_epoch.csv',
        fold_dir.name,
        test_ids,
        test_probs,
        test_targets,
        tuned_thresholds,
    )
    fixed_metrics, _, _ = compute_two_output_and_derived_metrics(
        test_targets, test_probs, [DEFAULT_THRESHOLD, DEFAULT_THRESHOLD]
    )
    tuned_metrics, _, _ = compute_two_output_and_derived_metrics(
        test_targets, test_probs, tuned_thresholds
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
        'fixed_test_f1_micro': float(fixed_metrics['two_outputs_f1_micro']),
        'fixed_test_f1_macro': float(fixed_metrics['two_outputs_f1_macro']),
        'tuned_test_f1_micro': float(tuned_metrics['two_outputs_f1_micro']),
        'tuned_test_f1_macro': float(tuned_metrics['two_outputs_f1_macro']),
        'fixed_clinical_f1_micro': float(fixed_metrics['clinical_three_states_f1_micro']),
        'fixed_clinical_f1_macro': float(fixed_metrics['clinical_three_states_f1_macro']),
        'tuned_clinical_f1_micro': float(tuned_metrics['clinical_three_states_f1_micro']),
        'tuned_clinical_f1_macro': float(tuned_metrics['clinical_three_states_f1_macro']),
        'threshold_Hemorragia': float(tuned_thresholds[0]),
        'threshold_Neumotorax': float(tuned_thresholds[1]),
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


def prepare_gradcam_top5_plan(summary_df: pd.DataFrame, run_root: Path):
    out_path = run_root / 'gradcam_top5_candidates.json'
    ok_df = summary_df[summary_df['status'] == 'ok'].copy()
    required_cols = ['tuned_oof_two_outputs_f1_micro', 'mean_tuned_test_f1_micro']
    if ok_df.empty or any(col not in ok_df.columns for col in required_cols):
        out_path.write_text(json.dumps([], ensure_ascii=False, indent=2), encoding='utf-8')
        return out_path
    top_df = ok_df.sort_values(
        ['tuned_oof_two_outputs_f1_micro', 'mean_tuned_test_f1_micro'],
        ascending=False,
    ).head(GRADCAM_TOP_K).copy()
    payload = top_df.to_dict(orient='records')
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return out_path


def run_model(model_name: str, input_profile: str, run_tag: str = '', output_root: Path | None = None, preprocessing_names: List[str] | None = None) -> Path:
    if model_name not in MODEL_NAMES:
        raise ValueError(f'Unsupported model: {model_name}')

    set_seed(SEED)
    output_root = OUTPUT_ROOT if output_root is None else Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_preprocessings = base.discover_preprocessings()
    preprocessings = list(all_preprocessings)
    if preprocessing_names:
        requested = set(preprocessing_names)
        preprocessings = [row for row in preprocessings if row['name'] in requested]
        observed = {row['name'] for row in preprocessings}
        if observed != requested:
            raise ValueError(f'Missing requested preprocessings: {sorted(requested - observed)}')
    selected_inputs = resolve_input_profiles(input_profile)

    clinical_source_df = pd.read_csv(base.CLINICAL_CSV, dtype={'patient_id': str})
    clinical_source_size = int(len(clinical_source_df))
    if clinical_source_size != 210:
        raise ValueError(f'Expected 210 rows in the clinical source, found {clinical_source_size}')
    labels_df = base.build_multilabel_dataframe(base.CLINICAL_CSV)
    target_eligible_size = int(len(labels_df))
    excluded_off_target = set(clinical_source_df['patient_id']) - set(labels_df['patient_id'])
    if target_eligible_size != 209 or excluded_off_target != {'27HASD'}:
        raise ValueError(
            f'Unexpected target-eligible cohort: size={target_eligible_size}, excluded={sorted(excluded_off_target)}'
        )
    common_ids = base.collect_common_ids(labels_df, all_preprocessings, selected_inputs)
    labels_df = labels_df[labels_df['patient_id'].isin(common_ids)].copy().reset_index(drop=True)
    if len(labels_df) != 204:
        raise ValueError(f'Expected 204 effective imaging patients, found {len(labels_df)}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = f'_{run_tag}' if run_tag else ''
    run_root = output_root / f'{model_name}_nestedcv_{input_profile}_{timestamp}{suffix}'
    run_root.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('CUDA is required for the full image-only experiment')
    split_indices = safe_outer_split(labels_df)

    setup = {
        'timestamp': timestamp,
        'model_name': model_name,
        'input_profile': input_profile,
        'outer_cv_strategy': 'StratifiedKFold_labelset_key',
        'n_outer_folds': N_OUTER_FOLDS,
        'inner_val_ratio': INNER_VAL_RATIO,
        'seed': SEED,
        'max_epochs': MAX_EPOCHS,
        'patience': PATIENCE,
        'lr': LEARNING_RATE,
        'dropout': DROPOUT,
        'weight_decay': WEIGHT_DECAY,
        'target_labels': TARGET_LABELS,
        'clinical_source_size': clinical_source_size,
        'target_eligible_cohort_size': target_eligible_size,
        'off_target_only_excluded_patients': sorted(excluded_off_target),
        'effective_imaging_cohort_size': int(len(labels_df)),
        'derived_state': 'Sin_complicacion = (Hemorragia == 0) and (Neumotorax == 0)',
        'threshold_grid': [float(value) for value in THRESHOLD_GRID],
        'threshold_selection_split': 'val_inner_best_checkpoint_only',
        'fixed_threshold_reference': DEFAULT_THRESHOLD,
        'inputs': selected_inputs,
        'preprocessings': [
            {
                'name': row['name'],
                'shape': row['shape'],
                'count': row['count'],
                'candidate_batch_sizes': candidate_batch_sizes_from_shape(row['shape']),
            }
            for row in preprocessings
        ],
        'cohort_size': int(len(labels_df)),
        'gradcam_top_k': GRADCAM_TOP_K,
        'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
    }
    (run_root / 'setup.json').write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding='utf-8')

    all_rows = []
    for preproc in preprocessings:
        for input_cfg in selected_inputs:
            for batch_size in candidate_batch_sizes_from_shape(preproc['shape']):
                config_name = f"{preproc['name']}__{input_cfg['name']}__bs{batch_size}"
                config_dir = run_root / config_name
                config_dir.mkdir(parents=True, exist_ok=True)
                print(f'START {config_name}')
                grad_accum_steps = grad_accum_steps_from_shape(preproc['shape'], batch_size)
                row = {
                    'model_name': model_name,
                    'preprocessing': preproc['name'],
                    'input_name': input_cfg['name'],
                    'batch_size': int(batch_size),
                    'grad_accum_steps': int(grad_accum_steps),
                    'effective_batch_size': int(batch_size * grad_accum_steps),
                    'shape': str(preproc['shape']),
                    'status': 'ok',
                    'error_type': None,
                    'error_message': None,
                }
                fold_rows = []
                try:
                    oof_parts = []
                    for fold_idx, (train_idx, test_idx) in enumerate(split_indices, start=1):
                        fold_dir = config_dir / f'fold_{fold_idx}'
                        fold_dir.mkdir(parents=True, exist_ok=True)
                        train_outer_ids = labels_df.iloc[train_idx]['patient_id'].astype(str).tolist()
                        test_outer_ids = labels_df.iloc[test_idx]['patient_id'].astype(str).tolist()
                        fold_summary = train_one_outer_fold(
                            model_name=model_name,
                            labels_df=labels_df,
                            preproc_name=preproc['name'],
                            input_cfg=input_cfg,
                            batch_size=batch_size,
                            train_outer_ids=train_outer_ids,
                            test_outer_ids=test_outer_ids,
                            fold_dir=fold_dir,
                            device=device,
                            fold_seed=SEED + fold_idx,
                        )
                        fold_summary['fold'] = fold_idx
                        fold_rows.append(fold_summary)
                        pred_path = fold_dir / 'test_predictions_best_epoch.csv'
                        if pred_path.exists():
                            oof_parts.append(pd.read_csv(pred_path, dtype={'patient_id': str}))
                    fold_df = pd.DataFrame(fold_rows)
                    fold_df.to_csv(config_dir / 'fold_summary.csv', index=False)
                    if len(fold_rows) != N_OUTER_FOLDS or len(oof_parts) != N_OUTER_FOLDS:
                        raise ValueError('Cannot aggregate without five completed folds and five prediction files')
                    if oof_parts:
                        oof_df = pd.concat(oof_parts, ignore_index=True)
                        expected_folds = {f'fold_{index}' for index in range(1, N_OUTER_FOLDS + 1)}
                        if set(oof_df['fold'].astype(str)) != expected_folds:
                            raise ValueError('OOF predictions do not contain exactly folds 1-5')
                        if oof_df['patient_id'].duplicated().any():
                            raise ValueError('Duplicated patients in OOF predictions')
                        expected_patient_ids = set(labels_df['patient_id'].astype(str))
                        if set(oof_df['patient_id'].astype(str)) != expected_patient_ids:
                            raise ValueError('Incomplete OOF imaging cohort')
                        oof_df.to_csv(config_dir / 'oof_predictions.csv', index=False)
                        y_true = oof_df[[f'true_{c}' for c in TARGET_LABELS]].values
                        y_prob = oof_df[[f'prob_{c}' for c in TARGET_LABELS]].values
                        fixed_pred = oof_df[[f'pred_fixed_{c}' for c in TARGET_LABELS]].values
                        tuned_pred = oof_df[[f'pred_tuned_{c}' for c in TARGET_LABELS]].values
                        fixed_oof_metrics, _ = compute_metrics_from_predictions(y_true, y_prob, fixed_pred)
                        tuned_oof_metrics, _ = compute_metrics_from_predictions(y_true, y_prob, tuned_pred)
                        (config_dir / 'oof_metrics_fixed_0_5.json').write_text(
                            json.dumps(fixed_oof_metrics, ensure_ascii=False, indent=2), encoding='utf-8'
                        )
                        (config_dir / 'oof_metrics_tuned.json').write_text(
                            json.dumps(tuned_oof_metrics, ensure_ascii=False, indent=2), encoding='utf-8'
                        )
                        row.update({f'fixed_oof_{key}': value for key, value in fixed_oof_metrics.items()})
                        row.update({f'tuned_oof_{key}': value for key, value in tuned_oof_metrics.items()})
                    metric_columns = [
                        column for column in fold_df.columns
                        if column.startswith(('fixed_', 'tuned_'))
                        and pd.api.types.is_numeric_dtype(fold_df[column])
                    ]
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
                    row['all_folds_present'] = bool(
                        len(fold_df) == N_OUTER_FOLDS
                        and set(fold_df['fold'].astype(int)) == set(range(1, N_OUTER_FOLDS + 1))
                    )
                except RuntimeError as e:
                    row['status'] = 'oom_or_runtime_error'
                    row['error_type'] = type(e).__name__
                    row['error_message'] = str(e)[:1000]
                    (config_dir / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                except Exception as e:
                    row['status'] = 'failed'
                    row['error_type'] = type(e).__name__
                    row['error_message'] = str(e)[:1000]
                    (config_dir / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
                all_rows.append(row)
                pd.DataFrame(all_rows).to_csv(run_root / 'summary_all_configs.csv', index=False)

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(run_root / 'summary_all_configs.csv', index=False)
    gradcam_plan = prepare_gradcam_top5_plan(summary_df, run_root)
    final_report = {
        'run_root': str(run_root),
        'model_name': model_name,
        'input_profile': input_profile,
        'total_configs': int(len(summary_df)),
        'ok_configs': int((summary_df['status'] == 'ok').sum()) if not summary_df.empty else 0,
        'failed_configs': int((summary_df['status'] != 'ok').sum()) if not summary_df.empty else 0,
        'gradcam_top5_candidates': str(gradcam_plan),
    }
    (run_root / 'final_report.json').write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(final_report, ensure_ascii=False, indent=2))
    if final_report['failed_configs'] != 0:
        raise RuntimeError(
            f"Image-only job contains {final_report['failed_configs']} failed configurations; "
            f"see {run_root / 'summary_all_configs.csv'}"
        )
    return run_root


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODEL_NAMES, required=True)
    parser.add_argument('--input-profile', choices=['all', 'focused', 'densenet_light'], default='')
    parser.add_argument('--run-tag', default='')
    parser.add_argument('--output-root', default=str(OUTPUT_ROOT))
    parser.add_argument('--preprocessing-names', nargs='+', default=[])
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    input_profile = args.input_profile or default_input_profile_for_model(args.model)
    run_model(
        args.model,
        input_profile=input_profile,
        run_tag=args.run_tag,
        output_root=Path(args.output_root),
        preprocessing_names=args.preprocessing_names,
    )


if __name__ == '__main__':
    main()
