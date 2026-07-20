import json
import math
import random
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121, resnet18
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path('/mnt/homeGPU/mcribilles/tfm')
CLINICAL_CSV = PROJECT_ROOT / 'clinical_data/210pacientes/clinical_data.csv'
PREPROC_ROOT = PROJECT_ROOT / 'volumenes_preprocesados'
OUTPUT_ROOT = PROJECT_ROOT / 'codigo/DL_multietiqueta/phase_a_single_job_runs'
TARGET_LABELS = ['Hemorragia', 'Neumotórax', 'Sin_complicacion']
PRIMARY_INPUTS = [
    {'name': 'lung_only_masked_ct', 'channels': ['ct_lung']},
    {'name': 'nodule_only_masked_ct', 'channels': ['ct_nodule']},
    {'name': 'ct_lung_nodule', 'channels': ['ct_lung', 'lung', 'nodule']},
    {'name': 'nodule_vessels', 'channels': ['ct_nodule', 'nodule', 'vessels']},
    {'name': 'ct_lung_nodule_vessels', 'channels': ['ct_lung', 'lung', 'nodule', 'vessels']},
]
MASK_DEPENDENCIES = {
    'lung': 'masks_lung',
    'nodule': 'masks_nodule',
    'vessels': 'masks_vessels',
    'trachea_bronchia': 'masks_trachea_bronchia',
}
CHANNEL_REQUIREMENTS = {
    'ct_lung': ['lung'],
    'ct_nodule': ['nodule'],
    'lung': ['lung'],
    'nodule': ['nodule'],
    'vessels': ['vessels'],
    'trachea_bronchia': ['trachea_bronchia'],
}
SEED = 17
N_FOLDS = 5
MAX_EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 1e-4
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
VAL_THRESHOLD = 0.5
USE_AMP = True
DROP_OFF_TARGET_ONLY = True
GRADCAM_TOP_K = 5


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_text(x):
    if pd.isna(x):
        return ''
    return str(x).strip()


def split_complications(raw_value: str):
    value = normalize_text(raw_value)
    if value == '' or value.lower() == 'x':
        return []
    parts = [p.strip() for p in value.split(',')]
    return [p for p in parts if p]


def build_multilabel_dataframe(clinical_csv: Path):
    df = pd.read_csv(clinical_csv).copy()
    df['patient_id'] = df['patient_id'].astype(str)
    df['Complicación'] = df['Complicación'].apply(normalize_text)
    df['Tipo de complicación'] = df['Tipo de complicación'].apply(normalize_text)
    parsed = df['Tipo de complicación'].apply(split_complications)
    df['Hemorragia'] = parsed.apply(lambda xs: int('Hemorragia' in xs))
    df['Neumotórax'] = parsed.apply(lambda xs: int('Neumotórax' in xs))
    df['Derrame_pleural'] = parsed.apply(lambda xs: int('Derrame pleural' in xs))
    off_target_only = (
        (df['Hemorragia'] == 0)
        & (df['Neumotórax'] == 0)
        & (df['Derrame_pleural'] == 1)
    )
    if DROP_OFF_TARGET_ONLY:
        df = df.loc[~off_target_only].copy()
    df['Sin_complicacion'] = ((df['Hemorragia'] == 0) & (df['Neumotórax'] == 0)).astype(int)
    df['labelset_key'] = df[TARGET_LABELS].astype(str).agg(''.join, axis=1)
    return df


def discover_preprocessings():
    rows = []
    for d in sorted([p for p in PREPROC_ROOT.iterdir() if p.is_dir()]):
        img_dir = d / 'npy' / 'images'
        sample = next(img_dir.glob('*.npy'), None) if img_dir.exists() else None
        if sample is None:
            continue
        arr = np.load(sample)
        rows.append({'name': d.name, 'shape': tuple(arr.shape), 'count': sum(1 for _ in img_dir.glob('*.npy'))})
    return rows


def spatial_shape(shape):
    if len(shape) == 3:
        return tuple(shape)
    if len(shape) == 4:
        if shape[0] <= 4:
            return tuple(shape[1:])
        if shape[-1] <= 4:
            return tuple(shape[:-1])
    return tuple(shape)


def candidate_batch_sizes_from_shape(shape):
    spatial = spatial_shape(shape)
    if spatial == (64, 64, 64):
        return [32, 16]
    return [16, 8]


def grad_accum_steps_from_shape(shape, batch_size):
    return 1


def required_masks_for_input(input_cfg):
    needed = set()
    for channel in input_cfg['channels']:
        needed.update(CHANNEL_REQUIREMENTS[channel])
    return sorted(needed)


def preprocess_channels(arr: np.ndarray):
    if arr.ndim == 3:
        return arr.astype(np.float32)[None, ...]
    if arr.ndim == 4:
        if arr.shape[0] <= 4:
            return arr.astype(np.float32)
        if arr.shape[-1] <= 4:
            return np.moveaxis(arr, -1, 0).astype(np.float32)
    raise ValueError(f'Unsupported image shape: {arr.shape}')


def collect_common_ids(labels_df, preprocessings, inputs):
    label_ids = set(labels_df['patient_id'].astype(str))
    common_ids = None
    for preproc in preprocessings:
        base = PREPROC_ROOT / preproc['name'] / 'npy'
        image_ids = {p.stem for p in (base / 'images').glob('*.npy')}
        for input_cfg in inputs:
            current = set(image_ids)
            for mask_key in required_masks_for_input(input_cfg):
                mask_ids = {p.stem for p in (base / MASK_DEPENDENCIES[mask_key]).glob('*.npy')}
                current &= mask_ids
            if common_ids is None:
                common_ids = set(current)
            else:
                common_ids &= set(current)
    return sorted(label_ids & (common_ids if common_ids is not None else set()))


class MultiInputVolumeDataset(Dataset):
    def __init__(self, labels_frame, patient_ids, preproc_name, input_cfg):
        self.df = labels_frame.set_index('patient_id')
        self.patient_ids = list(patient_ids)
        self.preproc_name = preproc_name
        self.input_cfg = input_cfg
        self.base_dir = PREPROC_ROOT / preproc_name / 'npy'
        self.image_dir = self.base_dir / 'images'
        self.mask_dirs = {k: self.base_dir / v for k, v in MASK_DEPENDENCIES.items()}

    def __len__(self):
        return len(self.patient_ids)

    def _load_mask(self, mask_key, pid):
        arr = np.load(self.mask_dirs[mask_key] / f'{pid}.npy').astype(np.float32)
        if arr.ndim == 4:
            if arr.shape[0] <= 4:
                arr = arr[0]
            elif arr.shape[-1] <= 4:
                arr = arr[..., 0]
        return (arr > 0).astype(np.float32)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        img = np.load(self.image_dir / f'{pid}.npy')
        img_channels = preprocess_channels(img)
        masks = {key: self._load_mask(key, pid) for key in required_masks_for_input(self.input_cfg)}
        out_channels = []
        for channel in self.input_cfg['channels']:
            if channel == 'ct_lung':
                for img_ch in img_channels:
                    out_channels.append(img_ch * masks['lung'])
            elif channel == 'ct_nodule':
                for img_ch in img_channels:
                    out_channels.append(img_ch * masks['nodule'])
            else:
                out_channels.append(masks[channel])
        x = np.stack(out_channels, axis=0).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, 0.0, 1.0)
        y = self.df.loc[pid, TARGET_LABELS].values.astype(np.float32)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), pid


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
    raise ValueError(f'Unsupported model: {model_name}')


def compute_pos_weight(labels_frame, patient_ids):
    y = labels_frame.set_index('patient_id').loc[patient_ids, TARGET_LABELS].values.astype(np.float32)
    pos = y.sum(axis=0)
    neg = len(patient_ids) - pos
    pos_weight = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'subset_accuracy': accuracy_score(y_true, y_pred),
        'hamming_loss': hamming_loss(y_true, y_pred),
        'jaccard_micro': jaccard_score(y_true, y_pred, average='micro', zero_division=0),
    }
    for idx, label in enumerate(TARGET_LABELS):
        metrics[f'f1_{label}'] = f1_score(y_true[:, idx], y_pred[:, idx], zero_division=0)
    return metrics


def train_one_fold(model_name, labels_df, preproc_name, input_cfg, batch_size, train_ids, val_ids, fold_dir, device):
    train_ds = MultiInputVolumeDataset(labels_df, train_ids, preproc_name, input_cfg)
    val_ds = MultiInputVolumeDataset(labels_df, val_ids, preproc_name, input_cfg)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    sample_x, _, _ = train_ds[0]
    in_channels = int(sample_x.shape[0])
    grad_accum_steps = max(1, int(grad_accum_steps_from_shape(sample_x.shape, batch_size)))
    model = build_model(model_name, in_channels, len(TARGET_LABELS)).to(device)
    pos_weight = compute_pos_weight(labels_df, train_ids).to(device)
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
        train_metrics = compute_metrics(train_targets, train_probs, threshold=VAL_THRESHOLD)

        model.eval()
        val_losses = []
        val_probs = []
        val_targets = []
        val_ids_local = []
        with torch.no_grad():
            for xb, yb, pids in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with autocast(enabled=(USE_AMP and device.type == 'cuda')):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                val_losses.append(loss.item())
                val_probs.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(yb.cpu().numpy())
                val_ids_local.extend(pids)

        val_loss = float(np.mean(val_losses)) if val_losses else np.nan
        val_probs = np.concatenate(val_probs, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_metrics = compute_metrics(val_targets, val_probs, threshold=VAL_THRESHOLD)
        scheduler.step(val_loss)

        history.append({
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
            'peak_alloc_gb': torch.cuda.max_memory_allocated() / (1024 ** 3),
            'peak_reserved_gb': torch.cuda.max_memory_reserved() / (1024 ** 3),
        })

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
    if best_state is not None:
        torch.save(best_state, fold_dir / 'best_model.pt')
        pred_df = pd.DataFrame(best_val_probs, columns=[f'prob_{c}' for c in TARGET_LABELS])
        pred_df.insert(0, 'patient_id', best_val_ids)
        pred_df.insert(1, 'fold', fold_dir.name)
        for idx, label in enumerate(TARGET_LABELS):
            pred_df[f'true_{label}'] = best_val_targets[:, idx].astype(int)
            pred_df[f'pred_{label}'] = (best_val_probs[:, idx] >= VAL_THRESHOLD).astype(int)
        pred_df.to_csv(fold_dir / 'val_predictions_best_epoch.csv', index=False)
    summary = {
        'best_epoch': int(best_epoch) if best_epoch is not None else None,
        'best_val_f1_micro': float(best_score) if best_epoch is not None else None,
        'peak_alloc_gb': float(history_df['peak_alloc_gb'].max()) if not history_df.empty else None,
        'peak_reserved_gb': float(history_df['peak_reserved_gb'].max()) if not history_df.empty else None,
        'num_epochs_ran': int(len(history_df)),
        'in_channels': int(sample_x.shape[0]),
        'grad_accum_steps': int(grad_accum_steps),
        'effective_batch_size': int(batch_size * grad_accum_steps),
    }
    return summary


def prepare_gradcam_top5_plan(summary_df: pd.DataFrame, run_root: Path):
    out_path = run_root / 'gradcam_top5_candidates.json'
    ok_df = summary_df[summary_df['status'] == 'ok'].copy()
    required_cols = ['mean_f1_micro', 'mean_f1_macro']
    if ok_df.empty or any(col not in ok_df.columns for col in required_cols):
        out_path.write_text(json.dumps([], ensure_ascii=False, indent=2), encoding='utf-8')
        return out_path
    top_df = ok_df.sort_values(required_cols, ascending=False).head(GRADCAM_TOP_K).copy()
    payload = top_df.to_dict(orient='records')
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return out_path


def run_phase_a(model_name: str):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available in this job')
    set_seed(SEED)
    device = torch.device('cuda')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_root = OUTPUT_ROOT / f'{model_name}_phase_a_{timestamp}'
    run_root.mkdir(parents=True, exist_ok=True)

    labels_df = build_multilabel_dataframe(CLINICAL_CSV)
    preprocessings = discover_preprocessings()
    common_ids = collect_common_ids(labels_df, preprocessings, PRIMARY_INPUTS)
    labels_df = labels_df[labels_df['patient_id'].isin(common_ids)].copy().reset_index(drop=True)

    setup = {
        'model_name': model_name,
        'seed': SEED,
        'n_folds': N_FOLDS,
        'max_epochs': MAX_EPOCHS,
        'patience': PATIENCE,
        'lr': LEARNING_RATE,
        'dropout': DROPOUT,
        'weight_decay': WEIGHT_DECAY,
        'target_labels': TARGET_LABELS,
        'inputs': PRIMARY_INPUTS,
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
        'gpu_name': torch.cuda.get_device_name(0),
    }
    (run_root / 'setup.json').write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding='utf-8')

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    split_indices = list(splitter.split(labels_df['patient_id'].tolist(), labels_df['labelset_key'].tolist()))

    all_rows = []
    for preproc in preprocessings:
        for input_cfg in PRIMARY_INPUTS:
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
                    for fold_idx, (train_idx, val_idx) in enumerate(split_indices, start=1):
                        fold_dir = config_dir / f'fold_{fold_idx}'
                        fold_dir.mkdir(parents=True, exist_ok=True)
                        train_ids = labels_df.iloc[train_idx]['patient_id'].tolist()
                        val_ids = labels_df.iloc[val_idx]['patient_id'].tolist()
                        fold_summary = train_one_fold(
                            model_name=model_name,
                            labels_df=labels_df,
                            preproc_name=preproc['name'],
                            input_cfg=input_cfg,
                            batch_size=batch_size,
                            train_ids=train_ids,
                            val_ids=val_ids,
                            fold_dir=fold_dir,
                            device=device,
                        )
                        fold_summary['fold'] = fold_idx
                        fold_rows.append(fold_summary)
                        pred_path = fold_dir / 'val_predictions_best_epoch.csv'
                        if pred_path.exists():
                            oof_parts.append(pd.read_csv(pred_path))
                    fold_df = pd.DataFrame(fold_rows)
                    fold_df.to_csv(config_dir / 'fold_summary.csv', index=False)
                    if oof_parts:
                        oof_df = pd.concat(oof_parts, ignore_index=True)
                        oof_df.to_csv(config_dir / 'oof_predictions.csv', index=False)
                        y_true = oof_df[[f'true_{c}' for c in TARGET_LABELS]].values
                        y_prob = oof_df[[f'prob_{c}' for c in TARGET_LABELS]].values
                        oof_metrics = compute_metrics(y_true, y_prob, threshold=VAL_THRESHOLD)
                        (config_dir / 'oof_metrics.json').write_text(json.dumps(oof_metrics, ensure_ascii=False, indent=2), encoding='utf-8')
                        row['oof_f1_micro'] = float(oof_metrics['f1_micro'])
                        row['oof_f1_macro'] = float(oof_metrics['f1_macro'])
                    row['mean_f1_micro'] = float(fold_df['best_val_f1_micro'].mean())
                    row['std_f1_micro'] = float(fold_df['best_val_f1_micro'].std(ddof=0))
                    row['mean_peak_alloc_gb'] = float(fold_df['peak_alloc_gb'].mean())
                    row['max_peak_alloc_gb'] = float(fold_df['peak_alloc_gb'].max())
                    row['max_peak_reserved_gb'] = float(fold_df['peak_reserved_gb'].max())
                    row['mean_epochs'] = float(fold_df['num_epochs_ran'].mean())
                    row['mean_in_channels'] = float(fold_df['in_channels'].mean())
                except RuntimeError as e:
                    msg = str(e)
                    row['status'] = 'oom_or_runtime_error'
                    row['error_type'] = type(e).__name__
                    row['error_message'] = msg[:1000]
                    (config_dir / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
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
        'total_configs': int(len(summary_df)),
        'ok_configs': int((summary_df['status'] == 'ok').sum()),
        'failed_configs': int((summary_df['status'] != 'ok').sum()),
        'gradcam_top5_candidates': str(gradcam_plan),
    }
    (run_root / 'final_report.json').write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(final_report, ensure_ascii=False))
    return run_root
