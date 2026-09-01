from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

MODULE_DIR = Path(__file__).resolve().parent
PARENT_DIR = MODULE_DIR.parent
PHASE_A_DIR = PARENT_DIR / "una_validacion_solo"
for path in (MODULE_DIR, PHASE_A_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import phase_a_serial_common as base
from two_point_five_d_core import (
    TARGET_LABELS,
    TwoPointFiveDDataset,
    build_resnet18_2d,
    validate_split_assignments,
)

PROJECT_ROOT = Path("/mnt/homeGPU/mcribilles/tfm")
DATA_ROOT = PROJECT_ROOT / "volumenes_preprocesados/resize_cube128_hu_m300_1400/npy"
REFERENCE_CONFIG = (
    PROJECT_ROOT
    / "codigo/DL_multietiqueta/val_externa_interna/runs"
    / "resnet18_nestedcv_focused_20260717_105720_resnet18_focused"
    / "resize_cube128__nodule_only_masked_ct__bs8"
)
SEED = 17
MAX_EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
THRESHOLD = 0.5
BATCH_SIZE = 32
NUM_WORKERS = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    probabilities = []
    targets = []
    patient_ids = []
    with torch.no_grad():
        for images, labels, batch_ids in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(labels.cpu().numpy())
            patient_ids.extend(str(value) for value in batch_ids)
    probability_matrix = np.concatenate(probabilities, axis=0)
    target_matrix = np.concatenate(targets, axis=0)
    metrics = base.compute_metrics(target_matrix, probability_matrix, threshold=THRESHOLD)
    return float(np.mean(losses)), probability_matrix, target_matrix, patient_ids, metrics


def prediction_frame(fold: int, patient_ids, probabilities, targets) -> pd.DataFrame:
    frame = pd.DataFrame(probabilities, columns=[f"prob_{label}" for label in TARGET_LABELS])
    frame.insert(0, "patient_id", patient_ids)
    frame.insert(1, "fold", f"fold_{fold}")
    for index, label in enumerate(TARGET_LABELS):
        frame[f"true_{label}"] = targets[:, index].astype(int)
        frame[f"pred_{label}"] = (probabilities[:, index] >= THRESHOLD).astype(int)
    return frame


def load_labels_and_split(fold: int):
    labels = base.build_multilabel_dataframe(base.CLINICAL_CSV).copy()
    labels["patient_id"] = labels["patient_id"].astype(str)
    split_path = REFERENCE_CONFIG / f"fold_{fold}" / "split_assignments.csv"
    split_frame = pd.read_csv(split_path)
    split_frame["patient_id"] = split_frame["patient_id"].astype(str)
    expected_ids = set(split_frame["patient_id"])
    validate_split_assignments(split_frame, expected_ids)
    labels = labels[labels["patient_id"].isin(expected_ids)].copy().reset_index(drop=True)
    if set(labels["patient_id"]) != expected_ids:
        raise RuntimeError("The clinical labels do not cover the reference 3D cohort")
    split_ids = {
        name: split_frame.loc[split_frame["split"] == name, "patient_id"].tolist()
        for name in ("train_inner", "val_inner", "test_outer")
    }
    return labels, split_frame, split_ids


def train_fold(representation: str, fold: int, run_dir: Path, max_epochs: int) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 2.5D training job")
    fold_seed = SEED + fold
    set_seed(fold_seed)
    device = torch.device("cuda")
    labels, split_frame, split_ids = load_labels_and_split(fold)
    output_dir = run_dir / representation / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_frame.to_csv(output_dir / "split_assignments.csv", index=False)

    datasets = {
        name: TwoPointFiveDDataset(labels, patient_ids, DATA_ROOT, representation)
        for name, patient_ids in split_ids.items()
    }
    loaders = {
        "train_inner": DataLoader(
            datasets["train_inner"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        ),
        "val_inner": DataLoader(
            datasets["val_inner"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        ),
        "test_outer": DataLoader(
            datasets["test_outer"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        ),
    }
    sample, _, _ = datasets["train_inner"][0]
    model = build_resnet18_2d(sample.shape[0], len(TARGET_LABELS), DROPOUT).to(device)
    pos_weight = base.compute_pos_weight(labels, split_ids["train_inner"]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = GradScaler(enabled=True)

    best_state = None
    best_score = -np.inf
    best_epoch = None
    best_val_payload = None
    patience_counter = 0
    history = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        train_probabilities = []
        train_targets = []
        for images, targets, _ in loaders["train_inner"]:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))
            train_probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
            train_targets.append(targets.detach().cpu().numpy())

        train_probability_matrix = np.concatenate(train_probabilities, axis=0)
        train_target_matrix = np.concatenate(train_targets, axis=0)
        train_metrics = base.compute_metrics(
            train_target_matrix,
            train_probability_matrix,
            threshold=THRESHOLD,
        )
        val_loss, val_probabilities, val_targets, val_ids, val_metrics = evaluate(
            model,
            loaders["val_inner"],
            criterion,
            device,
        )
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "train_f1_micro": train_metrics["f1_micro"],
                "train_f1_macro": train_metrics["f1_macro"],
                "val_f1_micro": val_metrics["f1_micro"],
                "val_f1_macro": val_metrics["f1_macro"],
                "peak_alloc_gb": torch.cuda.max_memory_allocated() / (1024**3),
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
            }
        )
        print(
            f"representation={representation} fold={fold} epoch={epoch:02d} "
            f"train_loss={np.mean(train_losses):.4f} val_loss={val_loss:.4f} "
            f"val_f1_micro={val_metrics['f1_micro']:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f}",
            flush=True,
        )
        if val_metrics["f1_micro"] > best_score + 1e-4:
            best_score = val_metrics["f1_micro"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            best_val_payload = (val_ids, val_probabilities.copy(), val_targets.copy())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best_state is None or best_val_payload is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "history.csv", index=False)
    torch.save(best_state, output_dir / "best_model.pt")
    val_ids, val_probabilities, val_targets = best_val_payload
    prediction_frame(fold, val_ids, val_probabilities, val_targets).to_csv(
        output_dir / "val_predictions_best_epoch.csv",
        index=False,
    )

    model.load_state_dict(best_state)
    test_loss, test_probabilities, test_targets, test_ids, test_metrics = evaluate(
        model,
        loaders["test_outer"],
        criterion,
        device,
    )
    prediction_frame(fold, test_ids, test_probabilities, test_targets).to_csv(
        output_dir / "test_predictions_best_epoch.csv",
        index=False,
    )
    summary = {
        "representation": representation,
        "fold": fold,
        "best_epoch": int(best_epoch),
        "best_val_f1_micro": float(best_score),
        "test_loss": test_loss,
        **{f"test_{key}": float(value) for key, value in test_metrics.items()},
        "num_epochs_ran": int(len(history_frame)),
        "in_channels": int(sample.shape[0]),
        "batch_size": BATCH_SIZE,
        "peak_alloc_gb": float(history_frame["peak_alloc_gb"].max()),
        "peak_reserved_gb": float(history_frame["peak_reserved_gb"].max()),
        "gpu_name": torch.cuda.get_device_name(0),
        "seed": fold_seed,
        "cohort_size": int(len(labels)),
        "data_root": str(DATA_ROOT),
        "reference_config": str(REFERENCE_CONFIG),
    }
    (output_dir / "fold_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=("axial5", "triplanar"), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_fold(args.representation, args.fold, args.run_dir, args.max_epochs)


if __name__ == "__main__":
    main()
