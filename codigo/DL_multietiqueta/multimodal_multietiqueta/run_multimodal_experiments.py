#!/usr/bin/env python3
"""Run 15 paired folds for clinical, image and multimodal models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121, resnet10, resnet18, resnet34, seresnet50
from sklearn.metrics import precision_score, recall_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

MODULE_DIR = Path(__file__).resolve().parent

from multimodal_common import (
    CLINICAL_FEATURES,
    MODES,
    TARGET_LABELS,
    AlignedMultimodalDataset,
    FusionClassifier,
    fit_clinical_transform,
    load_and_validate_split,
    transform_clinical,
    validate_binary_labels,
    validate_outer_test_partition,
)

volume_common = None

PROJECT_ROOT = Path("/mnt/homeGPU/mcribilles/tfm")
CLINICAL_PATH = PROJECT_ROOT / "clinical_data/210pacientes/clinical_data_multimodal.csv"
DEFAULT_MANIFEST_PATH = MODULE_DIR / "selected_configuration.json"
RUNS_ROOT = MODULE_DIR / "runs"
DEFAULT_RUN_NAME = "seresnet50_best_test_f1_15runs"
SEED = 17
MAX_EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
THRESHOLD = 0.5
IMAGE_PROJECTION_DIM = 256
CLINICAL_PROJECTION_DIM = 64
FUSION_HIDDEN_DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pinned_volume_helper(manifest: dict):
    helper_path = Path(manifest["volume_helper_path"])
    if not helper_path.exists():
        raise FileNotFoundError(f"Pinned volume helper not found: {helper_path}")
    observed_hash = file_sha256(helper_path)
    expected_hash = manifest["volume_helper_sha256"]
    if observed_hash != expected_hash:
        raise ValueError(
            f"Pinned volume helper changed: expected {expected_hash}, found {observed_hash}"
        )
    spec = importlib.util.spec_from_file_location("pinned_volume_common", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned volume helper: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if list(module.TARGET_LABELS) != TARGET_LABELS:
        raise ValueError("Pinned helper target labels do not match the multimodal experiment")
    return module


def load_manifest(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_dir = Path(manifest["source_config_dir"])
    if not source_dir.exists():
        raise FileNotFoundError(f"Source configuration not found: {source_dir}")
    if manifest["source_num_folds_completed"] != 5:
        raise ValueError("The selected source configuration does not contain five folds")
    supported = {"resnet10", "resnet18", "resnet34", "seresnet50", "densenet121"}
    if manifest["selected_model"] not in supported:
        raise ValueError(f"Unsupported selected model: {manifest['selected_model']}")
    return manifest


def load_clinical_table() -> pd.DataFrame:
    if not CLINICAL_PATH.exists():
        raise FileNotFoundError(
            f"Clinical table not found: {CLINICAL_PATH}. Run prepare_clinical_multimodal.py first."
        )
    frame = pd.read_csv(CLINICAL_PATH, dtype={"patient_id": str})
    if len(frame) != 210:
        raise ValueError(f"Expected 210 clinical patients, found {len(frame)}")
    if frame["patient_id"].duplicated().any():
        raise ValueError("Duplicated patient identifiers in clinical table")
    return frame.set_index("patient_id")


def source_fold_paths(manifest: dict) -> list[Path]:
    source_dir = Path(manifest["source_config_dir"])
    paths = [source_dir / f"fold_{fold}" / "split_assignments.csv" for fold in range(1, 6)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source split files: {missing}")
    return paths


def expected_cohort_from_splits(manifest: dict) -> set[str]:
    cohort_sets = []
    for path in source_fold_paths(manifest):
        frame = pd.read_csv(path, dtype={"patient_id": str})
        cohort_sets.append(set(frame["patient_id"]))
    first = cohort_sets[0]
    if any(current != first for current in cohort_sets[1:]):
        raise ValueError("The source folds do not use the same patient cohort")
    if len(first) != int(manifest["source_cohort_size"]):
        raise ValueError(
            f"Source cohort mismatch: expected {manifest['source_cohort_size']}, found {len(first)}"
        )
    validate_outer_test_partition(source_fold_paths(manifest), first)
    return first


def load_labels(expected_ids: set[str]) -> pd.DataFrame:
    if volume_common is None:
        raise RuntimeError("Pinned volume helper has not been loaded")
    labels = volume_common.build_multilabel_dataframe(volume_common.CLINICAL_CSV)
    labels["patient_id"] = labels["patient_id"].astype(str)
    validate_binary_labels(labels)
    labels = labels[labels["patient_id"].isin(expected_ids)].copy().reset_index(drop=True)
    observed = set(labels["patient_id"])
    if observed != expected_ids:
        raise ValueError(
            f"Label cohort mismatch; missing={sorted(expected_ids - observed)}, "
            f"extra={sorted(observed - expected_ids)}"
        )
    validate_binary_labels(labels)
    return labels


def build_volume_dataset(
    labels: pd.DataFrame,
    patient_ids: list[str],
    manifest: dict,
):
    input_cfg = {
        "name": manifest["selected_input_name"],
        "channels": manifest["selected_channels"],
    }
    return volume_common.MultiInputVolumeDataset(
        labels,
        patient_ids,
        manifest["selected_preprocessing"],
        input_cfg,
    )


def build_image_encoder(model_name: str, in_channels: int) -> tuple[nn.Module, int]:
    if model_name == "resnet10":
        encoder = resnet10(spatial_dims=3, n_input_channels=in_channels, num_classes=len(TARGET_LABELS))
        image_dim = int(encoder.fc.in_features)
        encoder.fc = nn.Identity()
        return encoder, image_dim
    if model_name == "resnet18":
        encoder = resnet18(spatial_dims=3, n_input_channels=in_channels, num_classes=len(TARGET_LABELS))
        image_dim = int(encoder.fc.in_features)
        encoder.fc = nn.Identity()
        return encoder, image_dim
    if model_name == "resnet34":
        encoder = resnet34(spatial_dims=3, n_input_channels=in_channels, num_classes=len(TARGET_LABELS))
        image_dim = int(encoder.fc.in_features)
        encoder.fc = nn.Identity()
        return encoder, image_dim
    if model_name == "seresnet50":
        encoder = seresnet50(spatial_dims=3, in_channels=in_channels, num_classes=len(TARGET_LABELS))
        image_dim = int(encoder.last_linear.in_features)
        encoder.last_linear = nn.Identity()
        return encoder, image_dim
    if model_name == "densenet121":
        encoder = DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=len(TARGET_LABELS),
            dropout_prob=DROPOUT,
        )
        image_dim = int(encoder.class_layers.out.in_features)
        encoder.class_layers.out = nn.Identity()
        return encoder, image_dim
    raise ValueError(f"Unsupported selected model: {model_name}")


def build_model(mode: str, in_channels: int, manifest: dict) -> FusionClassifier:
    if mode == "clinical_only":
        image_encoder = nn.Identity()
        image_dim = 2048
    else:
        image_encoder, image_dim = build_image_encoder(manifest["selected_model"], in_channels)
    model = FusionClassifier(
        mode=mode,
        image_encoder=image_encoder,
        image_dim=image_dim,
        clinical_dim=len(CLINICAL_FEATURES),
        image_projection_dim=IMAGE_PROJECTION_DIM,
        clinical_projection_dim=CLINICAL_PROJECTION_DIM,
        hidden_dim=FUSION_HIDDEN_DIM,
        out_dim=len(TARGET_LABELS),
        dropout=DROPOUT,
    )
    if mode == "image_only":
        for parameter in model.clinical_encoder.parameters():
            parameter.requires_grad = False
    elif mode == "clinical_only":
        for parameter in model.image_encoder.parameters():
            parameter.requires_grad = False
        for parameter in model.image_projection.parameters():
            parameter.requires_grad = False
    return model


def make_dataset(
    mode: str,
    labels: pd.DataFrame,
    clinical: pd.DataFrame,
    patient_ids: list[str],
    manifest: dict,
) -> AlignedMultimodalDataset:
    volume_dataset = None
    if mode != "clinical_only":
        volume_dataset = build_volume_dataset(labels, patient_ids, manifest)
    return AlignedMultimodalDataset(
        labels_frame=labels,
        clinical_frame=clinical,
        patient_ids=patient_ids,
        mode=mode,
        volume_dataset=volume_dataset,
    )


def move_batch(batch, mode: str, device: torch.device):
    image, clinical, target, patient_ids = batch
    image_input = None if mode == "clinical_only" else image.to(device, non_blocking=True)
    clinical_input = None if mode == "image_only" else clinical.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    return image_input, clinical_input, target, list(patient_ids)


def compute_full_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    metrics = {
        key: float(value)
        for key, value in volume_common.compute_metrics(
            targets,
            probabilities,
            threshold=THRESHOLD,
        ).items()
    }
    predictions = (probabilities >= THRESHOLD).astype(int)
    for index, label in enumerate(TARGET_LABELS):
        metrics[f"precision_{label}"] = float(
            precision_score(targets[:, index], predictions[:, index], zero_division=0)
        )
        metrics[f"recall_{label}"] = float(
            recall_score(targets[:, index], predictions[:, index], zero_division=0)
        )
    return metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    mode: str,
    device: torch.device,
):
    weighted_loss_sum = 0.0
    sample_count = 0
    probabilities = []
    targets = []
    patient_ids = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image, clinical, target, ids = move_batch(batch, mode, device)
            with autocast(enabled=device.type == "cuda"):
                logits = model(image, clinical)
                loss = criterion(logits, target)
            batch_size = int(target.shape[0])
            weighted_loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(target.cpu().numpy())
            patient_ids.extend(ids)
    probability_matrix = np.concatenate(probabilities, axis=0)
    target_matrix = np.concatenate(targets, axis=0)
    return (
        weighted_loss_sum / sample_count,
        probability_matrix,
        target_matrix,
        patient_ids,
        compute_full_metrics(target_matrix, probability_matrix),
    )


def prediction_frame(
    fold: int,
    patient_ids: list[str],
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(probabilities, columns=[f"prob_{label}" for label in TARGET_LABELS])
    frame.insert(0, "patient_id", patient_ids)
    frame.insert(1, "fold", fold)
    for index, label in enumerate(TARGET_LABELS):
        frame[f"true_{label}"] = targets[:, index].astype(int)
        frame[f"pred_{label}"] = (probabilities[:, index] >= THRESHOLD).astype(int)
    return frame


def load_state(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_run_configuration(args: argparse.Namespace, manifest: dict) -> dict:
    code_files = [
        MODULE_DIR / "run_multimodal_experiments.py",
        MODULE_DIR / "multimodal_common.py",
        MODULE_DIR / "prepare_clinical_multimodal.py",
    ]
    return {
        "selected_configuration": manifest,
        "modes": list(args.modes),
        "folds": list(args.folds),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "num_workers": int(args.num_workers),
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "threshold": THRESHOLD,
        "seed": SEED,
        "clinical_features": CLINICAL_FEATURES,
        "target_labels": TARGET_LABELS,
        "code_sha256": {path.name: file_sha256(path) for path in code_files},
    }


def configuration_id(configuration: dict) -> str:
    payload = json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def initialize_run_root(
    run_root: Path,
    configuration: dict,
    device: torch.device,
) -> str:
    run_root.mkdir(parents=True, exist_ok=True)
    run_config_id = configuration_id(configuration)
    config_path = run_root / "run_configuration.json"
    expected = {"run_config_id": run_config_id, "configuration": configuration}
    if config_path.exists():
        observed = json.loads(config_path.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError(
                "Run directory contains an incompatible configuration. Use a new run name."
            )
    else:
        config_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    setup_path = run_root / "setup.json"
    if not setup_path.exists():
        setup_path.write_text(
            json.dumps(
                {
                    "created_at": datetime.now().isoformat(),
                    "run_config_id": run_config_id,
                    "selection_status": configuration["selected_configuration"]["selection_status"],
                    "selection_warning": configuration["selected_configuration"]["selection_warning"],
                    "cohort_size": configuration["selected_configuration"]["source_cohort_size"],
                    "device": str(device),
                    "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return run_config_id


def load_completed_fold(fold_dir: Path, run_config_id: str) -> dict | None:
    complete_path = fold_dir / "complete.json"
    if not complete_path.exists():
        return None
    summary = json.loads(complete_path.read_text(encoding="utf-8"))
    if summary.get("run_config_id") != run_config_id:
        raise ValueError(f"Incompatible completed fold: {fold_dir}")
    required = [
        fold_dir / "best_model.pt",
        fold_dir / "history.csv",
        fold_dir / "split_assignments.csv",
        fold_dir / "clinical_transform.json",
        fold_dir / "val_predictions_best_epoch.csv",
        fold_dir / "test_predictions_best_epoch.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Completed fold has missing artifacts: {missing}")
    return summary


def train_fold(
    mode: str,
    fold: int,
    labels: pd.DataFrame,
    clinical_raw: pd.DataFrame,
    expected_ids: set[str],
    manifest: dict,
    run_root: Path,
    device: torch.device,
    max_epochs: int,
    patience: int,
    num_workers: int,
    run_config_id: str,
) -> dict:
    fold_dir = run_root / mode / f"fold_{fold}"
    completed = load_completed_fold(fold_dir, run_config_id)
    if completed is not None:
        print(f"SKIP mode={mode} fold={fold}: compatible complete artifact exists", flush=True)
        return completed
    fold_dir.mkdir(parents=True, exist_ok=True)
    split_path = Path(manifest["source_config_dir"]) / f"fold_{fold}" / "split_assignments.csv"
    split = load_and_validate_split(split_path, expected_ids)
    shutil.copy2(split_path, fold_dir / "split_assignments.csv")

    transform_params = fit_clinical_transform(clinical_raw, split["train_inner"])
    clinical = transform_clinical(clinical_raw, transform_params)
    (fold_dir / "clinical_transform.json").write_text(
        json.dumps(transform_params, indent=2),
        encoding="utf-8",
    )

    datasets = {
        name: make_dataset(mode, labels, clinical, ids, manifest)
        for name, ids in split.items()
    }
    generator = torch.Generator().manual_seed(SEED + fold)
    loaders = {
        "train_inner": DataLoader(
            datasets["train_inner"],
            batch_size=int(manifest["batch_size"]),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            generator=generator,
        ),
        "val_inner": DataLoader(
            datasets["val_inner"],
            batch_size=int(manifest["batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        ),
        "test_outer": DataLoader(
            datasets["test_outer"],
            batch_size=int(manifest["batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        ),
    }

    if mode == "clinical_only":
        in_channels = int(manifest["expected_input_channels"])
    else:
        sample_image, _, _, _ = datasets["train_inner"][0]
        in_channels = int(sample_image.shape[0])
    if in_channels != int(manifest["expected_input_channels"]):
        raise ValueError(
            f"Expected {manifest['expected_input_channels']} image channels, found {in_channels}"
        )

    set_seed(SEED + fold)
    model = build_model(mode, in_channels, manifest).to(device)
    train_ids = split["train_inner"]
    pos_weight = volume_common.compute_pos_weight(labels, train_ids).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = GradScaler(enabled=device.type == "cuda")
    best_path = fold_dir / "best_model.pt"
    best_score = float("-inf")
    best_epoch = None
    stale_epochs = 0
    history = []

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        epoch_probabilities = []
        epoch_targets = []
        for batch in loaders["train_inner"]:
            image, clinical_x, target, _ = move_batch(batch, mode, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(image, clinical_x)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(target.shape[0])
            epoch_loss_sum += float(loss.item()) * batch_size
            epoch_sample_count += batch_size
            epoch_probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
            epoch_targets.append(target.detach().cpu().numpy())

        train_probabilities = np.concatenate(epoch_probabilities, axis=0)
        train_targets = np.concatenate(epoch_targets, axis=0)
        train_metrics = compute_full_metrics(train_targets, train_probabilities)
        val_loss, _, _, _, val_metrics = evaluate(
            model,
            loaders["val_inner"],
            criterion,
            mode,
            device,
        )
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss_sum / epoch_sample_count,
                "val_loss": val_loss,
                "train_f1_micro": train_metrics["f1_micro"],
                "val_f1_micro": val_metrics["f1_micro"],
                "val_f1_macro": val_metrics["f1_macro"],
                "lr": optimizer.param_groups[0]["lr"],
                "peak_alloc_gb": (
                    torch.cuda.max_memory_allocated() / (1024**3)
                    if device.type == "cuda"
                    else 0.0
                ),
                "peak_reserved_gb": (
                    torch.cuda.max_memory_reserved() / (1024**3)
                    if device.type == "cuda"
                    else 0.0
                ),
            }
        )
        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
        print(
            f"mode={mode} fold={fold} epoch={epoch:02d} "
            f"train_loss={history[-1]['train_loss']:.4f} val_loss={val_loss:.4f} "
            f"val_f1_micro={val_metrics['f1_micro']:.4f}",
            flush=True,
        )

        if val_metrics["f1_micro"] > best_score + 1e-4:
            best_score = val_metrics["f1_micro"]
            best_epoch = epoch
            stale_epochs = 0
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(state, best_path)
            del state
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            print(f"Early stopping mode={mode} fold={fold} epoch={epoch}", flush=True)
            break

    if best_epoch is None or not best_path.exists():
        raise RuntimeError("No model checkpoint was selected")
    model.load_state_dict(load_state(best_path, device))
    val_loss, val_prob, val_targets, val_ids, val_metrics = evaluate(
        model,
        loaders["val_inner"],
        criterion,
        mode,
        device,
    )
    test_loss, test_prob, test_targets, test_ids, test_metrics = evaluate(
        model,
        loaders["test_outer"],
        criterion,
        mode,
        device,
    )
    prediction_frame(fold, val_ids, val_prob, val_targets).to_csv(
        fold_dir / "val_predictions_best_epoch.csv",
        index=False,
    )
    prediction_frame(fold, test_ids, test_prob, test_targets).to_csv(
        fold_dir / "test_predictions_best_epoch.csv",
        index=False,
    )
    summary = {
        "run_config_id": run_config_id,
        "mode": mode,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_f1_micro": best_score,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "n_train_inner": len(split["train_inner"]),
        "n_val_inner": len(split["val_inner"]),
        "n_test_outer": len(split["test_outer"]),
        "in_channels": in_channels,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "max_epochs": max_epochs,
        "patience": patience,
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    (fold_dir / "complete.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def aggregate_mode(
    run_root: Path,
    mode: str,
    expected_ids: set[str],
    run_config_id: str,
) -> dict:
    prediction_paths = [
        run_root / mode / f"fold_{fold}" / "test_predictions_best_epoch.csv"
        for fold in range(1, 6)
    ]
    missing = [str(path) for path in prediction_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot aggregate {mode}; missing predictions: {missing}")
    oof = pd.concat([pd.read_csv(path, dtype={"patient_id": str}) for path in prediction_paths])
    if oof["patient_id"].duplicated().any():
        raise ValueError(f"Duplicated OOF patients for mode {mode}")
    if set(oof["patient_id"]) != expected_ids:
        raise ValueError(f"Incomplete OOF cohort for mode {mode}")
    oof = oof.sort_values("patient_id").reset_index(drop=True)
    targets = oof[[f"true_{label}" for label in TARGET_LABELS]].to_numpy()
    probabilities = oof[[f"prob_{label}" for label in TARGET_LABELS]].to_numpy()
    metrics = compute_full_metrics(targets, probabilities)
    oof.to_csv(run_root / mode / "oof_predictions.csv", index=False)
    (run_root / mode / "oof_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    fold_summaries = []
    for fold in range(1, 6):
        summary = load_completed_fold(run_root / mode / f"fold_{fold}", run_config_id)
        if summary is None:
            raise FileNotFoundError(f"Missing completed fold: mode={mode}, fold={fold}")
        fold_summaries.append(summary)
    pd.DataFrame(fold_summaries).to_csv(run_root / mode / "fold_summary.csv", index=False)
    return {"mode": mode, "n_oof": len(oof), **metrics}


def aggregate_all(
    run_root: Path,
    expected_ids: set[str],
    run_config_id: str,
) -> None:
    rows = [
        aggregate_mode(run_root, mode, expected_ids, run_config_id)
        for mode in MODES
    ]
    reference = None
    for mode in MODES:
        frame = pd.read_csv(run_root / mode / "oof_predictions.csv", dtype={"patient_id": str})
        columns = ["patient_id", *[f"true_{label}" for label in TARGET_LABELS]]
        current = frame[columns].sort_values("patient_id").reset_index(drop=True)
        if reference is None:
            reference = current
        elif not current.equals(reference):
            raise ValueError("OOF patients or targets differ between modalities")
    pd.DataFrame(rows).to_csv(run_root / "oof_metrics_all_modes.csv", index=False)
    (run_root / "complete.json").write_text(
        json.dumps(
            {
                "run_config_id": run_config_id,
                "completed_at": datetime.now().isoformat(),
                "num_fold_trainings": 15,
                "modes": list(MODES),
                "folds_per_mode": 5,
                "cohort_size": len(expected_ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def smoke_test(
    labels: pd.DataFrame,
    clinical_raw: pd.DataFrame,
    expected_ids: set[str],
    manifest: dict,
    device: torch.device,
) -> None:
    split_path = Path(manifest["source_config_dir"]) / "fold_1" / "split_assignments.csv"
    split = load_and_validate_split(split_path, expected_ids)
    params = fit_clinical_transform(clinical_raw, split["train_inner"])
    clinical = transform_clinical(clinical_raw, params)
    for mode in MODES:
        set_seed(SEED + 1)
        ids = split["train_inner"][:2]
        dataset = make_dataset(mode, labels, clinical, ids, manifest)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        image, clinical_x, target, _ = move_batch(batch, mode, device)
        in_channels = int(manifest["expected_input_channels"])
        if image is not None:
            in_channels = int(image.shape[1])
        model = build_model(mode, in_channels, manifest).to(device)
        criterion = nn.BCEWithLogitsLoss()
        with autocast(enabled=device.type == "cuda"):
            logits = model(image, clinical_x)
            loss = criterion(logits, target)
        loss.backward()
        print(
            f"SMOKE_OK mode={mode} logits={tuple(logits.shape)} loss={loss.item():.4f}",
            flush=True,
        )
        del model, dataset, loader, batch, logits, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    global volume_common
    args = parse_args()
    invalid_folds = sorted(set(args.folds) - {1, 2, 3, 4, 5})
    if invalid_folds:
        raise ValueError(f"Invalid folds: {invalid_folds}")
    manifest = load_manifest(args.manifest)
    volume_common = load_pinned_volume_helper(manifest)
    expected_ids = expected_cohort_from_splits(manifest)
    clinical = load_clinical_table()
    if not expected_ids.issubset(set(clinical.index.astype(str))):
        raise ValueError("Selected imaging cohort is not contained in the clinical table")
    labels = load_labels(expected_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke_test:
        smoke_test(labels, clinical, expected_ids, manifest, device)
        return
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the full multimodal experiment")

    run_root = RUNS_ROOT / args.run_name
    configuration = build_run_configuration(args, manifest)
    run_config_id = initialize_run_root(run_root, configuration, device)

    for mode in args.modes:
        for fold in args.folds:
            train_fold(
                mode=mode,
                fold=fold,
                labels=labels,
                clinical_raw=clinical,
                expected_ids=expected_ids,
                manifest=manifest,
                run_root=run_root,
                device=device,
                max_epochs=args.max_epochs,
                patience=args.patience,
                num_workers=args.num_workers,
                run_config_id=run_config_id,
            )
    if set(args.folds) == {1, 2, 3, 4, 5}:
        if set(args.modes) == set(MODES):
            aggregate_all(run_root, expected_ids, run_config_id)
            print(f"ALL_15_COMPLETE run_root={run_root}", flush=True)
        else:
            rows = [aggregate_mode(run_root, mode, expected_ids, run_config_id) for mode in args.modes]
            pd.DataFrame(rows).to_csv(run_root / "oof_metrics_selected_modes.csv", index=False)
            (run_root / "complete_selected_modes.json").write_text(
                json.dumps(
                    {
                        "run_config_id": run_config_id,
                        "completed_at": datetime.now().isoformat(),
                        "num_fold_trainings": len(args.modes) * 5,
                        "modes": list(args.modes),
                        "folds_per_mode": 5,
                        "cohort_size": len(expected_ids),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"SELECTED_MODES_COMPLETE run_root={run_root}", flush=True)
    else:
        print("PARTIAL_RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
