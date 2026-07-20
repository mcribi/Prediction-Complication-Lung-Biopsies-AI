#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import random
import sqlite3
import traceback
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_START_METHOD", "spawn")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import faulthandler
faulthandler.enable()

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    coverage_error,
    f1_score,
    hamming_loss,
    jaccard_score,
    label_ranking_average_precision_score,
    label_ranking_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    zero_one_loss,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
except Exception:
    RandomOverSampler = None
    SMOTE = None

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_FILE = SCRIPT_DIR / "valid_configurations_radiomics_multilabel_210.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "radiomic_results" / "multilabel_210"
DEFAULT_LABELS_CSV = Path("/mnt/homeGPU/mcribilles/tfm/clinical_data/210pacientes/clinical_data_limpios.csv")
DEFAULT_TARGET_COLS = ["Derrame_pleural", "Hemorragia", "Neumotórax", "Sin_complicación"]
NO_COMPLICATION_COL = "Sin_complicación"


def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Radiomics multilabel classification runner for the 210-patient cohort."
    )
    parser.add_argument("config_index", type=int, help="Configuration index to execute")
    parser.add_argument("--config_file", type=str, default=str(DEFAULT_CONFIG_FILE))
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--features_csv", type=str, default=None)
    parser.add_argument("--labels_csv", type=str, default=str(DEFAULT_LABELS_CSV))
    parser.add_argument("--mapping_csv", type=str, default=None)
    parser.add_argument("--random_state", type=int, default=None)
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--job_suffix", type=str, default=None)
    parser.add_argument("--no_sqlite", action="store_true")
    return parser.parse_args()


def _norm_text(value: Any) -> str:
    value = "".join(ch for ch in unicodedata.normalize("NFKC", str(value)) if ch != "\ufeff")
    return value.strip()


def _key(value: Any) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return value.lower().strip()


def _rename_cols_like(df: pd.DataFrame, target_name: str, *aliases: str) -> bool:
    wanted = {_key(target_name)} | {_key(alias) for alias in aliases}
    mapping = {_key(col): col for col in df.columns}
    for key in wanted:
        if key in mapping:
            real_col = mapping[key]
            if real_col != target_name:
                df.rename(columns={real_col: target_name}, inplace=True)
            return True
    return False


def _norm_id_series(series: pd.Series) -> pd.Series:
    series = series.astype(str).str.strip()
    series = series.str.replace(r"[, ]+", "", regex=True)
    series = series.str.replace(r"_(seg|lung|nodule)$", "", regex=True)
    return series


def resolve_path(path_value: Optional[str], base_dir: Path) -> Optional[Path]:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_configurations(config_file: Path) -> List[Dict[str, Any]]:
    with config_file.open("r", encoding="utf-8") as f:
        configs = json.load(f)
    if not isinstance(configs, list):
        raise ValueError("config_file must contain a list of configurations")
    return configs


def extract_classifier_params(config: Dict[str, Any]) -> Dict[str, Any]:
    reserved = {
        "config_name",
        "features_csv",
        "labels_csv",
        "mapping_csv",
        "id_col",
        "target_cols",
        "label_mode",
        "use_masks",
        "n_splits",
        "random_state",
        "shuffle_folds",
        "scaler",
        "classifier",
        "dimred",
        "multilabel_strategy",
        "preproc_names",
        "save_models",
        "save_predictions",
        "output_subdir",
        "split_strategy",
        "sampler_name",
        "sampler_kwargs",
    }
    return {k: v for k, v in config.items() if k not in reserved}


def normalize_sampler_name(sampler_name: Optional[str]) -> str:
    if sampler_name in {None, "", "none", "None"}:
        return "none"
    name = str(sampler_name).strip().lower()
    alias_map = {
        "randomoversampler": "random_over_sampler",
        "random_over_sampler": "random_over_sampler",
        "ros": "random_over_sampler",
        "smote": "smote",
    }
    if name not in alias_map:
        raise ValueError(f"Unsupported sampler_name: {sampler_name}")
    return alias_map[name]


def build_sampler(sampler_name: str, random_state: int, sampler_kwargs: Optional[Dict[str, Any]], y: np.ndarray):
    sampler_kwargs = dict(sampler_kwargs or {})
    if sampler_name == "random_over_sampler":
        if RandomOverSampler is None:
            raise RuntimeError("imblearn is not available for RandomOverSampler")
        sampler_kwargs.setdefault("random_state", random_state)
        return RandomOverSampler(**sampler_kwargs)
    if sampler_name == "smote":
        if SMOTE is None:
            raise RuntimeError("imblearn is not available for SMOTE")
        bincount = np.bincount(np.asarray(y, dtype=int))
        positive_count = int(bincount[1]) if len(bincount) > 1 else 0
        if positive_count < 2:
            raise ValueError("SMOTE requires at least 2 positive samples in the training fold")
        requested_k = int(sampler_kwargs.get("k_neighbors", 5))
        sampler_kwargs["k_neighbors"] = max(1, min(requested_k, positive_count - 1))
        sampler_kwargs.setdefault("random_state", random_state)
        return SMOTE(**sampler_kwargs)
    raise ValueError(f"Unsupported sampler_name: {sampler_name}")


class BinarySamplerWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_estimator: Any,
        sampler_name: Optional[str] = None,
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ):
        self.base_estimator = base_estimator
        self.sampler_name = sampler_name
        self.sampler_kwargs = sampler_kwargs
        self.random_state = random_state

    def fit(self, x, y):
        x_arr = np.asarray(x)
        y_arr = np.asarray(y).astype(int).ravel()
        estimator = clone(self.base_estimator)
        sampler_name = normalize_sampler_name(self.sampler_name)
        if sampler_name == "none" or np.unique(y_arr).size < 2:
            estimator.fit(x_arr, y_arr)
        else:
            sampler = build_sampler(
                sampler_name=sampler_name,
                random_state=self.random_state,
                sampler_kwargs=self.sampler_kwargs,
                y=y_arr,
            )
            x_resampled, y_resampled = sampler.fit_resample(x_arr, y_arr)
            estimator.fit(x_resampled, y_resampled)
        self.estimator_ = estimator
        self.classes_ = getattr(estimator, "classes_", np.array([0, 1]))
        return self

    def predict(self, x):
        return self.estimator_.predict(x)

    def predict_proba(self, x):
        if hasattr(self.estimator_, "predict_proba"):
            return self.estimator_.predict_proba(x)
        raise AttributeError("Underlying estimator does not support predict_proba")

    def decision_function(self, x):
        if hasattr(self.estimator_, "decision_function"):
            return self.estimator_.decision_function(x)
        raise AttributeError("Underlying estimator does not support decision_function")


def get_output_dir(output_root: Path, config_index: int, config: Dict[str, Any], job_suffix: Optional[str]) -> Path:
    config_name = config.get("config_name") or f"config_{config_index:06d}"
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(config_name))
    if job_suffix:
        safe_name = f"{safe_name}__{job_suffix}"
    output_dir = output_root / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_features_csv(features_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(features_csv)
    if "patient_id" not in df.columns:
        raise ValueError("features_csv must contain patient_id")
    return df


def load_labels_csv(labels_csv: Path, id_col: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [_norm_text(col) for col in df.columns]
    ok_id = _rename_cols_like(df, id_col, "patient_id", "id paciente", "id_paciente", "paciente_id", "id")
    if not ok_id:
        raise ValueError(f"labels_csv must contain id column compatible with {id_col}")
    return df


def load_mapping_csv(mapping_csv: Optional[Path], id_col: str) -> Optional[pd.DataFrame]:
    if mapping_csv is None:
        return None
    if not mapping_csv.exists():
        raise FileNotFoundError(f"mapping_csv not found: {mapping_csv}")
    df = pd.read_csv(mapping_csv)
    if "patient_id" not in df.columns or id_col not in df.columns:
        raise ValueError(f"mapping_csv must contain patient_id and {id_col}")
    df = df.dropna(subset=[id_col]).copy()
    df[id_col] = df[id_col].astype(str).str.strip()
    return df


def merge_features_and_labels(
    df_feats: pd.DataFrame,
    df_labels: pd.DataFrame,
    id_col: str,
    target_cols: List[str],
    mapping_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report: Dict[str, Any] = {
        "merge_strategy": None,
        "n_features_rows_before": int(len(df_feats)),
        "n_features_patients_before": int(df_feats["patient_id"].nunique()),
        "n_labels_rows_before": int(len(df_labels)),
        "n_labels_patients_before": int(df_labels[id_col].nunique()),
    }

    missing_target_cols = [col for col in target_cols if col not in df_labels.columns]
    if missing_target_cols:
        raise ValueError(f"labels_csv is missing target columns: {missing_target_cols}")

    tmp = df_feats.merge(df_labels[[id_col] + target_cols], left_on="patient_id", right_on=id_col, how="left")
    direct_matches = int(tmp[target_cols].notna().all(axis=1).sum())
    if direct_matches > 0:
        merged = tmp
        strategy = "direct"
    else:
        df_feats_norm = df_feats.copy()
        df_labels_norm = df_labels.copy()
        df_feats_norm["_id_norm"] = _norm_id_series(df_feats_norm["patient_id"])
        df_labels_norm["_id_norm"] = _norm_id_series(df_labels_norm[id_col])
        tmp2 = df_feats_norm.merge(df_labels_norm[["_id_norm"] + target_cols], on="_id_norm", how="left")
        normalized_matches = int(tmp2[target_cols].notna().all(axis=1).sum())
        if normalized_matches > 0:
            merged = tmp2
            strategy = "normalized"
        elif mapping_df is not None:
            df_feats_map = df_feats.copy().merge(mapping_df[["patient_id", id_col]], on="patient_id", how="left")
            tmp3 = df_feats_map.merge(df_labels[[id_col] + target_cols], on=id_col, how="left")
            mapping_matches = int(tmp3[target_cols].notna().all(axis=1).sum())
            if mapping_matches == 0:
                raise ValueError("No labels matched after direct, normalized, and mapping merge")
            merged = tmp3
            strategy = "mapping"
        else:
            raise ValueError("No labels matched after direct and normalized merge")

    for aux_col in ["_id_norm", id_col]:
        if aux_col in merged.columns and aux_col != "patient_id":
            merged.drop(columns=[aux_col], inplace=True, errors="ignore")

    report["merge_strategy"] = strategy
    report["n_rows_after_merge"] = int(len(merged))
    report["n_patients_after_merge"] = int(merged["patient_id"].nunique())
    report["n_rows_with_all_targets"] = int(merged[target_cols].notna().all(axis=1).sum())
    return merged, report


def filter_preproc_names(df_feats: pd.DataFrame, preproc_names: Optional[List[str]]) -> pd.DataFrame:
    if not preproc_names:
        return df_feats
    if "preproc_name" not in df_feats.columns:
        raise ValueError("preproc_names requested but features_csv has no preproc_name column")
    out = df_feats[df_feats["preproc_name"].isin(preproc_names)].copy()
    if out.empty:
        raise ValueError(f"No rows left after filtering preproc_names={preproc_names}")
    return out


def _available_mask_kinds(df: pd.DataFrame) -> List[str]:
    if "mask_kind" not in df.columns:
        return []
    masks = []
    for value in df["mask_kind"].dropna().tolist():
        mask = str(value).strip()
        if mask:
            masks.append(mask)
    return sorted(set(masks))


def _parse_use_masks(use_masks: Any, available_masks: List[str]) -> Tuple[str, List[str], str]:
    if isinstance(use_masks, list):
        selected = [str(mask).strip() for mask in use_masks if str(mask).strip()]
        if not selected:
            raise ValueError("use_masks list is empty")
        if len(selected) == 1:
            return "single", selected, selected[0]
        return "merge", selected, "+".join(selected)

    use_masks_str = str(use_masks).strip()
    if not use_masks_str:
        raise ValueError("use_masks cannot be empty")

    if use_masks_str in {"all_merge", "available_merge"}:
        if not available_masks:
            raise ValueError("use_masks requested all_merge but no mask_kind column is available")
        return "merge", list(available_masks), use_masks_str

    if use_masks_str == "both_merge":
        return "merge", ["lung", "nodule"], use_masks_str

    if use_masks_str.startswith("merge:"):
        selected = [token.strip() for token in use_masks_str.split(":", 1)[1].split("+") if token.strip()]
        if not selected:
            raise ValueError("use_masks merge: specification is empty")
        return "merge", selected, use_masks_str

    if "+" in use_masks_str:
        selected = [token.strip() for token in use_masks_str.split("+") if token.strip()]
        if not selected:
            raise ValueError("use_masks combination is empty")
        if len(selected) == 1:
            return "single", selected, selected[0]
        return "merge", selected, use_masks_str

    return "single", [use_masks_str], use_masks_str


def _merge_selected_masks(df: pd.DataFrame, selected_masks: List[str], target_cols: List[str]) -> pd.DataFrame:
    labels = df[["patient_id"] + target_cols].drop_duplicates(subset=["patient_id"]).copy()
    meta_cols = [col for col in ["preproc_name", "image_path", "image_size", "image_spacing"] if col in df.columns]
    excluded_cols = {"patient_id", "mask_kind"} | set(target_cols) | set(meta_cols)
    feat_cols = [col for col in df.columns if col not in excluded_cols]
    if not feat_cols:
        raise ValueError("No feature columns available to merge across masks")

    parts = []
    for mask_kind in selected_masks:
        sub = df[df["mask_kind"] == mask_kind].copy()
        if sub.empty:
            raise ValueError(f"No rows found for requested mask_kind={mask_kind}")
        dup_ids = sub.loc[sub["patient_id"].duplicated(), "patient_id"].astype(str).unique().tolist()
        if dup_ids:
            preview = dup_ids[:10]
            raise ValueError(
                f"Mask {mask_kind} has duplicated patient_id rows after filtering; cannot build a unique wide merge. Examples: {preview}"
            )
        keep_cols = ["patient_id"] + meta_cols + feat_cols
        sub = sub[keep_cols].copy()
        rename_map = {col: f"{mask_kind}_{col}" for col in meta_cols + feat_cols}
        sub.rename(columns=rename_map, inplace=True)
        parts.append(sub)

    df_wide = parts[0]
    for part in parts[1:]:
        df_wide = df_wide.merge(part, on="patient_id", how="outer")
    return df_wide.merge(labels, on="patient_id", how="left")


def filter_mask_mode(df: pd.DataFrame, use_masks: Any, target_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report: Dict[str, Any] = {
        "use_masks_requested": use_masks,
        "available_mask_kinds_in_features": _available_mask_kinds(df),
        "selected_mask_kinds": [],
        "use_masks_resolved": use_masks,
        "mask_merge_mode": "not_applicable",
    }

    if "mask_kind" not in df.columns:
        return df, report

    available_masks = report["available_mask_kinds_in_features"]
    mode, selected_masks, resolved = _parse_use_masks(use_masks, available_masks)
    missing_masks = [mask for mask in selected_masks if mask not in available_masks]
    if missing_masks:
        raise ValueError(
            f"Requested masks not present in features_csv: {missing_masks}. Available mask_kind values: {available_masks}"
        )

    report["selected_mask_kinds"] = list(selected_masks)
    report["use_masks_resolved"] = resolved
    report["mask_merge_mode"] = mode

    if mode == "single":
        out = df[df["mask_kind"] == selected_masks[0]].copy()
        out.drop(columns=["mask_kind"], inplace=True)
        return out, report

    out = _merge_selected_masks(df=df, selected_masks=selected_masks, target_cols=target_cols)
    report["n_rows_after_mask_merge"] = int(len(out))
    report["n_patients_after_mask_merge"] = int(out["patient_id"].nunique())
    return out, report


def ensure_binary_targets(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in target_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=target_cols).copy()
    for col in target_cols:
        out[col] = out[col].astype(int)
        invalid = sorted(set(out[col].unique()) - {0, 1})
        if invalid:
            raise ValueError(f"Target column {col} contains values outside 0/1: {invalid}")
    return out


def apply_label_mode(df: pd.DataFrame, target_cols: List[str], label_mode: str) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    out = df.copy()
    if label_mode == "include_no_complication":
        final_target_cols = list(target_cols)
    elif label_mode == "exclude_no_complication":
        final_target_cols = [col for col in target_cols if col != NO_COMPLICATION_COL]
        if not final_target_cols:
            raise ValueError("No target columns remain after excluding no-complication label")
    elif label_mode == "derive_no_complication":
        complication_cols = [col for col in target_cols if col != NO_COMPLICATION_COL]
        if not complication_cols:
            raise ValueError("derive_no_complication requires complication columns other than Sin_complicacion")
        out[NO_COMPLICATION_COL] = (out[complication_cols].sum(axis=1) == 0).astype(int)
        final_target_cols = complication_cols + [NO_COMPLICATION_COL]
    else:
        raise ValueError(
            "label_mode must be include_no_complication, exclude_no_complication, or derive_no_complication"
        )

    info = {
        "label_mode": label_mode,
        "target_cols_input": list(target_cols),
        "target_cols_final": list(final_target_cols),
    }
    return out, final_target_cols, info


def summarize_label_support(df: pd.DataFrame, target_cols: List[str]) -> Dict[str, Any]:
    support = {col: int(df[col].sum()) for col in target_cols}
    label_cardinality = float(df[target_cols].sum(axis=1).mean()) if len(df) else 0.0
    label_density = float(label_cardinality / max(len(target_cols), 1))
    labelsets = df[target_cols].astype(str).agg("".join, axis=1).value_counts().to_dict() if len(df) else {}
    return {
        "label_support": support,
        "label_cardinality_mean": label_cardinality,
        "label_density": label_density,
        "labelset_counts": labelsets,
    }


def prepare_feature_columns(df: pd.DataFrame, target_cols: List[str]) -> List[str]:
    base_drop_cols = {"patient_id", "mask_kind"} | set(target_cols)
    metadata_suffixes = ("preproc_name", "image_path", "image_size", "image_spacing")

    def is_metadata_column(col: str) -> bool:
        if col in base_drop_cols:
            return True
        return any(col == suffix or col.endswith(f"_{suffix}") for suffix in metadata_suffixes)

    candidate_cols = [col for col in df.columns if not is_metadata_column(col)]
    if not candidate_cols:
        raise ValueError("No feature columns available after dropping metadata and target columns")

    numeric_feature_cols = []
    for col in candidate_cols:
        series_num = pd.to_numeric(df[col], errors="coerce")
        if series_num.notna().any():
            numeric_feature_cols.append(col)

    if not numeric_feature_cols:
        raise ValueError("No numeric feature columns remain after filtering non-numeric data")
    return numeric_feature_cols


def build_multilabel_strata(y: np.ndarray) -> np.ndarray:
    label_strings = np.array(["|".join(map(str, row.astype(int).tolist())) for row in y])
    values, counts = np.unique(label_strings, return_counts=True)
    count_map = dict(zip(values, counts))
    adjusted = np.array([
        label if count_map[label] >= 2 else "RARE_LABELSET"
        for label in label_strings
    ])
    return adjusted


def build_splitter(y: np.ndarray, n_splits: int, shuffle_folds: bool, random_state: int, split_strategy: str):
    if split_strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=shuffle_folds, random_state=random_state)
    if split_strategy in {"auto", "labelset_stratified"}:
        strata = build_multilabel_strata(y)
        unique_values, counts = np.unique(strata, return_counts=True)
        min_count = counts.min() if len(counts) else 0
        if len(unique_values) >= 2 and min_count >= n_splits:
            return StratifiedKFold(n_splits=n_splits, shuffle=shuffle_folds, random_state=random_state), strata
        if split_strategy == "labelset_stratified":
            raise ValueError(
                f"labelset_stratified requested but minimum labelset support is {min_count}, below n_splits={n_splits}"
            )
        return KFold(n_splits=n_splits, shuffle=shuffle_folds, random_state=random_state)
    raise ValueError("split_strategy must be auto, labelset_stratified, or kfold")


def build_base_classifier(classifier_name: str, classifier_params: Dict[str, Any], random_state: int):
    params = dict(classifier_params)
    if classifier_name == "RandomForest":
        return RandomForestClassifier(random_state=random_state, **params)
    if classifier_name == "ExtraTrees":
        return ExtraTreesClassifier(random_state=random_state, **params)
    if classifier_name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=random_state, **params)
    if classifier_name == "AdaBoost":
        return AdaBoostClassifier(random_state=random_state, **params)
    if classifier_name == "LogisticRegression":
        params.setdefault("max_iter", 1000)
        return LogisticRegression(random_state=random_state, **params)
    if classifier_name == "DecisionTree":
        return DecisionTreeClassifier(random_state=random_state, **params)
    if classifier_name == "KNN":
        return KNeighborsClassifier(**params)
    if classifier_name == "MLP":
        params.setdefault("max_iter", 500)
        return MLPClassifier(random_state=random_state, **params)
    if classifier_name == "SVM":
        kernel = params.pop("kernel", "rbf")
        C = params.pop("C", 1.0)
        gamma = params.pop("gamma", "scale")
        tol = params.pop("tol", 1e-3)
        max_iter = params.pop("max_iter", 5000)
        if kernel == "linear":
            return LinearSVC(C=C, tol=tol, max_iter=max_iter, **params)
        return SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=False,
            tol=tol,
            max_iter=max_iter,
            cache_size=1000,
            **params,
        )
    if classifier_name == "XGBoost":
        try:
            import xgboost as xgb
        except Exception as exc:
            raise RuntimeError(f"Could not import xgboost: {exc}")
        return xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            **params,
        )
    if classifier_name == "LightGBM":
        try:
            import lightgbm as lgb
        except Exception as exc:
            raise RuntimeError(f"Could not import lightgbm: {exc}")
        return lgb.LGBMClassifier(random_state=random_state, **params)
    if classifier_name == "CatBoost":
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:
            raise RuntimeError(f"Could not import catboost: {exc}")
        params.setdefault("verbose", False)
        params.setdefault("random_seed", random_state)
        return CatBoostClassifier(**params)
    if classifier_name == "TabPFN":
        try:
            from tabpfn import TabPFNClassifier
        except Exception as exc:
            raise RuntimeError(f"Could not import tabpfn: {exc}")
        if "random_state" in params and "seed" not in params:
            params["seed"] = params.pop("random_state")
        params.setdefault("seed", random_state)
        return TabPFNClassifier(**params)
    raise ValueError(f"Unsupported classifier: {classifier_name}")


def wrap_multilabel_classifier(
    base_classifier,
    multilabel_strategy: str,
    sampler_name: Optional[str],
    sampler_kwargs: Optional[Dict[str, Any]],
    random_state: int,
):
    wrapped_base = BinarySamplerWrapper(
        base_estimator=base_classifier,
        sampler_name=sampler_name,
        sampler_kwargs=sampler_kwargs,
        random_state=random_state,
    )
    if multilabel_strategy == "one_vs_rest":
        return OneVsRestClassifier(wrapped_base)
    if multilabel_strategy == "multi_output":
        return MultiOutputClassifier(wrapped_base)
    if multilabel_strategy == "classifier_chain":
        return ClassifierChain(wrapped_base)
    raise ValueError("multilabel_strategy must be one_vs_rest, multi_output, or classifier_chain")


def build_pipeline(
    classifier_name: str,
    classifier_params: Dict[str, Any],
    scaler_choice: Optional[str],
    dimred_choice: str,
    multilabel_strategy: str,
    random_state: int,
    sampler_name: Optional[str] = None,
    sampler_kwargs: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]

    pca_map = {"pca_99": 0.99, "pca_95": 0.95, "pca_90": 0.90}
    use_pca = dimred_choice in pca_map
    use_vt = dimred_choice == "varth_0.01"

    if use_pca:
        steps.append(("scaler_pca", StandardScaler()))
        steps.append(("pca", PCA(n_components=pca_map[dimred_choice], svd_solver="full", random_state=random_state)))
    elif use_vt:
        steps.append(("vt", VarianceThreshold(threshold=0.01)))
    elif dimred_choice not in {"none", None}:
        raise ValueError(f"Unsupported dimred choice: {dimred_choice}")

    force_scaler = classifier_name == "SVM"
    if not use_pca:
        if scaler_choice == "StandardScaler" or force_scaler:
            steps.append(("scaler", StandardScaler()))
        elif scaler_choice == "MinMaxScaler":
            steps.append(("scaler", MinMaxScaler()))
        elif scaler_choice in {None, "None", "none"}:
            pass
        else:
            raise ValueError(f"Unsupported scaler: {scaler_choice}")

    base_classifier = build_base_classifier(classifier_name, classifier_params, random_state=random_state)
    wrapped_classifier = wrap_multilabel_classifier(
        base_classifier,
        multilabel_strategy=multilabel_strategy,
        sampler_name=sampler_name,
        sampler_kwargs=sampler_kwargs,
        random_state=random_state,
    )
    steps.append(("classifier", wrapped_classifier))
    return Pipeline(steps)


def get_score_matrix(pipeline: Pipeline, x_test: np.ndarray) -> Optional[np.ndarray]:
    classifier = pipeline.named_steps["classifier"]
    try:
        if hasattr(classifier, "predict_proba"):
            proba = classifier.predict_proba(x_test)
            if isinstance(proba, list):
                columns = []
                for item in proba:
                    item = np.asarray(item)
                    if item.ndim == 2 and item.shape[1] >= 2:
                        columns.append(item[:, 1])
                    elif item.ndim == 1:
                        columns.append(item)
                    else:
                        columns.append(item[:, 0])
                return np.vstack(columns).T
            proba = np.asarray(proba)
            if proba.ndim == 3 and proba.shape[2] >= 2:
                return proba[:, :, 1]
            return proba
        if hasattr(classifier, "decision_function"):
            scores = classifier.decision_function(x_test)
            return np.asarray(scores)
    except Exception:
        return None
    return None


def safe_metric(metric_fn, *args, **kwargs):
    try:
        value = metric_fn(*args, **kwargs)
    except Exception:
        return None
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.item()
    try:
        value = float(value)
    except Exception:
        return None
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def is_probability_matrix(y_score: Optional[np.ndarray]) -> bool:
    if y_score is None:
        return False
    y_score = np.asarray(y_score)
    if y_score.size == 0 or not np.isfinite(y_score).all():
        return False
    return bool(((y_score >= 0.0) & (y_score <= 1.0)).all())


def compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray]) -> Dict[str, Any]:
    probability_like = is_probability_matrix(y_score)
    metrics = {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "zero_one_loss": float(zero_one_loss(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "hamming_score": float(1.0 - hamming_loss(y_true, y_pred)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_samples": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_samples": float(precision_score(y_true, y_pred, average="samples", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_samples": float(recall_score(y_true, y_pred, average="samples", zero_division=0)),
        "jaccard_micro": float(jaccard_score(y_true, y_pred, average="micro", zero_division=0)),
        "jaccard_macro": float(jaccard_score(y_true, y_pred, average="macro", zero_division=0)),
        "jaccard_weighted": float(jaccard_score(y_true, y_pred, average="weighted", zero_division=0)),
        "jaccard_samples": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
    }
    if y_score is not None:
        metrics["roc_auc_macro"] = safe_metric(roc_auc_score, y_true, y_score, average="macro")
        metrics["roc_auc_micro"] = safe_metric(roc_auc_score, y_true, y_score, average="micro")
        metrics["roc_auc_weighted"] = safe_metric(roc_auc_score, y_true, y_score, average="weighted")
        metrics["roc_auc_samples"] = safe_metric(roc_auc_score, y_true, y_score, average="samples")
        metrics["label_ranking_average_precision"] = safe_metric(
            label_ranking_average_precision_score,
            y_true,
            y_score,
        )
        metrics["label_ranking_loss"] = safe_metric(label_ranking_loss, y_true, y_score)
        metrics["coverage_error"] = safe_metric(coverage_error, y_true, y_score)
        if probability_like:
            metrics["average_precision_macro"] = safe_metric(
                average_precision_score,
                y_true,
                y_score,
                average="macro",
            )
            metrics["average_precision_micro"] = safe_metric(
                average_precision_score,
                y_true,
                y_score,
                average="micro",
            )
            metrics["average_precision_weighted"] = safe_metric(
                average_precision_score,
                y_true,
                y_score,
                average="weighted",
            )
            metrics["average_precision_samples"] = safe_metric(
                average_precision_score,
                y_true,
                y_score,
                average="samples",
            )
        else:
            metrics["average_precision_macro"] = None
            metrics["average_precision_micro"] = None
            metrics["average_precision_weighted"] = None
            metrics["average_precision_samples"] = None
    else:
        metrics["roc_auc_macro"] = None
        metrics["roc_auc_micro"] = None
        metrics["roc_auc_weighted"] = None
        metrics["roc_auc_samples"] = None
        metrics["average_precision_macro"] = None
        metrics["average_precision_micro"] = None
        metrics["average_precision_weighted"] = None
        metrics["average_precision_samples"] = None
        metrics["label_ranking_average_precision"] = None
        metrics["label_ranking_loss"] = None
        metrics["coverage_error"] = None
    return metrics


def compute_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    target_cols: List[str],
    fold_idx: Any,
    scope: str = "fold",
) -> List[Dict[str, Any]]:
    probability_like = is_probability_matrix(y_score)
    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    rows: List[Dict[str, Any]] = []
    for idx, label_name in enumerate(target_cols):
        tn, fp, fn, tp = confusion_matrix(y_true[:, idx], y_pred[:, idx], labels=[0, 1]).ravel()
        row = {
            "scope": scope,
            "fold": int(fold_idx),
            "label": label_name,
            "accuracy": float(accuracy_score(y_true[:, idx], y_pred[:, idx])),
            "balanced_accuracy": safe_metric(balanced_accuracy_score, y_true[:, idx], y_pred[:, idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1_values[idx]),
            "jaccard": float(jaccard_score(y_true[:, idx], y_pred[:, idx], zero_division=0)),
            "support": int(support[idx]),
            "prevalence": int(y_true[:, idx].sum()),
            "predicted_positives": int(y_pred[:, idx].sum()),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "specificity": None if (tn + fp) == 0 else float(tn / (tn + fp)),
            "npv": None if (tn + fn) == 0 else float(tn / (tn + fn)),
            "fpr": None if (fp + tn) == 0 else float(fp / (fp + tn)),
            "fnr": None if (fn + tp) == 0 else float(fn / (fn + tp)),
            "mcc": safe_metric(matthews_corrcoef, y_true[:, idx], y_pred[:, idx]),
            "cohen_kappa": safe_metric(cohen_kappa_score, y_true[:, idx], y_pred[:, idx]),
        }
        if y_score is not None and y_score.ndim == 2 and idx < y_score.shape[1]:
            row["roc_auc"] = safe_metric(roc_auc_score, y_true[:, idx], y_score[:, idx])
            if probability_like:
                row["average_precision"] = safe_metric(average_precision_score, y_true[:, idx], y_score[:, idx])
                row["brier_score"] = safe_metric(brier_score_loss, y_true[:, idx], y_score[:, idx])
            else:
                row["average_precision"] = None
                row["brier_score"] = None
        else:
            row["roc_auc"] = None
            row["average_precision"] = None
            row["brier_score"] = None
        rows.append(row)
    return rows


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def init_sqlite(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS multilabel_experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_index INTEGER,
                config_name TEXT,
                features_csv TEXT,
                labels_csv TEXT,
                use_masks TEXT,
                label_mode TEXT,
                multilabel_strategy TEXT,
                classifier TEXT,
                scaler TEXT,
                dimred TEXT,
                target_cols_json TEXT,
                classifier_params_json TEXT,
                fold INTEGER,
                subset_accuracy REAL,
                hamming_loss REAL,
                f1_micro REAL,
                f1_macro REAL,
                f1_samples REAL,
                precision_micro REAL,
                precision_macro REAL,
                recall_micro REAL,
                recall_macro REAL,
                roc_auc_macro REAL,
                roc_auc_micro REAL,
                n_train INTEGER,
                n_test INTEGER,
                timestamp TEXT
            )
            """
        )
        conn.commit()


def insert_fold_sqlite(db_path: Path, row: Dict[str, Any]) -> None:
    while True:
        try:
            with sqlite3.connect(db_path) as conn:
                c = conn.cursor()
                c.execute(
                    """
                    INSERT INTO multilabel_experiments (
                        config_index, config_name, features_csv, labels_csv, use_masks,
                        label_mode, multilabel_strategy, classifier, scaler, dimred,
                        target_cols_json, classifier_params_json, fold,
                        subset_accuracy, hamming_loss, f1_micro, f1_macro, f1_samples,
                        precision_micro, precision_macro, recall_micro, recall_macro,
                        roc_auc_macro, roc_auc_micro, n_train, n_test, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["config_index"],
                        row["config_name"],
                        row["features_csv"],
                        row["labels_csv"],
                        row["use_masks"],
                        row["label_mode"],
                        row["multilabel_strategy"],
                        row["classifier"],
                        row["scaler"],
                        row["dimred"],
                        json.dumps(row["target_cols"], ensure_ascii=False),
                        json.dumps(row["classifier_params"], ensure_ascii=False),
                        row["fold"],
                        row["subset_accuracy"],
                        row["hamming_loss"],
                        row["f1_micro"],
                        row["f1_macro"],
                        row["f1_samples"],
                        row["precision_micro"],
                        row["precision_macro"],
                        row["recall_micro"],
                        row["recall_macro"],
                        row["roc_auc_macro"],
                        row["roc_auc_micro"],
                        row["n_train"],
                        row["n_test"],
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()
                return
        except sqlite3.OperationalError:
            wait_seconds = random.randint(3, 10)
            time.sleep(wait_seconds)


def build_predictions_rows(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    target_cols: List[str],
    fold_idx: int,
) -> List[Dict[str, Any]]:
    rows = []
    for row_idx, patient_id in enumerate(patient_ids):
        row: Dict[str, Any] = {"fold": int(fold_idx), "patient_id": str(patient_id)}
        for label_idx, label_name in enumerate(target_cols):
            row[f"true__{label_name}"] = int(y_true[row_idx, label_idx])
            row[f"pred__{label_name}"] = int(y_pred[row_idx, label_idx])
            if y_score is not None and y_score.ndim == 2 and label_idx < y_score.shape[1]:
                row[f"score__{label_name}"] = float(y_score[row_idx, label_idx])
        rows.append(row)
    return rows


def aggregate_label_metrics(label_metrics_df: pd.DataFrame) -> pd.DataFrame:
    if label_metrics_df.empty:
        return pd.DataFrame()
    fold_scope_df = label_metrics_df[label_metrics_df["scope"] == "fold"].copy()
    if fold_scope_df.empty:
        return pd.DataFrame()
    numeric_cols = [
        col
        for col in fold_scope_df.columns
        if col not in {"scope", "fold", "label"} and pd.api.types.is_numeric_dtype(fold_scope_df[col])
    ]
    grouped = fold_scope_df.groupby("label", dropna=False)[numeric_cols].agg(["mean", "std", "min", "max"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def aggregate_fold_metrics(fold_metrics_df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = [
        col for col in fold_metrics_df.columns if col not in {"fold", "n_train", "n_test"}
    ]
    summary: Dict[str, Any] = {}
    for col in numeric_cols:
        if col in fold_metrics_df.columns:
            series = pd.to_numeric(fold_metrics_df[col], errors="coerce")
            summary[f"mean_{col}"] = None if series.dropna().empty else float(series.mean())
            summary[f"std_{col}"] = None if series.dropna().empty else float(series.std(ddof=0))
    return summary


def run_configuration(args: Any) -> None:
    config_file = Path(args.config_file).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_configs = load_configurations(config_file)
    config = all_configs[args.config_index]

    script_base = config_file.parent
    features_csv = resolve_path(args.features_csv or config.get("features_csv"), script_base)
    labels_csv = resolve_path(args.labels_csv or config.get("labels_csv") or str(DEFAULT_LABELS_CSV), script_base)
    mapping_csv = resolve_path(args.mapping_csv or config.get("mapping_csv"), script_base)
    id_col = config.get("id_col", "patient_id")
    target_cols = config.get("target_cols", list(DEFAULT_TARGET_COLS))
    label_mode = config.get("label_mode", "include_no_complication")
    use_masks = config.get("use_masks", "nodule")
    n_splits = int(config.get("n_splits", 5))
    random_state = int(args.random_state if args.random_state is not None else config.get("random_state", 42))
    shuffle_folds = bool(config.get("shuffle_folds", True))
    scaler_choice = config.get("scaler")
    classifier_name = config.get("classifier")
    dimred_choice = config.get("dimred", "none")
    multilabel_strategy = config.get("multilabel_strategy", "one_vs_rest")
    split_strategy = config.get("split_strategy", "auto")
    preproc_names = config.get("preproc_names")
    sampler_name = normalize_sampler_name(config.get("sampler_name"))
    sampler_kwargs = config.get("sampler_kwargs") or {}
    classifier_params = extract_classifier_params(config)

    if features_csv is None or labels_csv is None:
        raise ValueError("features_csv and labels_csv must be defined")

    output_dir = get_output_dir(output_root, args.config_index, config, args.job_suffix)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    run_context = {
        "config_index": int(args.config_index),
        "config_name": config.get("config_name", f"config_{args.config_index:06d}"),
        "config_file": str(config_file),
        "output_dir": str(output_dir),
        "features_csv": str(features_csv),
        "labels_csv": str(labels_csv),
        "mapping_csv": None if mapping_csv is None else str(mapping_csv),
        "id_col": id_col,
        "target_cols_requested": list(target_cols),
        "label_mode": label_mode,
        "use_masks": use_masks,
        "n_splits": n_splits,
        "random_state": random_state,
        "shuffle_folds": shuffle_folds,
        "scaler": scaler_choice,
        "classifier": classifier_name,
        "classifier_params": classifier_params,
        "dimred": dimred_choice,
        "multilabel_strategy": multilabel_strategy,
        "sampler_name": sampler_name,
        "sampler_kwargs": sampler_kwargs,
        "split_strategy": split_strategy,
        "preproc_names": preproc_names,
        "save_models": bool(args.save_models or config.get("save_models", False)),
        "save_predictions": bool(args.save_predictions or config.get("save_predictions", False)),
        "dry_run": bool(args.dry_run),
    }
    save_json(output_dir / "config_used.json", run_context)

    df_feats = load_features_csv(features_csv)
    df_feats = filter_preproc_names(df_feats, preproc_names)
    df_labels = load_labels_csv(labels_csv, id_col=id_col)
    mapping_df = load_mapping_csv(mapping_csv, id_col=id_col) if mapping_csv is not None else None

    merged_df, merge_report = merge_features_and_labels(
        df_feats=df_feats,
        df_labels=df_labels,
        id_col=id_col,
        target_cols=target_cols,
        mapping_df=mapping_df,
    )
    merged_df, mask_report = filter_mask_mode(merged_df, use_masks=use_masks, target_cols=target_cols)
    merged_df = ensure_binary_targets(merged_df, target_cols=target_cols)
    merged_df, final_target_cols, label_mode_info = apply_label_mode(
        merged_df,
        target_cols=target_cols,
        label_mode=label_mode,
    )

    merged_df = merged_df.dropna(subset=final_target_cols).copy()
    feature_cols = prepare_feature_columns(merged_df, final_target_cols)
    merged_df.loc[:, feature_cols] = merged_df[feature_cols].apply(pd.to_numeric, errors="coerce")

    coverage_report = {
        **merge_report,
        **mask_report,
        **label_mode_info,
        "n_rows_after_target_cleaning": int(len(merged_df)),
        "n_patients_after_target_cleaning": int(merged_df["patient_id"].nunique()),
        "feature_count": int(len(feature_cols)),
        "feature_columns_preview": feature_cols[:25],
        **summarize_label_support(merged_df, final_target_cols),
    }
    save_json(output_dir / "coverage_report.json", coverage_report)

    x = merged_df[feature_cols].values
    y = merged_df[final_target_cols].values.astype(int)
    patient_ids = merged_df["patient_id"].astype(str).values

    if args.dry_run:
        dry_run_summary = {
            **run_context,
            **coverage_report,
            "status": "dry_run_ok",
            "message": "Configuration and data validated. No training executed.",
        }
        save_json(output_dir / "summary.json", dry_run_summary)
        print(json.dumps(dry_run_summary, indent=2, ensure_ascii=False))
        return

    splitter_or_tuple = build_splitter(
        y=y,
        n_splits=n_splits,
        shuffle_folds=shuffle_folds,
        random_state=random_state,
        split_strategy=split_strategy,
    )
    if isinstance(splitter_or_tuple, tuple):
        splitter, strata = splitter_or_tuple
        split_iterator = splitter.split(x, strata)
        split_metadata = {
            "resolved_splitter": "StratifiedKFold",
            "stratification_basis": "labelset_signature",
        }
    else:
        splitter = splitter_or_tuple
        split_iterator = splitter.split(x)
        split_metadata = {
            "resolved_splitter": splitter.__class__.__name__,
            "stratification_basis": None,
        }

    pipeline = build_pipeline(
        classifier_name=classifier_name,
        classifier_params=classifier_params,
        scaler_choice=scaler_choice,
        dimred_choice=dimred_choice,
        multilabel_strategy=multilabel_strategy,
        random_state=random_state,
        sampler_name=sampler_name,
        sampler_kwargs=sampler_kwargs,
    )

    sqlite_db_path = output_dir / "results_multilabel_210.db"
    if not args.no_sqlite:
        init_sqlite(sqlite_db_path)

    fold_metric_rows: List[Dict[str, Any]] = []
    label_metric_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    oof_true_rows: List[np.ndarray] = []
    oof_pred_rows: List[np.ndarray] = []
    oof_score_rows: List[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(split_iterator, start=1):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        fold_patient_ids = patient_ids[test_idx]

        pipeline_fold = clone(pipeline)
        pipeline_fold.fit(x_train, y_train)
        y_pred = pipeline_fold.predict(x_test)
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        y_score = get_score_matrix(pipeline_fold, x_test)
        if y_score is not None:
            y_score = np.asarray(y_score)
            if y_score.ndim == 1:
                y_score = y_score.reshape(-1, 1)

        fold_metrics = compute_fold_metrics(y_true=y_test, y_pred=y_pred, y_score=y_score)
        fold_row = {
            "fold": int(fold_idx),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            **fold_metrics,
        }
        fold_metric_rows.append(fold_row)
        label_metric_rows.extend(
            compute_label_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_score=y_score,
                target_cols=final_target_cols,
                fold_idx=fold_idx,
                scope="fold",
            )
        )
        oof_true_rows.append(y_test)
        oof_pred_rows.append(y_pred)
        if y_score is not None:
            oof_score_rows.append(y_score)

        for idx in train_idx:
            split_rows.append({"fold": int(fold_idx), "split": "train", "patient_id": str(patient_ids[idx])})
        for idx in test_idx:
            split_rows.append({"fold": int(fold_idx), "split": "test", "patient_id": str(patient_ids[idx])})

        if run_context["save_predictions"]:
            prediction_rows.extend(
                build_predictions_rows(
                    patient_ids=fold_patient_ids,
                    y_true=y_test,
                    y_pred=y_pred,
                    y_score=y_score,
                    target_cols=final_target_cols,
                    fold_idx=fold_idx,
                )
            )

        if run_context["save_models"]:
            model_path = models_dir / f"model_cfg{args.config_index:06d}_fold{fold_idx}.pkl"
            joblib.dump(pipeline_fold, model_path)

        if not args.no_sqlite:
            sqlite_row = {
                "config_index": int(args.config_index),
                "config_name": run_context["config_name"],
                "features_csv": str(features_csv),
                "labels_csv": str(labels_csv),
                "use_masks": use_masks,
                "label_mode": label_mode,
                "multilabel_strategy": multilabel_strategy,
                "classifier": classifier_name,
                "scaler": scaler_choice,
                "dimred": dimred_choice,
                "target_cols": final_target_cols,
                "classifier_params": classifier_params,
                **fold_row,
            }
            insert_fold_sqlite(sqlite_db_path, sqlite_row)

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    label_metrics_df = pd.DataFrame(label_metric_rows)
    label_metrics_agg_df = aggregate_label_metrics(label_metrics_df)
    split_assignments_df = pd.DataFrame(split_rows)

    oof_y_true = np.vstack(oof_true_rows) if oof_true_rows else np.empty((0, len(final_target_cols)), dtype=int)
    oof_y_pred = np.vstack(oof_pred_rows) if oof_pred_rows else np.empty((0, len(final_target_cols)), dtype=int)
    oof_y_score = None
    if oof_score_rows and len(oof_score_rows) == len(oof_true_rows):
        oof_y_score = np.vstack(oof_score_rows)

    oof_metrics = compute_fold_metrics(oof_y_true, oof_y_pred, oof_y_score) if len(oof_y_true) else {}
    oof_label_metrics_df = (
        pd.DataFrame(
            compute_label_metrics(
                y_true=oof_y_true,
                y_pred=oof_y_pred,
                y_score=oof_y_score,
                target_cols=final_target_cols,
                fold_idx=0,
                scope="oof_all",
            )
        )
        if len(oof_y_true)
        else pd.DataFrame()
    )

    save_dataframe(output_dir / "fold_metrics.csv", fold_metrics_df)
    save_dataframe(output_dir / "label_metrics.csv", label_metrics_df)
    if not label_metrics_agg_df.empty:
        save_dataframe(output_dir / "label_metrics_agg.csv", label_metrics_agg_df)
    if not oof_label_metrics_df.empty:
        save_dataframe(output_dir / "oof_label_metrics.csv", oof_label_metrics_df)
    save_dataframe(output_dir / "split_assignments.csv", split_assignments_df)

    if run_context["save_predictions"] and prediction_rows:
        predictions_df = pd.DataFrame(prediction_rows)
        save_dataframe(output_dir / "predictions.csv", predictions_df)

    summary = {
        **run_context,
        **coverage_report,
        **split_metadata,
        **aggregate_fold_metrics(fold_metrics_df),
        **{f"oof_{key}": value for key, value in oof_metrics.items()},
        "n_folds_executed": int(len(fold_metrics_df)),
        "status": "completed",
    }
    if oof_metrics:
        save_json(output_dir / "oof_metrics.json", oof_metrics)
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    try:
        run_configuration(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
