#!/usr/bin/env python3
"""Reproducible subgroup discovery for the lung biopsy TFM cohort."""

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed
from scipy.stats import fisher_exact
from sklearn.model_selection import StratifiedKFold


ID_COL = "patient_id"
DEFAULT_TARGETS = ["Neumotórax", "Hemorragia", "Sin_complicación"]
ALL_CLINICAL_TARGETS = [
    "Derrame_pleural",
    "Hemorragia",
    "Neumotórax",
    "Sin_complicación",
    "Complicacion_binaria",
]
GEOMETRY_FEATURES = [
    "tamano_nodulo_mm",
    "profundidad_min_pleura_mm",
    "profundidad_centroidal_pleura_mm",
]
DROP_DESCRIPTORS = ["Sin_factor_de_riesgo", "Sin_patología_pulmonar"]
GROUPS = {
    "Fibrosis_any": ["Fibrosis", "Fibrosis_pulmonar"],
    "Dislipemia_any": ["Dislipemia", "Hipercolesterolemia", "Hiperlipemia"],
    "DM_any": ["DM", "DM1", "DM2"],
    "Tabac_any": ["Tabaquismo", "Extabaquismo", "Tabaquismo_pasivo"],
    "Alcohol_any": ["Alcoholismo", "Exalcoholismo"],
    "Enfisema_any": [
        "Enfisema",
        "Enfisema_centrolobulillar",
        "Enfisema_panacinar",
        "Enfisema_paraseptal",
    ],
    "CardioRisk": ["HTA", "Cardiopatía", "SCACEST", "Hiperuricemia"],
    "Urologic": ["HBP"],
}


@dataclass(frozen=True)
class Selector:
    feature: str
    operator: str
    value: float

    @property
    def description(self) -> str:
        if self.operator == "==":
            value = int(self.value) if float(self.value).is_integer() else self.value
        else:
            value = round(float(self.value), 6)
        return "{} {} {}".format(self.feature, self.operator, value)

    def mask(self, x: pd.DataFrame) -> np.ndarray:
        values = pd.to_numeric(x[self.feature], errors="coerce")
        if self.operator == "==":
            out = values == self.value
        elif self.operator == "<=":
            out = values <= self.value
        elif self.operator == ">":
            out = values > self.value
        else:
            raise ValueError("Unsupported operator: {}".format(self.operator))
        return out.fillna(False).to_numpy(dtype=bool)


@dataclass
class Rule:
    selectors: Tuple[Selector, ...]
    train_metrics: Dict[str, float]

    @property
    def description(self) -> str:
        return " AND ".join(selector.description for selector in self.selectors)

    @property
    def rule_key(self) -> str:
        return "|".join(selector.description for selector in self.selectors)

    @property
    def structure_key(self) -> str:
        return "|".join(
            "{} {}".format(selector.feature, selector.operator)
            for selector in self.selectors
        )

    def mask(self, x: pd.DataFrame) -> np.ndarray:
        out = np.ones(len(x), dtype=bool)
        for selector in self.selectors:
            out &= selector.mask(x)
        return out


@dataclass(frozen=True)
class Split:
    repeat: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item) for item in parse_csv_list(value)]


def parse_int_list(value: str) -> List[int]:
    return [int(item) for item in parse_csv_list(value)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cohort(df: pd.DataFrame, expected_n: int = 210) -> None:
    if ID_COL not in df.columns:
        raise ValueError("Missing required column: {}".format(ID_COL))
    if len(df) != expected_n:
        raise ValueError("Expected {} patients, found {}".format(expected_n, len(df)))
    if df[ID_COL].isna().any():
        raise ValueError("patient_id contains missing values")
    if df[ID_COL].nunique() != expected_n:
        raise ValueError("patient_id values must be unicos")


def _to_binary(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).clip(0, 1)


def _rowwise_any(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    present = [column for column in columns if column in df.columns]
    if not present:
        return pd.Series(0, index=df.index, dtype=int)
    return df[present].apply(_to_binary).max(axis=1).astype(int)


def apply_clinical_grouping(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sources = []
    for output, columns in GROUPS.items():
        out[output] = _rowwise_any(out, columns)
        sources.extend(column for column in columns if column in out.columns)
    drop_columns = sorted(set(sources + [c for c in DROP_DESCRIPTORS if c in out.columns]))
    return out.drop(columns=drop_columns, errors="ignore")


def filter_predictors(
    df: pd.DataFrame,
    targets: Sequence[str],
    min_feature_count: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    excluded = set([ID_COL] + ALL_CLINICAL_TARGETS + list(targets))
    candidates = [column for column in df.columns if column not in excluded]
    output = pd.DataFrame(index=df.index)
    audit_rows = []
    for column in candidates:
        values = pd.to_numeric(df[column], errors="coerce")
        keep = True
        reason = "kept"
        unique = sorted(values.dropna().unique().tolist())
        is_binary = set(unique).issubset({0, 1})
        if not unique:
            keep = False
            reason = "all_missing"
        elif is_binary:
            positive = int((values == 1).sum())
            negative = int((values == 0).sum())
            if min(positive, negative) < min_feature_count:
                keep = False
                reason = "rare_binary"
        elif values.nunique(dropna=True) < 2:
            keep = False
            reason = "constant"
        if keep:
            output[column] = values
        audit_rows.append(
            {
                "feature": column,
                "kept": keep,
                "reason": reason,
                "n_missing": int(values.isna().sum()),
                "n_unique": int(values.nunique(dropna=True)),
                "n_positive": int((values == 1).sum()) if is_binary else np.nan,
            }
        )
    return output, pd.DataFrame(audit_rows)


def compute_metrics(mask: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    y = np.asarray(y, dtype=int)
    if len(mask) != len(y):
        raise ValueError("mask and target must have the same length")
    n = len(y)
    n_covered = int(mask.sum())
    total_positive = int(y.sum())
    n_positive_covered = int(y[mask].sum()) if n_covered else 0
    global_prevalence = total_positive / n if n else np.nan
    coverage = n_covered / n if n else np.nan
    subgroup_prevalence = n_positive_covered / n_covered if n_covered else np.nan
    support = n_positive_covered / n if n else np.nan
    sensitivity = n_positive_covered / total_positive if total_positive else np.nan
    if n_covered and global_prevalence > 0:
        lift = subgroup_prevalence / global_prevalence
    else:
        lift = np.nan
    if n_covered:
        risk_difference = subgroup_prevalence - global_prevalence
        wracc = coverage * risk_difference
    else:
        risk_difference = np.nan
        wracc = 0.0
    a = n_positive_covered
    b = n_covered - n_positive_covered
    c = total_positive - n_positive_covered
    d = (n - n_covered) - c
    if n_covered and n_covered < n:
        odds_ratio, fisher_p = fisher_exact([[a, b], [c, d]], alternative="greater")
    else:
        odds_ratio, fisher_p = np.nan, np.nan
    return {
        "n": n,
        "n_covered": n_covered,
        "n_positive": total_positive,
        "n_positive_covered": n_positive_covered,
        "coverage": coverage,
        "support": support,
        "global_prevalence": global_prevalence,
        "subgroup_prevalence": subgroup_prevalence,
        "risk_difference": risk_difference,
        "lift": lift,
        "sensitivity": sensitivity,
        "wracc": wracc,
        "odds_ratio": odds_ratio,
        "fisher_p_unadjusted": fisher_p,
    }


def make_selectors(
    x: pd.DataFrame,
    numeric_quantiles: Sequence[float] = (0.25, 0.5, 0.75),
    include_binary_absence: bool = False,
) -> List[Selector]:
    selectors = []
    for feature in sorted(x.columns):
        values = pd.to_numeric(x[feature], errors="coerce")
        unique = sorted(values.dropna().unique().tolist())
        if not unique:
            continue
        if set(unique).issubset({0, 1}):
            binary_values = [0, 1] if include_binary_absence or feature == "Sexo_binaria" else [1]
            for value in binary_values:
                if value in unique:
                    selectors.append(Selector(feature, "==", float(value)))
        else:
            thresholds = values.quantile(list(numeric_quantiles)).dropna().unique().tolist()
            for threshold in sorted(set(float(v) for v in thresholds)):
                selectors.append(Selector(feature, "<=", threshold))
                selectors.append(Selector(feature, ">", threshold))
    return selectors


def _rank_rule(rule: Rule) -> Tuple[float, float, int, str]:
    metrics = rule.train_metrics
    return (
        float(metrics["wracc"]),
        float(metrics["coverage"]),
        -len(rule.selectors),
        rule.rule_key,
    )


def beam_search(
    x: pd.DataFrame,
    y: np.ndarray,
    max_depth: int = 3,
    min_coverage: float = 0.10,
    beam_width: int = 100,
    top_candidates: int = 100,
    include_binary_absence: bool = False,
) -> List[Rule]:
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    selectors = make_selectors(x, include_binary_absence=include_binary_absence)
    min_count = max(1, int(math.ceil(min_coverage * len(x))))
    selector_order = {selector: i for i, selector in enumerate(selectors)}
    all_rules = []
    beam = []
    for depth in range(1, max_depth + 1):
        candidates = []
        if depth == 1:
            selector_tuples = [(selector,) for selector in selectors]
        else:
            selector_tuples = []
            for previous in beam:
                used_features = {selector.feature for selector in previous.selectors}
                last_index = selector_order[previous.selectors[-1]]
                for selector in selectors[last_index + 1 :]:
                    if selector.feature not in used_features:
                        selector_tuples.append(previous.selectors + (selector,))
        seen_masks = set()
        for selector_tuple in selector_tuples:
            rule = Rule(selector_tuple, {})
            mask = rule.mask(x)
            covered = int(mask.sum())
            if covered < min_count or covered == len(x):
                continue
            packed = np.packbits(mask).tobytes()
            if packed in seen_masks:
                continue
            seen_masks.add(packed)
            metrics = compute_metrics(mask, y)
            if metrics["wracc"] <= 0:
                continue
            candidates.append(Rule(selector_tuple, metrics))
        candidates.sort(key=_rank_rule, reverse=True)
        beam = candidates[:beam_width]
        all_rules.extend(candidates[:top_candidates])
        if not beam:
            break
    unique = {}
    for rule in all_rules:
        old = unique.get(rule.rule_key)
        if old is None or _rank_rule(rule) > _rank_rule(old):
            unique[rule.rule_key] = rule
    return sorted(unique.values(), key=_rank_rule, reverse=True)[:top_candidates]


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(mask_a, mask_b).sum() / union)


def select_diverse_rules(
    rules: Sequence[Rule],
    x: pd.DataFrame,
    max_rules: int = 5,
    max_jaccard: float = 0.80,
) -> List[Rule]:
    selected = []
    selected_masks = []
    for rule in sorted(rules, key=_rank_rule, reverse=True):
        mask = rule.mask(x)
        if any(jaccard(mask, previous) > max_jaccard for previous in selected_masks):
            continue
        selected.append(rule)
        selected_masks.append(mask)
        if len(selected) >= max_rules:
            break
    return selected


def build_multilabel_strata(labels: pd.DataFrame, n_splits: int) -> np.ndarray:
    if labels.empty:
        raise ValueError("At least one stratification label is required")
    encoded = labels.astype(int).astype(str).agg("|".join, axis=1)
    counts = encoded.value_counts()
    rare = set(counts[counts < n_splits].index)
    if rare:
        encoded = encoded.map(lambda value: "rare" if value in rare else value)
    if encoded.value_counts().min() < n_splits:
        fallback = labels.sum(axis=1).clip(upper=2).astype(str)
        fallback_counts = fallback.value_counts()
        fallback_rare = set(fallback_counts[fallback_counts < n_splits].index)
        encoded = fallback.map(lambda value: "rare" if value in fallback_rare else value)
    if encoded.value_counts().min() < n_splits:
        raise ValueError("Not enough samples per multilabel stratum")
    return encoded.to_numpy()


def make_repeated_splits(
    labels: pd.DataFrame,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> List[Split]:
    strata = build_multilabel_strata(labels, n_splits)
    dummy = np.zeros(len(labels))
    output = []
    for repeat in range(n_repeats):
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + repeat,
        )
        for fold, (train_idx, test_idx) in enumerate(splitter.split(dummy, strata)):
            output.append(Split(repeat, fold, train_idx, test_idx))
    return output


def rule_to_row(
    rule: Rule,
    target: str,
    feature_set: str,
    split_name: str,
    metrics: Dict[str, float],
    repeat: Optional[int] = None,
    fold: Optional[int] = None,
) -> Dict[str, object]:
    row = {
        "target": target,
        "feature_set": feature_set,
        "split": split_name,
        "repeat": repeat,
        "fold": fold,
        "rule": rule.description,
        "rule_key": rule.rule_key,
        "structure_key": rule.structure_key,
        "rule_length": len(rule.selectors),
    }
    row.update(metrics)
    return row


def discover_full_rules(
    x: pd.DataFrame,
    targets: pd.DataFrame,
    feature_set: str,
    config: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, List[Rule]]]:
    rows = []
    selected_by_target = {}
    for target in targets.columns:
        y = targets[target].to_numpy(dtype=int)
        candidates = beam_search(
            x,
            y,
            max_depth=config.max_depth,
            min_coverage=config.min_coverage,
            beam_width=config.beam_width,
            top_candidates=config.top_candidates,
            include_binary_absence=config.include_binary_absence,
        )
        selected = select_diverse_rules(
            candidates,
            x,
            max_rules=config.top_rules,
            max_jaccard=config.max_jaccard,
        )
        selected_by_target[target] = selected
        for rule in selected:
            rows.append(
                rule_to_row(rule, target, feature_set, "full", rule.train_metrics)
            )
    return pd.DataFrame(rows), selected_by_target


def run_cross_validation(
    x: pd.DataFrame,
    targets: pd.DataFrame,
    stratification_labels: pd.DataFrame,
    feature_set: str,
    config: argparse.Namespace,
) -> pd.DataFrame:
    splits = make_repeated_splits(
        stratification_labels,
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        seed=config.seed,
    )
    rows = []
    for split in splits:
        x_train = x.iloc[split.train_idx].reset_index(drop=True)
        x_test = x.iloc[split.test_idx].reset_index(drop=True)
        for target in targets.columns:
            y_train = targets.iloc[split.train_idx][target].to_numpy(dtype=int)
            y_test = targets.iloc[split.test_idx][target].to_numpy(dtype=int)
            candidates = beam_search(
                x_train,
                y_train,
                max_depth=config.max_depth,
                min_coverage=config.min_coverage,
                beam_width=config.beam_width,
                top_candidates=config.top_candidates,
                include_binary_absence=config.include_binary_absence,
            )
            selected = select_diverse_rules(
                candidates,
                x_train,
                max_rules=config.top_rules,
                max_jaccard=config.max_jaccard,
            )
            for rule in selected:
                train_metrics = compute_metrics(rule.mask(x_train), y_train)
                test_metrics = compute_metrics(rule.mask(x_test), y_test)
                rows.append(
                    rule_to_row(
                        rule,
                        target,
                        feature_set,
                        "train",
                        train_metrics,
                        split.repeat,
                        split.fold,
                    )
                )
                rows.append(
                    rule_to_row(
                        rule,
                        target,
                        feature_set,
                        "test",
                        test_metrics,
                        split.repeat,
                        split.fold,
                    )
                )
    return pd.DataFrame(rows)


def summarize_stability(cv_rows: pd.DataFrame, total_splits: int) -> pd.DataFrame:
    if cv_rows.empty:
        return pd.DataFrame()
    test = cv_rows[cv_rows["split"] == "test"].copy()
    per_fold = (
        test.groupby(["feature_set", "target", "repeat", "fold", "structure_key"], as_index=False)
        .agg(
            rule_example=("rule", "first"),
            test_wracc=("wracc", "max"),
            test_coverage=("coverage", "mean"),
            test_prevalence=("subgroup_prevalence", "mean"),
            test_risk_difference=("risk_difference", "mean"),
            test_lift=("lift", "mean"),
        )
    )
    summary = (
        per_fold.groupby(["feature_set", "target", "structure_key"], as_index=False)
        .agg(
            rule_example=("rule_example", "first"),
            folds_selected=("fold", "size"),
            mean_test_wracc=("test_wracc", "mean"),
            median_test_wracc=("test_wracc", "median"),
            mean_test_coverage=("test_coverage", "mean"),
            mean_test_prevalence=("test_prevalence", "mean"),
            mean_test_risk_difference=("test_risk_difference", "mean"),
            mean_test_lift=("test_lift", "mean"),
        )
    )
    summary["selection_stability"] = summary["folds_selected"] / total_splits
    return summary.sort_values(
        ["feature_set", "target", "selection_stability", "mean_test_wracc"],
        ascending=[True, True, False, False],
    )


def _permutation_max_score(
    seed: int,
    x: pd.DataFrame,
    y: np.ndarray,
    config: argparse.Namespace,
) -> float:
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(y)
    rules = beam_search(
        x,
        permuted,
        max_depth=config.max_depth,
        min_coverage=config.min_coverage,
        beam_width=config.beam_width,
        top_candidates=1,
        include_binary_absence=config.include_binary_absence,
    )
    return float(rules[0].train_metrics["wracc"]) if rules else 0.0


def run_permutations(
    x: pd.DataFrame,
    targets: pd.DataFrame,
    feature_set: str,
    selected_by_target: Dict[str, List[Rule]],
    config: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for target in targets.columns:
        y = targets[target].to_numpy(dtype=int)
        seeds = [config.seed + 100000 + i for i in range(config.n_permutations)]
        null_scores = Parallel(n_jobs=config.n_jobs)(
            delayed(_permutation_max_score)(seed, x, y, config) for seed in seeds
        )
        null_scores = np.asarray(null_scores, dtype=float)
        for rule in selected_by_target.get(target, []):
            observed = float(rule.train_metrics["wracc"])
            p_adjusted = (1 + int((null_scores >= observed).sum())) / (
                config.n_permutations + 1
            )
            rows.append(
                {
                    "feature_set": feature_set,
                    "target": target,
                    "rule": rule.description,
                    "rule_key": rule.rule_key,
                    "observed_wracc": observed,
                    "permutation_p_search_adjusted": p_adjusted,
                    "n_permutations": config.n_permutations,
                    "null_max_wracc_mean": float(null_scores.mean()),
                    "null_max_wracc_q95": float(np.quantile(null_scores, 0.95)),
                }
            )
    return pd.DataFrame(rows)


def run_sensitivity(
    x: pd.DataFrame,
    targets: pd.DataFrame,
    feature_set: str,
    config: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for min_coverage in config.sensitivity_coverages:
        for max_depth in config.sensitivity_depths:
            for target in targets.columns:
                y = targets[target].to_numpy(dtype=int)
                candidates = beam_search(
                    x,
                    y,
                    max_depth=max_depth,
                    min_coverage=min_coverage,
                    beam_width=config.beam_width,
                    top_candidates=config.top_candidates,
                    include_binary_absence=config.include_binary_absence,
                )
                selected = select_diverse_rules(
                    candidates,
                    x,
                    max_rules=config.top_rules,
                    max_jaccard=config.max_jaccard,
                )
                for rank, rule in enumerate(selected, start=1):
                    row = rule_to_row(
                        rule, target, feature_set, "full_sensitivity", rule.train_metrics
                    )
                    row.update(
                        {
                            "rank": rank,
                            "min_coverage_config": min_coverage,
                            "max_depth_config": max_depth,
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def merge_extra_targets(
    clinical: pd.DataFrame,
    extra: pd.DataFrame,
    extra_targets: Sequence[str],
) -> pd.DataFrame:
    if ID_COL not in extra.columns:
        raise ValueError("Extra target file requires patient_id")
    missing_columns = [column for column in extra_targets if column not in extra.columns]
    if missing_columns:
        raise ValueError("Missing extra target columns: {}".format(missing_columns))
    if extra[ID_COL].duplicated().any():
        raise ValueError("Extra target patient_id values must be unicos")
    merged = clinical.merge(
        extra[[ID_COL] + list(extra_targets)],
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )
    if merged[list(extra_targets)].isna().any().any():
        raise ValueError("Missing values after merging extra targets")
    return merged


def load_inputs(config: argparse.Namespace):
    clinical_path = Path(config.clinical_csv)
    targets_path = Path(config.targets_csv)
    geometry_path = Path(config.geometry_csv)
    clinical = pd.read_csv(clinical_path)
    target_data = pd.read_csv(targets_path)
    geometry = pd.read_csv(geometry_path)
    validate_cohort(clinical, config.expected_patients)
    validate_cohort(target_data, config.expected_patients)
    validate_cohort(geometry, config.expected_patients)
    clinical[ID_COL] = clinical[ID_COL].astype(str)
    target_data[ID_COL] = target_data[ID_COL].astype(str)
    geometry[ID_COL] = geometry[ID_COL].astype(str)
    clinical_ids = set(clinical[ID_COL])
    if clinical_ids != set(target_data[ID_COL]):
        raise ValueError("Clinical and target patient sets differ")
    if clinical_ids != set(geometry[ID_COL]):
        raise ValueError("Clinical and geometry patient sets differ")
    if config.extra_target_csv:
        extra = pd.read_csv(config.extra_target_csv)
        extra[ID_COL] = extra[ID_COL].astype(str)
        target_data = merge_extra_targets(target_data, extra, config.extra_targets)
    for target in config.targets:
        if target not in target_data.columns:
            raise ValueError("Missing target: {}".format(target))
        values = set(pd.to_numeric(target_data[target], errors="coerce").dropna().unique())
        if not values.issubset({0, 1}):
            raise ValueError("Target is not binary: {}".format(target))
        if pd.to_numeric(target_data[target], errors="coerce").isna().any():
            raise ValueError("Target contains missing values: {}".format(target))
    predictors, feature_audit = filter_predictors(
        clinical,
        config.targets,
        config.min_feature_count,
    )
    targets = (
        target_data.set_index(ID_COL)
        .reindex(clinical[ID_COL])[config.targets]
        .apply(pd.to_numeric)
        .astype(int)
    )
    predictors.index = clinical[ID_COL]
    targets.index = predictors.index
    geometry = geometry.set_index(ID_COL).reindex(predictors.index)
    geometry_numeric = geometry[GEOMETRY_FEATURES].apply(pd.to_numeric, errors="coerce")
    complete_geometry = geometry_numeric.notna().all(axis=1)
    clinical_full = predictors.copy()
    clinical_matched = predictors.loc[complete_geometry].copy()
    clinical_geometry = pd.concat(
        [clinical_matched, geometry_numeric.loc[complete_geometry]], axis=1
    )
    feature_sets = {
        "clinical_full": (clinical_full, targets.loc[clinical_full.index]),
        "clinical_matched": (
            clinical_matched,
            targets.loc[clinical_matched.index],
        ),
        "clinical_geometry": (
            clinical_geometry,
            targets.loc[clinical_geometry.index],
        ),
    }
    requested = set(config.feature_sets)
    unknown = requested - set(feature_sets)
    if unknown:
        raise ValueError("Unknown feature sets: {}".format(sorted(unknown)))
    feature_sets = {name: value for name, value in feature_sets.items() if name in requested}
    metadata = {
        "clinical_csv": str(clinical_path),
        "clinical_sha256": sha256_file(clinical_path),
        "targets_csv": str(targets_path),
        "targets_sha256": sha256_file(targets_path),
        "geometry_csv": str(geometry_path),
        "geometry_sha256": sha256_file(geometry_path),
        "n_clinical_full": int(len(clinical_full)),
        "n_geometry_complete": int(complete_geometry.sum()),
        "n_geometry_incomplete": int((~complete_geometry).sum()),
    }
    return feature_sets, feature_audit, metadata


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    base = "/mnt/homeGPU/mcribilles/tfm/clinical_data/210pacientes"
    parser.add_argument(
        "--clinical-csv",
        default=base + "/clinical_data_multimodal.csv",
    )
    parser.add_argument(
        "--targets-csv",
        default=base + "/clinical_data_limpios_SD.csv",
    )
    parser.add_argument(
        "--geometry-csv",
        default=base + "/nodule_geometry_210.csv",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--expected-patients", type=int, default=210)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--include-binary-target", action="store_true")
    parser.add_argument("--extra-target-csv", default=None)
    parser.add_argument("--extra-targets", default="")
    parser.add_argument(
        "--feature-sets",
        default="clinical_full,clinical_matched,clinical_geometry",
    )
    parser.add_argument("--min-feature-count", type=int, default=5)
    parser.add_argument("--min-coverage", type=float, default=0.10)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=100)
    parser.add_argument("--top-candidates", type=int, default=100)
    parser.add_argument("--top-rules", type=int, default=5)
    parser.add_argument("--max-jaccard", type=float, default=0.80)
    parser.add_argument("--include-binary-absence", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--skip-permutations", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--sensitivity-coverages", default="0.075,0.10,0.15")
    parser.add_argument("--sensitivity-depths", default="1,2,3")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke", action="store_true")
    return parser


def normalize_config(config: argparse.Namespace) -> argparse.Namespace:
    config.targets = parse_csv_list(config.targets)
    if config.include_binary_target and "Complicacion_binaria" not in config.targets:
        config.targets.append("Complicacion_binaria")
    config.extra_targets = parse_csv_list(config.extra_targets)
    for target in config.extra_targets:
        if target not in config.targets:
            config.targets.append(target)
    if bool(config.extra_target_csv) != bool(config.extra_targets):
        raise ValueError("extra-target-csv and extra-targets must be used together")
    config.feature_sets = parse_csv_list(config.feature_sets)
    config.sensitivity_coverages = parse_float_list(config.sensitivity_coverages)
    config.sensitivity_depths = parse_int_list(config.sensitivity_depths)
    if config.smoke:
        config.feature_sets = ["clinical_full"]
        config.n_splits = 3
        config.n_repeats = 1
        config.n_permutations = min(config.n_permutations, 5)
        config.beam_width = min(config.beam_width, 30)
        config.top_candidates = min(config.top_candidates, 30)
        config.top_rules = min(config.top_rules, 3)
        config.sensitivity_coverages = [config.min_coverage]
        config.sensitivity_depths = [config.max_depth]
    if config.n_splits < 2 or config.n_repeats < 1:
        raise ValueError("Invalid cross-validation configuration")
    return config


def namespace_to_dict(config: argparse.Namespace) -> Dict[str, object]:
    return {key: value for key, value in vars(config).items()}


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)
    logging.info("Saved %s rows to %s", len(frame), path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = normalize_config(build_parser().parse_args(argv))
    if config.output_dir:
        output_dir = Path(config.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "smoke" if config.smoke else "full"
        output_dir = Path(__file__).resolve().parent / "results" / (mode + "_" + stamp)
    configure_logging(output_dir)
    start = time.time()
    logging.info("Starting subgroup discovery")
    logging.info("Device: CPU")
    logging.info("Targets: %s", config.targets)
    logging.info("Feature sets: %s", config.feature_sets)
    feature_sets, feature_audit, input_metadata = load_inputs(config)
    save_frame(feature_audit, output_dir / "feature_audit.csv")
    all_full = []
    all_cv = []
    all_permutations = []
    all_sensitivity = []
    for feature_set, (x, targets) in feature_sets.items():
        logging.info(
            "Feature set %s: %s patients, %s predictors",
            feature_set,
            len(x),
            x.shape[1],
        )
        stratification_columns = [
            target for target in ["Neumotórax", "Hemorragia"] if target in targets.columns
        ]
        if not stratification_columns:
            stratification_columns = list(targets.columns)
        strata = targets[stratification_columns]
        full_rows, selected = discover_full_rules(x, targets, feature_set, config)
        all_full.append(full_rows)
        cv_rows = run_cross_validation(x, targets, strata, feature_set, config)
        all_cv.append(cv_rows)
        if not config.skip_permutations and config.n_permutations > 0:
            permutation_rows = run_permutations(
                x, targets, feature_set, selected, config
            )
            all_permutations.append(permutation_rows)
        if not config.skip_sensitivity:
            sensitivity_rows = run_sensitivity(x, targets, feature_set, config)
            all_sensitivity.append(sensitivity_rows)
    full = pd.concat(all_full, ignore_index=True) if all_full else pd.DataFrame()
    cv = pd.concat(all_cv, ignore_index=True) if all_cv else pd.DataFrame()
    permutations = (
        pd.concat(all_permutations, ignore_index=True)
        if all_permutations
        else pd.DataFrame()
    )
    sensitivity = (
        pd.concat(all_sensitivity, ignore_index=True)
        if all_sensitivity
        else pd.DataFrame()
    )
    stability = summarize_stability(cv, config.n_splits * config.n_repeats)
    save_frame(full, output_dir / "rules_full.csv")
    save_frame(cv, output_dir / "cv_rules.csv")
    save_frame(stability, output_dir / "stability_summary.csv")
    save_frame(permutations, output_dir / "permutation_summary.csv")
    save_frame(sensitivity, output_dir / "sensitivity_rules.csv")
    metadata = {
        "created_at": datetime.now().isoformat(),
        "elapsed_seconds": time.time() - start,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "config": namespace_to_dict(config),
        "inputs": input_metadata,
        "privacy": "Outputs contain aggregate rules and no patient identifiers.",
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    logging.info("Completed in %.2f seconds", metadata["elapsed_seconds"])
    logging.info("Output directory: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
