#!/usr/bin/env python3

import math

import numpy as np


def select_hidden_positive_ids(frame, target, fraction, seed):
    if not 0.0 <= float(fraction) < 1.0:
        raise ValueError("fraction must be in [0, 1)")
    positives = sorted(frame.loc[frame[target].astype(int) == 1, "patient_id"].astype(str))
    if not positives or float(fraction) == 0.0:
        return []
    n_hidden = max(1, int(math.floor(len(positives) * float(fraction) + 0.5)))
    n_hidden = min(n_hidden, len(positives) - 1)
    rng = np.random.RandomState(int(seed))
    chosen = rng.choice(np.asarray(positives, dtype=object), size=n_hidden, replace=False)
    return sorted(map(str, chosen.tolist()))


def apply_hidden_labels(frame, target, hidden_ids):
    altered = frame.copy()
    altered["patient_id"] = altered["patient_id"].astype(str)
    altered["y_true"] = altered[target].astype(int)
    hidden = set(map(str, hidden_ids))
    invalid = hidden - set(altered.loc[altered["y_true"] == 1, "patient_id"])
    if invalid:
        raise ValueError("hidden_ids must identify positive rows")
    altered["is_hidden_positive"] = altered["patient_id"].isin(hidden).astype(int)
    altered["y_observed"] = altered["y_true"]
    altered.loc[altered["is_hidden_positive"] == 1, "y_observed"] = 0
    return altered


def build_task_rows(targets, fractions, concealment_seeds):
    rows = []
    for target in targets:
        for fraction in fractions:
            seeds = [0] if float(fraction) == 0.0 else list(concealment_seeds)
            for seed in seeds:
                rows.append(
                    {
                        "task_id": len(rows),
                        "target": str(target),
                        "hide_fraction": float(fraction),
                        "concealment_seed": int(seed),
                    }
                )
    return rows


def validate_hidden_ids(train_frame, target, hidden_ids, test_ids):
    hidden = set(map(str, hidden_ids))
    train_ids = set(train_frame["patient_id"].astype(str))
    test_ids = set(map(str, test_ids))
    if hidden & test_ids:
        raise ValueError("A hidden-positive mask contains outer-test patients")
    if not hidden.issubset(train_ids):
        raise ValueError("A hidden-positive mask contains patients outside outer training")
    true_positive_ids = set(
        train_frame.loc[train_frame[target].astype(int) == 1, "patient_id"].astype(str)
    )
    if not hidden.issubset(true_positive_ids):
        raise ValueError("A hidden-positive mask contains nonpositive patients")
