import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import pu3d_core as p


class TestPU3DCore(unittest.TestCase):
    def test_nnpu_risk_matches_definition_when_unlabeled_risk_nonnegative(self):
        logits = torch.tensor([1.2, 0.4, 0.7, 1.1])
        observed = torch.tensor([1.0, 1.0, 0.0, 0.0])
        prior = 0.3
        pos = logits[observed == 1]
        unl = logits[observed == 0]
        expected = prior * F.softplus(-pos).mean() + (
            F.softplus(unl).mean() - prior * F.softplus(pos).mean()
        )
        self.assertTrue(torch.allclose(p.nnpu_loss(logits, observed, prior), expected))

    def test_nnpu_risk_uses_nonnegative_correction(self):
        logits = torch.tensor([5.0, 5.0, -5.0, -5.0])
        observed = torch.tensor([1.0, 1.0, 0.0, 0.0])

        loss = p.nnpu_loss(logits, observed, class_prior=0.9)

        positive = logits[observed == 1]
        unlabeled = logits[observed == 0]
        negative_risk = (
            F.softplus(unlabeled).mean()
            - 0.9 * F.softplus(positive).mean()
        )

        self.assertLess(float(negative_risk), 0.0)
        self.assertTrue(torch.allclose(loss, -negative_risk))


    def test_estimate_c_and_prior_are_bounded(self):
        labels = np.array([1, 1, 0, 0, 0])
        probs = np.array([0.8, 0.6, 0.4, 0.2, 0.1])
        c = p.estimate_c(labels, probs)
        prior = p.estimate_class_prior(labels, c)
        self.assertAlmostEqual(c, 0.7)
        self.assertAlmostEqual(prior, (2 / 5) / 0.7)
        self.assertTrue(0.0 < prior < 1.0)

    def test_choose_threshold_prefers_gmean(self):
        y = np.array([1, 1, 0, 0])
        score = np.array([0.9, 0.7, 0.6, 0.1])
        threshold, metrics = p.choose_threshold(y, score)
        self.assertAlmostEqual(metrics["gmean"], 1.0)
        self.assertGreaterEqual(threshold, 0.6)
        self.assertLessEqual(threshold, 0.7)

    def test_binary_metrics_contains_ranking_threshold_and_calibration(self):
        y = np.array([1, 1, 0, 0])
        score = np.array([0.9, 0.8, 0.2, 0.1])
        m = p.binary_metrics(y, score >= 0.5, score)
        for key in ["average_precision", "roc_auc", "f1", "sensitivity", "specificity", "gmean", "mcc", "brier_score"]:
            self.assertIn(key, m)
        self.assertEqual(m["tp"], 2)
        self.assertEqual(m["tn"], 2)

    def test_load_outer_fold_has_no_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "folds.csv"
            pd.DataFrame({
                "fold": [1, 1, 1, 2, 2, 2],
                "split": ["train", "train", "test", "train", "test", "train"],
                "patient_id": ["a", "b", "c", "a", "b", "c"],
            }).to_csv(path, index=False)
            train, test = p.load_outer_fold(path, 1)
            self.assertEqual(train, ["a", "b"])
            self.assertEqual(test, ["c"])
            self.assertFalse(set(train) & set(test))

    def test_inner_split_keeps_test_external_unseen(self):
        df = pd.DataFrame({
            "patient_id": [str(i) for i in range(40)],
            "target": [0, 1] * 20,
        })
        outer_train = [str(i) for i in range(32)]
        external_test = set(str(i) for i in range(32, 40))
        train, val = p.make_inner_split(df, outer_train, "target", seed=7, val_ratio=0.25)
        self.assertFalse(set(train) & set(val))
        self.assertFalse((set(train) | set(val)) & external_test)
        self.assertEqual(set(train) | set(val), set(outer_train))


if __name__ == "__main__":
    unittest.main()
