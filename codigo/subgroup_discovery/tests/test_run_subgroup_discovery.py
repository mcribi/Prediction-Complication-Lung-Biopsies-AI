import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from run_subgroup_discovery import (
    Selector,
    apply_clinical_grouping,
    beam_search,
    build_multilabel_strata,
    build_parser,
    compute_metrics,
    load_inputs,
    make_repeated_splits,
    merge_extra_targets,
    normalize_config,
    select_diverse_rules,
    validate_cohort,
)


class TestMetrics(unittest.TestCase):
    def test_compute_metrics_uses_global_prevalence_for_wracc(self):
        y = np.array([1, 1, 0, 0, 0], dtype=int)
        mask = np.array([True, True, True, False, False])
        metrics = compute_metrics(mask, y)
        self.assertEqual(metrics["n_covered"], 3)
        self.assertEqual(metrics["n_positive_covered"], 2)
        self.assertAlmostEqual(metrics["coverage"], 3 / 5)
        self.assertAlmostEqual(metrics["subgroup_prevalence"], 2 / 3)
        self.assertAlmostEqual(metrics["global_prevalence"], 2 / 5)
        self.assertAlmostEqual(metrics["wracc"], (3 / 5) * ((2 / 3) - (2 / 5)))
        self.assertAlmostEqual(metrics["sensitivity"], 1.0)

    def test_compute_metrics_handles_empty_coverage(self):
        y = np.array([1, 0, 1], dtype=int)
        mask = np.zeros(3, dtype=bool)
        metrics = compute_metrics(mask, y)
        self.assertEqual(metrics["n_covered"], 0)
        self.assertTrue(np.isnan(metrics["subgroup_prevalence"]))
        self.assertTrue(np.isnan(metrics["lift"]))


class TestClinicalPreparation(unittest.TestCase):
    def test_grouping_is_reproducible_and_removes_source_columns(self):
        df = pd.DataFrame(
            {
                "Fibrosis": [1, 0],
                "Fibrosis_pulmonar": [0, 1],
                "DM": [0, 1],
                "DM1": [1, 0],
                "DM2": [0, 0],
                "HTA": [1, 0],
                "Cardiopatía": [0, 1],
                "SCACEST": [0, 0],
                "Hiperuricemia": [1, 0],
                "HBP": [0, 1],
            }
        )
        grouped = apply_clinical_grouping(df)
        self.assertListEqual(grouped["Fibrosis_any"].tolist(), [1, 1])
        self.assertListEqual(grouped["DM_any"].tolist(), [1, 1])
        self.assertListEqual(grouped["CardioRisk"].tolist(), [1, 1])
        self.assertListEqual(grouped["Urologic"].tolist(), [0, 1])
        self.assertNotIn("Fibrosis", grouped.columns)
        self.assertNotIn("Hiperuricemia", grouped.columns)

    def test_validate_cohort_rejects_wrong_size_and_duplicate_ids(self):
        good = pd.DataFrame({"patient_id": [f"p{i}" for i in range(210)]})
        validate_cohort(good, expected_n=210)
        with self.assertRaisesRegex(ValueError, "210"):
            validate_cohort(good.iloc[:-1], expected_n=210)
        duplicated = good.copy()
        duplicated.loc[1, "patient_id"] = duplicated.loc[0, "patient_id"]
        with self.assertRaisesRegex(ValueError, "unicos"):
            validate_cohort(duplicated, expected_n=210)

    def test_extra_targets_are_merged_before_target_validation(self):
        clinical = pd.DataFrame({"patient_id": ["p1", "p2"], "Edad": [60, 70]})
        extra = pd.DataFrame({"patient_id": ["p2", "p1"], "error_modelo": [1, 0]})
        merged = merge_extra_targets(clinical, extra, ["error_modelo"])
        self.assertListEqual(merged["error_modelo"].tolist(), [0, 1])
        with self.assertRaisesRegex(ValueError, "Missing values"):
            merge_extra_targets(
                clinical,
                extra.iloc[:1],
                ["error_modelo"],
            )

    def test_default_inputs_use_compact_predictors_and_separate_targets(self):
        config = build_parser().parse_args([])
        self.assertTrue(config.clinical_csv.endswith("/clinical_data_multimodal.csv"))
        self.assertTrue(config.targets_csv.endswith("/clinical_data_limpios_SD.csv"))
        self.assertTrue(config.geometry_csv.endswith("/nodule_geometry_210.csv"))

    def test_load_inputs_aligns_three_sources_and_keeps_compact_features(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            clinical = pd.DataFrame(
                {
                    "patient_id": ["p1", "p2", "p3"],
                    "Edad": [50, 60, 70],
                    "Sin_factor_de_riesgo": [1, 0, 0],
                    "Sin_patología_pulmonar": [0, 1, 0],
                }
            )
            targets = pd.DataFrame(
                {
                    "patient_id": ["p3", "p1", "p2"],
                    "Hemorragia": [1, 0, 1],
                    "Neumotórax": [0, 1, 0],
                    "Sin_complicación": [0, 0, 1],
                }
            )
            geometry = pd.DataFrame(
                {
                    "patient_id": ["p2", "p3", "p1"],
                    "tamano_nodulo_mm": [20.0, 30.0, 10.0],
                    "profundidad_min_pleura_mm": [2.0, 3.0, 1.0],
                    "profundidad_centroidal_pleura_mm": [5.0, 6.0, 4.0],
                }
            )
            clinical_path = tmp / "clinical.csv"
            targets_path = tmp / "targets.csv"
            geometry_path = tmp / "geometry.csv"
            clinical.to_csv(clinical_path, index=False)
            targets.to_csv(targets_path, index=False)
            geometry.to_csv(geometry_path, index=False)

            config = normalize_config(
                build_parser().parse_args(
                    [
                        "--clinical-csv",
                        str(clinical_path),
                        "--targets-csv",
                        str(targets_path),
                        "--geometry-csv",
                        str(geometry_path),
                        "--expected-patients",
                        "3",
                        "--min-feature-count",
                        "1",
                    ]
                )
            )
            feature_sets, audit, metadata = load_inputs(config)

            clinical_full, aligned_targets = feature_sets["clinical_full"]
            self.assertListEqual(
                clinical_full.columns.tolist(),
                ["Edad", "Sin_factor_de_riesgo", "Sin_patología_pulmonar"],
            )
            self.assertListEqual(aligned_targets["Hemorragia"].tolist(), [0, 1, 1])
            self.assertEqual(len(feature_sets["clinical_geometry"][0]), 3)
            self.assertTrue(audit["kept"].all())
            self.assertEqual(metadata["targets_csv"], str(targets_path))
            self.assertIn("targets_sha256", metadata)


class TestSearch(unittest.TestCase):
    def test_beam_search_finds_enriched_binary_rule(self):
        x = pd.DataFrame(
            {
                "risk": [1, 1, 1, 1, 0, 0, 0, 0],
                "noise": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        rules = beam_search(
            x,
            y,
            max_depth=2,
            min_coverage=0.25,
            beam_width=20,
            top_candidates=20,
        )
        self.assertTrue(rules)
        self.assertIn("risk == 1", rules[0].description)
        self.assertGreater(rules[0].train_metrics["wracc"], 0)

    def test_diversity_filter_removes_identical_coverage(self):
        x = pd.DataFrame({"a": [1, 1, 0, 0], "b": [1, 1, 0, 0]})
        y = np.array([1, 1, 0, 0])
        rules = beam_search(x, y, max_depth=1, min_coverage=0.25, beam_width=10, top_candidates=10)
        selected = select_diverse_rules(rules, x, max_rules=5, max_jaccard=0.8)
        self.assertEqual(len(selected), 1)

    def test_selector_numeric_threshold_is_applied_without_refitting(self):
        selector = Selector(feature="age", operator=">", value=60.0)
        train = pd.DataFrame({"age": [50.0, 70.0]})
        test = pd.DataFrame({"age": [59.0, 61.0, 100.0]})
        self.assertListEqual(selector.mask(train).tolist(), [False, True])
        self.assertListEqual(selector.mask(test).tolist(), [False, True, True])


class TestSplitting(unittest.TestCase):
    def test_multilabel_strata_and_repeated_splits_cover_each_sample_once_per_repeat(self):
        labels = pd.DataFrame(
            {
                "Neumotórax": [0] * 10 + [1] * 10 + [0] * 10 + [1] * 10,
                "Hemorragia": [0] * 20 + [1] * 20,
            }
        )
        strata = build_multilabel_strata(labels, n_splits=5)
        self.assertEqual(len(strata), 40)
        splits = make_repeated_splits(labels, n_splits=5, n_repeats=2, seed=13)
        self.assertEqual(len(splits), 10)
        for repeat in range(2):
            test_indices = []
            for item in splits:
                if item.repeat == repeat:
                    self.assertEqual(len(set(item.train_idx) & set(item.test_idx)), 0)
                    test_indices.extend(item.test_idx.tolist())
            self.assertListEqual(sorted(test_indices), list(range(40)))


if __name__ == "__main__":
    unittest.main()
