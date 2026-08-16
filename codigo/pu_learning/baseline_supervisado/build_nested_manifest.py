#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path("/mnt/homeGPU/mcribilles/tfm/codigo/pu_learning/baseline_supervisado")
RAD = Path("/mnt/homeGPU/mcribilles/tfm/codigo/radiomica_multietiqueta/radiomica_grids/multimask_clinical_grid_imbalance_small__2026-06-06__17-58-35")
FEATURES = {
    "clinical_geometry": RAD / "features/clinical_only__clin_sd_geom.csv",
    "radiomics": RAD / "features/radiomics__basic__lung__nodule__vessels.csv",
    "clinical_geometry_radiomics": RAD / "features/radiomics__basic__lung__nodule__vessels__clin_sd_geom.csv",
}
TARGETS = ["Hemorragia", "Neumotórax"]
FORMULATIONS = ["supervised_all_negatives", "supervised_clean_negatives", "pu_naive", "pu_elkan_noto_oof", "pu_bagging"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run-id", required=True); args = p.parse_args()
    run_dir = BASE / "nested_grid_runs" / args.run_id
    manifest_dir = run_dir / "manifest"; results = run_dir / "results"; logs = run_dir / "slurm_outputs"
    for d in [manifest_dir, results, logs]: d.mkdir(parents=True, exist_ok=True)
    labels = RAD / "labels/labels_main_drop_dpleural_only.csv"
    source_folds = BASE / "manifests/expanded_grid_20260811_1205/folds_labelset_stratified_s42.csv"
    source_cohort = BASE / "manifests/expanded_grid_20260811_1205/shared_cohort_patient_ids.csv"
    folds = pd.read_csv(source_folds); cohort = pd.read_csv(source_cohort)
    folds.to_csv(manifest_dir / "outer_folds.csv", index=False); cohort.to_csv(manifest_dir / "cohort_ids.csv", index=False)
    rows=[]; task=0
    for target in TARGETS:
        for feature_set, features_csv in FEATURES.items():
            for formulation in FORMULATIONS:
                rows.append({"task_id":task,"target":target,"feature_set":feature_set,"formulation":formulation,"features_csv":str(features_csv),"labels_csv":str(labels),"folds_csv":str(manifest_dir/"outer_folds.csv"),"cohort_ids_csv":str(manifest_dir/"cohort_ids.csv"),"output_root":str(results),"seed":42}); task+=1
    pd.DataFrame(rows).to_csv(manifest_dir/"manifest.csv",index=False)
    metadata={"run_id":args.run_id,"created_at":datetime.now().isoformat(),"n_tasks":len(rows),"targets":TARGETS,"feature_sets":list(FEATURES),"formulations":FORMULATIONS,"n_outer_folds":5,"inner_cv":"four source folds inside each outer-train partition","seed":42,"excluded_features":["Sexo_binaria"],"required_features":["Edad"],"source_hashes":{"labels":sha256(labels),"outer_folds":sha256(source_folds),"cohort_ids":sha256(source_cohort),**{k:sha256(v) for k,v in FEATURES.items()}},"selection":{"primary":"average_precision","tie_breakers":["gmean","mcc","candidate_index"],"threshold":"inner OOF maximize gmean then mcc then f1"}}
    (manifest_dir/"run_metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    print(json.dumps(metadata,indent=2,ensure_ascii=False))

if __name__ == "__main__": main()
