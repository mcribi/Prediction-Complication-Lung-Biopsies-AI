#!/bin/bash
#SBATCH --job-name=shap_r07_clean
#SBATCH --partition=muylarga
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/mnt/homeGPU/mcribilles/conda_envs/vc/bin/python}"
SCRIPT=$ROOT/run_shap_r07_clean.py
RESULT="${RESULT:-/mnt/homeGPU/mcribilles/tfm/codigo/radiomica_multietiqueta/radiomic_results_two_outputs_derived_thresholds_20260827/R07__two_outputs__derived_no_complication__nested_thresholds}"
BASE="${BASE:-/mnt/homeGPU/mcribilles/tfm/codigo/radiomica_multietiqueta/experiments/radiomica/run_radiomics_classification_210_resampling.py}"
TWO="${TWO_RUNNER:?Set TWO_RUNNER to run_radiomics_two_outputs_thresholds.py}"
FEATURES="${FEATURES:-/mnt/homeGPU/mcribilles/tfm/codigo/radiomica_multietiqueta/radiomica_grids/multimask_clinical_grid_massive_fix2__2026-06-03__12-53-45/features/radiomics__basic__lung__nodule__vessels__clin_full.csv}"
LABELS="${LABELS:-/mnt/homeGPU/mcribilles/tfm/codigo/radiomica_multietiqueta/radiomica_grids/multimask_clinical_grid_massive_fix2__2026-06-03__12-53-45/labels/labels_main_drop_dpleural_only.csv}"
VENDOR_ROOT="$(dirname "$ROOT")/vendor"
if [[ -d "$VENDOR_ROOT" ]]; then export PYTHONPATH="$VENDOR_ROOT:${PYTHONPATH:-}"; fi
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

$PYTHON -m py_compile "$SCRIPT"
$PYTHON -c "import shap, sklearn; print('shap', shap.__version__, 'sklearn', sklearn.__version__)"
rm -rf "$ROOT/output_smoke" "$ROOT/output_final"

$PYTHON "$SCRIPT" \
  --result-dir "$RESULT" \
  --output-dir "$ROOT/output_smoke" \
  --base-runner "$BASE" \
  --two-runner "$TWO" \
  --features-csv "$FEATURES" \
  --labels-csv "$LABELS" \
  --max-folds 1

$PYTHON "$SCRIPT" \
  --result-dir "$RESULT" \
  --output-dir "$ROOT/output_final" \
  --base-runner "$BASE" \
  --two-runner "$TWO" \
  --features-csv "$FEATURES" \
  --labels-csv "$LABELS"
