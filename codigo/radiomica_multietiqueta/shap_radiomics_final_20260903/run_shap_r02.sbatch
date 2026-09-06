#!/bin/bash
#SBATCH --job-name=shap_r02_final
#SBATCH --partition=muylarga
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/mnt/homeGPU/mcribilles/conda_envs/vc/bin/python}"
SCRIPT=$ROOT/code/run_shap_r02.py
RESULT="${RESULT:-/mnt/homeGPU/mcribilles/tfm/codigo/radiomica_multietiqueta/radiomic_results_two_outputs_derived_thresholds_20260827/R02__two_outputs__derived_no_complication__nested_thresholds}"
export PYTHONPATH="$ROOT/vendor:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

$PYTHON -m py_compile "$SCRIPT"
$PYTHON -c "import shap; print('shap', shap.__version__)"

$PYTHON "$SCRIPT" \
  --result-dir "$RESULT" \
  --output-dir "$ROOT/output/smoke_fold1_retry1" \
  --max-folds 1 \
  --skip-local

$PYTHON "$SCRIPT" \
  --result-dir "$RESULT" \
  --output-dir "$ROOT/output/final" \
  --max-display 20 \
  --waterfall-display 12
