#!/bin/bash
#SBATCH --job-name=shap_r07_figs
#SBATCH --partition=muylarga
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/mnt/homeGPU/mcribilles/conda_envs/vc/bin/python}"
VENDOR_ROOT="$(dirname "$ROOT")/vendor"
if [[ -d "$VENDOR_ROOT" ]]; then export PYTHONPATH="$VENDOR_ROOT:${PYTHONPATH:-}"; fi
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
rm -rf "$ROOT/output_publication"
$PYTHON -m py_compile "$ROOT/make_publication_figures_r07.py"
$PYTHON "$ROOT/make_publication_figures_r07.py" \
  --analysis-dir "$ROOT/output_final" \
  --output-dir "$ROOT/output_publication" \
  --max-display 15 \
  --waterfall-display 12
