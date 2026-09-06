#!/bin/bash
#SBATCH --job-name=shap_r02_figures
#SBATCH --partition=muylarga
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/mnt/homeGPU/mcribilles/conda_envs/vc/bin/python}"
export PYTHONPATH="$ROOT/vendor:${PYTHONPATH:-}"
export MPLBACKEND=Agg
$PYTHON -m py_compile "$ROOT/code/make_publication_figures.py"
$PYTHON "$ROOT/code/make_publication_figures.py" \
  --analysis-dir "$ROOT/output/final" \
  --output-dir "$ROOT/output/final/figures_memoria"
