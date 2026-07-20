#!/bin/bash
#SBATCH -J rad210ml
#SBATCH -p muylarga
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -o /home/mcribilles/tfm/slurm_outputs/%A_%a.out
#SBATCH -e /home/mcribilles/tfm/slurm_outputs/%A_%a.err

set -xeo pipefail
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU
export JOBLIB_MULTIPROCESSING=0
export LOKY_MAX_CPU_COUNT=1
export JOBLIB_START_METHOD=spawn

ENV_PATH="${ENV_PATH:-pyradiomics_env}"
CODE_ROOT="/home/mcribilles/tfm/codigo"
CODE_DIR="${CODE_DIR:-$CODE_ROOT/radiomica}"
SCRIPT="${SCRIPT:-$CODE_DIR/run_radiomics_classification_210.py}"
CONFIG_FILE="${CONFIG_FILE:?CONFIG_FILE is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/mcribilles/tfm/codigo/radiomic_results/multilabel_210}"
LABELS_CSV="${LABELS_CSV:-/home/mcribilles/tfm/clinical_data/210pacientes/clinical_data_limpios.csv}"
FEATURES_CSV="${FEATURES_CSV:-}"
MAPPING_CSV="${MAPPING_CSV:-}"
JOB_SUFFIX="${JOB_SUFFIX:-slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export PS1=${PS1-}
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
set -u

mkdir -p /home/mcribilles/tfm/slurm_outputs
mkdir -p "$OUTPUT_ROOT"
cd "$CODE_DIR"

CFG_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

echo "[SLURM] JOBID=$SLURM_JOB_ID ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID HOST=$(hostname) DATE=$(date)"
echo "[SLURM] SCRIPT=$SCRIPT"
echo "[SLURM] CONFIG_FILE=$CONFIG_FILE"
echo "[SLURM] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[SLURM] LABELS_CSV=$LABELS_CSV"
echo "[SLURM] FEATURES_CSV=${FEATURES_CSV:-<from-config>}"
echo "[SLURM] MAPPING_CSV=${MAPPING_CSV:-<none>}"
echo "[SLURM] JOB_SUFFIX=$JOB_SUFFIX"
echo "[SLURM] EXTRA_ARGS=$EXTRA_ARGS"
echo "[SLURM] PYTHON=$(command -v python)"
python -c "import sys; print('[PY]', sys.executable, sys.version)"

CMD=(python -u "$SCRIPT" "$CFG_INDEX" --config_file "$CONFIG_FILE" --output_root "$OUTPUT_ROOT" --labels_csv "$LABELS_CSV" --job_suffix "$JOB_SUFFIX")

if [[ -n "$FEATURES_CSV" ]]; then
  CMD+=(--features_csv "$FEATURES_CSV")
fi

if [[ -n "$MAPPING_CSV" ]]; then
  CMD+=(--mapping_csv "$MAPPING_CSV")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a EXTRA_ARR <<< "$EXTRA_ARGS"
  CMD+=("${EXTRA_ARR[@]}")
fi

echo "[SLURM] Running config index = $CFG_INDEX"
printf '[SLURM] COMMAND:'
printf ' %q' "${CMD[@]}"
printf '\n'

stdbuf -oL -eL "${CMD[@]}"

echo "[SLURM] DONE config index = $CFG_INDEX"
