#!/bin/bash
#SBATCH -J radiomics
#SBATCH -p dios
#SBATCH -w hera
#SBATCH --gres=gpu:0
#SBATCH -c 8
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.err

# --- logs en vivo ---
set -xeo pipefail
export PYTHONUNBUFFERED=1

# BLAS/OpenMP: evitar conflictos
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU

# --- rutas ---
ENV_PATH="/mnt/homeGPU/mcribilles/conda_envs/vc"
CODE_DIR="/mnt/homeGPU/mcribilles/tfm/codigo"
SCRIPT="$CODE_DIR/radiomica/run_radiomc.py"
CONFIG_FILE="$CODE_DIR/radiomica/valid_configurations_radiomics_unique.json"

# --- activar conda ---
export PS1=${PS1-}
set +u
source /opt/anaconda/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
set -u

echo "[SLURM] JOBID=$SLURM_JOB_ID HOST=$(hostname) DATE=$(date)"
echo "[SLURM] Using python: $(command -v python)"
python -c "import sys,os; print('[PY]', sys.executable, sys.version); print('[PY] CWD', os.getcwd())"

cd "$CODE_DIR"
echo "[SLURM] Now in: $(pwd)"

# --- contar configuraciones ---
TOTAL_CFGS=$(python - "$CONFIG_FILE" <<'PY'
import json,sys
with open(sys.argv[1],'r') as f:
    print(len(json.load(f)))
PY
)
echo "[SLURM] TOTAL_CFGS=$TOTAL_CFGS"

# --- ejecutar todas las configs en secuencia ---
for IDX in $(seq 0 $((TOTAL_CFGS-1))); do
  echo "[SLURM] Running config index = $IDX"
  stdbuf -oL -eL python -u "$SCRIPT" "$IDX"
done

echo "[SLURM] DONE ALL CONFIGS"
