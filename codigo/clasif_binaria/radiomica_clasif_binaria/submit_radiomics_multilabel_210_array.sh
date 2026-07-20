#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Uso: $0 /ruta/al/config.json [max_concurrency]" >&2
  exit 1
fi

CONFIG_FILE="$1"
MAX_CONCURRENCY="${2:-64}"
ENV_PATH="${ENV_PATH:-/mnt/homeGPU/mcribilles/conda_envs/vc}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/homeGPU/mcribilles/tfm/codigo/radiomic_results/multilabel_210}"
LABELS_CSV="${LABELS_CSV:-/mnt/homeGPU/mcribilles/tfm/clinical_data/210pacientes/clinical_data_limpios.csv}"
FEATURES_CSV="${FEATURES_CSV:-}"
MAPPING_CSV="${MAPPING_CSV:-}"
JOB_NAME="${JOB_NAME:-rad210ml}"
PARTITION="${PARTITION:-dios}"
SBATCH_FILE="/mnt/homeGPU/mcribilles/tfm/codigo/radiomica/run_radiomics_multilabel_210_array.sbatch"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: config file no existe: $CONFIG_FILE" >&2
  exit 1
fi

TOTAL_CFGS=$(python3 - "$CONFIG_FILE" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
data = json.loads(p.read_text())
if not isinstance(data, list):
    raise SystemExit('El config JSON debe contener una lista')
print(len(data))
PY
)

if [[ "$TOTAL_CFGS" -le 0 ]]; then
  echo "ERROR: el config file no contiene configuraciones" >&2
  exit 1
fi

ARRAY_SPEC="0-$((TOTAL_CFGS-1))%${MAX_CONCURRENCY}"

echo "[SUBMIT] CONFIG_FILE=$CONFIG_FILE"
echo "[SUBMIT] TOTAL_CFGS=$TOTAL_CFGS"
echo "[SUBMIT] ARRAY_SPEC=$ARRAY_SPEC"
echo "[SUBMIT] PARTITION=$PARTITION"
echo "[SUBMIT] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[SUBMIT] FEATURES_CSV=${FEATURES_CSV:-<from-config>}"
echo "[SUBMIT] LABELS_CSV=$LABELS_CSV"
echo "[SUBMIT] MAPPING_CSV=${MAPPING_CSV:-<none>}"
echo "[SUBMIT] EXTRA_ARGS=$EXTRA_ARGS"

sbatch \
  --job-name="$JOB_NAME" \
  --partition="$PARTITION" \
  --array="$ARRAY_SPEC" \
  --export=ALL,CONFIG_FILE="$CONFIG_FILE",ENV_PATH="$ENV_PATH",OUTPUT_ROOT="$OUTPUT_ROOT",LABELS_CSV="$LABELS_CSV",FEATURES_CSV="$FEATURES_CSV",MAPPING_CSV="$MAPPING_CSV",EXTRA_ARGS="$EXTRA_ARGS" \
  "$SBATCH_FILE"
