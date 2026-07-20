#!/usr/bin/env bash
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

if [[ -z "${CONFIG_JSON:-}" || -z "${OUTPUT_ROOT:-}" || -z "${SCRIPT:-}" || -z "${ENV_PYTHON:-}" ]]; then
  echo "Missing required env vars: CONFIG_JSON OUTPUT_ROOT SCRIPT ENV_PYTHON" >&2
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"
TMP_JSON="$(mktemp)"
trap 'rm -f "$TMP_JSON"' EXIT

"$ENV_PYTHON" - "$CONFIG_JSON" "$TASK_ID" > "$TMP_JSON" <<'PY'
import json
import sys
config_json = sys.argv[1]
task_id = int(sys.argv[2])
with open(config_json, encoding='utf-8') as f:
    configs = json.load(f)
item = configs[task_id]
print(json.dumps(item, ensure_ascii=False))
PY

PREPROC_NAME="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["preproc_name"])' "$TMP_JSON")"
PREPROC_DIR="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["preproc_dir"])' "$TMP_JSON")"
RUN_OUT_DIR="$OUTPUT_ROOT/$PREPROC_NAME"
mkdir -p "$RUN_OUT_DIR"

"$ENV_PYTHON" "$SCRIPT" \
  --preproc_dir "$PREPROC_DIR" \
  --preproc_name "$PREPROC_NAME" \
  --which_masks "all" \
  --feature_mode "both" \
  --out_dir "$RUN_OUT_DIR" \
  --out_prefix "radiomics_preprocessed" \
  --expected_total 210 \
  --skip_reduction \
  --save_raw_per_mask
