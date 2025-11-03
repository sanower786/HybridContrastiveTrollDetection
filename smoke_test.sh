#!/usr/bin/env bash
# run_smoke.sh  -- robust smoke test for HybridContrastiveTrollDetection
# Usage: ./run_smoke.sh
set -euo pipefail
IFS=$'\n\t'

echo "[SMOKE] Starting smoke_test at $(date)"

# repo root (dir of this script)
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO_ROOT"
echo "[SMOKE] Repo root: $REPO_ROOT"

# Python executable (can override with env PY=python3)
PY=${PY:-python}

RESULTS_DIR="${REPO_ROOT}/results"
LOG="${RESULTS_DIR}/smoke_test.log"
mkdir -p "$RESULTS_DIR"
rm -f "$LOG"

echo "[SMOKE] Using Python: $($PY -V 2>&1 || echo 'python not found')"

# 1) Toy data generation: prefer sample_data/generate_sample_data.py if present
TOY_GENERATOR="${REPO_ROOT}/sample_data/generate_sample_data.py"
SAMPLE_CSV="${REPO_ROOT}/data/sample_data.csv"
TOY_DATA_DIR="${REPO_ROOT}/sample_data/toy_dataset"

if [[ -f "$TOY_GENERATOR" ]]; then
  echo "[SMOKE] Found toy data generator: $TOY_GENERATOR"
  mkdir -p "$TOY_DATA_DIR"
  # run generator; it should write to toy_dataset or similar
  $PY "$TOY_GENERATOR" --outdir "$TOY_DATA_DIR" 2>&1 | tee -a "$LOG"
elif [[ -f "$SAMPLE_CSV" ]]; then
  echo "[SMOKE] No generator found, but sample CSV exists: $SAMPLE_CSV"
  mkdir -p "$TOY_DATA_DIR"
  cp "$SAMPLE_CSV" "${TOY_DATA_DIR}/sample_data.csv"
  echo "[SMOKE] Copied sample_data.csv to $TOY_DATA_DIR" | tee -a "$LOG"
else
  echo "[SMOKE] No generator or sample CSV found. Creating tiny synthetic CSV..."
  mkdir -p "$TOY_DATA_DIR"
  cat > "${TOY_DATA_DIR}/sample_data.csv" <<EOF
text,label
"this is a harmless comment",0
"you are an idiot",1
"thanks for sharing",0
"I will destroy your argument",1
EOF
  echo "[SMOKE] Tiny toy dataset created at ${TOY_DATA_DIR}/sample_data.csv" | tee -a "$LOG"
fi

# 2) Locate training script (prefer user-provided train_hybrid_model.py)
TRAIN_SCRIPT=""
if [[ -f "${REPO_ROOT}/train_hybrid_model.py" ]]; then
  TRAIN_SCRIPT="${REPO_ROOT}/train_hybrid_model.py"
elif [[ -f "${REPO_ROOT}/src/alignment/alignment_train.py" ]]; then
  TRAIN_SCRIPT="${REPO_ROOT}/src/alignment/alignment_train.py"
else
  # search in src for something that looks like a train script
  TRAIN_SCRIPT="$(find "${REPO_ROOT}/src" -type f -iname "*train*.py" | head -n 1 || true)"
fi

if [[ -z "$TRAIN_SCRIPT" ]]; then
  echo "[ERR] Could not find a training script (train_hybrid_model.py or src/*train*.py)." | tee -a "$LOG"
  echo "[SMOKE] Available files under repo root/src:" | tee -a "$LOG"
  ls -la "${REPO_ROOT}/src" 2>&1 | tee -a "$LOG" || true
  exit 2
fi

echo "[SMOKE] Found training script: $TRAIN_SCRIPT" | tee -a "$LOG"

# 3) Run training in smoke mode: small subset / one epoch
# The training script should accept arguments: --data, --epochs, --batch-size, etc.
# We try common flags; if not supported the script will error and we log that.

DATA_ARG="${TOY_DATA_DIR}/sample_data.csv"
EPOCHS=1
BATCH=16

echo "[SMOKE] Running training for 1 epoch on toy data..."
# run and log; keep the exit code to bubble up
set +e
$PY "$TRAIN_SCRIPT" --input "$DATA_ARG" --epochs $EPOCHS --batch_size $BATCH 2>&1 | tee -a "$LOG"
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo "[SMOKE] Training finished with exit code $EXIT_CODE" | tee -a "$LOG"

# 4) Post-check: list results
echo "[SMOKE] Results directory listing:" | tee -a "$LOG"
ls -la "$RESULTS_DIR" 2>&1 | tee -a "$LOG" || true

echo "[SMOKE] Tail of log (last 60 lines):"
tail -n 60 "$LOG" || true

echo "[SMOKE] Smoke test completed at $(date). Exit code: $EXIT_CODE"
exit $EXIT_CODE
