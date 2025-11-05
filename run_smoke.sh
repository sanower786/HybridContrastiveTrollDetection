#!/usr/bin/env bash
set -euo pipefail

echo "[SMOKE] Starting smoke test"

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO_ROOT" || exit 1

LOG_DIR="${REPO_ROOT}/results"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/smoke_test.log"

PY="${PY:-python}"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "Repo root: $REPO_ROOT"
log "Log file: $LOG_FILE"

# Create venv if missing
VENV_DIR="${REPO_ROOT}/.venv_smoke"
if [ ! -d "$VENV_DIR" ]; then
  log "Creating virtualenv at $VENV_DIR"
  $PY -m venv "$VENV_DIR"
fi

# Activation: check unix and windows-style
ACTIVATE_UNIX="$VENV_DIR/bin/activate"
ACTIVATE_WIN="$VENV_DIR/Scripts/activate"

if [ -f "$ACTIVATE_UNIX" ]; then
  log "Activating venv (unix): $ACTIVATE_UNIX"
  # shellcheck disable=SC1091
  . "$ACTIVATE_UNIX"
elif [ -f "$ACTIVATE_WIN" ]; then
  log "Activating venv (windows-gitbash): $ACTIVATE_WIN"
  # shellcheck disable=SC1091
  . "$ACTIVATE_WIN"
else
  echo "[ERR] No activate script found in $VENV_DIR (expected bin/activate or Scripts/activate)" | tee -a "$LOG_FILE"
  exit 3
fi

# Upgrade pip
log "Upgrading pip and setuptools"
python -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true

# Install smoke requirements if present
if [ -f "${REPO_ROOT}/requirements_smoke.txt" ]; then
  log "Installing smoke requirements from requirements_smoke.txt"
  pip install -r "${REPO_ROOT}/requirements_smoke.txt" 2>&1 | tee -a "$LOG_FILE" || true
else
  log "requirements_smoke.txt not found — installing minimal packages"
  pip install numpy pandas scikit-learn >/dev/null 2>&1 || true
fi

# Optional toy data generator
TOY_GEN="${REPO_ROOT}/sample_data/generate_sample_data.py"
TOY_DIR="${REPO_ROOT}/sample_data/toy_dataset"
if [ -f "$TOY_GEN" ]; then
  log "Generating toy dataset via $TOY_GEN -> $TOY_DIR"
  mkdir -p "$TOY_DIR"
  python "$TOY_GEN" --out-dir "$TOY_DIR" 2>&1 | tee -a "$LOG_FILE" || {
    log "Toy data generator failed (continuing if you have real data)"
  }
else
  log "No toy generator found. If you have a sample CSV, place it under data/sample_data.csv"
fi

# Find training script: check common names then search src/
TRAIN_SCRIPT=""
if [ -f "${REPO_ROOT}/train_hybrid_model.py" ]; then
  TRAIN_SCRIPT="${REPO_ROOT}/train_hybrid_model.py"
elif [ -f "${REPO_ROOT}/train.py" ]; then
  TRAIN_SCRIPT="${REPO_ROOT}/train.py"
else
  # search src/
  found=$(find "${REPO_ROOT}/src" -type f -iname "*train*.py" -print -quit 2>/dev/null || true)
  if [ -n "$found" ]; then
    TRAIN_SCRIPT="$found"
  fi
fi

if [ -z "$TRAIN_SCRIPT" ]; then
  log "ERROR: Could not find a training script (train_hybrid_model.py or train.py or src/*train*.py)."
  exit 4
fi

log "Found training script: $TRAIN_SCRIPT"

# Prepare smoke args (adapt if your script uses different flags)
SMOKE_EPOCHS=1
SMOKE_BATCH=16
SMOKE_INPUT=""
if [ -d "$TOY_DIR" ]; then
  SMOKE_INPUT="$TOY_DIR"
elif [ -f "${REPO_ROOT}/data/sample_data.csv" ]; then
  SMOKE_INPUT="${REPO_ROOT}/data/sample_data.csv"
fi

# Build command string (POSIX-friendly)
TRAIN_CMD="python \"$TRAIN_SCRIPT\" --epochs $SMOKE_EPOCHS --batch_size $SMOKE_BATCH"
if [ -n "$SMOKE_INPUT" ]; then
  TRAIN_CMD="$TRAIN_CMD --input \"$SMOKE_INPUT\""
fi

log "Running training (1 epoch) with command: $TRAIN_CMD"
# Run training
sh -c "$TRAIN_CMD" 2>&1 | tee -a "$LOG_FILE" || {
  log "Training script exited with non-zero code (check $LOG_FILE)."
  exit 5
}

log "Smoke training finished successfully. Check results/ and models/ for outputs."
log "Smoke test complete ✅"

# Try to deactivate if available
if command -v deactivate >/dev/null 2>&1; then
  deactivate >/dev/null 2>&1 || true
fi

exit 0
