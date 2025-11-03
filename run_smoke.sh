#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# run_smoke.sh
# Cross-platform smoke test runner (works with Git Bash on Windows, WSL, macOS, Linux)
# Creates a small venv, installs minimal dependencies, runs 1-epoch training, logs output
# ------------------------------------------------------------------------------

echo "[SMOKE] Starting smoke test"

# Repo root (directory containing this script)
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO_ROOT"

LOG_DIR="${REPO_ROOT}/results"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/smoke_test.log"

# Python to use (allow override)
PY="${PY:-python}"

echo "[SMOKE] Repo root: $REPO_ROOT"
echo "[SMOKE] Log file: $LOG_FILE"

# Simple timestamped logger function
log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# --- Create virtualenv if missing ---
VENV_DIR="${REPO_ROOT}/.venv_smoke"
if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtualenv at $VENV_DIR"
  $PY -m venv "$VENV_DIR"
fi

# Activation: support both unix and Windows layouts (Git Bash / WSL / Linux / macOS)
ACTIVATE_UNIX="$VENV_DIR/bin/activate"
ACTIVATE_WIN="$VENV_DIR/Scripts/activate"

if [[ -f "$ACTIVATE_UNIX" ]]; then
  log "Activating venv (unix): $ACTIVATE_UNIX"
  # shellcheck source=/dev/null
  source "$ACTIVATE_UNIX"
elif [[ -f "$ACTIVATE_WIN" ]]; then
  log "Activating venv (windows-gitbash): $ACTIVATE_WIN"
  # shellcheck source=/dev/null
  source "$ACTIVATE_WIN"
else
  echo "[ERR] No activate script found in $VENV_DIR (expected bin/activate or Scripts/activate)" | tee -a "$LOG_FILE"
  exit 3
fi

# Upgrade pip & basic tools
log "Upgrading pip and setuptools"
python -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true

# Install requirements for smoke
if [[ -f "${REPO_ROOT}/requirements_smoke.txt" ]]; then
  log "Installing smoke requirements from requirements_smoke.txt"
  pip install -r "${REPO_ROOT}/requirements_smoke.txt" 2>&1 | tee -a "$LOG_FILE"
else
  log "requirements_smoke.txt not found — installing minimal lightweight packages for smoke"
  # do not auto-install heavy packages like full torch by default
  pip install numpy pandas scikit-learn >/dev/null 2>&1 || true
fi

# Optional: generate toy data if generator exists
TOY_GEN="${REPO_ROOT}/sample_data/generate_sample_data.py"
TOY_DIR="${REPO_ROOT}/sample_data/toy_dataset"
if [[ -f "$TOY_GEN" ]]; then
  log "Generating toy dataset via $TOY_GEN -> $TOY_DIR"
  mkdir -p "$TOY_DIR"
  # Run generator; capture output
  python "$TOY_GEN" --out-dir "$TOY_DIR" 2>&1 | tee -a "$LOG_FILE" || {
    log "Toy data generator failed (continuing if you have real data)"
  }
else
  log "No toy generator found. If you have a sample CSV, place it under data/sample_data.csv"
fi

# Find training script: prefer top-level train_hybrid_model.py, then train.py, then search src/
CANDIDATES=(
  "${REPO_ROOT}/train_hybrid_model.py"
  "${REPO_ROOT}/train.py"
)
TRAIN_SCRIPT=""
for c in "${CANDIDATES[@]}"; do
  if [[ -f "$c" ]]; then
    TRAIN_SCRIPT="$c"
    break
  fi
done

if [[ -z "$TRAIN_SCRIPT" ]]; then
  # search src/ for something that looks like a train script
  TRAIN_SCRIPT="$(find "${REPO_ROOT}/src" -type f -iname "*train*.py" -print -quit 2>/dev/null || true) || true
fi

if [[ -z "$TRAIN_SCRIPT" ]]; then
  log "ERROR: Could not find a training script (train_hybrid_model.py or train.py or src/*train*.py)."
  log "Please add one or modify run_smoke.sh to point to your training entrypoint."
  exit 4
fi

log "Found training script: $TRAIN_SCRIPT"

# Prepare minimal args to keep smoke very small. If your script uses different CLI flags, edit below.
SMOKE_EPOCHS=1
SMOKE_BATCH=16
SMOKE_INPUT=""
# Prefer toy data if present
if [[ -d "$TOY_DIR" ]]; then
  SMOKE_INPUT="$TOY_DIR"
elif [[ -f "${REPO_ROOT}/data/sample_data.csv" ]]; then
  SMOKE_INPUT="${REPO_ROOT}/data/sample_data.csv"
fi

# Build command - adapt these flags if your training script expects different names
TRAIN_CMD=(python "$TRAIN_SCRIPT" --epochs "$SMOKE_EPOCHS" --batch_size "$SMOKE_BATCH")
# append input if the script supports --input or --data-root
if [[ -n "$SMOKE_INPUT" ]]; then
  TRAIN_CMD+=(--input "$SMOKE_INPUT")
fi

log "Running training (1 epoch) with command: ${TRAIN_CMD[*]}"
# Run the training and capture stdout/stderr to log
( "${TRAIN_CMD[@]}" ) 2>&1 | tee -a "$LOG_FILE" || {
  log "Training script exited with non-zero code (check $LOG_FILE)."
  exit 5
}

log "Smoke training finished successfully. Check results/ and models/ for outputs."
log "Smoke test complete ✅"

# Deactivate venv (if supported)
if command -v deactivate >/dev/null 2>&1; then
  deactivate >/dev/null 2>&1 || true
fi

exit 0
