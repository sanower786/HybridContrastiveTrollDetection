#!/usr/bin/env bash
# run_smoke.sh
# Usage: ./run_smoke.sh
# Creates venv (if needed), installs deps, runs smoke_test.py and writes smoke_test.log

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="results"
LOG_FILE="${LOG_DIR}/smoke_test.log"
VENV_DIR=".venv_smoke"

# Ensure results/ exists
mkdir -p "$LOG_DIR"

echo "=== Run Smoke Test ==="
echo "Working dir: $SCRIPT_DIR"
echo "Log file: $LOG_FILE"
echo "Virtualenv dir: $VENV_DIR"
echo

# If a virtualenv is already active, use it; otherwise create/use local venv
if [[ -n "${VIRTUAL_ENV-}" ]]; then
  echo "Using already active virtualenv: $VIRTUAL_ENV"
else
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtualenv in $VENV_DIR ..."
    python -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  echo "Activated virtualenv: $(which python)"
fi

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install dependencies (fail fast if requirements.txt missing)
if [[ -f "requirements.txt" ]]; then
  echo "Installing requirements from requirements.txt ..."
  pip install --no-cache-dir -r requirements.txt
else
  echo "WARNING: requirements.txt not found in repo root. Skipping pip install."
fi

# Run smoke test and capture logs
echo "Running smoke_test.py ..."
if [[ ! -f "smoke_test.py" ]]; then
  echo "ERROR: smoke_test.py not found in repo root." | tee "$LOG_FILE"
  exit 2
fi

# Run and tee output to log (preserves exit code)
set +e
python smoke_test.py 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo
echo "Smoke test finished with exit code: $EXIT_CODE"
echo "Logs stored at: $LOG_FILE"

# List results folder files (for CI artifact upload)
if [[ -d "$LOG_DIR" ]]; then
  echo
  echo "Contents of ${LOG_DIR}:"
  ls -la "$LOG_DIR" || true
fi

# Deactivate venv if we activated locally
if [[ -z "${VIRTUAL_ENV-}" ]]; then
  deactivate || true
fi

exit "$EXIT_CODE"
