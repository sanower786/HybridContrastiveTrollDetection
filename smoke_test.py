#!/usr/bin/env python3
"""
smoke_test.py
A lightweight end-to-end test for Hybrid Contrastive–Classification Troll Detection.
This script verifies that preprocessing, training, and evaluation pipelines execute successfully.
"""

import os
import subprocess
import sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "processed_data.npz")
RESULTS_DIR = os.path.join(ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "smoke_model.pt")

def run_cmd(cmd):
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(f"❌ Command failed: {' '.join(cmd)}")

def main():
    print("🧪 Starting smoke test for Hybrid Troll Detection")

    # Step 1. Check data exists
    if not os.path.exists(DATA_PATH):
        print("⚠️ processed_data.npz not found, running minimal preprocessing...")
        input_csv = os.path.join(ROOT, "data", "sample_data.csv")
        run_cmd([sys.executable, "preprocess.py", "--input", input_csv, "--output", DATA_PATH])

    assert os.path.exists(DATA_PATH), "❌ Data preprocessing failed!"

    # Step 2. Load data to ensure X,y structure is valid
    d = np.load(DATA_PATH, allow_pickle=True)
    assert "X" in d and "y" in d, "❌ processed_data.npz missing X or y"
    print(f"✅ Loaded data: X{d['X'].shape}, y{d['y'].shape}")

    # Step 3. Run a short training (1 epoch)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_cmd([
        sys.executable, "train_hybrid_model.py",
        "--data", DATA_PATH,
        "--epochs", "1",
        "--batch_size", "16",
        "--results_dir", RESULTS_DIR,
        "--save_name", "smoke_model.pt"
    ])

    # Step 4. Verify model checkpoint exists
    assert os.path.exists(MODEL_PATH), "❌ Model checkpoint not created!"

    # Step 5. Run evaluation on the model
    run_cmd([
        sys.executable, "evaluate.py",
        "--data", DATA_PATH,
        "--model", MODEL_PATH,
        "--out", RESULTS_DIR
    ])

    print("\n✅ Smoke test completed successfully — all core components ran without error.")
    print(f"Results and checkpoint available in {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
