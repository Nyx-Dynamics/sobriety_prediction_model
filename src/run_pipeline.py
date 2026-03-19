"""
Pipeline Runner — Sobriety Prediction Model
=============================================
Runs pipeline scripts sequentially, sharing a single RUN_ID
via the SUD_RUN_TIMESTAMP environment variable.

DATA_MODE controls which track runs:
  "synthetic" (default) — generates data first, then runs pipeline
  "real" — skips generation, expects patient_static.csv and patient_panel.csv in RAW_DIR

Usage:
    python src/run_pipeline.py                          # synthetic mode
    SUD_DATA_MODE=real python src/run_pipeline.py       # real data mode
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
SYNTHETIC_DIR = SRC_DIR / "synthetic"
PIPELINE_DIR = SRC_DIR / "pipeline"

# Set the shared timestamp BEFORE any child process imports config
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ["SUD_RUN_TIMESTAMP"] = RUN_TIMESTAMP

# Data mode
DATA_MODE = os.environ.get("SUD_DATA_MODE", "synthetic")

# Training mode — only train models if explicitly requested
TRAIN_MODELS = os.environ.get("SUD_TRAIN_MODELS", "false").lower() == "true"


def main():
    if DATA_MODE == "synthetic":
        scripts = [
            SYNTHETIC_DIR / "generate_synthetic_cohort.py",
            PIPELINE_DIR / "build_static_matrix.py",
            PIPELINE_DIR / "build_panel.py",
            PIPELINE_DIR / "build_outcomes.py",
            PIPELINE_DIR / "build_lstm_tensors.py",
            PIPELINE_DIR / "build_splits.py",
        ]
    else:
        scripts = [
            PIPELINE_DIR / "build_static_matrix.py",
            PIPELINE_DIR / "build_panel.py",
            PIPELINE_DIR / "build_outcomes.py",
            PIPELINE_DIR / "build_lstm_tensors.py",
            PIPELINE_DIR / "build_splits.py",
        ]

    # Append training steps if requested
    if TRAIN_MODELS:
        scripts.extend([
            PIPELINE_DIR / "train_bayesian.py",
            PIPELINE_DIR / "train_lstm.py",
        ])

    print("=" * 70)
    print(f"SOBRIETY PREDICTION MODEL — FULL PIPELINE")
    print(f"Run timestamp: {RUN_TIMESTAMP}")
    print(f"Run ID: run_{RUN_TIMESTAMP}")
    print(f"Data mode: {DATA_MODE}")
    print(f"Train models: {TRAIN_MODELS}")
    print("=" * 70)

    total_start = time.time()

    for i, script_path in enumerate(scripts, 1):
        script_name = f"{script_path.parent.name}/{script_path.name}"
        print(f"\n{'─' * 70}")
        print(f"[{i}/{len(scripts)}] Running {script_name} ...")
        print(f"{'─' * 70}\n")

        step_start = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            env={**os.environ, "SUD_RUN_TIMESTAMP": RUN_TIMESTAMP},
            cwd=str(SRC_DIR.parent),
        )

        elapsed = time.time() - step_start

        if result.returncode != 0:
            print(f"\n✗ FAILED: {script_name} (exit code {result.returncode})")
            print(f"  Elapsed: {elapsed:.1f}s")
            print(f"  Pipeline aborted.")
            sys.exit(result.returncode)

        print(f"\n✓ {script_name} completed in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE")
    print(f"Run ID: run_{RUN_TIMESTAMP}")
    print(f"Data mode: {DATA_MODE}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
