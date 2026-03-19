"""
Pipeline Runner — Sobriety Prediction Model
=============================================
Runs all 6 pipeline scripts sequentially, sharing a single RUN_ID
via the SUD_RUN_TIMESTAMP environment variable.

Usage:
    python src/run_pipeline.py
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

# Set the shared timestamp BEFORE any child process imports config
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ["SUD_RUN_TIMESTAMP"] = RUN_TIMESTAMP

SCRIPTS = [
    "generate_synthetic_cohort.py",
    "build_static_matrix.py",
    "build_panel.py",
    "build_outcomes.py",
    "build_lstm_tensors.py",
    "build_splits.py",
]

def main():
    print("=" * 70)
    print(f"SOBRIETY PREDICTION MODEL — FULL PIPELINE")
    print(f"Run timestamp: {RUN_TIMESTAMP}")
    print(f"Run ID: run_{RUN_TIMESTAMP}")
    print("=" * 70)

    total_start = time.time()

    for i, script_name in enumerate(SCRIPTS, 1):
        script_path = SRC_DIR / script_name
        print(f"\n{'─' * 70}")
        print(f"[{i}/{len(SCRIPTS)}] Running {script_name} ...")
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
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
