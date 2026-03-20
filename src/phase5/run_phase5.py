"""
Phase 5 Runner — Structural Equity Audit
==========================================
Runs Phase 5 scripts sequentially.

Usage:
    python src/phase5/run_phase5.py
    # or via main pipeline: SUD_PHASE5=true python src/run_pipeline.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path

PHASE5_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE5_DIR.parent.parent

SCRIPTS = [
    PHASE5_DIR / "download_datasets.py",           # Step 1: acquire data
    PHASE5_DIR / "prepare_teds.py",                # Step 2: process TEDS-D
    PHASE5_DIR / "prepare_caldata.py",             # Step 3: process CALDATA (optional)
    PHASE5_DIR / "generate_stratified_cohort.py",  # Step 4: synthetic fallback
    PHASE5_DIR / "pathway_attribution_model.py",   # Step 5: Bayesian model
    PHASE5_DIR / "equity_audit_figures.py",         # Step 6: figures
    PHASE5_DIR / "phase5_report.py",               # Step 7: report
]


def main():
    print("=" * 60)
    print("PHASE 5: STRUCTURAL EQUITY AUDIT — PIPELINE")
    print("=" * 60)

    total_start = time.time()

    for i, script_path in enumerate(SCRIPTS, 1):
        script_name = script_path.name
        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(SCRIPTS)}] Running {script_name} ...")
        print(f"{'─' * 60}\n")

        step_start = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            env=os.environ,
            cwd=str(PROJECT_ROOT),
        )

        elapsed = time.time() - step_start

        if result.returncode != 0:
            print(f"\n✗ FAILED: {script_name} (exit code {result.returncode})")
            sys.exit(result.returncode)

        print(f"\n✓ {script_name} completed in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"PHASE 5 COMPLETE — {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
