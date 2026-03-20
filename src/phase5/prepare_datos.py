"""
Phase 5: DATOS Urine Validation (DS0010 Only)
================================================
Drug Abuse Treatment Outcome Study (ICPSR 2258)
Only DS0010 (urine test results) was available as .tsv.

This script provides urine-validated outcome analysis to
cross-check NTIES self-reported abstinence.

Columns verified from inventory:
  caseid, modality, DRUGPAT, AGE, QAV_RACE
  COCR (cocaine urine result), AMPR (amphetamine urine result)
  QEV_COC, QEV_AMP (evaluation flags)

Nyx Dynamics LLC | MIT MicroMasters Capstone
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import numpy as np
import pandas as pd

np.random.seed(config.SEED)

raw_dir = config.PROJECT_ROOT / "data" / "phase5" / "raw"
out_dir = config.PROJECT_ROOT / "data" / "phase5" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

urine_path = raw_dir / "datos_followup_urine.tsv"

if not urine_path.exists():
    print("DATOS urine file not found — skipping")
    print(f"Expected: {urine_path}")
    sys.exit(0)

# ══════════════════════════════════════════════════════════════════════
# Load
# ══════════════════════════════════════════════════════════════════════

print("Loading DATOS DS0010 (urine test results)...")
df = pd.read_csv(urine_path, sep="\t", low_memory=False)
print(f"  Shape: {df.shape}")

# Key urine columns from inventory: COCR, AMPR, QEV_COC, QEV_AMP, COCHX
urine_result_cols = [c for c in ["COCR", "AMPR", "COCHX"] if c in df.columns]
print(f"  Urine result columns: {urine_result_cols}")

for col in urine_result_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  {col} value counts:")
    print(f"    {df[col].value_counts().head(6).to_dict()}")

# ══════════════════════════════════════════════════════════════════════
# Filter to cocaine/stimulant patients
# ══════════════════════════════════════════════════════════════════════

# DRUGPAT: 1=cocaine, 2=heroin, 3=alcohol, 4=other
if "DRUGPAT" in df.columns:
    print(f"\n  DRUGPAT: {df['DRUGPAT'].value_counts().sort_index().to_dict()}")
    stim_mask = df["DRUGPAT"].isin([1, 4])  # cocaine + other (stimulants)
    df_stim = df[stim_mask].copy()
    print(f"  After stimulant filter: {len(df_stim)}")
else:
    df_stim = df.copy()
    print("  No DRUGPAT column — using all rows")

# ══════════════════════════════════════════════════════════════════════
# Urine positivity analysis
# ══════════════════════════════════════════════════════════════════════

# COCR = cocaine urine result (typically 1=positive, 0 or 2=negative)
# AMPR = amphetamine urine result
results = {}

if "COCR" in df_stim.columns:
    cocr_valid = df_stim["COCR"].dropna()
    cocr_positive = (cocr_valid == 1).sum()
    cocr_total = len(cocr_valid)
    cocr_rate = cocr_positive / max(cocr_total, 1)
    print(f"\n  Cocaine urine: {cocr_positive}/{cocr_total} positive ({cocr_rate:.1%})")
    results["cocaine_urine_positive_n"] = int(cocr_positive)
    results["cocaine_urine_total_n"] = int(cocr_total)
    results["cocaine_urine_positive_rate"] = round(cocr_rate, 4)

if "AMPR" in df_stim.columns:
    ampr_valid = df_stim["AMPR"].dropna()
    ampr_positive = (ampr_valid == 1).sum()
    ampr_total = len(ampr_valid)
    ampr_rate = ampr_positive / max(ampr_total, 1)
    print(f"  Amphetamine urine: {ampr_positive}/{ampr_total} positive ({ampr_rate:.1%})")
    results["amphetamine_urine_positive_n"] = int(ampr_positive)
    results["amphetamine_urine_total_n"] = int(ampr_total)
    results["amphetamine_urine_positive_rate"] = round(ampr_rate, 4)

# Any stimulant positive
any_positive = pd.Series(False, index=df_stim.index)
for col in ["COCR", "AMPR"]:
    if col in df_stim.columns:
        any_positive = any_positive | (df_stim[col] == 1)
any_pos_rate = any_positive.mean()
print(f"  Any stimulant positive: {any_positive.sum()}/{len(df_stim)} ({any_pos_rate:.1%})")
results["any_stimulant_positive_rate"] = round(any_pos_rate, 4)
results["n_stimulant_patients"] = int(len(df_stim))
results["n_total"] = int(len(df))

# Demographics
if "modality" in df_stim.columns:
    print(f"\n  modality: {df_stim['modality'].value_counts().sort_index().to_dict()}")
    results["modality_distribution"] = df_stim["modality"].value_counts().sort_index().to_dict()

# ══════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════

results["note"] = (
    "DATOS full intake/followup files (DS0001, DS0006) were not available "
    "in delimited format. Download from ICPSR 2258 for full validation. "
    "This file contains urine test results only (DS0010)."
)

with open(out_dir / "datos_urine_validation.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 60}")
print(f"DATOS urine validation summary saved")
print(f"  {out_dir / 'datos_urine_validation.json'}")
print(f"{'=' * 60}")
print()
print("NOTE: DATOS full intake/followup files were not available.")
print("Download DS0001 and DS0006 from ICPSR 2258 in delimited")
print("format for full validation with self-report outcomes.")
