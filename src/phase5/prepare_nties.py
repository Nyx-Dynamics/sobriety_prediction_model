"""
Phase 5: NTIES Preparation (Primary Dataset)
===============================================
National Treatment Improvement Evaluation Study (ICPSR 2884)
CSAT-funded programs serving underserved populations.
True 12-month follow-up with outcome analysis sample (IN_4411).

Variable mapping verified against inventory 2026-03-19.

Nyx Dynamics LLC | MIT MicroMasters Capstone
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import numpy as np
import pandas as pd

np.random.seed(config.SEED)

raw_dir = config.PROJECT_ROOT / "data" / "phase5" / "raw"
out_dir = config.PROJECT_ROOT / "data" / "phase5" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

intake_path = raw_dir / "nties_intake_nriq.tsv"
followup_path = raw_dir / "nties_followup_npaq.tsv"
clinical_path = raw_dir / "nties_clinical_nclu.tsv"

if not intake_path.exists() or not followup_path.exists():
    print("NTIES files not found — skipping")
    print(f"Expected: {intake_path}")
    print(f"Expected: {followup_path}")
    sys.exit(0)

# ══════════════════════════════════════════════════════════════════════
# 1. Load and link
# ══════════════════════════════════════════════════════════════════════

print("Loading NTIES files...")
intake = pd.read_csv(intake_path, sep="\t", low_memory=False)
followup = pd.read_csv(followup_path, sep="\t", low_memory=False)
print(f"  NRIQ (intake):   {intake.shape}")
print(f"  NPAQ (followup): {followup.shape}")

clinical = None
if clinical_path.exists():
    clinical = pd.read_csv(clinical_path, sep="\t", low_memory=False)
    print(f"  NCLU (clinical): {clinical.shape}")

# Inner join intake + followup on CASEID
df = intake.merge(followup, on="CASEID", how="inner", suffixes=("_int", "_fu"))
print(f"  Merged (intake+followup): {len(df)}")

# Left join clinical
if clinical is not None:
    df = df.merge(clinical, on="CASEID", how="left", suffixes=("", "_clin"))
    print(f"  After clinical merge: {len(df)}")

# Keep only IN_4411 outcome analysis sample
df = df[df["IN_4411"] == 1].copy()
print(f"  Outcome analysis sample (IN_4411==1): {len(df)}")

# ══════════════════════════════════════════════════════════════════════
# 2. Filter to stimulant/cocaine/crack/meth
# ══════════════════════════════════════════════════════════════════════

# AFDRUG1: 1=alcohol, 2=cocaine/crack, 3=heroin, 4=other opiates, 5=other drugs
# Stimulant = 2 (cocaine/crack) or 5 (other drugs incl meth/stimulants)
print(f"\n  AFDRUG1 value counts (pre-filter):")
print(df["AFDRUG1"].value_counts().sort_index().to_string())

stimulant_codes = [2, 5]
stim_mask = df["AFDRUG1"].isin(stimulant_codes)
df = df[stim_mask].copy()
print(f"\n  After stimulant filter (AFDRUG1 in {stimulant_codes}): {len(df)}")

# ══════════════════════════════════════════════════════════════════════
# 3. Treatment intensity (0-5) from TRT2
# ══════════════════════════════════════════════════════════════════════

# TRT2: 1=OP non-methadone, 2=OP methadone, 3=short-term residential, 4=long-term residential
trt2_map = {
    1: 2,   # outpatient non-methadone
    2: 3,   # outpatient methadone (MAT)
    3: 4,   # short-term residential
    4: 5,   # long-term residential
}
# Use TRT2 preferentially; use _int suffix if duplicated by merge
trt_col = "TRT2_int" if "TRT2_int" in df.columns else "TRT2"
df["treatment_intensity"] = df[trt_col].map(trt2_map).fillna(1).astype(int)

# ══════════════════════════════════════════════════════════════════════
# 4. Monitoring intensity (0-4) from NCLU
# ══════════════════════════════════════════════════════════════════════

# URINFREQ: 1=regular urine testing → 3, 2=not regular → 1
# FREQCOUN: 1=weekly+ → +1 bonus, 2=biweekly → +0, 3=less → +0
urine_map = {1: 3, 2: 1}
if "URINFREQ" in df.columns:
    df["urine_score"] = df["URINFREQ"].map(urine_map).fillna(1).astype(int)
else:
    df["urine_score"] = 1

freq_bonus = {1: 1, 2: 0, 3: 0}
if "FREQCOUN" in df.columns:
    df["counsel_bonus"] = df["FREQCOUN"].map(freq_bonus).fillna(0).astype(int)
else:
    df["counsel_bonus"] = 0

df["monitoring_intensity"] = np.clip(
    df["urine_score"] + df["counsel_bonus"], 0, 4
).astype(int)

# ══════════════════════════════════════════════════════════════════════
# 5. Social support density (0-10)
# ══════════════════════════════════════════════════════════════════════

# R61: 1=own place → 7.0, 2=someone else's place → 4.0
r61_map = {1: 7.0, 2: 4.0}
df["housing_score"] = df["R61"].map(r61_map).fillna(5.0)

# R72: 1=living with someone → +1.5, 2=living alone → +0.0
r72_map = {1: 1.5, 2: 0.0}
df["living_bonus"] = df["R72"].map(r72_map).fillna(0.5)

df["social_support_density"] = np.clip(
    df["housing_score"] + df["living_bonus"], 0, 10
).round(1)

# ══════════════════════════════════════════════════════════════════════
# 6. Self-advocacy index (0-10)
# ══════════════════════════════════════════════════════════════════════

# R53: 1=<HS → 2.0, 2=HS/GED → 5.0, 3=some college → 7.0, 4=college grad → 9.0
r53_map = {1: 2.0, 2: 5.0, 3: 7.0, 4: 9.0}
df["educ_score"] = df["R53"].map(r53_map).fillna(4.0)

# R67: 1=insured → +2.0, 2=uninsured → +0.0
r67_map = {1: 2.0, 2: 0.0}
df["insur_bonus"] = df["R67"].map(r67_map).fillna(0.0)

df["self_advocacy_index"] = np.clip(
    df["educ_score"] + df["insur_bonus"], 0, 10
).round(1)

# ══════════════════════════════════════════════════════════════════════
# 7. Outcome from P4 (drug use status at 12-month follow-up)
# ══════════════════════════════════════════════════════════════════════

# P4: 1=abstinent entire period, 2=abstinent 6+ months,
#     3=reduced use, 4=same or worse. -3=skip/missing.
# event=0 for abstinent (1,2), event=1 for active use (3,4)
print(f"\n  P4 (outcome) distribution:")
print(df["P4"].value_counts().sort_index().to_string())

# Drop rows with missing outcome
df = df[df["P4"].isin([1, 2, 3, 4])].copy()
df["event"] = np.where(df["P4"].isin([1, 2]), 0, 1)
print(f"  After dropping missing P4: {len(df)}")
print(f"  Event rate: {df['event'].mean():.3f}")

# Cross-check with substance-specific use columns
# P400A2 (cocaine), P400A3 (crack), P400A10 (meth):
# -4 = did not use, -7 = refused, positive values = used
stim_use_cols = ["P400A2", "P400A3", "P400A10"]
available = [c for c in stim_use_cols if c in df.columns]
if available:
    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    any_stim_use = df[available].apply(
        lambda row: any(v >= 1 for v in row if pd.notna(v)), axis=1
    )
    cross_check_rate = any_stim_use.mean()
    print(f"  Cross-check: {cross_check_rate:.3f} had any stimulant use (P400A2/A3/A10)")

# ══════════════════════════════════════════════════════════════════════
# 8. Survival time from LOS (months → days)
# ══════════════════════════════════════════════════════════════════════

df["days_to_event"] = np.clip(
    (df["LOS"].fillna(3) * 30).astype(int), 1, 365
)

# ══════════════════════════════════════════════════════════════════════
# 9. SUD type
# ══════════════════════════════════════════════════════════════════════

# AFDRUG2: -4 = no secondary drug → stimulant only
# Any other positive code → polysubstance
df["sud_type"] = np.where(
    df["AFDRUG2"].isin([-4, -9]), 0, 1
)  # 0=stimulant, 1=polysubstance

# ══════════════════════════════════════════════════════════════════════
# 10. Demographics
# ══════════════════════════════════════════════════════════════════════

df["caseid"] = df["CASEID"].astype(str).apply(lambda x: f"NTIES-{x}")

# R5M1: 1=Male, 2=Female
df["sex"] = df["R5M1"].map({1: "Male", 2: "Female"}).fillna("Unknown")

# No direct age variable in NRIQ — use 35 as population mean
df["age_at_enrollment"] = 35

# R50: 1=first treatment, 2=second, 3=third, 4=four+
r50_map = {1: 0, 2: 1, 3: 2, 4: 3}
df["prior_tx_episodes"] = df["R50"].map(r50_map).fillna(0).astype(int)

# ══════════════════════════════════════════════════════════════════════
# 11. MH comorbidity from NCLU
# ══════════════════════════════════════════════════════════════════════

# PSYCHIAT: 1=psychiatric services available, 2=no
if "PSYCHIAT" in df.columns:
    df["mh_comorbidity"] = np.where(df["PSYCHIAT"] == 1, 1, 0)
else:
    df["mh_comorbidity"] = 0

# ══════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════

output_cols = [
    "caseid", "sud_type", "age_at_enrollment", "sex",
    "treatment_intensity", "monitoring_intensity",
    "social_support_density", "self_advocacy_index",
    "mh_comorbidity", "prior_tx_episodes", "event", "days_to_event",
]
result = df[output_cols].copy()
result.to_csv(out_dir / "nties_phase5.csv", index=False)

print(f"\n{'=' * 60}")
print(f"NTIES PHASE 5 PREPARATION COMPLETE")
print(f"{'=' * 60}")
print(f"  Output: {out_dir / 'nties_phase5.csv'}")
print(f"  N: {len(result)}")
print(f"  Event rate: {result['event'].mean():.3f}")
print(f"  SUD type: stimulant={( result['sud_type']==0).sum()}, "
      f"polysubstance={(result['sud_type']==1).sum()}")

print(f"\n  treatment_intensity:")
print(result["treatment_intensity"].value_counts().sort_index().to_string())

print(f"\n  Mean social_support_density by treatment_intensity:")
print(result.groupby("treatment_intensity")["social_support_density"].mean().round(1).to_string())

print(f"\n  MH comorbidity rate: {result['mh_comorbidity'].mean():.1%}")
print(f"  Mean days_to_event: {result['days_to_event'].mean():.0f}")
