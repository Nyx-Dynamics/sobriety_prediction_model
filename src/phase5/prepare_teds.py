"""
Phase 5: TEDS-D 2022 Preparation
==================================
Processes TEDS-D discharge data into the Phase 5 equity analysis format.
Falls back to synthetic stratified cohort if TEDS-D is not available.

Nyx Dynamics LLC | MIT MicroMasters Capstone
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import numpy as np
import pandas as pd
from glob import glob

np.random.seed(config.SEED)

raw_dir = config.PROJECT_ROOT / "data" / "phase5" / "raw"
out_dir = config.PROJECT_ROOT / "data" / "phase5" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

# ── Find the TEDS-D CSV ──────────────────────────────────────────────
teds_candidates = list(raw_dir.rglob("*.csv"))
teds_path = None
for c in teds_candidates:
    if "teds" in c.name.lower() or "tedsd" in c.name.lower():
        teds_path = c
        break
# Also check for the extracted zip structure
if teds_path is None:
    for c in teds_candidates:
        if c.stat().st_size > 1_000_000:  # large CSV likely the dataset
            teds_path = c
            break

if teds_path is None:
    print("TEDS-D not found in data/phase5/raw/.")
    print("Run: python src/phase5/download_datasets.py")
    print("Falling back to synthetic stratified cohort.")

    # Import and run synthetic generator as fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import importlib
    spec = importlib.util.spec_from_file_location(
        "generate_stratified_cohort",
        Path(__file__).resolve().parent / "generate_stratified_cohort.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.exit(0)

# ══════════════════════════════════════════════════════════════════════
# 1. Load
# ══════════════════════════════════════════════════════════════════════

print(f"Loading TEDS-D from: {teds_path}")
df = pd.read_csv(teds_path, low_memory=False)
print(f"TEDS-D loaded: {df.shape}")
print(f"Columns: {df.columns.tolist()[:20]}...")

# Normalize column names to uppercase
df.columns = df.columns.str.upper()

# ══════════════════════════════════════════════════════════════════════
# 2. Filter to stimulant and polysubstance
# ══════════════════════════════════════════════════════════════════════

stimulant_codes = [5, 6, 7]  # 5=cocaine/crack, 6=other stim, 7=meth
sud_mask = df["SUB1"].isin(stimulant_codes)
df = df[sud_mask].copy()
print(f"After stimulant filter (SUB1 in {stimulant_codes}): {len(df)} rows")

# Classify SUD type
df["sud_type"] = np.where(
    df["SUB2"].notna() & (df["SUB2"] != -9),
    "Polysubstance Use Disorder",
    "Stimulant Use Disorder",
)

# ══════════════════════════════════════════════════════════════════════
# 3. Treatment intensity (0-5)
# ══════════════════════════════════════════════════════════════════════

services_map = {
    1: 2,   # detox inpatient
    2: 1,   # detox ambulatory
    3: 4,   # rehab/residential non-hospital
    4: 5,   # rehab/residential hospital
    5: 3,   # ambulatory intensive (IOP)
    6: 2,   # ambulatory non-intensive
    7: 3,   # opioid treatment program
    8: 2,   # other
}
df["treatment_intensity"] = df["SERVICES"].map(services_map).fillna(0).astype(int)

# ══════════════════════════════════════════════════════════════════════
# 4. Monitoring intensity (0-3)
# ══════════════════════════════════════════════════════════════════════

# METHUSE is the MAT flag in TEDS-D 2022 (check available columns)
mat_col = None
for candidate in ["METHFLG", "METHUSE", "ALCHFLG"]:
    if candidate in df.columns:
        mat_col = candidate
        break

if mat_col:
    mat_map = {1: 3, 2: 3, 3: 2, 4: 2}  # methadone/bup→3, naltrexone/other→2
    df["mat_score"] = df[mat_col].map(mat_map).fillna(0).astype(int)
else:
    df["mat_score"] = 0

# Residential bonus
residential_bonus = df["treatment_intensity"].isin([4, 5]).astype(int)
df["monitoring_intensity"] = np.clip(df["mat_score"] + residential_bonus, 0, 4).astype(int)

# Cap: outpatient-only without MAT → max 1
outpatient_no_mat = (df["treatment_intensity"] <= 2) & (df["mat_score"] == 0)
df.loc[outpatient_no_mat, "monitoring_intensity"] = df.loc[outpatient_no_mat, "monitoring_intensity"].clip(upper=1)

# ══════════════════════════════════════════════════════════════════════
# 5. Social support density (0-10)
# ══════════════════════════════════════════════════════════════════════

livarag_map = {1: 1.0, 2: 4.0, 3: 7.0}
df["livarag_score"] = df["LIVARAG"].map(livarag_map).fillna(4.0)

employ_bonus_map = {1: 2.0, 2: 1.0, 3: 0.0, 4: 0.5}
df["employ_bonus"] = df["EMPLOY_D"].map(employ_bonus_map) if "EMPLOY_D" in df.columns else (
    df["EMPLOY"].map(employ_bonus_map) if "EMPLOY" in df.columns else 1.0
)
df["employ_bonus"] = df["employ_bonus"].fillna(1.0)

df["social_support_density"] = np.clip(df["livarag_score"] + df["employ_bonus"], 0, 10).round(1)

# ══════════════════════════════════════════════════════════════════════
# 6. Self-advocacy index (0-10)
# ══════════════════════════════════════════════════════════════════════

educ_map = {1: 1.0, 2: 3.0, 3: 5.0, 4: 7.0, 5: 9.0}
df["educ_score"] = df["EDUC"].map(educ_map).fillna(4.0)

priminc_bonus_map = {1: 2.0, 2: -1.0, 3: 1.0, 4: 0.0}
df["priminc_bonus"] = df["PRIMINC"].map(priminc_bonus_map).fillna(0.0)

df["self_advocacy_index"] = np.clip(df["educ_score"] + df["priminc_bonus"], 0, 10).round(1)

# ══════════════════════════════════════════════════════════════════════
# 7. Outcome variable
# ══════════════════════════════════════════════════════════════════════

# TEDS-D does not have relapse follow-up.
# Discharge reason is a structural proxy:
# completed=stable, dropout/terminated/incarcerated=failure.
# CALDATA provides the validation dataset with direct
# 1-year follow-up abstinence measurement.

reason_event_map = {
    1: 0,  # treatment completed → censored (survived)
    2: 1,  # dropped out / left AMA → failure
    3: 1,  # terminated by facility → failure
    4: 0,  # transferred → censored
    5: 1,  # incarcerated → failure
    6: 1,  # death → failure
    7: 0,  # other → censored
}
df["event"] = df["REASON"].map(reason_event_map).fillna(0).astype(int)

# ══════════════════════════════════════════════════════════════════════
# 8. Survival time (LOS)
# ══════════════════════════════════════════════════════════════════════

df["days_to_event"] = df["LOS"].fillna(1).clip(lower=1, upper=365).astype(int)

# ══════════════════════════════════════════════════════════════════════
# 9. MH comorbidity
# ══════════════════════════════════════════════════════════════════════

df["mh_comorbidity"] = np.where(df["PSYPROB"] == 1, 1, 0)

# ══════════════════════════════════════════════════════════════════════
# 10. Demographics
# ══════════════════════════════════════════════════════════════════════

# Age — TEDS codes age ranges, use midpoints
age_midpoint = {
    1: 14, 2: 16, 3: 19, 4: 22, 5: 27, 6: 32,
    7: 37, 8: 42, 9: 47, 10: 52, 11: 57, 12: 62,
}
df["age_at_enrollment"] = df["AGE"].map(age_midpoint).fillna(35).astype(int)

# Sex
gender_map = {1: "Male", 2: "Female"}
df["sex"] = df["GENDER"].map(gender_map).fillna("Female")

# Race
race_map = {
    1: "American Indian/Alaska Native", 2: "Asian",
    3: "Black", 4: "Hawaiian/Pacific Islander",
    5: "White", 6: "Other/Multiracial",
}
df["race_ethnicity"] = df["RACE"].map(race_map).fillna("Unknown")

# Prior treatment episodes
df["prior_tx_episodes"] = df["NOPRIOR"].fillna(0).clip(lower=0).astype(int)

# STFIPS for state
df["state"] = df["STFIPS"] if "STFIPS" in df.columns else np.nan

# ══════════════════════════════════════════════════════════════════════
# Build final output
# ══════════════════════════════════════════════════════════════════════

# Generate caseid
df["caseid"] = [f"TEDS-{i:06d}" for i in range(len(df))]

output_cols = [
    "caseid", "state", "sud_type", "age_at_enrollment", "sex",
    "race_ethnicity", "treatment_intensity", "monitoring_intensity",
    "social_support_density", "self_advocacy_index", "mh_comorbidity",
    "prior_tx_episodes", "event", "days_to_event",
]

result = df[output_cols].copy()
result.to_csv(out_dir / "teds_phase5.csv", index=False)

# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEDS-D 2022 — PHASE 5 PREPARATION COMPLETE")
print("=" * 60)
print(f"  Output: {out_dir / 'teds_phase5.csv'}")
print(f"  Total rows: {len(result)}")
print(f"  Event rate: {result['event'].mean():.3f}")

print(f"\n  treatment_intensity distribution:")
print(result["treatment_intensity"].value_counts(normalize=True).sort_index().round(3).to_string())

print(f"\n  Mean days_to_event by treatment_intensity:")
print(result.groupby("treatment_intensity")["days_to_event"].mean().round(1).to_string())

print(f"\n  social_support_density distribution:")
print(result["social_support_density"].describe().round(2).to_string())

print(f"\n  self_advocacy_index distribution:")
print(result["self_advocacy_index"].describe().round(2).to_string())

print(f"\n  Null counts:")
nulls = result.isnull().sum()
print(nulls[nulls > 0].to_string() if nulls.sum() > 0 else "  None")
