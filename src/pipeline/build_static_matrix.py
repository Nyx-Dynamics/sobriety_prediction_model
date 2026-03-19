"""
Build Static Feature Matrix (Steps 12-18)
==========================================
Sobriety Prediction Model | Nyx Dynamics LLC | MIT MicroMasters Capstone
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load source data (raw generator outputs) ─────────────────────────
static = pd.read_csv(config.RAW_DIR / "patient_static.csv")
panel = pd.read_csv(config.RAW_DIR / "patient_panel.csv")

# ── 12. patient_id — SHA-256 hash (simulate PHI anonymization) ───────
static["patient_id_hash"] = static["patient_id"].apply(
    lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
)

# ── 13. sud_type — stimulant=0, polysubstance=1 ──────────────────────
static["sud_type"] = (static["sud_primary"] == "Polysubstance Use Disorder").astype(int)

# ── 14. sud_severity — numeric 0-10 from DSM-5 categories ────────────
# DSM-5: Mild (2-3 criteria) → ~3, Moderate (4-5) → ~6, Severe (6+) → ~8
# Add noise to simulate ASI-like composite variability
severity_map = {"Mild": 3.0, "Moderate": 6.0, "Severe": 8.5}
static["sud_severity"] = static["sud_severity_baseline"].map(severity_map)
static["sud_severity"] += np.random.normal(0, 0.8, len(static))
static["sud_severity"] = static["sud_severity"].clip(0, 10).round(1)

# ── 15. prior_tx_episodes — already integer, pass through ────────────
# (already exists in source)

# ── 16. housing_stability — derive from panel life-event data ────────
# Aggregate housing instability across all visits: stable=0, unstable=1, homeless=2
housing_instab_rate = panel.groupby("patient_id")["le_housing_instability"].mean()
housing_map = pd.cut(
    housing_instab_rate,
    bins=[-0.01, 0.05, 0.15, 1.0],
    labels=[0, 1, 2]  # stable, unstable, homeless
).astype(int)
static["housing_stability"] = static["patient_id"].map(housing_map)

# ── 17. social_support — use baseline; impute missing from context ───
# If missing, derive from housing + emergency contact proxy
static["social_support"] = static["social_support_baseline"].copy()
# Simulate ~3% missingness
missing_mask = np.random.random(len(static)) < 0.03
static.loc[missing_mask, "social_support"] = np.nan

# Impute: use housing_stability as proxy (stable→6, unstable→4, homeless→2)
impute_vals = static["housing_stability"].map({0: 6.0, 1: 4.0, 2: 2.0})
static["social_support"] = static["social_support"].fillna(impute_vals)
static["social_support"] = static["social_support"].clip(0, 10).round(1)

# ── 18. MH diagnosis flags — rename to requested column names ────────
mh_rename = {
    "mh_major_depression": "mdd",
    "mh_ptsd": "ptsd",
    "mh_gad": "gad",
    "mh_adhd": "adhd",
    "mh_bipolar": "bipolar_i",
    "mh_personality_disorder": "bpd",
    "mh_psychotic_disorder": "psychosis",
}
for old, new in mh_rename.items():
    static[new] = static[old]

# Add bipolar_ii (split from bipolar: ~40% of bipolar cases are type II)
static["bipolar_ii"] = 0
bipolar_mask = static["bipolar_i"] == 1
n_bipolar = bipolar_mask.sum()
flip_to_ii = np.random.random(n_bipolar) < 0.40
static.loc[bipolar_mask, "bipolar_ii"] = flip_to_ii.astype(int)
static.loc[static["bipolar_ii"] == 1, "bipolar_i"] = 0  # mutually exclusive

# ── Assemble final static matrix ─────────────────────────────────────
output_cols = [
    "patient_id", "patient_id_hash",
    "sud_type", "sud_severity", "prior_tx_episodes",
    "housing_stability", "social_support",
    # Demographics (retain for modeling)
    "age_at_enrollment", "sex", "race_ethnicity",
    "route_of_admin", "treatment_modality", "insurance_status",
    "age_first_use", "substances_involved",
    # MH flags
    "mdd", "ptsd", "gad", "adhd", "bipolar_i", "bipolar_ii", "psychosis", "bpd",
    # Medical comorbidities
    "med_hcv", "med_hiv", "med_chronic_pain", "med_tbi_history",
    "med_liver_disease", "med_cardiovascular", "med_ms_autoimmune",
]

sud_static = static[output_cols].copy()

# ── Save ──────────────────────────────────────────────────────────────
sud_static.to_csv(config.STATIC_DIR / "sud_static.csv", index=False)

# ══════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STATIC FEATURE MATRIX — sud_static.csv")
print("=" * 70)
print(f"Shape: {sud_static.shape}")
print(f"\nDtypes:\n{sud_static.dtypes}")
print(f"\nFirst 3 rows (transposed):\n{sud_static.head(3).T}")

# ── Class imbalance check: sud_type ──────────────────────────────────
print("\n── Class Balance: sud_type ──")
counts = sud_static["sud_type"].value_counts()
print(counts)
ratio = counts.min() / counts.max()
print(f"Minority/Majority ratio: {ratio:.3f}")
if ratio < 0.5:
    print("⚠  WARNING: Significant class imbalance detected (ratio < 0.5)")
elif ratio < 0.8:
    print("⚠  MILD imbalance (ratio 0.5–0.8). Consider stratified splits.")
else:
    print("✓  Classes are reasonably balanced (ratio ≥ 0.8)")

# ── MH co-occurrence heatmap ─────────────────────────────────────────
mh_cols = ["mdd", "ptsd", "gad", "adhd", "bipolar_i", "bipolar_ii", "psychosis", "bpd"]
mh_matrix = sud_static[mh_cols]

# Co-occurrence: count patients with both conditions
n = len(mh_matrix)
co_occ = pd.DataFrame(index=mh_cols, columns=mh_cols, dtype=float)
for c1 in mh_cols:
    for c2 in mh_cols:
        co_occ.loc[c1, c2] = ((mh_matrix[c1] == 1) & (mh_matrix[c2] == 1)).sum() / n

print(f"\n── MH Co-occurrence Matrix (proportion of cohort) ──")
print(co_occ.round(3).to_string())

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    co_occ.astype(float), annot=True, fmt=".3f", cmap="YlOrRd",
    square=True, linewidths=0.5, ax=ax,
    cbar_kws={"label": "Proportion of cohort"}
)
ax.set_title("Mental Health Co-occurrence Matrix\n(N=2,000 synthetic SUD cohort)", fontsize=13)
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "mh_cooccurrence_heatmap.png", dpi=150)
print(f"\n✓ Heatmap saved: {config.FIGURES_DIR / 'mh_cooccurrence_heatmap.png'}")

# ── Flag patients missing >20% of features ───────────────────────────
# Count non-demographic features for missingness check
feature_cols = [c for c in output_cols if c not in ["patient_id", "patient_id_hash"]]
missing_pct = sud_static[feature_cols].isnull().mean(axis=1)
flagged = sud_static[missing_pct > 0.20]
print(f"\n── Missingness Check ──")
print(f"Total features checked: {len(feature_cols)}")
print(f"Patients missing >20% of features: {len(flagged)}")
if len(flagged) > 0:
    print(f"Flagged IDs: {flagged['patient_id'].tolist()}")
else:
    print("✓ No patients exceed 20% missingness threshold")

# Overall missingness summary
total_missing = sud_static[feature_cols].isnull().sum()
if total_missing.sum() > 0:
    print(f"\nPer-column null counts:")
    print(total_missing[total_missing > 0])
else:
    print("✓ Zero nulls across all feature columns")
