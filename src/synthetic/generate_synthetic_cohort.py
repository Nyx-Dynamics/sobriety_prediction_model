"""
Synthetic Longitudinal Cohort Generator (Phase 0 Integration)
==============================================================
Sobriety Prediction Model for Stimulant & Polysubstance Use Disorder
Nyx Dynamics LLC | MIT MicroMasters Capstone

Generates a realistic panel dataset of 2,000 patients with quarterly
observations over 3 years, using calibrated hazard parameters from
phase0_seed.py to produce distinct Remission vs Cycling latent classes.

Output columns are matched exactly to what build_static_matrix.py expects.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import Phase 0 calibrated parameters from same directory
from phase0_seed import (
    base_hazard_remission, base_hazard_cycling,
    mh_hazard_multipliers, tv_effects,
    stress_decay_rate, cascade_pathway_weight, involuntary_pathway_weight,
    p_remission_stimulant, p_remission_poly,
    mh_excess_penalty_per_dx, mh_excess_threshold,
)

np.random.seed(config.SEED)

# ── Configuration ──────────────────────────────────────────────────────
N_PATIENTS = 2000
MAX_QUARTERS = 12
ENROLLMENT_START = datetime(2021, 1, 1)
ENROLLMENT_END = datetime(2023, 6, 30)
STUDY_END = datetime(2024, 12, 31)

# ══════════════════════════════════════════════════════════════════════
# PATIENT-LEVEL (STATIC) FEATURES
# ══════════════════════════════════════════════════════════════════════

patient_ids = [f"SYN-{i:04d}" for i in range(1, N_PATIENTS + 1)]

# ── Demographics (updated clinical prevalence distributions) ──────────
age_at_enrollment = np.clip(
    np.random.normal(34, 9, N_PATIENTS).astype(int), 18, 65
)

sex = np.random.choice(
    ["Male", "Female", "Non-binary"],
    N_PATIENTS, p=[0.52, 0.38, 0.10]
)

race_ethnicity = np.random.choice(
    ["White", "Black", "Hispanic", "Asian", "Other/Multiracial"],
    N_PATIENTS, p=[0.48, 0.22, 0.18, 0.06, 0.06]
)

# ── SUD characteristics ───────────────────────────────────────────────
sud_primary = np.random.choice(
    ["Stimulant Use Disorder", "Polysubstance Use Disorder"],
    N_PATIENTS, p=[0.45, 0.55]
)

substance_categories = [
    "Methamphetamine", "Cocaine/Crack", "Opioids", "Alcohol",
    "Benzodiazepines", "Cannabis", "MDMA/Club drugs"
]

substances_involved = []
for i in range(N_PATIENTS):
    if sud_primary[i] == "Stimulant Use Disorder":
        primary = np.random.choice(["Methamphetamine", "Cocaine/Crack"])
        extras = []
        if np.random.random() < 0.3:
            extras = list(np.random.choice(
                [s for s in substance_categories if s != primary],
                size=1, replace=False
            ))
        substances_involved.append(";".join([primary] + extras))
    else:
        n_sub = np.random.randint(2, 5)
        subs = list(np.random.choice(substance_categories, size=n_sub, replace=False))
        substances_involved.append(";".join(subs))

sud_severity = np.random.choice(
    ["Mild", "Moderate", "Severe"],
    N_PATIENTS, p=[0.15, 0.35, 0.50]
)

route_of_admin = np.random.choice(
    ["Smoking", "Intranasal", "IV", "Oral", "Mixed"],
    N_PATIENTS, p=[0.35, 0.28, 0.18, 0.12, 0.07]
)

treatment_modality = np.random.choice(
    ["IOP", "Outpatient", "MAT", "Residential", "Detox_only", "None"],
    N_PATIENTS, p=[0.38, 0.25, 0.18, 0.12, 0.05, 0.02]
)

insurance = np.random.choice(
    ["Medicaid", "Uninsured", "Private", "Medicare", "VA"],
    N_PATIENTS, p=[0.42, 0.28, 0.22, 0.05, 0.03]
)

prior_tx_episodes = np.random.poisson(1.5, N_PATIENTS)

age_first_use = np.clip(
    (age_at_enrollment - np.random.exponential(12, N_PATIENTS)).astype(int),
    10, None
)
age_first_use = np.minimum(age_first_use, age_at_enrollment - 1)

social_support_baseline = np.clip(
    np.random.normal(4.5, 2.5, N_PATIENTS), 0, 10
).round(1)

# ── MH comorbidities (prevalence adjusted by SUD type + severity) ─────
mh_conditions = {
    "mh_major_depression":     0.45,
    "mh_ptsd":                 0.35,
    "mh_gad":                  0.30,
    "mh_bipolar":              0.15,
    "mh_adhd":                 0.25,
    "mh_personality_disorder": 0.12,
    "mh_psychotic_disorder":   0.06,
}

mh_data = {}
for cond, prev in mh_conditions.items():
    adj = np.where(sud_primary == "Polysubstance Use Disorder", 0.08, 0.0)
    adj += np.where(sud_severity == "Severe", 0.05, 0.0)
    mh_data[cond] = (np.random.random(N_PATIENTS) < (prev + adj)).astype(int)

# ── Medical comorbidities (updated prevalence) ────────────────────────
med_conditions = {
    "med_hiv":             0.08,
    "med_hcv":             0.22,
    "med_chronic_pain":    0.31,
    "med_tbi_history":     0.18,
    "med_liver_disease":   0.12,
    "med_cardiovascular":  0.09,
    "med_ms_autoimmune":   0.04,
}

med_data = {}
for cond, prev in med_conditions.items():
    adj = np.where(route_of_admin == "IV", 0.10, 0.0) if cond in ["med_hcv", "med_hiv"] else 0.0
    med_data[cond] = (np.random.random(N_PATIENTS) < (prev + adj)).astype(int)

# ── Latent class assignment (Phase 0 seed parameters) ─────────────────
# Base P(Remission) comes directly from seed, stratified by SUD type.
# Only adjustment: small penalty for severe MH burden (>3 diagnoses).
n_mh_per_patient = sum(mh_data[c] for c in mh_data)
latent_p_remission = np.where(
    sud_primary == "Stimulant Use Disorder",
    p_remission_stimulant,
    p_remission_poly,
)
# Subtract 0.05 per MH diagnosis above threshold of 3
mh_excess = np.clip(n_mh_per_patient - mh_excess_threshold, 0, None)
latent_p_remission = latent_p_remission - mh_excess * mh_excess_penalty_per_dx
latent_p_remission = np.clip(latent_p_remission, 0.15, 0.95)

latent_class = np.where(
    np.random.random(N_PATIENTS) < latent_p_remission, 0, 1
)  # 0 = Remission, 1 = Cycling

# ── Enrollment dates ──────────────────────────────────────────────────
enrollment_days = (ENROLLMENT_END - ENROLLMENT_START).days
enrollment_dates = [
    ENROLLMENT_START + timedelta(days=int(np.random.uniform(0, enrollment_days)))
    for _ in range(N_PATIENTS)
]

# ── Build static dataframe ────────────────────────────────────────────
static_df = pd.DataFrame({
    "patient_id": patient_ids,
    "enrollment_date": enrollment_dates,
    "sud_primary": sud_primary,
    "sud_severity_baseline": sud_severity,
    "prior_tx_episodes": prior_tx_episodes,
    "social_support_baseline": social_support_baseline,
    "age_at_enrollment": age_at_enrollment,
    "sex": sex,
    "race_ethnicity": race_ethnicity,
    "route_of_admin": route_of_admin,
    "treatment_modality": treatment_modality,
    "insurance_status": insurance,
    "age_first_use": age_first_use,
    "substances_involved": substances_involved,
    # MH diagnoses
    **mh_data,
    # Medical comorbidities
    **med_data,
    # Ground truth latent class (excluded from features downstream)
    "latent_class": latent_class,
})

# ══════════════════════════════════════════════════════════════════════
# PANEL (TIME-VARYING) DATA — hazard-driven sobriety trajectories
# ══════════════════════════════════════════════════════════════════════

rows = []

for i in range(N_PATIENTS):
    pid = patient_ids[i]
    enroll = enrollment_dates[i]
    lc = latent_class[i]
    sev = sud_severity[i]
    poly = sud_primary[i] == "Polysubstance Use Disorder"
    iv_user = route_of_admin[i] == "IV"
    ss = social_support_baseline[i]
    has_depression = mh_data["mh_major_depression"][i]
    has_ptsd = mh_data["mh_ptsd"][i]
    has_gad = mh_data["mh_gad"][i]
    tx = treatment_modality[i]

    # Base hazard from latent class
    base_h = base_hazard_remission if lc == 0 else base_hazard_cycling

    # Apply MH hazard multipliers to base hazard
    patient_hazard = base_h
    for mh_cond, multiplier in mh_hazard_multipliers.items():
        if mh_data[mh_cond][i]:
            patient_hazard *= multiplier

    # Clip patient hazard to reasonable range
    patient_hazard = min(patient_hazard, 0.95)

    # Determine number of quarters observed
    max_q = min(MAX_QUARTERS, ((STUDY_END - enroll).days) // 90)
    if sev == "Severe" and np.random.random() < 0.25:
        max_q = max(2, int(max_q * np.random.uniform(0.3, 0.7)))
    elif np.random.random() < 0.10:
        max_q = max(2, int(max_q * np.random.uniform(0.5, 0.85)))

    # Track state
    sober = False
    consecutive_sober = 0
    consecutive_using = 0
    cumulative_stress = 0.0

    for q in range(max_q):
        visit_date = enroll + timedelta(days=90 * q + np.random.randint(-7, 8))

        # ── Life events (stochastic) ──
        le_job_loss = int(np.random.random() < 0.06)
        le_housing_instability = int(np.random.random() < 0.08)
        le_incarceration = int(np.random.random() < 0.03)
        le_relationship_breakup = int(np.random.random() < 0.05)
        le_bereavement = int(np.random.random() < 0.02)
        le_new_relationship = int(np.random.random() < 0.04)
        le_got_employment = int(np.random.random() < 0.07)
        le_stable_housing = int(np.random.random() < 0.06)
        le_legal_problem = int(np.random.random() < 0.04)
        le_peer_relapse = int(np.random.random() < 0.05)

        # ── Cumulative stress (Phase 0 model) ──
        # Negative events accumulate via two pathways
        neg_event_sum = (
            le_job_loss * tv_effects["le_job_loss"]
            + le_housing_instability * tv_effects["le_housing_instability"]
            + le_incarceration * tv_effects["le_incarceration"]
            + le_relationship_breakup * tv_effects["le_relationship_breakup"]
            + le_bereavement * tv_effects["le_bereavement"]
            + le_legal_problem * tv_effects["le_legal_problem"]
            + le_peer_relapse * tv_effects["le_peer_relapse"]
        )
        pos_event_sum = (
            le_new_relationship * abs(tv_effects["le_new_relationship"])
            + le_got_employment * abs(tv_effects["le_got_employment"])
            + le_stable_housing * abs(tv_effects["le_stable_housing"])
        )

        stress_input = (
            cascade_pathway_weight * neg_event_sum
            + involuntary_pathway_weight * neg_event_sum
            - pos_event_sum
        )
        cumulative_stress = stress_decay_rate * cumulative_stress + stress_input
        cumulative_stress = np.clip(cumulative_stress, 0, 10)

        # ── Treatment adherence ──
        if tx == "None":
            tx_adherence = 0
        else:
            tx_adh_base = 70 if sober else 45
            # Treatment engagement improves slightly over time
            tx_adh_base += min(q * 2, 15)
            tx_adherence = np.clip(int(np.random.normal(tx_adh_base, 15)), 0, 100)

        # ── Meetings per month ──
        meetings_per_month = max(0, int(np.random.normal(
            8 if sober else 2, 3)))

        # ── Social support (drifts) ──
        social_support = np.clip(
            ss + (1.5 if sober else -0.8) * (q / MAX_QUARTERS)
            + np.random.normal(0, 0.5), 0, 10
        ).round(1)

        # ── Compute transition probability using Phase 0 hazard model ──
        # p_relapse = patient_hazard + time-varying effects + stress
        tv_hazard_adj = 0.0
        for le_name, le_val in [
            ("le_job_loss", le_job_loss),
            ("le_housing_instability", le_housing_instability),
            ("le_incarceration", le_incarceration),
            ("le_relationship_breakup", le_relationship_breakup),
            ("le_bereavement", le_bereavement),
            ("le_legal_problem", le_legal_problem),
            ("le_peer_relapse", le_peer_relapse),
            ("le_new_relationship", le_new_relationship),
            ("le_got_employment", le_got_employment),
            ("le_stable_housing", le_stable_housing),
        ]:
            tv_hazard_adj += le_val * tv_effects[le_name]

        # Protective effects from treatment engagement
        if tx_adherence > 70:
            tv_hazard_adj += tv_effects["tx_adherence_high"]
        if meetings_per_month >= 4:
            tv_hazard_adj += tv_effects["meetings_regular"]
        if social_support > 6.0:
            tv_hazard_adj += tv_effects["social_support_high"]

        # Momentum: sobriety builds stability
        if sober:
            momentum = -min(consecutive_sober * 0.04, 0.20)
        else:
            momentum = min(consecutive_using * 0.03, 0.12)

        # Treatment modality protective effect
        tx_protect = 0.0
        if tx in ["Residential", "MAT"]:
            tx_protect = -0.12
        elif tx in ["IOP", "Outpatient"]:
            tx_protect = -min(q * 0.015, 0.10)

        # Final relapse probability
        p_relapse = patient_hazard + tv_hazard_adj + momentum + tx_protect
        p_relapse += cumulative_stress * 0.03  # stress amplifies risk
        p_relapse = np.clip(p_relapse, 0.03, 0.95)

        # Sobriety = NOT relapsing
        sober = np.random.random() > p_relapse

        if sober:
            consecutive_sober += 1
            consecutive_using = 0
        else:
            consecutive_using += 1
            consecutive_sober = 0

        # ── PHQ-9 depression score (0-27) ──
        phq9_base = 8 if has_depression else 4
        phq9 = np.clip(int(np.random.normal(
            phq9_base - 3 if sober else phq9_base + 5, 3)), 0, 27)

        # ── GAD-7 anxiety score (0-21) ──
        gad7_base = 7 if has_gad else 3
        gad7 = np.clip(int(np.random.normal(
            gad7_base - 2 if sober else gad7_base + 4, 3)), 0, 21)

        # ── UDS result ──
        uds_positive = 0 if sober else 1
        if sober and np.random.random() < 0.03:
            uds_positive = 1
        if not sober and np.random.random() < 0.05:
            uds_positive = 0

        rows.append({
            "patient_id": pid,
            "quarter": q + 1,
            "visit_date": visit_date.strftime("%Y-%m-%d"),
            "phq9_score": phq9,
            "gad7_score": gad7,
            "tx_adherence_pct": tx_adherence,
            "sober_this_quarter": int(sober),
            "uds_positive": uds_positive,
            "meetings_per_month": meetings_per_month,
            "social_support_score": social_support,
            "le_job_loss": le_job_loss,
            "le_housing_instability": le_housing_instability,
            "le_incarceration": le_incarceration,
            "le_relationship_breakup": le_relationship_breakup,
            "le_bereavement": le_bereavement,
            "le_legal_problem": le_legal_problem,
            "le_peer_relapse": le_peer_relapse,
            "le_new_relationship": le_new_relationship,
            "le_got_employment": le_got_employment,
            "le_stable_housing": le_stable_housing,
        })

panel_df = pd.DataFrame(rows)

# ── Save ──────────────────────────────────────────────────────────────
static_df.to_csv(config.RAW_DIR / "patient_static.csv", index=False)
panel_df.to_csv(config.RAW_DIR / "patient_panel.csv", index=False)

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("SYNTHETIC COHORT GENERATION COMPLETE (Phase 0 Integration)")
print("=" * 70)

print(f"\n── Cohort size ──")
print(f"  Patients: {N_PATIENTS}")
print(f"  Panel observations: {len(panel_df)}")
print(f"  Quarters/patient: {panel_df.groupby('patient_id')['quarter'].count().describe().round(1).to_string()}")

print(f"\n── Latent class split ──")
lc_counts = static_df["latent_class"].value_counts().sort_index()
print(f"  Remission (0): {lc_counts.get(0, 0)} ({lc_counts.get(0, 0)/N_PATIENTS:.1%})")
print(f"  Cycling   (1): {lc_counts.get(1, 0)} ({lc_counts.get(1, 0)/N_PATIENTS:.1%})")

print(f"\n── Demographics ──")
for col in ["sex", "race_ethnicity", "route_of_admin", "treatment_modality", "insurance_status"]:
    print(f"\n  {col}:")
    dist = static_df[col].value_counts(normalize=True).round(3)
    for val, pct in dist.items():
        print(f"    {val}: {pct:.1%}")

print(f"\n  age_at_enrollment: mean={static_df['age_at_enrollment'].mean():.1f}, "
      f"std={static_df['age_at_enrollment'].std():.1f}, "
      f"range=[{static_df['age_at_enrollment'].min()}, {static_df['age_at_enrollment'].max()}]")

print(f"\n── SUD breakdown ──")
print(static_df["sud_primary"].value_counts().to_string())

print(f"\n── MH comorbidity prevalence ──")
mh_cols = [c for c in static_df.columns if c.startswith("mh_")]
for c in mh_cols:
    print(f"  {c}: {static_df[c].mean():.1%}")

print(f"\n── Medical comorbidity prevalence ──")
med_cols = [c for c in static_df.columns if c.startswith("med_")]
for c in med_cols:
    print(f"  {c}: {static_df[c].mean():.1%}")

print(f"\n── MH co-occurrence rates (top pairs) ──")
mh_matrix = static_df[mh_cols]
pairs = []
for j, c1 in enumerate(mh_cols):
    for c2 in mh_cols[j+1:]:
        rate = ((mh_matrix[c1] == 1) & (mh_matrix[c2] == 1)).mean()
        pairs.append((c1, c2, rate))
pairs.sort(key=lambda x: -x[2])
for c1, c2, rate in pairs[:8]:
    c1_short = c1.replace("mh_", "")
    c2_short = c2.replace("mh_", "")
    print(f"  {c1_short} + {c2_short}: {rate:.1%}")

print(f"\n── Sobriety rate by latent class ──")
merged = panel_df.merge(static_df[["patient_id", "latent_class"]], on="patient_id")
by_lc = merged.groupby("latent_class")["sober_this_quarter"].mean()
print(f"  Remission: {by_lc.get(0, 0):.1%}")
print(f"  Cycling:   {by_lc.get(1, 0):.1%}")

print(f"\n── Saved to ──")
print(f"  {config.RAW_DIR / 'patient_static.csv'}")
print(f"  {config.RAW_DIR / 'patient_panel.csv'}")
