"""
Synthetic Longitudinal Cohort Generator
========================================
Sobriety Prediction Model for Stimulant & Polysubstance Use Disorder
Nyx Dynamics LLC | MIT MicroMasters Capstone

Generates a realistic panel dataset of ~2,000 patients with quarterly
observations over 3 years (up to 12 time points per patient), informed
by clinical feature space from real EHR records.

Features are modeled after:
- MyChart longitudinal visit records
- Standard lab panels (metabolic, hepatic, hematologic, infectious)
- MH comorbidity profiles common in SUD populations
- Life event indicators from social determinants of health literature
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────
N_PATIENTS = 2000
MAX_QUARTERS = 12          # 3 years of quarterly observations
ENROLLMENT_START = datetime(2021, 1, 1)
ENROLLMENT_END = datetime(2023, 6, 30)
STUDY_END = datetime(2024, 12, 31)

# ── Patient-level (static) features ───────────────────────────────────
patient_ids = [f"SYN-{i:04d}" for i in range(1, N_PATIENTS + 1)]

age_at_enrollment = np.clip(
    np.random.normal(34, 9, N_PATIENTS).astype(int), 18, 65
)

sex = np.random.choice(
    ["Male", "Female", "Non-binary"],
    N_PATIENTS, p=[0.58, 0.38, 0.04]
)

race_ethnicity = np.random.choice(
    ["White", "Black", "Hispanic", "Asian", "Other/Multiracial"],
    N_PATIENTS, p=[0.42, 0.28, 0.20, 0.05, 0.05]
)

# Primary SUD diagnosis
sud_primary = np.random.choice(
    ["Stimulant Use Disorder", "Polysubstance Use Disorder"],
    N_PATIENTS, p=[0.45, 0.55]
)

# Specific substances involved (polysubstance gets 2-4)
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

# SUD severity at enrollment (DSM-5: mild 2-3, moderate 4-5, severe 6+)
sud_severity = np.random.choice(
    ["Mild", "Moderate", "Severe"],
    N_PATIENTS, p=[0.15, 0.35, 0.50]
)

# Route of administration (affects risk profile)
route_of_admin = np.random.choice(
    ["IV", "Smoking", "Intranasal", "Oral", "Mixed"],
    N_PATIENTS, p=[0.20, 0.30, 0.25, 0.10, 0.15]
)

# Treatment modality at enrollment
treatment_modality = np.random.choice(
    ["IOP", "Residential", "MAT", "Outpatient", "Detox_only", "None"],
    N_PATIENTS, p=[0.20, 0.15, 0.15, 0.25, 0.10, 0.15]
)

# ── MH comorbidities (binary, can have multiple) ─────────────────────
mh_conditions = {
    "mh_major_depression": 0.45,
    "mh_ptsd": 0.35,
    "mh_gad": 0.30,
    "mh_bipolar": 0.15,
    "mh_adhd": 0.25,
    "mh_personality_disorder": 0.12,
    "mh_psychotic_disorder": 0.06,
}

mh_data = {}
for cond, prev in mh_conditions.items():
    # Higher prevalence in polysubstance and severe
    adj = np.where(sud_primary == "Polysubstance Use Disorder", 0.08, 0.0)
    adj += np.where(sud_severity == "Severe", 0.05, 0.0)
    mh_data[cond] = (np.random.random(N_PATIENTS) < (prev + adj)).astype(int)

# Medical comorbidities (informed by real EHR feature space)
med_conditions = {
    "med_hcv": 0.18,             # Hepatitis C
    "med_hiv": 0.08,             # HIV
    "med_chronic_pain": 0.30,
    "med_tbi_history": 0.12,     # Traumatic brain injury
    "med_liver_disease": 0.10,
    "med_cardiovascular": 0.09,
    "med_ms_autoimmune": 0.03,   # MS or autoimmune conditions
}

med_data = {}
for cond, prev in med_conditions.items():
    adj = np.where(route_of_admin == "IV", 0.10, 0.0) if cond in ["med_hcv", "med_hiv"] else 0.0
    med_data[cond] = (np.random.random(N_PATIENTS) < (prev + adj)).astype(int)

# Insurance status
insurance = np.random.choice(
    ["Medicaid", "Private", "Medicare", "Uninsured", "VA"],
    N_PATIENTS, p=[0.40, 0.20, 0.10, 0.20, 0.10]
)

# Prior treatment episodes
prior_tx_episodes = np.random.poisson(1.5, N_PATIENTS)

# Age of first substance use
age_first_use = np.clip(
    (age_at_enrollment - np.random.exponential(12, N_PATIENTS)).astype(int),
    10, None
)
age_first_use = np.minimum(age_first_use, age_at_enrollment - 1)

# Social support score (0-10)
social_support_baseline = np.clip(
    np.random.normal(4.5, 2.5, N_PATIENTS), 0, 10
).round(1)

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
    "age_at_enrollment": age_at_enrollment,
    "sex": sex,
    "race_ethnicity": race_ethnicity,
    "sud_primary": sud_primary,
    "substances_involved": substances_involved,
    "sud_severity_baseline": sud_severity,
    "route_of_admin": route_of_admin,
    "treatment_modality": treatment_modality,
    "insurance_status": insurance,
    "prior_tx_episodes": prior_tx_episodes,
    "age_first_use": age_first_use,
    "social_support_baseline": social_support_baseline,
    **mh_data,
    **med_data,
})

# ── Panel (time-varying) data ─────────────────────────────────────────
rows = []

for i in range(N_PATIENTS):
    pid = patient_ids[i]
    enroll = enrollment_dates[i]
    sev = sud_severity[i]
    poly = sud_primary[i] == "Polysubstance Use Disorder"
    iv_user = route_of_admin[i] == "IV"
    ss = social_support_baseline[i]
    has_depression = mh_data["mh_major_depression"][i]
    has_ptsd = mh_data["mh_ptsd"][i]
    tx = treatment_modality[i]
    n_prior = prior_tx_episodes[i]

    # Base sobriety probability (higher = more likely sober at each visit)
    base_p = 0.45
    if sev == "Severe": base_p -= 0.15
    elif sev == "Mild": base_p += 0.10
    if poly: base_p -= 0.08
    if iv_user: base_p -= 0.07
    if tx in ["Residential", "MAT"]: base_p += 0.12
    elif tx == "None": base_p -= 0.10
    if has_depression: base_p -= 0.06
    if has_ptsd: base_p -= 0.04
    base_p += ss * 0.015
    if n_prior > 3: base_p -= 0.05

    # Determine number of quarters observed (some patients drop out)
    max_q = min(MAX_QUARTERS, ((STUDY_END - enroll).days) // 90)
    if sev == "Severe" and np.random.random() < 0.25:
        max_q = max(2, int(max_q * np.random.uniform(0.3, 0.7)))
    elif np.random.random() < 0.10:
        max_q = max(2, int(max_q * np.random.uniform(0.5, 0.85)))

    # Track sobriety state (Markov-ish with momentum)
    sober = False  # start in active use (just enrolled)
    consecutive_sober = 0
    consecutive_using = 0

    for q in range(max_q):
        visit_date = enroll + timedelta(days=90 * q + np.random.randint(-7, 8))

        # ── Time-varying probability adjustments ──
        p_sober = base_p

        # Momentum: consecutive sobriety increases stability
        if sober:
            p_sober += min(consecutive_sober * 0.04, 0.20)
        else:
            p_sober -= min(consecutive_using * 0.03, 0.12)

        # Treatment engagement improves over time
        if tx in ["IOP", "Residential", "MAT", "Outpatient"]:
            p_sober += min(q * 0.015, 0.10)

        # ── Life events (stochastic, time-varying) ──
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

        # Negative events decrease sobriety probability
        p_sober -= le_job_loss * 0.12
        p_sober -= le_housing_instability * 0.10
        p_sober -= le_incarceration * 0.08
        p_sober -= le_relationship_breakup * 0.07
        p_sober -= le_bereavement * 0.09
        p_sober -= le_legal_problem * 0.06
        p_sober -= le_peer_relapse * 0.10

        # Positive events increase sobriety probability
        p_sober += le_new_relationship * 0.05
        p_sober += le_got_employment * 0.08
        p_sober += le_stable_housing * 0.07

        # Clip probability
        p_sober = np.clip(p_sober, 0.03, 0.95)

        # Determine sobriety state this quarter
        sober = np.random.random() < p_sober

        if sober:
            consecutive_sober += 1
            consecutive_using = 0
        else:
            consecutive_using += 1
            consecutive_sober = 0

        # ── Lab values (informed by real EHR panels) ──
        # These vary based on sobriety and comorbidities
        ast = np.clip(np.random.normal(
            28 if sober else 45, 12 if sober else 25), 10, 300).round(0)
        alt = np.clip(np.random.normal(
            25 if sober else 42, 10 if sober else 22), 8, 280).round(0)
        ggt = np.clip(np.random.normal(
            30 if sober else 65, 15 if sober else 35), 5, 400).round(0)
        creatinine = np.clip(np.random.normal(
            0.95 if sober else 1.1, 0.15, ), 0.4, 3.5).round(2)

        # Hematologic
        wbc = np.clip(np.random.normal(
            7.0 if sober else 8.5, 2.0), 2.0, 20.0).round(1)
        hemoglobin = np.clip(np.random.normal(
            13.5 if sex[i] == "Female" else 14.8,
            1.2 if sober else 1.8), 7, 18).round(1)
        platelets = np.clip(np.random.normal(
            250 if sober else 210, 50), 50, 500).round(0)

        # Metabolic
        glucose_fasting = np.clip(np.random.normal(
            92 if sober else 105, 15), 55, 300).round(0)
        total_cholesterol = np.clip(np.random.normal(
            190, 35), 100, 350).round(0)
        triglycerides = np.clip(np.random.normal(
            130 if sober else 175, 50), 40, 500).round(0)

        # Infectious markers (for IV users / comorbid)
        if med_data["med_hiv"][i]:
            cd4_count = np.clip(np.random.normal(
                550 if sober else 380, 150), 50, 1500).round(0)
            viral_load_log = np.clip(np.random.normal(
                1.3 if sober else 3.2, 1.0), 0, 6).round(1)
        else:
            cd4_count = np.nan
            viral_load_log = np.nan

        # PHQ-9 depression score (0-27)
        phq9_base = 8 if has_depression else 4
        phq9 = np.clip(int(np.random.normal(
            phq9_base - 3 if sober else phq9_base + 5, 3)), 0, 27)

        # GAD-7 anxiety score (0-21)
        gad7_base = 7 if mh_data["mh_gad"][i] else 3
        gad7 = np.clip(int(np.random.normal(
            gad7_base - 2 if sober else gad7_base + 4, 3)), 0, 21)

        # PCL-5 PTSD score (0-80)
        pcl5_base = 38 if has_ptsd else 10
        pcl5 = np.clip(int(np.random.normal(
            pcl5_base - 8 if sober else pcl5_base + 10, 8)), 0, 80)

        # Social support (drifts from baseline)
        social_support = np.clip(
            ss + (1.5 if sober else -0.8) * (q / MAX_QUARTERS)
            + np.random.normal(0, 0.5), 0, 10
        ).round(1)

        # Treatment adherence (0-100%)
        if tx == "None":
            tx_adherence = 0
        else:
            tx_adh_base = 70 if sober else 45
            tx_adherence = np.clip(int(np.random.normal(tx_adh_base, 15)), 0, 100)

        # 12-step / peer support meetings per month
        meetings_per_month = max(0, int(np.random.normal(
            8 if sober else 2, 3)))

        # UDS result (urine drug screen — ground truth)
        uds_positive = 0 if sober else 1
        # Some noise: false negatives / delayed detection
        if sober and np.random.random() < 0.03:
            uds_positive = 1  # residual positive
        if not sober and np.random.random() < 0.05:
            uds_positive = 0  # test timing miss

        # Days since last use (self-report)
        if sober:
            days_since_last_use = int(consecutive_sober * 90
                                      + np.random.uniform(0, 30))
        else:
            days_since_last_use = max(0, int(np.random.exponential(5)))

        rows.append({
            "patient_id": pid,
            "quarter": q + 1,
            "visit_date": visit_date.strftime("%Y-%m-%d"),
            # Sobriety outcome
            "sober_this_quarter": int(sober),
            "uds_positive": uds_positive,
            "days_since_last_use": days_since_last_use,
            "consecutive_sober_quarters": consecutive_sober if sober else 0,
            # MH symptom scores
            "phq9_score": phq9,
            "gad7_score": gad7,
            "pcl5_score": pcl5,
            # Lab values
            "ast": ast,
            "alt": alt,
            "ggt": ggt,
            "creatinine": creatinine,
            "wbc": wbc,
            "hemoglobin": hemoglobin,
            "platelets": platelets,
            "glucose_fasting": glucose_fasting,
            "total_cholesterol": total_cholesterol,
            "triglycerides": triglycerides,
            "cd4_count": cd4_count,
            "viral_load_log10": viral_load_log,
            # Life events
            "le_job_loss": le_job_loss,
            "le_housing_instability": le_housing_instability,
            "le_incarceration": le_incarceration,
            "le_relationship_breakup": le_relationship_breakup,
            "le_bereavement": le_bereavement,
            "le_new_relationship": le_new_relationship,
            "le_got_employment": le_got_employment,
            "le_stable_housing": le_stable_housing,
            "le_legal_problem": le_legal_problem,
            "le_peer_relapse": le_peer_relapse,
            # Treatment & support
            "tx_adherence_pct": tx_adherence,
            "meetings_per_month": meetings_per_month,
            "social_support_score": social_support,
        })

panel_df = pd.DataFrame(rows)

# ── Outcome summary (one row per patient) ─────────────────────────────
outcome_rows = []
for pid in patient_ids:
    pt = panel_df[panel_df["patient_id"] == pid]
    n_obs = len(pt)
    n_sober = pt["sober_this_quarter"].sum()
    ever_sober = int(n_sober > 0)
    sustained_6mo = int((pt["consecutive_sober_quarters"] >= 2).any())
    sustained_12mo = int((pt["consecutive_sober_quarters"] >= 4).any())
    last_sober = pt.iloc[-1]["sober_this_quarter"]
    max_consec = pt["consecutive_sober_quarters"].max()
    pct_sober = round(n_sober / n_obs, 3)

    outcome_rows.append({
        "patient_id": pid,
        "n_observations": n_obs,
        "n_quarters_sober": n_sober,
        "pct_quarters_sober": pct_sober,
        "ever_achieved_sobriety": ever_sober,
        "sustained_sobriety_6mo": sustained_6mo,
        "sustained_sobriety_12mo": sustained_12mo,
        "sober_at_last_observation": last_sober,
        "max_consecutive_sober_quarters": max_consec,
    })

outcome_df = pd.DataFrame(outcome_rows)

# ── Save all datasets ─────────────────────────────────────────────────
static_df.to_csv(config.STATIC_DIR / "patient_static.csv", index=False)
panel_df.to_csv(config.PANEL_DIR / "patient_panel.csv", index=False)
outcome_df.to_csv(config.OUTCOMES_DIR / "patient_outcomes.csv", index=False)

# Also save the combined (merged) dataset
combined_df = panel_df.merge(static_df, on="patient_id", how="left")
combined_df.to_csv(config.RAW_DIR / "sobriety_cohort_full.csv", index=False)

print("=" * 70)
print("SYNTHETIC COHORT GENERATION COMPLETE")
print("=" * 70)

print(f"\n── Static (patient-level) ──")
print(f"Shape: {static_df.shape}")
print(f"Columns: {list(static_df.columns)}")

print(f"\n── Panel (time-varying) ──")
print(f"Shape: {panel_df.shape}")
print(f"Columns: {list(panel_df.columns)}")

print(f"\n── Outcomes (per-patient summary) ──")
print(f"Shape: {outcome_df.shape}")
print(f"Columns: {list(outcome_df.columns)}")

print(f"\n── Combined (merged) ──")
print(f"Shape: {combined_df.shape}")

print(f"\n── Dtypes (combined) ──")
print(combined_df.dtypes)

print(f"\n── Null counts ──")
nulls = combined_df.isnull().sum()
print(nulls[nulls > 0])

print(f"\n── First 3 rows (combined, transposed for readability) ──")
print(combined_df.head(3).T)

print(f"\n── Outcome distribution ──")
print(outcome_df.describe().round(2))

print(f"\n── SUD primary breakdown ──")
print(static_df["sud_primary"].value_counts())

print(f"\n── Sobriety rate by SUD type ──")
merged = outcome_df.merge(static_df[["patient_id", "sud_primary"]], on="patient_id")
print(merged.groupby("sud_primary")["pct_quarters_sober"].mean().round(3))
