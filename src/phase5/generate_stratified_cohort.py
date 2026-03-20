"""
Phase 5: Stratified Cohort Generator
======================================
Generates N=4000 patients across 4 resource strata to enable
structural equity audit via pathway attribution.

Nyx Dynamics LLC | MIT MicroMasters Capstone
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "synthetic"))
import config

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from phase0_seed import (
    base_hazard_remission, base_hazard_cycling,
    mh_hazard_multipliers, tv_effects,
    stress_decay_rate, cascade_pathway_weight, involuntary_pathway_weight,
    p_remission_stimulant, p_remission_poly,
    mh_excess_penalty_per_dx, mh_excess_threshold,
)

np.random.seed(config.SEED)

N_PER_STRATUM = 1000
N_TOTAL = 4000
MAX_QUARTERS = 12
ENROLLMENT_START = datetime(2021, 1, 1)
ENROLLMENT_END = datetime(2023, 6, 30)
STUDY_END = datetime(2024, 12, 31)

# ══════════════════════════════════════════════════════════════════════
# TRUE pathway log-hazard multipliers (model must recover these)
# Each unit DECREASE in the resource feature adds this to log-hazard
# ══════════════════════════════════════════════════════════════════════
TRUE_GAMMA = {
    "treatment_intensity": 0.15,
    "monitoring_intensity": 0.12,
    "social_support_density": 0.08,
    "self_advocacy_index": 0.10,
}

# ══════════════════════════════════════════════════════════════════════
# Stratum definitions
# ══════════════════════════════════════════════════════════════════════

STRATA = {
    "A": {
        "label": "High resource",
        "treatment_modality": "Residential",
        "treatment_intensity": (4, 5),       # residential_short to long
        "monitoring_intensity": (3, 4),       # weekly_uds to physician_program
        "social_support_range": (7.5, 10.0),
        "self_advocacy_range": (8.0, 10.0),
        "p_remission_boost": 0.05,           # slight boost on top of seed
    },
    "B": {
        "label": "Moderate resource",
        "treatment_modality": "IOP",
        "treatment_intensity": (3, 3),        # IOP
        "monitoring_intensity": (2, 2),        # monthly
        "social_support_range": (5.0, 7.5),
        "self_advocacy_range": (5.0, 8.0),
        "p_remission_boost": -0.15,
    },
    "C": {
        "label": "Low resource",
        "treatment_modality": "Outpatient",
        "treatment_intensity": (2, 2),         # outpatient
        "monitoring_intensity": (1, 1),         # quarterly
        "social_support_range": (2.5, 5.0),
        "self_advocacy_range": (2.0, 5.0),
        "p_remission_boost": -0.35,
    },
    "D": {
        "label": "Minimal resource",
        "treatment_modality": "None",
        "treatment_intensity": (0, 1),          # none to detox
        "monitoring_intensity": (0, 0),          # none
        "social_support_range": (0.0, 2.5),
        "self_advocacy_range": (0.0, 2.0),
        "p_remission_boost": -0.55,
    },
}

# ══════════════════════════════════════════════════════════════════════
# Generate patients
# ══════════════════════════════════════════════════════════════════════

substance_categories = [
    "Methamphetamine", "Cocaine/Crack", "Opioids", "Alcohol",
    "Benzodiazepines", "Cannabis", "MDMA/Club drugs"
]

mh_conditions = {
    "mh_major_depression": 0.45, "mh_ptsd": 0.35, "mh_gad": 0.30,
    "mh_bipolar": 0.15, "mh_adhd": 0.25,
    "mh_personality_disorder": 0.12, "mh_psychotic_disorder": 0.06,
}

static_rows = []
panel_rows = []
pid_counter = 0

for stratum_key, sdef in STRATA.items():
    for _ in range(N_PER_STRATUM):
        pid_counter += 1
        pid = f"EQ-{pid_counter:04d}"

        # Demographics
        age = np.clip(int(np.random.normal(34, 9)), 18, 65)
        sex = np.random.choice(["Male", "Female", "Non-binary"], p=[0.52, 0.38, 0.10])
        race = np.random.choice(
            ["White", "Black", "Hispanic", "Asian", "Other/Multiracial"],
            p=[0.48, 0.22, 0.18, 0.06, 0.06])
        sud = np.random.choice(
            ["Stimulant Use Disorder", "Polysubstance Use Disorder"], p=[0.45, 0.55])
        severity = np.random.choice(["Mild", "Moderate", "Severe"], p=[0.15, 0.35, 0.50])
        route = np.random.choice(
            ["Smoking", "Intranasal", "IV", "Oral", "Mixed"], p=[0.35, 0.28, 0.18, 0.12, 0.07])
        insurance = np.random.choice(
            ["Medicaid", "Uninsured", "Private", "Medicare", "VA"], p=[0.42, 0.28, 0.22, 0.05, 0.03])
        prior_tx = np.random.poisson(1.5)
        age_first = np.clip(int(age - np.random.exponential(12)), 10, age - 1)

        # Resource pathway features (stratum-specific)
        tx_lo, tx_hi = sdef["treatment_intensity"]
        tx_intensity = np.random.randint(tx_lo, tx_hi + 1)
        mon_lo, mon_hi = sdef["monitoring_intensity"]
        mon_intensity = np.random.randint(mon_lo, mon_hi + 1)
        ss_lo, ss_hi = sdef["social_support_range"]
        social_support = round(np.random.uniform(ss_lo, ss_hi), 1)
        adv_lo, adv_hi = sdef["self_advocacy_range"]
        self_advocacy = round(np.random.uniform(adv_lo, adv_hi), 1)

        # Composite social support density
        network_stability = np.clip(social_support / 10 + np.random.normal(0, 0.1), 0, 1)
        social_density = round((social_support + network_stability * 10) / 2, 1)

        # MH comorbidities
        mh_data = {}
        for cond, prev in mh_conditions.items():
            adj = 0.08 if sud == "Polysubstance Use Disorder" else 0.0
            adj += 0.05 if severity == "Severe" else 0.0
            mh_data[cond] = int(np.random.random() < (prev + adj))

        # Latent class assignment (seed-based + stratum boost)
        base_p = p_remission_stimulant if sud == "Stimulant Use Disorder" else p_remission_poly
        n_mh = sum(mh_data.values())
        mh_excess = max(0, n_mh - mh_excess_threshold)
        p_rem = base_p + sdef["p_remission_boost"] - mh_excess * mh_excess_penalty_per_dx
        p_rem = np.clip(p_rem, 0.05, 0.95)
        latent_class = 0 if np.random.random() < p_rem else 1

        # Resource-adjusted hazard
        base_h = base_hazard_remission if latent_class == 0 else base_hazard_cycling
        patient_h = base_h
        for mh_cond, mult in mh_hazard_multipliers.items():
            if mh_data.get(mh_cond, 0):
                patient_h *= mult

        # Apply TRUE pathway effects (resource deprivation increases hazard)
        # Reference = max resource level (Stratum A top)
        resource_penalty = (
            TRUE_GAMMA["treatment_intensity"] * (5 - tx_intensity)
            + TRUE_GAMMA["monitoring_intensity"] * (4 - mon_intensity)
            + TRUE_GAMMA["social_support_density"] * (10 - social_density) / 2
            + TRUE_GAMMA["self_advocacy_index"] * (10 - self_advocacy) / 2
        )
        patient_h *= np.exp(resource_penalty)
        patient_h = min(patient_h, 0.95)

        # Enrollment
        enroll = ENROLLMENT_START + timedelta(
            days=int(np.random.uniform(0, (ENROLLMENT_END - ENROLLMENT_START).days)))

        # Number of quarters
        max_q = min(MAX_QUARTERS, ((STUDY_END - enroll).days) // 90)
        if severity == "Severe" and np.random.random() < 0.25:
            max_q = max(2, int(max_q * np.random.uniform(0.3, 0.7)))
        elif np.random.random() < 0.10:
            max_q = max(2, int(max_q * np.random.uniform(0.5, 0.85)))

        static_rows.append({
            "patient_id": pid,
            "stratum": stratum_key,
            "enrollment_date": enroll.strftime("%Y-%m-%d"),
            "sud_primary": sud,
            "sud_severity_baseline": severity,
            "prior_tx_episodes": prior_tx,
            "age_at_enrollment": age,
            "sex": sex,
            "race_ethnicity": race,
            "route_of_admin": route,
            "treatment_modality": sdef["treatment_modality"],
            "insurance_status": insurance,
            "age_first_use": age_first,
            # Resource pathway features
            "treatment_intensity": tx_intensity,
            "monitoring_intensity": mon_intensity,
            "social_support_density": social_density,
            "self_advocacy_index": self_advocacy,
            # MH
            **mh_data,
            "latent_class": latent_class,
        })

        # Panel generation
        sober = False
        consec_sober = 0
        consec_using = 0

        for q in range(max_q):
            visit_date = enroll + timedelta(days=90 * q + np.random.randint(-7, 8))

            # Life events
            le = {
                "le_job_loss": int(np.random.random() < 0.06),
                "le_housing_instability": int(np.random.random() < 0.08),
                "le_incarceration": int(np.random.random() < 0.03),
                "le_relationship_breakup": int(np.random.random() < 0.05),
                "le_bereavement": int(np.random.random() < 0.02),
                "le_legal_problem": int(np.random.random() < 0.04),
                "le_peer_relapse": int(np.random.random() < 0.05),
                "le_new_relationship": int(np.random.random() < 0.04),
                "le_got_employment": int(np.random.random() < 0.07),
                "le_stable_housing": int(np.random.random() < 0.06),
            }

            # TV hazard adjustments
            tv_adj = sum(le[k] * tv_effects[k] for k in le if k in tv_effects)

            # Protective time-varying
            tx_adh = np.clip(int(np.random.normal(
                70 if sober else 45, 15)) + min(q * 2, 15), 0, 100) if sdef["treatment_modality"] != "None" else 0
            meetings = max(0, int(np.random.normal(8 if sober else 2, 3)))
            ss_t = np.clip(social_support + (1.5 if sober else -0.8) * (q / MAX_QUARTERS)
                           + np.random.normal(0, 0.5), 0, 10)

            if tx_adh > 70:
                tv_adj += tv_effects["tx_adherence_high"]
            if meetings >= 4:
                tv_adj += tv_effects["meetings_regular"]
            if ss_t > 6.0:
                tv_adj += tv_effects["social_support_high"]

            momentum = -min(consec_sober * 0.04, 0.20) if sober else min(consec_using * 0.03, 0.12)

            p_relapse = patient_h + tv_adj + momentum
            p_relapse = np.clip(p_relapse, 0.03, 0.95)
            sober = np.random.random() > p_relapse

            if sober:
                consec_sober += 1
                consec_using = 0
            else:
                consec_using += 1
                consec_sober = 0

            phq9 = np.clip(int(np.random.normal(
                5 if sober else 13, 3) if mh_data["mh_major_depression"] else
                np.random.normal(2 if sober else 7, 3)), 0, 27)
            gad7 = np.clip(int(np.random.normal(
                5 if sober else 11, 3) if mh_data["mh_gad"] else
                np.random.normal(1 if sober else 5, 3)), 0, 21)
            uds = 0 if sober else 1
            if sober and np.random.random() < 0.03:
                uds = 1
            if not sober and np.random.random() < 0.05:
                uds = 0

            panel_rows.append({
                "patient_id": pid,
                "quarter": q + 1,
                "visit_date": visit_date.strftime("%Y-%m-%d"),
                "phq9_score": phq9,
                "gad7_score": gad7,
                "tx_adherence_pct": tx_adh,
                "sober_this_quarter": int(sober),
                "uds_positive": uds,
                "meetings_per_month": meetings,
                "social_support_score": round(ss_t, 1),
                **le,
            })

static_df = pd.DataFrame(static_rows)
panel_df = pd.DataFrame(panel_rows)

# ── Save ──────────────────────────────────────────────────────────────
out_dir = config.PROJECT_ROOT / "data" / "phase5" / "stratified"
out_dir.mkdir(parents=True, exist_ok=True)
static_df.to_csv(out_dir / "patient_static_stratified.csv", index=False)
panel_df.to_csv(out_dir / "patient_panel_stratified.csv", index=False)

# ── Summary ───────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 5: STRATIFIED COHORT GENERATED")
print("=" * 60)
print(f"  Total patients: {N_TOTAL}")
print(f"  Panel observations: {len(panel_df)}")

# Compute relapse rates per stratum
for sk in ["A", "B", "C", "D"]:
    st_pids = static_df[static_df["stratum"] == sk]["patient_id"]
    st_panel = panel_df[panel_df["patient_id"].isin(st_pids)]
    n_pts = len(st_pids)
    # Relapse = ever had uds_positive after a sober quarter
    relapse_pids = set()
    for pid in st_pids:
        pt = st_panel[st_panel["patient_id"] == pid].sort_values("quarter")
        ever_sober = False
        for _, row in pt.iterrows():
            if row["sober_this_quarter"] == 1:
                ever_sober = True
            elif ever_sober and row["uds_positive"] == 1:
                relapse_pids.add(pid)
                break
        if not ever_sober:
            # Never sober = treatment failure
            if (pt["uds_positive"] == 1).any():
                relapse_pids.add(pid)

    relapse_rate = len(relapse_pids) / n_pts
    mean_sober = st_panel.groupby("patient_id")["sober_this_quarter"].mean().mean()
    print(f"\n  Stratum {sk} ({STRATA[sk]['label']}):")
    print(f"    N={n_pts}, Relapse rate={relapse_rate:.1%}, Mean sobriety={mean_sober:.1%}")

print(f"\n  Saved to: {out_dir}")
