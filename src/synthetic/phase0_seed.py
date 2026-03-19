"""
Phase 0 Seed Parameters
========================
Calibrated hazard parameters for the two-class latent survival model.
These drive the synthetic data generator to produce clinically realistic
relapse trajectories with distinct Remission vs Cycling subpopulations.

Nyx Dynamics LLC | MIT MicroMasters Capstone
"""

# ── Latent class base hazards (per-quarter transition probabilities) ──
# Remission class: lower baseline hazard of relapse
base_hazard_remission = 0.06
# Cycling class: higher baseline hazard — frequent relapse/recovery cycles
base_hazard_cycling = 0.28

# ── MH comorbidity hazard multipliers ─────────────────────────────────
# Each multiplies the base hazard when the condition is present
mh_hazard_multipliers = {
    "mh_major_depression": 1.45,
    "mh_ptsd":             1.55,
    "mh_gad":              1.25,
    "mh_bipolar":          1.60,
    "mh_adhd":             1.20,
    "mh_personality_disorder": 1.70,
    "mh_psychotic_disorder":   1.80,
}

# ── Time-varying covariate effects on hazard ──────────────────────────
# Positive values increase relapse hazard; negative values are protective
tv_effects = {
    "le_job_loss":              0.18,
    "le_housing_instability":   0.15,
    "le_incarceration":         0.12,
    "le_relationship_breakup":  0.10,
    "le_bereavement":           0.14,
    "le_legal_problem":         0.08,
    "le_peer_relapse":          0.16,
    "le_new_relationship":     -0.06,
    "le_got_employment":       -0.10,
    "le_stable_housing":       -0.09,
    "tx_adherence_high":       -0.20,   # adherence > 70%
    "meetings_regular":        -0.12,   # meetings >= 4/month
    "social_support_high":     -0.08,   # social support > 6.0
}

# ── Cumulative stress model parameters ────────────────────────────────
stress_decay_rate = 0.80       # exponential decay per quarter
# Weights for stress accumulation pathways
cascade_pathway_weight = 0.65  # stress → MH worsening → relapse
involuntary_pathway_weight = 0.35  # external shock → direct relapse

# ── Latent class assignment probabilities ─────────────────────────────
# Base probability of being in the Remission class (vs Cycling)
# Modified by severity, polysubstance status, MH burden at enrollment
latent_class_base_remission_prob = 0.42

# Severity adjustment: Severe → -0.15, Moderate → -0.05, Mild → +0.10
severity_remission_adj = {"Severe": -0.15, "Moderate": -0.05, "Mild": 0.10}

# Polysubstance penalty
polysubstance_remission_adj = -0.10

# Per-MH-diagnosis penalty on remission probability
mh_remission_penalty_per_dx = -0.04
