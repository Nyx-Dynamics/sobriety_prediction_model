"""
Phase 5: Bayesian Pathway Attribution Model
=============================================
Estimates log-hazard contribution of each structural resource pathway
using transfer learning from Phase 3 MH effects.

Nyx Dynamics LLC | MIT MicroMasters Capstone
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

np.random.seed(config.SEED)

# ── Load data: priority order NTIES > DATOS > TEDS-D > CALDATA > synthetic
proc_dir = config.PROJECT_ROOT / "data" / "phase5" / "processed"
nties_path = proc_dir / "nties_phase5.csv"
datos_path = proc_dir / "datos_phase5.csv"
teds_path = proc_dir / "teds_phase5.csv"
caldata_path = proc_dir / "caldata_phase5.csv"
synthetic_path = config.PROJECT_ROOT / "data" / "phase5" / "stratified" / "patient_static_stratified.csv"

validation_df = None

if nties_path.exists():
    df = pd.read_csv(nties_path)
    data_source = "NTIES (ICPSR 2884)"
    print(f"[phase5] Primary: NTIES N={len(df)}")
    if datos_path.exists():
        validation_df = pd.read_csv(datos_path)
        print(f"[phase5] Validation: DATOS N={len(validation_df)}")
elif datos_path.exists():
    df = pd.read_csv(datos_path)
    data_source = "DATOS (ICPSR 2258)"
    print(f"[phase5] Primary: DATOS N={len(df)}")
elif teds_path.exists():
    df = pd.read_csv(teds_path)
    data_source = "TEDS-D 2022"
    print(f"[phase5] Primary: TEDS-D 2022 N={len(df)}")
elif synthetic_path.exists():
    df = pd.read_csv(synthetic_path)
    data_source = "Synthetic stratified cohort"
    print(f"[phase5] WARNING: Using synthetic data — run prepare_nties.py first")
else:
    raise FileNotFoundError(
        "No Phase 5 data found. Run: python src/phase5/run_phase5.py"
    )

# Subsample for first run if TEDS-D is very large
if data_source == "TEDS-D 2022" and len(df) > 50000:
    print(f"[phase5] Subsampling to 50,000 for initial run")
    df = df.sample(n=50000, random_state=config.SEED).reset_index(drop=True)

# ── Load Phase 3 beta_mh posteriors (transfer learning) ──────────────
bayes_dir = config.PROJECT_ROOT / "models" / "bayesian"
summary_p3 = pd.read_csv(bayes_dir / "bayesian_summary.csv", index_col=0)

mh_param_names = [f"beta_mh[{i}]" for i in range(8)]
beta_mh_fixed = np.array([summary_p3.loc[n, "mean"] for n in mh_param_names])

# ── Build MH matrix ──────────────────────────────────────────────────
# For TEDS-D: single mh_comorbidity column → broadcast to all 8 MH slots
# For synthetic: full per-diagnosis columns available
if "mh_comorbidity" in df.columns and "mh_major_depression" not in df.columns:
    # TEDS-D format: single binary MH flag
    mh_val = df["mh_comorbidity"].values.astype(float)
    X_mh = np.column_stack([mh_val] * 8)  # uniform effect across MH dimensions
else:
    # Synthetic format: per-diagnosis columns
    mh_cols_8 = ["mh_major_depression", "mh_ptsd", "mh_gad", "mh_adhd",
                 "mh_bipolar", "mh_bipolar", "mh_psychotic_disorder", "mh_personality_disorder"]
    X_mh = np.zeros((len(df), 8))
    for j, col in enumerate(mh_cols_8):
        if col in df.columns:
            X_mh[:, j] = df[col].values
    X_mh[:, 5] = 0  # bipolar_ii not split

# ── Resource pathway features ────────────────────────────────────────
resource_cols = ["treatment_intensity", "monitoring_intensity",
                 "social_support_density", "self_advocacy_index"]
R = df[resource_cols].values.astype(np.float64)

R_mean = R.mean(axis=0)
R_std = R.std(axis=0) + 1e-8
R_z = (R - R_mean) / R_std

# ── Outcomes ──────────────────────────────────────────────────────────
if "event" in df.columns and "days_to_event" in df.columns:
    # TEDS-D or pre-built outcomes
    event = df["event"].values.astype(np.int64)
    quarters = np.clip((df["days_to_event"].values / 90).astype(int), 1, 12)
elif "patient_id" in df.columns:
    # Synthetic: build from panel
    panel = pd.read_csv(config.PROJECT_ROOT / "data" / "phase5" / "stratified" / "patient_panel_stratified.csv")
    outcome_rows = []
    for pid in df["patient_id"]:
        pt = panel[panel["patient_id"] == pid].sort_values("quarter")
        n_obs = len(pt)
        ever_sober = False
        ev = 0
        ev_q = n_obs
        for _, row in pt.iterrows():
            if row["sober_this_quarter"] == 1:
                ever_sober = True
            elif ever_sober and row["uds_positive"] == 1:
                ev = 1
                ev_q = row["quarter"]
                break
        if ev == 0 and not ever_sober and (pt["uds_positive"] == 1).any():
            if np.random.random() < 0.65:
                ev = 1
                ev_q = int(pt[pt["uds_positive"] == 1].iloc[-1]["quarter"])
        outcome_rows.append({"event": ev, "quarters": int(ev_q)})
    outcomes_df = pd.DataFrame(outcome_rows)
    event = outcomes_df["event"].values.astype(np.int64)
    quarters = np.clip(outcomes_df["quarters"].values.astype(int), 1, 12)
else:
    raise ValueError("Cannot determine outcomes from data")

N = len(df)
K_resource = len(resource_cols)

# MH effect (fixed from Phase 3)
mh_effect_fixed = X_mh @ beta_mh_fixed  # (N,)

print("=" * 60)
print("PHASE 5: PATHWAY ATTRIBUTION MODEL")
print("=" * 60)
print(f"  N patients: {N}")
print(f"  Event rate: {event.mean():.3f}")
print(f"  Resource features: {resource_cols}")
print(f"  Phase 3 beta_mh (fixed): {beta_mh_fixed.round(3)}")

# ══════════════════════════════════════════════════════════════════════
# PyMC Model
# ══════════════════════════════════════════════════════════════════════

with pm.Model() as pathway_model:

    # ── Latent class baseline hazards ──
    log_h0_rem = pm.Normal("log_h0_rem", mu=-3.5, sigma=0.5)
    log_h0_cyc = pm.Normal("log_h0_cyc", mu=-2.0, sigma=0.5)

    # ── Resource pathway effects (THE PARAMETERS OF INTEREST) ──
    gamma = pm.Normal("gamma", mu=0, sigma=0.3, shape=K_resource)

    # ── Latent class mixing weight ──
    p_rem = pm.Beta("p_remission", alpha=2, beta=2)

    # ── Per-patient log-hazard ──
    resource_effect = pm.math.dot(R_z, gamma)  # (N,)

    # Class-specific hazards with MH (fixed) and resource effects
    h_rem = pm.math.exp(log_h0_rem + mh_effect_fixed + resource_effect)
    h_cyc = pm.math.exp(log_h0_cyc + mh_effect_fixed + resource_effect)

    h_rem_c = pm.math.clip(h_rem, 1e-8, 1 - 1e-8)
    h_cyc_c = pm.math.clip(h_cyc, 1e-8, 1 - 1e-8)

    # ── Discrete-time survival likelihood (mixture) ──
    log_surv_rem = (quarters - 1) * pm.math.log(1 - h_rem_c)
    log_surv_cyc = (quarters - 1) * pm.math.log(1 - h_cyc_c)

    log_lik_rem = pm.math.switch(
        event,
        log_surv_rem + pm.math.log(h_rem_c),
        quarters * pm.math.log(1 - h_rem_c),
    )
    log_lik_cyc = pm.math.switch(
        event,
        log_surv_cyc + pm.math.log(h_cyc_c),
        quarters * pm.math.log(1 - h_cyc_c),
    )

    log_mix = pm.math.logsumexp(
        pm.math.stack([
            pm.math.log(p_rem) + log_lik_rem,
            pm.math.log(1 - p_rem) + log_lik_cyc,
        ], axis=0),
        axis=0,
    )

    pm.Potential("log_likelihood", log_mix.sum())

    # ── Sample ──
    trace = pm.sample(
        draws=2000, tune=1000, chains=4,
        target_accept=0.9,
        random_seed=config.SEED,
        return_inferencedata=True,
    )

# ── Save outputs ──────────────────────────────────────────────────────
out_dir = config.PROJECT_ROOT / "models" / "phase5"
out_dir.mkdir(parents=True, exist_ok=True)

trace.to_netcdf(str(out_dir / "pathway_trace.nc"))
summary = az.summary(trace, var_names=["gamma", "log_h0_rem", "log_h0_cyc", "p_remission"],
                     hdi_prob=0.94)
summary.to_csv(out_dir / "pathway_summary.csv")

gamma_samples = trace.posterior["gamma"].values.reshape(-1, K_resource)
np.save(out_dir / "gamma_posteriors.npy", gamma_samples)

print(f"\n✓ Saved: {out_dir / 'pathway_trace.nc'}")
print(f"✓ Saved: {out_dir / 'pathway_summary.csv'}")

# ── Print results ─────────────────────────────────────────────────────
gamma_mean = gamma_samples.mean(axis=0)
gamma_hdi_lo = np.percentile(gamma_samples, 3, axis=0)
gamma_hdi_hi = np.percentile(gamma_samples, 97, axis=0)

# Resource ranges (Stratum A max to Stratum D min, standardized)
resource_ranges_raw = np.array([5 - 0, 4 - 0, 10 - 0, 10 - 0], dtype=float)
resource_ranges_z = resource_ranges_raw / R_std

print("\n── Pathway Attribution (posterior means, 94% HDI) ──")
pathway_names = ["Treatment access", "Monitoring continuity",
                 "Social support density", "Clinical self-advocacy"]
for k in range(K_resource):
    hr = np.exp(gamma_mean[k] * resource_ranges_z[k])
    print(f"  {pathway_names[k]:>25}: gamma={gamma_mean[k]:.3f} "
          f"[{gamma_hdi_lo[k]:.3f}, {gamma_hdi_hi[k]:.3f}]  "
          f"HR(A→D)={hr:.2f}")

# ── Population-attributable fractions ─────────────────────────────────
# PAF_k = 1 - exp(-gamma_k * mean_deprivation_k)
mean_R_z = R_z.mean(axis=0)
total_resource_hazard = np.exp((gamma_mean * mean_R_z).sum())

print("\n── Population-Attributable Fractions ──")
pafs = {}
for k in range(K_resource):
    # Counterfactual: remove pathway k's contribution
    cf_hazard = np.exp(sum(gamma_mean[j] * mean_R_z[j] for j in range(K_resource) if j != k))
    paf_k = (total_resource_hazard - cf_hazard) / total_resource_hazard
    pafs[resource_cols[k]] = paf_k
    print(f"  {pathway_names[k]:>25}: PAF = {paf_k:.1%}")

total_paf = sum(pafs.values())
print(f"  {'Total explained':>25}: {total_paf:.1%}")

# Save PAFs for report
results = {
    "gamma_mean": gamma_mean.tolist(),
    "gamma_hdi_lo": gamma_hdi_lo.tolist(),
    "gamma_hdi_hi": gamma_hdi_hi.tolist(),
    "pathway_names": pathway_names,
    "resource_cols": resource_cols,
    "pafs": {k: float(v) for k, v in pafs.items()},
    "R_mean": R_mean.tolist(),
    "R_std": R_std.tolist(),
    "resource_ranges_raw": resource_ranges_raw.tolist(),
}

import json
with open(out_dir / "pathway_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Saved: {out_dir / 'pathway_results.json'}")
