"""
Latent-Class Bayesian Survival Model (Phase 3)
================================================
Sobriety Prediction Model | Nyx Dynamics LLC | MIT MicroMasters Capstone

Fits a discrete-time mixture survival model with two latent classes
(Remission vs Cycling) using PyMC. MH diagnosis flags modulate
per-class hazards via log-linear effects.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(config.SEED)

# ── Load data ─────────────────────────────────────────────────────────
static = pd.read_csv(config.STATIC_DIR / "sud_static.csv")
outcomes = pd.read_csv(config.OUTCOMES_DIR / "sud_outcomes.csv")
train_idx = np.load(config.LSTM_DIR / "train_idx.npy")
test_idx = np.load(config.LSTM_DIR / "test_idx.npy")

# ── Build feature matrix ──────────────────────────────────────────────
mh_cols = ["mdd", "ptsd", "gad", "adhd", "bipolar_i", "bipolar_ii", "psychosis", "bpd"]
numeric_cols = ["sud_severity", "social_support", "prior_tx_episodes", "sud_type"]
feature_cols = numeric_cols + mh_cols

# Align to outcome ordering
merged = outcomes.merge(static, on="patient_id", suffixes=("", "_static"))
X_all = merged[feature_cols].values.astype(np.float64)
event_all = merged["event"].values.astype(np.int64)
days_all = merged["days_to_event"].values.astype(np.float64)

# Split
X_train = X_all[train_idx]
event_train = event_all[train_idx]
days_train = days_all[train_idx]

X_test = X_all[test_idx]
event_test = event_all[test_idx]
days_test = days_all[test_idx]

N_train = len(train_idx)
N_test = len(test_idx)
K_mh = len(mh_cols)
K_total = len(feature_cols)

# Standardize continuous features (train-fit)
cont_idx = [0, 1, 2]  # sud_severity, social_support, prior_tx_episodes
train_mean = X_train[:, cont_idx].mean(axis=0)
train_std = X_train[:, cont_idx].std(axis=0) + 1e-8
X_train[:, cont_idx] = (X_train[:, cont_idx] - train_mean) / train_std
X_test[:, cont_idx] = (X_test[:, cont_idx] - train_mean) / train_std

# Convert days to discrete quarters for hazard computation
quarters_train = np.clip((days_train / 90).astype(int), 1, 12)
quarters_test = np.clip((days_test / 90).astype(int), 1, 12)
MAX_Q = 12

print("=" * 70)
print("BAYESIAN LATENT-CLASS SURVIVAL MODEL")
print("=" * 70)
print(f"Train: {N_train}, Test: {N_test}")
print(f"Features: {feature_cols}")
print(f"Train event rate: {event_train.mean():.3f}")

# ══════════════════════════════════════════════════════════════════════
# PyMC Model
# ══════════════════════════════════════════════════════════════════════

# Extract MH columns from feature matrix (indices 4-11)
X_mh_train = X_train[:, len(numeric_cols):].astype(np.float64)

with pm.Model() as survival_model:

    # ── Latent class membership: Z_i ~ Bernoulli(sigmoid(X @ alpha)) ──
    alpha = pm.Normal("alpha", mu=0, sigma=1, shape=K_total)
    logit_p_rem = pm.math.dot(X_train, alpha)
    p_remission = pm.math.sigmoid(logit_p_rem)

    # ── Class-specific baseline log-hazards ──
    log_h0_rem = pm.Normal("log_h0_rem", mu=-3.5, sigma=0.5)
    log_h0_cyc = pm.Normal("log_h0_cyc", mu=-2.0, sigma=0.5)

    # ── MH diagnosis effects on hazard (shared across classes) ──
    beta_mh = pm.Normal("beta_mh", mu=0, sigma=0.5, shape=K_mh)

    # ── Per-patient log-hazard contribution from MH ──
    mh_effect = pm.math.dot(X_mh_train, beta_mh)

    # ── Per-class hazard per quarter ──
    h_rem = pm.math.exp(log_h0_rem + mh_effect)  # (N_train,)
    h_cyc = pm.math.exp(log_h0_cyc + mh_effect)  # (N_train,)

    # ── Discrete-time survival likelihood (mixture) ──
    # S(t) = prod_{q=1}^{t-1} (1-h)  for survived quarters
    # f(t) = S(t-1) * h              for event at quarter t
    # For censored: L = S(t_censor)
    # Mixture: L = p_rem * L_rem + (1-p_rem) * L_cyc

    # Compute log-likelihoods per class
    h_rem_clipped = pm.math.clip(h_rem, 1e-8, 1 - 1e-8)
    h_cyc_clipped = pm.math.clip(h_cyc, 1e-8, 1 - 1e-8)

    # Log-survival to observed time: sum of log(1-h) for survived quarters
    log_surv_rem = (quarters_train - 1) * pm.math.log(1 - h_rem_clipped)
    log_surv_cyc = (quarters_train - 1) * pm.math.log(1 - h_cyc_clipped)

    # Log-density at event time (for events) or log-survival (for censored)
    log_lik_rem = pm.math.switch(
        event_train,
        log_surv_rem + pm.math.log(h_rem_clipped),     # event
        quarters_train * pm.math.log(1 - h_rem_clipped)  # censored
    )
    log_lik_cyc = pm.math.switch(
        event_train,
        log_surv_cyc + pm.math.log(h_cyc_clipped),
        quarters_train * pm.math.log(1 - h_cyc_clipped)
    )

    # Mixture log-likelihood via logsumexp
    log_mix = pm.math.logsumexp(
        pm.math.stack([
            pm.math.log(p_remission) + log_lik_rem,
            pm.math.log(1 - p_remission) + log_lik_cyc,
        ], axis=0),
        axis=0,
    )

    pm.Potential("log_likelihood", log_mix.sum())

    # ── Sample ──
    trace = pm.sample(
        draws=1000, tune=500, chains=2,
        target_accept=0.9,
        random_seed=config.SEED,
        return_inferencedata=True,
    )

# ══════════════════════════════════════════════════════════════════════
# Save outputs
# ══════════════════════════════════════════════════════════════════════

model_dir = config.PROJECT_ROOT / "models" / "bayesian"
model_dir.mkdir(parents=True, exist_ok=True)

# Save trace
trace.to_netcdf(str(model_dir / "bayesian_trace.nc"))
print(f"\n✓ Trace saved: {model_dir / 'bayesian_trace.nc'}")

# Summary table
summary = az.summary(trace, var_names=["log_h0_rem", "log_h0_cyc", "beta_mh", "alpha"],
                     hdi_prob=0.94)
summary.to_csv(model_dir / "bayesian_summary.csv")
print(f"✓ Summary saved: {model_dir / 'bayesian_summary.csv'}")

# ── Posterior class probabilities for test patients ───────────────────
alpha_samples = trace.posterior["alpha"].values  # (chains, draws, K_total)
alpha_mean = alpha_samples.mean(axis=(0, 1))
logit_test = X_test @ alpha_mean
p_rem_test = 1 / (1 + np.exp(-logit_test))
np.save(model_dir / "posterior_class_probs.npy", p_rem_test)
print(f"✓ Posterior class probs saved: {model_dir / 'posterior_class_probs.npy'}")

# ══════════════════════════════════════════════════════════════════════
# Print results
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("POSTERIOR MH HAZARD RATIOS (exp(beta_mh))")
print("=" * 70)

beta_samples = trace.posterior["beta_mh"].values  # (chains, draws, K_mh)
beta_flat = beta_samples.reshape(-1, K_mh)
hr_samples = np.exp(beta_flat)

hr_results = []
for k, mh_name in enumerate(mh_cols):
    hr_mean = hr_samples[:, k].mean()
    hr_lo, hr_hi = np.percentile(hr_samples[:, k], [3, 97])
    hr_results.append((mh_name, hr_mean, hr_lo, hr_hi))
    print(f"  {mh_name:>12}: HR = {hr_mean:.3f}  [{hr_lo:.3f}, {hr_hi:.3f}]")

# Top 3 risk factors
hr_results.sort(key=lambda x: -x[1])
print(f"\nTop 3 risk factors by posterior mean HR:")
for rank, (name, hr, lo, hi) in enumerate(hr_results[:3], 1):
    print(f"  {rank}. {name}: HR = {hr:.3f}  [{lo:.3f}, {hi:.3f}]")

# Test set remission estimate
pct_rem_test = (p_rem_test > 0.5).mean() * 100
print(f"\nTest patients in remission class (P>0.5): {pct_rem_test:.1f}%")
print(f"Mean P(remission) on test set: {p_rem_test.mean():.3f}")

# ── Baseline hazards ──
log_h0_rem_samples = trace.posterior["log_h0_rem"].values.flatten()
log_h0_cyc_samples = trace.posterior["log_h0_cyc"].values.flatten()
print(f"\nBaseline hazards (per quarter):")
print(f"  Remission: {np.exp(log_h0_rem_samples.mean()):.4f} "
      f"[{np.exp(np.percentile(log_h0_rem_samples, 3)):.4f}, "
      f"{np.exp(np.percentile(log_h0_rem_samples, 97)):.4f}]")
print(f"  Cycling:   {np.exp(log_h0_cyc_samples.mean()):.4f} "
      f"[{np.exp(np.percentile(log_h0_cyc_samples, 3)):.4f}, "
      f"{np.exp(np.percentile(log_h0_cyc_samples, 97)):.4f}]")

# ══════════════════════════════════════════════════════════════════════
# Posterior plot: 8-panel MH effects
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()

for k, mh_name in enumerate(mh_cols):
    ax = axes[k]
    samples = beta_flat[:, k]
    hr_samp = np.exp(samples)
    hr_mean = hr_samp.mean()
    hdi_lo, hdi_hi = np.percentile(hr_samp, [3, 97])

    ax.hist(hr_samp, bins=50, color="#2c7bb6", alpha=0.7, density=True, edgecolor="white")

    # Shade 94% HDI
    x_fill = np.linspace(hdi_lo, hdi_hi, 100)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(hr_samp)
    ax.fill_between(x_fill, kde(x_fill), alpha=0.3, color="#fdae61")

    # Null line (HR=1)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.5, label="HR=1 (null)")
    # Posterior mean
    ax.axvline(hr_mean, color="#d7191c", linestyle="-", linewidth=2, label=f"Mean={hr_mean:.2f}")

    ax.set_title(f"{mh_name}\nHR={hr_mean:.2f} [{hdi_lo:.2f}, {hdi_hi:.2f}]", fontsize=10)
    ax.set_xlabel("Hazard Ratio")
    if k == 0:
        ax.legend(fontsize=8)

fig.suptitle("Posterior MH Diagnosis Hazard Ratios (94% HDI)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "posterior_mh_effects.png", dpi=150)
print(f"\n✓ Figure saved: {config.FIGURES_DIR / 'posterior_mh_effects.png'}")
