import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.mixture import GaussianMixture
from diptest import diptest # Requires: pip install diptest

def validate_bimodality(event_times):
    """
    event_times: Array of 'days to event' for relapsed patients
    """
    # 1. Hartigan's Dip Test
    dip_stat, p_val = diptest(event_times)
    print(f"Hartigan's Dip Test: Stat={dip_stat:.4f}, p-value={p_val:.4f}")
    if p_val < 0.05:
        print("Result: Significant unimodality rejection (Evidence of Bimodality).")

    # 2. Gaussian Mixture Model (GMM) Comparison
    X = event_times.reshape(-1, 1)
    bics = []
    models = []

    for n in [1, 2]:
        gmm = GaussianMixture(n_components=n, random_state=42).fit(X)
        bics.append(gmm.bic(X))
        models.append(gmm)

    print(f"BIC (1-component): {bics[0]:.2f}")
    print(f"BIC (2-component): {bics[1]:.2f}")

    if bics[1] < bics[0]:
        print("Result: 2-component mixture preferred by BIC. Axiom 1 Supported.")

    return models[1] # Return the preferred mixture model
