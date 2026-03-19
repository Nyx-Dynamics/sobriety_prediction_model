import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

def visualize_temporal_signatures(h_T, latent_classes):
    """
    h_T: (N, hidden_dim) final hidden states from LSTM
    latent_classes: (N,) binary labels (0=Cycling, 1=Remission)
    """
    # 1. PCA for initial reduction
    pca_result = PCA(n_components=2).fit_transform(h_T)

    # 2. t-SNE for manifold visualization
    tsne_result = TSNE(n_components=2, perplexity=30).fit_transform(h_T)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot PCA
    sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=latent_classes,
                    palette='coolwarm', ax=ax[0])
    ax[0].set_title("PCA: LSTM Hidden State Space")

    # Plot t-SNE
    sns.scatterplot(x=tsne_result[:,0], y=tsne_result[:,1], hue=latent_classes,
                    palette='coolwarm', ax=ax[1])
    ax[1].set_title("t-SNE: Temporal Signature Clustering")

    plt.suptitle("Verification of Latent Class Linear Separability")
    plt.show()
