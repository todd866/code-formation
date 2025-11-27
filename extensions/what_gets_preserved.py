#!/usr/bin/env python3
"""
What Gets Preserved? Understanding Dimensionality Reduction

When we reduce 512-D to k-D, how does the network decide what to keep?

Answer: The network learns to preserve variance along directions that minimize
reconstruction error under noise. This naturally preserves "semantic" structure
and discards noise/fine details.

Author: Ian Todd
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_formation import generate_ring_data, generate_ring_labels, train_network

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def analyze_what_gets_preserved():
    print("=" * 70)
    print("WHAT GETS PRESERVED IN DIMENSIONALITY REDUCTION?")
    print("=" * 70)

    # Generate data
    print("\n1. Generating 512-D ring manifold...")
    data, theta = generate_ring_data(n_samples=2000, input_dim=512)
    true_labels = generate_ring_labels(theta)
    data_np = data.numpy()

    # Continuous color gradient (not discrete categories)
    colors = plt.cm.hsv(theta.numpy() / (2 * np.pi))

    # PCA on input
    print("2. Analyzing input structure with PCA...")
    pca = PCA(n_components=10)
    proj = pca.fit_transform(data_np)

    print(f"\n   Variance explained:")
    print(f"     PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"     PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
    print(f"     PC3-10: {sum(pca.explained_variance_ratio_[2:])*100:.1f}% total")
    print(f"\n   → Input is ~2D (PC1-2 = ring, PC3+ = noise)")

    # Train at different k
    print("\n3. Training encoders at k = 1, 2, 4, 8...")
    k_values = [1, 2, 4, 8]
    results = {}

    for k in k_values:
        code, error, model = train_network(k, data, noise_std=0.3, epochs=150, verbose=False)

        # Compute ARI
        if k == 1:
            cluster_labels = np.digitize(code.flatten(), np.linspace(code.min(), code.max(), 7)) - 1
        else:
            kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(code)
        ari = adjusted_rand_score(true_labels, cluster_labels)

        # Correlation with PCs
        pc_corrs = []
        for d in range(min(k, 2)):
            corr1 = np.abs(np.corrcoef(code[:, d], proj[:, 0])[0, 1])
            corr2 = np.abs(np.corrcoef(code[:, d], proj[:, 1])[0, 1])
            pc_corrs.append(max(corr1, corr2))

        results[k] = {'code': code, 'error': error, 'ari': ari, 'pc_corrs': pc_corrs}
        print(f"     k={k}: error={error:.5f}, ARI={ari:.3f}, PC corr={pc_corrs}")

    # Create clean 2x4 figure
    print("\n4. Creating visualization...")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    # Row 1: Input structure and PCA
    ax = axes[0, 0]
    ax.scatter(proj[:, 0], proj[:, 1], c=colors, alpha=0.5, s=12)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Input in PC space\n(ring manifold)', fontweight='bold', fontsize=10)
    ax.set_aspect('equal')

    ax = axes[0, 1]
    ax.bar(range(1, 6), pca.explained_variance_ratio_[:5] * 100, color='steelblue', edgecolor='black')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance (%)')
    ax.set_title('PC1-2 capture ring\n(rest is noise)', fontweight='bold', fontsize=10)
    ax.set_xticks(range(1, 6))

    ax = axes[0, 2]
    ax.scatter(theta.numpy(), proj[:, 0], c=colors, alpha=0.3, s=8)
    ax.set_xlabel('θ (angle)')
    ax.set_ylabel('PC1')
    ax.set_title('PC1 ≈ cos(θ)', fontweight='bold', fontsize=10)

    ax = axes[0, 3]
    ax.scatter(theta.numpy(), proj[:, 1], c=colors, alpha=0.3, s=8)
    ax.set_xlabel('θ (angle)')
    ax.set_ylabel('PC2')
    ax.set_title('PC2 ≈ sin(θ)', fontweight='bold', fontsize=10)

    # Row 2: What each k preserves
    for idx, k in enumerate(k_values):
        ax = axes[1, idx]
        code = results[k]['code']
        ari = results[k]['ari']

        if k == 1:
            ax.scatter(theta.numpy(), code[:, 0], c=colors, alpha=0.5, s=12)
            ax.set_xlabel('θ')
            ax.set_ylabel('Code')
            title = f'k=1: Line collapse\nARI={ari:.2f}'
        else:
            ax.scatter(code[:, 0], code[:, 1], c=colors, alpha=0.5, s=12)
            ax.set_xlabel('Code dim 1')
            ax.set_ylabel('Code dim 2')
            ax.set_aspect('equal', adjustable='datalim')
            if k == 2:
                title = f'k=2: Discretizes!\nARI={ari:.2f}'
            else:
                title = f'k={k}: Ring preserved\nARI={ari:.2f}'

        ax.set_title(title, fontweight='bold', fontsize=10)

    plt.suptitle('How does the network decide what to preserve?\n'
                 'Answer: High-variance directions that survive noise (≈ top PCs)',
                 fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, 'what_gets_preserved.png'), dpi=150, bbox_inches='tight')
    print(f"\nSaved to output/what_gets_preserved.png")
    plt.show()

    # Summary
    print("\n" + "=" * 70)
    print("THE ANSWER")
    print("=" * 70)
    print("""
The encoder minimizes ||x - decode(encode(x) + noise)||²

This preserves:
  ✓ High-variance directions (PC1-2 = ring structure)
  ✓ Noise-robust features (categorical boundaries)

This discards:
  ✗ Low-variance directions (PC3+ = embedding noise)
  ✗ Fine details destroyed by channel noise

Result: The code correlates strongly with top PCs (~0.8-0.9).
At k=2, noise forces discretization into 6 stable codes.
""")
    print("=" * 70)


if __name__ == "__main__":
    analyze_what_gets_preserved()
