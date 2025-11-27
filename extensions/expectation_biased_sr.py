#!/usr/bin/env python3
"""
Expectation-Biased Stochastic Resonance via Variable Dimensionality

The key insight: NOISE INCREASES DIMENSIONALITY.

When a signal is ambiguous (sitting between two codes), it's effectively
1-dimensional - collapsed onto the boundary between basins. Adding noise
EXPANDS the representation into higher dimensions, allowing the system
to "explore" and find the correct attractor basin.

This reframes stochastic resonance as a DIMENSIONALITY phenomenon:
- Low noise: signal is trapped in low-D subspace (on boundary)
- Optimal noise: dimensionality expands enough to escape to basin
- High noise: too much dimensionality, random wandering

Expectation bias then acts by:
- Pre-expanding dimensionality in the "expected" direction
- Reducing the noise needed to reach correct basin

Author: Ian Todd
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_formation import generate_ring_data, generate_ring_labels, train_network

# Configuration
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RAINBOW = ['#E41A1C', '#FF7F00', '#FFFF33', '#4DAF4A', '#377EB8', '#984EA3']


def compute_effective_dimensionality(data, threshold=0.95):
    """
    Compute effective dimensionality as number of PCA components
    needed to explain threshold% of variance.
    """
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    if data.shape[0] < 2:
        return 1

    pca = PCA()
    pca.fit(data)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    return np.searchsorted(cumvar, threshold) + 1


def compute_participation_ratio(data):
    """
    Participation ratio: a continuous measure of dimensionality.
    PR = (sum of eigenvalues)^2 / sum of eigenvalues^2

    For data spread evenly across d dimensions, PR ≈ d.
    For data along one direction, PR ≈ 1.
    """
    if len(data.shape) == 1 or data.shape[0] < 2:
        return 1.0

    # Add tiny jitter to prevent zero covariance in constant signals
    data = data + np.random.randn(*data.shape) * 1e-9

    # Center the data
    centered = data - data.mean(axis=0)

    # Compute covariance eigenvalues
    cov = np.cov(centered.T)
    if cov.ndim == 0:
        return 1.0

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Numerical stability

    if len(eigenvalues) == 0:
        return 1.0

    # Participation ratio
    pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    return pr


def create_ambiguous_signals(code_centroids, n_signals=500, ambiguity=0.4):
    """
    Create signals that sit BETWEEN codes (on low-D boundary).
    """
    n_codes = len(code_centroids)
    signals = []
    true_labels = []

    for i in range(n_signals):
        true_code = i % n_codes
        distractor = (true_code + 1) % n_codes
        signal = (1 - ambiguity) * code_centroids[true_code] + ambiguity * code_centroids[distractor]
        signals.append(signal)
        true_labels.append(true_code)

    return np.array(signals), np.array(true_labels)


def sr_with_dimensionality_tracking(signals, code_centroids, noise_level,
                                     bias_toward=None, bias_strength=0.0,
                                     n_timesteps=10):
    """
    Stochastic resonance with dimensionality tracking over time.

    Simulates a dynamical process where:
    1. Signal starts at ambiguous position
    2. Noise expands dimensionality
    3. Bias pulls toward expected code
    4. System settles into attractor basin

    Returns decoded labels AND trajectory of effective dimensionality.
    """
    n_signals = signals.shape[0]
    n_codes = len(code_centroids)
    dim = signals.shape[1]

    # Track trajectories
    trajectories = np.zeros((n_signals, n_timesteps, dim))
    trajectories[:, 0, :] = signals

    # Dynamics
    for t in range(1, n_timesteps):
        # Current position
        current = trajectories[:, t-1, :]

        # Add noise (expands dimensionality)
        noise = np.random.randn(*current.shape) * noise_level

        # Apply expectation bias (directional expansion)
        if bias_toward is not None and bias_strength > 0:
            for i in range(n_signals):
                expected = code_centroids[bias_toward[i]]
                bias_vec = bias_strength * (expected - current[i])
                noise[i] += bias_vec

        # Attractor dynamics: pull toward nearest code
        for i in range(n_signals):
            distances = [np.linalg.norm(current[i] - c) for c in code_centroids]
            nearest = np.argmin(distances)
            attractor_pull = 0.3 * (code_centroids[nearest] - current[i])
            trajectories[:, t, :] = current + noise + attractor_pull

    # Final decoding
    final_pos = trajectories[:, -1, :]
    decoded = np.zeros(n_signals, dtype=int)
    for i in range(n_signals):
        distances = [np.linalg.norm(final_pos[i] - c) for c in code_centroids]
        decoded[i] = np.argmin(distances)

    # Compute dimensionality at each timestep
    dim_trajectory = []
    for t in range(n_timesteps):
        pr = compute_participation_ratio(trajectories[:, t, :])
        dim_trajectory.append(pr)

    return decoded, trajectories, dim_trajectory


def run_sr_experiment():
    """
    Demonstrate that SR works via dimensionality expansion.
    """
    print("=" * 70)
    print("STOCHASTIC RESONANCE AS DIMENSIONALITY EXPANSION")
    print("=" * 70)
    print("\nKey insight: Noise INCREASES effective dimensionality,")
    print("allowing ambiguous signals to escape low-D boundaries.")

    # Step 1: Train code formation network
    print("\n1. Training code formation network (k=2)...")
    data, theta = generate_ring_data(n_samples=2000)
    true_labels_train = generate_ring_labels(theta)

    code_np, error, model = train_network(channel_dim=2, data=data, noise_std=0.3, epochs=150)

    # Step 2: Extract learned codes
    print("2. Extracting learned code centroids...")
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    kmeans.fit(code_np)
    code_centroids = kmeans.cluster_centers_

    # Compute inter-code distance
    distances = []
    for i in range(6):
        for j in range(i+1, 6):
            distances.append(np.linalg.norm(code_centroids[i] - code_centroids[j]))
    mean_dist = np.mean(distances)

    # Step 3: Create ambiguous signals
    print("3. Creating ambiguous signals (low-D: on basin boundaries)...")
    ambiguity = 0.42
    test_signals, test_labels = create_ambiguous_signals(code_centroids, n_signals=500, ambiguity=ambiguity)

    # Measure initial dimensionality
    initial_dim = compute_participation_ratio(test_signals)
    print(f"   Initial effective dimensionality: {initial_dim:.2f}")
    print(f"   (Should be ~1 since signals are on 1D boundaries)")

    # Step 4: Sweep noise levels
    print("4. Running dimensionality-aware SR sweep...")

    noise_levels = np.linspace(0, mean_dist * 0.6, 20)
    n_timesteps = 8
    n_trials = 30

    conditions = {
        'No bias': (None, 0.0),
        'Correct bias': (test_labels, 0.25),
        'Wrong bias': ((test_labels + 3) % 6, 0.25),
    }

    results_acc = {name: [] for name in conditions}
    results_dim = {name: [] for name in conditions}

    for noise in noise_levels:
        for name, (bias_labels, bias_strength) in conditions.items():
            trial_accs = []
            trial_dims = []

            for _ in range(n_trials):
                decoded, trajectories, dim_traj = sr_with_dimensionality_tracking(
                    test_signals, code_centroids, noise,
                    bias_toward=bias_labels, bias_strength=bias_strength,
                    n_timesteps=n_timesteps
                )
                trial_accs.append(np.mean(decoded == test_labels))
                trial_dims.append(np.mean(dim_traj))  # Average dimensionality over trajectory

            results_acc[name].append(np.mean(trial_accs))
            results_dim[name].append(np.mean(trial_dims))

    # Step 5: Create figure
    print("5. Generating visualization...")

    fig = plt.figure(figsize=(15, 10))

    # Panel (a): Code space with ambiguous signals
    ax1 = fig.add_subplot(2, 3, 1)

    colors = [RAINBOW[l] for l in true_labels_train]
    ax1.scatter(code_np[:, 0], code_np[:, 1], c=colors, alpha=0.15, s=10)

    for i, c in enumerate(code_centroids):
        ax1.scatter([c[0]], [c[1]], c=RAINBOW[i], s=250, marker='o',
                   edgecolors='black', linewidths=2, zorder=10)

    # Show ambiguous signals on boundaries
    for i in range(0, 50, 8):
        ax1.scatter([test_signals[i, 0]], [test_signals[i, 1]],
                   c='black', s=60, marker='x', linewidths=2, zorder=8)

    ax1.set_xlabel('Code dim 1')
    ax1.set_ylabel('Code dim 2')
    ax1.set_title('(a) Codes and ambiguous signals\n× on low-D boundaries', fontweight='bold')
    ax1.set_aspect('equal', adjustable='datalim')

    # Panel (b): SR curves (accuracy)
    ax2 = fig.add_subplot(2, 3, 2)

    colors_line = {'No bias': '#666666', 'Correct bias': '#D55E00', 'Wrong bias': '#0072B2'}

    for name, accs in results_acc.items():
        lw = 3 if name == 'No bias' else 2.5
        ls = '-' if name != 'Wrong bias' else '--'
        ax2.plot(noise_levels / mean_dist, accs, ls, color=colors_line[name],
                label=name, linewidth=lw)

    # Mark SR peak
    no_bias_acc = results_acc['No bias']
    peak_idx = np.argmax(no_bias_acc)
    if 0 < peak_idx < len(no_bias_acc) - 1:
        ax2.scatter([noise_levels[peak_idx] / mean_dist], [no_bias_acc[peak_idx]],
                   c='#666666', s=200, marker='*', zorder=10, edgecolors='white', linewidths=1.5)
        ax2.annotate('SR peak', xy=(noise_levels[peak_idx] / mean_dist, no_bias_acc[peak_idx]),
                    xytext=(noise_levels[peak_idx] / mean_dist + 0.1, no_bias_acc[peak_idx] + 0.05),
                    fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))

    ax2.axhline(y=1/6, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Noise level (× code spacing)')
    ax2.set_ylabel('Decoding accuracy')
    ax2.set_title('(b) Stochastic resonance curves', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # Panel (c): Dimensionality curves - THE KEY INSIGHT
    ax3 = fig.add_subplot(2, 3, 3)

    for name, dims in results_dim.items():
        lw = 3 if name == 'No bias' else 2.5
        ls = '-' if name != 'Wrong bias' else '--'
        ax3.plot(noise_levels / mean_dist, dims, ls, color=colors_line[name],
                label=name, linewidth=lw)

    # Mark optimal dimensionality point
    if 0 < peak_idx < len(results_dim['No bias']) - 1:
        ax3.axvline(x=noise_levels[peak_idx] / mean_dist, color='gray', linestyle=':', alpha=0.7)
        ax3.text(noise_levels[peak_idx] / mean_dist + 0.02, ax3.get_ylim()[1] * 0.9,
                'Optimal\nnoise', fontsize=9, color='gray')

    ax3.set_xlabel('Noise level (× code spacing)')
    ax3.set_ylabel('Effective dimensionality (PR)')
    ax3.set_title('(c) Noise expands dimensionality', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel (d): Trajectory visualization for one ambiguous signal
    ax4 = fig.add_subplot(2, 3, 4)

    # Run one trajectory at optimal noise
    optimal_noise = noise_levels[peak_idx]
    _, traj_nobias, _ = sr_with_dimensionality_tracking(
        test_signals[:1], code_centroids, optimal_noise,
        n_timesteps=15
    )
    _, traj_bias, _ = sr_with_dimensionality_tracking(
        test_signals[:1], code_centroids, optimal_noise,
        bias_toward=test_labels[:1], bias_strength=0.25,
        n_timesteps=15
    )

    # Plot background codes
    for i, c in enumerate(code_centroids):
        circle = plt.Circle((c[0], c[1]), mean_dist * 0.25, fill=True,
                           color=RAINBOW[i], alpha=0.2, zorder=1)
        ax4.add_patch(circle)
        ax4.scatter([c[0]], [c[1]], c=RAINBOW[i], s=150, zorder=5,
                   edgecolors='black', linewidths=1.5)

    # Plot trajectories
    traj_nb = traj_nobias[0]
    traj_b = traj_bias[0]

    ax4.plot(traj_nb[:, 0], traj_nb[:, 1], 'o-', color='#666666',
            markersize=6, linewidth=1.5, alpha=0.7, label='No bias')
    ax4.plot(traj_b[:, 0], traj_b[:, 1], 's-', color='#D55E00',
            markersize=6, linewidth=1.5, alpha=0.7, label='With bias')

    # Mark start
    ax4.scatter([test_signals[0, 0]], [test_signals[0, 1]],
               c='black', s=100, marker='x', linewidths=3, zorder=10)
    ax4.annotate('Start\n(low-D)', xy=(test_signals[0, 0], test_signals[0, 1]),
                xytext=(test_signals[0, 0] - 1, test_signals[0, 1] + 1),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))

    ax4.set_xlabel('Code dim 1')
    ax4.set_ylabel('Code dim 2')
    ax4.set_title('(d) Example trajectories\nNoise enables basin escape', fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_aspect('equal', adjustable='datalim')

    # Panel (e): Dimensionality over time within a trial
    ax5 = fig.add_subplot(2, 3, 5)

    # Get time-resolved dimensionality
    _, _, dim_traj_low = sr_with_dimensionality_tracking(
        test_signals, code_centroids, noise_levels[2],
        n_timesteps=12
    )
    _, _, dim_traj_opt = sr_with_dimensionality_tracking(
        test_signals, code_centroids, optimal_noise,
        n_timesteps=12
    )
    _, _, dim_traj_high = sr_with_dimensionality_tracking(
        test_signals, code_centroids, noise_levels[-3],
        n_timesteps=12
    )

    timesteps = np.arange(len(dim_traj_opt))
    ax5.plot(timesteps, dim_traj_low, 'o-', color='#999999', label=f'Low noise', linewidth=2)
    ax5.plot(timesteps, dim_traj_opt, 's-', color='#D55E00', label=f'Optimal noise', linewidth=2.5)
    ax5.plot(timesteps, dim_traj_high, '^-', color='#0072B2', label=f'High noise', linewidth=2)

    ax5.set_xlabel('Time step')
    ax5.set_ylabel('Effective dimensionality (PR)')
    ax5.set_title('(e) Dimensionality dynamics\nOptimal noise → controlled expansion', fontweight='bold')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Panel (f): Schematic of the mechanism
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_xlim(-2.5, 2.5)
    ax6.set_ylim(-0.5, 3)
    ax6.axis('off')
    ax6.set_title('(f) Mechanism: SR as dimensionality expansion', fontweight='bold')

    # Draw three scenarios
    scenarios = [
        (-1.7, 'Low noise', '#999999', 'Stuck on\nboundary\n(dim ≈ 1)'),
        (0, 'Optimal', '#D55E00', 'Expands to\nescape basin\n(dim ≈ 1.5)'),
        (1.7, 'High noise', '#0072B2', 'Random walk\n(dim → 2)'),
    ]

    for x, label, color, desc in scenarios:
        # Draw two basins
        ax6.plot([x-0.4, x], [0.8, 0.3], '-', color=RAINBOW[0], linewidth=3, alpha=0.5)
        ax6.plot([x, x+0.4], [0.3, 0.8], '-', color=RAINBOW[1], linewidth=3, alpha=0.5)

        # Ball position
        if label == 'Low noise':
            ball_pos = (x, 0.35)
            ax6.annotate('', xy=(x-0.1, 0.35), xytext=(x+0.1, 0.35),
                        arrowprops=dict(arrowstyle='<->', color=color, lw=1))
        elif label == 'Optimal':
            ball_pos = (x-0.25, 0.5)
            # Show expansion
            ax6.annotate('', xy=(x-0.35, 0.45), xytext=(x, 0.35),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
        else:
            ball_pos = (x+0.1, 1.0)
            # Show random expansion
            for angle in [0, 60, 120, 180, 240, 300]:
                dx = 0.15 * np.cos(np.radians(angle))
                dy = 0.15 * np.sin(np.radians(angle))
                ax6.annotate('', xy=(x + dx, 0.6 + dy), xytext=(x, 0.5),
                            arrowprops=dict(arrowstyle='->', color=color, lw=1, alpha=0.5))

        ax6.scatter([ball_pos[0]], [ball_pos[1]], c='black', s=80, zorder=10)
        ax6.text(x, 2.2, label, ha='center', fontsize=10, fontweight='bold', color=color)
        ax6.text(x, 1.5, desc, ha='center', fontsize=8, color='gray')

    ax6.text(0, -0.3, 'Noise increases dimensionality → enables escape from low-D boundary',
            ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, 'stochastic_resonance.png'), dpi=150, bbox_inches='tight')
    print(f"\nSaved to output/stochastic_resonance.png")
    plt.show()

    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    no_bias_acc = results_acc['No bias']
    peak_idx = np.argmax(no_bias_acc)
    baseline = no_bias_acc[0]
    peak_acc = no_bias_acc[peak_idx]

    print(f"\n• Without noise: {baseline:.1%} accuracy (trapped on boundary)")
    print(f"• At SR peak: {peak_acc:.1%} accuracy")
    print(f"• Improvement: +{(peak_acc - baseline)*100:.1f} percentage points")

    print(f"\n• Initial dimensionality: {initial_dim:.2f} (signals on 1D boundary)")
    print(f"• Dimensionality at SR peak: {results_dim['No bias'][peak_idx]:.2f}")
    print(f"• Dimensionality at high noise: {results_dim['No bias'][-1]:.2f}")

    correct_bias_acc = results_acc['Correct bias'][peak_idx]
    print(f"\n• With correct bias: {correct_bias_acc:.1%} accuracy")
    print(f"• Bias adds: +{(correct_bias_acc - peak_acc)*100:.1f} percentage points")

    print("\n" + "=" * 70)
    print("THE DIMENSIONALITY INTERPRETATION")
    print("=" * 70)
    print("""
Why does stochastic resonance work?

1. AMBIGUOUS SIGNALS ARE LOW-DIMENSIONAL
   - They sit on the boundary between attractor basins
   - Effectively 1D: constrained to the decision boundary

2. NOISE INCREASES DIMENSIONALITY
   - Random fluctuations expand the representation
   - The system can now "explore" perpendicular to the boundary
   - This enables escape into the correct basin

3. OPTIMAL NOISE = OPTIMAL DIMENSIONALITY
   - Too little: stays trapped on 1D boundary
   - Just right: expands enough to find correct basin
   - Too much: expands into random wandering (2D chaos)

4. EXPECTATION BIAS = DIRECTED EXPANSION
   - Bias adds dimensionality in a specific direction
   - Points toward the expected attractor
   - Reduces the noise needed for correct decoding

This connects code formation → SR → predictive processing:
   Codes create basins → Noise explores basins → Predictions bias exploration
""")
    print("=" * 70)


if __name__ == "__main__":
    run_sr_experiment()
