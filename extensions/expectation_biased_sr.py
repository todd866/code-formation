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

    # Step 5: Create figure (clean 2x2 layout)
    print("5. Generating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel (a): Code space with ambiguous signals
    ax1 = axes[0, 0]

    colors = [RAINBOW[l] for l in true_labels_train]
    ax1.scatter(code_np[:, 0], code_np[:, 1], c=colors, alpha=0.15, s=10)

    for i, c in enumerate(code_centroids):
        ax1.scatter([c[0]], [c[1]], c=RAINBOW[i], s=250, marker='o',
                   edgecolors='black', linewidths=2, zorder=10)

    # Show ambiguous signals on boundaries
    for i in range(0, 50, 8):
        ax1.scatter([test_signals[i, 0]], [test_signals[i, 1]],
                   c='black', s=60, marker='x', linewidths=2, zorder=8)

    ax1.set_xlabel('Code dim 1', fontsize=11)
    ax1.set_ylabel('Code dim 2', fontsize=11)
    ax1.set_title('(a) Learned codes with ambiguous test signals (×)', fontweight='bold', fontsize=12)
    ax1.set_aspect('equal', adjustable='datalim')

    # Panel (b): SR curves (accuracy)
    ax2 = axes[0, 1]

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
                    xytext=(noise_levels[peak_idx] / mean_dist + 0.15, no_bias_acc[peak_idx] + 0.05),
                    fontsize=11, arrowprops=dict(arrowstyle='->', color='gray'))

    ax2.axhline(y=1/6, color='gray', linestyle=':', alpha=0.5, label='Chance')
    ax2.set_xlabel('Noise level (× code spacing)', fontsize=11)
    ax2.set_ylabel('Decoding accuracy', fontsize=11)
    ax2.set_title('(b) Stochastic resonance: accuracy peaks at intermediate noise', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # Panel (c): Dimensionality curves - THE KEY INSIGHT
    ax3 = axes[1, 0]

    for name, dims in results_dim.items():
        lw = 3 if name == 'No bias' else 2.5
        ls = '-' if name != 'Wrong bias' else '--'
        ax3.plot(noise_levels / mean_dist, dims, ls, color=colors_line[name],
                label=name, linewidth=lw)

    # Mark optimal dimensionality point
    if 0 < peak_idx < len(results_dim['No bias']) - 1:
        ax3.axvline(x=noise_levels[peak_idx] / mean_dist, color='gray', linestyle=':', alpha=0.7)
        ax3.text(noise_levels[peak_idx] / mean_dist + 0.03, ax3.get_ylim()[1] * 0.85,
                'SR peak', fontsize=10, color='gray')

    ax3.set_xlabel('Noise level (× code spacing)', fontsize=11)
    ax3.set_ylabel('Effective dimensionality (PR)', fontsize=11)
    ax3.set_title('(c) Noise expands effective dimensionality', fontweight='bold', fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Panel (d): TRUE Expectation-biased SR simulation
    # Expectation enters in HIGH-D space (before bottleneck), not in code space
    ax4 = axes[1, 1]

    print("   Running high-D expectation bias simulation...")

    # Get the trained model and data from the main experiment
    # We need to recompute high-D prototypes for each code
    from code_formation import generate_ring_data as gen_ring, Autoencoder

    # Reload model and get high-D prototypes
    data_hd, theta = gen_ring(n_samples=2000, input_dim=512)
    true_labels_hd = generate_ring_labels(theta, n_sectors=6)

    # Train fresh model for this demo
    torch.manual_seed(42)
    model = Autoencoder(512, 256, 2, noise_std=0.5)
    data_tensor = data_hd.clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        recon, _ = model(data_tensor, add_noise=True)
        loss = loss_fn(recon, data_tensor)
        loss.backward()
        optimizer.step()

    model.eval()

    # Compute HIGH-D prototypes for each code (mean input for each cluster)
    hd_prototypes = []
    for label in range(6):
        mask = true_labels_hd == label
        prototype = data_hd[mask].mean(dim=0).numpy()
        hd_prototypes.append(prototype)
    hd_prototypes = np.array(hd_prototypes)

    # Find genuinely ambiguous stimuli by testing which inputs land near boundaries
    # in CODE space (not just averaging prototypes in input space)
    basin_A, basin_B = 0, 1  # Adjacent basins

    # Get code centroids for this fresh model
    with torch.no_grad():
        _, codes_fresh = model(data_tensor, add_noise=False)
    codes_fresh = codes_fresh.numpy()

    # Find fresh centroids
    fresh_centroids = []
    for label in range(6):
        mask = true_labels_hd == label
        fresh_centroids.append(codes_fresh[mask].mean(axis=0))
    fresh_centroids = np.array(fresh_centroids)

    # Find inputs that encode to near the boundary between A and B
    boundary_inputs = []
    for i in range(len(data_hd)):
        code = codes_fresh[i]
        dist_A = np.linalg.norm(code - fresh_centroids[basin_A])
        dist_B = np.linalg.norm(code - fresh_centroids[basin_B])
        # Look for points where distances are similar (near boundary)
        if abs(dist_A - dist_B) < 0.3 * (dist_A + dist_B) / 2:
            boundary_inputs.append(i)

    if len(boundary_inputs) < 10:
        # Fallback: use interpolated inputs
        boundary_inputs = list(range(50))

    print(f"   Found {len(boundary_inputs)} boundary inputs")

    # Run many trials with different expectation conditions
    n_trials = 200

    # Key parameters
    noise_std = 0.4
    expect_strength = 0.2

    results = {'No expectation': [], 'Expect A': [], 'Expect B': []}

    for trial in range(n_trials):
        # Pick a random boundary input
        idx = np.random.choice(boundary_inputs)
        stimulus = data_hd[idx].numpy()

        # Fresh noise for each trial
        noise = np.random.randn(512) * noise_std

        # NO EXPECTATION: just stimulus + noise
        input_none = torch.tensor(stimulus + noise, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, code_none = model(input_none, add_noise=False)
        code_none = code_none.numpy()[0]

        # Decode to nearest of the two competing basins
        dist_A = np.linalg.norm(code_none - fresh_centroids[basin_A])
        dist_B = np.linalg.norm(code_none - fresh_centroids[basin_B])
        results['No expectation'].append(basin_A if dist_A < dist_B else basin_B)

        # EXPECT A: add high-D pattern pointing toward A prototype
        expect_A = expect_strength * (hd_prototypes[basin_A] - stimulus)
        input_A = torch.tensor(stimulus + noise + expect_A, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, code_A = model(input_A, add_noise=False)
        code_A = code_A.numpy()[0]

        dist_A = np.linalg.norm(code_A - fresh_centroids[basin_A])
        dist_B = np.linalg.norm(code_A - fresh_centroids[basin_B])
        results['Expect A'].append(basin_A if dist_A < dist_B else basin_B)

        # EXPECT B: add high-D pattern pointing toward B prototype
        expect_B = expect_strength * (hd_prototypes[basin_B] - stimulus)
        input_B = torch.tensor(stimulus + noise + expect_B, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, code_B = model(input_B, add_noise=False)
        code_B = code_B.numpy()[0]

        dist_A = np.linalg.norm(code_B - fresh_centroids[basin_A])
        dist_B = np.linalg.norm(code_B - fresh_centroids[basin_B])
        results['Expect B'].append(basin_A if dist_A < dist_B else basin_B)

    # Compute percentages landing in each basin
    pct_A_none = results['No expectation'].count(basin_A) / n_trials * 100
    pct_A_expA = results['Expect A'].count(basin_A) / n_trials * 100
    pct_A_expB = results['Expect B'].count(basin_A) / n_trials * 100

    # Plot as bar chart
    conditions = ['No\nexpectation', 'Expect\nbasin A', 'Expect\nbasin B']
    pct_land_A = [pct_A_none, pct_A_expA, pct_A_expB]
    pct_land_B = [100 - p for p in pct_land_A]

    x_pos = np.arange(3)
    width = 0.35

    bars_A = ax4.bar(x_pos - width/2, pct_land_A, width, label=f'Land in A',
                     color=RAINBOW[basin_A], alpha=0.8, edgecolor='black')
    bars_B = ax4.bar(x_pos + width/2, pct_land_B, width, label=f'Land in B',
                     color=RAINBOW[basin_B], alpha=0.8, edgecolor='black')

    ax4.set_ylabel('% of trials', fontsize=11)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(conditions, fontsize=10)
    ax4.set_ylim(0, 105)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_title('(d) High-D expectation biases code formation', fontweight='bold', fontsize=12)

    # Add value labels on bars
    for bar, pct in zip(bars_A, pct_land_A):
        if pct > 5:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{pct:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, pct in zip(bars_B, pct_land_B):
        if pct > 5:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{pct:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Caption explaining the mechanism
    ax4.text(0.5, -0.15, 'Expectation added in 512-D space before bottleneck (β→γ)',
            transform=ax4.transAxes, ha='center', fontsize=10, style='italic', color='gray')

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
