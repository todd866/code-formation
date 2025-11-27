# Code Formation Demo

Demonstrates that discrete codes emerge spontaneously when continuous high-dimensional data passes through a bandwidth-limited **noisy** channel.

## The Key Insight: You Need BOTH

Clustering requires **two ingredients working together**:

| Condition | Bottleneck | Noise | Result |
|-----------|:----------:|:-----:|--------|
| k=2, σ=0 | ✓ | ✗ | Continuous ring preserved |
| k=32, σ=0.5 | ✗ | ✓ | Continuous ring preserved |
| k=2, σ=0.5 | ✓ | ✓ | **Discrete clusters emerge!** |

Neither alone is sufficient. The **interaction** of bottleneck and noise causes clustering.

## Why Both Are Required

**BOTTLENECK (k)** - sets the geometry:
- Compresses high-D data into k dimensions
- Small k → points get crowded together
- Large k → plenty of room to spread out

**NOISE (σ)** - sets the grain size:
- Smears each code point into an uncertainty sphere
- Nearby points become indistinguishable
- Information capacity: C ≈ k · log(1 + SNR)

**THE MECHANISM:**
1. Bottleneck compresses the space → points get closer
2. Noise adds uncertainty → nearby points collide/overlap
3. Overlapping codes = confusion = information loss
4. Network learns to space codes apart → discrete clusters
5. Gap between clusters = safety margin against noise

This is a **topological phase transition**: continuous O(2) symmetry → discrete C₆ symmetry, driven by the interaction of compression and noise.

## Quick Start

```bash
pip install torch numpy matplotlib scikit-learn
python code_formation.py
```

## Output

The script generates `output/codes_by_channel_dim.png` showing:
- **Bottleneck only** (k=2, σ=0): Ring stays continuous
- **Noise only** (k=32, σ=0.5): Ring stays continuous
- **Both** (k=2, σ=0.5): Discrete clusters emerge
- **Explanation panel**: Why both ingredients are needed

## Files

```
code_formation_demo/
├── code_formation.py      # Main simulation with control experiment
├── output/                # Generated figures
└── extensions/
    ├── what_gets_preserved.py    # Shows encoder learns top PCs
    └── expectation_biased_sr.py  # SR as dimensionality expansion
```

## Technical Notes

- **Nonlinearity matters**: ReLU activations allow clustering. Linear would just be PCA.
- **Noise = Temperature**: Higher σ forces sharper phase transitions.
- **ARI metric**: Measures cluster alignment with 6-sector division. Network doesn't see these labels during training.
