# Extensions

Run from the main `code_formation_demo/` directory:

```bash
python extensions/what_gets_preserved.py
python extensions/expectation_biased_sr.py
```

Output goes to `output/`.

## 1. What Gets Preserved?

How does the network decide what to keep when reducing 512-D → k-D?

**Answer**: It preserves high-variance directions that survive noise (≈ top PCs).

## 2. Stochastic Resonance

**Key insight**: Noise increases dimensionality.

Ambiguous signals sit on low-D boundaries between attractor basins. Noise expands the representation, enabling escape to the correct basin.
