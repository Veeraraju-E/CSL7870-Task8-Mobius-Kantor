# CSL7870-Task8-Mobius-Kantor

Codebase for (Optional) Task 8

## Overview

**Exp_A.py**: Enhanced Weisfeiler–Lehman (WL) hash analysis on TUDataset AIDS graphs to study robustness under single-edge perturbations. Identifies isomorphic groups, perturbs graphs (add/remove one edge), recomputes WL hashes, and summarizes isomorphism preservation rates.

**Exp_B.ipynb**: Jupyter notebook that validates WL-based isomorphic group detection by visualizing graph structures and verifying true isomorphism using NetworkX's GraphMatcher. Analyzes structural properties beyond WL test (degree sequences, triangles, cycles).

## Key Features

**Exp_A.py**:

- WL hashing–based grouping of isomorphic graphs (PyTorch Geometric → NetworkX conversion).
- Noise robustness experiment with controlled single-edge perturbations per graph.
- Directed-graph safe connectivity analysis (uses weak connectivity for directed graphs).
- Publication-grade figures saved to disk (matplotlib only), no interactive windows.

**Exp_B.ipynb**:

- Interactive visualization of isomorphic groups as network graphs.
- True isomorphism verification using NetworkX GraphMatcher.
- Structural property analysis beyond WL test (degree sequences, triangles, cycles).

## Implementation Notes

- WL Hash: Iterative color refinement using node labels (`data.x`) as initial colors.
- Perturbations: Exactly one edge is added or removed per perturbed copy when possible.
- Connectivity: Calls are robust to directed graphs using `nx.is_weakly_connected(g)` when applicable.
- Reproducibility: `random`, `numpy`, and `torch` seeds are fixed at script entry.

## Running the Experiments

**Exp_A.py**:

```
python Exp_A.py
```

Downloads/loads AIDS dataset and executes the noise robustness pipeline end-to-end.

**Exp_B.ipynb**:
Open in Jupyter notebook and run cells sequentially. Provides interactive analysis and visualization of isomorphic groups.

## Outputs

**Exp_A.py**: Figures saved under `figures/` (300 DPI):

- `overview.png`: Multi-panel summary (maintenance rate, operation analysis, distributions, connectivity).
- `hash_similarity.png`: Heatmap of WL-hash equivalence among original and perturbed graphs.

**Exp_B.ipynb**: Interactive visualizations of isomorphic groups and structural property analysis.

## Dataset

TUDataset AIDS is fetched to the default root (`/tmp/AIDS`). Ensure internet access for the initial run.
