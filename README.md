# CSL7870-Task8-Mobius-Kantor

Codebase for (Optional) Task 8

## Overview

`Exp_A.py` conducts an enhanced Weisfeiler–Lehman (WL) hash analysis on the TUDataset AIDS graphs to study robustness under single-edge perturbations. It identifies isomorphic groups, perturbs graphs (add/remove one edge), recomputes WL hashes, and summarizes how often isomorphism is preserved.

## Key Features

- WL hashing–based grouping of isomorphic graphs (PyTorch Geometric → NetworkX conversion).
- Noise robustness experiment with controlled single-edge perturbations per graph.
- Directed-graph safe connectivity analysis (uses weak connectivity for directed graphs).
- Publication-grade figures saved to disk (matplotlib only), no interactive windows.

## Implementation Notes

- WL Hash: Iterative color refinement using node labels (`data.x`) as initial colors.
- Perturbations: Exactly one edge is added or removed per perturbed copy when possible.
- Connectivity: Calls are robust to directed graphs using `nx.is_weakly_connected(g)` when applicable.
- Reproducibility: `random`, `numpy`, and `torch` seeds are fixed at script entry.

## Dataset

TUDataset AIDS is fetched to the default root (`/tmp/AIDS`). Ensure internet access for the initial run.
