# Experiment Protocol (reference)

See docs/method_overview.md for the concise protocol. This page records the
exact parameters used in the reported experiments.

| Parameter | Value |
|---|---|
| Shared / specific genes | 108 / 892 (common_genes.tsv of the benchmark) |
| Fine / coarse classes | 8 / 6 |
| Hidden / latent dims | 256 / 128 |
| Input transform | log1p |
| Epochs | 15 (fixed; final checkpoint; no early stopping) |
| Optimizer | Adam lr=1e-3, wd=1e-4 |
| Batch size | 1,024 |
| Gradient clip | max norm 5.0 |
| Loss | CE(fine) + CE(coarse) |
| Panel-specific dropout | 20% of cells, training only |
| Folds | patient-level 3-fold (12/13/12 patients) |
| Seeds | {42, 7, 99, 123, 2024}, one process per (seed, fold) |
| Fit/val core split | 20% of fit cores, RandomState(42), logging only |
| GPU | one RTX 3090; peak ~0.7 GB; 13-19 s per fold (free GPU) |
| ECE | 15 equal-width bins, no temperature scaling |
| Brier | multi-class, per-cell sum of squared errors averaged |
| Patient estimand | per-patient accuracy, seed-averaged, 37-patient equal mean |
| Bootstrap | patient-clustered, B=2,000, rng seed 20260331 |
| Multiple testing | BH; families stated per analysis |
