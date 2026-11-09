# Method Overview

M1 (manuscript Section 3.3-3.5 / Methods 4.4-4.6):

- Two branches encode the 108 shared genes and the 892 CosMx panel-specific
  genes into 128-d representations (two-layer MLPs with BatchNorm + ReLU,
  log1p input).
- A cell-level scalar gate `g = sigmoid(MLP([z_sh; z_sp]))` in `[B,1]` is
  broadcast across the 128 dimensions (scalar per cell, not per dimension).
- Fusion `z = g * z_sh + (1-g) * z_sp` is a per-cell convex combination.
- Fine (8-class) and coarse (6-class) linear heads share `z`; training loss
  is the equal-weight sum of the two cross-entropy terms.
- During training, the whole panel-specific input is zeroed for 20% of cells
  (matching the cross-platform zeroing at inference).

Retrained controls (manuscript Methods 4.6 / Results 5.5):

- M1-Fixed: same architecture, gate replaced by the constant 0.5.
- M1-Global: same architecture, gate replaced by one global scalar
  g = sigmoid(alpha) shared by all cells; it converges to ≈0.493.

Key results (see the manuscript for full reporting):

- Patient-level paired accuracy difference (per-patient mean over five seeds,
  then equal-weight over 37 patients): M1 vs M1-Fixed +0.52 pp
  (95% CI +0.29 to +0.79; p = 1.1e-5); M1 vs M1-Global +0.43 pp
  (95% CI +0.18 to +0.69; p = 4.7e-5).
- Cross-platform evidence is indirect (gate behavior, composition, regional
  lineage concordance); no per-cell ground truth exists on the target
  platforms.

# Experiment Protocol

- Data: CosMx 2023 multi-tissue tTMA1 (150 cores, 37 patients, 169,283 cells;
  RNA_nbclust candidate labels, not expert gold standard).
- Split: patient-level three-fold CV; within-fold 20% core validation split
  (logging only, RandomState(42)).
- Training: log1p, 15 epochs, Adam (lr 1e-3, wd 1e-4), batch 1,024,
  grad clip 5.0, 20% panel-specific dropout, final-epoch checkpoint,
  five pre-specified seeds {42, 7, 99, 123, 2024}; one process per
  (seed, fold).
- Evaluation: equal-fold and pooled conventions reported separately;
  sample and population SDs reported; ECE (15 equal-width bins, no
  temperature scaling); multi-class Brier.
- Statistics: exploratory (not pre-registered); the seed-averaged patient
  estimand was defined after the initial analyses and before the retrained
  controls; Wilcoxon signed-rank; patient-clustered bootstrap B=2,000
  (rng 20260331); Cohen's dz; BH across the two primary comparisons.
