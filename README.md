# Cell-Level Gated Fusion of Shared and CosMx Panel-Specific Gene Representations for Cell-Type Annotation across Heterogeneous Imaging-Based Spatial Transcriptomic Panels

Author: hanshitao (Han Shitao)

## Project Overview

This project studies cell-type annotation for imaging-based spatial
transcriptomics across heterogeneous gene panels. M1 encodes the shared
genes and the CosMx panel-specific genes in two separate branches and fuses
their representations with a cell-level adaptive gate. The shared genes
provide a common feature space that transfers across platforms, while the
panel-specific genes supply complementary cell-type-related information;
the gate adaptively balances the two sources based on each cell's own
representations. Cross-platform analyses on the target platforms are
indirect (gate behavior, composition, and region-level biological
concordance): no per-cell accuracy on target platforms is claimed, because
no per-cell ground truth exists there.

## Model Architecture

![M1 Model Architecture](assets/m1_architecture.png)

- **Shared branch**: 108 shared genes → MLP → 128-d representation
- **Panel-specific branch**: 892 CosMx-specific genes → MLP → 128-d
  representation
- **Gate**: concatenated representations → MLP → sigmoid → **one scalar per
  cell** (cell-level scalar gating, not feature-wise gating)
- **Fusion**: z = g × z_shared + (1 − g) × z_specific
- **Fine head**: 8 classes; **Coarse head**: 6 classes

## Key Results

On the CosMx 2023 multi-tissue tTMA1 benchmark (37 patients, 169,283 cells,
patient-level three-fold cross-validation):

- Accuracy: **0.8993 ± 0.0052** (equal-fold mean of the three test folds,
  ± sample SD across folds; pooled 0.8992)
- Macro-F1: **0.8168 ± 0.0148** (fold-mean macro-F1, ± sample SD across
  folds; pooled 0.8703)
- Patient-level paired accuracy difference: M1 vs M1-Fixed **+0.5204 pp**
  (p = 1.1e-5); M1 vs M1-Global **+0.4262 pp** (p = 4.7e-5)
- The full-panel linear reference (LR-A) is slightly more accurate than M1
  (about +0.25 pp); M1 is not claimed to outperform all baselines.

## Selected Experimental Results

- [public_results/model_performance.csv](public_results/model_performance.csv) — closed-set accuracy, macro-F1, ECE, and Brier of M1 and the internal baselines.
- [public_results/gating_mechanism_comparison.csv](public_results/gating_mechanism_comparison.csv) — patient-level paired statistics of the retrained fixed-gate and global-gate controls.
- [public_results/biological_enrichment.csv](public_results/biological_enrichment.csv) — complete GO/KEGG/Reactome enrichment table after removing the 33 canonical label-defining markers.

See [public_results/README.md](public_results/README.md) for file descriptions and the original canonical file names.

## Environment

- Python 3.11
- PyTorch ≥ 2.1 (verified: 2.9.1 for CPU tests; 2.14.1+cu130 on one
  RTX 3090 for training)
- CUDA: optional — CPU suffices for the included tests

```bash
conda env create -f environment.yml
conda activate cosmx-gated
python tests/test_models.py
```

Model weights, training checkpoints, and raw datasets are not included in
this repository.

## Citation and License

Author: hanshitao (Han Shitao). The manuscript is not yet formally
published; no DOI or publication metadata is provided. See
[CITATION_PENDING.md](CITATION_PENDING.md).

License: [LICENSE_PENDING.md](LICENSE_PENDING.md) — the repository is
publicly viewable, but no open-source license has been selected yet; all
rights are reserved by default, and public access does not grant permission
to use, copy, modify, or redistribute the code.
