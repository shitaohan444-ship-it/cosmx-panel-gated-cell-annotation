# Cell-Level Gated Fusion of Shared and CosMx Panel-Specific Gene Representations for Cell-Type Annotation across Heterogeneous Imaging-Based Spatial Transcriptomic Panels

Author: hanshitao (Han Shitao)

## Project Overview

This project studies cell-type annotation across heterogeneous
imaging-based spatial transcriptomic panels. M1 uses dual-branch
representations — shared genes and CosMx panel-specific genes — fused by a
cell-level gate. The shared genes provide cross-platform cell-identity
information, while the panel-specific genes supplement cell-type-related
information; the cell-level gate adaptively balances the two per cell.

## Key Results

On the CosMx 2023 multi-tissue tTMA1 benchmark (37 patients, 169,283 cells,
patient-level three-fold cross-validation, five seeds):

- Accuracy: **0.8993 ± 0.0052** (equal-fold mean)
- Macro-F1: **0.8168 ± 0.0148** (fold mean)
- Patient-level paired accuracy difference: M1 vs M1-Fixed **+0.5204 pp**;
  M1 vs M1-Global **+0.4262 pp**
- Note: the full-panel linear reference (LR-A) is slightly more accurate
  than M1 (about +0.25 pp); M1 is not claimed to outperform all baselines.

## Selected Experimental Results

Three representative aggregated result files are published on the
`phase395-selected-results-release` branch:

- [Model Performance](https://github.com/shitaohan444-ship-it/cosmx-panel-gated-cell-annotation/blob/phase395-selected-results-release/public_results/model_performance.csv) — closed-set accuracy, macro-F1, ECE, and Brier of M1 and the internal baselines.
- [Gating Mechanism Comparison](https://github.com/shitaohan444-ship-it/cosmx-panel-gated-cell-annotation/blob/phase395-selected-results-release/public_results/gating_mechanism_comparison.csv) — patient-level paired statistics of the retrained fixed-gate and global-gate controls.
- [Biological Enrichment](https://github.com/shitaohan444-ship-it/cosmx-panel-gated-cell-annotation/blob/phase395-selected-results-release/public_results/biological_enrichment.csv) — complete GO/KEGG/Reactome enrichment table computed after removing the 33 canonical label-defining markers.

## Environment

- Python 3.11 (see [`environment.yml`](environment.yml))
- PyTorch ≥ 2.1 (see [`requirements.txt`](requirements.txt); verified with
  2.9.1 for CPU tests and 2.14.1+cu130 on one RTX 3090)
- CUDA: optional — CPU suffices for the included tests

```bash
conda env create -f environment.yml
conda activate cosmx-gated
python tests/test_models.py
```

Model weights, training checkpoints, and raw datasets are not included in
this repository.

## Citation

Author: hanshitao (Han Shitao)

Citation files: [`CITATION_PENDING.md`](CITATION_PENDING.md) — the
manuscript is not yet formally published, so no DOI or publication metadata
is provided.
