# cosmx-panel-gated-cell-annotation

Code for **cell-level gated fusion of shared and CosMx panel-specific gene
representations for cell-type annotation across heterogeneous imaging-based
spatial transcriptomic panels**.

> **Status**: private repository. The manuscript is under review preparation;
> this repository will remain private until the authors approve public
> release and a license is chosen (see `LICENSE_PENDING.md`).

## Project Overview

Imaging-based spatial transcriptomics platforms (CosMx, MERSCOPE, Xenium)
measure different gene panels. In the multi-platform FFPE benchmark used in
our study, only 108 of the 1,000 CosMx genes are shared across all three
panels. This repository contains the reference implementation of **M1**, a
dual-branch annotator that encodes the 108 shared genes and the 892
CosMx panel-specific genes in separate branches and fuses their 128-d
representations with a **cell-level adaptive scalar gate**.

## Research Objective

Test whether a cell-level adaptive gate provides a measurable annotation
benefit over two retrained controls that differ *only* in the gating
mechanism: **M1-Fixed** (g = 0.5 for every cell) and **M1-Global** (one
globally learned scalar g = sigmoid(alpha), which converges to ≈0.493).

## Method Overview / Model Architecture

```
x_shared (108)  -> MLP(108->256->128, BN+ReLU) -> z_sh
x_specific(892) -> MLP(892->256->128, BN+ReLU) -> z_sp
g = sigmoid(MLP([z_sh; z_sp] -> 128 -> 1))  in (0,1)^{B x 1}   (M1)
z = g * z_sh + (1 - g) * z_sp
fine head: Linear(128 -> 8)   coarse head: Linear(128 -> 6)
```

| Model | Gating mechanism | Trainable gate params |
|---|---|---|
| M1 (`src/models/m1.py`) | per-cell scalar gate, broadcast across 128 dims | 33,025 |
| M1-Fixed (`src/models/m1_fixed.py`) | g = 0.5 constant (no gate network) | 0 |
| M1-Global (`src/models/m1_global.py`) | one global scalar g = sigmoid(alpha) | 1 |

Training protocol (identical for all three models): log1p input, 15 epochs,
Adam (lr 1e-3, wd 1e-4), batch 1,024, grad clip 5.0, loss =
CE(fine) + CE(coarse), 20% panel-specific dropout during training, patient-
level three-fold CV with five pre-specified seeds {42, 7, 99, 123, 2024},
fixed final-epoch checkpoint (no early stopping).

## Data Requirements

**No data ships with this repository.** The experiments use the public
multi-platform FFPE benchmark (Wang et al., 2025; Zenodo 16848917, CC-BY-4.0)
and external annotations (Zenodo 15411357; ExperimentHub EH8567). Users must
obtain these datasets through their official channels.

For training/evaluation, provide a preprocessed `.npz` with the schema
documented in `src/data.py` (X_shared 108 / X_specific 892 / y_fine / y_coarse /
folds / core_arr), matching the in-house `build_data5` outputs.

## Installation / Environment

```bash
conda env create -f environment.yml
conda activate cosmx-gated
pip install -r requirements.txt   # if not using conda
```

Python 3.11; PyTorch >= 2.1 (CPU suffices for tests; CUDA optional).

## Training

```bash
python scripts/train.py --config configs/experiment_config.example.yaml \
    --model m1 --seed 42 --fold 1 --out-dir runs/m1_s42
```

See `configs/experiment_config.example.yaml` for path templates (replace
placeholders with your local layout).

## Evaluation / Statistical Analysis / Calibration

```bash
python scripts/counterfactual.py --config ... --seed 42 --fold 1 \
    --ckpt runs/m1_s42/model_fold1.pt --out counterfactual.csv
python scripts/patient_stats.py --config ... --out patient_stats.csv
python scripts/calibration.py --config ... --out calibration.csv
```

Primary statistical estimand: per-patient accuracy averaged over the five
seeds, then equal-weight mean over 37 patients; two-sided Wilcoxon
signed-rank; patient-clustered bootstrap 95% CI (B = 2,000, rng 20260331);
Cohen's dz; BH over the two primary comparisons.

## Biological Interpretation / Figure Reproduction

The frozen attribution (Integrated Gradients), enrichment, spatial, and
cross-platform analysis scripts from the original runs are provided verbatim
under `run_code_reference/` (they retain the original hard-coded server
paths and are intended as provenance, not as turn-key reproductions); the
config-driven equivalents for the core mechanism analyses live in
`scripts/`. Figure scripts that generated the manuscript figures are in
`run_code_reference/charts/`.

## Reproducibility

- Model and metric code in `src/` is byte-identical to the run code
  (except the documented import localizations; see
  `docs/DIFFERENCES_FROM_RUN_CODE.md`).
- `tests/test_models.py` verifies the parameter counts, gate behavior and
  training step on CPU with synthetic inputs.
- Original experiment artifacts (predictions, checkpoints) are **not**
  included in this repository.

## Citation

Citation information is pending the manuscript publication; see
`CITATION_PENDING.md`. Until then, please cite the manuscript preprint once
available, and contact the authors for further information.

## License

No license has been selected yet; see `LICENSE_PENDING.md`. Do not assume
an open-source license applies.
