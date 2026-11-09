# Reproduction Guide

This repository contains the model/metric code and the config-driven
analysis scripts. Full reproduction of the manuscript numbers additionally
requires the public datasets and (for exact replication of every table) the
original experiment artifacts (predictions/checkpoints), which are not
shipped here.

## 1. Data acquisition (official channels)

| Data | Source |
|---|---|
| Multi-platform FFPE benchmark (CosMx/Xenium/MERSCOPE tTMA data) | Wang et al., 2025; Zenodo record 16848917 (CC-BY-4.0) |
| Pathologist region annotations (IDC/ILC) | Zenodo record 15411357 (CC-BY-4.0) |
| Bhuva expression data | Bioconductor ExperimentHub EH8567 |

Users must comply with the terms of these records. This repository includes
no data, no patient-level information, and no predictions.

## 2. Preprocessing

Export a preprocessed `.npz` with the schema in `src/data.py`
(X_shared/X_specific/y_fine/y_coarse/folds/core_arr). The original pipeline
(build_data5) reads the CosMx h5ad plus the metadata manifests
(sceneA_final_manifest.csv, core_split_manifest.csv, celltype_mapping.tsv,
common_genes.tsv); a verbatim copy of that pipeline is available in
`run_code_reference/train5.py`.

## 3. Model checks (CPU, no data)

```bash
python tests/test_models.py
```

Verifies parameter counts (358,671 / 325,646 / 325,647), gate behavior and
a training step.

## 4. Training

```bash
python scripts/train.py --config configs/experiment_config.example.yaml \
  --model m1 --seed 42 --fold 1 --out-dir runs/m1_s42
```

Repeat for models m1_fixed / m1_global and all (seed, fold) pairs.

## 5. Analyses

- Counterfactual interventions: `scripts/counterfactual.py`
- Patient-level paired statistics: `scripts/patient_stats.py`
- Calibration: `scripts/calibration.py`
- Provenance scripts (verbatim, hard-coded server paths):
  `run_code_reference/` — attribution_ig, enrichment, spatial, crossplatform,
  charts, and the original training scripts.

## 6. Limitations of this release

- Reference scripts under `run_code_reference/` retain original absolute
  paths and are provided for provenance, not turn-key execution.
- The manuscript figures were produced with the chart scripts referenced in
  the figure QC reports (not shipped here); figure data CSVs are available
  from the authors' manuscript package on request.
