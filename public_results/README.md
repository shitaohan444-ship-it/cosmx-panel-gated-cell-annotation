# Selected Experimental Results

Three representative result files from the manuscript experiments.
Full provenance (original paths and SHA256) is recorded in the Phase 3.9
publication-policy package; the original canonical file names are listed
below.

## 1. model_performance.csv

- **Content**: closed-set accuracy of M1, LR-A, LR-B, LR-C, M8, and M9 —
  per-fold accuracy, equal-fold mean with sample/population SD, pooled
  accuracy, fold-mean and pooled macro-F1, balanced accuracy, ECE (15
  equal-width bins), and multi-class Brier.
- **Key results**: M1 accuracy 0.8993 ± 0.0052 (equal-fold); macro-F1
  0.8168 ± 0.0148; LR-A exceeds M1 by about 0.25 pp (reported as-is).
- **Paper location**: Table 2 / Figure 3 / Results 5.2.
- **Original file**: lr_variants_recompute.csv (Phase 3.1, 01_Numerical_Repair).

## 2. gating_mechanism_comparison.csv

- **Content**: patient-level paired statistics of the retrained gating
  controls: per-patient accuracy averaged over five seeds then averaged
  over 37 patients; two-sided Wilcoxon p; Cohen's dz; patient-clustered
  bootstrap 95% CI (B = 2,000); per-seed values; BH across the two
  comparisons. Aggregate only — no patient IDs.
- **Key results**: M1 vs M1-Fixed +0.5204 pp (p = 1.1e-5);
  M1 vs M1-Global +0.4262 pp (p = 4.7e-5).
- **Paper location**: Table 4 / Figure 7 / Results 5.5.
- **Original file**: phase32_patient_paired_statistics.csv
  (Phase 3.2, 06_Statistical_Analysis).

## 3. biological_enrichment.csv

- **Content**: complete GO/KEGG/Reactome enrichment table (25,331 terms)
  computed after removing the 33 canonical label-defining markers, with
  branch-matched backgrounds (86 shared / 881 specific genes); per term:
  class, branch, library, overlap, raw p, and BH-FDR under three families
  (within-library / per class-branch / all-tests). Significant and
  non-significant terms are both retained.
- **Key results**: shared branch has no significant terms after the
  complete-family correction; specific-branch enrichment remains for six
  of eight classes; fibroblast (b) resolves to broad ECM terms. These are
  attribution-level associations, not causal claims.
- **Paper location**: Results 5.8 / Supplementary Table S7 / Fig S4.
- **Original file**: enrichment_reanalysis_full.csv
  (Phase 3.1, 03_Enrichment_Reanalysis).
