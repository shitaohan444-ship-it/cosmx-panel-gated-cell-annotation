# Differences from the run code

This release was assembled from the exact scripts that produced the
manuscript results. The following differences are the only ones:

| Release file | Original | Difference |
|---|---|---|
| src/models/m1.py | src/models/experiment5/model5.py (mlp, GateFusion5) + src/models/experiment6/model6.py (M1Model6) | mlp/GateFusion5 are defined locally in the same file instead of imported from the package; all code bodies byte-identical |
| src/models/m1_fixed.py, m1_global.py | 10_Phase32_Results/01_Model_Implementation/model_phase32.py | Class bodies byte-identical; `mlp` imported from `.m1` instead of `src.models.experiment5.model5`; file split into two |
| src/metrics.py | 09_Phase31_Results/10_Scripts/phase31_common.py | None (byte-identical) |
| configs/model_config.example.yaml | configs/experiment5.yaml | None (byte-identical) |
| scripts/train.py | scripts/experiment6/train6.py + train_phase32.py | Data source replaced by user-provided preprocessed npz (same schema); paths config-driven; training loop line-identical |
| scripts/counterfactual.py | phase31_gate_counterfactual.py | Paths config-driven; intervention math identical (incl. permutation rng scheme) |
| scripts/patient_stats.py | phase32_stats.py | Paths config-driven; estimands/statistics identical |
| scripts/calibration.py | 13_Phase35.../phase35_calibration.py | Paths config-driven; ECE/Brier implementation identical |
| run_code_reference/* | original run scripts | None (verbatim copies; retain hard-coded server paths by design, for provenance) |

No model logic, no metric definition, and no statistical procedure was
modified. The tests in `tests/test_models.py` pin the parameter counts and
gate behavior to the manuscript values.
