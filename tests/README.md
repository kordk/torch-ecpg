# Tests for tecpg

This directory contains validation, regression, performance, and smoke tests for the `tecpg` pipeline and its associated tools. The tests are organized below by theme.

## Running the tests

Most tests import `tecpg`, so install the package in editable mode first and
run the tests from this `tests/` directory:

```bash
# from the repository root
pip install --editable .

cd tests
```

The suite is a mix of `pytest`-style tests, `unittest` modules, plain
executable scripts, and shell scripts. The individual "How to run" lines below
give the exact command for each file. The Python tests are all `pytest`
compatible, so the quickest way to run the whole Python suite is:

```bash
# from the tests/ directory
pytest
```

You can also run a single file or a single test:

```bash
pytest test_permute_skeleton.py
pytest test_permute_subsample.py::test_select_null_population_determinism
```

The tests are deterministic and run on CPU; a GPU is not required (GPU-specific
behavior is mocked). Running `pytest` requires `pytest` to be installed
(`pip install pytest`); the shell scripts (`*.sh`) are run directly and are not
collected by `pytest`.

## 1. Accuracy and Method Comparison

These tests validate the mathematical correctness of `tecpg`'s regression implementations against established statistical libraries (like `statsmodels`) and ensure consistency across different backend methods.

- **`test_accuracy.py`**
  - **Purpose:** Validates the accuracy of `tecpg`'s regression results against `statsmodels` (OLS). It generates synthetic data, runs `tecpg` regression, and randomly selects pairs to compare.
  - **Output:** Prints a comparison table showing metrics like Max Abs Diff and Mean Abs Diff, and creates scatter plots showing the correlation of values if the `validation_utils` module supports it.
  - **How to run:** `python test_accuracy.py`
- **`test_mlr_comparison.py`**
  - **Purpose:** Compares the two implementation backends of `tecpg` (`regression_full` and `tecpg_mlr_qr`) to ensure both methods produce consistent results across different chunking and region filtration scenarios.
  - **Output:** Prints a detailed comparison summary of max/mean absolute and relative differences, and correlation.
  - **How to run:** `python test_mlr_comparison.py`
- **`test_correctness_harness.py`**
  - **Purpose:** Correctness test harness that encodes *correct* behavior independently of current output, as a precondition for later statistics fixes. Contains oracle/differential tests (batched QR regression vs. plain OLS; BH-FDR vs. `statsmodels`; `p_boot` vs. a hand-rolled empirical computation), a committed structural fingerprint of the dummy `all` pipeline, and invariant/property tests. Tests tied to a pending audit fix are marked `xfail` with the audit item (e.g. `C1` for the missing `p_boot` `1/B` floor) so the fix flips them from `xfail` to `pass`; the fingerprint flags unintended structural drift.
  - **Reference artifact:** `fingerprint_all_pipeline.json` is the *only* committed reference (no stored output parquets, no network access). Regenerating it re-blesses structure and requires a reviewed reason.
  - **How to run:** `pytest test_correctness_harness.py`
  - **Regenerate fingerprint:** `python tests/test_correctness_harness.py --regenerate-fingerprint`
- **`test_bootstrap_qr.py`**
  - **Purpose:** Validates the `qr_bootstrap` MLR backend (`tecpg/bootstrap.py`). Confirms the batched QR + triangular-solve regression matches `torch.linalg.lstsq` on well-conditioned inputs and behaves on degenerate designs, runs the bootstrap end-to-end, and checks that the CLI forwards integrated-gradients kwargs and the default output filename.
  - **Output:** Test pass/fail output from `pytest`.
  - **How to run:** `pytest test_bootstrap_qr.py`
- **`test_recalculate_pvalues.py`**
  - **Purpose:** Verifies that the internal helper script for recalculating p-values correctly applies the Student's t-distribution survival function.
  - **Output:** Test pass/fail output from `unittest`.
  - **How to run:** `python -m unittest test_recalculate_pvalues.py`
- **`test_logit.py`** & **`test_logit_transform.py`**
  - **Purpose:** Verifies that the logit transformation applied to input arrays (pandas and torch) maintains mathematical invariants.
  - **Output:** Test pass/fail output from `unittest`.
  - **How to run:** `python -m unittest test_logit.py` and `python -m unittest test_logit_transform.py`

## 2. Data Processing and Tools

These tests ensure that standalone scripts and pipeline utilities correctly process data, merge outputs, and generate functional networks.

- **`test_exportBipartiteNetwork.py`**
  - **Purpose:** Tests the `exportBipartiteNetwork.py` tool to ensure it correctly parses pipeline outputs and formats them for Cytoscape.
  - **Output:** Test pass/fail output from `unittest`.
  - **How to run:** `python -m unittest test_exportBipartiteNetwork.py`
- **`test_merge_tool.py`**
  - **Purpose:** Tests the `mergeOutputs.py` tool to verify it accurately concatenates chunked parquet or CSV files.
  - **Output:** Test pass/fail output from `unittest`.
  - **How to run:** `python -m unittest test_merge_tool.py`
- **`test_preprocessPcaCovariates.py`**
  - **Purpose:** Verifies that the PCA covariate preprocessing step behaves correctly under simulated datasets.
  - **Output:** Test pass/fail output from `pytest`.
  - **How to run:** `pytest test_preprocessPcaCovariates.py`
- **`test_runEnrichment.py`**
  - **Purpose:** Tests the `runEnrichment.py` script to ensure functional enrichment mapping correctly summarizes pipeline results.
  - **Output:** Test pass/fail output from `unittest`.
  - **How to run:** `python -m unittest test_runEnrichment.py`
- **`validation_utils.py`**
  - **Purpose:** A utility file containing helper functions (like `run_statsmodels_ols`) used by the accuracy validation tests. *Not run directly.*

## 3. Performance, Profiling, and Resources

These tests evaluate the dynamic scaling, GPU utilization, memory management, and profiling mechanics of the host execution environment.

- **`test_auto_scale.py`** & **`test_host_profile.py`**
  - **Purpose:** Unit tests for host-profile classification and thread auto-sizing logic, guaranteeing appropriate chunk and thread scaling without needing a GPU.
  - **Output:** Test pass/fail output from `pytest`.
  - **How to run:** `pytest test_auto_scale.py test_host_profile.py`
- **`test_bottleneck.py`**
  - **Purpose:** Verifies the `analyze_bottleneck` logic correctly diagnoses performance bottlenecks like low RAM or GPU underutilization.
  - **Output:** Test pass/fail output from `pytest`.
  - **How to run:** `pytest test_bottleneck.py`
- **`test_cuda_alloc_conf.py`**
  - **Purpose:** Tests the pre-torch setup of `PYTORCH_CUDA_ALLOC_CONF` to ensure proper CUDA memory pressure tuning is applied.
  - **Output:** Test pass/fail output from `pytest`.
  - **How to run:** `pytest test_cuda_alloc_conf.py`
- **`test_gpu_monitor.py`** & **`test_gpu_monitor_mock.py`**
  - **Purpose:** Tests the `ThermalMonitor` logging and GPU uuid matching functionality utilizing `pynvml` (mocked where necessary).
  - **Output:** Test pass/fail output from `pytest` and `unittest`.
  - **How to run:** `pytest test_gpu_monitor.py` and `python -m unittest test_gpu_monitor_mock.py`
- **`test_save_queue_depth_invariant.py`**
  - **Purpose:** A static AST regression test ensuring the `save_queue_depth` variable is not incorrectly shadowed, preventing a pipeline crash.
  - **Output:** Test pass/fail output from `pytest`.
  - **How to run:** `pytest test_save_queue_depth_invariant.py`
- **`profiling_verdict_test.sh`**
  - **Purpose:** Verifies the shell logic inside `profiling.sh` for interpreting tecpg logs and identifying performance verdicts.
  - **Output:** Success message or script failure.
  - **How to run:** `./profiling_verdict_test.sh`

## 4. Edge Cases and External Integration

Tests focusing on specific regression prevention and compatibility with external standards.

- **`test_fdr_schema_mismatch.py`**
  - **Purpose:** Tests the pipeline's robustness against parquet schema mismatches generated by earlier steps (like `summarizeOutput_parquet.py`).
  - **Output:** Test pass/fail output from `unittest`.
  - **How to run:** `python -m unittest test_fdr_schema_mismatch.py`
- **`test_per_feature_ig.py`**
  - **Purpose:** Simulates MLR stage 9 per-feature integrated gradients logic.
  - **Output:** Test pass/fail output from `pytest`.
  - **How to run:** `pytest test_per_feature_ig.py`
- **`test_ucsc_integration.py`**
  - **Purpose:** Synthetic test harness for the UCSC WG-6 integration logic in `generate_annotations2.py`, validating chromosome naming and coordinates.
  - **Output:** Exit code 0 indicates all passed.
  - **How to run:** `python test_ucsc_integration.py`

## 5. Environment and Smoke Tests

Basic verification scripts to quickly determine if the environment is set up correctly and the code compiles/imports.

- **`verify_env.sh`**
  - **Purpose:** Checks the Conda environment and Git branch status to ensure development requirements are met.
  - **Output:** Terminal output indicating environment health.
  - **How to run:** `./verify_env.sh`
- **`test_imports_smoke.py`** & **`test_imports_mocked.py`**
  - **Purpose:** Basic tests checking if internal components can be imported cleanly, sometimes with external libraries mocked out.
  - **Output:** Test pass/fail output or `"Imports OK"`.
  - **How to run:** `python test_imports_smoke.py` and `python test_imports_mocked.py`
- **`test_minimal_config.sh`**
  - **Purpose:** End-to-end smoke test validating the CPU-only execution path (`--host-profile minimum`) on a synthetic dataset.
  - **Output:** Console output detailing the pipeline steps and assertion results.
  - **How to run:** `./test_minimal_config.sh`

## 6. Permutation Testing

These tests cover the permutation-based null-distribution backend
(`tecpg/permute.py`), which estimates empirical p-values by permuting samples
and accumulating a null statistic distribution.

- **`test_permute_skeleton.py`**
  - **Purpose:** End-to-end smoke test of `tecpg_mlr_qr_permute` on a small synthetic fixture, asserting the output file is written with the expected schema and row count.
  - **Output:** Test pass/fail output from `pytest`.
  - **How to run:** `pytest test_permute_skeleton.py`
- **`test_permute_oracle.py`**
  - **Purpose:** Differential/oracle test comparing `_compute_observed_statistic` (batched QR regression) against a hand-rolled plain OLS t-statistic to confirm the observed statistics match.
  - **Output:** Test pass/fail output from `pytest`.
  - **How to run:** `pytest test_permute_oracle.py`
- **`test_permute_residualize.py`**
  - **Purpose:** Verifies the `_residualize_and_permute` step: an oracle check against a strong-covariate fixture, an identity invariant, and determinism under a fixed seed.
  - **Output:** Test pass/fail output from `pytest`.
  - **How to run:** `pytest test_permute_residualize.py`
- **`test_permute_accumulate.py`**
  - **Purpose:** Tests `_accumulate_null` for memory-bounded streaming accumulation (histogram + bounded top-k), histogram and top-k correctness, determinism, per-pair stratification, and calibration sanity.
  - **Output:** Test pass/fail output from `pytest`.
  - **How to run:** `pytest test_permute_accumulate.py`
- **`test_permute_subsample.py`**
  - **Purpose:** Tests `_select_null_population` subsampling behavior (correct subset sizes, determinism, warning/full-population fallback, zero-count errors) and permutation with subsampling enabled.
  - **Output:** Test pass/fail output from `pytest`.
  - **How to run:** `pytest test_permute_subsample.py`
- **`test_permute_mask.py`**
  - **Purpose:** Verifies `_compute_trans_mask` cis/trans classification semantics using chromosome and position annotations.
  - **Output:** Test pass/fail output from `pytest`.
  - **How to run:** `pytest test_permute_mask.py`

## Troubleshooting

### `TypeError: C function scipy.spatial._qhull._barycentric_coordinates has wrong signature`

If you encounter this error when running tests, it indicates a version mismatch between `scipy` and `numpy` (or other compiled extensions) in your environment.

**Fix:**
Upgrade `scipy` to a version compatible with your `numpy` installation (typically `scipy>=1.12.0`). You can try reinstalling dependencies:

```bash
pip install --upgrade scipy numpy
# or re-install all requirements
pip install -r ../requirements.txt
```