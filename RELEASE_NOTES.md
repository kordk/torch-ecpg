# Release Notes — torch-ecpg v1.2.9

**Summary of changes on the `dev` branch since the last release (`main`).**

This release introduces significant new features for the MLR (Multiple Linear Regression) pipeline, including a new `lstsq` computation backend, logit-transform support, reservoir sampling, GPU thermal monitoring, and several post-processing tools. Numerous bug fixes improve GPU device selection, memory reporting, CLI argument handling, and test reliability.

---

## New Features

### MLR `lstsq` Backend (PR #45)
- Implemented a new `torch.linalg.lstsq`-based MLR backend (`tecpg_mlr_lstsq`) as an alternative to the existing manual regression approach.
- Registered the `lstsq` backend in the CLI under the `mlr` command.

### Logit Transform for Beta-to-M-value Conversion (PRs #54, #55, #56, #61)
- Added `--logit-transform` CLI flag to `mlr` and `mlr_single` commands for converting methylation Beta values to M-values using `log2(beta / (1 - beta))`, with clamping to `[1e-6, 1 - 1e-6]`.
- Implemented `logit_transform_torch` and `logit_transform_pandas` helper functions in `tecpg/helper.py`.
- Added validation protocol with runtime logging for the Beta-to-M-value conversion, including input bounds checking, clamping frequency, and output distribution metrics.

### Reservoir Sampling (PRs #72, #73)
- Added an online Reservoir Sampling algorithm to the `mlr lstsq` backend via the `--reservoir-count` CLI flag.
- Uses vectorized `torch.rand` to maintain a rolling sample pool of computation pairs prior to `p_thresh` filtering.
- Outputs an unbiased sample pool to `sample_reservoir.csv` in the output directory.
- Added initialization and final logging for reservoir sampling progress.

### GPU Thermal Monitoring and Throttling (PRs #59, #60)
- Created `tecpg/gpu_monitor.py` for NVIDIA Management Library (NVML) integration and thermal logic.
- Added `--thermal-threshold` and `--thermal-wait` CLI options to `mlr` and `mlr_single` commands.
- Integrated thermal monitoring into `tecpg_mlr_lstsq`, `regression_full`, and `regression_single` computation loops.
- Implemented `gpu_guardian` context manager for safe NVML initialization and shutdown.
- Added memory and thermal reporting to MLR functions.

### GPU and RAM Usage Logging (PR #58)
- Added GPU VRAM and system RAM usage logging to MLR methods via `Logger.memory_check()`.

### Version Reporting (PR #68)
- Added `--version` CLI flag using `click.version_option`.
- Unified version source: `setup.py` now reads the version from `tecpg/__init__.py` to ensure a single source of truth.
- Version is logged at startup via `logger.info`.

### Post-Processing Tools

#### Recalculate P-values Tool (PR #69)
- Added `tools/recalculate_pvalues.py` to recalculate p-values from regression results using `scipy.stats.t.sf` with `float64` precision, addressing limitations of the default `float32` output.
- Supports multiprocessing and chunk-based processing for memory efficiency.

#### Summarize Output Tool (PRs #70, #74, #75, #78)
- Added `tools/summarizeOutput.py` for memory-efficient summarization of large CSV regression outputs, including:
  - Unique gene (`gt_id`) and CpG (`mt_id`) counts.
  - P-value histogram generation.
  - Genomic Inflation Factor (Lambda) calculation from reservoir file t-statistics using `float64` precision.
  - QQ plot generation (`qq_plot.png`) with rasterized rendering for efficiency.
  - Benjamini-Hochberg FDR threshold discovery on large unsorted datasets.
- Added detailed `--help` descriptions documenting the script's metrics and usage.

#### Merge Outputs Tool (PR #71)
- Added `tools/mergeOutputs.py` for merging chunked CSV output files into a single consolidated result.

### User Options Reporting (PR #63)
- Added user options reporting and parameter logging at the start of MLR runs for troubleshooting.

### Flexible Input File Paths (PR #65)
- `read_dataframes` now accepts an optional `file_names` list, allowing specific data files to be read from outside the default input directory.
- CLI commands (`corr`, `mlr`, `mlr_single`, `chunks`) pass explicit file paths to `read_dataframes`.

---

## Bug Fixes

### GPU Monitor Device Selection (PRs #66, #67)
- **UUID-based GPU selection**: Implemented robust GPU UUID-based monitoring to prevent selecting the wrong physical GPU when PyTorch and NVML device enumeration differ (e.g., due to `CUDA_VISIBLE_DEVICES`).
- **Index mismatch fix**: Replaced strict UUID lookup and index fallback with iterative search across all NVML devices, with normalized UUID and name matching (case-insensitive, whitespace-stripped).

### GPU Memory Reporting (PR #76)
- Fixed GPU memory reporting to include `torch.cuda.memory_reserved()` alongside `torch.cuda.memory_allocated()`, addressing under-reporting of GPU cache size.
- Added strategic `memory_check()` calls inside `tecpg_mlr_lstsq` to capture peak memory during tensor allocation, regression, residual calculation, and p-value derivation.

### Summarize Output P-value Precision (PR #78)
- Fixed `summarizeOutput.py` to recalculate p-values on the fly from t-statistics using `scipy.stats.t.sf` in `float64`, preventing underflow issues where `float32` GPU p-values became exactly `0.0`, leading to incorrect FDR cutoffs and histogram gaps.

### CLI Argument Passing (PR #62)
- Refactored `mlr` and `mlr_single` CLI commands to use keyword argument dictionaries (`**kwargs`), fixing an issue where `logit_transform` and other arguments could be misaligned due to positional argument fragility.

### Undefined Variable in Reservoir Sampling Logs (PR #77)
- Replaced references to undefined `meth_count` and `gene_count` variables with `len(M)` and `len(G)` for correct calculation of `expected_items` in the reservoir sampling log message.
- Updated import paths in test files for tool scripts.

### Input File Path Handling (PR #65)
- Fixed `ValueError` when the default data directory is missing or empty but specific files are provided via CLI.

### Environment Verification (PR #57)
- Fixed `verify_env.sh` to change to a neutral directory before checking the installed `tecpg` version, preventing stale metadata from a local `.egg-info` directory from being picked up.

### Scipy Compatibility (PR #49)
- Upgraded `scipy` requirement to `>=1.12.0` to resolve `TypeError` caused by binary incompatibility between older `scipy` and newer `numpy` versions.

### Test Import Precedence (PR #51)
- Fixed test failures by using `sys.path.insert(0, ...)` instead of `sys.path.append(...)` to ensure local source code is imported over an installed package missing newer modules.

### Test Accuracy Tolerance (PR #50)
- Increased estimate tolerance from `1e-4` to `2e-4` in accuracy tests to accommodate expected `float32` vs `float64` precision differences.

---

## Testing Improvements

- **MLR lstsq validation test** (PR #48): Added independent validation test comparing `lstsq` backend results against `statsmodels` OLS.
- **MLR comparison summary** (PR #52): Added human-readable summary output to the MLR comparison test.
- **MLR scatter plots** (PR #53): Added scatter plots for visual comparison of MLR backends across estimate, t-statistic, p-value, and standard error metrics.
- **Logit transform tests** (PRs #54, #55, #56, #61): Added `tests/test_logit.py` and `tests/test_logit_transform.py` to verify transformation logic, mathematical invariants, symmetry, and clamping.
- **GPU monitor mock tests** (PRs #66, #67): Added `tests/test_gpu_monitor_mock.py` to verify UUID-based GPU selection with mocked hardware.
- **Merge tool tests** (PR #71): Added `tests/test_merge_tool.py` to verify CSV chunk merging.
- **Recalculate p-values tests** (PR #69): Added `tests/test_recalculate_pvalues.py` to verify high-precision p-value recalculation.
- **Test documentation** (PR #49): Added `tests/README.md` documenting test purpose and usage.
- **Validation utilities** (PR #48): Added `tests/validation_utils.py` and `tests/verify_env.sh` for environment validation.

---

## Documentation

- Added `tests/README.md` with test descriptions, usage instructions, and troubleshooting (PR #49).
- Added detailed `--help` descriptions to `summarizeOutput.py` documenting all output metrics (PR #74).

---

## Infrastructure and Maintenance

- Version bumped from `0.0.1` to `1.2.9-dev` through incremental updates (1.1.0 → 1.2.0 → 1.2.1 → 1.2.5 → 1.2.6 → 1.2.7 → 1.2.9).
- Added `nvidia-ml-py` to `requirements.txt` for GPU monitoring.
- Upgraded `scipy>=1.12.0` in `requirements.txt`.
- Consolidated tool scripts under the `tools/` directory.
- Removed generated test output files and plots from version control.

---

## Files Changed (25 files, +3,149 / −87 lines)

| Area | Files |
|------|-------|
| Core | `tecpg/processing.py`, `tecpg/cli.py`, `tecpg/helper.py`, `tecpg/gpu_monitor.py`, `tecpg/logger.py`, `tecpg/import_data.py`, `tecpg/regression_full.py`, `tecpg/regression_single.py`, `tecpg/pearson_full.py` |
| Tools | `tools/summarizeOutput.py`, `tools/recalculate_pvalues.py`, `tools/mergeOutputs.py` |
| Tests | `tests/test_accuracy.py`, `tests/test_gpu_monitor_mock.py`, `tests/test_logit.py`, `tests/test_logit_transform.py`, `tests/test_merge_tool.py`, `tests/test_mlr_comparison.py`, `tests/test_recalculate_pvalues.py`, `tests/validation_utils.py`, `tests/verify_env.sh`, `tests/README.md` |
| Config | `setup.py`, `requirements.txt`, `tecpg/__init__.py` |
