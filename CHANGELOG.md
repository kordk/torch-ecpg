# Changelog

All notable changes to **Torch-eCpG** are documented in this file.

The current development version on the `dev` branch is **1.12.2-dev**.
The most recent released version on `main` is **1.0.0** (`__version__ = '0.0.1'`).

Entries below describe the work accumulated on `dev` since the last
release on `main`, grouped by the version bump that landed each set of
changes. Each version section is organized into **Features**,
**Improvements / Performance**, and **Bug Fixes** where applicable.

---

## 1.12.2-dev

### Bug Fixes
- Correct sample size assignment to avoid a `TypeError` during GPU profiling.

## 1.12.1-dev

### Bug Fixes
- Resolve a `gpu_monitor` typo and a thermal-reading race condition in GPU
  monitoring.

## 1.12.0-dev

### Features
- Add rich per-chunk profiling and diagnostic bottleneck heuristics to help
  identify whether runs are GPU-, I/O-, or CPU-bound.

## 1.11.0-dev

### Improvements / Performance
- Optimize MLR GPU usage with a QR-decomposition cache and non-blocking
  NVML calls to reduce GPU-monitoring overhead and improve utilization.

## 1.10.1-dev

### Bug Fixes
- Guard the `tecpg/__main__.py` entry point against multiprocessing
  re-entry recursion and fix the function signature in `regression_single`.

## 1.10.0-dev

### Improvements / Performance
- Increase GPU utilization by moving chunk saving to a `ProcessPoolExecutor`,
  decoupling I/O from GPU compute.

## 1.9.0-dev

### Improvements / Performance
- Memory backpressure on the save queue, streaming `to_csv`, and explicit
  per-chunk reference freeing to bound peak RAM during long runs.

## 1.8.1-dev

### Features
- Add `--save-threads` CLI option to control the asynchronous save pool.

### Improvements / Performance
- Replace covariate tensor `repeat` with `expand` to substantially reduce
  memory usage in the MLR path.

## 1.8.0-dev

### Features
- Add a full analysis bash pipeline script `pipeline.sh` with robust
  argument parsing, descriptive logging, timestamps, and data-existence
  checks.
- Integrate PCA covariates preprocessing into the pipeline and add a
  standalone PCA preprocessing script for covariate generation.

### Improvements / Performance
- Switch asynchronous CSV saving to a `ThreadPoolExecutor` to prevent
  fork-bombing RSS memory.
- Refactor `tecpg_mlr_lstsq` for lower memory footprint.
- Update `pipeline.sh` defaults (modest M and G chunk sizes,
  `PYTHONUNBUFFERED=1`).

### Bug Fixes
- Fix mismatched labels between data matrices by forcing string types.

## 1.7.2-dev

### Improvements / Performance
- Memory optimizations in the `lstsq` MLR backend.
- Pipeline script logging, timestamps, and data-existence checks.

## 1.7.1-dev

### Features
- Add a configurable bootstrap batch size to the CLI to resolve out-of-memory
  errors during bootstrapping.

### Bug Fixes
- Update bootstrap-list generation logic and reporting.

## 1.7.0-dev

### Features
- New tool `tools/createBootstrapList.py` to filter eQTMs for bootstrapping.
- Implement an `lstsq` bootstrap subcommand in `tecpg`.

### Bug Fixes
- Handle headers in the background BED file in `summarizeOutput_parquet`.
- Correct the ENCODE DNase download URL.

## 1.6.8-dev

### Features
- Add ENCODE Enrichment Analysis to `summarizeOutput_parquet.py`.
- Add regional functional enrichment using `gseapy` and `mygene`.
- Circos plot improvements: cytoband labels and a custom legend.

### Bug Fixes
- Handle missing coordinates in `plotCircos.py` to avoid `IntCastingNaNError`.

## 1.6.6-dev

### Features
- Add a standalone script to generate Circos plots for eQTM architecture.

### Bug Fixes
- Identify the correct t-statistic column for downstream calculations.

## 1.6.5-dev

### Features
- Decouple Integrated Gradients (IG) outputs and allow covariate filtering
  via a file argument.

## 1.6.4-dev

### Features
- Add a stacked proportional saliency chart to the summarization script.

## 1.6.3-dev

### Features
- Comparative plotting mode for eQTM visualizations.
- Regional FDR summaries and top-hits output in
  `summarizeOutput_parquet.py`.
- New `tools/visualizeFindings.py` script for multi-omic plots.
- Support for GENCODE GTF annotations in `assignRegionToEcpg_parquet.py`.
- Add `verify_alignment` to `assignRegionToEcpg_parquet.py`.

### Improvements
- Keep all rows and columns in `assignRegionToEcpg_parquet.py`.
- Simplify CLI arguments in `visualizeFindings.py`.

## 1.6.2-dev

### Features
- Add `summaryParquetToCsv.py` for converting Parquet outputs to CSV.

## 1.6.1-dev

### Features
- Support GFF annotation format in `assignRegionToEcpg_parquet.py`.

## 1.6.0-dev

### Features
- Integrate Integrated Gradients (IG) into the `lstsq` MLR backend.
- Add `summarizeOutput_parquet.py` for Parquet outputs with FDR
  calculation.

### Bug Fixes
- Fix schema mismatch for empty chunks in `mergeOutputs.py`.

## 1.5.1-dev

### Features
- Add estimated time remaining (ETA) to `lstsq` chunk logging.

## 1.5.0-dev

### Features
- Add `tools/assignRegionToEcpg_parquet.py` script.

### Improvements
- Move `assignRegionToEcpg.py` under `tools/`.

## 1.4.2-dev

### Features
- Add a tool to recalculate p-values for Parquet output.

## 1.4.1-dev

### Features
- Support Parquet and Snappy/ZSTD compression in `mergeOutputs.py`.

## 1.4.0-dev

### Features
- Optional `--permute-label-test` flag for the `lstsq` method.
- New `tecpg data mesa` command for downloading and processing the MESA
  dataset.

## 1.3.3-dev

### Improvements
- Add a statistical power audit to degrees-of-freedom logging.

## 1.3.2-dev

### Features
- Add subsampling parameters to the `mlr` command.

## 1.3.0–1.3.1-dev

### Features
- Implement FDR threshold discovery, QQ plot, and genomic-inflation
  (lambda) calculation in summarization.
- Add `tools/summarizeOutput.py` for evaluating regression output.
- Add `tecpg/tools/mergeOutputs.py` for merging CSV chunks.
- Add reservoir sampling to the `mlr` `lstsq` backend, with initialization
  and final logging.
- Add a standalone script to recalculate p-values with high precision.

### Improvements
- Detailed help description for summarization metrics.

### Bug Fixes
- Recalculate p-values from t-statistics for main chunks
  (`summarizeOutput`).
- Fix undefined `meth_count` / `gene_count` variables in reservoir-sampling
  logs and update test tool paths.
- Fix GPU memory reporting to include reserved memory and track peak usage
  in `tecpg_mlr_lstsq`.

## 1.2.6–1.2.9-dev

### Features
- Add version reporting to the CLI and unify the version source.

### Bug Fixes
- Robust GPU UUID-based monitoring and selection (replaces brittle name /
  index matching).
- Fix GPU thermal monitoring incorrectly selecting an idle GPU due to
  index mismatch.

## 1.2.5-dev

### Improvements
- Refactor logging for chunk saving and add peak memory checks.
- Add user-options reporting and parameter logging for troubleshooting.
- Refactor CLI to use keyword arguments for MLR functions.

### Bug Fixes
- Enable `tecpg` to read specific data files outside the default input
  directory.

## 1.2.0–1.2.1-dev

### Features
- Add `--logit-transform` option to MLR commands, with validation
  protocol and runtime logging for Beta-to-M-value conversion.
- Add memory and thermal reporting to MLR functions, including GPU
  thermal monitoring and throttling and GPU/RAM usage logging.
- Add scatter plots for MLR comparison and validation tests.
- Update tests to include logit-transformed M-values; add docstring to
  the `mlr` command.

### Bug Fixes
- Fix environment verification to check the installed package version.

## 1.1.0-dev

### Features
- Implement the `torch.linalg.lstsq` MLR backend.
- Add a validation test for the MLR `lstsq` backend.
- Add a human-readable summary to the MLR comparison test.

### Bug Fixes
- Fix test failures by prioritizing local imports over the installed
  package.
- Fix `TypeError` in tests by upgrading the SciPy requirement; add
  `tests/README.md`.
- Increase estimate tolerance for `float32` precision in the accuracy
  test.
