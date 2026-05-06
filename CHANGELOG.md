# Changelog

All notable changes to **Torch-eCpG** are documented in this file.

The current development version on the `dev` branch is **1.21.0-dev**.
The most recent released version on `main` is **1.0.0** (`__version__ = '0.0.1'`).

Entries below describe the work accumulated on `dev` since the last
release on `main`, grouped by the version bump that landed each set of
changes. Each version section is organized into **Features**,
**Improvements / Performance**, and **Bug Fixes** where applicable.

---

## 1.21.0-dev

### Breaking Changes
- The `tecpg run mlr` subcommand's `--gene-loci-per-chunk` and
  `--meth-loci-per-chunk` options no longer accept the `-g` / `-m`
  short forms. The short forms collided visually with the top-level
  group's `-g, --gene-file` / `-m, --meth-file` flags, and with the
  new anchoring semantics (PR 2 lets the user pin one of the two
  chunk dimensions and auto-derive the other) the overlap was
  becoming a real source of confusion. Picking different single
  letters (e.g. `-G` / `-M`) would only have shifted the collision —
  those are already taken by `--gene-annot` / `--meth-annot` at the
  top level — so the short forms are dropped and only the
  unambiguous long forms remain. Users who passed `-g <N> -m <N>` to
  `tecpg run mlr ...` need to replace those with
  `--gene-loci-per-chunk <N> --meth-loci-per-chunk <N>`. The
  `pipeline.sh`, `profiling.sh`, `tests/test_minimal_config.sh`,
  `docker-related/running_test_data.txt`, and `README.md` examples
  in this repo have been updated. The unrelated `data dummy` and
  `chunks` subcommands' own `-g`/`-m` short flags are unchanged.

### Improvements / Performance
- `_auto_chunk_sizes` (`tecpg/cli.py`) gains anchored-mode support:
  when the user supplies exactly one of `-g` / `-m` the helper now
  honors that anchor and auto-derives the other dimension from the
  same 80%-of-budget memory target used for the fully-automatic case
  (CUDA `mem_get_info` on GPU, `psutil.virtual_memory().available` on
  CPU). Anchoring is honored on any host class because it is an
  explicit user request; the historical "minimum hosts never auto-set
  chunk sizes when the user supplies neither flag" invariant is
  preserved unchanged. `pinned_g` is satisfied via bisection over
  `mt_count` against the existing peak-memory estimators in
  `tecpg.tool`; `pinned_m` is satisfied directly by feeding the
  anchored value in as `mt_count`.
- Drop the `meth_chunk = min(mt_count, 40000)` ceiling on the
  fully-automatic path: the RAM/GPU budget is the binding constraint
  (the estimator is already keyed on `mt_count`), and the ceiling
  could only force extra outer iterations on hosts where a larger
  `-m` fit fine. Combined with PR 1's smaller post-PR-1 inner-kernel
  footprint this lets the auto-sizer pick more aggressive `-m`
  values when the budget allows.
- Estimator docs (`_auto_chunk_sizes`) updated to reflect the
  post-PR-1 inner-kernel footprint: the per-CpG `(B, S, T, P)`
  tensors are now realized at the active methylation column only
  (K=1) when `methylation_only=True` and no IG is requested, which
  is what the `full_output=False` branch of the estimator already
  implicitly modeled. The `2 *` factor in the results-peak formula
  is preserved as a conservative upper bound on transient overlap
  during the in-place buffer assembly.

### Tests
- `tests/test_host_profile.py` gains coverage for the anchored modes
  (`pinned_m`, `pinned_g`, both pinned), and for the absence of the
  old `40000` methylation ceiling on the auto path.

## 1.20.1-dev

### Improvements / Performance
- PR 1 (Inner-kernel peak-memory reduction): drop the late
  `torch.cat([B, S, T, P])` in the lstsq path in favor of an
  in-place pre-allocated buffer that is filled and freed
  incrementally (A5); free `X` immediately after QR when deep IG is
  not requested (A6); and slice `B`/`S` to the active methylation
  column before forming `T = B / S` and `P = normal_p(T)` when
  `methylation_only=True` and neither analytical nor deep IG is
  requested (A1–A2). Bit-equivalent output verified against the
  pre-PR-1 baseline on CPU via `tests/test_mlr_comparison.py` and
  the "All Region" demo variants.



### Features
- Add `--host-profile {auto,minimum,server}` (envvar `TECPG_HOST_PROFILE`)
  to the top-level CLI. `auto` (default) detects the host class from
  physical CPU count and total RAM (`<12 cores or <32 GB` ⇒ `minimum`,
  otherwise `server`). The resolved profile drives defaults for
  save-pool size, output format, prefetch depth, and chunk auto-sizing.
  Explicit per-flag overrides (`--save-threads`, `--output-format`,
  `--gene-loci-per-chunk`, `--meth-loci-per-chunk`,
  `--prefetch-chunks`) always win.

### Improvements / Performance
- `--output-format` gains an `auto` default that resolves to `parquet`
  on server-class hosts and `csv` on minimum-class hosts. CSV behavior
  on laptop-class hosts is unchanged.
- The chunked save pool now uses a `ThreadPoolExecutor` for the
  parquet path (pyarrow releases the GIL, so per-chunk DataFrames no
  longer have to be pickled across `spawn`-mode workers). The CSV path
  keeps the historical `ProcessPoolExecutor`.
- `_auto_save_threads`: cap on server-class hosts lowered from 32 to
  8. Profiling on RAID6/dm-crypt LUNs (klabdev) showed the underlying
  device saturates well before 32 concurrent writers; extra workers
  only added kernel-writeback CPU cost and cross-process pickle
  traffic.
- Auto-scale `gene_loci_per_chunk` and `meth_loci_per_chunk` on
  server-class hosts when the user does not supply `-g`/`-m`. Uses the
  same memory heuristics as the existing `tecpg chunks` subcommand
  (80% of free GPU memory, or 80% of available system RAM in CPU
  mode). Minimum-class hosts never auto-set chunk sizes.
- `--prefetch-chunks` auto-resolution now reports `0` when CUDA is
  unavailable or `--host-profile=minimum`, fixing a `pin_memory()`
  crash on CPU-only systems.

### Tests
- `tests/test_minimal_config.sh`: end-to-end smoke test verifying the
  pipeline still runs on a simulated minimum-config host (8 cores /
  16 GB RAM, CPU-only) under `--host-profile minimum` for auto-,
  CSV-, and Parquet-format output.
- `tests/test_host_profile.py`: unit tests for `_host_class`,
  `_auto_save_threads`, and `_auto_chunk_sizes`.
- `tests/test_auto_scale.py`: updated for the new save-thread cap and
  gains coverage for `_host_class`.

## 1.15.1-dev

### Bug Fixes
- `profiling.sh`: stop counting `GpuIdle` as a throttle and prioritize the
  save-bound verdict so diagnostic conclusions correctly attribute
  bottlenecks (adds `tests/profiling_verdict_test.sh`).

## 1.15.0-dev

### Features
- Add `profiling.sh`, a bash-based GPU diagnostic tool that drives
  `nvidia-smi`, `top`, `vmstat`, and `pidstat` alongside PyTorch debug
  output to evaluate bottleneck origins, with a matrix sweep over
  prefetching, larger chunks, TF32, and BLAS thread configurations and
  an environment-annotated tarball of results. Documented in
  `docs/profiling.md`.
- Add `AGENTS.md` and vendor the Clean Code and *A Philosophy of
  Software Design* mini rule sets under `docs/agent-rules/` for use by
  AI coding agents.

### Bug Fixes
- `profiling.sh`: prevent aborts under `set -euo pipefail` when text
  searches (e.g. for `PROFILE` chunks or `Verdict`) find no matches,
  add a fallback message when `VERDICT` is empty, and surface the tail
  of `tecpg.log` on non-zero `tecpg` exits.
- `profiling.sh`: replace the invalid `--profile-equivalent` flag with
  the `TECPG_PROFILE=1` environment variable so `PROFILE chunk` log
  lines are actually produced.
- `profiling.sh`: gracefully handle missing `tecpg.log` and
  `chunk_profile_summary.txt` files (fall back to `NA` metrics) instead
  of leaking `grep`/`tail` "No such file or directory" errors.
- `profiling.sh`: write `tecpg` output to `$cell_dir/tecpg_out` so
  `tecpg`'s output-directory initialization no longer wipes the
  profiler's own logs (`tecpg.log`, `nvidia-smi-query.csv`,
  `chunk_profile_summary.txt`).
- `profiling.sh`: remove the hardcoded 90-second matrix-run cap that
  was overriding the user-supplied `-D` duration, and evaluate
  `extract_metrics` outside the `metrics=$(run_workload ...)` subshell
  so runtime log lines are no longer swallowed or mixed into
  `matrix_csv` output.
- `profiling.sh`: cap `bigger_chunks` cell sizes (`g <= 1000`,
  `s <= 40000`; defaults reduced to 20000 / 500 for `gtp` and `mesa`)
  to avoid CUDA OOM on 22–24 GB devices, and place
  `--blas-threads` as a global `tecpg` option before the subcommand to
  fix `Error: No such option: --blas-threads`.
- `profiling.sh`: add step-status summary logging and search for the
  `Verdict:` string instead of relying on `tail -n 1`.

## 1.14.0-dev

### Improvements / Performance
- Auto-scale `--save-threads` based on available CPUs and remove
  per-chunk `torch.cuda.empty_cache()` calls to reduce GPU
  synchronization overhead. Adds `tests/test_auto_scale.py`.

## 1.13.1-dev

### Bug Fixes
- Calculate prefetch chunk coordinates directly from the lookahead
  index in `_tecpg_mlr_lstsq_inner` so a saturated prefetch queue no
  longer leaves `lookahead_start` stale, which previously caused
  oversized chunks and downstream `IndexError` dimension mismatches.

## 1.13.0-dev

### Features
- Add `--prefetch-chunks` (via `concurrent.futures.ThreadPoolExecutor`
  in `tecpg_mlr_lstsq`) to overlap chunk preparation with GPU compute.
- Add a `--blas-threads` global CLI option implemented as a
  pre-import environment shim in `tecpg/__main__.py` so BLAS thread
  caps take effect before NumPy/PyTorch import.
- Add gap-fill diagnostics: `TECPG_SAVE_THREADS` fallback for
  `--save-threads`; report `save_threads_effective`,
  `prefetch_chunks_effective`, `blas_threads_effective`, logical and
  physical CPU counts, and thread counts in the startup banner;
  emit per-chunk `gpu_idle_between_chunks_ms`, `save_queue_depth`, and
  `prefetch_fill` metrics with an end-of-run statistical summary.
- Document the new performance-tuning options in `README.md`.

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
