# Changelog

All notable changes to **Torch-eCpG** are documented in this file.

The current development version on the `dev` branch is **1.27.6-dev**.
The most recent released version on `main` is **1.0.0** (`__version__ = '0.0.1'`).

Entries below describe the work accumulated on `dev` since the last
release on `main`, grouped by the version bump that landed each set of
changes. Each version section is organized into **Features**,
**Improvements / Performance**, and **Bug Fixes** where applicable.

---

## 1.27.6-dev

### Improvements / Performance
- Refactor missing annotation logging in
  `tools/assignRegionToEcpg_parquet.py` (PR #236). Removes per-pair
  `Annotation missing` log statements that previously flooded the
  main log. Missing probe IDs (gene and methylation) are collected
  along with their exclusion reason (`missing_id` or `missing_chrom`)
  and written to a sidecar file named `annotation_missing_ids.txt`
  alongside the output parquet. A single summary line is emitted to
  the main log with the unique counts of missing genes and CpGs and
  the absolute path to the sidecar file.

## 1.27.5-dev

### Bug Fixes
- Skip BED6 annotation rows with missing coordinates in
  `tools/assignRegionToEcpg_parquet.py` and `tools/generate_annotations.py`.
  `readAnnotationFileToDict` previously crashed on `int('')` when
  comprehensive annotation files retained unmapped probes as
  NA-coordinate rows. The BED6 branch now reads and validates
  `chrom`/`start`/`end` before inserting a dict entry; rows with
  missing or NA-like coordinates (`''`, `NA`, `<NA>`, `NAN`)
  increment `nskip` and are skipped so unmapped probes leave no key
  behind and are correctly treated as absent downstream. Verified on
  the `gtp` annotate stage: 5937 NA loci skipped from `G.bed6` and
  703 from `M.bed6`, with no `int()` crash and no `chr22.0`
  coordinate failures.

## 1.27.4-dev

### Features
- Rewrite of `tools/generate_annotations.py` introducing a corrected,
  validated multi-source HT-12 mapping pipeline with provenance
  tracking. Fixes chromosome-labeling and probe-dropping bugs that
  produced spurious X/Y eCpG enrichment in the eQTM Circos plots.
  - **Chromosome handling:** add `clean_geo_chromosome()` to
    normalize labels to canonical tokens (`chr1`-`chr22`, `X`, `Y`,
    `M`/`MT`) and reject pipe-delimited contigs, unplaced/alt
    scaffolds (`NT_`/`NW_`/`GL`/`KI`, `*_random`, `*_hap`),
    pseudoautosomal labels (`chrXY`/`chrYX`), and stray header text.
    Float-contaminated labels (`chr22.0`) are corrected via
    dtype-safe normalization. A single `chr`-prefixed convention is
    enforced through one `write_bed6()` gate with a pre-write
    validation guard, and `normalize_chrom` no longer serializes
    `pd.NA` as `chr<NA>`.
  - **Probe retention and recovery:** unmapped probes are retained
    with NA coordinates and a preserved probe ID for downstream
    ID-level joins. Three coordinate sources are layered by priority:
    Re-Annotator -> GEO -> UCSC WG-6, falling back to NA, and each
    probe is tagged with a provenance value. A UCSC `illuminaProbes`
    (WG-6, hg19) loader is added as a recovery source, converting
    UCSC 0-based starts to the 1-based convention used elsewhere
    (`+1`) and degrading gracefully if the file is absent. Net
    effect: HT-12 hg19 valid mappings rise 42,692 -> 51,553
    (ReAnnotator 34,936 / UCSC_WG6 8,861 / GEO 7,756; NA 5,937).
  - Adds `tests/test_ucsc_integration.py` with 28 synthetic checks
    covering chromosome cleaning, 0-based -> 1-based conversion,
    source priority, contig rejection, NA-row ID retention, and
    `chr<NA>` prevention. Runs offline with no downloads.
- Regenerated comprehensive BED6 annotations for EPIC and HT-12
  (hg19/hg38) produced by the rewritten generator.

## 1.27.3-dev

### Bug Fixes
- Fix the Docker runtime stage in `docker-related/Dockerfile`: unpin
  R and add the runtime shared libraries `libxml2`, `libcurl4`, and
  `libssl3` required for R package loading.

## 1.27.2-dev

### Features
- Updated comprehensive BED6 annotations for EPIC and HT-12
  (`demo/annoEPIC_comprehensive.*.bed6` and
  `demo/annoHT12_comprehensive.*.bed6`).

## 1.27.1-dev

### Features
- `tools/generate_annotations.py` now uses Re-Annotator data as the
  primary HT-12 coordinate source with a GEO-based fallback.

## 1.27.0-dev

### Features
- Add Re-Annotator (J. Arloth) HT-12 v4 annotation data
  (`demo/reannotator_humanHt12v4.txt`), sourced from
  <https://sourceforge.net/projects/reannotator/files/annotations/humanHt12v4.txt/download>
  (see <https://www.biorxiv.org/content/10.1101/019596v1> and
  <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0139516>).
- New pipeline restart option (PR #230). `pipeline.sh` accepts a
  `-s` / `--start-stage` argument so the pipeline can be resumed
  from an intermediate stage (e.g. `pca`, `map`, `summarize`).
  Context variables (`DF`, `TOTAL_TESTS`) are calculated
  independently of earlier stage execution, and all major steps are
  wrapped in a conditional switch so they run sequentially from the
  chosen starting point.
- `pipeline.sh` checks for comprehensive annotation files
  (`annoEPIC_comprehensive.hg19.bed6` and
  `annoHT12_comprehensive.hg19.bed6`) before copying them for both
  the GTP and MESA dataset paths, and gracefully falls back to the
  standard annotation files when they are absent (PR #231).
- Rebuild the Docker image against `dev` using a multi-stage build
  with full pipeline support (PR #233), and add
  `docker-related/REBUILD_DESIGN.md` design memo (PR #232).

### Bug Fixes
- Fix cross-stage R library copy and de-duplicate package install in
  the Docker image (PR #234).

## 1.26.7-dev

### Features
- Add extensive raw diagnostic logging to `tools/plotCircos.py`
  before merging/filtering to diagnose coordinate-mismatch and
  lookup failures (PR #229).

## 1.26.6-dev

- Version bump only; no functional changes.

## 1.26.5-dev

### Features
- Add `tools/generate_annotations.py` script to generate
  comprehensive BED6 annotations (PR #228).
- Edge filtering for bipartite network export and visualization
  (PR #226). `tools/exportBipartiteNetwork.py` gains optional
  `--min-effect` and `--max-boot-p` arguments to reduce "hairball"
  visualizations; filtering is applied in order (effect size ->
  bootstrap p-value -> top K), and the node table implicitly drops
  isolated nodes based on the remaining edges.
  `tools/visualizeBipartiteNetwork.py` logs initial counts of loaded
  edges and nodes for visibility into filtering, and `pipelinePost.sh`
  uses sensible filtering defaults in Stage 4. Unit tests cover
  filtering and graceful fallback for missing statistical columns.

### Improvements / Performance
- `tools/plotCircos.py` reports detailed reasons for excluded
  CpG-Gene pairs (PR #227), distinguishing missing/invalid
  methylation coordinates, missing/invalid gene coordinates, and
  cases where the methylation or gene chromosome is not present in
  the cytoband annotation file.

## 1.26.3-dev

### Features
- New `pipelinePost.sh` script that automates downstream
  visualizations and network analyses (PR #223).

### Improvements / Performance
- `tools/plotCircos.py` improves missing-data reporting (PR #224),
  breaking down the exact reasons pairs are excluded (Missing
  Methylation Coordinates, Missing Gene Coordinates, Methylation
  Chromosome not in Cytoband, Gene Chromosome not in Cytoband). Also
  fixes a local test failure in `test_pearson_single.py` caused by a
  scoping error involving the `time` variable.

### Bug Fixes
- Switch Circos plot generation from hg38 to hg19 (PR #225).
  `pipelinePost.sh` now downloads the hg19 `cytoBand.txt` file from
  UCSC, and `tools/plotCircos.py` help text is updated to match this
  requirement for the default GTP and MESA datasets.

## 1.26.2-dev

### Bug Fixes
- Handle duplicate edges in `tools/visualizeBipartiteNetwork.py`
  (PR #222). `cytoscape_edges.csv` inputs may contain duplicate
  `(Source, Target)` pairs (e.g. from multiple genomic region
  assignments), which previously caused a `ValueError` during
  `pandas.pivot` operations for UMAP and heatmaps. `prepare_network`
  now detects duplicates, keeps the edge with the maximum weight,
  and saves the dropped rows to `dropped_duplicate_edges.csv` in
  `args.out_dir`, with a log line reporting the count of dropped
  duplicates.

## 1.26.1-dev

### Bug Fixes
- Prevent `KeyError` on Parquet -> Pandas conversion (PR #221).
  When `tecpg` chunks are merged or output directly from PyTorch
  logic to Parquet, `gt_id` can end up as the DataFrame index
  instead of a column. Downstream scripts such as
  `tools/assignRegionToEcpg_parquet.py` and `tools/visualizeFindings.py`
  expected `gt_id` as a column and crashed with `KeyError` otherwise.
  This release conditionally injects `reset_index()` across all such
  Parquet readers to safely recover the index into columns.

## 1.26.0-dev

### Improvements / Performance
- Improve bottleneck logs and correct the ETA calculation in the
  PyTorch MLR/Pearson runs (PR #220).

## 1.25.3-dev

### Bug Fixes
- Ignore `sample_reservoir.csv` when merging outputs (PR #219).
  The `sample_reservoir.csv` file produced by `tecpg` is an
  unfiltered random sample used for metric generation and its schema
  differs from the standard chunked Parquet/CSV outputs. Previously
  `tools/mergeOutputs.py` matched it via `*.*`, causing schema
  mismatches and a downstream `KeyError: 'gt_id'` in
  `tools/assignRegionToEcpg_parquet.py`. The file is now explicitly
  filtered out of the file-processing list.

## 1.25.2-dev

### Bug Fixes
- Fix PyTorch MLR failure with dummy data and a Parquet `gt_id`
  index issue (PR #218).

## 1.25.1-dev

### Bug Fixes
- Fix `tools/mergeOutputs.py` unique counts for Parquet files
  (PR #217). When reading from Parquet, `batch.to_pandas()` returns
  `gt_id` and `mt_id` in `df.index` due to how PyArrow converts
  Parquet index metadata, but the script attempted to read them as
  standard columns and therefore reported `0` unique genes and CpGs.
  `df.reset_index()` is now called when reading the batches so the
  indices are exposed as columns and counted correctly.

## 1.25.0-dev

### Bug Fixes
- Avoid pandas type inference for integer columns with nulls in
  Parquet conversion (PR #215). `tools/recalculate_pvalues_parquet.py`
  now reads the `mt_t` column via native PyArrow array access
  (`to_numpy(zero_copy_only=False)`) instead of `batch.to_pandas()`,
  preventing Pandas from upcasting PyArrow nullable `int64` columns
  (such as annotations with missing mappings) into `float64` arrays
  and breaking Parquet output schemas across chunks.
- Fix Parquet chunk merging to prevent 1,000,000 hit-limit
  contamination (PR #216).

## 1.24.4-dev

### Features
- `tools/exportBipartiteNetwork.py` gains an `--out-dir` argument
  (PR #211). The script now accepts an explicit output directory,
  creates it if it does not exist, and writes the edges and nodes
  tables there via `os.path.join(args.out_dir, ...)` instead of the
  current working directory.

## 1.24.3-dev

### Features
- `tools/exploreOmics.py` now reports on **both original and
  processed** methylation and expression files and emits a
  consolidated HTML report (PR #213). The script accepts four input
  files (orig + processed for methylation and expression), generates
  per-dataset QC metrics and plots labelled `_orig` / `_processed`,
  and aggregates the figures and stats into a single HTML page.
  `pipeline.sh` is updated to pass all four inputs and to ensure
  `G_orig.csv` is present before invoking the script.

## 1.24.2-dev

### Features
- Advanced bipartite visualizations added to
  `tools/visualizeBipartiteNetwork.py` (PR #212):
  - `plot_bi_adjacency_heatmap` renders clustered co-regulation
    modules with Seaborn.
  - `plot_arc_diagram` produces a clean horizontal arc representation
    of regulatory bridges.
  - `project_bipartite_to_unipartite` collapses the 2-mode graph to a
    1-mode gene or CpG graph using `count`, `sum`, or `hypergeometric`
    edge-weight methods.
  - Edge weights are validated for `inf`/`-inf` before plotting.
  - All three new charts are hooked into the existing `main()`
    pipeline.

### Bug Fixes
- `tools/visualizeBipartiteNetwork.py` no longer raises `TypeError`
  when `Region` values are empty (PR #210). Missing `Region` entries
  are filled with `'Undefined'` before sorting, the script exits
  early with a clear error if the `Region` column is absent
  entirely, and a per-region count summary is logged.
- Disable the logit transformation for the `gtp` dataset in
  `pipeline.sh` (the dataset is already on a suitable scale).

### Changes
- Revert the default `p-thresh` value from `0.00001` back to `0.001`
  (supersedes the 1.22.3-dev change in PR #196).

## 1.24.1-dev

### Features
- New **Bipartite Network Visualization Suite**
  (`tools/visualizeBipartiteNetwork.py`, PR #209):
  - Parses annotated edges and nodes CSVs produced by
    `exportBipartiteNetwork.py`.
  - Renders an Energy-Minimized Bipartite Network via `networkx` +
    `ForceAtlas2` (`fa2`).
  - Renders a UMAP of Regulatory Beta-Diversity using `umap-learn`
    with a Bray–Curtis metric.
  - Renders a Regulatory Degree Distribution plot (Seaborn) with the
    x-axis clipped to the 99th percentile for readability.
- `tools/exportBipartiteNetwork.py` now extracts and includes the
  `Region` column for CpG nodes in the node table export.

### Improvements
- Rename `tools/export_cytoscape.py` (introduced in 1.24.0-dev) to
  `tools/exportBipartiteNetwork.py`, and remove its previous
  `tests/test_export_cytoscape.py` (the cytoscape exporter and the
  bipartite exporter are the same tool under a clearer name). A
  broken import inside the remaining
  `tests/test_exportBipartiteNetwork.py` is also fixed so the test
  runs again.
- New runtime dependencies added to `requirements.txt`: `networkx`,
  `fa2`, `umap-learn`.

## 1.24.0-dev

### Features
- New **Cytoscape bipartite network exporter**,
  `tools/export_cytoscape.py` (PR #208), which parses an eQTM
  Parquet file and emits filtered Cytoscape-formatted **node** and
  **edge** tables. Records are sorted by `mt_ig` with a fallback to
  `abs(mt_t)`. Required columns are validated and optional columns
  are handled gracefully (e.g. `region` defaults to `'Undefined'`).
  (Renamed to `tools/exportBipartiteNetwork.py` in 1.24.1-dev.)

### Tests
- `tests/test_export_cytoscape.py` covers the exporter (sorting
  precedence, optional-column fallbacks, schema validation). (Removed
  in 1.24.1-dev when the script was renamed.)

## 1.23.2-dev

### Features
- Integrate `tools/exploreOmics.py` into `pipeline.sh` (PR #205).
  External / `src` imports are removed in favor of local helpers
  (`setup_logger`, `set_random_seed`, `ensure_dir`), and the script
  runs immediately after Stage 1 to write QC outputs for methylation
  (`M.csv`) and expression (`G.csv`) into a dataset-specific `qc/`
  directory.
- Add `tools/exploreOmics.py` (initial version): explores omics data,
  computes QC metrics, and generates summary reports and
  visualizations.

### Bug Fixes
- Apply the floor + `log2(x + 1)` gene-expression transform during
  initial data preparation in `tecpg/mesa.py` and `tecpg/gtp.py`
  (using `log2(max(x, 0) + 1)`), and remove the duplicate
  `--log2-transform` invocation from `pipeline.sh` so
  `residualize_pca.sh` no longer transforms the data a second time
  (PR #206).

## 1.23.1-dev

### Features
- `tools/estimateCellProportions.R` now supports **M-values** for
  EpiDISH (PR #204). The script detects negative values in the
  methylation matrix at runtime — indicative of M-values, e.g. from
  the MESA cohort — and applies the inverse log2 transform
  `(2^M) / (2^M + 1)` in-memory to produce beta-values before
  calling EpiDISH. GTP beta-scores are passed through unchanged and
  the on-disk inputs are not modified.

## 1.23.0-dev

### Features
- `tools/visualizeFindings.py` now prefers the empirical bootstrap
  p-value column (`p_boot`) over `precise_mt_p` and `mt_p` when
  loading data and selecting top hits (PR #199). Volcano, Manhattan,
  and Scatter plot labels and annotations show which p-value metric
  is being plotted.
- `tools/visualizeFindings.py` further refactored to generate a
  **full set of plots for every p-value column present** in the
  parquet schema (PR #200). Output filenames are prefixed
  `bootstrapP_`, `preciseP_`, or `mtP_` to differentiate them; falls
  back to `mt_p` when neither bootstrap nor precise columns exist.
  The matrix dependency is now checked lazily, only when scatter
  plots are requested.

### Bug Fixes
- Add NaN diagnostics, error handling, and M-value awareness to the
  preparation pipeline (PR #203):
  - `tecpg/mesa.py` and `tecpg/gtp.py` properly detect row-wise NaNs,
    drop the offending loci, and print clear diagnostics with drop
    counts.
  - `tools/residualize_pca.py` errors early when it sees values
    `<= -1` before `log2(x + 1)` rather than silently producing
    NaNs.
  - `pipeline.sh` only passes `--logit-transform` for the `gtp`
    dataset; MESA already provides M-values so the flag is omitted
    there.
- `tools/plotCircos`: filter `cytoBand.txt` down to the standard
  chromosomes the script initializes to avoid `pycircos` `KeyError`
  on alternate contigs, log any Parquet links excluded because their
  chromosome is missing from the filtered annotations, and
  monkeypatch `pycircos.Circos.draw_scaffold` for Matplotlib 3.10+
  compatibility (strict numpy 0-d scalar type-checking previously
  crashed the script; PR #201).

### Chores
- Remove temporary test files (`patch.py`, `patch_main.py`,
  `patch_scatter.py`, `test_data/`, `test_out/`) accidentally
  committed in an earlier PR, and update `.gitignore` to prevent
  reintroduction (PR #202).

## 1.22.5-dev

### Features
- New `tools/install_dependencies.R` script that identifies and
  installs all required R packages used by the repository's tools
  (`pheatmap`, `EpiDISH`, `sva`,
  `IlluminaHumanMethylationEPICanno.ilm10b4.hg19`, `ExperimentHub`)
  via `BiocManager` for both CRAN and Bioconductor sources (PR #198).

## 1.22.4-dev

### Bug Fixes
- Resolve missing scatter and Manhattan plots in
  `tools/visualizeFindings.py` (PR #197):
  - Cast sample-ID indexes and columns of the `M`, `G`, and `C`
    DataFrames to strings so the matrix intersections are no longer
    empty, restoring comparative scatter plots.
  - Add sample-ID debugging logs for the matrix intersections to
    aid troubleshooting.
  - Rename column lookups `mt_chr` / `mt_pos` → `mt_chrom` /
    `mt_chromStart` to match the actual Parquet schema, re-enabling
    Manhattan plot generation.

## 1.22.3-dev

### Bug Fixes
- `pipeline.sh`: write the `mlr_run.log` `tee` target outside the
  `tecpg` output directory (PR #195). `tecpg` calls
  `helper.initialize_dir()` on the user-supplied output directory at
  startup, which `rmtree`s its contents — if `tee` had already
  created `mlr_run.log` inside that directory the log file was
  deleted immediately, breaking downstream parsing steps. The log
  is now written to the current working directory instead.

### Changes
- Update the default `p-thresh` value to `0.00001` and emit a
  console log line announcing the active threshold (PR #196). (This
  default is later reverted to `0.001` in 1.24.2-dev.)

## 1.22.2-dev

### Bug Fixes
- Fix severe OOM regression in the `_auto_chunk_sizes` chunk auto-sizer
  (`tecpg/cli.py`) when Integrated Gradients is enabled on tight GPUs.
  On the GTP-scale dataset (336341 methylation × 39352 gene × 340
  samples) running on an L4 (~22 GB free VRAM) with `--compute-ig`,
  the estimator's `constants_bytes` term alone exceeded the 80%-of-VRAM
  budget, so the no-anchor branch of `_auto_chunk_sizes` returned a
  negative chunk size and silently fell back to
  `(gt_count // 4, mt_count // 4) = (9838, 84085)` — roughly 5–10×
  larger than the previously-known-safe `(15000, 1000)` static values
  — causing OOM on the first chunk. Three coordinated changes:
  - **IG-aware estimator.** `estimate_loci_per_chunk_e_peak` and
    `estimate_loci_per_chunk_results_peak` (`tecpg/tool.py`) gain
    `compute_ig` / `compute_ig_deep` parameters. Analytical IG charges
    one extra `(M, S, K)`-equivalent constants term (for the
    `X_diff_mean` transient before reduction) plus a 1.5× per-locus
    factor; deep IG additionally charges another two `(M, S, K)`
    equivalents and a 4× per-locus factor (for the retained autograd
    graph and per-step interpolated activations). Factors are
    conservative on purpose — under-estimating peak causes OOM,
    over-estimating only forces extra outer iterations.
  - **Bisection-based fallback.** When the estimator returns < 1 at
    `mt_count` (no-anchor branch), instead of `(gt_count // 4,
    mt_count // 4)` the helper now bisects over `mt ∈ [1, mt_count]`
    for the largest meth chunk that admits a non-trivial gene chunk
    (g_floor=64, falling back to 1 only on extreme tightness). The
    same bisection helper is reused in the anchor-g branch (replacing
    its previous `mt_count // 4` fallback). The anchor-m branch's
    naive `gt_count // 4` fallback is replaced with `gene_chunk=1`
    plus a loud warning, since the user has anchored mt and there is
    nothing to bisect.
  - **Loud diagnostics + VRAM safety ceiling.** The "estimate < 1"
    log line is upgraded from `info` to `warning` and now names all
    the estimator inputs (target bytes, samples, mt, gt, covars,
    `compute_ig`, `compute_ig_deep`) so this regression cannot recur
    silently. As belt-and-suspenders, when IG is enabled and effective
    free VRAM is ≤ 24 GB the no-anchor pair is clamped to
    `(gene ≤ 2000, meth ≤ 20000)`, and at ≤ 48 GB to
    `(gene ≤ 4000, meth ≤ 40000)`. The clamp is not applied to
    anchored modes — anchoring is an explicit user request.

### Tests
- `tests/test_host_profile.py` gains five new tests covering the
  IG-aware estimator, the bisection-based fallback (vs. the old
  naive `// 4` quartering), and the safety ceiling (clamps with IG
  on tight VRAM, no-op without IG, no-op when anchored). The helper
  signature gains an optional `target_bytes=` parameter so the
  tests are deterministic on CPU.

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
