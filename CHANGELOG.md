# Changelog

All notable changes to **Torch-eCpG** are documented in this file.

The current development version on the `dev` branch is **2.0.0b2.dev48**.
The most recent released version on `main` is **1.0.0** (`__version__ = '0.0.1'`).

As of `2.0.0b2.dev0` the project version scheme migrated to the
[PEP 440](https://peps.python.org/pep-0440/) standard, replacing the
older `X.Y.Z-dev` suffix with pre-release / development tags such as
`2.0.0b1` (beta) and `2.0.0b2.devN`.

Entries below describe the work accumulated on `dev` since the last
release on `main`, grouped by the version bump that landed each set of
changes. Each version section is organized into **Features**,
**Improvements / Performance**, and **Bug Fixes** where applicable.

---

## 2.0.0b2.dev48

### Features
- Add a region annotation stage to `pipelinePermute.sh` (PR #322).
- Consume the canonical region column for the 7-way `eval_permute`
  stratify (PR #321).

### Improvements / Performance
- Use unbuffered `python3` calls in `pipelinePermute.sh`.
- Revise the `qr_permute` status and validation details.

### Bug Fixes
- Restore strict fallback byte-identity in `eval` and strengthen the
  associated tests.
- Fix a `pipelinePermute.sh` bug and add `gtpsub` to `pipeline.sh`.

### Tests
- Resolve a missing header in the `test_smoke_summarize_permute` mock
  BED files.

## 2.0.0b2.dev47

### Features
- Update `pipelinePermute.sh` to support the `--reservoir` flag and add a
  summary step (PR #320).

### Improvements / Performance
- Demote `lambda_excess` from the stratify verdict gating (PR #319).
- Set `MLR_IG_COVARIATES='all'` in `pipeline.sh`.
- Revise architecture notes and chromosome handling.

### Bug Fixes
- Fix a bug in `tools/summarize_permute.py`.

## 2.0.0b2.dev46

### Bug Fixes
- Output the true raw dataframes `M_orig.csv` and `G_orig.csv` (PR #318).

## 2.0.0b2.dev45

### Improvements / Performance
- Make the master-consistency guard advisory and recalibrate its
  tolerance (PR #317).
- Update `pipelinePermute.sh`.

## 2.0.0b2.dev44

### Bug Fixes
- Intersect the master universe with the normalized M/G in `qr_permute`
  (PR #316).

## 2.0.0b2.dev43

### Bug Fixes
- Fix the QC report comparison by properly saving and retaining the
  original data (PR #315).

## 2.0.0b2.dev42

### Improvements / Performance
- Refactor `qr_permute` as a consumer path with an additive master merge
  (PR #314).

### Documentation
- Enhance the documentation for the `qr_permute` method.

### Bug Fixes
- Fix the `qr_permute` signature and update the realignment test suite.
- Move `exploreOmics.py` to the end of `pipelinePre.sh` (PR #313).

## 2.0.0b2.dev41

### Improvements / Performance
- Chunk the pair dimension in the observed-statistic path of `qr_permute`
  (PR #311).
- Add progress logging to the `qr_permute` path (PR #312).

### Bug Fixes
- Fix duplicate logging in the permute function.
- Fix `pipelinePre.sh` so dummy data correctly skips EpiDISH.

### Tests
- Harden the chunk-coverage guard using real loop counts.

## 2.0.0b2.dev40

### Features
- Add the `subsample_loci` tool and the `gtpsub` (subsampled genotype)
  dataset to the pipeline (PR #310).
- Add the `pipelinePermute.sh` script and documentation (PR #308).

### Improvements / Performance
- Exclude EpiDISH for the `gtpsub` dataset in `pipelinePre.sh`.

### Documentation
- Update the CHANGELOG for `2.0.0b2.dev26` through `dev39` (PR #307).
- Update `tests/README.md` to the current standard (CI, `pytest.ini`,
  undocumented tests).

### Bug Fixes
- Remove dummy output files that were erroneously committed (PR #309).

## 2.0.0b2.dev39

### Bug Fixes
- Resolve `eval_permute` annotation-reader failures and `NaN`-chromosome
  handling, and fix a `TypeError` from `pd.NA` boolean evaluation in the
  `eval_permute` tests (PR #306).

## 2.0.0b2.dev38

### Features
- Support running `qr_permute` from the CLI at `--all` with real BED6
  annotations (PR #305).

### Documentation
- Revise the `qr_permute` status and output documentation.

### Tests
- Fix regression-test integrity for annotation load/pass at `--all`.

## 2.0.0b2.dev37

### Features
- Add a standalone read-only `eval_permute` diagnostic and scaffold its
  evaluation harness (PR #303).

### Improvements / Performance
- Add a minimal CI test gate that runs `pytest`, and allow the required Node
  version for the checkout / setup-python actions (PR #304).

### Bug Fixes
- Address evaluation bugs in `eval_permute` and its tests, and fix test-suite
  collisions and label-strata robustness.

## 2.0.0b2.dev36

### Features
- Add permutation output finalization and thresholding (`qr_permute` chunk 9)
  (PR #301).

### Documentation
- Update the `qr_permute` status and clarify its limitations.

## 2.0.0b2.dev35

### Features
- Implement the GPD peaks-over-threshold tail for `qr_permute` chunk 8
  (PR #300).

### Tests
- Test tail continuity against the epsilon handoff and clean up scripts.

## 2.0.0b2.dev34

### Features
- Implement `_score_observed` to compute empirical permutation p-values
  (PR #299).

### Documentation
- Document the permute / bootstrap tests and add run guidance (PR #298).

## 2.0.0b2.dev33

### Features
- Build out the `qr_permute` null pipeline: real cis/trans masking via a shared
  helper (chunk 3, PR #294), null-population subsampling (chunk 4, PR #295),
  design-fixed Freedman–Lane permutation of the response (chunk 5, PR #296),
  and streaming null accumulation with trans null stratification (chunk 6,
  PR #297).

### Bug Fixes
- Remove a duplicate test and fix an overflow bound in chunk 6.

### Documentation
- Revise the `qr_permute` status and implementation details.

## 2.0.0b2.dev32

### Improvements / Performance
- Make parquet the default output format for all MLR methods; `--output-format
  auto` now resolves to `parquet` regardless of host profile (PR #293).

## 2.0.0b2.dev31

### Features
- Implement the batched PyTorch QR solver for `_compute_observed_statistic`
  and the observed-statistic path for `qr_permute` chunk 2 (PR #292).

### Documentation
- Create documentation for the `qr_permute` method.

## 2.0.0b2.dev30

### Features
- Add the `qr_permute` walking skeleton (chunk 0 + chunk 1) (PR #291).

### Improvements / Performance
- Refactor the chunking logs in `tecpg_mlr_qr` to use the standard log
  formatting (PR #289).

### Documentation
- Update the README dev version and note the PEP 440 versioning scheme
  (PR #290).

## 2.0.0b2.dev29

### Bug Fixes
- Fix the `tecpg run --help` output for the mlr commands (PR #288).

## 2.0.0b2.dev28

### Improvements / Performance
- Update the MLR CLI descriptions and hide `mlr-single` from non-debug help
  (PR #287).

### Bug Fixes
- Change the debug short flag from `-d` to `-D` to prevent collisions with
  dataset flags in bash scripts (PR #286).

## 2.0.0b2.dev27

### Improvements / Performance
- Change the `memory_check` log level to debug to reduce log noise (PR #285).

## 2.0.0b2.dev26

### Improvements / Performance
- Add `diagnoseExpressionPCs.py` for GTP to `pipelinePre.sh`.

## 2.0.0b2.dev25

### Features
- Add between-sample quantile normalization for gene expression (`G`),
  applied immediately after the existing floor and `log2` transforms in
  `tecpg/gtp.py`, `tecpg/mesa.py`, and `tecpg/test_data.py` (PR #281).
  Includes a finite-input guard that rejects `NaN`/`inf` before computing,
  and exact pandas rank-mapping so tied values average across ranks.

## 2.0.0b2.dev24

### Improvements / Performance
- Add diagnostics for the dominant expression-PC concentration observed in
  `residualize_pca`.

## 2.0.0b2.dev23

### Bug Fixes
- Drop near-degenerate cell types before the reference drop and INT. The
  RPC reference panel sometimes fails to resolve a cell type (e.g.
  eosinophils nonzero in only ~1.5% of samples), leaving a near-constant
  column that INT collapses to a single tie-rank and that weakly
  reintroduces collinearity with the intercept. `drop_degenerate_cells()`
  removes cell types nonzero in fewer than `--min-nonzero-frac` of samples
  (default 0.5; 0 disables), guards against fewer than two survivors, and
  falls back to the most-abundant remaining type when a pinned
  `--reference` is itself dropped.

### Improvements / Performance
- Key the input-scale diagnostic FLAG off each covariate's scale-adjusted
  `|beta|` (`mean|IG| / input_MAD`) relative to methylation's, rather than
  the largest raw input MAD, so it distinguishes real coefficient dominance
  from input-scale effects.

## 2.0.0b2.dev22

### Bug Fixes
- Break cell-proportion collinearity before appending EpiDISH RPC fractions
  to the covariate matrix. The fractions sum to ~1 per sample, making the
  cell-type columns collinear with the intercept and smearing one shared
  coefficient across them (uninterpretable per-cell coefficients and IG
  attributions). Drop one reference cell type (default most-abundant; pin
  with `--reference` for cross-cohort reproducibility) and apply a
  rank-based inverse-normal transform (Blom c=3/8) to the remaining cell
  columns (`--no-int` to skip). Non-cell covariates pass through untouched.
- Restrict every saliency aggregate in `tools/evaluateSaliency.py` to rows
  that actually carry IG (the bootstrapped subset), dropping the silent
  `fillna(0)` that poisoned fraction distributions, rank bands, and
  per-feature proportions. Adds top-50 tables ranked by `|mt_ig|` and
  `p_boot`, effect and input-scale diagnostics, and a `--frac-exclude`
  filter.

## 2.0.0b2.dev21

### Features
- Implement the tecpg batch-2/3 completion pass (PR #280). Removes the
  `DF=96` fallback so degrees of freedom are derived unconditionally from
  `C.csv`, with the `M7-DF` `C.shape.meta` cross-check failing closed before
  DF computation. Hardens the correctness harness with an independent
  bootstrap p-value oracle, a strict `p_boot_floor` assertion, a seed
  round-trip subprocess check across PyArrow formats, and fingerprint
  comparison (`np.isclose`, atol=1e-5) on `p_min`/`p_max`.

### Documentation
- Update the eCpG filtering & prioritization living document against `dev`
  HEAD (PR #279).

## 2.0.0b2.dev20

### Features
- Seed the bootstrap resample draw and record the seed in the outputs for
  reproducibility (PR #278).

### Improvements / Performance
- Add before/after row-count logging at silent drop sites (observability
  only) (PR #278).

## 2.0.0b2.dev19

### Bug Fixes
- Statistics-trust fixes, Batch 2 (PR #277). Floor `p_boot` at its true
  empirical resolution (`1 / finite resample count`) in float64 so a
  one-sided resample distribution reports the smallest representable
  p-value instead of 0. Require `TOTAL_TESTS` (the BH-FDR denominator) to be
  extracted from the mlr log with no placeholder fallback, aborting the
  pipeline on failure. Add an `M7-DF` stage-boundary check that validates
  `C.csv` still matches the `(samples, covars)` shape recorded in
  `C.shape.meta` and asserts `DF > 0` before recalculating p-values.

## 2.0.0b2.dev18

### Improvements / Performance
- Add a tecpg correctness test harness with seeded `test_data` and a
  committed structural fingerprint (PR #276). See
  `tests/test_correctness_harness.py`; regenerate the reference with
  `--regenerate-fingerprint`.

## 2.0.0b2.dev17

### Improvements / Performance
- Demote verbose bootstrap memory logs from INFO to DEBUG (PR #275).

## 2.0.0b2.dev16

### Bug Fixes
- Write non-chunked bootstrap output to `bootstrap_merged.<ext>`, preserving
  the configured file extension via `os.path.splitext` (PR #274).

## 2.0.0b2.dev15

### Bug Fixes
- Fix `TOTAL_TESTS` and reservoir-deletion bugs (PR #273). The end-of-run
  summary block was skipped when `chunking=False`, so `TOTAL_TESTS` was
  never logged for `pipeline.sh`; it now always logs. `save_dataframes` also
  called `initialize_dir` on the final non-chunked save, wiping the freshly
  generated `sample_reservoir.csv`; a `clear_dir=False` flag now preserves
  it.

## 2.0.0b2.dev14

### Bug Fixes
- Forward the IG keyword arguments to `tecpg_mlr_qr_bootstrap` in the
  `qr_bootstrap` CLI branch so per-feature IG columns are emitted; adds a
  CLI regression test for the forwarding (PR #272).

## 2.0.0b2.dev13

### Breaking Changes
- Hard rename of the MLR computation methods from `lstsq`,
  `lstsq_bootstrap`, and `manual` to `qr`, `qr_bootstrap`, and
  `legacy_normal_eq` respectively, to reflect the underlying QR
  decomposition + triangular solve (PR #271). All associated Python symbols,
  CLI arguments, and documentation are updated; there are no
  backward-compatible aliases.

### Bug Fixes
- Make `test_cross_method_equivalence` deterministic and non-empty by
  seeding the fixtures and passing `p_thresh=1.0`, so it actually exercises
  numerical equivalence instead of comparing empty frames (PR #271).

## 2.0.0b2.dev12

### Features
- Add Integrated Gradients (IG) computation to the MLR bootstrap (PR #268).
  Adds `compute_ig`, `compute_ig_deep`, `ig_baseline`, and
  `ig_covariates_filter` arguments, integrating Captum IG and analytical IG
  derived from the original (non-bootstrapped) regression coefficients, with
  end-to-end tests for the IG column outputs.

## 2.0.0b2.dev11

### Features
- Replace the bootstrap `lstsq` solver with batched QR decomposition and a
  degenerate resample guard (PR #265). On CUDA, `gels` behaves unreliably on
  rank-deficient arrays while CPUs silently fall back to min-norm solutions;
  the update standardizes batched least-squares with explicit QR. A guard
  catches any `nan`/`inf` produced by rank-deficient combinations, counts and
  logs them, and restricts summary statistics (`mean`, `std`,
  `torch.quantile`) to the valid subset only. Robust production-level tests
  are added to verify the guard under degenerate inputs.

### Improvements / Performance
- Update `tests/README.md` with thematic organization and script descriptions
  (PR #264).

## 2.0.0b2.dev10

### Features
- Split `pipeline.sh` into `pipelinePre.sh` (preprocessing stages 1–2) and a
  leaner core `pipeline.sh` (stages 3–9) (PR #260). `pipelinePre.sh` covers
  data preparation, EpiDISH cell-proportion estimation, and PCA; `pipeline.sh`
  enforces precondition guards that error if preprocessed outputs (`M.csv`,
  `G.csv`, `C.csv`, BED6 annotations) are absent, directing users to run
  `pipelinePre.sh` first.
- Extract functional enrichment into a standalone `tools/runEnrichment.py`
  (PR #262). Removes ~690 lines of enrichment logic from
  `tools/summarizeOutput_parquet.py`; the new tool reads FDR and IG Parquet
  inputs directly, adds retry-with-backoff for `gseapy.enrichr` calls,
  validates Enrichr library names against the live API, and caps per-region
  gene lists. Restores the two-sided Fisher's exact test for ENCODE enrichment
  p-values (previously switched silently to `alternative='greater'`). Adds
  URL-scheme validation in `download_encode_files` (Bandit B310) and excludes
  the generated `p_value_histogram.png` via `.gitignore`.

### Improvements / Performance
- Wire `tools/runEnrichment.py` into `pipelinePost.sh` as a new final stage
  `[7/7]` (PR #263). Adds `ENRICHMENT_DIR`/`SUMMARIZED_PARQUET` variables,
  renumbers existing stages, and adds a comment in `pipeline.sh` pointing to
  the extracted tool.
- Update `README.md` to reflect the two-step `pipelinePre.sh` →
  `pipeline.sh` workflow (PR #261, PR #263). Adds a dedicated "Preprocessing
  pipeline" section, rewrites the "Full analysis pipeline" section to cover
  stages 3–9, and corrects the `--bootstrap-iterations` default from 100 to
  1000. The tools reference is updated to add `runEnrichment.py` and clarify
  that enrichment is no longer bundled inside `summarizeOutput_parquet.py`.
- Update `docs/ecpg-filtering-prioritization.md` with the revised stage maps,
  enrichment table (tool name, line citations, default libraries, output
  paths), and Enrichr-library parameter citation (PR #263).

## 2.0.0b2.dev9

### Bug Fixes
- Correctly apply `--ig-covariates` to the bootstrap MLR stage so that
  per-feature IG attribution is emitted (PR #256). `pipeline.sh`
  previously passed only `--compute-ig`, leaving `mt_ig` (scalar
  methylation attribution) as the sole output and producing
  broken/degenerate `evaluateSaliency.py` results. Two stage-scoped
  configuration variables now control per-feature IG generation:
  `MLR_IG_COVARIATES="none"` (Stage 3, to avoid intermediate-file
  bloat) and `BOOTSTRAP_IG_COVARIATES="all"` (Stage 9, to compute
  full per-feature saliency on top candidates at ~1MB cost).
  Downstream scripts read Parquet schemas dynamically and propagate
  the `<covariate>_ig` columns to `bootstrap_merged.parquet`
  automatically. Documentation is updated to reflect the
  stage-scoped configuration, and a synthetic `test_per_feature_ig.py`
  verifies the per-feature IG does not collapse to uniform fractions.

## 2.0.0b2.dev8

### Features
- Add `tools/evaluateSaliency.py` for integrated-gradients (IG)
  distribution analysis (PR #254, PR #255). Parses the
  bootstrap-merged Parquet to produce console reports and decay
  curves of the `mt_ig` saliency distribution, chunking the read with
  `iter_batches` to avoid memory saturation on massive (153M-row)
  datasets such as MESA. Detects the saliency inflection point using
  `kneed` when available and falls back to normalized chord-distance
  geometry otherwise, degrading gracefully when per-feature IG scores
  lack covariate columns. The step is wired into `pipelinePost.sh`,
  and a synthetic `tests/test_evaluateSaliency.py` fixture is added.

## 2.0.0b2.dev7

### Bug Fixes
- Harden functional enrichment against transient and configuration
  failures in `tools/summarizeOutput_parquet.py` (PR #254). Adds retry
  logic with exponential backoff for `gseapy.enrichr` calls to absorb
  transient network errors (e.g. HTTP 504), validates library names
  against `gseapy.get_library_name()` and updates the deprecated
  `WikiPathways_2021_Human` to `WikiPathways_2024_Human`, adds
  `--enrichment-max-genes` to rank and cap submitted genes by lowest
  p-value (avoiding Enrichr payload-size limits), and adds a
  `--dry-run-enrichment` flag to simulate API calls and failure
  recovery.

## 2.0.0b2.dev6

### Bug Fixes
- Fix a Parquet schema-mismatch error in FDR summarization in
  `tools/summarizeOutput_parquet.py` (PR #253). When writing result
  chunks, Pandas converted nullable integer coordinates to `float64`
  if `NaN`s were present but kept `int64` otherwise, causing PyArrow
  to reject later chunks. The schema retrieved from the first chunk is
  now applied explicitly to all subsequent chunks, and the
  `mt_chromStart` / `gt_chromStart` coordinate columns are coerced to
  pandas' nullable `Int64` type to prevent float conversion. A unit
  test verifies chunk-writing stability under schema variability.

## 2.0.0b2.dev5

### Improvements / Performance
- Sideload `plotCircos.py` missing-coordinate exclusions to a sidecar
  file instead of flooding the main log (PR #252).

## 2.0.0b2.dev4

### Features
- Add a `--min-per-region` floor limit to the bootstrap pair-selection
  script `tools/createBootstrapList.py` (PR #118, default 200). A
  "sandwich" calculation guarantees a minimum number of pairs per
  region when available, and the output summary table now displays
  `(FLOOR)` and `(CAPPED)` labels alongside the configured floor.

### Improvements / Performance
- Update `pipeline.sh` bootstrap configuration: per-region floor
  (`> 4500/region`), `< 10,000` max cap, and 1000 bootstrap
  iterations.
- Increase the default `NETWORK_TOP_K` to 5000.

## 2.0.0b2.dev3

### Improvements / Performance
- Increase the default `NETWORK_TOP_K` to 1000.

## 2.0.0b2.dev2

### Improvements / Performance
- Update `.github/dependabot.yml` configuration for Dependabot.

## 2.0.0b2.dev1

### Features
- Add the eCpG filtering & prioritization living document
  (`docs/ecpg-filtering-prioritization.md`) and link it from the
  README (PR #241).

### Bug Fixes
- Fix Illumina probe ID translation in
  `tools/summarizeOutput_parquet.py` (PR #242). Renames
  `clean_and_translate_ensembl_ids` to `clean_and_translate_gene_ids`
  and adds mapping for Illumina `ILMN_*` array probe IDs (e.g. the GTP
  cohort): primary resolution via Re-Annotator
  (`demo/reannotator_humanHt12v4.txt`), then a secondary fallback
  using `demo/ucsc_illuminaProbes.hg19.txt` or
  `demo/annoHT12.hg19.bed6` with GENCODE v49lift37 `pyranges`
  intersection, while preserving the generic `mygene` strategy for
  `ensembl.gene` arrays (e.g. MESA). Also adds URL-scheme validation
  to `download_gencode_gtf` (only `http://`/`https://`), resolving a
  Bandit B310 finding.

## 2.0.0b2.dev0

### Improvements / Performance
- Migrate the project version scheme to the PEP 440 standard.

### Bug Fixes
- Refactor eCpG region classification to contiguous, symmetric 5'/3'
  labels (PR #240). Replaces the previous CIS / DISTAL classifications
  with bidirectional `CIS5`, `CIS3`, `DISTAL5`, `DISTAL3` variants
  derived from the 5' and 3' coordinates, closing the unassigned gaps
  for same-chromosome pairs in both `assignRegionToEcpg_parquet.py`
  and the legacy `assignRegionToEcpg.py` (both strands), preventing
  `UNKNOWN` dead-zones. Dictionary keys (`my_typeCountH`) and
  downstream reporting tools (`createBootstrapList.py`,
  `test_exportBipartiteNetwork.py`) are updated to the new annotation
  strings; existing plot scripts accept the new categories without
  hardcoded color conflicts.

## 2.0.0b1

### Improvements / Performance
- Bump version to `2.0.0b1` for the Beta release.

## 1.27.8-dev

### Features
- Add the UCSC hg19 `illuminaProbes.txt` annotation
  (`demo/ucsc_illuminaProbes.hg19.txt`), sourced from the UCSC
  golden-path database, as a probe-mapping recovery source.

### Improvements / Performance
- Refresh README legacy sections, highlight the `pipeline.sh` stages,
  and add a Demo datasets section (PR #239).

## 1.27.7-dev

### Improvements / Performance
- Update the README to reflect the `dev` branch pipeline and changelog,
  and add a migration example for the removed `-g`/`-m` short flags
  (PR #238).
- Regenerate the comprehensive BED6 annotation files for EPIC and
  HT-12 (hg19/hg38).

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
