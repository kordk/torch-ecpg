# Tools and helper scripts

The `tools/` directory contains the supporting scripts driven by
`pipelinePre.sh`, `pipeline.sh`, `pipelinePost.sh`, and `pipelinePermute.sh`.
They can also be invoked standalone; each accepts `--help` for its options.
The R-based tools additionally require the R packages installed by
`Rscript tools/install_dependencies.R` — see
[R dependencies](../README.md#r-dependencies-pipeline-and-tools-only) in the
README.

Which script runs at which pipeline stage is documented under
[Pipeline stages](../README.md#pipeline-stages) in the README.

## Contents

* [Data preparation and QC](#data-preparation-and-qc)
* [Annotation](#annotation)
* [Mapping post-processing](#mapping-post-processing)
* [Influence and permutation diagnostics](#influence-and-permutation-diagnostics)
* [Bootstrapping](#bootstrapping)
* [Visualization and network analysis](#visualization-and-network-analysis)
* [Benchmarking and profiling](#benchmarking-and-profiling)

## Data preparation and QC

* `tools/generateProbeBlacklist.sh` / `generateProbeBlacklist.R` —
  build a probe blacklist (SNP-affected, cross-reactive, sex-chromosome) from
  the DMRcatedata ExperimentHub lists, scoped to the array in use. Defaults to
  **450k**; pass `epic` or `both` as the second argument, or set `METH_ARRAY`
  for `pipelinePre.sh`. The DMRcate source lists span 450K and EPICv1; the
  array argument selects which manifest to scope the output to. Output is
  `probes_blacklist.csv` with columns `Probe_ID,Reason` (Reason is one or more
  of SNP / CROSSREACTIVE / SEXCHROM, `;`-joined).
  The superseded `generateEpicProbeBlacklist.sh` / `_v2.R` derived
  sex-chromosome probes from the EPIC manifest alone and so missed 450K probes
  with no EPIC counterpart.
* `tools/exclude_blacklisted_probes.py` — drop blacklisted CpGs from
  `M_orig.csv` to produce `M.csv`.
* `tools/exploreOmics.py` — QC metrics, plots, and a consolidated HTML
  report for the original and processed methylation/expression matrices.
* `tools/estimateCellProportions.R` / `estimateCellProportions.sh` — run
  EpiDISH for immune cell-proportion estimation; M-value aware.
* `tools/residualize_pca.py` / `residualize_pca.sh` — residualize against
  covariates and emit principal-component covariates.
* `tools/preprocessPcaCovariates.py` — PCA preprocessing for covariates
  used by `pipelinePre.sh`.
* `tools/ancestry_probes_report.py` — evaluate methylation-derived ancestry
  instruments and emit scores, a probe table, and an HTML/JSON report.
* `tools/mergeCovariateColumns.py` — merge selected sidecar columns (e.g.
  ancestry components) into the covariate matrix.
* `tools/encodeCategorical.py` — expand integer-coded categorical covariates
  into indicator columns with a minimum cell-size guard.
* `tools/subsample_loci.py` — subsample rows (loci, never samples) of a
  matrix; used to build the `gtpsub` dataset.
* `tools/install_dependencies.R` — install all R packages required by
  the tools (`pheatmap`, `EpiDISH`, `sva`, `ExperimentHub`, `minfi`, and the
  `IlluminaHumanMethylation450kanno.*` / `IlluminaHumanMethylationEPICanno.*`
  manifests) via `BiocManager`, then verify each one loads.

## Annotation

* `tools/generate_annotations.py` — regenerate comprehensive HT-12 / EPIC
  BED6 annotations from Re-Annotator, GEO, and UCSC sources, with
  provenance tracking.
* `tools/annotation_io.py` — shared annotation readers (transparently reads
  gzipped files) and drop-if-ambiguous probe/symbol → gene-model resolvers.
* `tools/build_probe_gene_model.py` — derive the ILMN probe → gene-model map
  (`probe_gene_model.tsv`) from a GENCODE GTF, or a synthetic fixed-span map
  for `dummy`.
* `tools/assignRegionToEcpg_parquet.py` and `tools/assignRegionToEcpg.py` —
  Parquet- and CSV-based region assignment into the seven strand-aware
  labels (PROMOTER / GENEBODY / CIS5 / CIS3 / DISTAL5 / DISTAL3 / TRANS). The
  Parquet variant takes the `--gene-model` map and writes a sidecar
  `annotation_missing_ids.txt` of unmatched probes. It is the supported
  entry point; the CSV variant is retained for pre-Parquet outputs only.

See [`annotation.md`](annotation.md) for annotation sources, the BED6
contract, and the probe-gene model.

## Mapping post-processing

* `tools/mergeOutputs.py` — merge per-chunk CSV/Parquet outputs into a
  single file (skips `sample_reservoir.csv`).
* `tools/recalculate_pvalues_parquet.py` / `recalculate_pvalues.py` —
  recompute p-values from t-statistics with high precision, replacing the
  normal-CDF approximation with Student's-t.
* `tools/summarizeOutput_parquet.py` / `summarizeOutput.py` — global
  BH-FDR, top-hits table, QQ / histogram / saliency plots, and regional FDR
  summaries. (Functional/ENCODE enrichment lives in `tools/runEnrichment.py`.)
* `tools/runEnrichment.py` — standalone functional (Enrichr/`gseapy`/`mygene`)
  and optional ENCODE ChromHMM enrichment of significant genes. Reads the FDR
  summary (`--fdr-input summarized.parquet`) and/or the bootstrap IG ranking
  (`--ig-input bootstrap_merged.parquet`) selected via `--rank-by fdr ig`, and
  is run as the final stage of `pipelinePost.sh`.
* `tools/summaryParquetToCsv.py` — Parquet→CSV converter for summary
  files.
* `tools/summarizeEnrichment.py` — self-contained HTML summary of the
  enrichment results.
* `tools/chromatin_features.py` / `tools/chromatinEnrichment_parquet.py` —
  interval index over chromatin-feature tracks and the Kennedy Fig. 6
  chromatin enrichment (Fisher exact statistics, BH q-values, and a
  `--plot` two-panel heatmap coloured by log odds ratio).

## Influence and permutation diagnostics

* `tools/flagInfluence_parquet.py` — derive `mt_influence_flag` from the
  mapper's `mt_h_max` leverage column and emit an influence QC JSON.
* `tools/calibration_bridge.py`, `tools/fig_influence_dose_response.py`,
  `tools/influence_diagnostic_panels.py`, `tools/influence_pair_anatomy.py`,
  `tools/diagnose_se_ratio_trend.py`, `tools/se_ratio_trend_report.py` —
  influence calibration against bootstrap fragility, dose-response and
  SE-ratio figures and reports.
* `tools/influence_qc_report.py` — consolidated FastQC-style HTML report over
  the influence artifacts.
* `tools/eval_permute.py` — read-only audit of a `qr_permute` parquet,
  producing `eval_permute_report.json`.
* `tools/summarize_permute.py` / `tools/read_permute_diagnostics.py` /
  `tools/plot_permute_diagnostics.py` — permutation summaries, the 7-way
  region table, and diagnostic plots across cohorts.
* `tools/permute_qc_report.py` — self-contained HTML QC report for a
  permutation run.
* `tools/annotate_permute_p.py` / `tools/join_precise_p_permute.py` — write
  `p_permute` / `fdr_permute` onto the mainline catalogs.
* `tools/build_gene_anchored_master.py` — assemble the cis near-gene pairs and
  the reservoir trans/distal pairs into the master scored by `qr_permute`.
* `tools/reservoir_to_parquet.py` — convert `sample_reservoir.csv` into a
  master parquet.
* `tools/compare_perm_vs_analytic.py` — compare permutation and analytic
  p-values.

See [`mlr_qr_permute.md`](mlr_qr_permute.md) for the permutation backend's
design and output columns.

## Bootstrapping

* `tools/createBootstrapList.py` — pick the top hits (by p-value, with
  per-region floors and caps) to feed the `qr_bootstrap` MLR backend.
* `tools/annotate_bootstrap_concordance.py` — raw bootstrap / analytic
  concordance scores and a distribution summary.

See [`bootstrap_qr_unification.md`](bootstrap_qr_unification.md) for the
shared QR path behind the `qr` and `qr_bootstrap` backends.

## Visualization and network analysis

* `tools/plotCircos.py` — Circos plots of the eQTM architecture. Uses the
  hg19 UCSC `cytoBand.txt` (downloaded automatically by `pipelinePost.sh`)
  and reports detailed reasons for excluded CpG-Gene pairs.
* `tools/visualizeFindings.py` — volcano, Manhattan, and scatter plots;
  emits a full set of plots for each available p-value column
  (`p_boot`, `precise_mt_p`, `mt_p`) with prefixed filenames.
* `tools/evaluateSaliency.py` — integrated-gradients saliency diagnostics,
  with an optional `--frac-exclude` pass that removes expression-derived IG
  from the saliency denominator.
* `tools/ig_qc_report.py` — self-contained HTML QC report over the IG
  columns: coverage, whether `|mt_ig|` is an independent ranking axis
  (against `|t|`), what drives its magnitude, the methylation share of
  attribution — with an optional `--frac-exclude` (e.g. `'Exp_PC*_ig'`)
  second denominator reported alongside the raw one — and per-region IG.
* `tools/plotRegionProportions.py` — regional composition plots.
* `tools/exportBipartiteNetwork.py` — Cytoscape-formatted node and edge
  tables (with optional `--min-effect`, `--max-boot-p`, `--max-fdr`, and
  `--top-k` filtering and an explicit `--out-dir`).
* `tools/visualizeBipartiteNetwork.py` — ForceAtlas2-based energy-minimized
  bipartite network, UMAP of regulatory β-diversity, regulatory degree
  distribution, clustered bipartite adjacency heatmap, a signed `mt_est`
  heatmap, a hypergeometric gene–gene projection, `--per-region` stratified
  figures, and arc diagrams; handles duplicate edges by keeping the
  maximum-weight pair.

See [`integrated_gradients.md`](integrated_gradients.md) for how the IG
saliency columns are computed.

## Benchmarking and profiling

* `pipelineBenchmarkKennedy.sh` / `tools/benchmark_kennedy.py` — comparison
  against the Kennedy et al. benchmark, standardizing thresholds (1e-5 and
  1e-11) across cohorts, with an eligibility decomposition (testable vs
  blacklisted vs otherwise absent Kennedy pairs), a probe-blacklist audit,
  effect-size / t-statistic / sign concordance on the shared pairs,
  influence-stratified recovery, and a region-composition crosswalk to
  Kennedy's four categories.
* `tools/diagnose_overlap.py` / `tools/check_catalog_grid.py` — overlap and
  catalog-grid consistency diagnostics for benchmark comparisons.
* `tools/io_microbench.py` — IO microbenchmarks for the save pool.
* `profiling.sh` — bottleneck diagnostic harness; see
  [`profiling.md`](profiling.md).
