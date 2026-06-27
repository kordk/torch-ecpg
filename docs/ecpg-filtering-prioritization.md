# eCpG Filtering & Prioritization — Living Document

> **Verified against:** `kordk/torch-ecpg` branch `dev`, HEAD
> `8145d172c3e02d5c74f84a31cc3b488f2c8bc316`. All `path:line` citations below are
> reproducible at that commit.
>
> **Status:** Living document. Update this file whenever the pipeline scripts
> (`pipelinePre.sh`, `pipeline.sh`, `pipelinePost.sh`), the `tecpg` package, or any
> of the `tools/*.py` helpers change the way eCpGs are computed, filtered,
> prioritized, tested for enrichment, or visualized.
>
> **Scope:** Describes the end-to-end data flow for the **GTP** dataset run with
> the **`all`** mapping mode: `./pipelinePre.sh --dataset gtp` (preprocessing),
> then `./pipeline.sh -d gtp -m all` (mapping & prioritization), then
> `./pipelinePost.sh gtp` (visualization, network & enrichment). The
> `cis`/`distal`/`trans` modes share the same machinery; only the candidate-pair
> set differs.

---

## 1. What this pipeline produces

Torch-eCpG maps **expression Quantitative Trait Methylation (eQTM)**
associations: for every candidate (CpG, gene) pair it fits a multiple linear
regression of gene expression on methylation (adjusted for covariates), then
filters and prioritizes the resulting **eCpGs** through a sequence of
statistical and biological layers, and finally visualizes them as plots and a
bipartite CpG↔Gene network.

```
   Methylation (M) ─┐
   Expression  (G) ─┼──►  MLR (qr) ──►  eCpG table  ──►  filter / prioritize  ──►  figures + network
   Covariates  (C) ─┘        + IG          (per pair)        (regions, p, FDR,
                                                              precise p, bootstrap)
```

---

## 2. Input data (GTP, `all`)

| File (in `data_gtp/`) | Shape | Meaning |
|-----------------------|-------|---------|
| `M.csv`  | CpG loci × samples | Methylation β/M-values (post blacklist filter). Default name `tecpg/config.py:10`. |
| `G.csv`  | Genes × samples    | Gene expression (HT-12). |
| `C.csv`  | Samples × covariates | Covariates + EpiDISH cell proportions + Expression/Methylation PCs. Produced by `pipelinePre.sh:191-221`. |
| `annot_gtp/M.bed6` | — | CpG genomic coordinates (EPIC, hg19). |
| `annot_gtp/G.bed6` | — | Gene genomic coordinates (HT-12, hg19). |

Sample labels (columns of M and G, index of C) are verified and **trimmed to the
shared intersection** so that `M.columns == G.columns == C.index`
(`tecpg/helper.py:84-130`, intersection at `:100-107`).

Degrees of freedom used for precise p-values:
`DF = SAMPLES − COVARS − 2` (subtracting the methylation term and the
intercept), computed dynamically in `pipeline.sh:160-207` (`DF=$((SAMPLES -
COVARS - 2))` at `:194`). `pipelinePre.sh` records the expected `(samples,
covars)` shape in `C.shape.meta` so `pipeline.sh` can cross-check it before
deriving DF (`pipeline.sh:171-192`).

---

## 3. End-to-end stage map

The original single pipeline has been **split into three drivers**. Preprocessing
(prep / cell-proportion / PCA) now lives in `pipelinePre.sh`; mapping &
prioritization in `pipeline.sh`; visualization, network & enrichment in
`pipelinePost.sh`. (The scripts still print legacy `[n/9]` log labels.)

### 3a. `pipelinePre.sh` — preprocessing (3 stages)

```
 ┌────────────────────────────────────────────────────────────────────────────┐
 │ STAGE                     TOOL / COMMAND                 KEY OUTPUT        │
 ├────────────────────────────────────────────────────────────────────────────┤
 │ [1]   prep                tecpg data gtp                 M/G/C_orig.csv    │
 │       + blacklist         tools/exclude_blacklisted_…    M.csv             │
 │       + QC                tools/exploreOmics.py          data_gtp/qc/      │
 │ [1.5] cell_prop           tools/estimateCellProportions  C_post_cellTypes  │
 │ [2]   pca                 tools/residualize_pca.sh (G,M) C.csv (+PCs)      │
 └────────────────────────────────────────────────────────────────────────────┘
```

Restartable via `--start-stage` (`pipelinePre.sh:43`,
order: `prep → cell_prop → pca`). The PCA stage also writes `C.shape.meta`
(`pipelinePre.sh:217-218`).

### 3b. `pipeline.sh` — mapping & prioritization (7 stages, labeled [3]–[9])

```
 ┌────────────────────────────────────────────────────────────────────────────┐
 │ STAGE                     TOOL / COMMAND                 KEY OUTPUT        │
 ├────────────────────────────────────────────────────────────────────────────┤
 │ [3]   map  ★              tecpg run mlr --mlr-method qr --all     output chunks    │
 │             --compute-ig    (per-pair betas, t, p, IG)                     │
 │ [4]   merge               tools/mergeOutputs.py          merged.parquet    │
 │ [5]   annotate ★          tools/assignRegionToEcpg_…     annotated.parquet │
 │ [6]   precise_p ★         tools/recalculate_pvalues_…    annotated_pcalc   │
 │ [7]   summarize ★         tools/summarizeOutput_parquet  summarized.parquet│
 │             + FDR + plots   (BH-FDR, QQ, hist, saliency)                   │
 │ [8]   boot_list ★         tools/createBootstrapList.py   bootstrap_list.csv│
 │ [9]   bootstrap ★         tecpg run mlr --mlr-method qr_bootstrap bootstrap_merged │
 └────────────────────────────────────────────────────────────────────────────┘
   ★ = a filtering / prioritization / statistics step (detailed below)
```

`pipeline.sh` requires `M.csv`, `G.csv`, `C.csv`, `G.bed6`, `M.bed6` to already
exist (produced by `pipelinePre.sh`; checked at `pipeline.sh:145-151`).
Restartable via `--start-stage` (`pipeline.sh:85`, valid options
`all, map, merge, annotate, precise_p, summarize, boot_list, bootstrap`; order:
`map → merge → annotate → precise_p → summarize → boot_list → bootstrap`).

### 3c. `pipelinePost.sh` — visualization, network & enrichment (7 stages)

Input: `output_gtp/bootstrap_merged.parquet` (master table after Stage 9,
checked at `pipelinePost.sh:33`) and `output_gtp/summarized.parquet` (FDR summary
from Stage 7).

```
 [1] cytoBand.txt           (download hg19 cytobands from UCSC)
 [2] plotCircos.py          → output_gtp/plots/circos_*.{png,pdf}
 [3] visualizeFindings.py   → output_gtp/plots/{volcano,manhattan,region,scatter}*
 [4] exportBipartiteNetwork → output_gtp/network/cytoscape_{nodes,edges}.csv
 [5] visualizeBipartiteNetwork → output_gtp/network/*.png
 [6] evaluateSaliency.py    → output_gtp/plots/saliency_*
 [7] runEnrichment.py       → output_gtp/enrichment/ (functional + optional ENCODE, CSV only)
```

Network export filter defaults (`pipelinePost.sh:26-27`):
`NETWORK_TOP_K=5000`, `NETWORK_MAX_BOOT_P=0.05`.

---

## 4. Per-pair statistics (MLR `qr`)

`tecpg run mlr --mlr-method qr --all --compute-ig` fits, for each (CpG j,
gene i) pair:

```
   G_i  =  β0 (intercept)  +  β_mt · M_j  +  Σ β_k · C_k   +  ε
```

(The CLI accepts three MLR methods: `legacy_normal_eq`, `qr`, `qr_bootstrap`
— `tecpg/cli.py:791`.)

For the methylation term (`mt`) it emits four statistics, plus IG
(`tecpg/processing.py:340-343, 361-362`):

| Column | Meaning |
|--------|---------|
| `mt_est` | β coefficient (effect size) of methylation on expression |
| `mt_err` | standard error of β |
| `mt_t`   | t-statistic = `mt_est / mt_err` |
| `mt_p`   | p-value (fast normal-CDF approximation) |
| `mt_ig`  | Integrated Gradients saliency (see below) |

(`--compute-ig` also emits per-covariate `*_ig` columns when IG covariates are
requested; with covariates the `*_est/_err/_t/_p` quadruple is also produced for
`const` and each covariate.)

### Integrated Gradients (saliency)

Analytical IG (`tecpg/processing.py:512-517, 731-733`):

```
   IG_analytical = mean_s|X − X̄| · |β|        (intercept excluded)
```

where `X̄` is the per-feature mean baseline (`X.mean(dim=1)`, the default
`ig_baseline='mean'`), `mean_s|X − X̄|` is the per-feature mean absolute
deviation over samples, and `|β|` is the absolute regression coefficient. The
intercept (index 0) is dropped (`B[:, :, 1:]`).

This is the per-pair feature-importance / saliency used downstream to rank
edges for the Circos plot and network export. A slower Captum-based variant
(`--compute-ig-deep`, requires `--p-thresh`; `tecpg/cli.py:886,1135-1136`,
`tecpg/bootstrap.py:218`) is available but **not** used by `pipeline.sh`.

---

## 5. Filtering & prioritization layers

Each layer narrows or re-ranks the eCpG set. From most permissive to most
selective:

```
   ALL CpG×Gene pairs (mlr --all)
        │  Stage 5: region annotation requires p ≤ 1e-6 (PVALCUTOFF)
        ▼
   Pairs with assigned region  (PROMOTER / GENEBODY / CIS5 / CIS3 /
        │                        DISTAL5 / DISTAL3 / TRANS)
        │  Stage 6: precise float64 two-sided t p-value (precise_mt_p)
        ▼
   Pairs with precise_mt_p
        │  Stage 7: Benjamini-Hochberg global FDR (fdr_est)
        ▼   FDR < 0.05  (and FDR < 0.01 reported)
   Significant eCpGs
        │  Stage 8: top hits per region (ranked by p / IG / |est|);
        ▼   pipeline run uses --rank-by p-value, --min-per-region 4500,
        │   --max-per-region 10000 (tool defaults: 10%, min 200, cap 2000)
   Bootstrap candidate list  (bootstrap_list.csv)
        │  Stage 9: empirical bootstrap (1000 iters, batch 10) → p_boot, CI
        ▼
   Robust eCpGs (used as master table for post-processing)
        │  pipelinePost network export: top-k 5000 AND p_boot ≤ 0.05
        ▼
   Network nodes & edges
```

### 5a. Region assignment — `assignRegionToEcpg_parquet.py`

Only pairs with `mt_p ≤ PVALCUTOFF = 1e-6` are annotated
(`assignRegionToEcpg_parquet.py:19`). Region is strand-aware; thresholds
(`:22-31`): promoter ±2,500 bp around TSS, cis window 50,000 bp, distal beyond.

ASCII view of regions relative to a **+strand** gene (TSS at left):

```
        −50kb        −2.5kb    TSS=========gene body=========TES      +50kb
   ───────┼──────────────┼────────┼────────────────────────────┼─────────┼──────►
  DISTAL5 │    CIS5      │ PROMOT.│         GENEBODY           │  CIS3   │ DISTAL3
          │ (upstream)   │ ±2.5kb │                            │(downstr)│
```

| Region | Definition (relative to gene, strand-aware) |
|--------|---------------------------------------------|
| `PROMOTER` | within ±2,500 bp of the TSS |
| `GENEBODY` | inside the gene body |
| `CIS5` | 2.5 kb–50 kb upstream of TSS |
| `CIS3` | 0–50 kb downstream of gene end |
| `DISTAL5` / `DISTAL3` | >50 kb up/downstream, same chromosome |
| `TRANS` | CpG and gene on **different chromosomes** (`:290`) |

Positive-strand region logic is at `:308-326`; negative-strand at `:340-358`.
The script also normalizes `chr`-prefix mismatches between the two annotations
(`:434-452`) and strips Ensembl ID version suffixes when the GTF is versioned
but the parquet is not (`:477-485`).

Output adds CpG/gene coordinates + `region`; unmapped IDs are logged to
`annotation_missing_ids.txt` (`:407`).

### 5b. Precise p-values — `recalculate_pvalues_parquet.py`

Recomputes a high-precision two-sided p from `mt_t` using the Student-t
survival function at the pipeline DF (`recalculate_pvalues_parquet.py:79`):

```
   precise_mt_p = 2 · sf(|mt_t|, df)        (float64, chunked)
```

### 5c. Global FDR — `summarizeOutput_parquet.py`

Benjamini-Hochberg using the genome-wide `--total-tests` (extracted
dynamically from the mlr log, `pipeline.sh:237-253`), so FDR is valid even
though only a filtered subset is materialized
(`summarizeOutput_parquet.py:446-454`):

```
   fdr_est_i = p_i · total_tests / rank_i        (monotone step-down → q-values)
```

The pipeline runs with `--calculate-fdr`, so `summarized.parquet` carries a
`fdr_est` column; both FDR < 0.05 and FDR < 0.01 thresholds are reported
(`:410-411`). (A separate, mutually-exclusive `--assign-fdr-passfail` mode would
instead append a boolean `is_significant = precise_mt_p ≤ p_max_fdr` column,
`:550`; the pipeline does not use it.)

Genomic inflation **λ_GC** is estimated globally from a reservoir targeting
~1,000,000 p-values (1.2× oversampling probability, `:204-206`), computed at
`:341-346`:

```
   λ_GC = median(χ²_obs, df=1) / 0.4549
```

### 5d. Bootstrap prioritization — `createBootstrapList.py` + `qr_bootstrap`

`createBootstrapList.py` selects the top **`--percent` (default 10%)** of hits
**per region**, floored at **`--min-per-region` (default 200)** and hard-capped
at **`--max-per-region` (default 2000)**, ranked by one of: `p-value` (asc),
`ig_score` (`mt_ig` desc), or `magnitude` (`|mt_est|` desc)
(`createBootstrapList.py:12-18`). Pairs are globally de-duplicated. The GTP
pipeline run **overrides** the per-region bounds: `--rank-by p-value
--min-per-region 4500 --max-per-region 10000` (`pipeline.sh:317`).

`tecpg run mlr --mlr-method qr_bootstrap` (1000 iterations, batch 10,
`pipeline.sh:333`) then resamples samples with replacement and emits
(`tecpg/bootstrap.py:303-363`):

| Column | Meaning |
|--------|---------|
| `mt_est_boot_mean` | mean β across bootstrap resamples |
| `mt_est_boot_std`  | std of β (robustness) |
| `ci_low`, `ci_high`| 2.5% / 97.5% percentile CI |
| `p_boot` | empirical two-sided p = `2 · min(P(β≤0), P(β≥0))`, floored at `1/finite_count` |
| `degenerate_resamples` | count of degenerate (non-finite) resamples for the pair |
| `<covariate>_ig` | per-feature integrated gradients for covariates (enabled by `BOOTSTRAP_IG_COVARIATES="all"` in Stage 9, `pipeline.sh:32`) |

Results are left-joined onto the master parquet → `bootstrap_merged.parquet`
(`tecpg/bootstrap.py:402`).

---

## 6. Enrichment tests

Functional and ENCODE enrichment were moved out of `summarizeOutput_parquet.py`
into the standalone `tools/runEnrichment.py`, which runs as the final stage of
`pipelinePost.sh` (Stage 7). It draws significant genes from the FDR summary
(`summarized.parquet`, via `--rank-by fdr`) and/or the bootstrap IG ranking
(`bootstrap_merged.parquet`, via `--rank-by ig`). The tool writes **CSV results
only — it produces no figures.**

| Test | Where | Method | Output |
|------|-------|--------|--------|
| ENCODE ChromHMM enrichment | `runEnrichment.py:249-343` | **Fisher's exact** of significant-eCpG overlap with ENCODE chromatin-state track vs. background BED (`--encode-enrichment`, `--background-bed`). Fold-enrichment + BH-adjusted p. | `encode_enrichment_results.csv` (`:341`) |
| Functional / pathway enrichment | `runEnrichment.py:345-419` | **Enrichr** (`gseapy`) on significant genes per region against `GO_Biological_Process_2021`, `KEGG_2021_Human`, `WikiPathway_2023_Human` (override with `--enrichment-libraries`, `:433`); keep Adj-P < 0.05. | `enrichment_results/{region}_{method}_{library}_enrichment.csv` (`:410`) |
| Gene–gene co-regulation | `visualizeBipartiteNetwork.py:404-430` | **Hypergeometric** test on shared CpG targets when building the unipartite gene projection; edge weight = −log10(p) (capped 100, `:430`). | edges of `UnipartiteProjection.png` |

---

## 7. Visualizations

### 7a. Diagnostics (Stage 7, `summarizeOutput_parquet.py`)

| Figure | Content |
|--------|---------|
| `qq_plot.png` | observed vs expected −log10(p) with λ_GC |
| `p_value_histogram.png` | 100-bin p-value distribution |
| `saliency_profile_top50.png` | stacked IG feature contributions for top-50 hits |

(Moved into `output_gtp/` by `pipeline.sh:307`.)

### 7b. Findings (`visualizeFindings.py --all`)

Generates a full plot set for **each available** p-column, prefixing filenames
accordingly: `p_boot` → `bootstrapP_`, `precise_mt_p` → `preciseP_`; `mt_p` →
`mtP_` is used only as a fallback when neither of the first two is present
(`:464-485`). Stratified subsampling thresholds: `HITS_P=1e-5`, `SIGNAL_P=0.05`,
signal frac 5%, noise frac 0.5% (`:16-19`).

| Figure | Content |
|--------|---------|
| `*_volcano_plot.png` | effect size (`mt_est`) vs −log10(p), colored by region; top-10 genes labeled |
| `*_manhattan_plot.png` | CpG chromosomal position vs −log10(p) with `1e-5` threshold line |
| `*_region_breakdown.png` | counts of significant hits per region |
| `comparative_scatter_*` / `scatter_*` | raw vs covariate-adjusted M-vs-G scatter (Pearson r, regression line) for top hits |

### 7c. Circos (`plotCircos.py`)

Curved CpG→gene links scaled by `mt_ig` saliency, red (`mt_est>0`) / blue
(`mt_est<0`), over 1 Mb density tracks. `--top-n` 5000 links, `--top-n-trans`
2000 (`:49,56`).

```
            chr1
        ___/    \___                 outer ring  = all-eCpG density (grey)
      /   .-link-.   \               inner ring  = top-saliency density (orange)
   chrY      ⌣        chr2           ribbons     = CpG↔gene, width ∝ mt_ig
     |      ⌢ ⌣ ⌢      |             red=+effect, blue=−effect
      \   '-link-'   /
        \___    ___/
            chrN
```

Outputs: `circos_top_saliency.{png,pdf}` (`:649`), `circos_trans_only.{png,pdf}`
(`:659`), `circos_diagnostic.log` (`:10`).

### 7d. Saliency evaluation (`evaluateSaliency.py`)

Run as Stage 6 of `pipelinePost.sh`; writes to `output_gtp/plots/`:
`saliency_profile_ranks_{start}_{end}.png` (`:291`), `saliency_fraction_hist.png`
(`:302`), `saliency_vs_{effect_col}.png` (`:313`),
`saliency_fraction_by_region.png` (`:325`), `saliency_fraction_decay_curve.png`
(`:344`), `saliency_magnitude_decay_curve.png` (`:360`). Default `--rank-by
mt_ig` (`:142`).

---

## 8. Network nodes & edges

### 8a. Export — `exportBipartiteNetwork.py`

Filter order (`:51-82`): `--min-effect` (|`mt_est`|) → `--max-boot-p`
(`p_boot`) → `--top-k` ranked by `mt_ig` saliency (fallback |`mt_t`|).
`pipelinePost.sh` uses top-k 5000, max-boot-p 0.05 (the tool's own `--top-k`
default is 10000, `:15`).

```
   cytoscape_edges.csv                         cytoscape_nodes.csv
   ┌────────────────────────────────┐          ┌──────────────────────────────┐
   │ Source (mt_id)                 │          │ Node_ID                      │
   │ Target (gt_id)                 │          │ Chrom, Start, Strand         │
   │ Interaction (region)           │          │ Node_Type  = CpG | Gene      │
   │ mt_est, mt_p, mt_t             │          │ Region     (CpG only)        │
   │ fdr_est, *_ig (abs_t only if   │          │                              │
   │   mt_ig absent)                │          │                              │
   └────────────────────────────────┘          └──────────────────────────────┘
```

(Edge columns built at `:86-98`; `abs_t` is appended only on the |`mt_t`|
fallback path, `:89-90`.)

Bipartite structure:

```
        CpG nodes              Gene nodes
       (left, by region)      (right)
        cg0001 ───── mt_ig ───► ENSG…A
        cg0002 ──┬────────────► ENSG…B
                 └────────────► ENSG…C
        cg0003 ───────────────► ENSG…B
        edge label = region (PROMOTER / CIS5 / TRANS …)
        edge weight = mt_ig saliency
```

### 8b. Network figures — `visualizeBipartiteNetwork.py`

Edge weight = `mt_ig` (fallback `abs_t`, `:48-53`); `--threshold` default 0.5
(`:21`); duplicate edges resolved by max weight (→ `dropped_duplicate_edges.csv`,
`:80-89`).

| Figure | Content |
|--------|---------|
| `EnergyMinimizedBipartiteNetwork.png` | ForceAtlas2 layout, CpGs (by region) ↔ genes (`:200`) |
| `UMAPofRegulatoryBetaDiversity.png` | UMAP (Bray-Curtis) of CpG regulatory profiles (`:232`) |
| `RegulatoryDegreeDistribution.png` | KDE of CpG out-degree, faceted by region (`:485`) |
| `BiclusteredBiAdjacencyHeatmap.png` | hierarchically clustered CpG×Gene weight heatmap (`:276`) |
| `ArcDiagram.png` | bipartite arc layout, arc height ∝ weight (`:360`) |
| `UnipartiteProjection.png` | gene–gene projection (hypergeometric-weighted, see §6; `:541`) |

---

## 9. Key thresholds & defaults (quick reference)

| Parameter | Value | Source |
|-----------|-------|--------|
| Region annotation p cutoff | `1e-6` | `assignRegionToEcpg_parquet.py:19` |
| Promoter window | ±2,500 bp | `:30-31` |
| Cis window | 50,000 bp | `:26` |
| Distal offset | 50,000 bp | `:22` |
| FDR significance | `< 0.05` (and `< 0.01`) | `summarizeOutput_parquet.py:410-411` |
| Bootstrap list defaults (percent / floor / cap) | 10% / 200 / 2000 per region | `createBootstrapList.py:14-18` |
| Bootstrap list values used by pipeline (min / max) | 4500 / 10000 per region | `pipeline.sh:317` |
| Bootstrap iterations / batch | 1000 / 10 | `pipeline.sh:333` |
| Network top-k / max boot p (pipeline) | 5000 / 0.05 | `pipelinePost.sh:26-27` |
| Network top-k (tool default) | 10000 | `exportBipartiteNetwork.py:15` |
| Circos top-n / top-n-trans | 5000 / 2000 | `plotCircos.py:49,56` |
| Enrichr libraries | GO BP 2021, KEGG 2021, WikiPathway 2023 | `runEnrichment.py:433` |

---

## 10. Maintenance checklist

When changing the pipeline, update the relevant section here:

- [ ] New/renamed stage in `pipelinePre.sh` / `pipeline.sh` / `pipelinePost.sh` → §3
- [ ] New per-pair statistic or IG change → §4
- [ ] Changed threshold, region rule, FDR, or bootstrap behavior → §5, §9
- [ ] New or modified enrichment test → §6
- [ ] New or renamed figure / network attribute → §7, §8
- [ ] Re-verify the dev HEAD hash and every `path:line` citation when lines drift
