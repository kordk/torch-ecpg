# eCpG Filtering & Prioritization — Living Document

> **Status:** Living document. Update this file whenever the pipeline scripts
> (`pipeline.sh`, `pipelinePost.sh`), the `tecpg` package, or any of the
> `tools/*.py` helpers change the way eCpGs are computed, filtered, prioritized,
> tested for enrichment, or visualized.
>
> **Scope:** Describes the end-to-end data flow for the **GTP** dataset run with
> the **`all`** mapping mode (`./pipeline.sh -d gtp -m all` followed by
> `./pipelinePost.sh gtp`). The `cis`/`distal`/`trans` modes share the same
> machinery; only the candidate-pair set differs.

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
   Expression  (G) ─┼──►  MLR (lstsq) ──►  eCpG table  ──►  filter / prioritize  ──►  figures + network
   Covariates  (C) ─┘        + IG          (per pair)        (regions, p, FDR,
                                                              precise p, bootstrap)
```

---

## 2. Input data (GTP, `all`)

| File (in `data_gtp/`) | Shape | Meaning |
|-----------------------|-------|---------|
| `M.csv`  | CpG loci × samples | Methylation β/M-values (post blacklist filter). Source `tecpg/config.py:10`. |
| `G.csv`  | Genes × samples    | Gene expression (HT-12). |
| `C.csv`  | Samples × covariates | Covariates + EpiDISH cell proportions + Expression/Methylation PCs. |
| `annot_gtp/M.bed6` | — | CpG genomic coordinates (EPIC, hg19). |
| `annot_gtp/G.bed6` | — | Gene genomic coordinates (HT-12, hg19). |

Samples are intersected and aligned so that `M.columns == G.columns == C.index`
(`tecpg/processing.py:175`).

Degrees of freedom used for precise p-values:
`DF = SAMPLES − COVARS − 2` (subtracting the methylation term and the
intercept), computed dynamically in `pipeline.sh:275-284`.

---

## 3. End-to-end stage map

### 3a. `pipeline.sh` — mapping & prioritization (9 stages)

```
 ┌────────────────────────────────────────────────────────────────────────────┐
 │ STAGE                     TOOL / COMMAND                 KEY OUTPUT        │
 ├────────────────────────────────────────────────────────────────────────────┤
 │ [1]   prep                tecpg data gtp                 M/G/C_orig.csv    │
 │       + blacklist         tools/exclude_blacklisted_…    M.csv             │
 │       + QC                tools/exploreOmics.py          data_gtp/qc/      │
 │ [1.5] cell_prop           tools/estimateCellProportions  C_post_cellTypes  │
 │ [2]   pca                 tools/residualize_pca.sh (G,M) C.csv (+PCs)      │
 │ [3]   map  ★              tecpg run mlr --lstsq --all     output chunks    │
 │             --compute-ig    (per-pair betas, t, p, IG)                     │
 │ [4]   merge               tools/mergeOutputs.py          merged.parquet    │
 │ [5]   annotate ★          tools/assignRegionToEcpg_…     annotated.parquet │
 │ [6]   precise_p ★         tools/recalculate_pvalues_…    annotated_pcalc   │
 │ [7]   summarize ★         tools/summarizeOutput_parquet  summarized.parquet│
 │             + FDR + plots   (BH-FDR, QQ, hist, saliency, enrichment)       │
 │ [8]   boot_list ★         tools/createBootstrapList.py   bootstrap_list.csv│
 │ [9]   bootstrap ★         tecpg run mlr --lstsq_bootstrap bootstrap_merged │
 └────────────────────────────────────────────────────────────────────────────┘
   ★ = a filtering / prioritization / statistics step (detailed below)
```

Stages are restartable via `--start-stage` (`pipeline.sh:40`,
order: `prep → cell_prop → pca → map → merge → annotate → precise_p →
summarize → boot_list → bootstrap`).

### 3b. `pipelinePost.sh` — visualization & network (5 stages)

Input: `output_gtp/bootstrap_merged.parquet` (the master table after Stage 9).

```
 [1] cytoBand.txt           (download hg19 cytobands from UCSC)
 [2] plotCircos.py          → output_gtp/plots/circos_*.{png,pdf}
 [3] visualizeFindings.py   → output_gtp/plots/{volcano,manhattan,region,scatter}*
 [4] exportBipartiteNetwork → output_gtp/network/cytoscape_{nodes,edges}.csv
 [5] visualizeBipartiteNetwork → output_gtp/network/*.png
```

Network export filter defaults (`pipelinePost.sh:24-25`):
`NETWORK_TOP_K=500`, `NETWORK_MAX_BOOT_P=0.05`.

---

## 4. Per-pair statistics (MLR `lstsq`)

`tecpg run mlr --mlr-method lstsq --all --compute-ig` fits, for each (CpG j,
gene i) pair:

```
   G_i  =  β0 (intercept)  +  β_mt · M_j  +  Σ β_k · C_k   +  ε
```

For the methylation term (`mt`) it emits four statistics, plus IG
(`tecpg/processing.py:316-319, 709-710`):

| Column | Meaning |
|--------|---------|
| `mt_est` | β coefficient (effect size) of methylation on expression |
| `mt_err` | standard error of β |
| `mt_t`   | t-statistic = `mt_est / mt_err` |
| `mt_p`   | p-value (fast normal-CDF approximation) |
| `mt_ig`  | Integrated Gradients saliency (see §5) |

(`--compute-ig` also emits per-covariate `*_ig` columns; with covariates the
`*_est/_err/_t/_p` quadruple is also produced for `const` and each covariate.)

### Integrated Gradients (saliency)

Analytical IG (`tecpg/processing.py:715-722`):

```
   IG_analytical = mean_centered(X) · |β|        (intercept excluded)
```

This is the per-pair feature-importance / saliency used downstream to rank
edges for the Circos plot and network export. A slower Captum-based variant
(`--compute-ig-deep`, requires `--p-thresh`) is available but **not** used by
`pipeline.sh`.

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
        │  Stage 7: Benjamini-Hochberg global FDR (fdr_est, is_significant)
        ▼   FDR < 0.05  (and FDR < 0.01 reported)
   Significant eCpGs
        │  Stage 8: top % per region (ranked by p / IG / |est|)
        ▼   default --percent 0.10, capped --max-per-region 2000
   Bootstrap candidate list  (bootstrap_list.csv)
        │  Stage 9: empirical bootstrap (100 iters) → p_boot, CI
        ▼
   Robust eCpGs (used as master table for post-processing)
        │  pipelinePost network export: top-k 500 AND p_boot ≤ 0.05
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

Output adds CpG/gene coordinates + `region`; unmapped IDs are logged to
`annotation_missing_ids.txt`.

### 5b. Precise p-values — `recalculate_pvalues_parquet.py`

Recomputes a high-precision two-sided p from `mt_t` using the Student-t
survival function at the pipeline DF (`recalculate_pvalues_parquet.py:79`):

```
   precise_mt_p = 2 · sf(|mt_t|, df)        (float64, chunked)
```

### 5c. Global FDR — `summarizeOutput_parquet.py`

Benjamini-Hochberg using the genome-wide `--total-tests` (extracted
dynamically from the mlr log, `pipeline.sh:305-315`), so FDR is valid even
though only a filtered subset is materialized
(`summarizeOutput_parquet.py:471-517`):

```
   fdr_est_i = p_i · total_tests / rank_i        (monotone step-down → q-values)
   is_significant = fdr_est < 0.05               (0.01 also reported)
```

Genomic inflation **λ_GC** is estimated from ~1.2M reservoir-sampled p-values
(`:311-315, 416-419`):

```
   λ_GC = median(χ²_obs, df=1) / 0.4549
```

### 5d. Bootstrap prioritization — `createBootstrapList.py` + `lstsq_bootstrap`

`createBootstrapList.py` selects the top **`--percent` (default 10%)** of hits
**per region**, hard-capped at **`--max-per-region` (default 2000)**, ranked by
one of: `p-value` (asc), `ig_score` (`mt_ig` desc), or `magnitude` (`|mt_est|`
desc). Pairs are globally de-duplicated. `pipeline.sh:377` ranks by `p-value`.

`tecpg run mlr --mlr-method lstsq_bootstrap` (100 iterations, batch 10) then
resamples samples with replacement and emits (`tecpg/bootstrap.py:200-235`):

| Column | Meaning |
|--------|---------|
| `mt_est_boot_mean` | mean β across bootstrap resamples |
| `mt_est_boot_std`  | std of β (robustness) |
| `ci_low`, `ci_high`| 2.5% / 97.5% percentile CI |
| `p_boot` | empirical two-sided p = `2 · min(P(β≤0), P(β≥0))` |
| `<covariate>_ig` | per-feature integrated gradients for covariates (enabled by `BOOTSTRAP_IG_COVARIATES="all"` in Stage 9) |

Results are left-joined onto the master parquet → `bootstrap_merged.parquet`.

---

## 6. Enrichment tests

| Test | Where | Method | Output |
|------|-------|--------|--------|
| ENCODE ChromHMM enrichment | `summarizeOutput_parquet.py:106-132, 650-850` | **Fisher's exact** of significant-eCpG overlap with ENCODE chromatin-state track vs. background BED (`--encode-enrichment`, `--background-bed`). Fold-enrichment + BH-adjusted p. | `encode_enrichment_results.csv`, `encode_enrichment_heatmap.png` (log2 fold enrichment by region × state) |
| Functional / pathway enrichment | `summarizeOutput_parquet.py:856-933` | **Enrichr** (`gseapy`) on significant genes per region against `GO_Biological_Process_2021`, `KEGG_2021_Human`, `WikiPathways_2021_Human`; keep Adj-P < 0.05. | `enrichment_results/{region}_{library}_enrichment.csv` + plots |
| Gene–gene co-regulation | `visualizeBipartiteNetwork.py:404-431` | **Hypergeometric** test on shared CpG targets when building the unipartite gene projection; edge weight = −log10(p) (capped 100). | edges of `UnipartiteProjection.png` |

---

## 7. Visualizations

### 7a. Diagnostics (Stage 7, `summarizeOutput_parquet.py`)

| Figure | Content |
|--------|---------|
| `qq_plot.png` | observed vs expected −log10(p) with λ_GC |
| `p_value_histogram.png` | 100-bin p-value distribution |
| `saliency_profile_top50.png` | stacked IG feature contributions for top-50 hits |

### 7b. Findings (`visualizeFindings.py --all`)

Auto-selects the best p-column (`p_boot` → `precise_mt_p` → `mt_p`) and prefixes
filenames accordingly (`:456-468`). Stratified subsampling thresholds:
`HITS_P=1e-5`, `SIGNAL_P=0.05`, signal frac 5%, noise frac 0.5% (`:16-19`).

| Figure | Content |
|--------|---------|
| `*_volcano_plot.png` | effect size (`mt_est`) vs −log10(p), colored by region; top-10 genes labeled |
| `*_manhattan_plot.png` | CpG chromosomal position vs −log10(p) with `1e-5` threshold line |
| `*_region_breakdown.png` | counts of significant hits per region |
| `comparative_scatter_*` / `scatter_*` | raw vs covariate-adjusted M-vs-G scatter (Pearson r, regression line) for top hits |

### 7c. Circos (`plotCircos.py`)

Curved CpG→gene links scaled by `mt_ig` saliency, red (`mt_est>0`) / blue
(`mt_est<0`), over 1 Mb density tracks. `--top-n` 5000 links, `--top-n-trans`
2000 (`:50,56`).

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

Outputs: `circos_top_saliency.{png,pdf}`, `circos_trans_only.{png,pdf}`,
`circos_diagnostic.log`.

---

## 8. Network nodes & edges

### 8a. Export — `exportBipartiteNetwork.py`

Filter order (`:51-80`): `--min-effect` (|`mt_est`|) → `--max-boot-p`
(`p_boot`) → `--top-k` ranked by `mt_ig` saliency (fallback |`mt_t`|).
`pipelinePost.sh` uses top-k 500, max-boot-p 0.05.

```
   cytoscape_edges.csv                         cytoscape_nodes.csv
   ┌────────────────────────────────┐          ┌──────────────────────────────┐
   │ Source (mt_id)                 │          │ Node_ID                      │
   │ Target (gt_id)                 │          │ Chrom, Start, Strand         │
   │ Interaction (region)           │          │ Node_Type  = CpG | Gene      │
   │ mt_est, mt_p, mt_t, abs_t      │          │ Region     (CpG only)        │
   │ fdr_est, *_ig                  │          │                              │
   └────────────────────────────────┘          └──────────────────────────────┘
```

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

Edge weight = `mt_ig` (fallback `abs_t`); `--threshold` default 0.5; duplicate
edges resolved by max weight (→ `dropped_duplicate_edges.csv`).

| Figure | Content |
|--------|---------|
| `EnergyMinimizedBipartiteNetwork.png` | ForceAtlas2 layout, CpGs (by region) ↔ genes |
| `UMAPofRegulatoryBetaDiversity.png` | UMAP (Bray-Curtis) of CpG regulatory profiles |
| `RegulatoryDegreeDistribution.png` | KDE of CpG out-degree, faceted by region |
| `BiclusteredBiAdjacencyHeatmap.png` | hierarchically clustered CpG×Gene weight heatmap |
| `ArcDiagram.png` | bipartite arc layout, arc height ∝ weight |
| `UnipartiteProjection.png` | gene–gene projection (hypergeometric-weighted, see §6) |

---

## 9. Key thresholds & defaults (quick reference)

| Parameter | Value | Source |
|-----------|-------|--------|
| Region annotation p cutoff | `1e-6` | `assignRegionToEcpg_parquet.py:19` |
| Promoter window | ±2,500 bp | `:30-31` |
| Cis window | 50,000 bp | `:26` |
| Distal offset | 50,000 bp | `:22` |
| FDR significance | `< 0.05` (and `< 0.01`) | `summarizeOutput_parquet.py:473-474` |
| Bootstrap list percent / cap | 10% / 2000 per region | `createBootstrapList.py:14-17` |
| Bootstrap iterations / batch | 100 / 10 | `pipeline.sh:387` |
| Network top-k / max boot p | 500 / 0.05 | `pipelinePost.sh:24-25` |
| Circos top-n / top-n-trans | 5000 / 2000 | `plotCircos.py:50,56` |
| Enrichr libraries | GO BP 2021, KEGG 2021, WikiPathways 2021 | `summarizeOutput_parquet.py:863` |

---

## 10. Maintenance checklist

When changing the pipeline, update the relevant section here:

- [ ] New/renamed stage in `pipeline.sh` / `pipelinePost.sh` → §3
- [ ] New per-pair statistic or IG change → §4
- [ ] Changed threshold, region rule, FDR, or bootstrap behavior → §5, §9
- [ ] New or modified enrichment test → §6
- [ ] New or renamed figure / network attribute → §7, §8
