# eCpG Filtering & Prioritization — Living Document

> **Verified against:** `kordk/torch-ecpg` branch `dev`, HEAD
> `c94ef0f1c84b6fb8c2ac8072a427956c3c2ce87c` (2026-07-30). All `path:line`
> citations below are reproducible at that commit.
>
> **Previous revision** was anchored at `8145d17` (2026-06-27), 216 commits
> behind. That revision predates the entire `qr_permute` subsystem, the
> `pipelinePermute.sh` driver, and the `evaluateSaliency.py` rewrite. Several of
> its claims about the filtering cascade were also incorrect **as of that
> commit** — see §11.
>
> **Status:** Living document. Update this file whenever the pipeline scripts
> (`pipelinePre.sh`, `pipeline.sh`, `pipelinePost.sh`, `pipelinePermute.sh`), the
> `tecpg` package, or any of the `tools/*.py` helpers change the way eCpGs are
> computed, filtered, prioritized, tested for enrichment, or visualized.
>
> **Scope:** Two purposes. §1–§5 and §7–§10 describe the end-to-end data flow for
> the **GTP** dataset run with the **`all`** mapping mode. **§6 is the decision
> document** for the outstanding question — how to select loci for evaluation
> after mapping, given three candidate axes (`precise_mt_p`/`fdr_est`, `p_boot`,
> `mt_ig`) that are not independent of one another. §11 records defects and
> code/doc drift found during this review.

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
                                              │
                                              └──►  qr_permute  ──►  calibration verdict
                                                   (separate chain, §5e)
```

---

## 2. Input data (GTP, `all`)

| File (in `data_gtp/`) | Shape | Meaning |
|-----------------------|-------|---------|
| `M.csv`  | CpG loci × samples | Methylation β/M-values (post blacklist filter). Default name `tecpg/config.py:10`. |
| `G.csv`  | Genes × samples    | Gene expression (HT-12). |
| `C.csv`  | Samples × covariates | Covariates + EpiDISH cell proportions + Expression/Methylation PCs. Produced by `pipelinePre.sh`. |
| `annot_gtp/M.bed6` | — | CpG genomic coordinates (EPIC, hg19). |
| `annot_gtp/G.bed6` | — | Gene genomic coordinates (HT-12, hg19). |

Sample labels (columns of M and G, index of C) are verified and **trimmed to the
shared intersection** so that `M.columns == G.columns == C.index`
(`tecpg/helper.py`).

Degrees of freedom used for precise p-values:
`DF = SAMPLES − COVARS − 2` (subtracting the methylation term and the
intercept), computed dynamically at `pipeline.sh:199`. `pipelinePre.sh` records
the expected `(samples, covars)` shape in `C.shape.meta` so `pipeline.sh` can
cross-check it before deriving DF (`pipeline.sh:174-197`); a mismatch aborts
rather than silently shifting DF.

---

## 3. End-to-end stage map

Four drivers. Preprocessing in `pipelinePre.sh`; mapping & prioritization in
`pipeline.sh`; visualization, network & enrichment in `pipelinePost.sh`;
permutation calibration in `pipelinePermute.sh` (an independent chain that
consumes a mapping artifact and produces a **verdict**, not a catalog — §5e).

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

Restartable via `--start-stage` (order: `prep → cell_prop → pca`). The PCA stage
also writes `C.shape.meta`.

### 3b. `pipeline.sh` — mapping & prioritization (7 stages, labeled [3]–[9])

```
 ┌────────────────────────────────────────────────────────────────────────────┐
 │ STAGE                     TOOL / COMMAND                 KEY OUTPUT        │
 ├────────────────────────────────────────────────────────────────────────────┤
 │ [3]   map  ★              tecpg run mlr --mlr-method qr --all              │
 │             --compute-ig    (per-pair betas, t, p, IG)   output chunks     │
 │                             + reservoir (pre-filter)     sample_reservoir  │
 │ [4]   merge               tools/mergeOutputs.py          merged.parquet    │
 │ [5]   annotate            tools/assignRegionToEcpg_…     annotated.parquet │
 │ [6]   precise_p ★         tools/recalculate_pvalues_…    annotated_pcalc   │
 │ [7]   summarize ★         tools/summarizeOutput_parquet  summarized.parquet│
 │             + FDR + plots   (BH-FDR, QQ, hist, saliency)                   │
 │ [8]   boot_list ★         tools/createBootstrapList.py   bootstrap_list.csv│
 │ [9]   bootstrap ★         tecpg run mlr --mlr-method qr_bootstrap          │
 │                                                          bootstrap_merged  │
 └────────────────────────────────────────────────────────────────────────────┘
   ★ = a filtering / prioritization / statistics step (detailed below)
```

Note that **[5] annotate is not marked ★**: it annotates, it does not filter
(§5a, §11-D1). `pipeline.sh` requires `M.csv`, `G.csv`, `C.csv`, `G.bed6`,
`M.bed6` to already exist. Restartable via `--start-stage` (`pipeline.sh:85`,
options `all, map, merge, annotate, precise_p, summarize, boot_list,
bootstrap`).

### 3c. `pipelinePost.sh` — visualization, network & enrichment (7 stages)

Input: `output_gtp/bootstrap_merged.parquet` (master table after Stage 9,
checked at `pipelinePost.sh:33`) and `output_gtp/summarized.parquet` (FDR
summary from Stage 7).

```
 [1] cytoBand.txt           (download hg19 cytobands from UCSC)
 [2] plotCircos.py          → output_gtp/plots/circos_*.{png,pdf}
 [3] visualizeFindings.py   → output_gtp/plots/{volcano,manhattan,region,scatter}*
 [4] exportBipartiteNetwork → output_gtp/network/cytoscape_{nodes,edges}.csv
 [5] visualizeBipartiteNetwork → output_gtp/network/*.png
 [6] evaluateSaliency.py    → output_gtp/plots/saliency_*, effect_vs_mad, …
 [7] runEnrichment.py       → output_gtp/enrichment/ (functional + optional ENCODE, CSV only)
```

Network export filter defaults (`pipelinePost.sh:26-27`):
`NETWORK_TOP_K=5000`, `NETWORK_MAX_BOOT_P=0.05`.

### 3d. `pipelinePermute.sh` — permutation calibration (independent chain)

```
 [*] reservoir_to_parquet / build_gene_anchored_master  → permute master
 [*] assignRegionToEcpg_parquet (chr-native, no filter) → master + region
 [*] tecpg run mlr --mlr-method qr_permute              → permutation_results
 [*] eval_permute.py + summarize_permute.py             → eval_permute_report.json
 [*] permute_qc_report.py                               → permute_qc_report.html
```

`VALID_STAGES=("all" "permute" "eval")` (`pipelinePermute.sh:144`). Entrances:
`--master-parquet`, `--reservoir`, or `--cis-enrich` (exactly one required,
`:132`). `--mapping cis` is refused (`:137-141`) because the null is trans-global.

**This chain writes no column onto the mainline catalog.** Its product is a
per-region calibration verdict. See §5e and §6.

---

## 4. Per-pair statistics (MLR `qr`)

`tecpg run mlr --mlr-method qr --all --compute-ig` fits, for each (CpG j,
gene i) pair:

```
   G_i  =  β0 (intercept)  +  β_mt · M_j  +  Σ β_k · C_k   +  ε
```

(The CLI accepts three MLR methods plus the permutation consumer:
`legacy_normal_eq`, `qr`, `qr_bootstrap`, `qr_permute`.)

For the methylation term (`mt`) it emits four statistics, plus IG
(`tecpg/processing.py:338-362`):

| Column | Meaning |
|--------|---------|
| `mt_est` | β coefficient (effect size) of methylation on expression |
| `mt_err` | standard error of β |
| `mt_t`   | t-statistic = `mt_est / mt_err` |
| `mt_p`   | p-value (fast normal-CDF approximation, **float32**) |
| `mt_ig`  | Integrated Gradients saliency (see below) |

`pipeline.sh:31` sets `MLR_IG_COVARIATES="all"`, so Stage 3 also emits a
per-covariate `<covariate>_ig` column and the full `*_est/_err/_t/_p` quadruple
for `const` and each covariate. (The comment block above that assignment still
describes the old `'none'` default — §11-D6.)

### 4a. `mt_p` is float32 and saturates

`mt_p` is computed as `erf(−|t|/√2) + 1` in float32 (`tecpg/processing.py:39`).
Adding a value near `−1` to `1` is catastrophic cancellation: the sum lands in
`[0.5, 1)` where float32 spacing is `2⁻²⁴`, so the smallest representable
non-zero result is **5.96e-08** and anything below flushes to exactly `0` at
`|t| ≈ 5.68`. Stage 6 fixes this on the catalog by recomputing in float64
(§5b). **`mt_p` must never be used as a ranking fallback.** It survives in the
pipeline in exactly one load-bearing place: the mapper's own `-p` threshold
(§5-0), which is applied to the float32 column before Stage 6 exists.

### 4b. Integrated Gradients (saliency)

Analytical IG (`tecpg/processing.py:508, 513, 733`):

```
   IG_analytical = mean_s|X − X̄| · |β|        (intercept excluded)
```

where `X̄` is the per-feature mean baseline (`X.mean(dim=1)`, the default
`ig_baseline='mean'`), `mean_s|X − X̄|` is the per-feature mean absolute
deviation over samples, and `|β|` is the absolute regression coefficient. The
intercept (index 0) is dropped (`B[:, :, 1:]`). A slower Captum-based variant
(`--compute-ig-deep`, requires `--p-thresh`) exists but is **not** used by
`pipeline.sh`.

For the methylation term this reduces to a two-factor product:

```
   mt_ig = MAD(M_j) · |β_mt|
```

**This is not an independent statistic.** It is an exact, deterministic
re-weighting of the t-statistic — see §6.2 and Appendix A. Its properties as a
ranking axis follow from that identity, and are the reason §6 treats it as a
*filter on a known confound* rather than as a third line of evidence.

---

## 5. Filtering & prioritization layers

Each layer narrows or re-ranks the eCpG set. From most permissive to most
selective. **This cascade has been corrected against the code** — the previous
revision misstated stages 0, 5a, and 5c/5d (§11).

```
   ALL CpG×Gene pairs (mlr --all)                       GTP `all`: ~2e10 pairs
        │  Stage 3: MAPPER p-threshold, -p default 1e-3, applied to FLOAT32 mt_p
        ▼   (tecpg/cli.py:799, tecpg/processing.py:950)      ◄── the real first gate
   Catalog (p ≤ 1e-3)                                   + sample_reservoir.csv
        │                                                 (uniform, PRE-filter)
        │  Stage 5: region annotation — ANNOTATES, DOES NOT FILTER
        ▼   (rows lacking annotation get region=NULL and are retained)
   Catalog + region label
        │  Stage 6: precise float64 two-sided t p-value (precise_mt_p)
        ▼
   Catalog + precise_mt_p
        │  Stage 7: Benjamini-Hochberg global FDR (fdr_est), denominator TOTAL_TESTS
        ▼   FDR < 0.05 and < 0.01 are REPORTED; no rows are removed
   Catalog + fdr_est                                     ◄── fdr_est never gates
        │  Stage 8: per-region QUOTA, ranked by precise_mt_p
        ▼   min 4500 / max 10000 per region (pipeline.sh:318)
   Bootstrap candidate list (bootstrap_list.csv)         ~8 regions × ≤10000
        │  Stage 9: empirical bootstrap (1000 iters, batch 10) → p_boot, CI
        ▼   LEFT-JOINED onto the full catalog; non-candidates get NaN
   bootstrap_merged.parquet  (full catalog rows; p_boot and mt_ig NaN outside list)
        │  pipelinePost network export: p_boot ≤ 0.05 (NaN → dropped) AND top-k 5000 by mt_ig
        ▼
   Network nodes & edges                                 ◄── coverage = the Stage 8 quota
```

The single most consequential structural fact: **everything downstream of Stage
8 sees only what the Stage 8 quota admitted.** `p_boot` and (post-join) `mt_ig`
are NaN outside the candidate list, and both the network filter and the Circos
ranking treat NaN as "drop" or "sort last". The quota is therefore not a
prioritization refinement — it is the effective definition of the analysed set.
This is the limitation motivating §6.

### 5-0. Mapper p-threshold — the real first gate

`tecpg/cli.py:799` declares `-p/--p-thresh` with `default=0.001`.
`pipeline.sh:228` does not override it. `tecpg/processing.py:950` applies
`p_indices = P[:, p_col] <= p_thresh` and `:963` subsets the results. So the
catalog written to disk is the **float32 `p ≤ 1e-3`** subset of the full grid.

Two consequences worth stating plainly:

1. The catalog's inclusion boundary is set by the one column §4a says is
   untrustworthy. Near the boundary (`p ≈ 1e-3`, `|t| ≈ 3.3`) float32 is
   nowhere near its cancellation floor, so the *boundary itself* is
   numerically fine; the floor bites at `p < 6e-8`, deep inside the retained
   set. The threshold is sound; the ranking within it is not, which is exactly
   what Stage 6 repairs.
2. `TOTAL_TESTS` (the BH denominator) is extracted from the mlr log
   (`pipeline.sh:236-253`) and is the **full grid**, not the catalog row count.
   This is correct and load-bearing: FDR is computed over the universe actually
   tested, and `pipeline.sh` fails closed if the extraction fails.

Alongside the filtered catalog, Stage 3 writes `sample_reservoir.csv` — a
uniform sample taken **before** p-filtration, auto-sized at
`min(1e6, 1% of grid)` when `--reservoir-count` is not given
(`tecpg/cli.py:1342-1345`), written by `tecpg/processing.py:1330`. It is the only
artifact spanning the full p-range, which is why λ_GC and the permutation
calibration are both computed from it and not from the catalog.

### 5a. Region assignment — `assignRegionToEcpg_parquet.py`

**This stage annotates; it does not filter.** `PVALCUTOFF = 1e-6` is defined at
`:19` and logged at `:602`, but is never compared against anything: `mt_p =
row[pval_col]` (`:225`) is assigned and never read, and the exclusion counter
`npvalx` (`:185`) is printed at `:421` but never incremented. Every row that
enters is emitted; rows lacking usable annotation are emitted with `region =
None` and logged to `annotation_missing_ids.txt`. See §11-D1.

Region is strand-aware; thresholds at `:22-31`: promoter ±2,500 bp around TSS,
cis window 50,000 bp, distal beyond.

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

The labels partition rows exactly (an if/elif chain, one label per row). This
tool is the **single region authority**; `createBootstrapList`, `runEnrichment`,
`eval_permute` and the QC report all consume its `region` column rather than
re-deriving it. The script normalizes `chr`-prefix mismatches (`:434-452`) and
strips Ensembl ID version suffixes when the GTF is versioned but the parquet is
not (`:477-485`).

Note the asymmetry that the QC work surfaced: the taxonomy bounds
`DISTAL5`/`CIS5`/`PROMOTER` on the **TSS** but `CIS3`/`DISTAL3` on the **TES**,
with `GENEBODY` straddling both. On array platforms `GENEBODY` is largely
unreachable because it requires an annotated span exceeding the promoter window,
so pairs an RNA-seq gene model would place in the body fall to neighbouring
labels. Region composition is therefore **not directly comparable between array
and RNA-seq cohorts**.

### 5b. Precise p-values — `recalculate_pvalues_parquet.py`

Recomputes a high-precision two-sided p from `mt_t` using the Student-t
survival function at the pipeline DF (`:79`):

```
   precise_mt_p = 2 · sf(|mt_t|, df)        (float64, chunked, appended)
```

Reaches ~1e-300. This is **the p-value of record** for ranking and for FDR.

### 5c. Global FDR — `summarizeOutput_parquet.py`

Benjamini-Hochberg using the genome-wide `--total-tests`, so FDR is valid even
though only a filtered subset is materialized:

```
   fdr_est_i = p_i · total_tests / rank_i        (monotone step-down → q-values)
```

The pipeline runs with `--calculate-fdr` (`pipeline.sh:305`), so
`summarized.parquet` carries an `fdr_est` column; both FDR < 0.05 and FDR < 0.01
thresholds are **reported to stdout** (`:441-470`) but **no rows are removed**.
`fdr_est` is a computed, carried column; nothing in the pipeline gates on it
(§11-D3).

Since Chunk A the p-column and FDR-column are parameterized (`--p-column` /
`--fdr-column`, defaults `precise_mt_p` / `fdr_est`, `:147-148`), with three
guards: an explicitly named p-column that is absent **fails closed** rather than
falling back to the t-statistic (`:207-214`); naming an existing column as
`--fdr-column` is refused (`:218-222`); and a row whose source p is null yields
a **null** FDR rather than `1.0` (`:592-593`), keeping "not assessed"
distinguishable from "assessed and not significant". This parameterization is
what lets a second BH pass write `fdr_permute` beside `fdr_est` instead of over
it (§5e).

Genomic inflation **λ_GC** is estimated from the reservoir targeting ~1,000,000
p-values (1.2× oversampling, `:235-237`), computed at `:374-375`:

```
   λ_GC = median(χ²_obs, df=1) / 0.4549
```

`:378` warns at `λ > 1.1`. **Treat this warning as advisory only.** λ_GC
presumes a mostly-null test space — valid in GWAS/eQTL, not in eQTM, and least
of all in cis where a large fraction of pairs carry real signal. The permutation
work found the near-gene λ gradient (`PROMOTER` 1.107 > `CIS5` 1.081 > `CIS3`
1.051 > distal ~1.01 ≈ `TRANS` 1.006) to be close to a restatement of where the
eQTMs are. λ is **not** an appropriate optimization target for this pipeline.

### 5d. Bootstrap prioritization — `createBootstrapList.py` + `qr_bootstrap`

**The candidate list is a per-region quota, not a significance filter.** The
selection is (`createBootstrapList.py:149-151`):

```
   target      = int(total_hits × percent)
   floor_val   = max(target, min_per_region)
   final_count = min(total_hits, floor_val, max_per_region)
```

Tool defaults are 10% / 200 / 2000. The GTP pipeline run **overrides** them to
`--rank-by p-value --min-per-region 4500 --max-per-region 10000`
(`pipeline.sh:318`). With those values, any region holding ≥4,500 catalog rows
contributes at least 4,500 pairs **regardless of their p-values**, and any
region holding >10,000 contributes exactly 10,000. Ranking by `precise_mt_p`
(`:54-63`) decides *which* pairs within a region, never *how many*. Pairs are
sorted globally then sliced per region, and the concatenation is de-duplicated
globally with `keep="first"` (`:179`).

The practical effect is deliberate coverage balancing: `TRANS`, which dominates
the raw catalog, is capped at the same 10,000 as `PROMOTER`, which does not. That
is a defensible design for a *diagnostic* — it guarantees every region is
represented — but it means the candidate set is **not** "the significant pairs",
and downstream consumers that treat it as such are over-claiming.

`tecpg run mlr --mlr-method qr_bootstrap` (1000 iterations, batch 10,
`pipeline.sh:334`) then resamples samples with replacement and emits
(`tecpg/bootstrap.py:303-363`):

| Column | Meaning |
|--------|---------|
| `mt_est_boot_mean` | mean β across bootstrap resamples (finite draws only) |
| `mt_est_boot_std`  | std of β (robustness) |
| `ci_low`, `ci_high`| 2.5% / 97.5% percentile CI (`:328-329`) |
| `p_boot` | `2 · min(P(β≤0), P(β≥0))`, clamped at 1, floored at `1/finite_count` (`:331-343`) |
| `degenerate_resamples` | count of non-finite (rank-deficient) resamples for the pair |
| `<covariate>_ig` | per-feature IG, enabled by `BOOTSTRAP_IG_COVARIATES="all"` (`pipeline.sh:32`) |

Results are **left-joined onto the full master** → `bootstrap_merged.parquet`
(`tecpg/bootstrap.py:402`). Two consequences that the previous revision missed:

- `bootstrap_merged.parquet` is **not a row subset** of the catalog. It carries
  every catalog row; `p_boot` is NaN for the ~99.9% not in the candidate list.
- The join **drops the master's `mt_ig` first** (`:392`, `cols_to_drop` includes
  `ig_columns`) and replaces it with the bootstrap-stage IG. So the genome-wide
  Stage-3 `mt_ig` is **discarded**, and `mt_ig` in `bootstrap_merged.parquet` is
  NaN outside the candidate list. `evaluateSaliency.py:455-462` documents this
  and restricts every saliency aggregate to the IG-bearing rows accordingly.
  See §11-D4.

### 5e. Permutation calibration — `qr_permute` (§3d chain)

`qr_permute` is a **post-mapping consumer**, mirroring `qr_bootstrap`: it reads
a master parquet for the observed `mt_t` and the pair universe, builds a
design-fixed Freedman–Lane null from the supplied `M`/`G`/`C`, scores the
universe against it, and merges `perm_mt_p` back. Full method description lives
in `docs/mlr_qr_permute.md`; only what bears on selection is repeated here.

**What it settled (GTP only).** The `--cis-enrich` run on full GTP returned
`single_global_null_adequate` with `divergent_regions = []`. All seven regions
calibrate to the analytic reference at a bulk median `|log10(p_perm/p_ana)|` of
~0.0027; the largest near-gene departure is **3.162e-05** (`PROMOTER`) against a
tolerance of 0.5. The near-gene family carried 1,098,442 bulk pairs against
`MIN_REGION_BULK_N = 100`.

**What that licenses.** `eval_permute.py:compute_analytic_p` does not read a
stored column — it recomputes `2·t.sf(|t|, df)` in float64 from `mt_t`, the same
expression `recalculate_pvalues_parquet.py:79` uses. **The quantity the
permutation validated is exactly `precise_mt_p`.** The verdict is that the
analytic null model holds, per region, in the bulk band.

**Four qualifications that bound its use in §6.**

1. **Bulk only.** At B = 10 the empirical permutation p cannot resolve below
   `1/(10 × 3,793,007) ≈ 2.64e-08`. The tail — where the findings are — is
   unmeasured in either direction. Per-region tail ratios order exactly as
   signal density predicts, which is consistent with a resolution artefact *and*
   with genuine signal; the two are not separable at this permutation count.
2. **Per-dataset.** MESA and the oncology cohort (dbGaP phs003863) each need
   their own reservoir + enrichment run and their own verdict. Nothing is
   inherited from GTP.
3. **Detectable, not absent.** Mann–Whitney rejects distributional identity with
   `TRANS` for the three well-powered near-gene strata (`CIS5` p = 1.2e-10,
   `PROMOTER` p = 7.0e-07, `CIS3` p = 1.3e-02) while both distal strata — with
   14× more bulk pairs than `CIS5` — do not. The defensible claim is a bounded
   measurement, not an equivalence: a real, ordered offset of 3.2e-05 in median
   log₁₀ ratio, i.e. a factor of 1.00007 between the two p-values.
4. **Not on the catalog.** `p_permute`/`fdr_permute` exist only in
   `tools/annotate_permute_p.py` and its tests. `grep -rn annotate_permute_p
   --include=*.sh` returns nothing; `pipelinePermute.sh:144` has no `[5/5]`
   stage. **No catalog on disk carries either column.**

The pending annotation (Chunk C) is additive by contract:

| Column | Written by | Meaning |
|---|---|---|
| `p_permute` | `tools/annotate_permute_p.py` | `precise_mt_p` copied forward for rows whose region is licensed; null elsewhere. Licensing predicate is `{R : status ∈ (ok, reference)} − set(divergent_regions)` (`:59-62`) — divergent regions carry `status: 'ok'`, so status alone is not the predicate. |
| `fdr_permute` | `tools/summarizeOutput_parquet.py --p-column p_permute --fdr-column fdr_permute` | BH over `p_permute`, sharing `fdr_est`'s `TOTAL_TESTS` denominator. Null where `p_permute` is null. |

When every region is licensed — the GTP case — `fdr_permute` is identical to
`fdr_est` elementwise by construction. That is a self-validating oracle, and
also the honest statement of what the permutation currently adds to GTP
selection: **confidence in the existing axis, not a new one.**

---

## 6. The selection question: `precise_mt_p`/`fdr_est` vs `p_boot` vs `mt_ig`

This section is the discussion document. The framing that matters is that the
three candidate axes are **not three independent lines of evidence**. One pair
is an exact monotone re-expression; another pair is an exact deterministic
transform with known confounds; only one axis is genuinely independent, and it
is the one with the worst coverage.

### 6.1 What each quantity actually measures

| Quantity | Measures | Coverage on `bootstrap_merged.parquet` | Resolution floor |
|---|---|---|---|
| `precise_mt_p` | Evidence against `H0: β_mt = 0` under the fitted normal-error model | **All** catalog rows | ~1e-300 |
| `fdr_est` | The same evidence, re-expressed as an expected false-discovery proportion at that rank | All catalog rows | — |
| `p_boot` | Sign stability of `β_mt` under resampling of samples | **Candidate list only** (~≤80k rows); NaN elsewhere | `1/finite_count` ≈ **1e-3** |
| `mt_ig` | Typical absolute contribution of methylation to the fitted expression value | **Candidate list only** after the Stage-9 join (§5d) | — |
| `perm_mt_p` | Empirical rank of `|t|` against a data-driven null | Permutation chain only; never joined to the catalog | ~2.6e-08 at B=10 |

### 6.2 The relationships between them

**(a) `fdr_est` is a monotone function of `precise_mt_p`. It is not a second
axis.** BH computes `p·N/rank` and then enforces monotonicity by step-down
(`summarizeOutput_parquet.py:481-487`), so `fdr_est` is non-decreasing in `p`.
Ranking by `fdr_est` and ranking by `precise_mt_p` produce **identical
orderings**. What FDR adds is not discrimination but *interpretation*: it
converts a rank into a stated error rate, which is what makes a threshold
defensible. Choosing "between precise p and FDR" is therefore not a real choice
— they are the same axis at different units. The real choice is between that
axis and the other two.

**(b) `mt_ig` is an exact deterministic function of `|t|` and three confound
factors.** From `mt_ig = MAD(M)·|β_mt|` and `t = β_mt / SE(β_mt)`:

```
                        σ_ε        MAD(M)              1
   mt_ig  =  |t|  ·  ───────  ·  ─────────  ·  ──────────────────
                        √n         SD(M)         √(1 − R²_{M~[1,C]})
                        ▲            ▲                   ▲
                  gene residual  probe shape       probe–covariate
                     scale        (0.45–0.94)       collinearity
```

Verified numerically against an independent numpy OLS across randomized designs
(n = 80–400, 2–12 covariates, gaussian / bimodal / heavy-tailed probes):
**max |ratio − 1| = 6.7e-16**. Derivation in Appendix A.

Read the identity as a decomposition of what ranking by IG does *relative to*
ranking by evidence:

- **σ_ε (gene residual scale)** — the dominant term, and a pure confound for
  eQTM purposes. Two pairs with identical statistical evidence rank differently
  by IG solely because one gene is noisier. In the verification runs a pair at
  `|t| = 0.78` outranked one at `|t| = 4.48` on IG because σ_ε was 6× larger.
- **MAD(M)/SD(M) (probe shape)** — dimensionless, spanning roughly 0.45
  (heavy-tailed) to 0.94 (bimodal) in the verification runs. Bimodal probes —
  exactly the mQTL-driven ones — are systematically up-weighted ~2× relative to
  probes with outlier-driven variance.
- **1/√(1 − R²)** — up-weights probes collinear with the covariates, i.e.
  precisely the probes whose adjusted effect is least identifiable.

Two invariance properties, verified with the response held fixed:

- `mt_ig` is **invariant to a linear rescaling of methylation** (β compensates
  exactly): ×40 on M leaves `mt_ig` at 0.41397 and `|t|` at 16.648, unchanged.
  So the β-value-vs-M-value *unit* question does not move IG. A **logit**
  reparameterization is not a rescale and does move it (`mt_ig` 0.41397 →
  0.39229, `|t|` 16.648 → 16.596).
- `mt_ig` is **not invariant to the expression scale of the gene**: ×1000 on G
  gives `mt_ig` ×1000 exactly, with `|t|` unchanged. This is the σ_ε term, and
  it is why raw `mt_ig` is not comparable across genes.

**(c) `mt_ig_frac` removes the gene-scale term; it does not remove the others.**
`evaluateSaliency.py:495` computes `mt_ig / Σ_k |IG_k|`. Because every term in
numerator and denominator carries the same expression units, the ratio is exactly
invariant to gene rescaling (verified: Δ = 1.4e-17 under ×1000). It is the
defensible normalization if IG is used at all. It does **not** remove the probe-shape
or collinearity terms, and its absolute level remains uninterpretable as
"importance": `evaluateSaliency.py:301-313` reports the per-feature input MAD
alongside IG precisely so that a small methylation fraction can be read as "this
feature varies little in this cohort" rather than "this feature does not matter".
Note also that with `MLR_IG_COVARIATES="all"` the denominator includes the
**expression PCs**, which are near-proxies for the outcome and dominate it —
hence `--frac-exclude` (`:434-438`). A `mt_ig_frac` computed without excluding
`Exp_PC*_ig` is close to meaningless.

**(d) How far apart do IG and evidence rankings actually fall?** Under a
simulation with per-gene residual scale spanning 10^±0.7 and per-probe MAD
spanning 10^−2.3…10^−0.9 (n = 340, 20,000 pairs), Spearman ρ(|t|, `mt_ig`) =
0.626, top-100 overlap **0/100**, top-1000 overlap **29/1000**. Of the top-1000
by `mt_ig`, 971 are not top-1000 by `|t|`; their median `|t|` is 1.71 (vs 4.25)
and their median two-sided p is 0.088.

> **This number is illustrative, not measured.** It depends on the assumed
> spread of σ_ε and MAD, which I could not check against the real catalog from
> this review. **The measurement already exists in the codebase**:
> `evaluateSaliency.py` produces `saliency_fraction_vs_standardized_effect.png`
> and `effect_vs_mad.png` for exactly this. Running it on the tuned GTP catalog
> and reading the real ρ and top-K overlap should be the first empirical input
> to this decision. If the real ρ is ≫0.626 the concern shrinks; if it is near
> 0.626 the two axes are selecting substantially different pairs and the choice
> is consequential.

**(e) `p_boot` is genuinely independent — and structurally limited.** It is the
only axis that does not reduce to a transform of `mt_t`: it resamples samples
and asks whether the sign of `β_mt` is stable, which interrogates influential
observations and non-normality that the analytic p assumes away. Three limits:

- **Floor.** `1/finite_count` ≈ 1e-3 at 1000 iterations. It cannot express
  evidence beyond that and cannot reach any genome-wide threshold.
- **Coverage.** Defined only on the Stage-8 quota. This is the limitation in the
  original question.
- **Equivalence to the CI.** `p_boot ≤ 0.05` ⟺ `2·min(P(β≤0), P(β≥0)) ≤ 0.05`
  ⟺ `min ≤ 0.025` ⟺ the 2.5–97.5 percentile CI excludes zero. So
  `--max-boot-p 0.05` (`pipelinePost.sh:27`) is exactly the rule "`ci_low` and
  `ci_high` share a sign". The `<=`/`>=` inclusive convention at
  `tecpg/bootstrap.py:331-332` makes it marginally more conservative when exact
  zeros occur.

**(f) `perm_mt_p` does not currently add an axis to GTP.** Since every GTP region
calibrated, `p_permute` for GTP would be `precise_mt_p` verbatim and
`fdr_permute` identical to `fdr_est`. Its contribution is **licensing** — it
converts "we assume the analytic null" into "we checked the analytic null, in
the bulk, per region, on this cohort". Where it would become a distinct axis is
a cohort whose verdict differs, which is exactly why the columns are specified
as standard rather than omitted-when-redundant.

### 6.3 Summary of the dependency structure

```
                   mt_t  (the one measured quantity)
                     │
      ┌──────────────┼───────────────────────────┐
      │              │                           │
 precise_mt_p   mt_ig = |t| · σ_ε/√n            (resampling — does not
      │              · MAD/SD · 1/√(1−R²)        pass through mt_t)
   fdr_est                │                           │
 (monotone in p)     mt_ig_frac                    p_boot
                    (gene-scale-free)          (floor 1e-3, quota-only)
      ▲                   ▲                           ▲
      │                   │                           │
  licensed by       confound filter,            independent, but
  perm_mt_p         not evidence                coverage-limited
  (bulk, GTP)
```

One measured quantity, one exact monotone re-expression, one exact confounded
transform, and one genuinely independent but structurally limited statistic.

### 6.4 Candidate selection schemes

Framed as options for discussion, not a recommendation to adopt one now.

**S0 — status quo.** Per-region quota on `precise_mt_p` (4500/10000) → bootstrap
→ network gated on `p_boot ≤ 0.05` and top-5000 by `mt_ig`.
*Property:* guarantees regional coverage. *Cost:* the analysed set is defined by
a quota, not by evidence; the final network's IG ranking operates only on quota
survivors; nothing anywhere uses `fdr_est`.

**S1 — FDR-thresholded, quota-capped.** Replace the `min-per-region` floor with
an `fdr_est` threshold, keeping `max-per-region` as a compute cap. Selection
becomes "all pairs at FDR < q, capped at N per region", with the cap breach
logged per region.
*Property:* the analysed set has a stated error rate; regions with no signal
contribute nothing rather than 4,500 forced entries. *Cost:* loses guaranteed
regional coverage, which is a real loss for the region-comparison analyses; the
cap still binds in `TRANS`. Requires deciding what a region contributing zero
pairs means for the enrichment and network products.

**S2 — S1 plus stratified FDR.** As S1, but BH within region rather than
globally, so the near-gene strata are not swamped by the trans test space.
*Property:* matches the region-specific-biology goal directly. *Cost:* changes
the meaning of `fdr_est`; needs a per-region `TOTAL_TESTS`, which the mapper can
supply for `region != 'all'` runs but not from a single `--all` map without
additional bookkeeping. Also interacts with the permutation verdict, which is
currently a *global* licensing statement.

**S3 — evidence for selection, IG for annotation.** Select on
`precise_mt_p`/`fdr_est` (S1 or S2); carry `mt_ig_frac` (with `--frac-exclude
'Exp_PC*_ig'`) as a **descriptive attribute** and as a confound flag, not as a
ranking axis. Replace the network's `--top-k` ranking by `mt_ig` with a ranking
by the selection axis, and keep IG as an edge attribute.
*Property:* removes σ_ε from the selection path entirely, which §6.2(b) shows is
the dominant confound. *Cost:* loses the "large absolute contribution" notion
that IG captures and p-values do not; some genuinely large-effect pairs in noisy
genes will rank lower.

**S4 — two-stage: evidence gate, then stability confirmation.** Select
candidates on `fdr_est` (S1/S2), bootstrap them, and report `p_boot` /
`ci_low`,`ci_high` as a **confirmation** column rather than a filter — i.e. a
pair is "reported" on FDR and "robust" on CI sign agreement, with both stated.
*Property:* uses each axis for what it can support; the 1e-3 floor stops
mattering because `p_boot` is no longer being asked to rank. *Cost:* two numbers
to explain in the paper; requires deciding what to do with pairs that pass FDR
and fail the sign check (report as non-robust, or drop).

### 6.5 Open decisions

- **Does the permutation verdict change the selection axis?** For GTP, no — it
  licenses the existing one. The decision it does force is whether to *state*
  the licensing (report `p_permute`/`fdr_permute` as standard columns, per Chunk
  C) or to leave it as a methods-section claim about `precise_mt_p`. The former
  is the stated plan and keeps the schema stable across cohorts.
- **Measure the real IG-vs-evidence divergence before deciding S3.** Run
  `evaluateSaliency.py` on the tuned GTP catalog with `--covariates-csv` and
  `--methylation-csv` supplied and `--frac-exclude 'Exp_PC*_ig'`, and read ρ and
  the top-K overlap from `saliency_fraction_vs_standardized_effect.png` and
  `effect_vs_mad.png`. §6.2(d)'s numbers are simulation.
- **Regional coverage vs stated error rate.** S1–S2 trade the quota's guaranteed
  coverage for an interpretable threshold. Because the region-specific-biology
  question is a stated project goal, this is not obviously the right trade and
  should be decided explicitly rather than by default.
- **`fdr_est` is currently computed and never used.** Whichever scheme is
  chosen, either wire it into selection or state in the methods that it is
  reported for interpretation only. The present state — a computed column that
  gates nothing while the pipeline gates on an unstated float32 `p ≤ 1e-3` — is
  the hardest configuration to defend in review.
- **`runEnrichment.py`'s "FDR path" does not use FDR** (§11-D5). Resolve before
  any enrichment result is published.
- **`--top-k` has three disagreeing values** (README, branch proposal, code
  default `10000` at `exportBipartiteNetwork.py:15` vs pipeline's `5000`).
  Single source of truth still unresolved.
- **Per-dataset re-read.** Every §5e conclusion is GTP-only. MESA and the
  oncology cohort each need their own verdict before the same selection scheme
  is applied to them.

---

## 7. Enrichment tests

Functional and ENCODE enrichment live in the standalone `tools/runEnrichment.py`,
run as the final stage of `pipelinePost.sh`. It draws genes from the FDR summary
(`summarized.parquet`, `--rank-by fdr`) and/or the bootstrap IG ranking
(`bootstrap_merged.parquet`, `--rank-by ig`); `pipelinePost.sh:86` passes both.
The tool writes **CSV results only — no figures.**

| Test | Where | Method | Output |
|------|-------|--------|--------|
| ENCODE ChromHMM enrichment | `runEnrichment.py:262-343` | **Fisher's exact** of significant-eCpG overlap with ENCODE chromatin-state track vs. background BED (`--encode-enrichment`, `--background-bed`). Fold-enrichment + BH-adjusted p. | `encode_enrichment_results.csv` |
| Functional / pathway enrichment | `runEnrichment.py:345-419` | **Enrichr** (`gseapy`) on significant genes per region against `GO_Biological_Process_2021`, `KEGG_2021_Human`, `WikiPathway_2023_Human` (override with `--enrichment-libraries`, `:433`); keep Adj-P < 0.05. | `enrichment_results/{region}_{method}_{library}_enrichment.csv` |
| Gene–gene co-regulation | `visualizeBipartiteNetwork.py:404-430` | **Hypergeometric** test on shared CpG targets when building the unipartite gene projection; edge weight = −log10(p), capped 100. | edges of `UnipartiteProjection.png` |

**The IG path selects by inflection, not by a threshold** (`:521-529`): it ranks
by `mt_ig_frac`, calls `detect_inflection`, and takes everything above the knee.
Note that this path computes `mt_ig_frac` over **all** `*_ig` columns
(`:521-522`) with no equivalent of `--frac-exclude`, so the expression PCs are in
the denominator — see §6.2(c).

**The FDR path does not use FDR** — see §11-D5.

---

## 8. Visualizations

### 8a. Diagnostics (Stage 7, `summarizeOutput_parquet.py`)

| Figure | Content |
|--------|---------|
| `qq_plot.png` | observed vs expected −log10(p) with λ_GC (from the reservoir) |
| `p_value_histogram.png` | 100-bin p-value distribution |
| `saliency_profile_top50.png` | stacked IG feature contributions for top-50 hits |

Moved into `output_gtp/` by `pipeline.sh:308`.

### 8b. Findings (`visualizeFindings.py --all`)

Generates a full plot set for **each available** p-column, prefixing filenames
accordingly: `p_boot` → `bootstrapP_`, `precise_mt_p` → `preciseP_`; `mt_p` →
`mtP_` only as a fallback when neither is present (`:466-485`). Stratified
subsampling thresholds: `HITS_P_THRESH=1e-5`, `SIGNAL_P_THRESH=0.05`, signal
frac 5%, noise frac 0.5% (`:16-19`).

Because `p_boot` is NaN outside the candidate list (§5d), the `bootstrapP_`
plot set describes the quota, while the `preciseP_` set describes the catalog.
They are not two views of the same population.

| Figure | Content |
|--------|---------|
| `*_volcano_plot.png` | effect size (`mt_est`) vs −log10(p), colored by region; top-10 genes labeled |
| `*_manhattan_plot.png` | CpG chromosomal position vs −log10(p) with `1e-5` threshold line |
| `*_region_breakdown.png` | counts of significant hits per region |
| `comparative_scatter_*` / `scatter_*` | raw vs covariate-adjusted M-vs-G scatter (Pearson r, regression line) for top hits |

### 8c. Circos (`plotCircos.py`)

Curved CpG→gene links scaled by `mt_ig` saliency, red (`mt_est>0`) / blue
(`mt_est<0`), over 1 Mb density tracks. `--top-n` 5000 links, `--top-n-trans`
2000 (`:47-57`); ranking is `sort_values(by='mt_ig', ascending=False)` (`:324`).
Since `mt_ig` is NaN outside the candidate list, the Circos links are drawn from
the quota, not the catalog.

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

### 8d. Saliency evaluation (`evaluateSaliency.py`)

Run as Stage 6 of `pipelinePost.sh`. Substantially expanded since the previous
revision (+415 lines); it now includes the diagnostics that make the §6.2 IG
argument checkable on real data.

| Figure | Content | Line |
|--------|---------|------|
| `effect_vs_mad.png` | methylation MAD vs \|`mt_est`\| (log-log) — **the mechanism**: large \|β\| tracks low methylation variability | `:257` |
| `saliency_fraction_vs_effect_mad.png` | frac vs \|`mt_est`\|, colored by MAD — the high-\|β\| floor points are the low-MAD probes | `:277` |
| `saliency_fraction_vs_standardized_effect.png` | frac vs \|partial r\| (with `--df`) or \|`mt_t`\| — **variance/error-bounded**, so coefficient blow-up no longer stretches the axis | `:292` |
| `input_scale_vs_ig.png` | per-feature input MAD beside IG attribution, separating "large attribution because large input scale" from "large coefficient" | `:411` |
| `saliency_profile_ranks_{start}_{end}.png` | stacked proportional saliency by rank window | `:666` |
| `saliency_fraction_hist.png` | distribution of `mt_ig_frac` | `:676` |
| `saliency_vs_{effect_col}.png` | saliency against the chosen effect column | `:687` |
| `saliency_fraction_by_region.png` | `mt_ig_frac` faceted by canonical region | `:707` |
| `saliency_fraction_decay_curve.png` / `saliency_magnitude_decay_curve.png` | ranked decay with inflection detection | `:730`, `:747` |

Default `--rank-by mt_ig` (`:419`). Useful flags that `pipelinePost.sh` does
**not** currently pass: `--df` (upgrades the standardized-effect plot from
\|`mt_t`\| to partial r), `--covariates-csv` (enables the input-scale
diagnostic), `--methylation-csv` (independent MAD rather than the recovered
`mt_ig/|mt_est|`, which is an algebraic re-expression and not an independent
measurement), and `--frac-exclude` (removes expression PCs from the
`mt_ig_frac` denominator). All four should be supplied for the §6.5 measurement.

---

## 9. Network nodes & edges

### 9a. Export — `exportBipartiteNetwork.py`

Filter order (`:51-82`): `--min-effect` (|`mt_est`|) → `--max-boot-p` (`p_boot`)
→ `--top-k` ranked by `mt_ig` (fallback |`mt_t`|). `pipelinePost.sh` uses top-k
5000, max-boot-p 0.05; the tool's own `--top-k` default is 10000 (`:15`).

Both middle and last filters are quota-confined: `df['p_boot'] <= 0.05` (`:64`)
evaluates False for NaN, so every non-bootstrapped pair is dropped; and `mt_ig`
is NaN there too. **The network's population is the Stage-8 candidate list,
intersected with the CI-sign-agreement rule (§6.2e), ranked by IG.**

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

Edge columns built at `:86-98`; `abs_t` appended only on the |`mt_t`| fallback
path (`:89-90`). Note the edge table carries `mt_p` (float32) and `fdr_est` but
**not** `precise_mt_p` (`:86`) — §11-D7.

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

### 9b. Network figures — `visualizeBipartiteNetwork.py`

Edge weight = `mt_ig` (fallback `abs_t`, `:48-53`); `--threshold` default 0.5
(`:21`); duplicate edges resolved by max weight → `dropped_duplicate_edges.csv`.

| Figure | Content |
|--------|---------|
| `EnergyMinimizedBipartiteNetwork.png` | ForceAtlas2 layout, CpGs (by region) ↔ genes |
| `UMAPofRegulatoryBetaDiversity.png` | UMAP (Bray-Curtis) of CpG regulatory profiles |
| `RegulatoryDegreeDistribution.png` | KDE of CpG out-degree, faceted by region |
| `BiclusteredBiAdjacencyHeatmap.png` | hierarchically clustered CpG×Gene weight heatmap |
| `ArcDiagram.png` | bipartite arc layout, arc height ∝ weight |
| `UnipartiteProjection.png` | gene–gene projection (hypergeometric-weighted, §7) |

Because edge weight is `mt_ig` and `--threshold` is an absolute cut on it, the
threshold inherits the gene-scale dependence of §6.2(b): it is not comparable
across genes. If S3 is adopted, this is a second site to change.

---

## 10. Key thresholds & defaults (quick reference)

| Parameter | Value | Source |
|-----------|-------|--------|
| **Mapper p-threshold (the real first gate)** | **`1e-3`, float32 `mt_p`** | `tecpg/cli.py:799`, applied `processing.py:950` |
| Reservoir size (auto) | `min(1e6, 1% of grid)` | `tecpg/cli.py:1342-1345` |
| Region annotation p cutoff | **none — `PVALCUTOFF` is dead code** | `assignRegionToEcpg_parquet.py:19` (§11-D1) |
| Promoter window | ±2,500 bp | `assignRegionToEcpg_parquet.py:30-31` |
| Cis window | 50,000 bp | `:26` |
| Distal offset | 50,000 bp | `:22` |
| DF | `SAMPLES − COVARS − 2` | `pipeline.sh:199` |
| FDR thresholds reported (no filtering) | `< 0.05`, `< 0.01` | `summarizeOutput_parquet.py:441-442` |
| BH denominator | `TOTAL_TESTS` (full grid, from mlr log) | `pipeline.sh:236-253` |
| λ_GC warning | `> 1.1` (advisory only, §5c) | `summarizeOutput_parquet.py:378` |
| Bootstrap list defaults (percent / floor / cap) | 10% / 200 / 2000 per region | `createBootstrapList.py:14-19` |
| Bootstrap list values used by pipeline (min / max) | 4500 / 10000 per region | `pipeline.sh:318` |
| Bootstrap iterations / batch | 1000 / 10 | `pipeline.sh:334` |
| `p_boot` floor | `1/finite_count` ≈ `1e-3` | `tecpg/bootstrap.py:343` |
| Bootstrap CI | 2.5% / 97.5% percentile | `tecpg/bootstrap.py:328-329` |
| Network top-k / max boot p (pipeline) | 5000 / 0.05 | `pipelinePost.sh:26-27` |
| Network top-k (tool default) | 10000 | `exportBipartiteNetwork.py:15` |
| Circos top-n / top-n-trans | 5000 / 2000 | `plotCircos.py:49,56` |
| visualizeFindings hits / signal p | `1e-5` / `0.05` | `visualizeFindings.py:16-17` |
| Enrichr libraries | GO BP 2021, KEGG 2021, WikiPathway 2023 | `runEnrichment.py:433` |
| Permutation eval: bulk band / tolerance / min region n | `0.05` / `0.5` / `100` | `eval_permute.py:18-22` |
| Permutation resolution floor (GTP, B=10) | `≈2.64e-08` | `docs/mlr_qr_permute.md` §8 |

---

## 11. Defects and drift found during this review

Recorded, not fixed. Each is verified at `c94ef0f`. None is in scope for this
document's update; they are listed so the §6 discussion is not built on a
misreading of the pipeline.

**D1 — `PVALCUTOFF` in `assignRegionToEcpg_parquet.py` is dead code.**
`:19` defines it, `:602` logs it, nothing compares against it. `mt_p =
row[pval_col]` (`:225`) is assigned and never read; `npvalx` (`:185`) is printed
at `:421` and never incremented. *Impact:* documentation only — the previous
revision of this file asserted a `1e-6` gate in three places (§5, §5a, §9) that
has never existed. The log line actively misleads. *Fix options:* delete the
constant and the log line, or implement the filter. Deleting is safer; the
mapper's `-p` already provides a gate and adding a second one would silently
change every downstream count.

**D2 — the effective first gate is undocumented.** `-p` defaults to `1e-3`
(`tecpg/cli.py:799`) and `pipeline.sh` never sets it. The catalog's inclusion
rule is therefore implicit, and it is applied to the float32 column. Not a bug,
but it should be explicit in `pipeline.sh` so it appears in the run log and in
the methods section.

**D3 — `fdr_est` is computed and never used as a filter.** Grep across `tools/`
and `tecpg/` finds it only as a carried edge attribute
(`exportBipartiteNetwork.py:91-92`) and in `benchmark_kennedy.py` column lists.
`createBootstrapList.py` does not gate on it. *Impact:* the pipeline reports an
FDR-controlled result set to stdout that no artifact reflects. This is the
single most important item for §6.

**D4 — the Stage-9 join discards the genome-wide `mt_ig`.**
`tecpg/bootstrap.py:392` includes `ig_columns` in `cols_to_drop`, removing the
master's Stage-3 `mt_ig` before the left join; the join then repopulates it only
for candidate-list pairs. *Impact:* `bootstrap_merged.parquet` — the input to
Circos, the network export, and `evaluateSaliency` — has `mt_ig` NaN on ~99.9%
of rows, so every IG-ranked product is quota-confined.
`evaluateSaliency.py:455-462` handles this correctly and logs the coverage;
`plotCircos.py` and `exportBipartiteNetwork.py` do not log it. *Fix option:*
suffix rather than drop, so the genome-wide and bootstrap IG are both retained.

**D5 — `runEnrichment.py`'s FDR path compares raw p against the FDR threshold.**
`:468`: `sig_mask = chunk_p_vals <= args.fdr_threshold`, where `chunk_p_vals` is
`precise_mt_p` (`:462`) and `args.fdr_threshold` defaults to `0.05` (`:427`).
This selects `precise_mt_p ≤ 0.05`, not `fdr_est ≤ 0.05`. On a catalog already
cut at `p ≤ 1e-3` it admits **every row**. *Impact:* the "FDR" enrichment gene
set is the whole catalog per region. This invalidates any enrichment result
produced from that path. *Fix:* read `fdr_est` where present; fail closed
otherwise.

**D6 — `pipeline.sh` comment contradicts its own value.** `:26-30` states Stage
3 "defaults to `'none'`" for per-covariate IG; `:31` sets
`MLR_IG_COVARIATES="all"` (changed at `4767bf0`, 2026-07-19). The stated
rationale (>5GB per intermediate at 153M rows) was the reason for `'none'` and
still applies. Either the comment or the value is wrong.

**D7 — the network edge table carries `mt_p`, not `precise_mt_p`.**
`exportBipartiteNetwork.py:86` hardcodes `['mt_id','gt_id','region','mt_est','mt_p']`.
Cytoscape edges therefore carry the float32 column that saturates at 5.96e-08 —
which is precisely the range the top-5000 edges occupy. *Fix:* prefer
`precise_mt_p` when present.

**D8 — silent-truncation path in `summarizeOutput_parquet.py`** (previously
tracked). Caught exceptions can return exit 0 with a valid but truncated
parquet.

**D9 — `final_transcript.txt` committed at repo root** (previously tracked),
with test counts matching no commit in history.

---

## 12. Maintenance checklist

When changing the pipeline, update the relevant section here:

- [ ] New/renamed stage in any of the four drivers → §3
- [ ] New per-pair statistic or IG change → §4, and re-derive §6.2 / Appendix A
- [ ] Changed threshold, region rule, FDR, or bootstrap behavior → §5, §10
- [ ] Any change to what selects the analysed set → §5, §6 (this is the one that
      matters most; §6 is only useful if it describes the live cascade)
- [ ] New permutation verdict, or a verdict for a new cohort → §5e, §6.5
- [ ] New or modified enrichment test → §7
- [ ] New or renamed figure / network attribute → §8, §9
- [ ] Defect in §11 fixed or newly found → §11
- [ ] Re-verify the dev HEAD hash and every `path:line` citation when lines drift

---

## Appendix A — derivation of the IG identity

**Claim.** For the methylation term of the fitted MLR,

```
                        σ_ε        MAD(M)              1
   mt_ig  =  |t|  ·  ───────  ·  ─────────  ·  ──────────────────
                        √n         SD(M)         √(1 − R²_{M~[1,C]})
```

where `σ_ε` is the residual standard error, `MAD(M) = mean_s|M − M̄|`,
`SD(M)` uses the population (`1/n`) convention to match MAD's, and
`R²_{M~[1,C]}` is the coefficient of determination of methylation regressed on
the intercept and covariates.

**Derivation.** tecpg computes `mt_ig = MAD(M)·|β_mt|`
(`tecpg/processing.py:513, 733`). By definition `|β_mt| = |t|·SE(β_mt)`. For
OLS, the standard error of a single coefficient is

```
   SE(β_mt) = σ_ε / ( SD_pop(M) · √n · √(1 − R²_{M~[1,C]}) )
```

— the Frisch–Waugh form: the denominator is the norm of the component of `M`
orthogonal to `[1, C]`, which is `√n · SD_pop(M) · √(1 − R²)`. Substituting:

```
   mt_ig = MAD(M) · |t| · σ_ε / ( SD_pop(M) · √n · √(1 − R²) )
```

which regroups to the claim.

**Verification.** `ig_identity_check.py` fits an independent numpy `lstsq` OLS
across randomized designs — `n` ∈ [80, 400], 2–12 covariates, methylation drawn
as gaussian / bimodal (±1 with jitter) / heavy-tailed (t₂), each with injected
collinearity against `C[:,0]` — and compares `mt_ig` computed by tecpg's formula
against the prediction. Result:

```
   n   kc   shape      |t|      mt_ig  predicted    ratio  MAD/SD  R2_M~C  sig_eps
 370   11   gauss   0.1149    0.01682    0.01682 1.000000  0.8052  0.0778   3.3588
 187    8 bimodal   0.7778    0.21169    0.21169 1.000000  0.8766  0.2633   3.6441
 235    3 heavy-t   6.7929    0.43519    0.43519 1.000000  0.5862  0.0064   1.6700
 182    5   gauss   4.4780    0.18908    0.18908 1.000000  0.7761  0.2950   0.6162
 102    7 bimodal   0.2130    0.06603    0.06603 1.000000  0.9401  0.1019   3.1565
 174    6 heavy-t   7.6716    0.62502    0.62502 1.000000  0.4515  0.0890   2.2717

   max |ratio − 1| = 6.7e-16
```

Note rows 2 and 4: `|t| = 0.78` yields `mt_ig = 0.212` while `|t| = 4.48` yields
`mt_ig = 0.189`. The reversal is driven by `σ_ε` (3.64 vs 0.62). This is the
concrete form of the §6.2(b) concern.

**Invariance checks** (`ig_frac_check2.py`, response held fixed):

```
                                      |t|       mt_ig  mt_ig_frac
baseline (beta-values)           16.64801     0.41397    0.150775
(a) methylation x40 (linear)     16.64801     0.41397    0.150775   ← mt_ig invariant
(b) expression x1000             16.64801   413.97164    0.150775   ← frac invariant
(c) beta -> M-value (logit)      16.59613     0.39229    0.144261   ← not a rescale
```

(a) A linear rescale of methylation leaves `mt_ig` exactly unchanged, because
`β` compensates. (b) A rescale of expression multiplies `mt_ig` by the same
factor but leaves `mt_ig_frac` unchanged to 1.4e-17 — the gene-scale term
cancels in the ratio. (c) The logit reparameterization is not a rescale and
moves all three quantities slightly.

**Practical consequences.**

1. `mt_ig` ranks pairs by contribution *in the gene's own expression units*, so
   it is not comparable across genes. `mt_ig_frac` is the comparable version.
2. `mt_ig` does not divide by the standard error, so a large contribution
   estimated imprecisely outranks a smaller one estimated precisely.
3. Neither `mt_ig` nor `mt_ig_frac` removes the probe-shape or collinearity
   terms.
4. IG is not an independent measurement of importance and cannot corroborate a
   p-value: both are functions of the same fit. Its legitimate use is as a
   *descriptive* attribute (how much of the fitted expression this methylation
   accounts for) and as a **confounding filter** — flagging pairs whose apparent
   effect rests on a near-invariant probe or on covariate collinearity.

> The two verification scripts are shipped alongside this document
> (`ig_identity_check.py`, `ig_frac_check2.py`). They depend only on numpy and
> scipy, take under a second, and are the forced-fail check for §6.2: perturb
> either formula and the ratio column departs from 1.
