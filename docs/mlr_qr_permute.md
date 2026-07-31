# `qr_permute` — Permutation-Null Significance for eQTM Mapping

> **Status: under active development — bulk calibration complete on GTP; tail unvalidated.** The **Phase-1 trans pipeline is complete** and `qr_permute` produces real per-pair p-values on a finalized, seeded, provenance-stamped output schema. The `--cis-enrich` run on full GTP returned **`single_global_null_adequate` with `divergent_regions = []`**: the corrected map produced 16,526,333 cis pairs, the assembled master carried 17,525,137 pairs, and the near-gene family cleared the coverage floor with **1,098,442** bulk pairs against `MIN_REGION_BULK_N` = 100. All seven regions calibrate; the largest near-gene departure from `TRANS` is **3.162e-05** against a tolerance of 0.5, and the verdict is invariant from 0.5 down to 1e-4 (first flipping at 3e-5). **Consequence: on GTP a single global null is adequate and the Phase-2 per-gene cis Beta is not required.** Four qualifications matter. First, the quantity validated is the **`float64` analytic p recomputed from `mt_t`** — the same `2·t.sf(|t|, df)` that `recalculate_pvalues_parquet.py` stores as `precise_mt_p` — so the validation transfers directly to the mainline catalog's precise column. Second, the verdict is **per-dataset**; MESA and the oncology cohort must be run and read independently and inherit nothing. Third, this validates the **bulk, not the tail**: at B = 10 the permutation p cannot resolve below ~2.6e-08, so the extreme tail is *unmeasured* in either direction, and that bites hardest exactly where the findings are (§8). Fourth, the near-gene agreement is **detectable but negligible rather than exact** — Mann–Whitney rejects distributional identity with `TRANS` at p = 1.2e-10 for `CIS5`, at an effect size of 3.2e-05 in median log₁₀ ratio, i.e. a factor of 1.00007 between the two p-values (§5). **Phase 4 is complete** (`36f3af8`): `pipelinePermute.sh [5/5]` annotates the mainline catalogs with `p_permute` and `fdr_permute`, additively and without touching `mt_p`, `precise_mt_p` or `fdr_est` (§5, §3.10). The columns exist; whether a given study *uses* them is a per-study decision. Do **not** use `perm_mt_p` for inference until this notice is updated — Phase 4 wires the plumbing, it does not validate the tail. See [§5 Implementation status](#5-implementation-status).

**Applies to:** `tecpg run mlr --mlr-method qr_permute` (torch-eCpG v2, `dev`).
**Audience:** method reviewers, maintainers, and (eventually) users.
**Related:** analytic p (`mt_p`), bootstrap diagnostic (`qr_bootstrap` / `p_boot`).

> **Architecture note.** `qr_permute` is a **post-mapping consumer**: it reads the mapping's output parquet (`--master-parquet`) for the observed statistic and the pair universe, then merges `perm_mt_p` back onto it — mirroring `qr_bootstrap` (§4). This consumer form is **live on `dev`**; the earlier **self-contained** form (which generated its own universe and recomputed the observed statistic) has been fully replaced (§5).

---

## In plain language

**What it does.** eQTM mapping looks at pairs — a spot where DNA methylation is measured, and a gene whose activity is measured — and asks whether the two move together across people. Every pair gets a number for how strongly they appear linked. The hard part is separating a *real* link from one that could easily arise by chance, across an enormous number of pairs.

**The idea.** Instead of trusting a formula to judge chance, `qr_permute` measures chance directly. It repeatedly **shuffles** the data so that any true methylation–expression link is deliberately broken, and records how strong a link still appears by luck alone. Repeating this many times builds a picture of what pure noise looks like *for this dataset*. Each real pair is then compared against that picture: a link stronger than almost anything the shuffling produced is unlikely to be a fluke.

**Why this way.** It fixes two weaknesses of the usual formula-based p-value. The formula quietly rounds the smallest probabilities down to zero, so it can't tell the strongest associations apart; and it assumes the data behave in an idealized way that real biology often doesn't. Measuring chance by shuffling sidesteps both — it is built from the data's own noise, it produces a result for every pair, and it can still gauge how rare even the strongest signals are.

**What you get.** One p-value per pair (`perm_mt_p`): small means the association is unlikely to be chance, large means it's unremarkable. Because methylation near a gene ("cis") and methylation far away or on other chromosomes ("trans") behave very differently, the two are ultimately meant to be judged separately so each gets a fair yardstick.

> **For now:** the machinery is built and the shuffling has confirmed that the formula-based p-values are trustworthy in the ordinary range, for methylation near genes and far from them alike, on one dataset. Two things it has *not* confirmed: the very smallest p-values — the strongest findings — because ten shuffles cannot resolve that deep; and anything about the other cohorts, which have to be checked separately. Treat the current output as validated in the ordinary range and provisional at the extremes.

---

## 1. Overview

`qr_permute` is a permutation-based significance method for expression–methylation (eQTM) association mapping. It is intended to become the **primary per-pair significance measure**, complementing — not replacing — the two existing measures:

- the **analytic** p-value (`mt_p`): a two-sided normal-approximation tail derived from the regression t-statistic; and
- the **bootstrap** diagnostic (`qr_bootstrap`, `p_boot`): a resampling-based stability measure computed on a user-supplied candidate pair set.

Unlike either, `qr_permute` estimates a **data-driven null** and can resolve very small p-values via parametric tail extrapolation. Like `qr_bootstrap`, it is a **post-mapping step**: the user runs the primary mapping first, and `qr_permute` consumes its output parquet (`--master-parquet`) — reading the observed statistic and the pair universe the mapping already produced, scoring that universe against the permutation null, and merging `perm_mt_p` back onto it. An optional `--pairs-file` narrows the scored set to a candidate subset; by default the universe is the entire master. It is implemented as an isolated parallel method alongside the existing solvers.

---

## 2. Motivation

`qr_permute` addresses three limitations of the existing significance measures:

1. **Analytic-p underflow — in the mapping's stored column only.** As written by the mapping, `mt_p` is computed in `float32` as `erf(−|t|/√2) + 1`. Adding a value near `−1` to `1` is catastrophic cancellation: the sum lands in `[0.5, 1)` where `float32` spacing is `2⁻²⁴`, so the smallest representable non-zero result is **5.96e-08** and anything below flushes to exactly `0` at `|t| ≈ 5.68`. Measured on the GTP master: smallest non-zero stored p 5.96e-08, and **2,092 pairs at exactly zero** — 26% of the tail band, unrankable against each other. This is a `float32` *epsilon* limit under cancellation, not the `float32` range limit (~1.18e-38). **It is already resolved on the mainline catalog**: pipeline stage `[6/9]` (`recalculate_pvalues_parquet.py`) recomputes `2·t.sf(|t|, df)` in `float64` and stores it as `precise_mt_p`, which reaches ~1e-300. The underflow therefore persists only in artifacts that bypass `[6/9]` — the cis write-all map and the reservoir, hence the permute master. This weakens motivation (1) considerably: permutation's contribution is **calibration**, not resolution.
2. **Bootstrap coverage and floor.** `p_boot` exists only for the `--pairs-file` subset and floors at `1/(resamples)` (~`1e-3` at 1000 resamples). It is a candidate-set *stability* diagnostic by design and cannot serve as a genome-wide primary statistic or reach the exploratory `1e-6` cutoff.
3. **Parametric mis-calibration.** The analytic p assumes normal, correctly-specified errors. A permutation null is model-free and absorbs residual mis-calibration (non-normal residuals, heteroscedasticity, confounding not captured by the covariates). This is a more principled handle on genomic inflation than tuning covariate models toward `λ = 1`: if inflation reflects a mis-specified null *shape*, permutation recalibrates it directly; if it reflects real signal, permutation removes the signal and leaves it.

---

## 3. Method

### 3.1 Test statistic — the pivotal t

For a CpG `m` and gene `g`, fit the linear model

```
G_g = β0 + β_m · M_m + C · γ + ε
```

and test `H0: β_m = 0`. The reported statistic is the **methylation-coefficient t**:

```
t = β_m / SE(β_m)
```

with the intercept and covariates `C` treated as nuisance. `qr_permute` pools and scores the **t-statistic**, not the raw coefficient `β_m`. The reason is that the null spread of `β_m` scales with the CpG's variance — a `β_m` that is extreme for a low-variance probe is unremarkable for a high-variance one — whereas the t-statistic is approximately **pivotal**: given a shared design and constant degrees of freedom, its null distribution is common across pairs, which is what makes a *pooled* null valid (§3.8).

### 3.2 Null construction — design-fixed Freedman–Lane

The null is generated by permutation under a **Freedman–Lane** scheme, which is correct for a covariate-adjusted partial coefficient:

1. Regress the response on the **reduced** model `[1, C]` (covariates only, no methylation); obtain fitted values `Ĝ_reduced` and residuals `e`.
2. Permute the residuals over samples: `e → e*`.
3. Form the permuted response `G* = Ĝ_reduced + e*`.
4. Refit the **full** model `G* ~ [1, M_m, C]` and take the methylation-coefficient t.

Two properties are essential:

- **Covariate correctness.** A naive shuffle of sample labels breaks the methylation–covariate relationship and mis-specifies the *partial* coefficient's null, because methylation correlates strongly with the covariates (cell composition, age, batch). Permuting reduced-model residuals preserves that structure. This is also why the pre-existing `--permute-label-test` (a whole-column shuffle, used as a coarse negative control) is **not** a valid adjusted null and is not reused as one.
- **Design-fixed ⇒ cheap.** Only the *response* is permuted; the design `[1, M, C]` is identical across permutations. The reduced-model residualization is computed **once** (a single factorization of `[1, C]` applied to all genes) and the full design is invariant to the permutation, so each permutation is a cheap re-solve rather than a re-factorization. Permuting a *predictor* instead would change the design every permutation and cost roughly `P×` more — that formulation must be avoided.

### 3.3 Null subsampling

The pooled-null assumption (§3.8) — that the per-pair null is shared — also licenses estimating the null from a **representative subsample** rather than from every pair:

- **Trans / global:** a uniform random sample of loci (drawn independently on the M and G axes; the null pairs are their product) suffices to characterize the single global null.
- **Cis (planned):** per gene, the null population is that gene's cis-window CpGs (the full local test space), not a random draw.

The **observed** statistic is **read from the master parquet for every reported pair** (the mapping already computed it); only the *null* is subsampled. Conflating the two — letting the null subsample also define the reported set — would silently forfeit the master's universe and is explicitly disallowed: the subsample flags feed only the null population, never the scored set. A non-positive subsample count fails closed rather than yielding an empty null.

### 3.4 Stratification — cis vs trans

The two strata have different test geometry and are ultimately meant to take different null constructions, mirroring the cis/trans test-space asymmetry:

- **Trans → pooled global null + generalized-Pareto (GPD) tail** *(implemented)*. Trans has no window over which a per-gene min-p is defined, so a single global null is the natural object. The scored null population is stratified by **chromosome** — a null pair is *trans* iff its CpG and gene lie on different chromosomes — using the shared region-mask helper, independent of which pairs happen to populate the master (§4).
- **Cis → per-gene Beta-approximation** *(planned — Phase 2, not yet implemented)*: fit a Beta distribution to each gene's permutation min-p over its window, correcting local multiplicity and enabling smooth tail extrapolation. The Beta is fit in the **t-domain** (max `|t|` per permutation, converted at the end) so the `float32` analytic-p underflow cannot corrupt the significant tail.

**Current behavior:** because the cis Beta is not yet built, **all reported pairs — cis and trans alike — are currently scored against the trans-global null.** Whether the cis stratum ultimately needs its own null (rather than sharing the global one) is the open *stratify-or-not* question, to be decided from the Phase-3 evaluation evidence (§5, §8).

**Two distinct notions of "region" — null vs diagnostic.** Null-*pair* stratification (above) is a binary chromosome test: a null pair is trans iff its CpG and gene are on different chromosomes. That is all the null needs, and it is deliberately coarse. The **evaluation** (§5, Phase 3a), by contrast, partitions the *observed* pairs with the project's canonical, strand-aware, TSS-relative **7-way** taxonomy — `TRANS`, `DISTAL5`, `CIS5` (2.5–50 kb upstream), `PROMOTER` (±2.5 kb of TSS), `GENEBODY`, `CIS3` (≤50 kb past the gene end), `DISTAL3` — assigned once by `tools/assignRegionToEcpg_parquet.py`, which is the single source of truth already consumed downstream (`createBootstrapList`, `runEnrichment`). This matters because a same-chromosome heuristic labels *every* same-chrom pair "cis," folding the far `DISTAL5`/`DISTAL3` pairs in with the genuine near-gene window; the near-gene family is only `CIS5 ∪ PROMOTER ∪ GENEBODY ∪ CIS3`. The evaluation therefore **consumes the canonical `region` column** rather than re-deriving strata (§4, §5).

**Chromosome equality is not a naive comparison.** Assigning strata requires canonicalizing chromosome labels before testing equality: annotation columns arrive as bare integers, `chr`-prefixed strings, mixed case, or `float64` (pandas infers a float column whenever any value is missing, silently turning `1` into `1.0`). A raw `==` across two differently-typed annotation sources evaluates false for *every* pair, which would relabel the entire cis stratum as trans **with no error raised**. The canonical form (a single shared helper, `tecpg/chrom.py:canonicalize_chrom`, used by both the null's annotation-normalization and the mapping's region-filtration path — §4) strips a `chr` prefix and a trailing `.0`, uppercases, maps `X`/`Y`/`MT`, and treats a missing/unmappable chromosome as a **dropped locus** rather than guessing a stratum.

**Universe ∩ normalized annotations.** The mapping produces the master over the full M × G (a `region='all'` map needs no annotations and drops nothing), whereas the null here is built over the **normalized** M/G — loci with a missing or unmappable chromosome are dropped during canonicalization. The master universe is therefore a strict superset of the scoreable set. Before scoring (and before the consistency guard), `qr_permute` **intersects** the master's `(mt_id, gt_id)` universe with the normalized M/G index: dropped-locus pairs cannot be chromosome-stratified, so by default they are **silently excluded** with a count logged, surviving in the merged output with `NaN perm_mt_p`. When a `--pairs-file` explicitly names such a pair, that is a usage error and **raises** (§3.8). This recurs on real cohorts — the GTP and MESA annotation files both carry NaN-chromosome loci.

### 3.5 Scoring and tail extrapolation

The empirical two-sided p for an observed `|t|` is the fraction of null `|t|` at least as large, floored by the finite null size:

```
p_emp = max( #{ |t_null| ≥ |t_obs| } / N_null ,  1 / (N_null + 1) )
```

The count is read from the accumulated null representation (§3.7): a reverse-cumulative sum over the histogram bins plus the overflow count, taking the bin that contains `|t_obs|` (a slightly conservative convention). Empirical counting is reliable only to the extent of **effective** independent draws. Because probes are in LD and genes co-express, the effective number of independent draws is far smaller than the nominal count, and the permutation count is the binding constraint in the deep tail. Below the reliable range, a **parametric tail** extends p past the empirical floor:

- **Trans (implemented):** a **generalized-Pareto (GPD)** peaks-over-threshold fit to the retained tail exceedances, in `float64`. The threshold `u` is the smallest retained exceedance; above `u` the p-value is `(N_u / N) · SF_GPD(|t_obs| − u)`, which meets the exceedance probability `N_u / N` continuously at `u` and extends smoothly below the `1/(N_null+1)` floor. The result is clamped strictly positive so a bounded-support fit (`ξ < 0`) can never map to exactly `0`. If the tail is degenerate — too few exceedances for a fit, or the fit returns non-finite parameters — the method **falls back to the empirical p** unchanged (a warning is logged); the pipeline still yields valid, if unextended, p-values.
- **Cis (planned):** a Beta on the p-values (§3.4).

The parametric tail is not a compromise for lack of compute; it is the **lower-variance** estimator in the extreme (a p estimated from `k` tail exceedances has relative standard error ≈ `1/√k`).

### 3.6 Resolution targets

Resolution is set by the multiple-testing denominator, not chosen for its own sake:

| Stratum | Correction | Smallest meaningful p |
|--------|-----------|-----------------------|
| Cis | per-gene window (~few hundred CpGs → ~`1e-4`), then FDR over ~28k genes | ~`2e-6` |
| Trans | Bonferroni over ~`4.5e5` CpGs × ~`4.7e4` probes ≈ `2e10` pairs | ~`2.4e-12` |

The **number of permutations** is governed by **tail-shape convergence** — whether the GPD shape parameter `ξ` and the extrapolated genome-wide quantile stabilize as permutations increase — **not** by the target p. Reaching `1e-12` by counting would require ~`1e12` tail draws and is the wrong estimator; the GPD tail is both necessary and more honest there. Any value reported beyond the correction threshold should be given as `< threshold`, not as a precise number, since past that point the extrapolation's assumptions (Pareto-like tail, exchangeability) are the limiting uncertainty rather than the arithmetic.

> The ξ-convergence diagnostic that would *measure* this is not yet available: it requires the null accumulator, which is not persisted (§5, Phase 3b). Until then, the permutation count is set by judgment, not evidence.

### 3.7 Null storage

At trans scale the null cannot be materialized as a list (`~2e10` pairs × permutations). It is accumulated in **fixed memory**, independent of permutation count: a **fixed-resolution `|t|`-histogram** (with an overflow count for values beyond its range) plus a bounded **top-K tail-exceedance buffer** that retains the largest `|t|` seen exactly. The empirical body is read from the histogram; the GPD tail is fit to the exact exceedances in the buffer. Accumulation is streaming — each permutation folds into the same fixed structures, and the retained footprint does not grow with the number of permutations.

**These structures are in-memory only and are discarded when the run exits** — only the per-pair output is written (§3.9). That is the reason the Phase-3 diagnostics split into arms that can be computed from the output and arms that cannot (§5).

On the observed/reported side, a computed p for every pair does not imply writing every pair: the output supports an optional p-threshold (§3.9).

### 3.8 Validity conditions

- **Exchangeability / constant df.** The pooled null requires `df = n − k` constant across all pairs and no per-pair missingness that varies `n`. In the current pipeline `df` is a single run-level scalar (`df = n_samples − covariates − 2`) and missingness is removed globally before the solve, so this holds. If per-pair `df` ever varied, both the pooled null and the subsampling shortcut would break.
- **Pivotal statistic.** Pooling requires a scale-free statistic; the t-statistic qualifies, the raw coefficient does not (§3.1).
- **Annotations required.** Because the scored null is stratified by chromosome (§3.4), the method requires methylation and expression annotations. Running without them **fails closed** (raises) rather than silently scoring against an unstratified null. Unusable annotations follow a coverage-aware split (§3.4): a master pair whose locus was dropped in normalization (missing/unmappable chromosome) is **excluded** from scoring by default — it carries `NaN perm_mt_p` — while a `--pairs-file` that explicitly names such a pair **raises**.
- **Design consistency with the master.** Because the observed `mt_t` is read from the master while the null is built from the supplied `M`/`G`/`C`, the covariate design and `df` behind the master's statistic must match those of the null — otherwise the observed and null values reference two different models, and the p-values are confidently wrong with no error raised. The method surfaces this with an **advisory** sampled equivalence check: it recomputes the observed t for a random subset (~256) of master pairs from the provided `M`/`G`/`C` and compares to the stored `mt_t`. On divergence it emits a prominent, **self-diagnosing** warning — reporting `max|dt|` and `corr(stored, recomputed)` — and continues; it does **not** raise. The warning is the diagnostic: `corr ≈ 1.0` marks a benign `float32` execution-order divergence between the mapping's and permute's QR solvers (same algorithm, same `df`) on real-scale data, whereas a `corr` well below 1.0 or a large `max|dt|` indicates a genuine design/`df` mismatch — in which case the p-values are invalid. The tolerance (`rtol = atol = 1e-2`) is calibrated to that real-data `float32` floor (`max|dt| ≈ 7.7e-3` observed on gtpsub, 340 samples), not to clean fixture data. It is advisory rather than fatal so a benign excess cannot abort a multi-hour GPU run; the separate `--pairs-file` usage errors (§3.4) stay fatal.

### 3.9 Output schema

The output is the **master parquet with `perm_mt_p` merged on** — keyed on `(mt_id, gt_id)`, additive, and `mt_p` is never overwritten — mirroring how `qr_bootstrap` merges its columns. It therefore carries the master's columns (including the observed `mt_t`, **read not recomputed**) plus `perm_mt_p` and the run's `seed` and `n_perm`. Master rows that were not scored (e.g. outside a `--pairs-file` subset) carry `NaN perm_mt_p`. Provenance is additionally stamped into the **parquet** schema metadata as `tecpg_perm_seed`, `tecpg_perm_n_perm`, `tecpg_perm_n_reported`, and `tecpg_perm_n_null_pairs` (the size of the trans-filtered null grid, without which the empirical resolution floor cannot be reconstructed — §8).

`tecpg_perm_n_reported` records the **pre-threshold** universe size. This matters: scoring computes a p for every reported pair, but an optional `output_p_threshold` writes only pairs at or below a cutoff (genome scale cannot materialize `~2e10` rows). The default writes **all** reported pairs, preserving the full FDR universe on disk; when a threshold is used, `tecpg_perm_n_reported` is what keeps a downstream BH-FDR correction honest about the universe it was drawn from (Phase 4).

> **One current gap.** (i) The metadata keys are written on the **parquet** path only — a CSV run keeps the `seed` and `n_perm` *columns* but loses `n_reported`, so a thresholded CSV artifact cannot reconstruct its FDR universe. Prefer parquet for permutation output.

---

### 3.10 Column reference

Every per-pair quantity the pipeline computes, the artifact it first appears in, and what it means. Columns are **carried forward** from the artifact that creates them: each stage writes a new file and appends, so a column present at one row of this table is present in every artifact below it in the same chain. The two chains are independent — the mainline catalogs and the permutation output are separate files built from separate runs, and no column crosses between them.

`(mt_id, gt_id)` is the join key throughout: the methylation probe and the gene/transcript of a tested pair.

#### Mainline chain — `pipeline.sh`

| Column | First appears in | Written by | Description |
|---|---|---|---|
| `mt_id`, `gt_id` | `merged.parquet` | `tecpg run mlr --mlr-method qr` (stage 3 of 9) | Pair identity. Emitted as a two-level index by the mapper and promoted to columns during merge. |
| `mt_est`, `mt_err`, `mt_t`, `mt_p` | `merged.parquet` | same | Methylation coefficient, its standard error, the t-statistic, and the two-sided analytic p. `mt_p` is **float32** and saturates below roughly `1e-7`; see `precise_mt_p`. |
| `const_est`, `const_err`, `const_t`, `const_p` | `merged.parquet` | same | The intercept term's regression outputs. |
| `<covariate>_est`, `_err`, `_t`, `_p` | `merged.parquet` | same | One quartet per column of `C.csv`, named after the covariate. |
| `mt_ig`, `<covariate>_ig` | `merged.parquet` | same, with `--compute-ig` | Integrated-gradients attribution per pair. Present only when IG is requested; the covariate set is controlled by the IG covariate filter. |
| `mt_chrom`, `mt_chromStart`, `mt_strand` | `annotated.parquet` | `tools/assignRegionToEcpg_parquet.py` (stage 5 of 9) | Probe coordinates, joined from `M.bed6`. |
| `gt_chrom`, `gt_chromStart`, `gt_strand` | `annotated.parquet` | same | Gene coordinates, joined from `G.bed6`. `gt_chromStart` is the TSS. |
| `region` | `annotated.parquet` | same | The canonical 7-way label: `PROMOTER`, `GENEBODY`, `CIS5`, `CIS3`, `DISTAL5`, `DISTAL3`, `TRANS`. One label per row, assigned by an if/elif chain, so the labels partition the rows exactly. Null where annotation was missing. This tool is the single region authority; nothing downstream re-derives it. |
| `precise_mt_p` | `annotated_pcalc.parquet` | `tools/recalculate_pvalues_parquet.py` (stage 6 of 9) | The two-sided analytic p recomputed in **float64** from the stored `mt_t` and the run's degrees of freedom. This is the p-value of record for ranking and for FDR — `mt_p` is retained unchanged beside it but must not be used as a fallback, since float32 cancellation places an artificial floor near `5.96e-08`. |
| `fdr_est` | `summarized.parquet` | `tools/summarizeOutput_parquet.py` (stage 7 of 9) | Benjamini–Hochberg FDR over `precise_mt_p`, with the step-down monotonicity constraint applied. The denominator is `TOTAL_TESTS`, the full mapping grid, not the row count of the catalog — the catalog is the top-N surviving the mapping threshold, so the correction must reference the universe actually tested. |
| `is_significant` | `summarized.parquet` | same, with `--assign-fdr-passfail` | Boolean pass/fail against the FDR threshold. **Not produced on the golden path**; `pipeline.sh` does not pass this flag. |
| `mt_est_boot_mean`, `mt_est_boot_std` | `bootstrap_merged.parquet` | `tecpg run mlr --mlr-method qr_bootstrap` (stage 9 of 9) | Mean and standard deviation of the methylation coefficient across resamples, over finite draws only. |
| `ci_low`, `ci_high` | `bootstrap_merged.parquet` | same | The 2.5% and 97.5% quantiles of the resampled coefficient — a 95% percentile confidence interval. |
| `p_boot` | `bootstrap_merged.parquet` | same | Two-sided empirical bootstrap p: twice the smaller of the proportions of resampled coefficients at or below and at or above zero, clamped at 1. Floored at `1/finite_count` for that pair, so a strictly one-sided resample distribution reports the smallest representable value rather than exactly zero. |
| `degenerate_resamples` | `bootstrap_merged.parquet` | same | Count of resamples that produced a non-finite estimate (rank-deficient draws). These are excluded from every statistic above, and this count is the per-pair denominator adjustment. |
| `seed` | `bootstrap_merged.parquet` | same | The RNG seed actually used, stamped on every row so a run is reproducible from its artifact. |

Bootstrap runs only over the pairs selected into `bootstrap_list.csv`, with `summarized.parquet` as its master, so `bootstrap_merged.parquet` is a row subset of the catalog carrying all of the columns above.

#### Permutation chain — `pipelinePermute.sh`

The permutation master is assembled independently of the mainline catalog. `tools/build_gene_anchored_master.py` and `tools/reservoir_to_parquet.py` both subset to `{mt_id, gt_id, mt_t}`, adding `mt_p` only when both sources carry it. **`precise_mt_p` is not present in the permutation chain**: the reservoir and the cis write-all map both bypass mainline stage 6 of 9, which is the only place that column is ever computed.

| Column | First appears in | Written by | Description |
|---|---|---|---|
| `mt_id`, `gt_id`, `mt_t` | the master parquet | `tools/build_gene_anchored_master.py` or `tools/reservoir_to_parquet.py` | Pair identity and the observed t-statistic, **read from the mapping run and never recomputed**. `mt_t` is what the permutation null scores against. |
| `mt_p` | the master parquet | same | Carried through only when present in every source. The float32 analytic p, subject to the same saturation floor noted above. |
| `region` | the annotated master | `tools/assignRegionToEcpg_parquet.py` (permute stage 1) | The same 7-way label from the same authority as the mainline chain. Rides through the permute merge so the evaluation can stratify on it. Skipped with `--no-assign-regions`, in which case the evaluation falls back to a coarser two-way split. |
| `perm_mt_p` | `permutation_results.parquet` | `tecpg run mlr --mlr-method qr_permute` (permute stage 2) | The permutation-null p for the pair: the empirical tail probability of the observed \|t\| against the chromosome-stratified, design-fixed Freedman–Lane null, with a generalized-Pareto fit beyond the empirical threshold. **float64.** Null for master rows that were not scored. |
| `seed`, `n_perm` | `permutation_results.parquet` | same | Run-level constants stamped on every row: the RNG seed and the number of permutations. Also written to parquet schema metadata as `tecpg_perm_seed` and `tecpg_perm_n_perm`, alongside `tecpg_perm_n_reported` and `tecpg_perm_n_null_pairs`. |

`tools/eval_permute.py` and `tools/summarize_permute.py` are **readers**. They consume `permutation_results.parquet` and emit `eval_permute_report.json` and a markdown summary; neither writes a column, and the report carries a per-region calibration verdict rather than any p-value.

#### Phase 4 mainline annotation — `pipelinePermute.sh [5/5]`

| Column | Target artifact | Written by | Description |
|---|---|---|---|
| `p_permute` | `*.permute.parquet` | `tools/annotate_permute_p.py` (stage 5 of 5) | The analytic p-value **licensed by the permutation verdict**: `precise_mt_p` copied forward for rows whose region calibrated against the permutation null, null elsewhere. It is a licensed selection of an existing quantity, not an independently measured one — `perm_mt_p` is the measured permutation p. A region is licensed when its status is `ok` or `reference` **and** it is absent from `divergent_regions`; note that divergent regions carry status `ok`, so status alone is not the predicate. |
| `fdr_permute` | `*.permute.parquet` | `tools/summarizeOutput_parquet.py` (stage 5 of 5) | Benjamini–Hochberg FDR over `p_permute`, sharing `fdr_est`'s `TOTAL_TESTS` denominator. Null wherever `p_permute` is null, so "not assessed" stays distinguishable from "assessed and not significant". When every region in a catalog is licensed, this column is identical to `fdr_est` by construction; it diverges — upward, more conservative — as strata drop out. |

Both are additive: the stage writes a new file and neither `mt_p`, `precise_mt_p`, nor `fdr_est` is modified. Downstream ranking is unchanged — `tools/createBootstrapList.py` continues to rank on `precise_mt_p`.

> **Retention.** Every stage in both chains writes a new artifact rather than editing in place, and every intermediate is kept: `merged` → `annotated` → `annotated_pcalc` → `summarized` → `bootstrap_merged` on the mainline side, and the master → `permutation_results` on the permutation side. Writes are additive by contract — `region` fails closed if the column already exists, `precise_mt_p` is appended, and `perm_mt_p`, `p_boot`, and `fdr_permute` refuse or drop only their own prior column on a re-run. Permutation output is retained for independent evaluation whether or not a given study elects to use it downstream.

---

## 4. Architecture

`qr_permute` is registered as a `--mlr-method` value and implemented in its own module, `tecpg/permute.py`, mirroring `qr_bootstrap`. Key architectural choices:

- **Isolated parallel method.** During development it touches **no shared solver and no output-processing script**. Existing analyses (`legacy_normal_eq`, `qr`, `qr_bootstrap`, all `tools/`) run unaffected on the same checkout. Byte-for-byte identity of existing methods is the standing contract, enforced by the committed fingerprint (`tests/fingerprint_all_pipeline.json`), which is never modified to make a test pass.
- **Coverage vs stratification are independent axes.** Coverage — which pairs are reported — is defined by the **master parquet** (optionally narrowed by `--pairs-file`), not by an internally generated product. The *scoring* stratum is derived per pair from its own chromosomes (§3.4), so a pair receives the same p regardless of which master it came from, as long as the design matches. Region-mask logic lives in a single shared helper (`helper.py:compute_region_mask`), used for the null stratification.
- **Post-mapping data-flow.** `qr_permute` follows the **`qr_bootstrap` shape**: read the master parquet, compute a new per-pair quantity, merge it back on `(mt_id, gt_id)`. The observed statistic is not recomputed — it is read from the master. What is genome-wide is the *null estimation* (subsampled), not an observed scan; the reported universe is exactly the master's (optionally narrowed by `--pairs-file`).
- **Solve is null-only, with a consistency guard.** The regression kernel implemented within `permute.py` (as `bootstrap.py` does) is used to build the **null** and to verify the master. The observed statistic comes from the master; a **sampled equivalence guard** recomputes the observed t for a random subset of master pairs from the supplied `M`/`G`/`C` and **warns (advisory)** if it diverges from the stored `mt_t` — a self-diagnosing message distinguishing benign `float32` noise from a real design/`df` mismatch, without aborting the run (§3.8). Numerical equivalence is guaranteed by a three-way oracle: consume-path (read) == recompute-path == independent per-pair OLS (numpy).
- **Diagnostics are read-only and torch-free.** The evaluation script (§5, Phase 3) consumes permutation output only; it imports no torch and no `tecpg` runtime module, so it runs anywhere and is testable without a GPU environment.
- **Single region authority, consumed not re-derived.** Region labels come from one place — `tools/assignRegionToEcpg_parquet.py` — and everything downstream (the eval, `createBootstrapList`, `runEnrichment`) reads its `region` column. The evaluation's per-region stratify (§5) consumes that column when present and falls back to the legacy same-chromosome split only when it is absent, so the eval speaks the same region currency as the rest of the pipeline rather than maintaining a second, coarser definition. The reproducible path is a **region-annotation stage in `pipelinePermute.sh`** that runs `assignRegionToEcpg_parquet.py` on the resolved master (chr-prefix-native, no p-filter) before permute, so `region` rides through the permute merge into the output for the eval to read; it is idempotent (skips a master that already carries `region`), bundled with the permute stage under the start-stage gate (so an eval-only restart does not re-annotate), and switchable off with `--no-assign-regions`. No cis-specific mapping is run — the full-map reservoir already spans every region.
- **One chromosome canonicalizer.** Chromosome-label canonicalization (strip a `chr` prefix, uppercase, map `X`/`Y`/`MT` to signed sentinels, coerce, treat unmappable as a dropped locus — §3.4) lives in a single torch-free helper, `tecpg/chrom.py:canonicalize_chrom`, consumed by both `permute.py`'s null annotation-normalization and the mapping's `region != 'all'` filtration path. Consolidating it fixed a latent `.astype(int)` crash on `chr`-prefixed annotations in the windowed-map path (which every real cohort hits), so the cis write-all map runs on GTP/MESA annotations directly.
- **Windowed-map predicate correctness (int64 bounds, ordered interval).** The cis/distal membership predicate derives its window bounds by multiplying the gene's strand by the window magnitude. Two latent defects lived there, both exposed only once a real ±1 Mb cis map was run at scale. First, the strand tensor is `int8` at every call site, and `int8 × ~1e6` **overflows** (`torch.tensor([1], dtype=torch.int8) * -1000000` → `-64`), collapsing the requested window to an effective ~±64 bp — a full-GTP cis map returned 2,741 pairs where millions were expected. Second, multiplying by strand flips the window's orientation but also **reverses the two bounds**, so a symmetric window became the empty interval `(+1e6, −1e6)` for every negative-strand gene. The fix widens the strand operand to `int64` immediately before the multiply and orders the bounds with `minimum`/`maximum`, applied identically to `compute_region_mask` and to `regression_full`'s duplicated predicate (they must move together, since their equivalence is itself a test). It is a **point-of-use** cast: the `int8` tensor construction is deliberately left alone, as it is built in three places and chunked downstream. Note the failure mode that hid this — the only pre-existing cis coverage was a qr-vs-`regression_full` equivalence oracle in which *both sides overflowed identically and therefore agreed*; the guard is now an independent `int64` reference oracle (`tests/test_region_mask.py`). `region='all'` skips the predicate entirely, so the mainline catalog and the permutation null (trans-only, no window arithmetic) are unaffected by both bugs and by their fix.
- **Cis-window enrichment tooling.** When the reservoir's near-gene coverage is too thin (`insufficient_near_gene_coverage`, §5), the near-gene calibration is powered by enrichment rather than by a denser uniform sample. `tools/build_gene_anchored_master.py` assembles a cis write-all map's near-gene pairs with the reservoir's trans/distal pairs — concatenate, dedupe on `(mt_id, gt_id)`, and **fail-closed** on a `mt_t` disagreement beyond tolerance for any pair present in both (a real disagreement means the two runs used different covariate designs). It carries no region logic; `assignRegionToEcpg_parquet.py` relabels the assembled master canonically. The whole enrichment path — cis write-all map → `mergeOutputs` → assemble → annotate → permute → per-region eval — is one `pipelinePermute.sh --cis-enrich` invocation, mirroring the existing map→merge flow.
- **Deferred downstream integration.** Wiring `perm_mt_p` into the downstream significance selectors and the BH-FDR universe is a deliberate final phase, undertaken only after the method is validated, so that in-progress output can never affect existing pipelines. The FDR step is not a new implementation: it reuses `tools/summarizeOutput_parquet.py`'s BH machinery, fed `perm_mt_p` and the `tecpg_perm_n_reported` universe (§5, Phase 4).

---

## 5. Implementation status

The build has **four phases**, delivered as a **walking skeleton**: the pipeline was first wired as trivial stubs (freezing interfaces and the output contract), then each stage replaced by real logic one **chunk** at a time, each guarded by an oracle and a **forced-fail proof** (green → red on a deliberately injected bug → green on revert).

- **Phase 1 — trans-global null** *(complete, chunks 0–9)*. Design-fixed Freedman–Lane; streaming fixed-memory null; empirical p + GPD tail; finalized output. All stage functions are real, and the null-side schema is settled (§3.9): `perm_mt_p`, `seed`, `n_perm`, parquet provenance metadata, and the optional output threshold. **Complete is not validated** — see Phase 3.
- **Realignment to the post-mapping-consumer architecture** *(complete)*. Phase 1 was first built as a **self-contained** method that generated its own universe and recomputed the observed statistic; it has been corrected to the `qr_bootstrap`-parallel form documented throughout: consume `--master-parquet` for the observed `mt_t` and the pair universe, the optional `--pairs-file` subset, the **universe ∩ normalized-M/G intersection** (§3.4), the **advisory** consistency guard (§3.8, §4), and the merge of `perm_mt_p` onto the master (§3.9). The null machinery, accumulator, GPD tail, and diagnostics are unchanged.
- **Phase 2 — cis Beta-approximation** *(not required for GTP; conditional per dataset)*. Per-gene min-p Beta null for the cis stratum (§3.4), fit in the t-domain, reusing Phase 1's residualize/refit primitives and the shared cis mask as each gene's local test set. It was gated on the cis-window enrichment verdict, and **that gate has now resolved for GTP in the negative**: the enrichment run returned `single_global_null_adequate`, so the near-gene regions are adequately served by the trans-global null and the Beta machinery is unnecessary for this cohort. It remains conditionally pending for **MESA** and the **oncology cohort**, each of which gets its own enrichment run and its own verdict; the machinery is built only if one of them shows a near-gene stratum diverging from `TRANS`.
- **Phase 3 — evaluation / diagnostics** *(partially delivered)*. A standalone read-only script. Its arms divide by what the persisted output can support (§3.7):
  - **3a — output-derivable** *(built)*: calibration of `perm_mt_p` against a `float64` analytic reference recomputed from `mt_t`; null sanity via genomic inflation and a uniformity test on the trans (mostly-null) bulk; and the **stratify-or-not decision** in its *calibration-divergence* form. When the output carries a canonical `region` column (§3.4, §4) the decision is **7-way per-region**: each region's bulk-band departure from the analytic reference (median `|log10(p_perm / p_ana)|`) is compared against the `TRANS` reference stratum, region by region, with a per-region minimum-count floor (`MIN_REGION_BULK_N`) marking under-populated regions `insufficient_data` so the verdict is never driven by noise. The recommendation keys on the **effect size** of the divergence for the near-gene regions (`CIS5`/`PROMOTER`/`GENEBODY`/`CIS3`) relative to `TRANS`; the two-sample test statistic is reported but non-gating (at genome scale its p-value collapses to zero from sample size alone). Three outcomes: **single global null adequate**, **stratification warranted** (with the divergent regions named), or **insufficient near-gene coverage** (the per-region gate that routes to enrichment, §8). Reporting `DISTAL5`/`DISTAL3` on par lets the analysis separate a genuine *cis-window* effect (near-gene diverges, distal does not → per-gene Beta) from a *same-chromosome structural* effect (near-gene and distal diverge together → a per-chromosome null, not per-gene Beta) — a distinction the old same-chromosome 2-way split could not make. When no `region` column is present, the eval falls back to the legacy 2-way (same-chrom) split. Genomic inflation (`lambda_trans`, `lambda_cis`, `lambda_excess`) is reported as a descriptive diagnostic but does **not** gate the decision — `lambda_GC` presumes a mostly-null test space, which holds for GWAS/eQTL but not eQTM, and least of all in cis where a large fraction of pairs carry real signal, so `lambda_cis > 1` is expected biology rather than miscalibration (§2). The legitimate concern that broad cis signal leaks into the bulk and masquerades as null divergence is left to a calibration-native detector — a uniformity test on the cis-bulk analytic p, mirroring the trans-bulk test — deferred rather than adjudicated by lambda (§8).
  - **3b — sidecar-gated** *(deferred)*: ξ / extrapolated-quantile convergence (§3.6), literal null flatness, and the rigorous *null-shape* stratify comparison. These need the null accumulator itself, which is not persisted, so they are stubbed behind an optional sidecar input whose contract is frozen in the consumer. Whether to add the corresponding writer to the core module is deliberately left to the 3a evidence.
  
  **3a run status (GTP subset).** The calibration diagnostic needs a master that spans the full p-value range (a mostly-null *bulk*); a threshold-filtered master (e.g. `bootstrap_merged.parquet`, significant pairs only) yields an empty bulk, a `lambda` of ~27 that is a filter artifact rather than inflation, and a skipped verdict. The right calibration master is the mapping's **`sample_reservoir.csv`** — a uniform sample taken *before* p-value filtration (`--reservoir-count`), carrying `mt_t` over the full range — converted to parquet and consumed like any other master (`pipelinePermute.sh --reservoir` automates the CSV→parquet conversion, the region-annotation, the two-step run, and the summary; `tools/summarize_permute.py` renders the report into a text digest plus figures). On that master the **trans** bulk calibration is near-perfect (median `|log10(p_perm/p_ana)|` ≈ 0.003, `lambda_trans` ≈ 1.007) and the trans null's bulk calibration is validated.

  **7-way per-region run (full GTP).** The reservoir-first per-region evaluation has been run — one `pipelinePermute.sh --reservoir` invocation, annotate → permute → per-region eval. `TRANS` and both **well-powered** distal strata calibrate to the analytic reference and to each other: `TRANS` bulk median `|log10(p_perm/p_ana)|` ≈ 0.0027 (`lambda_trans` ≈ 1.006), `DISTAL5`/`DISTAL3` departures within ~1e-5 of `TRANS` (`divergent_regions = []`). The near-gene regions, however, were essentially empty in a uniform sample (a ~1M-pair reservoir held only ~100 near-gene pairs, ~91 in the bulk band, below `MIN_REGION_BULK_N`), so the eval correctly returned **`insufficient_near_gene_coverage`** rather than a noise-driven cis verdict. Two readings follow. First, because the well-powered distal strata calibrate exactly like trans, a *per-chromosome structural* divergence is ruled out — whatever the cis window does is a near-gene-*specific* effect, so post-enrichment the routing is binary (near-gene calibrated ⇒ skip Phase 2; near-gene diverges ⇒ per-gene Beta). Second, the coverage gate routes to the **cis-window enrichment** path, which is now built (§4, §8): a chr-prefix-native cis write-all map assembled with the reservoir's trans/distal pairs, re-annotated, permuted, and evaluated per region. The `Counts by Region` line remains the free coverage gate; the eval thresholds (`BULK_LO`, `TOLERANCE`, `MIN_REGION_BULK_N`) remain provisional placeholders to be re-set from the enrichment run and from MESA. **The verdict is per-dataset:** MESA and the oncology cohort each carry their own confounding structure, so the near-gene calibration — and therefore the mainline p method (§8) — must be re-read for each, not inherited from GTP.

  **Enrichment run (full GTP) — complete.** The first `pipelinePermute.sh --cis-enrich` attempt validated the wrapper end-to-end but produced only **2,741** cis pairs from a requested ±1 Mb window and failed at the assemble step. Three independent defects, all since fixed: the windowed-map predicate's **`int8` overflow** and **negative-strand bound reversal** (§4), which together meant the "±1 Mb" window was really ~±64 bp on positive-strand genes and *empty* on negative-strand ones; and the assembly tool failing to promote `mt_id`/`gt_id` from the map parquet's **named index** (the map writes them as an index, and `mergeOutputs`' parquet→parquet path is a raw Arrow passthrough that preserves it).

  With those corrected the run completed cleanly. The map returned **16,526,333** cis pairs — a 6,029× recovery, and an independent confirmation of both window fixes on real data. `DISTAL5` and `DISTAL3` came out within **0.68%** of each other (7,674,941 vs 7,727,379), which is the direct signature of the negative-strand fix: a symmetric window applied to both strands produces a symmetric split, and pre-fix the `−` strand contributed nothing. Assembly produced a **17,525,137**-pair master (16,526,333 cis + 1,000,000 reservoir − 1,196 overlapping, with no `mt_t` disagreement across the overlap). Region counts: `trans` 931,682 · `distal5` 7,674,941 · `cis5` 549,615 · `promoter` 104,252 · `genebody` 5,177 · `cis3` 514,447 · `distal3` 7,727,379.

  **Verdict: `single_global_null_adequate`, `divergent_regions = []`.** The near-gene family (`CIS5`+`PROMOTER`+`GENEBODY`+`CIS3`) totalled **1,098,442** bulk pairs against `MIN_REGION_BULK_N` = 100 — roughly 10,980× headroom, so `insufficient_near_gene_coverage` is comprehensively off the table and every one of the seven regions is `ok`. (An earlier revision of this document quoted 1,173,491 here; that is `n_cis`, the total near-gene pair count, not the bulk-band subset the coverage gate tests.) All seven calibrate to the analytic reference at a bulk median `|log10(p_perm/p_ana)|` of ~0.0027, across strata spanning 4,884 to 7.3M bulk pairs. The largest near-gene departure is **3.162e-05** (`PROMOTER`) against a tolerance of 0.5 — a margin of ~15,800×, with the verdict unchanged down to a tolerance of 1e-4 and first flipping at 3e-5. Three findings are recorded rather than buried. **(i) The reference quantity is `precise_mt_p`.** `eval_permute.py:compute_analytic_p` does not read a stored column; it recomputes `2·scipy.stats.t.sf(|t|, df)` from `mt_t` in `float64` — the same expression `recalculate_pvalues_parquet.py` uses — so what the permutation validated is exactly the array the mainline `precise_mt_p` column already holds, and the `float32` floor of the stored `mt_p` (§2) never entered the comparison. **(ii) The near-gene offset is detectable, not absent.** Mann–Whitney rejects distributional identity with `TRANS` for the three well-powered near-gene strata — `CIS5` p = 1.2e-10, `PROMOTER` p = 7.0e-07, `CIS3` p = 1.3e-02 — while *both* distal strata, carrying 14× more bulk pairs than `CIS5`, do not (p = 0.137, 0.140). Power is therefore not the explanation: there is a real, ordered near-gene offset, ranked `PROMOTER` > `CIS5` > `CIS3` > distal ≈ trans, the same order as λ and as the tail ratio. This *strengthens* the result rather than undermining it, because it shows the test is not insensitive — it resolves a difference at p = 1e-10 and measures that difference as 3.2e-05, a factor of 1.00007 between the two p-values. The honest claim is a bounded measurement, not an equivalence. **(iii) The direction is consistent.** Permutation p exceeds analytic p for 99.555% of the 16,582,226 bulk pairs at a median ratio of 1.00632, and the sign-flip rate is flat across all seven regions (0.41–0.45%), so the offset is a shift in the body of the ratio distribution rather than a change in the fraction of reversals.
- **Phase 4 — mainline annotation** *(complete)*. The verdict changes what this phase has to be. Because every GTP stratum calibrates, `p_permute` for a calibrated stratum **is** the validated analytic p — which is the array `precise_mt_p` already contains — so there is no per-pair permutation to merge and no brute-force scan. Two decisions bound the work. First, `p_permute` and `fdr_permute` are added as **standard columns** rather than omitted where redundant, so the schema is stable across cohorts and downstream consumers can rely on them for a dataset whose verdict differs. Second, the ordering forbids doing it in `[7/9]`: that stage runs inside `pipeline.sh`, before any permutation exists, and repopulating it would invalidate the bootstrap candidate list and everything `pipelinePost.sh` consumes. The annotation is therefore a **new final stage, `pipelinePermute.sh [5/5]`**, built in three chunks.

  *Delivered in three chunks.* **A (`7dfa516`)** parameterized `tools/summarizeOutput_parquet.py` with `--p-column`/`--fdr-column`, defaulting to `precise_mt_p`/`fdr_est` so the existing `pipeline.sh` invocation is unchanged, and added three guards: an explicitly named p-column that is absent **fails closed** rather than falling back to the t-statistic; naming an existing column as `--fdr-column` is refused; and a row whose source p is null yields a **null** FDR rather than `1.0`, keeping *not assessed* distinguishable from *assessed and not significant*. **B (`b11ee1e`)** added `tools/annotate_permute_p.py`, which copies `precise_mt_p` into `p_permute` for the licensed strata. **Licensing is keyed on the per-region status *and* the `divergent_regions` list, not on status alone** — `eval_permute.py` records a divergent region with `status: 'ok'` and marks divergence only by list membership, so a status-only predicate would license precisely the strata the verdict excluded. The predicate is `{R : status in (ok, reference)} - set(divergent_regions)`. **C (`36f3af8`)** added the wrapper stage `[5/5]`, applied to `summarized.parquet` and `bootstrap_merged.parquet`, writing to a new `*.permute.parquet` in both cases.

  Two properties of the stage are worth stating because they are checkable. The BH denominator is the **mapping grid `TOTAL_TESTS`** — the same denominator `fdr_est` uses — and *not* the permute run's `tecpg_perm_n_reported`, which is the permute universe and roughly three orders of magnitude smaller; the tool's `total_tests >= total_rows` guard does not catch a wrong-but-large value, so the wrapper takes it as an explicit flag and cross-checks it against `mlr_run_<ds>.log`. And on a **fully licensed** catalog `fdr_permute` is identical to `fdr_est` elementwise by construction, which makes any deviation a wiring defect; as strata drop out the columns diverge, always **upward** (more conservative) on the survivors, because unlicensed rows leave the BH pool and ranks shift down.

  **No selector is changed** — `createBootstrapList.py` keeps ranking on `precise_mt_p`, so the candidate list and all of `pipelinePost.sh` stay bit-identical; `pipelinePost.sh` continues to read the un-annotated `bootstrap_merged.parquet`. Promoting `p_permute` to a ranking axis, and repointing `pipelinePost.sh`, remain deliberate decisions for later.
- **Existing-method byte-identity.** `legacy_normal_eq`, `qr`, `qr_bootstrap`, and all `tools/` behave identically; the fingerprint JSON is never modified to make a test pass.
- **Forced-fail proof.** Every guard and every real computation is verified by injecting a bug, observing the test fail, then reverting — a passing test that does not fail on injection is not accepted as verification. This applies to read-only diagnostics as well as to core code: a green suite is not evidence of coverage, and assertions that run under the wrong test name are worse than absent ones.
- **Oracle / differential tests.** Correctness is checked against independent implementations — for the observed statistic, a **three-way equivalence** that the value read from the master equals both an in-module recompute and an independent per-pair OLS (numpy `lstsq`); the read-vs-recompute leg proves the consume-path plumbing, and the numpy leg breaks the circularity of two internal paths sharing one solver. Also: a single-permutation numpy Freedman–Lane for the residualization; a direct null recount for the empirical p; GPD parameter recovery on simulated Pareto data for the tail — not against stored golden outputs. Where a diagnostic re-implements a pipeline quantity, it must use the **same estimator** as the pipeline (e.g. the same GPD fitting method), or it measures something other than what it claims to.
- **Design-fixed Freedman–Lane.** The null permutes the reduced-model response residuals with the design held fixed (§3.2); permuting a predictor is a correctness *and* cost error.
- **Pivotal statistic.** The pooled/scored statistic is the t-stat, never raw β (§3.1).
- **Bounded-memory accumulation.** The null footprint is fixed and independent of permutation count (§3.7).
- **Per-pair floor and positivity.** The empirical p is floored per pair at `1 / (N_null + 1)`; the GPD extension is clamped strictly positive.
- **Fail-closed / fail-safe guards.** Missing annotations, non-positive subsample counts, an empty post-intersection universe, `--pairs-file` pairs absent from the master, and a `--pairs-file` pair whose locus was dropped in normalization all fail closed (raise); a degenerate tail fit fails safe to the empirical p. A dropped-locus master pair *not* named in a `--pairs-file` is excluded, not raised (§3.4).
- **Master-consistency guard (advisory).** The observed `mt_t` read from the master is checked against the supplied `M`/`G`/`C` by a sampled equivalence spot-check; a design/`df` mismatch (or a sampled pair absent from the data) emits a loud self-diagnosing warning and continues rather than raising (§3.8, §4). Its test is that the warning **fires** on a real mismatch and **stays silent** on a match — both directions asserted.
- **Additive merge.** `perm_mt_p` is merged onto the master on `(mt_id, gt_id)`; `mt_p` is never overwritten and unscored rows fall through to `NaN`.
- **Seed / determinism.** Permutation draws follow a recorded seed (default 42); the same seed reproduces the null, and the seed is written to the output and its parquet metadata (§3.9).

---

## 6. Correctness contracts

These are enforced for every chunk:

- **Existing-method byte-identity.** `legacy_normal_eq`, `qr`, `qr_bootstrap`, and all `tools/` behave identically; the fingerprint JSON is never modified to make a test pass.
- **Forced-fail proof.** Every guard and every real computation is verified by injecting a bug, observing the test fail, then reverting — a passing test that does not fail on injection is not accepted as verification. This applies to read-only diagnostics as well as to core code: a green suite is not evidence of coverage, and assertions that run under the wrong test name are worse than absent ones.
- **Oracle / differential tests.** Correctness is checked against independent implementations — for the observed statistic, a **three-way equivalence** that the value read from the master equals both an in-module recompute and an independent per-pair OLS (numpy `lstsq`); the read-vs-recompute leg proves the consume-path plumbing, and the numpy leg breaks the circularity of two internal paths sharing one solver. Also: a single-permutation numpy Freedman–Lane for the residualization; a direct null recount for the empirical p; GPD parameter recovery on simulated Pareto data for the tail — not against stored golden outputs. Where a diagnostic re-implements a pipeline quantity, it must use the **same estimator** as the pipeline (e.g. the same GPD fitting method), or it measures something other than what it claims to.
- **Design-fixed Freedman–Lane.** The null permutes the reduced-model response residuals with the design held fixed (§3.2); permuting a predictor is a correctness *and* cost error.
- **Pivotal statistic.** The pooled/scored statistic is the t-stat, never raw β (§3.1).
- **Bounded-memory accumulation.** The null footprint is fixed and independent of permutation count (§3.7).
- **Per-pair floor and positivity.** The empirical p is floored per pair at `1 / (N_null + 1)`; the GPD extension is clamped strictly positive.
- **Fail-closed / fail-safe guards.** Missing annotations, non-positive subsample counts, an empty post-intersection universe, `--pairs-file` pairs absent from the master, and a `--pairs-file` pair whose locus was dropped in normalization all fail closed (raise); a degenerate tail fit fails safe to the empirical p. A dropped-locus master pair *not* named in a `--pairs-file` is excluded, not raised (§3.4).
- **Master-consistency guard (advisory).** The observed `mt_t` read from the master is checked against the supplied `M`/`G`/`C` by a sampled equivalence spot-check; a design/`df` mismatch (or a sampled pair absent from the data) emits a loud self-diagnosing warning and continues rather than raising (§3.8, §4). Its test is that the warning **fires** on a real mismatch and **stays silent** on a match — both directions asserted.
- **Additive merge.** `perm_mt_p` is merged onto the master on `(mt_id, gt_id)`; `mt_p` is never overwritten and unscored rows fall through to `NaN`.
- **Seed / determinism.** Permutation draws follow a recorded seed (default 42); the same seed reproduces the null, and the seed is written to the output and its parquet metadata (§3.9).

---

## 7. Usage

> The trans pipeline produces real p-values, but the method is **not yet validated** (see the status note at the top). Current output should not be used for inference.

`qr_permute` is a **two-step** flow: map first to produce the master, then permute consuming it.

```
# 1. Map — produce the master parquet (observed mt_t over your universe).
tecpg run mlr --mlr-method qr --output-format parquet
#    → e.g. <output_dir>/<mapping>.parquet

# 2. Permute — consume that master; score its universe against the null.
tecpg run mlr --mlr-method qr_permute \
    --master-parquet <mapping>.parquet \
    --permutations 1000 \
    --subsample-mt-count <N> --subsample-g-count <N> \
    --seed 42 \
    --output-format parquet
    # optional: --pairs-file candidates.csv   (score only a subset)
```

**Reproducible per-region path.** The end-to-end calibration run is wrapped by `pipelinePermute.sh --reservoir`, which converts the mapping's `sample_reservoir.csv` to a master, **annotates it with the canonical `region` column** (`assignRegionToEcpg_parquet.py`), runs the permute (the `region` column rides through the merge), and evaluates **per region** — one command whose `Counts by Region` line doubles as the near-gene coverage gate (§5). `--no-assign-regions` skips the annotation, in which case the eval falls back to the legacy 2-way (same-chrom) strata.

**Mainline annotation stage.** The final stage of `pipelinePermute.sh`, `[5/5]`, annotates the mainline catalogs with `p_permute` and `fdr_permute` (§3.10). It runs `annotate_permute_p.py` and then `summarizeOutput_parquet.py` over `summarized.parquet` and `bootstrap_merged.parquet`, writing each to a new `*.permute.parquet` and leaving the input untouched.

```bash
./pipelinePermute.sh --dataset <ds> --master-parquet <master> \
    --start-stage eval --total-tests <grid size>
```

`--total-tests` is **required** whenever the stage annotates a catalog, and must be the **mapping grid** — the same `TOTAL_TESTS` `pipeline.sh` uses for `fdr_est`, not the permute run's `tecpg_perm_n_reported`. It is cross-checked against `mlr_run_<ds>.log` when that log is present. `--no-annotate-mainline` skips the stage, which is the right choice for a smoke run: annotation off a low-`B` verdict produces a real-looking catalog from a licensing decision that was never meant to stand.

The stage fails closed on a missing eval report, on a catalog lacking `region` or `precise_mt_p`, on a `--total-tests` that disagrees with the log, and on a row-count mismatch after either step. It skips — with a logged status, not an error — when a mainline catalog is absent (the permute pipeline is legitimately run standalone) or already annotated. A terminal summary block names what was annotated and what was skipped.

**Cis-window enrichment path.** When the reservoir's near-gene coverage is insufficient (§5), `pipelinePermute.sh --cis-enrich` runs the enrichment end-to-end from the same entrance: a cis write-all map (`--cis`, `-p 1.0`, over a generous `--cis-window`, over-captured then relabelled canonically) → `mergeOutputs` → `build_gene_anchored_master.py` (assemble with `sample_reservoir.csv`) → annotate → permute → per-region eval. `--start-stage permute` reuses an existing assembled master rather than re-running the map.

Relevant options:

- `--mlr-method qr_permute` — select the permutation method (qr-family, post-mapping consumer).
- `--master-parquet` — **required.** The mapping output supplying the observed `mt_t` and the `(mt_id, gt_id)` universe. Its covariate design/`df` must match the supplied `M`/`G`/`C` (§3.8, checked fail-closed).
- `--pairs-file` — **optional.** A candidate subset (`mt_id`, `gt_id` CSV); default universe = the entire master. A pair absent from the master fails closed.
- `--permutations` — number of permutations (default 100). Governed by tail-shape convergence, not the target p (§3.6).
- `--all` / `--cis` / `--distal` / `--trans` — standard region flags; under the consumer model the reported set is the master (optionally narrowed by `--pairs-file`), not these flags. The scoring stratum is derived per pair (§4).
- `--subsample-mt-count` / `--subsample-g-count` — random loci selection for null estimation (§3.3).
- `--seed` — permutation/subsample seed, recorded with the output.
- `--output-p-threshold` — writes only pairs at or below this permutation p-value cutoff. Pre-threshold size is retained in metadata.

**Annotations are required.** Because the null is chromosome-stratified (§3.4, §3.8), methylation and expression annotations must be supplied; running without them raises rather than silently producing an unstratified null.

**Output:** the master with `perm_mt_p` (and the run's `seed`, `n_perm`) merged on — `permutation_results.{parquet,csv}` by default. It carries the master's columns (including the observed `mt_t`) plus `perm_mt_p`; unscored rows carry `NaN perm_mt_p`, and `mt_p` is left untouched. An explicit `--output-file` is honored verbatim; the default writes `permutation_results.<ext>` into the output directory. **Parquet is recommended** — the CSV path does not carry the provenance metadata (§3.9).

### 7.1 QC report

`tools/permute_qc_report.py` renders a single self-contained HTML QC report from the
evaluation output, modelled on FastQC: a navigation sidebar of modules, each with a
status badge, a stated **Purpose**, a figure and/or table, and an **Interpretation**
block. Figures are base64 data URIs and CSS is inline, so the file opens offline with no
sibling files and no script element.

```bash
python3 tools/permute_qc_report.py \
    --report          output_gtp/eval_permute_report.json \
    --perm-output     output_gtp/permutation_results.parquet --df 321 \
    --gene-annotation annot_gtp/G.bed6 \
    --meth-annotation annot_gtp/M.bed6 \
    --dataset gtp --out output_gtp/permute_qc_report.html
```

Only `--report` is required. Modules whose inputs are absent render an `INFO` panel
naming every missing flag, and the report always generates.

Twelve modules, in render order: `run-provenance`, `region-composition`,
`bulk-calibration`, `calibration-direction`, `stratification`, `verdict-robustness`,
`permutation-resolution`, `tail-behaviour`, `analytic-p-precision`, `cis-pair-density`,
`gene-span-distribution`, `tss-distance-by-region`. **`permutation-resolution` precedes
`tail-behaviour` by design** — tail ratios are bounded below by the resolution floor and
are uninterpretable without it.

The report is a **renderer, not an authority**: every calibration statistic and the
verdict come from `eval_permute.py`, and the only quantities computed here are those
absent from the report JSON (the resolution floor, the tolerance sweep, and the
correctness diagnostics that read the parquet or the annotations directly). Region
labels are read from the canonical `region` column and never re-derived; annotation
parsing and every taxonomy boundary are imported from `assignRegionToEcpg_parquet.py`.

Two modules are deliberately unbadged. **Genomic inflation** appears only as a column in
the stratification table: λ measures inflation against a mostly-null expectation that
near-gene eQTM pairs violate by construction, so its regional gradient largely restates
where signal lives rather than measuring calibration quality (§5, 3a). **Tail behaviour**
carries an `INFO` badge under all conditions because a large tail ratio is confounded
with both the permutation resolution floor and signal density, and no calibrated
threshold separates those causes at present permutation counts (§8).

---

## 8. Limitations and open decisions

- **Validated in the bulk, across all seven regions, on GTP only.** The enrichment run (§5) validates the `float64` analytic p — the `precise_mt_p` quantity — in the mostly-null bulk band (0.05 ≤ p ≤ 1) for every region including near-gene, on one cohort. **What that licenses is the null model**: that the residualized statistic follows t₃₂₁ under the null in every region class. Given a correct null model the analytic tail follows exactly from the t distribution, so the tail claim rests on that model assumption with the bulk as the evidence it holds where it can be checked — it is not an extrapolation of the permutation. Two limits bound the claim: it is **per-dataset**, and it is **bulk-only** (see below).
- **Open — the tail is unmeasured at B = 10, and that is where the findings are.** The empirical permutation p cannot resolve below `1 / (n_perm × n_null_pairs)` = `1 / (10 × 3,793,007)` ≈ **2.64e-08**; below it, values come from the GPD extrapolation rather than counted exceedances. Any pair whose true p sits far past that limit receives a floored or extrapolated permutation p, so `p_perm / p_ana` there measures distance to the floor and says nothing about whether the analytic p is right. The per-region tail ratios order exactly as signal density predicts (`TRANS` 90th-percentile ratio 2.3×, rising monotonically to `PROMOTER` 6.5e8×), which is consistent with a resolution artefact **and** with genuine signal; the two are not separable at this permutation count. Note also that the two floors sit in the same decade — permutation 2.64e-08, stored analytic 5.96e-08 (§2) — so at B = 10 permutation buys only ~2.3× of resolution over the stored column. **No conclusion about the analytic tail should be drawn from the current run in either direction**, and the limitation bites hardest precisely where the catalog's headline findings live: the significant near-gene pairs. The open decision is whether to fund a higher-B run (the floor falls linearly: B = 100 → 2.6e-09, B = 1,000 → 2.6e-10) or to scope the published claim to the bulk and say so. Current position: **scope to the bulk and declare the limitation.** Reported without a status badge in the QC report for this reason.
- **Resolved (`0225916`) — the QC report's resolution module computed the wrong floor.** `build_permutation_resolution_module` took `min(observed non-zero p)` as the floor. Because values past the empirical limit come from the GPD extrapolation, that quantity sat at 9e-17 — 2.9e8× below the true floor — so "tail pairs at floor" was a tautology (the sample minimum is attained by at least one pair), "implied effective null draws ≈ 1/floor" inverted a fitted quantile as though it were a draw count, and the module returned `PASS` on a resolution finding nothing had measured. Its own interpretation text described the floor correctly throughout; only the implementation disagreed. The fix persists `tecpg_perm_n_null_pairs` (§3.9), derives `perm_resolution_floor` in `eval_permute.py` as an authority statistic, redefines `frac_tail_at_floor` against it, and returns `INFO` rather than `PASS` when the floor is unknown. `--n-null-pairs` supplies the value for outputs written before the stamp, so existing runs need not be repeated.
- **Resolved (`0225916`) — two QC badges asserted more than their data supported.** Gene span warned whenever most spans were short, which is the expected state on any array platform and therefore a false positive on every array cohort; it now flags only non-positive spans and spans longer than any known human gene, both malformed under either platform. TSS distance returned `PASS` while three of its six rows carried no applicable check; the uncheckable rows now read `n/a (span-dependent)` and the badge states its scope.
- **Resolved — the "empty decade" was a counting artefact, and it is `float32`.** On the assembled GTP master `P < 1e-8` and `P < 1e-9` both returned 2,092 pairs, suggesting zero pairs in `[1e-9, 1e-8)`. There is no gap. `reportPvalues` is cumulative and zero is less than every threshold, so the **2,092 exact zeros** were counted into both bins; the smallest genuinely non-zero stored p is **5.96e-08**, above 1e-7, so the decade is empty by construction. The QC module confirms it: shallowest empty decade = None once zeros are accounted for. An earlier revision ruled out `float32` by citing the *range* limit (1.18e-38); the mechanism is the *epsilon* limit under cancellation, `2⁻²⁴` = 5.96e-08, exactly as §2 describes. Consequence for the catalog: 2,092 pairs — 26% of the tail band — are unrankable by the stored `mt_p`. Already resolved on the mainline by `precise_mt_p` (§2); it remains a property of the permute master.
- **Out of scope — expression-feature span and the `GENEBODY` label.** The gene annotation is intentionally method-agnostic: it is a drop-in for array and RNA-seq alike, and a probe is treated as a measure *of a gene* without regard to the probe's own extent. A consequence is that the `GENEBODY` label requires an annotated span exceeding the promoter window, so on array cohorts it is largely unreachable and the pairs an RNA-seq gene model would place in the gene body fall to the neighbouring labels; region composition is therefore not directly comparable between array and RNA-seq cohorts. This **cannot affect any calibration verdict** — the receiving regions are themselves calibrated, and redistributing pairs among calibrated strata cannot create divergence. Exploring the choice of annotation source is left to individual users and is explicitly **not** a maintained concern of this method.
- **Open — `ks_trans_bulk_vs_uniform` is biased by construction.** The null-sanity arm compares the trans bulk analytic p against an untruncated uniform, but the bulk sample is truncated to `[bulk_lo, 1.0]`. A perfectly calibrated null simulated through the same code path yields `stat ≈ 0.0500, p = 0`, essentially identical to the observed value; rescaling the same sample to `[0, 1]` yields `p ≈ 0.56`. It is advisory and gates nothing, so no verdict is affected, and it is deliberately **not rendered** in the QC report. Fix: rescale to `(p − bulk_lo) / (1 − bulk_lo)` before the test.
- **Open — the TSS-distance QC panel mixes two anchors.** The region taxonomy bounds `DISTAL5`/`CIS5`/`PROMOTER` on the TSS (`gt_chromStart`) but `CIS3`/`DISTAL3` on the TES (`gt_chromEnd`), with `GENEBODY` straddling both. The QC panel plots every region against the TSS, so in that coordinate the three 3-prime bands translate by the gene span and overlap once pooled over the gene-length distribution, while the 5-prime bands are non-overlapping by construction. The `±200,000` axis limit is a display clip, not a data boundary: `DISTAL3` has no upper bound in the taxonomy and its data runs to the cis-window edge. Nothing is mis-assigned, and this is why the three 3-prime rows read `n/a (span-dependent)`. Deriving a second, TES-anchored distance for those rows would make `CIS3` and `DISTAL3` fully band-checkable and `GENEBODY` partially so — a real increase in QC coverage rather than presentation polish — and is the open option.
- **Open — `perm_mt_p` and `precise_mt_p` have never shared a row.** The mainline chain carries `precise_mt_p`; the permutation chain carries `perm_mt_p`; the two are separate files built from separate runs and no column crosses between them (§3.10). Both master builders subset to `{mt_id, gt_id, mt_t}`, and both the reservoir and the cis write-all map bypass the mainline stage that computes `precise_mt_p`, so it is absent from the permutation chain by construction. `p_permute` does not close this: it is a licensed copy of the analytic p, not the measured permutation p. Independent evaluation of the two quantities therefore requires bringing `perm_mt_p` onto the mainline catalog by a left join on `(mt_id, gt_id)` — additive, null where the permute run did not cover the pair, with coverage reported rather than assumed. `precise_mt_p` is **not** recomputed to achieve this: a second derivation of a quantity the pipeline already owns can diverge silently from the first.

- **Open — `seed` and `n_perm` collide across the two chains.** `tecpg/bootstrap.py` and `tecpg/permute.py` each stamp run-level columns literally named `seed` and `n_perm` on every row, and neither guards against the other. They are unambiguous today only because the two artifacts are never joined; any joined artifact (see above) makes them ambiguous. A rename or namespaced key is a prerequisite for the join, not a cleanup afterwards.

- **A single trans-global null; no cis-specific null exists.** Every reported pair — near-gene included — is scored against the chromosome-stratified trans null (§3.4). Phase 2 would have built a cis-specific null; the verdict is the finding that GTP does not need one. This means the permutation p offers nothing cis-specific to prefer over the analytic p for cis pairs, and should not be described as though it did.
- **Report the near-gene agreement as a bounded measurement, not an equivalence.** The near-gene strata are detectably offset from `TRANS` (§5, finding ii). The defensible sentence is "all seven regions calibrate to within 3.2e-05 in median log₁₀ p-ratio; the near-gene strata are detectably but negligibly offset, in the direction and rank order predicted by signal density" — not "the two agree". Correspondingly, λ remains **advisory and non-gating**: the near-gene λ gradient (`PROMOTER` 1.107 > `CIS5` 1.081 > `CIS3` 1.051 > distal ~1.01 ≈ `TRANS` 1.006) is close to a restatement of where the eQTMs are, and re-promoting it to a scientific finding is a recurring temptation to resist.
- **Extrapolation ceiling.** p-values are trustworthy to the resolution the test count demands (§3.6); beyond that they are reported as `< threshold`, since the tail model's assumptions become the dominant uncertainty.
- **Resolved — cis/trans mask reuse.** The region-mask logic is factored into a shared helper (`helper.py:compute_region_mask`) — a single source of truth for both the qr path and `permute.py`.
- **Resolved — null-pair stratification.** The accumulated null is stratified by chromosome to the trans stratum (§3.4); the earlier unmasked cross-product placeholder is gone.
- **Resolved — post-mapping architecture.** `qr_permute` consumes the mapping's master parquet and merges `perm_mt_p` back, mirroring `qr_bootstrap`, rather than recomputing a self-contained universe (§4). An **advisory** sampled consistency guard flags a supplied `M`/`G`/`C` that does not match the master's design (§3.8). The decision is settled and the code has completed the transition (§5).
- **Resolved — output-thresholding policy.** Write-all by default, with an optional p-cutoff and a recorded pre-threshold universe size (§3.9).
- **Resolved (GTP) — stratify-or-not.** The enrichment run settled it: `single_global_null_adequate`, `divergent_regions = []`, near-gene **bulk** 1,098,442 pairs against a floor of 100, largest near-gene `Δ vs TRANS` = 3.162e-05 against a tolerance of 0.5. A single global null carries for GTP and Phase 2 is unnecessary there. The reservoir-first run had already ruled out the *per-chromosome structural* branch (both distal strata calibrating to `TRANS`), leaving the near-gene window as the only open question; the enrichment run answered it. **The remaining work is per-dataset**: MESA and the oncology cohort each need their own reservoir + enrichment run and their own verdict, never inherited. Eval thresholds remain provisional; GTP shows the verdict invariant from 0.5 to 1e-4, first flipping at 3e-5, which argues for re-setting `TOLERANCE_MEDIAN_LOG10_RATIO_DIFF` to **0.05** — a value with an a-priori reading (10^0.05 = 1.12, i.e. "the two p-values differ by more than 12%"), 1,580× above the observed maximum, and not reverse-engineered from the result. Confirm against MESA before freezing.
- **Open — analytic+FDR for trans (cost-saving fork), now well-supported.** The bulk calibration (§5) is strong evidence the trans analytic p is well-calibrated, and the distal strata calibrating to `TRANS` extends that to the whole same-chromosome space, so the trans stratum could use the analytic p with BH-FDR (matching tensorQTL) instead of the expensive permutation-GPD layer. This is the trans arm of the Phase-4 `p_permute`/`fdr_permute` annotation (§5): where a stratum calibrates, `p_permute` is the validated analytic p and `fdr_permute` reuses `summarizeOutput_parquet.py` over it. Whether to adopt it for the production catalog is a decision to take per dataset once each near-gene verdict is in.
- **Open — a calibration-native cis-bulk contamination check.** With genomic inflation demoted from the stratify verdict (§5, 3a), the legitimate concern that broad, weak cis signal leaks into the bulk band and inflates the calibration-divergence `delta` is currently *stated but unhandled*. The principled replacement is a uniformity test on the cis-bulk analytic p (mirroring the existing trans-bulk test), **not** `lambda`; it is deferred until the 3a evidence shows it is actually needed.
- **Open — GPD threshold selection.** The tail threshold is provisionally the smallest retained exceedance (all of the top-K buffer); a higher threshold may be warranted, to be informed by the ξ-convergence diagnostic — which is itself sidecar-gated (§5, Phase 3b), so this remains open longer than originally planned.
- **Open — null-state persistence.** The accumulator and GPD fit are discarded at exit (§3.7), which is what gates the Phase-3b arms. Whether to persist them to a sidecar artifact is deliberately deferred until the 3a evidence shows the rigorous null-shape comparison is actually needed.
- **Open — CSV provenance and CLI threshold exposure.** The parquet-only metadata and the unexposed `output_p_threshold` flag (§3.9) are both small gaps to close before Phase 4 relies on them.
- **Open — downstream/FDR integration.** Two of the three Phase-4 chunks are merged (§5); the wrapper stage that invokes them is the remaining critical-path item. Until it lands, no catalog on disk carries `p_permute` or `fdr_permute`.

---

## 9. References

- Freedman, D. & Lane, D. (1983). *A nonstochastic interpretation of reported significance levels.* Journal of Business & Economic Statistics 1(4):292–298. — the residual-permutation scheme (§3.2).
- Ongen, H. et al. (2016). *Fast and efficient QTL mapper for thousands of molecular phenotypes* (FastQTL). Bioinformatics 32(10):1479–1485. — the Beta-approximation for cis (§3.4).
- Taylor-Weiner, A. et al. (2019). *Scaling computational genomics to millions of individuals with GPUs* (tensorQTL). Genome Biology 20:228. — GPU permutation QTL mapping and the per-gene Beta null.
- Generalized Pareto / extreme-value peaks-over-threshold tail modeling for permutation p-values (§3.5–3.6).
- Kober, K.M. et al. (2024). torch-eCpG. BMC Bioinformatics 25:71. — the base tecpg tool.
