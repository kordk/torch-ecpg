# `qr_permute` — Permutation-Null Significance for eQTM Mapping

> **Status: under active development.** The **Phase-1 trans pipeline is complete** — `qr_permute` produces real per-pair p-values for the **trans** stratum (empirical p with a GPD tail extension), on a finalized, seeded, provenance-stamped output schema. **It is not yet ready for inference.** The Phase-3a diagnostics have now been run on real data (GTP subset, via the reservoir calibration master — §5): in the mostly-null bulk the permutation and analytic p-values agree to within ~1% and `lambda_trans ≈ 1.0`, so the **trans analytic null looks well-calibrated**, and the stratify verdict is `single_global_null_adequate` — but on a *uniform* sample whose cis stratum is a weak proxy for the cis window, so a **gene-anchored cis-window confirmation is still pending** (§8). The cis stratum still has no dedicated null, so **every reported pair is currently scored against the trans-global null**, and `perm_mt_p` is not wired into the downstream FDR pipeline (Phase 4). Do **not** use `perm_mt_p` for inference until this notice is updated. See [§5 Implementation status](#5-implementation-status).

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

> **For now:** the trans machinery is built and producing real p-values, but the method is still being validated and the cis-specific yardstick is not in place yet (see the status note at the top). Treat current output as a work in progress, not as final significance.

---

## 1. Overview

`qr_permute` is a permutation-based significance method for expression–methylation (eQTM) association mapping. It is intended to become the **primary per-pair significance measure**, complementing — not replacing — the two existing measures:

- the **analytic** p-value (`mt_p`): a two-sided normal-approximation tail derived from the regression t-statistic; and
- the **bootstrap** diagnostic (`qr_bootstrap`, `p_boot`): a resampling-based stability measure computed on a user-supplied candidate pair set.

Unlike either, `qr_permute` estimates a **data-driven null** and can resolve very small p-values via parametric tail extrapolation. Like `qr_bootstrap`, it is a **post-mapping step**: the user runs the primary mapping first, and `qr_permute` consumes its output parquet (`--master-parquet`) — reading the observed statistic and the pair universe the mapping already produced, scoring that universe against the permutation null, and merging `perm_mt_p` back onto it. An optional `--pairs-file` narrows the scored set to a candidate subset; by default the universe is the entire master. It is implemented as an isolated parallel method alongside the existing solvers.

---

## 2. Motivation

`qr_permute` addresses three limitations of the existing significance measures:

1. **Analytic-p underflow.** `mt_p` is computed in `float32` as `erf(−|t|/√2) + 1`. Adding a value near `−1` to `1` is catastrophic cancellation, so the result underflows to exactly `0` past `|t| ≈ 6`. Below roughly `1e-7` the analytic p collapses to `0` rather than degrading gracefully, so it cannot resolve or rank the strongest associations. (A separate, cheap fix for the analytic path is the cancellation-free `erfc(|t|/√2)` reformulation; it is orthogonal to the permutation work and addresses *resolution*, not *calibration*.)
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

**Chromosome equality is not a naive comparison.** Assigning strata requires canonicalizing chromosome labels before testing equality: annotation columns arrive as bare integers, `chr`-prefixed strings, mixed case, or `float64` (pandas infers a float column whenever any value is missing, silently turning `1` into `1.0`). A raw `==` across two differently-typed annotation sources evaluates false for *every* pair, which would relabel the entire cis stratum as trans **with no error raised**. The canonical form strips a `chr` prefix and a trailing `.0`, uppercases, and treats a missing/unmappable chromosome as a **dropped locus** rather than guessing a stratum.

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

The output is the **master parquet with `perm_mt_p` merged on** — keyed on `(mt_id, gt_id)`, additive, and `mt_p` is never overwritten — mirroring how `qr_bootstrap` merges its columns. It therefore carries the master's columns (including the observed `mt_t`, **read not recomputed**) plus `perm_mt_p` and the run's `seed` and `n_perm`. Master rows that were not scored (e.g. outside a `--pairs-file` subset) carry `NaN perm_mt_p`. Provenance is additionally stamped into the **parquet** schema metadata as `tecpg_perm_seed`, `tecpg_perm_n_perm`, and `tecpg_perm_n_reported`.

`tecpg_perm_n_reported` records the **pre-threshold** universe size. This matters: scoring computes a p for every reported pair, but an optional `output_p_threshold` writes only pairs at or below a cutoff (genome scale cannot materialize `~2e10` rows). The default writes **all** reported pairs, preserving the full FDR universe on disk; when a threshold is used, `tecpg_perm_n_reported` is what keeps a downstream BH-FDR correction honest about the universe it was drawn from (Phase 4).

> **One current gap.** (i) The metadata keys are written on the **parquet** path only — a CSV run keeps the `seed` and `n_perm` *columns* but loses `n_reported`, so a thresholded CSV artifact cannot reconstruct its FDR universe. Prefer parquet for permutation output.

---

## 4. Architecture

`qr_permute` is registered as a `--mlr-method` value and implemented in its own module, `tecpg/permute.py`, mirroring `qr_bootstrap`. Key architectural choices:

- **Isolated parallel method.** During development it touches **no shared solver and no output-processing script**. Existing analyses (`legacy_normal_eq`, `qr`, `qr_bootstrap`, all `tools/`) run unaffected on the same checkout. Byte-for-byte identity of existing methods is the standing contract, enforced by the committed fingerprint (`tests/fingerprint_all_pipeline.json`), which is never modified to make a test pass.
- **Coverage vs stratification are independent axes.** Coverage — which pairs are reported — is defined by the **master parquet** (optionally narrowed by `--pairs-file`), not by an internally generated product. The *scoring* stratum is derived per pair from its own chromosomes (§3.4), so a pair receives the same p regardless of which master it came from, as long as the design matches. Region-mask logic lives in a single shared helper (`helper.py:compute_region_mask`), used for the null stratification.
- **Post-mapping data-flow.** `qr_permute` follows the **`qr_bootstrap` shape**: read the master parquet, compute a new per-pair quantity, merge it back on `(mt_id, gt_id)`. The observed statistic is not recomputed — it is read from the master. What is genome-wide is the *null estimation* (subsampled), not an observed scan; the reported universe is exactly the master's (optionally narrowed by `--pairs-file`).
- **Solve is null-only, with a consistency guard.** The regression kernel implemented within `permute.py` (as `bootstrap.py` does) is used to build the **null** and to verify the master. The observed statistic comes from the master; a **sampled equivalence guard** recomputes the observed t for a random subset of master pairs from the supplied `M`/`G`/`C` and **warns (advisory)** if it diverges from the stored `mt_t` — a self-diagnosing message distinguishing benign `float32` noise from a real design/`df` mismatch, without aborting the run (§3.8). Numerical equivalence is guaranteed by a three-way oracle: consume-path (read) == recompute-path == independent per-pair OLS (numpy).
- **Diagnostics are read-only and torch-free.** The evaluation script (§5, Phase 3) consumes permutation output only; it imports no torch and no `tecpg` runtime module, so it runs anywhere and is testable without a GPU environment.
- **Deferred downstream integration.** Wiring `perm_mt_p` into the downstream significance selectors and the BH-FDR universe is a deliberate final phase, undertaken only after the method is validated, so that in-progress output can never affect existing pipelines.

---

## 5. Implementation status

The build has **four phases**, delivered as a **walking skeleton**: the pipeline was first wired as trivial stubs (freezing interfaces and the output contract), then each stage replaced by real logic one **chunk** at a time, each guarded by an oracle and a **forced-fail proof** (green → red on a deliberately injected bug → green on revert).

- **Phase 1 — trans-global null** *(complete, chunks 0–9)*. Design-fixed Freedman–Lane; streaming fixed-memory null; empirical p + GPD tail; finalized output. All stage functions are real, and the null-side schema is settled (§3.9): `perm_mt_p`, `seed`, `n_perm`, parquet provenance metadata, and the optional output threshold. **Complete is not validated** — see Phase 3.
- **Realignment to the post-mapping-consumer architecture** *(complete)*. Phase 1 was first built as a **self-contained** method that generated its own universe and recomputed the observed statistic; it has been corrected to the `qr_bootstrap`-parallel form documented throughout: consume `--master-parquet` for the observed `mt_t` and the pair universe, the optional `--pairs-file` subset, the **universe ∩ normalized-M/G intersection** (§3.4), the **advisory** consistency guard (§3.8, §4), and the merge of `perm_mt_p` onto the master (§3.9). The null machinery, accumulator, GPD tail, and diagnostics are unchanged.
- **Phase 2 — cis Beta-approximation** *(pending)*. Per-gene min-p Beta null for the cis stratum (§3.4), fit in the t-domain, reusing Phase 1's residualize/refit primitives and the shared cis mask as each gene's local test set. Until it lands, cis pairs are scored against the trans-global null. **Gated on the Phase-3 stratify-or-not evidence** — the machinery is only built if the evidence says the cis null actually diverges from trans.
- **Phase 3 — evaluation / diagnostics** *(partially delivered)*. A standalone read-only script. Its arms divide by what the persisted output can support (§3.7):
  - **3a — output-derivable** *(built)*: calibration of `perm_mt_p` against a `float64` analytic reference recomputed from `mt_t`; null sanity via genomic inflation and a uniformity test on the trans (mostly-null) bulk; and the **stratify-or-not decision** in its *calibration-divergence* form — comparing how cis and trans each depart from the analytic reference in the bulk band. The recommendation keys on the **effect size** of that divergence (the median `log10(p_perm / p_ana)` difference between strata), with the two-sample test statistic reported but non-gating: at genome scale any such test's p-value collapses to zero from sample size alone and carries no information. Two outcomes: **stratification warranted** or **single global null adequate**. Genomic inflation (`lambda_trans`, `lambda_cis`, `lambda_excess`) is reported as a descriptive diagnostic but does **not** gate the decision — `lambda_GC` presumes a mostly-null test space, which holds for GWAS/eQTL but not eQTM, and least of all in cis where a large fraction of pairs carry real signal, so `lambda_cis > 1` is expected biology rather than miscalibration (§2). The legitimate concern that broad cis signal leaks into the bulk and masquerades as null divergence is left to a calibration-native detector — a uniformity test on the cis-bulk analytic p, mirroring the trans-bulk test — deferred rather than adjudicated by lambda (§8).
  - **3b — sidecar-gated** *(deferred)*: ξ / extrapolated-quantile convergence (§3.6), literal null flatness, and the rigorous *null-shape* stratify comparison. These need the null accumulator itself, which is not persisted, so they are stubbed behind an optional sidecar input whose contract is frozen in the consumer. Whether to add the corresponding writer to the core module is deliberately left to the 3a evidence.
  
  **3a has now been run on the GTP subset.** The calibration diagnostic needs a master that spans the full p-value range (a mostly-null *bulk*); a threshold-filtered master (e.g. `bootstrap_merged.parquet`, significant pairs only) yields an empty bulk, a `lambda` of ~27 that is a filter artifact rather than inflation, and a skipped verdict. The right calibration master is the mapping's **`sample_reservoir.csv`** — a uniform sample taken *before* p-value filtration (`--reservoir-count`), carrying `mt_t` over the full range — converted to parquet and consumed like any other master (`pipelinePermute.sh --reservoir` automates the CSV→parquet conversion and the two-step run; `tools/summarize_permute.py` renders the report into a text digest plus figures). On that master the bulk calibration is near-perfect (median `|log10(p_perm/p_ana)|` ≈ 0.003, `lambda_trans` ≈ 1.007, `lambda_cis` ≈ 1.016) and the verdict is `single_global_null_adequate` (Δ ≈ 2e-6, Mann-Whitney p ≈ 0.6, KS p ≈ 0.76). This validates the **trans** null's bulk calibration but not the cis window (§8, gene-anchored). The eval thresholds remain provisional placeholders to be re-set from full GTP/MESA.
- **Phase 4 — downstream / FDR integration** *(pending — deferred to last, gated on validation)*. The only phase that edits shared downstream code.

**Granular per-chunk status lives in the code, not this document.** Each stage function in `tecpg/permute.py` carries a `# CHUNK N` tag. This document tracks status at phase level to stay accurate through routine merges.

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

---

## 8. Limitations and open decisions

- **Partially validated (trans bulk).** On the GTP subset (reservoir calibration master, §5) the trans analytic null is well-calibrated in the bulk (permutation vs analytic agree to <1%, `lambda_trans` ≈ 1). Cis-window calibration and the full stratify-or-not decision are not yet settled (gene-anchored run pending, below), and `perm_mt_p` is not wired to FDR (Phase 4). Do not use `perm_mt_p` for inference yet.
- **Cis has no dedicated null yet.** Until Phase 2 lands, cis pairs are scored against the trans-global null (§3.4).
- **Extrapolation ceiling.** p-values are trustworthy to the resolution the test count demands (§3.6); beyond that they are reported as `< threshold`, since the tail model's assumptions become the dominant uncertainty.
- **Resolved — cis/trans mask reuse.** The region-mask logic is factored into a shared helper (`helper.py:compute_region_mask`) — a single source of truth for both the qr path and `permute.py`.
- **Resolved — null-pair stratification.** The accumulated null is stratified by chromosome to the trans stratum (§3.4); the earlier unmasked cross-product placeholder is gone.
- **Resolved — post-mapping architecture.** `qr_permute` consumes the mapping's master parquet and merges `perm_mt_p` back, mirroring `qr_bootstrap`, rather than recomputing a self-contained universe (§4). An **advisory** sampled consistency guard flags a supplied `M`/`G`/`C` that does not match the master's design (§3.8). The decision is settled and the code has completed the transition (§5).
- **Resolved — output-thresholding policy.** Write-all by default, with an optional p-cutoff and a recorded pre-threshold universe size (§3.9).
- **Open — stratify-or-not, cis window.** The uniform-subsample (reservoir) run returned `single_global_null_adequate` with cis and trans indistinguishable in the bulk (§5) — but its cis stratum is same-chromosome pairs, ~98% distal, a weak proxy for the cis window, so 'adequate' is *suspicious not settled*. The deciding run is a **gene-anchored cis-window master** (each gene's window CpGs, not a uniform draw), then permute + eval: if that also comes back adequate, Phase 2's per-gene Beta is unnecessary and a single global null carries; if the cis-window stratum diverges, Phase 2 is warranted. The design still assumes two strata (§3.4); **coverage is the master's universe regardless of how this resolves.**
- **Open — analytic+FDR for trans (cost-saving fork).** The bulk calibration result (§5) is strong evidence the trans analytic p is well-calibrated, so the trans stratum could use the analytic p with BH-FDR (matching tensorQTL) instead of the expensive permutation-GPD layer, making the most expensive layer evidence-contingent. Whether to adopt this for the production catalog is a decision to take once the gene-anchored cis result is in.
- **Open — a calibration-native cis-bulk contamination check.** With genomic inflation demoted from the stratify verdict (§5, 3a), the legitimate concern that broad, weak cis signal leaks into the bulk band and inflates the calibration-divergence `delta` is currently *stated but unhandled*. The principled replacement is a uniformity test on the cis-bulk analytic p (mirroring the existing trans-bulk test), **not** `lambda`; it is deferred until the 3a evidence shows it is actually needed.
- **Open — GPD threshold selection.** The tail threshold is provisionally the smallest retained exceedance (all of the top-K buffer); a higher threshold may be warranted, to be informed by the ξ-convergence diagnostic — which is itself sidecar-gated (§5, Phase 3b), so this remains open longer than originally planned.
- **Open — null-state persistence.** The accumulator and GPD fit are discarded at exit (§3.7), which is what gates the Phase-3b arms. Whether to persist them to a sidecar artifact is deliberately deferred until the 3a evidence shows the rigorous null-shape comparison is actually needed.
- **Open — CSV provenance and CLI threshold exposure.** The parquet-only metadata and the unexposed `output_p_threshold` flag (§3.9) are both small gaps to close before Phase 4 relies on them.
- **Open — downstream/FDR integration timing.** Deferred to a final phase (Phase 4), gated on validation.

---

## 9. References

- Freedman, D. & Lane, D. (1983). *A nonstochastic interpretation of reported significance levels.* Journal of Business & Economic Statistics 1(4):292–298. — the residual-permutation scheme (§3.2).
- Ongen, H. et al. (2016). *Fast and efficient QTL mapper for thousands of molecular phenotypes* (FastQTL). Bioinformatics 32(10):1479–1485. — the Beta-approximation for cis (§3.4).
- Taylor-Weiner, A. et al. (2019). *Scaling computational genomics to millions of individuals with GPUs* (tensorQTL). Genome Biology 20:228. — GPU permutation QTL mapping and the per-gene Beta null.
- Generalized Pareto / extreme-value peaks-over-threshold tail modeling for permutation p-values (§3.5–3.6).
- Kober, K.M. et al. (2024). torch-eCpG. BMC Bioinformatics 25:71. — the base tecpg tool.
