# `qr_permute` — Permutation-Null Significance for eQTM Mapping

> **Status: under active development.** The **Phase-1 trans pipeline is functionally complete** — `qr_permute` now produces real per-pair p-values for the **trans** stratum (empirical p with a GPD tail extension), replacing the earlier `perm_mt_p = 0.5` placeholder. **It is not yet ready for inference.** Its calibration has not been validated (the diagnostics of Phase 3 are pending); the cis stratum has no dedicated null yet, so **every reported pair is currently scored against the trans-global null**; the output schema is not finalized (chunk 9); and `perm_mt_p` is not wired into the downstream FDR pipeline (Phase 4). Do **not** use `perm_mt_p` for inference until validation is complete and this notice is updated. See [§5 Implementation status](#5-implementation-status).

**Applies to:** `tecpg run mlr --mlr-method qr_permute` (torch-eCpG v2, `dev`).
**Audience:** method reviewers, maintainers, and (eventually) users.
**Related:** analytic p (`mt_p`), bootstrap diagnostic (`qr_bootstrap` / `p_boot`).

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

Unlike either, `qr_permute` estimates a **data-driven null** covering **all pairs** and can resolve very small p-values via parametric tail extrapolation. It is a genome-wide method (no `--pairs-file`) and is implemented as an isolated parallel method alongside the existing solvers.

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

The **observed** statistic is still computed for **every reported pair**; only the *null* is subsampled. Conflating the two — scoring against a null built from the same subsample that also defines the reported set — would silently forfeit the all-pairs property and is explicitly disallowed in the implementation. A non-positive subsample count fails closed rather than yielding an empty null.

### 3.4 Stratification — cis vs trans

The two strata have different test geometry and are ultimately meant to take different null constructions, mirroring the cis/trans test-space asymmetry:

- **Trans → pooled global null + generalized-Pareto (GPD) tail** *(implemented)*. Trans has no window over which a per-gene min-p is defined, so a single global null is the natural object. The scored null population is stratified by **chromosome** — a null pair is *trans* iff its CpG and gene lie on different chromosomes — using the shared region-mask helper, independent of the user's coverage flag (§4).
- **Cis → per-gene Beta-approximation** *(planned — Phase 2, not yet implemented)*: fit a Beta distribution to each gene's permutation min-p over its window, correcting local multiplicity and enabling smooth tail extrapolation. The Beta is fit in the **t-domain** (max `|t|` per permutation, converted at the end) so the `float32` analytic-p underflow cannot corrupt the significant tail.

**Current behavior:** because the cis Beta is not yet built, **all reported pairs — cis and trans alike — are currently scored against the trans-global null.** Whether the cis stratum ultimately needs its own null (rather than sharing the global one) is the open *stratify-or-not* question, to be decided from the Phase-3 evaluation evidence (§5, §8).

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

### 3.7 Null storage

At trans scale the null cannot be materialized as a list (`~2e10` pairs × permutations). It is accumulated in **fixed memory**, independent of permutation count: a **fixed-resolution `|t|`-histogram** (with an overflow count for values beyond its range) plus a bounded **top-K tail-exceedance buffer** that retains the largest `|t|` seen exactly. The empirical body is read from the histogram; the GPD tail is fit to the exact exceedances in the buffer. Accumulation is streaming — each permutation folds into the same fixed structures, and the retained footprint does not grow with the number of permutations. The observed/reported side is likewise intended to be emitted under an output-thresholding policy rather than writing the full cross-product (chunk 9).

### 3.8 Validity conditions

- **Exchangeability / constant df.** The pooled null requires `df = n − k` constant across all pairs and no per-pair missingness that varies `n`. In the current pipeline `df` is a single run-level scalar (`df = n_samples − covariates − 2`) and missingness is removed globally before the solve, so this holds. If per-pair `df` ever varied, both the pooled null and the subsampling shortcut would break.
- **Pivotal statistic.** Pooling requires a scale-free statistic; the t-statistic qualifies, the raw coefficient does not (§3.1).
- **Annotations required.** Because the scored null is stratified by chromosome (§3.4), the method requires methylation and expression annotations. Running without them **fails closed** (raises) rather than silently scoring against an unstratified null.

---

## 4. Architecture

`qr_permute` is registered as a `--mlr-method` value and implemented in its own module, `tecpg/permute.py`, mirroring `qr_bootstrap`. Key architectural choices:

- **Isolated parallel method.** During development it touches **no shared solver and no output-processing script**. Existing analyses (`legacy_normal_eq`, `qr`, `qr_bootstrap`, all `tools/`) run unaffected on the same checkout. Byte-for-byte identity of existing methods is the standing contract, enforced by the committed fingerprint (`tests/fingerprint_all_pipeline.json`), which is never modified to make a test pass.
- **Coverage vs stratification are independent axes.** The user's region flag (`--all`/`--cis`/`--distal`/`--trans`) is a *coverage* filter selecting which pairs are reported; the *scoring* stratum is derived per pair from its own chromosomes (§3.4), so a pair receives the same p regardless of the coverage flag. Region-mask logic lives in a single shared helper (`helper.py:compute_region_mask`).
- **Genome-wide data-flow.** Unlike `qr_bootstrap` (candidate `--pairs-file` + master-parquet merge), `qr_permute` scans genome-wide with subsampled null estimation. Its data-flow therefore resembles the `qr` full-scan path, not the bootstrap subset path.
- **Self-contained solve.** The regression kernel is implemented within `permute.py` (as `bootstrap.py` does), replicating the qr path's formula, with an independent per-pair OLS oracle guaranteeing numerical equivalence.
- **Deferred downstream integration.** Wiring `perm_mt_p` into the downstream significance selectors and the BH-FDR universe is a deliberate final phase, undertaken only after the method is validated, so that in-progress output can never affect existing pipelines.

---

## 5. Implementation status

The build has **four phases**, delivered as a **walking skeleton**: the pipeline was first wired as trivial stubs (freezing interfaces and the output contract), then each stage replaced by real logic one **chunk** at a time, each guarded by an oracle and a **forced-fail proof** (green → red on a deliberately injected bug → green on revert).

- **Phase 1 — trans-global null** *(nearly complete)*. All-pairs coverage; design-fixed Freedman–Lane; streaming null; empirical p + GPD tail; finalized output. Chunks 0–9. **The four null-estimation stages are all implemented** — Freedman–Lane residualization, streaming histogram + tail-buffer accumulation, empirical scoring with the `1/(N+1)` floor, and the GPD tail. **Remaining: chunk 9 (output finalization)** — adding the observed t (`mt_t`), the seed, and `n_perm` to the output schema and settling the output-thresholding policy.
- **Phase 2 — cis Beta-approximation** *(pending)*. Per-gene min-p Beta null for the cis stratum (§3.4), fit in the t-domain, reusing Phase 1's residualize/refit primitives and the shared cis mask as each gene's local test set. Until it lands, cis pairs are scored against the trans-global null.
- **Phase 3 — evaluation / diagnostics script** *(pending — begin now)*. Standalone, read-only: calibration vs the analytic p, GPD ξ / quantile convergence, null sanity, and the cis-vs-trans comparison that **decides** whether stratification is warranted. Its first cut is due now — the trans tail is in place, and this script produces the evidence for the stratify-or-not decision and for validating calibration before the method is trusted.
- **Phase 4 — downstream / FDR integration** *(pending — deferred to last, gated on validation)*. The only phase that edits shared downstream code.

**Granular per-chunk status lives in the code, not this document.** Each stage function in `tecpg/permute.py` carries a `# CHUNK N` tag; the stage functions are now all real. This document tracks status at phase level to stay accurate through routine merges.

---

## 6. Correctness contracts

These are enforced for every chunk:

- **Existing-method byte-identity.** `legacy_normal_eq`, `qr`, `qr_bootstrap`, and all `tools/` behave identically; the fingerprint JSON is never modified to make a test pass.
- **Forced-fail proof.** Every guard and every real computation is verified by injecting a bug, observing the test fail, then reverting — a passing test that does not fail on injection is not accepted as verification.
- **Oracle / differential tests.** Correctness is checked against independent implementations (per-pair OLS via numpy `lstsq` for the observed t; a single-permutation numpy Freedman–Lane for the residualization; a direct null recount for the empirical p; GPD parameter recovery on simulated Pareto data for the tail) — not against stored golden outputs.
- **Design-fixed Freedman–Lane.** The null permutes the reduced-model response residuals with the design held fixed (§3.2); permuting a predictor is a correctness *and* cost error.
- **Pivotal statistic.** The pooled/scored statistic is the t-stat, never raw β (§3.1).
- **Bounded-memory accumulation.** The null footprint is fixed and independent of permutation count (§3.7).
- **Per-pair floor and positivity.** The empirical p is floored per pair at `1 / (N_null + 1)`; the GPD extension is clamped strictly positive.
- **Fail-closed / fail-safe guards.** Missing annotations and non-positive subsample counts fail closed (raise); a degenerate tail fit fails safe to the empirical p.
- **Seed / determinism.** Permutation draws follow a recorded seed (default 42); the same seed reproduces the null, and the seed will be written to output metadata (chunk 9).

---

## 7. Usage

> The trans pipeline produces real p-values, but the method is **not yet validated** (see the status note at the top). Current output should not be used for inference.

```
tecpg run mlr --mlr-method qr_permute \
    --permutations 1000 \
    --trans \
    --subsample-mt-count <N> --subsample-g-count <N> \
    --seed 42 \
    --output-format parquet
```

Relevant options:

- `--mlr-method qr_permute` — select the permutation method (qr-family; no `--pairs-file`).
- `--permutations` — number of permutations (default 100). Governed by tail-shape convergence, not the target p (§3.6).
- `--all` / `--cis` / `--distal` / `--trans` — coverage selection (the standard region flags); scoring stratum is derived per pair (§4).
- `--subsample-mt-count` / `--subsample-g-count` — random loci selection for null estimation (§3.3).
- `--seed` — permutation/subsample seed, recorded with the output.

**Annotations are required.** Because the null is chromosome-stratified (§3.4, §3.8), methylation and expression annotations must be supplied; running without them raises rather than silently producing an unstratified null.

**Output:** `permutation_results.{parquet,csv}`. The frozen contract is the columns `mt_id`, `gt_id`, `perm_mt_p`; the finalized schema (chunk 9) will additionally carry the observed t (`mt_t`), the seed, and `n_perm`. An explicit `--output-file` is honored verbatim; the default writes `permutation_results.<ext>` into the output directory.

---

## 8. Limitations and open decisions

- **Not yet validated.** The trans pipeline produces real p-values, but calibration has not been checked and the stratify-or-not decision has not been made (both await Phase 3). Do not use `perm_mt_p` for inference yet.
- **Cis has no dedicated null yet.** Until Phase 2 lands, cis pairs are scored against the trans-global null (§3.4).
- **Extrapolation ceiling.** p-values are trustworthy to the resolution the test count demands (§3.6); beyond that they are reported as `< threshold`, since the tail model's assumptions become the dominant uncertainty.
- **Resolved — cis/trans mask reuse.** The region-mask logic is factored into a shared helper (`helper.py:compute_region_mask`) — a single source of truth for both the qr path and `permute.py`.
- **Resolved — null-pair stratification.** The accumulated null is stratified by chromosome to the trans stratum (§3.4); the earlier unmasked cross-product placeholder is gone.
- **Open — stratify-or-not calibration.** Whether to fit separate cis/trans nulls or a single global null is an empirical question, decided from the evaluation script (Phase 3): does the cis null diverge enough from trans to warrant the per-gene Beta machinery? The design assumes two strata (statistically motivated, §3.4); **coverage is all-pairs regardless of how this resolves.**
- **Open — GPD threshold selection.** The tail threshold is provisionally the smallest retained exceedance (all of the top-K buffer); a higher threshold may be warranted, to be informed by the ξ-convergence diagnostic (Phase 3).
- **Open — output-thresholding policy.** Which reported pairs are written at genome scale (all vs a p-cutoff); a computed p for every pair does not imply materializing `~2e10` rows (chunk 9).
- **Open — downstream/FDR integration timing.** Deferred to a final phase (Phase 4), gated on validation.

---

## 9. References

- Freedman, D. & Lane, D. (1983). *A nonstochastic interpretation of reported significance levels.* Journal of Business & Economic Statistics 1(4):292–298. — the residual-permutation scheme (§3.2).
- Ongen, H. et al. (2016). *Fast and efficient QTL mapper for thousands of molecular phenotypes* (FastQTL). Bioinformatics 32(10):1479–1485. — the Beta-approximation for cis (§3.4).
- Taylor-Weiner, A. et al. (2019). *Scaling computational genomics to millions of individuals with GPUs* (tensorQTL). Genome Biology 20:228. — GPU permutation QTL mapping and the per-gene Beta null.
- Generalized Pareto / extreme-value peaks-over-threshold tail modeling for permutation p-values (§3.5–3.6).
- Kober, K.M. et al. (2024). torch-eCpG. BMC Bioinformatics 25:71. — the base tecpg tool.

---

*This is a living document maintained alongside the implementation; update it as phases progress and as the open decisions above are resolved. Granular per-chunk status is tracked in the code (`tecpg/permute.py` chunk comments), not here.*
