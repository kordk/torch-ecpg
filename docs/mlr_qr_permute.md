# `qr_permute` — Permutation-Null Significance for eQTM Mapping

> **Status: under active development.** The walking skeleton is merged; the observed-statistic stage is in progress. **`qr_permute` does not yet produce valid p-values** — the current build emits a placeholder `perm_mt_p = 0.5` for every pair. Do **not** use `perm_mt_p` for inference until this notice is removed. See [§5 Implementation status](#5-implementation-status).

**Applies to:** `tecpg run mlr --mlr-method qr_permute` (torch-eCpG v2, `dev`).
**Audience:** method reviewers, maintainers, and (eventually) users.
**Related:** analytic p (`mt_p`), bootstrap diagnostic (`qr_bootstrap` / `p_boot`).

---

## In plain language

**What it does.** eQTM mapping looks at pairs — a spot where DNA methylation is measured, and a gene whose activity is measured — and asks whether the two move together across people. Every pair gets a number for how strongly they appear linked. The hard part is separating a *real* link from one that could easily arise by chance, across an enormous number of pairs.

**The idea.** Instead of trusting a formula to judge chance, `qr_permute` measures chance directly. It repeatedly **shuffles** the data so that any true methylation–expression link is deliberately broken, and records how strong a link still appears by luck alone. Repeating this many times builds a picture of what pure noise looks like *for this dataset*. Each real pair is then compared against that picture: a link stronger than almost anything the shuffling produced is unlikely to be a fluke.

**Why this way.** It fixes two weaknesses of the usual formula-based p-value. The formula quietly rounds the smallest probabilities down to zero, so it can't tell the strongest associations apart; and it assumes the data behave in an idealized way that real biology often doesn't. Measuring chance by shuffling sidesteps both — it is built from the data's own noise, it produces a result for every pair, and it can still gauge how rare even the strongest signals are.

**What you get.** One p-value per pair (`perm_mt_p`): small means the association is unlikely to be chance, large means it's unremarkable. Because methylation near a gene ("cis") and methylation far away or on other chromosomes ("trans") behave very differently, the two are judged separately so each gets a fair yardstick.

> **For now:** this method is still being built and does **not** yet produce real p-values (see the status note at the top). This summary describes how it will work once complete.

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
- **Design-fixed ⇒ cheap.** Only the *response* is permuted; the design `[1, M, C]` is identical across permutations. The per-CpG factorization is therefore computed **once** and reused, so each permutation is a matvec against the cached factor rather than a re-factorization. Permuting a *predictor* instead would change the design every permutation and cost roughly `P×` more — that formulation must be avoided.

### 3.3 Null subsampling

The pooled-null assumption (§3.8) — that the per-pair null is shared — also licenses estimating the null from a **representative subsample** rather than from every pair:

- **Trans / global:** a uniform random sample of pairs (order `1e6`–`1e7`) suffices to characterize the single global null.
- **Cis:** per gene, the null population is that gene's cis-window CpGs (the full local test space), not a random draw.

The **observed** statistic is still computed for **every reported pair**; only the *null* is subsampled. Conflating the two — scoring against a null built from the same subsample that also defines the reported set — would silently forfeit the all-pairs property and is explicitly disallowed in the implementation.

### 3.4 Stratification — cis vs trans

The two strata have different test geometry and take different null constructions, mirroring the cis/trans test-space asymmetry:

- **Cis → per-gene Beta-approximation** (tensorQTL / FastQTL style): fit a Beta distribution to each gene's permutation min-p over its window, correcting local multiplicity and enabling smooth tail extrapolation. The Beta is fit in the **t-domain** (max `|t|` per permutation, converted at the end) so the `float32` analytic-p underflow cannot corrupt the significant tail.
- **Trans → pooled global null + generalized-Pareto (GPD) tail.** Trans has no window over which a per-gene min-p is defined, so a single global null is the natural object.

### 3.5 Scoring and tail extrapolation

The empirical two-sided p for an observed `|t|` is the fraction of null `|t|` at least as large, floored by the finite null size:

```
p_emp = max( #{ |t_null| ≥ |t_obs| } / N_null ,  1 / (N_null + 1) )
```

Empirical counting is reliable only to the extent of **effective** independent draws. Because probes are in LD and genes co-express, the effective number of independent draws is far smaller than the nominal count, and the permutation count is the binding constraint in the deep tail. Below the reliable range, a **parametric tail** — Beta on the p-values (cis) or a GPD fit to the exceedances (trans) — extends p past the empirical floor. The parametric tail is not a compromise for lack of compute; it is the **lower-variance** estimator in the extreme (a p estimated from `k` tail exceedances has relative standard error ≈ `1/√k`).

### 3.6 Resolution targets

Resolution is set by the multiple-testing denominator, not chosen for its own sake:

| Stratum | Correction | Smallest meaningful p |
|--------|-----------|-----------------------|
| Cis | per-gene window (~few hundred CpGs → ~`1e-4`), then FDR over ~28k genes | ~`2e-6` |
| Trans | Bonferroni over ~`4.5e5` CpGs × ~`4.7e4` probes ≈ `2e10` pairs | ~`2.4e-12` |

The **number of permutations** is governed by **tail-shape convergence** — whether the GPD shape parameter `ξ` and the extrapolated genome-wide quantile stabilize as permutations increase — **not** by the target p. Reaching `1e-12` by counting would require ~`1e12` tail draws and is the wrong estimator; the GPD tail is both necessary and more honest there. Any value reported beyond the correction threshold should be given as `< threshold`, not as a precise number, since past that point the extrapolation's assumptions (Pareto-like tail, exchangeability) are the limiting uncertainty rather than the arithmetic.

### 3.7 Null storage

At trans scale the null cannot be materialized as a list (`~2e10` pairs × permutations). It is accumulated as a **fixed-resolution t-histogram** plus a retained **tail-exceedance buffer** (streaming), from which the empirical body and the GPD tail are estimated. The observed/reported side is likewise emitted under an output-thresholding policy rather than writing the full cross-product.

### 3.8 Validity conditions

- **Exchangeability / constant df.** The pooled null requires `df = n − k` constant across all pairs and no per-pair missingness that varies `n`. In the current pipeline `df` is a single run-level scalar (`df = n_samples − covariates − 2`) and missingness is removed globally before the solve, so this holds. If per-pair `df` ever varied, both the pooled null and the subsampling shortcut would break.
- **Pivotal statistic.** Pooling requires a scale-free statistic; the t-statistic qualifies, the raw coefficient does not (§3.1).

---

## 4. Architecture

`qr_permute` is registered as a `--mlr-method` value and implemented in its own module, `tecpg/permute.py`, mirroring `qr_bootstrap`. Key architectural choices:

- **Isolated parallel method.** During development it touches **no shared solver and no output-processing script**. Existing analyses (`legacy_normal_eq`, `qr`, `qr_bootstrap`, all `tools/`) run unaffected on the same checkout. Byte-for-byte identity of existing methods is enforced by the committed fingerprint (`tests/fingerprint_all_pipeline.json`), which must remain unchanged.
- **Genome-wide data-flow.** Unlike `qr_bootstrap` (candidate `--pairs-file` + master-parquet merge), `qr_permute` scans genome-wide with subsampled null estimation. Its data-flow therefore resembles the `qr` full-scan path, not the bootstrap subset path.
- **Self-contained solve.** The regression kernel is implemented within `permute.py` (as `bootstrap.py` does), replicating the qr path's formula, with an independent per-pair OLS oracle guaranteeing numerical equivalence.
- **Deferred downstream integration.** Wiring `perm_mt_p` into the downstream significance selectors and the BH-FDR universe is a deliberate final phase, undertaken only after the method is validated, so that placeholder or in-progress output can never affect existing pipelines.

---

## 5. Implementation status

The method is built as a **walking skeleton**: the full pipeline is first wired as trivial stubs (freezing the interfaces and the output contract), then each stage is replaced by real logic one **chunk** at a time, each guarded by an oracle and a **forced-fail proof** (green → red on a deliberately injected bug → green on revert).

Current phase: **trans-global null (Phase 1).**

| Chunk | Stage | Status |
|------:|-------|--------|
| 0 | Method registration + placeholder output contract | ✅ merged |
| 1 | All stages stubbed, wired end-to-end (`perm_mt_p = 0.5`) | ✅ merged |
| 2 | Real observed statistic (pivotal t) | 🚧 in progress |
| 3 | Real cis/trans mask | ⏳ planned |
| 4 | Real null subsampling | ⏳ planned |
| 5 | Design-fixed Freedman–Lane (single permutation) | ⏳ planned |
| 6 | Iterate permutations + streaming accumulation | ⏳ planned |
| 7 | Real scoring (empirical p + floor) | ⏳ planned |
| 8 | GPD tail (trans) | ⏳ planned |
| 9 | Output finalization (`mt_t`, seed, `n_perm`, thresholding) | ⏳ planned |

**Subsequent phases:** cis Beta-approximation (§3.4); a standalone evaluation script (calibration vs the analytic p, `ξ`/quantile convergence, null sanity); adaptive permutation count; and — last, and deliberately deferred — downstream selector and FDR integration.

---

## 6. Correctness contracts

These are enforced for every chunk:

- **Existing-method byte-identity.** `legacy_normal_eq`, `qr`, `qr_bootstrap`, and all `tools/` behave identically; the fingerprint JSON is never modified to make a test pass.
- **Forced-fail proof.** Every guard and every real computation is verified by injecting a bug, observing the test fail, then reverting — a passing test that does not fail on injection is not accepted as verification.
- **Oracle / differential tests.** Correctness is checked against independent implementations (e.g. per-pair OLS via numpy `lstsq` for the observed t, mirroring `test_oracle_qr_regression_vs_plain_ols`), not against stored golden outputs.
- **Design-fixed Freedman–Lane.** The null permutes the reduced-model response residuals with the design held fixed (§3.2); permuting a predictor is a correctness *and* cost error.
- **Pivotal statistic.** The pooled/scored statistic is the t-stat, never raw β (§3.1).
- **Per-pair floor.** The empirical p is floored per pair at `1 / (N_null + 1)`, using that pair's finite null count.
- **Seed / determinism.** Permutation draws follow a recorded seed (default 42); the same seed reproduces the null, and the seed is written to output metadata.

---

## 7. Usage

> The method is not yet functional; the invocation below is documented for completeness and currently produces placeholder output.

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
- `--all` / `--cis` / `--distal` / `--trans` — stratum selection (the standard region flags).
- `--subsample-mt-count` / `--subsample-g-count` — random loci selection for null estimation (§3.3).
- `--seed` — permutation/subsample seed, recorded with the output.

**Output:** `permutation_results.{parquet,csv}`. The frozen contract is the columns `mt_id`, `gt_id`, `perm_mt_p`; the finalized schema (chunk 9) will additionally carry the observed t (`mt_t`), the seed, and `n_perm`. An explicit `--output-file` is honored verbatim; the default writes `permutation_results.<ext>` into the output directory.

---

## 8. Limitations and open decisions

- **Not yet functional.** Only the skeleton and (in progress) the observed statistic are implemented; `perm_mt_p` is placeholder until the pipeline is complete and validated.
- **Extrapolation ceiling.** p-values are trustworthy to the resolution the test count demands (§3.6); beyond that they are reported as `< threshold`, since the tail model's assumptions become the dominant uncertainty.
- **Open — cis/trans mask reuse.** Whether the region-mask construction is factored into a shared helper (single source of truth, small edit to the qr path) or duplicated in `permute.py` (maximum isolation, divergence risk) is decided at chunk 3.
- **Open — output-thresholding policy.** Which reported pairs are written (all vs a p-cutoff) at genome scale is decided at chunk 9; a computed p for every pair does not imply materializing `~2e10` rows.
- **Open — downstream/FDR integration timing.** Deferred to a final phase, gated on method validation.

---

## 9. References

- Freedman, D. & Lane, D. (1983). *A nonstochastic interpretation of reported significance levels.* Journal of Business & Economic Statistics 1(4):292–298. — the residual-permutation scheme (§3.2).
- Ongen, H. et al. (2016). *Fast and efficient QTL mapper for thousands of molecular phenotypes* (FastQTL). Bioinformatics 32(10):1479–1485. — the Beta-approximation for cis (§3.4).
- Taylor-Weiner, A. et al. (2019). *Scaling computational genomics to millions of individuals with GPUs* (tensorQTL). Genome Biology 20:228. — GPU permutation QTL mapping and the per-gene Beta null.
- Generalized Pareto / extreme-value tail modeling for permutation p-values (§3.5–3.6).
- Kober, K.M. et al. (2024). torch-eCpG. BMC Bioinformatics 25:71. — the base tecpg tool.

---

*This is a living document maintained alongside the implementation; update it as chunks land and as the open decisions above are resolved.*
