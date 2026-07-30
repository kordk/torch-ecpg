# Integrated Gradients (`*_ig`) — What tecpg Actually Computes

> **Verified against:** `kordk/torch-ecpg` branch `dev`, HEAD
> `c94ef0f1c84b6fb8c2ac8072a427956c3c2ce87c`. All `path:line` citations below are
> reproducible at that commit.
>
> **Status:** Living document. Update whenever `tecpg/processing.py` changes the
> IG code paths, `LinearForwardWrapper.forward` stops being linear, or the
> `*_ig` output schema changes.
>
> **Scope:** The `--compute-ig` / `--compute-ig-deep` options of the `qr` MLR
> method, the `*_ig` columns they emit, and how those columns should — and
> should not — be described in reports and manuscripts.

---

## In plain language

`*_ig` is **not** a machine-learning result and **not** a second model. No
network is trained and no nonlinearity is introduced. IG in tecpg is the
regression coefficient re-expressed on a different scale: the size of `β`
multiplied by how far that predictor actually moves in your samples.

`β` answers *"how fast does expression move per unit of methylation?"*
`*_ig` answers *"how much expression does this CpG actually account for across
the methylation range present in this cohort?"*

---

## 1. The identity

Both IG paths compute the same quantity. For predictor column `k`:

```
IG_k  =  |β_k|  ×  mean_s |x_sk − baseline_k|
```

This is exact, not an approximation. Integrated Gradients attributes a model
output to its inputs by integrating the gradient along a path from a baseline
to the observed input. The model here is linear, so the gradient is the constant
`β` everywhere on that path and the integral collapses to the closed form above.

Consequences that follow directly:

- **`*_ig` is unsigned.** `.abs()` is applied at `processing.py:733` and
  `processing.py:1019`. Direction lives only in `mt_est`.
- **`*_ig` cannot rank differently from `|β|` within a single CpG**, because the
  multiplier depends only on the design matrix, never on expression.
- **`*_ig` carries no nonlinearity.** Thresholds, saturation and CpG×CpG
  interaction are invisible to it, because the fit has none.

---

## 2. Code map

| Step | Location |
|---|---|
| Baseline `X_baseline` selected (`mean` or `zero`) | `processing.py:507-510` |
| `X_diff_mean = (X - X_baseline).abs().mean(dim=1)` | `processing.py:513` |
| Analytical IG `= X_diff_mean * B[:, :, 1:].abs()` | `processing.py:733` |
| Covariate column selection (`ig_col_indices`) | `processing.py:736`, `1025` |
| Flatten to `(G·M, K)`, then region mask | `processing.py:793`, `808` |
| Concatenated into `current_results` | `processing.py:851` |
| p-threshold filter applied | `processing.py:963` |
| Deep path: `LinearForwardWrapper` (`x.matmul(w)`) | `processing.py:968-976` |
| Deep path: `ig.attribute(..., n_steps=50)` | `processing.py:1016` |
| Output column names (`*_ig`) | `processing.py:344-360` |
| Working precision `DTYPE = torch.float32` | `config.py:26` |

The `w` passed to `LinearForwardWrapper` is `B_full_filtered[i]`
(`processing.py:998`) — the OLS coefficient vector from the QR solve. It is not
learned; the `torch.nn.Module` is a container so Captum will accept the
function.

---

## 3. The two flags

They are **mutually exclusive** (`cli.py:1184`). They compute the same number;
they differ in coverage, cost and provenance.

| | `--compute-ig` (analytical) | `--compute-ig-deep` (Captum) |
|---|---|---|
| Coverage | Full (M × G) grid, before filtering | Only rows passing `--p-thresh` |
| `--p-thresh` | Optional | **Required** (`cli.py:1189`) |
| Reaches the reservoir | Yes (sampled pre-filtration, `:853`) | No |
| Execution | One vectorised op on the grid | Python loop, 50 fwd/bwd passes per hit |
| Peak memory | `X` freed after QR (`:538-539`) | `X` held across the whole gene loop |
| Numerics | Closed form | 50-step Riemann sum of a constant integrand |

**Agreement.** Over 25 random trials (S=340, K=8), worst relative divergence
between the two paths:

| precision | worst relative difference |
|---|---|
| `float32` (tecpg default) | 2.79e-07 |
| `float64` | 6.99e-10 |

i.e. they agree to float32 epsilon. Not bitwise identical, never scientifically
different.

**Which to use.** Use `--compute-ig`. The deep path buys nothing while
`LinearForwardWrapper.forward` is `x.matmul(w)`; it exists as scaffolding for a
future nonlinear forward function. Its only present-day use is as a CI
equivalence oracle — a weak one, since both paths consume the same `β` and the
same `X_baseline`, so it can only catch divergence in the attribution
arithmetic, never an error in either input.

---

## 4. Baseline choice (`--ig-baseline`, default `mean`)

This choice changes what the score *means*, not just its scale.

- **`mean`** (default) — `X_diff_mean` becomes the mean absolute deviation of
  the predictor, a **spread** measure (≈ 0.798·SD for a Gaussian predictor).
  IG then reads as *realized contribution*: slope × how much this CpG actually
  varies. This is the interpretation assumed everywhere below.
- **`zero`** — for methylation β-values, which are non-negative, `mean |x − 0|`
  is the mean methylation **level**, not its spread. IG then rewards
  uniformly-high-methylation CpGs regardless of whether they vary at all. Do not
  use `zero` for methylation predictors unless you intend exactly that.

---

## 5. Output schema

Columns are `mt_ig` plus one `<covariate>_ig` per covariate selected by
`--ig-covariates` (all) or `--ig-covariates-list` (`processing.py:344-360`).

**Both flags emit identical column names.** There is no `deep_` prefix, so a
catalog carries no record of which path produced its `*_ig` values — provenance
lives only in the run flags and logs. Record the flag in run metadata if the
distinction will ever matter downstream.

---

## 6. Interpretation

Because the covariates sit in the same design matrix, `mt_ig` and the
`<covariate>_ig` columns are on one common scale. That comparison is the main
thing IG adds over `mt_est`:

> Is this CpG accounting for more of the fitted expression than age, sex or
> cell composition already do?

A useful triage using `mt_est` and `mt_ig` together:

| | Reading |
|---|---|
| large `\|mt_est\|`, large `mt_ig` | Steep effect on a CpG that genuinely varies here. Highest priority. |
| large `\|mt_est\|`, small `mt_ig` | Potent per unit, near-invariant in these samples. Real but latent — do not discard; it may dominate in a cohort where the CpG does vary. |
| small `\|mt_est\|`, large `mt_ig` | Modest slope acting across a wide methylation range. Easily missed by an effect-size cutoff alone. |
| small `\|mt_est\|`, small `mt_ig` | Neither steep nor variable. Deprioritise. |

---

## 7. Limitations

- **Not causal.** IG ranks associations. A high `mt_ig` can belong to a
  passenger CpG tracking an unmeasured driver.
- **Not a confounding filter by construction.** IG attributes faithfully to
  whatever is in the design. It is informative *because* covariates share the
  scale, not because it removes confounding.
- **Cohort-dependent.** `X_diff_mean` is a property of your sample, so `*_ig`
  moves under resampling, subsetting or a change in cohort composition. `mt_est`
  does not (in expectation). Do not compare `*_ig` across datasets.
- **Not an arbiter of eQTM importance.** `β`, FDR and bootstrap reliability
  remain the arbiters. IG share is not an optimisation target.
- **Linear throughout.** See §1.

---

## 8. Reporting language

Accurate:

- "Variance-weighted effect size" / "realized contribution."
- "Gradient attribution computed on the fitted linear model
  (Integrated Gradients; Sundararajan et al., ICML 2017; Captum)."

Inaccurate, and likely to draw reviewer challenge:

- "AI-" or "machine-learning-based" scoring, "deep learning interpretability",
  or any phrasing implying a trained or nonlinear model.
- Any claim that IG detects nonlinearity, thresholds or interaction effects.
- Any directional claim from `*_ig` alone — it is unsigned.

---

## 9. Correctness contracts

1. `--compute-ig` and `--compute-ig-deep` must remain mutually exclusive while
   they share the `*_ig` column names (`cli.py:1184`).
2. `early_meth_slice` must stay disabled when either IG flag is set
   (`processing.py:714-718`); otherwise `B[:, :, 1:]` at `:733` indexes an
   already-truncated tensor.
3. Deep-path index arithmetic `m_idx = orig_flat_idx % mt_count`
   (`processing.py:1008`) is correct **only** while `:787` flattens as
   `permute(1, 0, 2).reshape(-1, K)` → `(G, M, K)`. Any change to that reshape
   invalidates it.
4. If `LinearForwardWrapper.forward` ever becomes nonlinear, §1 and the
   equivalence in §3 both cease to hold, and this document must be revised
   before the `*_ig` columns are interpreted again.
