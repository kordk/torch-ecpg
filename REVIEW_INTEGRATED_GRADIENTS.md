# Review: Integrated Gradients (IG) Feature Implementation

**Branch:** `feat/integrated-gradients-lstsq-12123322807659440520`  
**Base:** `dev` (at commit `b4b06ef` — v1.3.0-dev)  
**Commits:** 3 (feat + 2 fixes)  
**Primary file changed:** `tecpg/processing.py` (+~130 lines of IG logic)

---

## Part 1: Implementation Overview (For Integration Planning Discussion)

### 1.1 What was implemented

The feature branch adds **Integrated Gradients (IG)** — a feature importance / attribution method — to the existing `tecpg_mlr_lstsq` regression pipeline. Two computation paths are provided:

| Path | CLI Flag | Library | Speed | Scope |
|------|----------|---------|-------|-------|
| **Analytical IG** | `--compute-ig` | None (pure PyTorch) | Fast (batch) | All results |
| **Deep IG** | `--compute-ig-deep` | Captum (`IntegratedGradients`) | Slow (per-hit loop) | Significant hits only |

Both paths produce a new `_ig` suffix column appended to the output alongside `_est`, `_err`, `_t`, and `_p`.

### 1.2 Parameters added

Three new parameters were introduced to `tecpg_mlr_lstsq()` and exposed via the CLI:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compute_ig` | `bool` | `False` | Enable analytical (fast) IG computation |
| `compute_ig_deep` | `bool` | `False` | Enable Captum-based (slow) IG computation |
| `ig_baseline` | `str` | `'mean'` | Baseline method: `'mean'` (per-CpG sample mean) or `'zero'` (zeros) |

These replaced the previously prototyped `subsample_mt_count`, `subsample_g_count`, `seed`, and `permute_label_test` parameters, which were removed in this branch.

### 1.3 Analytical IG path (`--compute-ig`)

The analytical path exploits the fact that for a **linear model** `y = Xw`, the Integrated Gradients attribution simplifies to a closed-form expression:

```
IG_i = (x_i - x_baseline_i) * w_i
```

The implementation computes this in batch across all methylation loci (M) and gene expression loci (G) in the current chunk:

```python
# X: (M, S, K), X_baseline: (M, 1, K), B (weights): (M, G, K)
diff_X = X - X_baseline                         # (M, S, K)
mean_abs_diff_X = diff_X.abs().mean(dim=1)       # (M, K)
IG = mean_abs_diff_X.unsqueeze(1) * B.abs()      # (M, G, K)
```

This is a **memory-efficient optimization**: instead of materializing the full `(M, S, G, K)` tensor, it separates the sample-dimension reduction from the gene-dimension broadcasting.

### 1.4 Deep IG path (`--compute-ig-deep`)

The deep path uses Captum's `IntegratedGradients` class with a lightweight `torch.nn.Module` wrapper:

```python
class LstsqWrapper(torch.nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.w = w
        self.b = b
    def forward(self, x):
        return x.matmul(self.w) + self.b
```

Key design decisions:
- **Only runs on statistically significant hits** (after p-value thresholding) to manage computational cost
- Uses `n_steps=50` interpolation steps (Captum default)
- Iterates over significant hits **one at a time** in a Python loop
- Maps flat indices back to `(gene_idx, methylation_idx)` pairs using `torch.meshgrid`
- Retrieves full weight vectors from `lstsq_result.solution` (not the potentially-sliced `methylation_only` weights)
- Computes mean absolute attribution across samples: `attributions.abs().mean(dim=0)`

### 1.5 Output format changes

When IG is enabled, the output columns gain an `_ig` suffix:
- With `methylation_only=True`: columns are `mt_est, mt_err, mt_t, mt_p, mt_ig`
- With `methylation_only=False`: columns include `const_ig, mt_ig, <covar>_ig, ...`

The `_ig` value represents the **mean absolute Integrated Gradient attribution** — a measure of feature importance for each predictor in the regression.

### 1.6 Where IG fits in the workflow

```
Data Loading → Chunking (Meth × Gene) → Design Matrix X Construction
    → lstsq Regression → B, S, T, P computation
        → [NEW] Analytical IG computation (if --compute-ig)
    → Flatten & Region Masking → Methylation-only slicing
        → P-value thresholding
            → [NEW] Deep IG computation (if --compute-ig-deep, post-threshold)
    → Results collection / Reservoir sampling → Output saving
```

### 1.7 Changes to other files

| File | Change |
|------|--------|
| `tecpg/cli.py` | Replaced subsampling/permutation CLI options with `--compute-ig`, `--compute-ig-deep`, `--ig-baseline`. IG flags are only passed to the `lstsq` method. Removed the MESA data command. |
| `tecpg/regression_full.py` | Simplified the degrees-of-freedom log message (cosmetic only). |
| `tecpg/regression_single.py` | Simplified the degrees-of-freedom log message (cosmetic only). |
| `tecpg/mesa.py` | **Deleted** (unrelated cleanup). |
| `tools/assignRegionToEcpg_parquet.py` | **Deleted** (unrelated cleanup). |
| `tools/recalculate_pvalues_parquet.py` | **Deleted** (unrelated cleanup). |
| `tools/mergeOutputs.py` | Simplified (unrelated cleanup). |

### 1.8 Dependencies

- **Analytical IG**: No new dependencies (uses existing PyTorch operations).
- **Deep IG**: Requires `captum` (Facebook's model interpretability library for PyTorch). The import is lazy — only attempted when `compute_ig_deep=True`, with a clear error message if not installed.

---

## Part 2: Analytic Design and Code Analysis

### 2.1 Analytical IG: Mathematical correctness

The analytical path computes:

```
IG(feature_i) = mean_over_samples(|x_i - baseline_i|) × |w_i|
```

**Observation:** Standard Integrated Gradients for a linear model `f(x) = w^T x + b` with baseline `x'` is:

```
IG_i(x) = (x_i - x'_i) × w_i
```

The implementation takes the **mean absolute value** across samples, which produces a per-CpG-locus aggregate importance score rather than a per-sample attribution. The mathematical reformulation `mean(|a * b|) = mean(|a|) * |b|` (used in the code) is valid **only because `w` is constant across samples** (it depends on the CpG–gene pair, not on individual samples). This is correct.

**Note on interpretation:** Taking absolute values means this produces an *unsigned importance magnitude*. Signed attributions (which could indicate direction of effect) are not preserved. This is a design choice, not a bug, but should be documented for users.

### 2.2 Deep IG: Correctness considerations

The deep path correctly:
- Wraps the linear model as a `torch.nn.Module` for Captum compatibility
- Uses the **full** weight vector from `lstsq_result.solution[m_idx, :, g_idx]` (not the methylation-only slice)
- Sets the intercept to 0 in `LstsqWrapper` (intercept does not affect gradients)

**Potential issue — Index mapping after combined region + p-value filtering:**

The code constructs a `full_mask` by combining `region_mask` and `p_indices`:

```python
full_mask = torch.ones(chunk_len * mt_count, dtype=torch.bool, device=device)
if region != 'all':
    full_mask = region_indices_list[-1]
if p_thresh is not None:
    temp = torch.zeros_like(full_mask)
    temp[full_mask] = p_indices
    full_mask = temp
```

This is subtle: `p_indices` is a boolean mask relative to the already-region-filtered subset, and the code correctly re-expands it into the full `(G*M)` flat space. This logic appears correct but is fragile — if the ordering of operations upstream changes, this mapping could silently produce wrong index lookups. Unit tests covering this interaction are recommended.

**Performance concern:** The deep path iterates in a Python `for` loop over each significant hit, instantiating a new `LstsqWrapper` and `IntegratedGradients` object per hit. For datasets with many significant hits, this would be very slow. The code includes a log warning about this.

### 2.3 Placement within the computation loop

```
Outer loop: methylation chunks (meth_chunk_index)
  Inner loop: gene chunks (gene_chunk_index)
    1. Construct design matrix X: (M, S, K)
    2. Solve lstsq → B (weights): (M, K, G)
    3. Compute S (standard errors), T, P
    ─── Analytical IG inserted here (pre-flatten) ───
    4. Permute/flatten → (G*M, K)
    5. Region mask filtering
    6. Methylation-only slicing
    7. Assemble current_results (B, S, T, P [, IG])
    8. Reservoir sampling
    9. P-value thresholding
    ─── Deep IG inserted here (post-threshold) ───
    10. Save results
```

**Key design observations:**

- **Analytical IG is computed at step 3**, operating on the full `(M, G, K)` tensors before any filtering. This is the correct placement for batch efficiency — it avoids re-indexing into filtered subsets and leverages the same GPU tensors already in memory.

- **Deep IG is computed at step 9**, after p-value filtering. This is intentional — it limits the expensive per-hit computation to only statistically significant results. However, it requires complex index remapping to trace back to the original `(m_idx, g_idx)` coordinates for accessing `X` and `lstsq_result.solution`.

- **X_baseline is computed once per methylation chunk** (at the chunk level, before the gene loop), which is correct since the baseline depends only on the design matrix `X` (which varies per methylation chunk, not per gene chunk).

### 2.4 Memory efficiency analysis

| Operation | Tensor Shape | Memory (M=1000, G=500, S=200, K=10) |
|-----------|-------------|--------------------------------------|
| `X` | `(M, S, K)` | 8 MB (float32) |
| `X_baseline` | `(M, 1, K)` | 40 KB |
| `diff_X` | `(M, S, K)` | 8 MB |
| `mean_abs_diff_X` | `(M, K)` | 40 KB |
| `IG` | `(M, G, K)` | 20 MB |
| **Avoided naive approach** | `(M, S, G, K)` | **4 GB** |

The analytical path's memory optimization (factoring out the sample dimension) is significant — it avoids an `O(M × S × G × K)` intermediate tensor and instead uses `O(M × S × K) + O(M × G × K)`.

**Concern with `diff_X`:** The `diff_X = X - X_baseline` tensor is `(M, S, K)` and lives alongside the existing `X` tensor. This temporarily doubles the memory for `X`. A potential optimization would be to compute `mean_abs_diff_X` in-place or to use `X.sub(X_baseline).abs_().mean(dim=1)` to avoid the intermediate.

### 2.5 GPU/CPU handling

The IG computation inherits the device placement from the existing pipeline (`X`, `B`, etc. are already on the configured device). No explicit device transfers are needed for the analytical path.

For the deep path, the `LstsqWrapper` model and Captum operations operate on tensors already on the GPU, which is correct.

### 2.6 Interaction with other features

| Feature | Interaction |
|---------|-------------|
| **Region filtering** (cis/trans/distal) | Analytical IG is computed before region filtering, then filtered along with B/S/T/P. Deep IG uses remapped indices. Both are correct. |
| **Methylation-only mode** | Analytical IG is sliced to column index 1 (methylation). Deep IG uses `mean_abs_attr[1:2]`. Both are correct. |
| **P-value thresholding** | Analytical IG is filtered along with other results. Deep IG runs only on filtered results. Both are correct. |
| **Reservoir sampling** | IG columns are included in `current_results` before reservoir sampling, so they are preserved. |
| **Gene/methylation chunking** | IG is computed within each chunk pair, consistent with the existing loop structure. |
| **`p_only` mode** | When `p_only=True`, IG is not included (only P values returned). This is correct — `p_only` is for quick screening. |
| **Logit transform** | IG baseline is computed after logit transformation of `X`, so attributions reflect the transformed feature space. This should be documented. |

### 2.7 Code quality observations

**Strengths:**
1. Clean separation of the two IG paths with minimal code duplication
2. Memory-efficient analytical path with clear mathematical documentation in comments
3. Lazy import of `captum` with an informative error message
4. Appropriate use of existing logging infrastructure
5. IG flags are only passed to the `lstsq` method (guarded in CLI)

**Areas for improvement:**

1. **Comment density in deep path:** The deep IG loop contains several "thinking out loud" comments (e.g., `"Wait, we permuted B earlier..."`, `"Actually, it's easier to..."`) that reflect prototype development and should be cleaned up for production.

2. **Hardcoded `n_steps=50`:** The number of interpolation steps for Captum's IG is hardcoded. Consider exposing this as a parameter for advanced users, or at minimum document the choice.

3. **⚠️ Bug: No validation of `ig_baseline` parameter:** The `ig_baseline` parameter accepts any string but only handles `'mean'` and `'zero'`. An invalid value silently skips baseline creation, and `X_baseline` will be undefined when accessed later, causing a **runtime `NameError`**. This is a critical issue — input validation must be added (e.g., raise `ValueError` for unsupported baselines, or use `click.Choice` in the CLI, which is already done at the CLI level but not in the `tecpg_mlr_lstsq` function itself).

4. **⚠️ Ambiguous flag interaction: `compute_ig` + `compute_ig_deep`:** Both flags can be set simultaneously, but the code guards with `compute_ig and not compute_ig_deep`, meaning the analytical path is silently skipped when `compute_ig_deep` is also set. This precedence rule is implicit and undocumented. **Recommendation:** Either (a) enforce mutual exclusivity at the CLI level with a `click.UsageError`, or (b) clearly document that `--compute-ig-deep` takes precedence, or (c) allow both to run and produce separate output columns (e.g., `_ig_analytical` and `_ig_deep`).

5. **Variable lifetime of `lstsq_result`:** The deep path accesses `lstsq_result.solution[m_idx, :, g_idx]` after the gene inner loop has progressed, which means `lstsq_result` from the *current* gene chunk is still in scope. This is correct but relies on Python's scoping rules — the `lstsq_result` variable is reassigned each gene chunk iteration but accessed within the same iteration.

6. **Prototype artifacts in the branch:** The branch also deletes `mesa.py`, `assignRegionToEcpg_parquet.py`, `recalculate_pvalues_parquet.py`, and simplifies `mergeOutputs.py`. These are unrelated cleanup changes that should be separated from the IG feature for a clean integration.

### 2.8 Recommendations for integration into `dev`

1. **Cherry-pick only the IG-related changes** from the feature branch, excluding the unrelated file deletions and log message simplifications.

2. **Add input validation** for `ig_baseline` and enforce mutual exclusivity (or define precedence) between `--compute-ig` and `--compute-ig-deep`.

3. **Add unit tests** covering:
   - Analytical IG output for a known small dataset (verify against manual calculation)
   - Deep IG output consistency with analytical IG for a linear model (they should agree)
   - Correct behavior with region filtering + p-value thresholding + IG
   - Edge cases: no significant hits (deep path should handle gracefully), single CpG, single gene

4. **Clean up the deep path comments** — remove prototype "thinking" comments; keep explanatory ones.

5. **Document the feature** in README.md with:
   - When to use analytical vs. deep IG
   - Interpretation of `_ig` column values
   - Performance expectations (analytical: negligible overhead; deep: ~seconds per hit)
   - Baseline choice guidance (mean vs. zero)

6. **Consider exposing `n_steps`** as an advanced parameter for the deep path.

7. **Consider the `diff_X` memory optimization** mentioned in §2.4 for large-scale runs.

8. **Performance profiling** of the deep path with realistic hit counts to establish guidelines for when it's practical.
