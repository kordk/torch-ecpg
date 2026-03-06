# Review Report: MLR Methods 'manual' and 'lstsq' and Validation Tests

## 1. Scope

This report reviews the two multiple linear regression (MLR) implementations in the `tecpg` package and their associated validation tests:

- **'manual' method**: `regression_full()` in `tecpg/regression_full.py` — solves the normal equations via explicit matrix inverse.
- **'lstsq' method**: `tecpg_mlr_lstsq()` in `tecpg/processing.py` — solves the least-squares problem via `torch.linalg.lstsq` and QR decomposition.
- **Validation tests**: `tests/test_accuracy.py`, `tests/test_mlr_comparison.py`, and `tests/validation_utils.py`.

Both methods implement the regression model **G ~ Intercept + M + C₁ + C₂ + ...** for every pair of methylation site (M) and gene expression locus (G), and both produce four output statistics per coefficient: estimate (est), standard error (err), t-statistic (t), and p-value (p).

This report focuses on major analytical issues that could impact study findings and opportunities for expanding quality-control evaluations. Code efficiency and optimization are out of scope.

---

## 2. Implementation Review

### 2.1 Design Matrix Construction

Both methods construct the design matrix identically:

```
X = [ones | M | C]     shape: (batch, n_samples, n_coefficients)
```

where `ones` is the intercept column, `M` is the methylation values column, and `C` contains covariate columns. The column ordering is consistent between methods and with the output labeling convention (`const`, `mt`, covariate names).

**Assessment**: No issues identified. The design matrix construction is correct and consistent.

### 2.2 Degrees of Freedom

Both methods compute degrees of freedom as:

```python
nrows, ncols = C.shape[0], C.shape[1] + 1
df = nrows - ncols - 1
```

This yields `df = N - (n_covariates + 1) - 1 = N - n_covariates - 2`, which correctly accounts for `N` observations minus the total number of estimated parameters (`1 intercept + 1 methylation + n_covariates`).

**Assessment**: Correct in both methods.

### 2.3 Coefficient Estimation

**Manual method** (`regression_full.py`):
```python
XtXi = Xt.bmm(X).inverse()
XtXi_Xt = XtXi.bmm(Xt)
B = XtXi_Xt.matmul(Y)
```
Solves via the normal equations: β = (X'X)⁻¹X'Y.

**Lstsq method** (`processing.py`):
```python
lstsq_result = torch.linalg.lstsq(X, Y_expanded)
B = lstsq_result.solution
```
Solves via `torch.linalg.lstsq`, which internally uses a numerically stable decomposition.

**Assessment**: Both are mathematically correct formulations for the OLS estimator. The normal equations approach in the manual method is more susceptible to numerical issues with ill-conditioned matrices (see Section 3.2).

### 2.4 Standard Error Computation

**Manual method**:
```python
XtXi_diag_sqrt = torch.diagonal(XtXi, dim1=1, dim2=2).sqrt()
E = Y - X @ B
scalars = sqrt(sum(E²)) / sqrt(df)
S = XtXi_diag_sqrt * scalars
```

**Lstsq method**:
```python
Q, R = torch.linalg.qr(X, mode='reduced')
R_inv = torch.linalg.inv(R)
XtXi_diag_sqrt = (R_inv.pow(2).sum(dim=2)).sqrt()
E = Y - X @ B
Sigma = (RSS / df).sqrt()
S = XtXi_diag_sqrt * Sigma
```

Both compute `SE(β_j) = σ̂ × √((X'X)⁻¹_jj)`, where `σ̂ = √(RSS/df)`. The lstsq method obtains `diag((X'X)⁻¹)` through QR factorization: since X = QR, we have (X'X)⁻¹ = R⁻¹(R⁻¹)' and `diag((X'X)⁻¹)_j = Σ_k (R⁻¹_jk)²`.

**Assessment**: Both are mathematically equivalent and correctly implement the standard OLS standard error formula. The QR-based approach in the lstsq method is more numerically stable.

### 2.5 T-Statistic Computation

Both methods compute:
```python
T = B / S
```

**Assessment**: This correctly computes the t-statistic as `t = β̂ / SE(β̂)`. No issues.

### 2.6 P-Value Computation

Both methods use the same p-value function, `create_normal_p`:

```python
def create_normal_p(device, dtype):
    scalar = torch.tensor(2, device=device, dtype=dtype).sqrt().reciprocal().neg()
    def prob(value):
        return torch.erf(scalar * value.abs()) + 1
    return prob
```

This computes `P = 1 - erf(|t| / √2)`, which is the **two-tailed p-value under the standard normal distribution**.

The standard OLS p-value uses the Student's t distribution with `df` degrees of freedom. The normal distribution is used here as an approximation because, as documented in the `regression_full` docstring: *"Torch does not currently support the Student's T CDF function or any function that would help to implement it in python."*

**Assessment**: This is the most significant analytical concern (see Section 3.1 for detailed discussion).

---

## 3. Major Analytical Issues

### 3.1 P-Value Normal Approximation (Both Methods) — HIGH

**Issue**: Both methods compute p-values using the standard normal CDF instead of the Student's t CDF.

**Impact**: The Student's t distribution has heavier tails than the standard normal. For the same absolute t-statistic value, the Student's t two-tailed p-value is **larger** than the normal two-tailed p-value. Consequently, the normal approximation produces p-values that are **smaller (more significant)** than the true Student's t p-values. This means the tool is **anti-conservative**: it will produce more false positives than the nominal significance level implies.

**Magnitude**: The discrepancy depends on the degrees of freedom:
- At **df ≥ 100**, the approximation error is generally negligible for most practical purposes (relative error in p-values typically < 5% for moderate t-values).
- At **df = 30**, the error becomes noticeable, especially near significance thresholds (e.g., p ≈ 0.05).
- At **df < 20**, the error can be substantial and may meaningfully affect study conclusions.

**Example**: For df = 20 and t = 2.086 (critical value for two-tailed α = 0.05 under Student's t), the normal approximation yields p ≈ 0.037, a 26% underestimation relative to the true p = 0.05.

**Mitigation**: Studies with large sample sizes relative to the number of covariates (yielding df >> 30) are minimally affected. The impact is most pronounced in studies with small sample sizes or many covariates. Users should be aware that p-values from this tool will tend to be smaller than those from standard statistical software (e.g., R's `lm()`, Python's `statsmodels`).

**Note**: A Student's t PDF function (`create_studentt_p`) exists in `regression_full.py` but is commented out. This function computes the PDF, not the CDF, so it would not yield correct p-values if uncommented. The CDF is what is needed for p-values.

### 3.2 Numerical Stability with Ill-Conditioned Matrices (Manual Method) — MEDIUM

**Issue**: The manual method computes `(X'X).inverse()` directly using `torch.Tensor.inverse()`. For ill-conditioned design matrices (e.g., highly correlated covariates, near-constant methylation values, or near-collinear features), the matrix inverse can be numerically inaccurate or fail entirely.

**Impact**: Coefficients, standard errors, t-statistics, and p-values may be unreliable when the condition number of X'X is large. In extreme cases (singular or near-singular X'X), results may contain NaN or Inf values without any error or warning being raised.

**Comparison**: The lstsq method uses `torch.linalg.lstsq` for coefficients (which handles rank deficiency more gracefully) but still uses `torch.linalg.inv(R)` for the standard error computation. While the QR-based inversion is more numerically stable than direct (X'X) inversion, it can still fail for rank-deficient matrices.

**Mitigation**: Neither method checks the condition number or rank of the design matrix, and neither guards against or reports NaN/Inf in the output.

### 3.3 Float32 Precision (Both Methods) — LOW-MEDIUM

**Issue**: Both methods use `torch.float32` (single precision, ~7 decimal digits) as configured in `tecpg/config.py` (`DTYPE = torch.float32`). The independent validation reference (`statsmodels`) uses float64 (double precision, ~15 decimal digits).

**Impact**: The validation tests demonstrate that float32 produces results within `2e-4` for estimates and `1e-3` for t-statistics relative to float64 reference values. While this is acceptable for most applications, accumulated precision loss could matter for:
- Very small or very large coefficient values.
- P-values near significance thresholds, where the rounding error could shift results across the threshold.
- Downstream analyses that depend on precise p-values (e.g., FDR correction on a large set of tests).

**Note**: The use of float32 is likely a deliberate design choice to enable GPU computation with reduced memory usage.

### 3.4 Annotation Validation Bug in Manual Method — LOW

**Issue**: In `regression_full.py` line 149, the annotation check contains a duplicated condition:

```python
if region != 'all' and (G_annot is None or G_annot is None):
```

This checks `G_annot is None` twice instead of checking both `G_annot` and `M_annot`. The lstsq method (`processing.py` line 71) has the correct check:

```python
if region != 'all' and (G_annot is None or M_annot is None):
```

**Impact**: In the manual method, if `M_annot` is `None` while `G_annot` is provided, and a non-'all' region is specified, the validation passes incorrectly. This would lead to a runtime error later when `M_annot` is accessed. The practical impact is limited because the error would still surface (as a crash rather than a clear validation message), and the 'all' region (the most common use case) is unaffected.

### 3.5 Duplicate `create_normal_p` Definition — LOW

**Issue**: The `create_normal_p` function is defined independently in both `regression_full.py` and `processing.py` with identical implementations. This duplication introduces a risk of the implementations diverging if one is modified without updating the other.

**Impact**: Currently no issue, as both definitions are identical. However, a future change to the p-value computation in one method that is not propagated to the other would cause the two methods to produce different p-values silently.

---

## 4. Review of Validation Tests

### 4.1 `test_accuracy.py` — Validation Against Independent Reference

**What it does**: Generates synthetic data (100 samples, 500 M loci, 100 G loci), runs `regression_full` (manual method), selects 100 random pairs, and compares results against `statsmodels.OLS`.

**Strengths**:
- Tests against a well-established independent reference (`statsmodels`).
- Checks both beta-value and logit-transformed (M-value) inputs.
- Reports quantitative metrics (mean/max absolute differences).
- Generates scatter plots for visual inspection.

**Issues and Gaps**:

1. **Only validates the manual method**: The test calls `regression_full` but does not independently validate `tecpg_mlr_lstsq` against `statsmodels`. The lstsq method is only validated indirectly via `test_mlr_comparison.py` (comparing it to the manual method). An independent validation of the lstsq method against `statsmodels` would provide additional confidence.

2. **No fixed random seed**: The test uses `random.sample()` for selecting validation pairs and `generate_data()` for creating synthetic data, neither of which uses a fixed seed. This means test results are non-reproducible, making it difficult to debug intermittent failures.

3. **P-value tolerance is not enforced**: The test explicitly acknowledges that p-values will differ due to the normal approximation and only reports the average difference without enforcing any tolerance. While the difference is expected, having a bound on the acceptable p-value discrepancy (which depends on df) would document the expected behavior and catch unexpected regressions.

4. **Validation sample size is small**: Only 100 out of 50,000 total pairs are validated. While this is a practical trade-off, it provides limited coverage. Extreme or boundary conditions in the data may not be sampled.

5. **Only methylation coefficients are validated**: The test only compares `mt_est`, `mt_err`, `mt_t`, `mt_p`. Intercept and covariate coefficients are not validated against the reference, though the comparison test (`test_mlr_comparison.py`) does run with `methylation_only=False`.

### 4.2 `test_mlr_comparison.py` — Cross-Method Consistency

**What it does**: Generates synthetic data and runs both `regression_full` and `tecpg_mlr_lstsq` with identical inputs, comparing the outputs. Tests multiple scenarios: all-pairs, chunked, cis-region, trans-region, and both beta and logit-transformed inputs.

**Strengths**:
- Tests multiple region modes (all, cis, trans).
- Tests chunking behavior (meth_loci_per_chunk).
- Tests logit transformation.
- Uses `pd.testing.assert_frame_equal` with defined tolerances.
- Tests with `methylation_only=False`, validating all coefficients.
- Reports detailed comparison metrics (max/mean absolute and relative differences, correlation).

**Issues and Gaps**:

1. **Tolerances may mask real discrepancies**: The tolerance `rtol=1e-3, atol=1e-3` is relatively generous. While this is appropriate for comparing float32 implementations, it could mask a systematic bias if one method had a subtle formula error that produced results within this tolerance range. A test that checks for zero-mean differences (i.e., no systematic bias) would complement the tolerance check.

2. **No fixed random seed**: Same reproducibility concern as `test_accuracy.py`.

3. **Small test data size**: Uses 50 samples, 50 M loci, 50 G loci. This is computationally practical but does not test behavior at scales where numerical issues (accumulated float32 error, memory-related edge cases) might emerge.

4. **No edge case testing**: The test does not exercise conditions such as:
   - Collinear or near-collinear covariates.
   - Constant (zero-variance) methylation or gene expression values.
   - Very few samples relative to parameters (small df).
   - Missing or NaN values in the input.

5. **`distal` region not tested**: The test covers `all`, `cis`, and `trans` regions but omits `distal`.

### 4.3 `validation_utils.py` — Helper Functions

**What it does**: Provides `run_statsmodels_ols()` for independent OLS computation, `compare_results()` for computing differences, and `save_scatter_plot()` for visualization.

**Strengths**:
- `run_statsmodels_ols` correctly constructs the design matrix with the same column ordering as tecpg (const, mt, covariates).
- Includes index alignment validation.

**Issues and Gaps**:

1. **No validation of covariate coefficients in `run_statsmodels_ols` output**: The function only returns methylation-related results (`mt_est`, `mt_err`, `mt_t`, `mt_p`). Extending it to return all coefficients would enable more comprehensive validation.

2. **`compare_results` only computes absolute differences**: Relative differences are computed only for estimates (`rel_diff_est`). Adding relative differences for all metrics would provide a more complete picture.

---

## 5. Recommendations for Quality Control Expansion

### 5.1 Add Student's t P-Value Reference Comparison

**Recommendation**: Add a test that quantifies the p-value discrepancy between the normal approximation and the exact Student's t p-value as a function of degrees of freedom. This test should:
- Compute both normal and Student's t p-values for the same t-statistics.
- Verify that the discrepancy falls within theoretically expected bounds.
- Document the df threshold above which the approximation is acceptable for a given tolerance.

This would give users and reviewers confidence about when the approximation is and is not adequate for their study design.

### 5.2 Add Condition Number / Rank Checks

**Recommendation**: Add tests that exercise ill-conditioned inputs:
- Include covariates that are highly correlated or exactly collinear.
- Include constant-value methylation sites.
- Verify that the methods either produce correct results, raise informative errors, or flag unreliable outputs (e.g., NaN/Inf detection).

### 5.3 Add Edge Case Tests

**Recommendation**: Expand the test suite with:
- **Minimum df**: Test with the smallest valid number of samples (e.g., n_samples = n_parameters + 2, yielding df = 1). Verify behavior is correct or failure is graceful.
- **Known-result regression**: Use a hand-crafted dataset with analytically known regression coefficients and statistics to validate both methods against exact expected values (not dependent on another software package).
- **Boundary p-values**: Test with t-statistics chosen so that the p-value is very close to common thresholds (0.05, 0.01, etc.) and verify correct classification on both sides of the threshold.

### 5.4 Reproducible Tests via Fixed Seeds

**Recommendation**: Set fixed random seeds at the beginning of each test function:
```python
import random, numpy as np
random.seed(42)
np.random.seed(42)
```
This ensures reproducibility for debugging and makes test results deterministic.

### 5.5 Validate All Coefficients Against Independent Reference

**Recommendation**: Extend `test_accuracy.py` to validate intercept and covariate coefficients (not only methylation) against `statsmodels`. While the comparison test (`test_mlr_comparison.py`) uses `methylation_only=False`, having all coefficients independently validated against a reference would increase confidence that the full regression output is correct.

### 5.6 Add NaN/Inf Output Checks

**Recommendation**: Add assertions in the tests (and optionally in the implementations) that verify no NaN or Inf values appear in the output. For example:
```python
assert not res_df.isnull().any().any(), "Output contains NaN values"
assert np.isfinite(res_df.select_dtypes(include=[np.number]).values).all(), "Output contains Inf values"
```

### 5.7 Add `distal` Region Test

**Recommendation**: Add a test case in `test_mlr_comparison.py` for the `distal` region, which is currently the only region mode not covered by the comparison test.

### 5.8 Test the P-Value Filtration Boundary

**Recommendation**: Add a test that runs both methods with a `p_thresh` value and verifies:
- All returned results have p-values ≤ `p_thresh`.
- No results that should pass the filter are omitted.
- Both methods return the same set of results after filtration.

### 5.9 Add Systematic Bias Detection

**Recommendation**: In the comparison test, add a check that the mean difference between methods is approximately zero (e.g., via a one-sample t-test on the differences). This would detect systematic biases that fall within the absolute tolerance but are non-random.

### 5.10 Validate lstsq Method Independently Against External Reference

**Recommendation**: Add a test analogous to `test_accuracy.py` that validates `tecpg_mlr_lstsq` directly against `statsmodels`, rather than relying solely on the cross-method comparison. This would ensure that any bug shared between both implementations is still caught.

---

## 6. Summary

| Issue | Severity | Methods Affected | Impact on Study Findings |
|-------|----------|-----------------|-------------------------|
| Normal approximation for p-values | HIGH | Both | Anti-conservative p-values; more false positives, especially at low df |
| Numerical instability with ill-conditioned matrices | MEDIUM | Both (manual more so) | Potential for incorrect or NaN results without warning |
| Float32 precision | LOW-MEDIUM | Both | Rounding near significance thresholds; generally acceptable |
| Annotation validation bug (`G_annot` checked twice) | LOW | Manual only | Unclear error message when `M_annot` is missing; does not affect 'all' region |
| Duplicate `create_normal_p` definition | LOW | Both | Risk of silent divergence in future |

The coefficient estimation (est), standard error (err), and t-statistic (t) computations are analytically correct in both methods and have been validated to be consistent with each other and with `statsmodels` OLS within float32 tolerance. The primary analytical concern is the p-value computation, which uses a normal approximation rather than the exact Student's t distribution. For studies with adequate sample sizes (df >> 30), this approximation is acceptable. For studies with small sample sizes or many covariates, users should exercise caution with the reported p-values.

The existing validation tests provide a solid foundation, confirming cross-method consistency and approximate agreement with an independent reference. The main opportunities for improvement are: adding independent validation of the lstsq method, testing edge cases and ill-conditioned inputs, enforcing reproducible random seeds, and quantifying the p-value approximation error as a function of degrees of freedom.
