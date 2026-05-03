# LMM Evaluation and Implementation Strategy

## A. Verdict on the finding

The finding is **correct** and accurately identifies both the bottleneck of Mixed Models and the optimal path forward for GPU-accelerated workloads.

**Assessment:**
* Torch-eCpG achieves its high throughput by treating the regression problem as a massive batch of Ordinary Least Squares (OLS) equations. It computes cross-products (`Xt.bmm(X).inverse()`) or QR decompositions (`torch.linalg.qr`) that are highly optimized on the GPU and reused across chunks.
* A full Linear Mixed Model (LMM) natively requires an iterative solver (such as Restricted Maximum Likelihood, REML) to estimate variance components for *every single* CpG-gene pair. Doing this would destroy the batching advantage of the GPU and make the pipeline computationally intractable.
* The "Kennedy-style" / EMMAX approach (Estimating the variance components once under the null model, constructing the cohort-level covariance matrix $V$, and using its inverse square root $V^{-1/2}$ to whiten the data) converts the Generalized Least Squares (GLS) problem back into an OLS problem.
* **Current State:** The existing architecture is strictly OLS/MLR. However, because OLS on whitened data is mathematically equivalent to GLS, the architecture is **GLS-ready**. We do not need to rewrite the inner regression loops.

## B. Minimal-change implementation strategy

The most effective strategy is the **Whitened OLS** bridge. This preserves the existing batched regression engine entirely.

1. **Estimate Cohort Covariance (Once):**
   Using a Null Model (containing only covariates $C$ and random effects like a Kinship matrix), estimate the variance components (e.g., genetic variance vs. environmental variance) using PyTorch on the GPU. Construct the total phenotypic covariance matrix $V$.
2. **Compute the Whitening Transform (Once):**
   Compute the Cholesky decomposition or eigen-decomposition of $V^{-1}$ to obtain the whitening matrix $W = V^{-1/2}$. This matrix is of size $N \times N$ (where $N$ is the number of samples).
3. **Whiten the Input Data:**
   Transform the inputs by multiplying them by $W$:
   * $Y_{whitened} = W Y$ (Gene expression)
   * $X_{whitened} = W X$ (where $X$ contains covariates $C$ and the methylation locus $M$)
4. **Reuse Existing Machinery:**
   Pass the whitened $M$, $G$, and $C$ matrices directly into the existing `_tecpg_mlr_lstsq_inner` or `_regression_full_inner` loops. The existing QR solvers will transparently solve the GLS problem at OLS speeds.

## C. Repository touchpoints

By design, this strategy requires extremely few modifications to the core engine:

* **What can stay unchanged:**
  The core OLS algorithms (`tecpg_mlr_lstsq`, `regression_full`), chunking logic, data loading, GPU device management, output formatting, and parallel save executors.
* **Main modules to touch:**
  * `tecpg/cli.py`: Add CLI arguments for the LMM backend (e.g., `--kinship-matrix`, `--lmm`).
  * `tecpg/processing.py` and `tecpg/regression_full.py`: Add a small interception block at the beginning of the functions (before chunking begins) to apply the whitening transformation if LMM is enabled.
* **New abstractions:**
  * `tecpg/lmm.py`: A new module encapsulating the variance component estimation (REML) and generation of the whitening matrix $W$.

## D. Suggested API / backend design

**CLI API:**
Users opt-in to the LMM by providing a covariance/kinship matrix.
```bash
tecpg run mlr --all -g 100 -m 10000 \
    --kinship data/kinship.parquet \
    --mlr-method lstsq
```

**Python API:**
```python
def tecpg_mlr_lstsq(
    M: pandas.DataFrame, G: pandas.DataFrame, C: pandas.DataFrame,
    kinship: Optional[pandas.DataFrame] = None, # New Argument
    # ... existing args ...
):
    if kinship is not None:
        from tecpg.lmm import compute_whitening_matrix
        W = compute_whitening_matrix(C, kinship)

        # Apply transformation to dataframes/tensors before OLS
        M = apply_whitening(M, W)
        G = apply_whitening(G, W)
        C = apply_whitening(C, W)

    # Proceed with existing OLS implementation...
```

**Output Compatibility:**
Since the batched OLS loop is unchanged, it will emit coefficients, standard errors, T-statistics, and P-values in the exact same format. The coefficients from the whitened OLS are the unbiased GLS estimates. Feature importance (Integrated Gradients) can reuse the transformed linear model, meaning no downstream analysis code needs to change.

## E. Risks and mitigation

* **Numerical Stability:** Inverting the cohort covariance matrix $V$ can be unstable if $V$ is ill-conditioned (e.g., highly correlated subjects).
  * *Mitigation:* Use `torch.linalg.eigh` (eigen-decomposition) instead of Cholesky, and clamp small eigenvalues to a small positive threshold (jitter) before taking the inverse square root.
* **Memory/Performance Bottlenecks:** $W$ is $N \times N$. While negligible for small cohorts, memory scales quadratically. Multiplying $W$ across millions of loci could be slow.
  * *Mitigation:* $N$ is typically in the hundreds or thousands (e.g., $N=5,000 \rightarrow W$ is 100MB). This easily fits in VRAM. Matrix multiplication $W \times M$ can be done chunk-by-chunk on the GPU seamlessly since the existing code already iterates in chunks.
* **Feature Importance / Interpretability:** Whitening entangles sample dimensions. While IG scores for the *features* (CpGs, Covariates) are mathematically sound in the whitened space, they lose their strictly independent per-sample interpretation.
  * *Mitigation:* Document this explicitly. For global feature importance, the aggregated IG scores remain highly informative.
* **Full LMM vs Whitened Bridge:** A full LMM per pair is infeasible. The "Whitened OLS" bridge is the only viable way to retain GPU batching advantages.

## F. Recommended first milestone / MVP

1. **Milestone 1 (The exact GLS bridge):**
   Write a utility script or offline step that generates the whitening matrix $W$ for a cohort using standard tools (like GEMMA, PyTorch, or a custom R script).
2. **Milestone 2:**
   Add a `--whitening-matrix` flag to `tecpg run mlr`. Modify `tecpg/processing.py` to simply multiply $Y$, $M$, and $C$ by this matrix immediately after they are loaded onto the GPU, but before `Q, R = torch.linalg.qr(X)` is called.
3. **Milestone 3 (Validation):**
   Compare the p-values and coefficients of this simple whitened-OLS against an established, slow LMM solver (like `statsmodels.regression.mixed_linear_model` or EMMAX) for 100 CpG-Gene pairs. Once they match perfectly, we prove the architecture works.
4. **Milestone 4 (Native Estimation):**
   Implement the PyTorch-based Null Model REML solver natively in Torch-eCpG (`tecpg/lmm.py`) so users only have to pass a Kinship matrix.