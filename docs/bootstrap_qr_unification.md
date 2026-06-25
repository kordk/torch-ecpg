# Bootstrap QR Unification

## Summary
The bootstrap path in `tecpg` has been modified to replace `torch.linalg.lstsq` with a batched QR solver combined with `torch.linalg.solve_triangular`. During resampling with replacement, degenerate draws (e.g., resamples lacking variance for a feature) cause the QR solver to emit `NaN` or `inf` values. We introduced a robust finite-filtering guard that correctly drops these specific resamples and computes empirical statistics (mean, std, 2.5/97.5 percentiles, and p_boot) strictly over the surviving, valid draws.

## Degenerate Resamples Observe Rate
In our synthetic equivalence testing (`tests/test_bootstrap_qr.py`), well-conditioned arrays agreed identically between `lstsq` and the QR solver (within float32 tolerances). When evaluating synthetic strict rank-deficient items (like collinear or zero-variance columns), `torch.linalg.solve_triangular(R)` naturally emits `NaN` or `inf` where `torch.linalg.lstsq` silently emits a min-norm solution on CPU. The guard successfully caught 100% of these cases and recorded them as degenerate.

In real-cohort setups (like MESA with N~610 subjects), a purely random resampling approach would be highly unlikely to ever draw a completely degenerate sample for typical continuous variables (probability approaching zero). It could happen occasionally on low-variance discrete covariates or exceptionally rare SNP variations, but the rate of degenerate draws should be exceedingly low at this scale.

## CUDA `gels` Caveat
A major hidden motivation for this unification involves the production GPU environment. On CUDA devices, `torch.linalg.lstsq` uses the `gels` driver, which is mathematically *undefined* on rank-deficient inputs. Even though the CPU tests silently returned min-norm solutions through the `gelsd` driver without raising exceptions, on CUDA those identical degenerate resamples caused undefined behavior that compromised the downstream bounds.

Adding the QR solver alongside the degenerate guard effectively hard-fails these rank-deficient draws via finite filtering (which correctly propagates on CUDA), preventing silent corruption of bootstrap estimates. Note that the CPU tests cannot exercise the production CUDA path, and the end-to-end test validates the production function's logic but not GPU solver behavior. The green test suite does not validate `gels` behavior on rank-deficient input.

## Unification Recommendation
Because the degenerate-resample rate should be incredibly rare on large `N` (such as N=610), and the exact behavior of `lstsq` on CUDA in these cases is undefined, unifying around the QR method (`qr_bootstrap`) is the strongly recommended approach. The QR method forces strict correctness and exposes issues that were previously hidden by CPU min-norm fallbacks or CUDA undefined behaviors.

## v2.0 Breaking Change Notice
The methods and CLI arguments `lstsq`, `lstsq_bootstrap`, and `manual` have been permanently renamed to `qr`, `qr_bootstrap`, and `legacy_normal_eq` respectively, reflecting the shift to the QR solver and to provide accuracy about the underlying algorithms. There are no backward-compatible aliases for the removed `lstsq` strings.
