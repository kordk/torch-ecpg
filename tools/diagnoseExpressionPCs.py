#!/usr/bin/env python3
"""
Diagnose the dominant expression-PC concentration seen in residualize_pca.

Question: residualizing G on (Sex, age, cells) leaves ~90% of variance in the top
5 PCs for expression but only ~15% for methylation. Is that top axis a global
array-intensity (technical) effect that between-sample normalization would remove,
or structured (possibly biological) variance we should not regress out of mt_est?

This script re-derives the expression residual PCs and reports:
  1. Per-PC scree (is it PC1 alone, or spread across PC1-5?).
  2. Correlation of each top PC with per-sample mean expression (the global
     intensity / "brightness" proxy) -- the signature of an un-normalized array axis.
  3. Correlation of each top PC with the covariates (sanity: should be ~0 after
     residualization).
  4. PC1 loading sign uniformity (a near-uniform-sign PC1 is a global axis).
  5. A quantile-normalization counterfactual: re-residualize + re-PCA after
     between-sample QN and report the new top-5 share. If QN collapses the 90%,
     the dominant axis was an un-normalized intensity effect and the proper fix
     is normalizing expression upstream, not regressing out a PC.

Inputs are the same files residualize_pca consumed:
  --expression data_gtp/G.csv          (features x samples, log2 already applied)
  --covariates data_gtp/C_post_cellTypes.csv
Covariate columns default to the residualization formula
(Sex age B NK CD4T CD8T Mono), intersected with what is present.
"""
import argparse
import sys
import numpy as np
import pandas as pd

DEFAULT_COVARS = ["Sex", "age", "B", "NK", "CD4T", "CD8T", "Mono"]


def residualize(Y, X):
    """Y: samples x features; X: samples x p (incl. intercept). Return residuals."""
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return Y - X @ beta


def pca_scree(R, n=10):
    """R: samples x features. Center features, SVD, return (ratios[n], scores[:,:n], Vt[:n])."""
    Rc = R - R.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
    var = S ** 2
    ratio = var / var.sum()
    k = min(n, len(S))
    scores = U[:, :k] * S[:k]
    return ratio[:k], scores, Vt[:k]


def quantile_normalize(G):
    """Quantile-normalize a features x samples frame so every sample shares one
    distribution (classic Bolstad QN). Returns features x samples."""
    arr = G.to_numpy(dtype=float)
    # mean of sorted values across samples, per rank position
    sorted_cols = np.sort(arr, axis=0)
    rank_means = sorted_cols.mean(axis=1)
    out = np.empty_like(arr)
    for j in range(arr.shape[1]):
        # rank within sample (average ties), map to rank_means
        order = np.argsort(arr[:, j], kind="mergesort")
        ranks = np.empty(arr.shape[0], dtype=float)
        ranks[order] = np.arange(arr.shape[0])
        out[:, j] = rank_means[ranks.astype(int)]
    return pd.DataFrame(out, index=G.index, columns=G.columns)


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_block(G, C, covar_cols, n_pcs, label):
    # align samples
    samples = [s for s in G.columns if s in C.index]
    Gs = G[samples]
    Cs = C.loc[samples, covar_cols]
    Y = Gs.to_numpy(float).T                      # samples x features
    X = np.column_stack([np.ones(len(samples)), Cs.to_numpy(float)])
    R = residualize(Y, X)
    ratios, scores, Vt = pca_scree(R, n=n_pcs)

    print(f"\n=== {label} ===")
    print("Per-PC variance explained (of total residual variance):")
    cum = 0.0
    for i, r in enumerate(ratios):
        cum += r
        print(f"  PC{i+1}: {r*100:6.2f}%   cumulative {cum*100:6.2f}%")

    # global intensity proxy = per-sample mean expression (input scale)
    global_mean = Gs.mean(axis=0).to_numpy(float)   # per sample
    print("\nCorrelation of each PC with per-sample mean expression (intensity proxy):")
    for i in range(scores.shape[1]):
        print(f"  PC{i+1}: r = {corr(scores[:, i], global_mean):+.3f}")

    print("\nCorrelation of each PC with covariates (should be ~0 post-residualization):")
    for ci, c in enumerate(covar_cols):
        rs = [f"PC{i+1}:{corr(scores[:, i], Cs[c].to_numpy(float)):+.2f}"
              for i in range(min(3, scores.shape[1]))]
        print(f"  {c:>6}: " + "  ".join(rs))

    # PC1 loading sign uniformity
    l1 = Vt[0]
    frac_pos = float(np.mean(l1 > 0))
    print(f"\nPC1 loading sign uniformity: {max(frac_pos, 1-frac_pos)*100:.1f}% of genes "
          f"share one sign (near 100% => global all-genes-together axis).")
    return float(ratios[:5].sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expression", required=True, help="G.csv (features x samples, log2 applied)")
    ap.add_argument("--covariates", required=True, help="C_post_cellTypes.csv (samples x covariates)")
    ap.add_argument("--covar-cols", nargs="+", default=DEFAULT_COVARS,
                    help=f"Residualization covariates (default: {' '.join(DEFAULT_COVARS)})")
    ap.add_argument("--n-pcs", type=int, default=10, help="PCs to report (default 10)")
    ap.add_argument("--no-qn", action="store_true", help="Skip the quantile-normalization counterfactual")
    args = ap.parse_args()

    G = pd.read_csv(args.expression, index_col=0)
    C = pd.read_csv(args.covariates, index_col=0)
    C.index = C.index.astype(str)
    G.columns = G.columns.astype(str)

    covar_cols = [c for c in args.covar_cols if c in C.columns]
    missing = [c for c in args.covar_cols if c not in C.columns]
    if missing:
        print(f"WARNING: covariates not found and skipped: {missing}", file=sys.stderr)
    if not covar_cols:
        print("ERROR: none of the requested covariate columns are present.", file=sys.stderr)
        sys.exit(1)
    print(f"Expression: {G.shape} (features x samples); covariates used: {covar_cols}")

    top5_raw = run_block(G, C, covar_cols, args.n_pcs, "AS-IS (no between-sample normalization)")

    if not args.no_qn:
        Gqn = quantile_normalize(G)
        top5_qn = run_block(Gqn, C, covar_cols, args.n_pcs, "AFTER quantile normalization")
        print("\n--- Counterfactual verdict ---")
        print(f"Top-5 PC share: as-is {top5_raw*100:.1f}%  ->  quantile-normalized {top5_qn*100:.1f}%")
        drop = top5_raw - top5_qn
        if drop >= 0.25:
            print("QN collapses the dominant axis substantially. The 90% was largely an "
                  "un-normalized global-intensity (technical) effect; the proper fix is to "
                  "normalize expression upstream, then re-derive the PCs -- not to regress "
                  "out a 90% PC as a covariate.")
        else:
            print("QN does NOT collapse the dominant axis. The concentration is not just an "
                  "intensity artifact; inspect PC1 loadings/correlates before using these PCs "
                  "as covariates, as regressing them out may remove biological signal.")


if __name__ == "__main__":
    main()
