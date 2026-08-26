#!/usr/bin/env python3
"""compare_perm_vs_analytic.py -- do the permutation p-values differ from the
analytic ones, and where?

    python3 compare_perm_vs_analytic.py output_gtp/permutation_results.parquet [df]

Bins pairs by |mt_t| and reports median log10(perm_p / analytic_p) in each bin,
so the answer is resolved as a function of depth rather than collapsed to one
number. Also reports the analytic p floor and how many pairs sit on it.

Positive log ratio  => permutation p is LARGER  => analytic p was anti-conservative.
Negative log ratio  => permutation p is SMALLER => analytic p was conservative.

Read-only.
"""
import sys

import numpy as np
import pandas as pd
import scipy.stats

path = sys.argv[1]
df_override = float(sys.argv[2]) if len(sys.argv) > 2 else None

d = pd.read_parquet(path)
print("rows: {:,}".format(len(d)))
print("columns: {}".format(sorted(d.columns.tolist())))

if 'perm_mt_p' not in d.columns:
    sys.exit("no perm_mt_p column -- is this the permute output?")

# Prefer the high-precision analytic p when present.
acol = next((c for c in ('precise_mt_p', 'mt_p') if c in d.columns), None)
if acol is None:
    sys.exit("no analytic p column (looked for precise_mt_p, mt_p)")
print("analytic p column: {}".format(acol))
if 'precise_mt_p' in d.columns and 'mt_p' in d.columns:
    same = np.isclose(d['precise_mt_p'], d['mt_p'], rtol=1e-6).mean()
    print("precise_mt_p == mt_p for {:.1%} of rows".format(same))

t = np.abs(d['mt_t'].to_numpy(dtype=np.float64))
pa = d[acol].to_numpy(dtype=np.float64)
pp = d['perm_mt_p'].to_numpy(dtype=np.float64)

# ---- analytic floor -------------------------------------------------------
print("\n=== ANALYTIC p FLOOR ===")
pos = pa[pa > 0]
mn = pos.min() if pos.size else 0.0
print("  min nonzero {:<12} : {:.6e}".format(acol, mn))
print("  pairs at that floor       : {:,}".format(int((pa == mn).sum())))
print("  distinct values < 1e-7    : {:,}".format(int(np.unique(pa[pa < 1e-7]).size)))
print("  pairs with p == 0         : {:,}".format(int((pa == 0).sum())))
print("  min perm_mt_p             : {:.6e}".format(float(pp[pp > 0].min()) if (pp > 0).any() else 0.0))
print("""
  If 'distinct values' is tiny while 'pairs at floor' is large, the analytic p
  has saturated and cannot rank those pairs. Those are exactly the pairs the
  permutation can newly discriminate.""")

# ---- ratio by |t| bin -----------------------------------------------------
ok = (pa > 0) & (pp > 0) & np.isfinite(t)
lr = np.log10(pp[ok] / pa[ok])
tt = t[ok]

edges = [0, 2, 3, 4, 4.5, 5, 6, 8, 10, 15, 20, 30, np.inf]
print("\n=== median log10(perm_p / analytic_p) BY |t| ===")
print("  {:>12} {:>12} {:>14} {:>14} {:>12}".format(
    "|t| bin", "n", "median log10", "ratio", "median perm_p"))
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (tt >= lo) & (tt < hi)
    n = int(m.sum())
    if n == 0:
        continue
    med = float(np.median(lr[m]))
    print("  {:>12} {:>12,} {:>14.4f} {:>14} {:>12.3e}".format(
        "[{:g},{:g})".format(lo, hi), n, med,
        "{:.3g}x".format(10 ** med), float(np.median(pp[ok][m]))))

print("""
  Read down the 'median log10' column. Near 0 in the low bins and growing
  positive in the high bins is the signature of an analytic null that is fine
  in the bulk and anti-conservative in the tail -- the permutation assigns
  LARGER p-values than the t-distribution does, because the permuted null
  produces large |t| more often than t theory predicts.""")

# ---- what it costs at a decision threshold --------------------------------
print("\n=== EFFECT AT COMMON THRESHOLDS ===")
for thr in (1e-5, 1e-6, 1e-8):
    na = int((pa < thr).sum())
    npm = int((pp < thr).sum())
    both = int(((pa < thr) & (pp < thr)).sum())
    print("  p < {:.0e} : analytic {:>9,}   permute {:>9,}   both {:>9,}   "
          "analytic-only {:>9,}".format(thr, na, npm, both, na - both))

if df_override:
    print("\n=== NULL SHAPE CHECK (df={:g}) ===".format(df_override))
    for q in (1e-4, 1e-5, 1e-6):
        t_exp = scipy.stats.t.isf(q / 2, df_override)
        print("  t-theory says |t| > {:.3f} occurs with prob {:.0e}".format(t_exp, q))
    print("  Compare against the sidecar's gpd_u and null max |t|. A null max")
    print("  far above what t theory allows at N draws is direct evidence the")
    print("  permutation null is heavier-tailed than the analytic assumption.")
