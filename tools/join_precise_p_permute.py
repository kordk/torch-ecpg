#!/usr/bin/env python3
"""join_precise_p_permute.py -- attach float64 precise_mt_p (and leverage) from the
mainline catalog onto a permute output, and report join coverage.

    python3 join_precise_p_permute.py \
        --permute   output_mesa/permutation_results.parquet \
        --catalog   output_mesa/summarized.parquet \
        --out       output_mesa/permutation_results.precise.parquet

The permute chain builds its master from a fresh cis map plus the reservoir,
both of which branch off upstream of pipeline.sh stage [6/9], so neither
carries precise_mt_p. The mainline catalog does. This joins them on
(mt_id, gt_id) so the permutation p-values can be compared against a float64
analytic comparator rather than a float32-truncated one.

Coverage is expected to be partial: reservoir-sourced trans/distal pairs need
not appear in the mainline catalog. The report below quantifies that, broken
down by region and by |t|, so the gap is visible rather than assumed.

Read-only with respect to its inputs.
"""
import argparse

import numpy as np
import pandas as pd

FLOAT32_MIN_P = 2.0 ** -24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--permute', required=True)
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--out', default=None,
                    help='write the joined frame here (optional)')
    ap.add_argument('--extra-cols', default='mt_h_max',
                    help='comma-separated additional catalog columns to carry '
                         'over when present (default: mt_h_max)')
    a = ap.parse_args()

    extra = [c.strip() for c in a.extra_cols.split(',') if c.strip()]

    perm = pd.read_parquet(a.permute)
    print("permute : {:,} rows".format(len(perm)))

    import pyarrow.parquet as pq
    want = ["mt_id", "gt_id", "precise_mt_p"]
    have = set(pq.ParquetFile(a.catalog).schema_arrow.names)
    if 'precise_mt_p' not in have:
        raise SystemExit("catalog has no precise_mt_p -- run pipeline.sh stage [6/9] first")
    want += [c for c in extra if c in have]
    cat = pd.read_parquet(a.catalog, columns=want)
    print("catalog : {:,} rows  (carrying {})".format(len(cat), want[2:]))

    m = perm.merge(cat, on=['mt_id', 'gt_id'], how='left')
    hit = m['precise_mt_p'].notna()

    print("\n=== JOIN COVERAGE ===")
    print("  matched: {:,} / {:,}  ({:.2%})".format(int(hit.sum()), len(m),
                                                    hit.mean()))

    if 'region' in m.columns:
        print("\n  by region:")
        g = m.groupby('region', dropna=False)['precise_mt_p']
        for r, sub in g:
            n = len(sub)
            k = int(sub.notna().sum())
            print("    {:<12} {:>12,} / {:>12,}   {:>7.2%}".format(
                str(r), k, n, k / n if n else 0.0))

    t = np.abs(m['mt_t'].to_numpy(np.float64))
    print("\n  by |t| (coverage where it matters):")
    for lo, hi in [(0, 4), (4, 6), (6, 10), (10, 20), (20, np.inf)]:
        sel = (t >= lo) & (t < hi)
        n = int(sel.sum())
        if not n:
            continue
        k = int(hit[sel].sum())
        print("    |t| [{:g},{:g}){:<6} {:>12,} / {:>12,}   {:>7.2%}".format(
            lo, hi, '', k, n, k / n))

    # What the float64 comparator recovers.
    if 'mt_p' in m.columns:
        z = (m['mt_p'].to_numpy(np.float64) == 0)
        zc = z & hit.to_numpy()
        print("\n=== WHAT float64 RECOVERS ===")
        print("  pairs with float32 mt_p == 0        : {:,}".format(int(z.sum())))
        print("  ... of those, joined to precise_mt_p: {:,}".format(int(zc.sum())))
        if zc.any():
            pv = m.loc[zc, 'precise_mt_p'].to_numpy(np.float64)
            nz = pv[pv > 0]
            print("  ... precise_mt_p still exactly 0    : {:,}".format(
                int((pv == 0).sum())))
            if nz.size:
                print("  ... min / median precise_mt_p       : {:.3e} / {:.3e}".format(
                    nz.min(), np.median(nz)))
                print("  ... spans {:.0f} orders of magnitude".format(
                    np.log10(nz.max()) - np.log10(nz.min())))

    if a.out:
        m.to_parquet(a.out, index=False)
        print("\nwrote {}".format(a.out))
        print("Pass this to plot_permute_diagnostics.py; it prefers precise_mt_p\n"
              "over mt_p automatically and will report which column it used.")


if __name__ == '__main__':
    main()
