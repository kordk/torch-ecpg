#!/usr/bin/env python3
"""read_permute_diagnostics.py -- GTP vs MESA, side by side.

    python3 read_permute_diagnostics.py output_gtp output_mesa

Reads the null sidecar and the eval report from each run and prints the four
things worth reading first: tail resolution, GPD threshold stability, per-region
calibration precision, and the stratum verdicts.

Read-only. Touches nothing.
"""
import json
import os
import sys

import numpy as np

dirs = sys.argv[1:] or ['output_gtp', 'output_mesa']


def load(d):
    out = {'name': os.path.basename(d.rstrip('/')), 'sidecar': None, 'report': None}
    sc = os.path.join(d, 'permutation_results.perm_null.npz')
    if os.path.exists(sc):
        try:
            out['sidecar'] = dict(np.load(sc, allow_pickle=False))
        except Exception as e:
            out['sidecar_err'] = str(e)
    rp = os.path.join(d, 'eval_permute_report.json')
    if os.path.exists(rp):
        try:
            out['report'] = json.load(open(rp))
        except Exception as e:
            out['report_err'] = str(e)
    return out


runs = [load(d) for d in dirs]
W = 22


def row(label, vals, fmt="{}"):
    cells = "".join(("{:>%d}" % W).format(fmt.format(v) if v is not None else "-")
                    for v in vals)
    print("  {:<26}{}".format(label, cells))


print("=" * (26 + W * len(runs) + 2))
print("  {:<26}{}".format("", "".join(("{:>%d}" % W).format(r['name']) for r in runs)))
print("=" * (26 + W * len(runs) + 2))

# ---------------------------------------------------------------- tail depth
print("\n--- TAIL RESOLUTION ---")
have = [r['sidecar'] for r in runs]
if not any(h is not None for h in have):
    print("  No sidecars found. Was the run made after the sidecar PR landed?")
else:
    def g(h, k, f=float):
        return None if h is None or k not in h else f(h[k])

    row("permutations", [g(h, 'perm_n_perm', int) for h in have], "{:,}")
    row("null draws (N)", [g(h, 'total_count', int) for h in have], "{:,}")
    row("empirical floor",
        [None if h is None else 1.0 / (int(h['total_count']) + 1) for h in have], "{:.3e}")
    row("null max |t|",
        [None if h is None else float(np.asarray(h['topk_values']).max()) for h in have], "{:.4f}")
    row("observed max |t|", [g(h, 'observed_max_abs_t') for h in have], "{:.4f}")
    row("EXTRAPOLATION GAP",
        [None if h is None else
         float(h['observed_max_abs_t']) - float(np.asarray(h['topk_values']).max())
         for h in have], "{:.4f}")
    row("overflow fraction",
        [None if h is None else int(h['overflow_count']) / int(h['total_count'])
         for h in have], "{:.3e}")
    row("gpd status", [None if h is None else str(h['gpd_status']) for h in have])
    row("gpd u (run)", [g(h, 'gpd_u') for h in have], "{:.4f}")
    row("gpd xi (run)", [g(h, 'gpd_xi') for h in have], "{:.5f}")
    print("""
  Read: a large extrapolation gap means the top hits' perm_mt_p is GPD
  extrapolation, not empirical resolution -- a disclosure item regardless of
  the p-value. overflow fraction well above ~0 means T_MAX=10.0 is clipping
  real null mass and the histogram's range is mis-set.""")

# ------------------------------------------------------------- xi convergence
print("\n--- GPD THRESHOLD STABILITY (xi sweep) ---")
for r in runs:
    s = (r['report'] or {}).get('arms', {}).get('sidecar', {})
    sweep = s.get('xi_sweep') or []
    print("\n  {}: status={}  xi_spread={}".format(
        r['name'], s.get('status', '-'),
        "-" if s.get('xi_spread') is None else "{:.5f}".format(s['xi_spread'])))
    if sweep:
        print("    {:>8} {:>12} {:>12} {:>12}".format("quantile", "u", "n_exc", "xi"))
        for w in sweep:
            print("    {:>8.2f} {:>12.4f} {:>12,} {:>12}".format(
                w['quantile'], w['u'], w['n_exceedances'],
                "None" if w['xi'] is None else "{:.5f}".format(w['xi'])))
print("""
  Read: xi stable across rungs => u = topk.min() was defensible. xi drifting
  with the threshold => the threshold is doing the work, not the data, and the
  provisional u should be revisited before any tail p-value is trusted.""")

# ------------------------------------------------------------- per-region
print("\n--- PER-REGION CALIBRATION ---")
for r in runs:
    st = (r['report'] or {}).get('arms', {}).get('stratify_decision', {})
    pr = st.get('per_region', {}) or {}
    print("\n  {}: verdict={}  divergent={}".format(
        r['name'], st.get('verdict', '-'), st.get('divergent_regions', '-')))
    if not pr:
        print("    no per_region block")
        continue
    print("    {:<10} {:>10} {:>10} {:>20} {:>9} {:>14}".format(
        "region", "n_bulk", "delta", "CI", "margin", "verdict"))
    for R, d in pr.items():
        if d.get('status') != 'ok':
            print("    {:<10} {:>10} {:>10} {:>20} {:>9} {:>14}".format(
                R, d.get('n_bulk', '-'), '-', '-', '-', d.get('status', '-')))
            continue
        lo, hi = d.get('delta_ci_lo'), d.get('delta_ci_hi')
        m = d.get('delta_ci_margin')
        print("    {:<10} {:>10,} {:>10.4f} {:>20} {:>9} {:>14}".format(
            R, d['n_bulk'], d['delta_vs_trans'],
            "-" if lo is None else "({:+.3f},{:+.3f})".format(lo, hi),
            "-" if m is None else "{:.4f}".format(m),
            d.get('delta_ci_verdict', '-')))
    if all('delta_ci_verdict' not in d for d in pr.values()):
        print("    [WARN] no delta CI fields -- was Task A merged before this run?")
print("""
  Read the MARGIN, not the label. With TOLERANCE=0.5 the equivalence band spans
  a 3.2x p-ratio, wide enough that a CI of (-0.49,-0.04) and one of
  (-0.02,+0.02) both read 'equivalent'. Margins near zero across near-gene
  strata would say the tolerance is too loose to bind on this data.""")

print("\n" + "=" * (26 + W * len(runs) + 2))
print("""  Cross-cohort: differences between the two columns are the first real
  evidence on whether tail behaviour and near-gene calibration are
  cohort-specific. Verdicts are per-dataset and per-annotation; neither
  licenses anything about the other.""")
