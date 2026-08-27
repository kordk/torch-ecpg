#!/usr/bin/env python3
"""influence_diagnostic_panels.py — cross the influence and bootstrap screens.

Consumes the flagged catalog and the bootstrap-merged catalog and produces the
figures that show what the two screens do to the significant catalog:

  influence_joint_plane.png       leverage vs bootstrap fragility, per pair
  influence_volcano_flagged.png   where flagged pairs sit in the volcano
  influence_cumulative_flagged.png running fraction flagged down the p ranking
  influence_screen_agreement.png  overlap of the two screens on covered pairs

plus influence_panels.json with every number the figures display, so a report
can quote them without recomputing.

Usage:
  python3 tools/influence_diagnostic_panels.py --dataset gtp \
      --catalog output_gtp/summarized.influence.parquet \
      --boot    output_gtp/bootstrap_merged.parquet \
      --out-dir output_gtp/influence_panels
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

COL_FLAG = '#c2452d'
COL_OK = '#3b6ea5'
COL_MUTE = '#9aa5b1'
REGION_ORDER = ['TRANS', 'DISTAL5', 'DISTAL3', 'CIS5', 'CIS3', 'PROMOTER',
                'GENEBODY']


def read_cols(path, wanted):
    names = set(pq.ParquetFile(path).schema_arrow.names)
    df = pq.read_table(path, columns=[c for c in wanted if c in names]).to_pandas()
    if df.index.names != [None]:
        df = df.reset_index()
    return df


def save(fig, out_dir, stem, written):
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'{stem}.{ext}')
        fig.savefig(p, dpi=200, bbox_inches='tight')
        if ext == 'png':
            written.append(p)
    plt.close(fig)


def panel_joint_plane(df, out_dir, written, J, h_c_max, thr):
    """Leverage excess vs bootstrap fragility, one point per covered pair."""
    w = df.dropna(subset=['h_excess', 'sign_frac'])
    if w.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=200)
    flagged = w['mt_influence_flag'] == True                    # noqa: E712
    ax.scatter(w.loc[~flagged, 'h_excess'], w.loc[~flagged, 'sign_frac'],
               s=7, alpha=0.30, color=COL_OK, edgecolors='none',
               label='not flagged')
    ax.scatter(w.loc[flagged, 'h_excess'], w.loc[flagged, 'sign_frac'],
               s=9, alpha=0.45, color=COL_FLAG, edgecolors='none',
               label='leverage-flagged')
    ax.axhline(0.025, color=COL_MUTE, ls=':', lw=1)
    ymax = ax.get_ylim()[1]
    ax.text(ax.get_xlim()[0], 0.027, '  bootstrap CI spans zero above this line',
            fontsize=8, ha='left', va='bottom', color='#555')
    if thr is not None:
        ax.axvline(thr, color='k', ls='--', lw=1)
        ax.annotate('leverage threshold', xy=(thr, ymax * 0.97),
                    xytext=(6, 0), textcoords='offset points', fontsize=8,
                    va='top', ha='left', color='#333')
    ax.set_xscale('symlog', linthresh=1e-3)
    ax.set_xlabel('leverage above the covariate floor  (h_excess)')
    ax.set_ylabel('bootstrap sign instability\n(minority-sign fraction of resamples)')
    ax.set_title('Two independent screens, one point per bootstrapped '
                 'significant pair', fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, out_dir, 'influence_joint_plane', written)

    # quadrant counts (fragile = CI spans zero)
    frag = w['ci_cross'] == True                                # noqa: E712
    J['joint_plane'] = {
        'n_pairs': int(len(w)),
        'flagged_and_fragile': int((flagged & frag).sum()),
        'flagged_not_fragile': int((flagged & ~frag).sum()),
        'fragile_not_flagged': int((~flagged & frag).sum()),
        'neither': int((~flagged & ~frag).sum()),
        'h_c_max': h_c_max, 'threshold': thr,
    }


def panel_volcano(df, out_dir, written, J):
    """Where the flagged pairs sit in the significance landscape."""
    need = {'mt_est', 'mt_p', 'mt_influence_flag'}
    if not need <= set(df.columns):
        return
    w = df.dropna(subset=['mt_est', 'mt_p']).copy()
    if w.empty:
        return
    if len(w) > 400_000:
        w = w.sample(400_000, random_state=0)
    # mt_p is float32 in the catalog and underflows to exactly 0 for the
    # strongest pairs; floor at the smallest positive value present so those
    # pairs plot at the top of the axis instead of becoming inf.
    p = w['mt_p'].astype('float64').to_numpy()
    pos = p[p > 0]
    floor = pos.min() if pos.size else 1e-300
    y = -np.log10(np.maximum(p, floor))
    flagged = (w['mt_influence_flag'] == True).to_numpy()        # noqa: E712
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=200)
    ax.scatter(w['mt_est'][~flagged], y[~flagged], s=4, alpha=0.20,
               color=COL_MUTE, edgecolors='none', label='not flagged')
    ax.scatter(w['mt_est'][flagged], y[flagged], s=6, alpha=0.55,
               color=COL_FLAG, edgecolors='none', label='leverage-flagged')
    ax.set_xlabel('effect estimate (mt_est)')
    ax.set_ylabel('-log10 p')
    ax.set_title('Flagged pairs are not marginal: they sit at the top of the '
                 'volcano', fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, out_dir, 'influence_volcano_flagged', written)
    top = w.nsmallest(min(1000, len(w)), 'mt_p')
    J['volcano'] = {
        'n_plotted': int(len(w)),
        'frac_flagged_overall': float(flagged.mean()),
        'frac_flagged_top1000_by_p':
            float((top['mt_influence_flag'] == True).mean()),  # noqa: E712
    }


def panel_cumulative(df, out_dir, written, J):
    """Running fraction flagged as you walk down the p-value ranking."""
    if 'mt_p' not in df.columns or 'mt_influence_flag' not in df.columns:
        return
    w = df.dropna(subset=['mt_p']).sort_values('mt_p')
    if w.empty:
        return
    f = (w['mt_influence_flag'] == True).to_numpy().astype(float)  # noqa: E712
    run = np.cumsum(f) / np.arange(1, len(f) + 1)
    idx = np.unique(np.round(np.logspace(0, np.log10(len(f)), 400)).astype(int))
    idx = idx[idx >= 1] - 1
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=200)
    ax.plot(idx + 1, run[idx] * 100, color=COL_FLAG, lw=1.8)
    ax.axhline(f.mean() * 100, color=COL_MUTE, ls='--', lw=1,
               label=f'catalog-wide {f.mean() * 100:.1f}%')
    ax.set_xscale('log')
    ax.set_xlabel('rank by p-value (log scale)')
    ax.set_ylabel('% flagged among the top N')
    ax.set_title('Flagged pairs concentrate at the strongest end of the '
                 'catalog', fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, out_dir, 'influence_cumulative_flagged', written)
    marks = {}
    for n in (100, 1000, 10000, 100000):
        if n <= len(f):
            marks[f'top_{n}'] = float(run[n - 1])
    J['cumulative'] = {'catalog_wide': float(f.mean()), 'running': marks}


def panel_agreement(df, out_dir, written, J):
    """Do the two screens catch the same pairs?"""
    w = df.dropna(subset=['sign_frac'])
    if w.empty or 'mt_influence_flag' not in w.columns:
        return
    flagged = (w['mt_influence_flag'] == True).to_numpy()        # noqa: E712
    frag = (w['ci_cross'] == True).to_numpy()                   # noqa: E712
    both = int((flagged & frag).sum())
    lev_only = int((flagged & ~frag).sum())
    boot_only = int((~flagged & frag).sum())
    neither = int((~flagged & ~frag).sum())
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    labels = ['caught by both', 'leverage only', 'bootstrap only',
              'neither screen']
    vals = [both, lev_only, boot_only, neither]
    colors = ['#8c2f22', COL_FLAG, '#e0a800', COL_OK]
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
    total = sum(vals)
    for b, v in zip(bars, vals[::-1]):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                f'  {v:,} ({v / total * 100:.1f}%)', va='center', fontsize=9)
    ax.set_xlabel('bootstrapped significant pairs')
    ax.set_title('Screen agreement on the pairs where both can be evaluated',
                 fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(0, max(vals) * 1.28)
    save(fig, out_dir, 'influence_screen_agreement', written)
    J['agreement'] = {'both': both, 'leverage_only': lev_only,
                      'bootstrap_only': boot_only, 'neither': neither,
                      'n': total}


def main():
    ap = argparse.ArgumentParser(
        description='Figures crossing the influence and bootstrap screens.')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--catalog', required=True,
                    help='flagged catalog (summarized.influence.parquet)')
    ap.add_argument('--boot', help='bootstrap-merged catalog; omit if the '
                                   'bootstrap columns are on --catalog')
    ap.add_argument('--fdr-column', default='fdr_est')
    ap.add_argument('--fdr-threshold', type=float, default=0.05)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()

    if not HAVE_MPL:
        sys.exit('matplotlib is required for this tool.')
    os.makedirs(a.out_dir, exist_ok=True)

    key = ['mt_id', 'gt_id']
    df = read_cols(a.catalog, key + ['region', 'mt_est', 'mt_err', 'mt_t',
                                     'mt_p', a.fdr_column, 'mt_h_max',
                                     'mt_influence_flag', 'p_boot', 'ci_low',
                                     'ci_high', 'mt_est_boot_std'])
    for c in ('mt_id', 'mt_h_max'):
        if c not in df.columns:
            sys.exit(f'--catalog lacks required column {c}')
    if 'mt_influence_flag' not in df.columns:
        sys.exit('--catalog lacks mt_influence_flag; run '
                 'tools/flagInfluence_parquet.py with -o first.')
    for c in key:
        df[c] = df[c].astype(str)

    if a.boot:
        b = read_cols(a.boot, key + ['p_boot', 'ci_low', 'ci_high',
                                     'mt_est_boot_std'])
        for c in key:
            b[c] = b[c].astype(str)
        b = b.dropna(subset=['p_boot']).drop_duplicates(key)
        df = df.drop(columns=[c for c in ('p_boot', 'ci_low', 'ci_high',
                                          'mt_est_boot_std')
                              if c in df.columns], errors='ignore')
        df = df.merge(b, on=key, how='left', validate='many_to_one')

    h_c_max = None
    try:
        md = pq.ParquetFile(a.catalog).schema_arrow.metadata or {}
        h_c_max = float(md[b'tecpg_influence_h_c_max'].decode())
        thr = float(md[b'tecpg_influence_threshold'].decode())
    except Exception:                                           # noqa: BLE001
        thr = None
    df['h_excess'] = (df['mt_h_max'] - h_c_max) if h_c_max is not None \
        else df['mt_h_max']

    if a.fdr_column in df.columns:
        sig = df[df[a.fdr_column] <= a.fdr_threshold].copy()
    else:
        sig = df.copy()
    if 'p_boot' in sig.columns:
        sig['sign_frac'] = sig['p_boot'] / 2.0
        sig['ci_cross'] = (sig['ci_low'] <= 0) & (sig['ci_high'] >= 0)
    else:
        sig['sign_frac'] = np.nan
        sig['ci_cross'] = np.nan

    J = {'dataset': a.dataset, 'catalog': a.catalog, 'boot': a.boot,
         'n_rows': int(len(df)), 'n_significant': int(len(sig)),
         'fdr_column': a.fdr_column, 'fdr_threshold': a.fdr_threshold}
    written = []
    panel_joint_plane(sig, a.out_dir, written, J, h_c_max, thr)
    panel_volcano(sig, a.out_dir, written, J)
    panel_cumulative(sig, a.out_dir, written, J)
    panel_agreement(sig, a.out_dir, written, J)

    with open(os.path.join(a.out_dir, 'influence_panels.json'), 'w') as fh:
        json.dump(J, fh, indent=1, default=str)
    print(f'wrote {len(written)} figure(s) (png + pdf) to {a.out_dir}')
    for p in written:
        print(f'  {p}')
    print(f'  {os.path.join(a.out_dir, "influence_panels.json")}')


if __name__ == '__main__':
    main()
