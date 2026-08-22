#!/usr/bin/env python3
"""Poster figures: influence diagnostic vs bootstrap fragility (two separate figures).

Reads calibration_bridge.json; writes into --out-dir:
  influence_dose_response.(png|pdf)   sign-instability rate by leverage bin
                                      (paired bars: all regions / TRANS)
  influence_se_ratio.(png|pdf)        median bootstrap/analytic SE ratio by bin

Usage:
  python3 fig_influence_dose_response.py \
      --json output_gtp/calibration_bridge/calibration_bridge.json \
      --out-dir output_gtp/calibration_bridge
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

H_C_MAX = 0.186711   # GTP covariate floor; threshold line = floor + DELTA
DELTA = 0.1

COL_ALL = '#3b6ea5'
COL_TR = '#c2452d'


def load(json_path):
    J = json.load(open(json_path))
    rows = J['breakdown_by_leverage']
    bins = [r['hbin'] for r in rows]
    n_all = [int(str(r['n']).replace(',', '')) for r in rows]
    ci_all = [float(r['ci_cross']) * 100 for r in rows]
    se_all = [float(r['med_se_ratio']) for r in rows]
    tr = {r['hbin']: float(r['ci_cross']) * 100
          for r in J.get('breakdown_by_leverage_trans', [])}
    ci_tr = [tr.get(b, np.nan) for b in bins]
    return bins, n_all, ci_all, ci_tr, se_all


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9)


def fig_dose_response(bins, n_all, ci_all, ci_tr, out_dir):
    x = np.arange(len(bins))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=200)
    ax.bar(x - w / 2, ci_all, w, label='All regions', color=COL_ALL)
    ax.bar(x + w / 2, ci_tr, w, label='TRANS only', color=COL_TR)
    for xi in range(len(bins)):
        top = np.nanmax([ci_all[xi], ci_tr[xi]])
        ax.text(xi, top + 2.5, f'n={n_all[xi]:,}', ha='center', va='bottom',
                fontsize=8, color='#444')
    ax.text(0, ci_all[0] + 11, f'{ci_all[0]:.1f}%', ha='center', fontsize=10,
            fontweight='bold', color=COL_ALL)
    thr = H_C_MAX + DELTA
    xpos = 0.5 + (thr - 0.25) / 0.25
    ax.axvline(xpos, color='k', ls='--', lw=1)
    ax.text(xpos - 0.09, 42, f'flag threshold (h > floor + {DELTA})',
            fontsize=8, rotation=90, ha='center', va='center', color='#333')
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_xlabel('per-CpG max sample leverage (mt_h_max)', fontsize=10)
    ax.set_ylabel('% of significant pairs with\nsign-instability under bootstrap',
                  fontsize=10)
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, loc='upper left', fontsize=9)
    style(ax)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'influence_dose_response.{ext}'),
                    bbox_inches='tight')
    plt.close(fig)


def fig_se_ratio(bins, n_all, se_all, out_dir):
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=200)
    ax.plot(x, se_all, 'o-', color=COL_ALL, ms=6, lw=1.8)
    for xi, v in enumerate(se_all):
        ax.annotate(f'{v:.2f}', (xi, v), xytext=(0, 7),
                    textcoords='offset points', ha='center', fontsize=9)
    ax.axhline(1.0, color='#888', lw=1, ls=':')
    ax.text(len(bins) - 0.55, 1.03, 'well-specified (ratio = 1)', fontsize=8,
            color='#666', va='bottom', ha='right')
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_xlabel('per-CpG max sample leverage (mt_h_max)', fontsize=10)
    ax.set_ylabel('median SE ratio\n(bootstrap / analytic)', fontsize=10)
    ax.set_ylim(0, max(se_all) * 1.25)
    style(ax)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'influence_se_ratio.{ext}'),
                    bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    bins, n_all, ci_all, ci_tr, se_all = load(a.json)
    fig_dose_response(bins, n_all, ci_all, ci_tr, a.out_dir)
    fig_se_ratio(bins, n_all, se_all, a.out_dir)
    print(f'wrote influence_dose_response.png/.pdf and '
          f'influence_se_ratio.png/.pdf -> {a.out_dir}')


if __name__ == '__main__':
    main()
