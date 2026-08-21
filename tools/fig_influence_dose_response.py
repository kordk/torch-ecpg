#!/usr/bin/env python3
"""Poster figure: bootstrap sign-instability vs per-CpG max leverage (GTP).

Reads calibration_bridge.json; writes influence_dose_response.(png|pdf).
Usage: python3 fig_influence_dose_response.py \
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

H_C_MAX = 0.186711
DELTA = 0.1

ap = argparse.ArgumentParser()
ap.add_argument('--json', required=True)
ap.add_argument('--out-dir', required=True)
a = ap.parse_args()
J = json.load(open(a.json))

bins = [r['hbin'] for r in J['breakdown_by_leverage']]
n_all = [int(r['n'].replace(',', '')) if isinstance(r['n'], str) else int(r['n'])
         for r in J['breakdown_by_leverage']]
ci_all = [float(r['ci_cross']) * 100 for r in J['breakdown_by_leverage']]
se_all = [float(r['med_se_ratio']) for r in J['breakdown_by_leverage']]
tr = {r['hbin']: float(r['ci_cross']) * 100
      for r in J.get('breakdown_by_leverage_trans', [])}
ci_tr = [tr.get(b, np.nan) for b in bins]

x = np.arange(len(bins))
w = 0.38
fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
b1 = ax.bar(x - w / 2, ci_all, w, label='All regions', color='#3b6ea5')
b2 = ax.bar(x + w / 2, ci_tr, w, label='TRANS only', color='#c2452d')
for xi, (v, n) in enumerate(zip(ci_all, n_all)):
    ax.text(xi - w / 2, v + 1.5, f'n={n:,}', ha='center', va='bottom',
            fontsize=7, rotation=0)
ax.annotate(f'{ci_all[0]:.1f}%', (x[0] - w / 2, ci_all[0]),
            xytext=(x[0] - w / 2, ci_all[0] + 9), ha='center', fontsize=9,
            fontweight='bold', arrowprops=dict(arrowstyle='-', lw=0.6))
ax.annotate(f'{ci_all[-1]:.1f}%', (x[-1] - w / 2, ci_all[-1] + 1),
            xytext=(x[-1] - 1.0, ci_all[-1] + 4), ha='center', fontsize=9,
            fontweight='bold', arrowprops=dict(arrowstyle='-', lw=0.6))
# flag threshold lands between bins 1 and 2 boundary region (floor+delta=0.287)
thr = H_C_MAX + DELTA
ax.axvline(0.5 + (thr - 0.25) / 0.25, color='k', ls='--', lw=1)
ax.text(0.5 + (thr - 0.25) / 0.25 + 0.05, 97,
        f'flag threshold\n(h > floor + {DELTA})', fontsize=7, va='top')
ax.set_xticks(x)
ax.set_xticklabels(bins)
ax.set_xlabel('per-CpG max sample leverage (mt_h_max)')
ax.set_ylabel('% of significant pairs with\nsign-instability under bootstrap')
ax.set_ylim(0, 104)
ax.legend(frameon=False, loc='upper left', fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
# se_ratio inset
ins = ax.inset_axes([0.60, 0.18, 0.36, 0.30])
ins.plot(x, se_all, 'o-', color='#555', ms=3, lw=1)
ins.axhline(1.0, color='#aaa', lw=0.7, ls=':')
ins.set_title('median SE ratio\n(bootstrap / analytic)', fontsize=6.5)
ins.set_xticks(x)
ins.set_xticklabels(bins, fontsize=5, rotation=45)
ins.tick_params(labelsize=6)
ins.spines[['top', 'right']].set_visible(False)
fig.suptitle('Leverage predicts fragility of eQTM associations (GTP, N=340)',
             fontsize=10)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(a.out_dir, f'influence_dose_response.{ext}'),
                bbox_inches='tight')
print('wrote influence_dose_response.png/.pdf ->', a.out_dir)
