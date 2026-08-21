#!/usr/bin/env python3
"""
calibration_bridge.py — cross mt_h_max (single-point leverage flag) against
bootstrap-derived fragility on the bootstrapped ("covered") pairs.

Purpose: convert "X% of significant TRANS rows are leverage-suspect" into a
measured collapse rate, to inform the influence-flag threshold choice.

Breakdown instruments (derived from stored bootstrap summaries; no resamples
are stored):
  sign_frac        = p_boot / 2      minority-sign fraction of resamples
                                     (exact above the 1/B floor; B=1000 -> floor 5e-4)
  ci_cross         = ci_low <= 0 <= ci_high   (>=2.5% minority mass; binary)
  break10/break25  = sign_frac >= 0.10 / 0.25 (graded collapse)
  se_ratio         = mt_est_boot_std / mt_err (continuous misspecification instrument)

Inputs: a master parquet carrying mt_h_max (I-0), and bootstrap columns either
on the same file or on a separate bootstrap-merged parquet joined on
(mt_id, gt_id). Report-only; writes nothing to any input.

Usage (klabdev, tecpg-dev env, repo root):
  python3 calibration_bridge.py \
      --master output_gtp/summarized.parquet \
      [--boot output_gtp/bootstrap_merged.parquet] \
      --covariates data_gtp/C.csv \
      --out-dir output_gtp/calibration_bridge
If the boot columns (p_boot, ci_low, ci_high, mt_est_boot_std) are already on
--master, omit --boot.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BOOT_COLS = ['p_boot', 'ci_low', 'ci_high', 'mt_est_boot_std',
             'degenerate_resamples']
CANDIDATE_ABS = [0.3, 0.5, 0.7, 0.9, 0.95]
CANDIDATE_FLOOR = [0.05, 0.10, 0.20, 0.30, 0.50]
REGION_ORDER = ['TRANS', 'DISTAL5', 'DISTAL3', 'CIS5', 'CIS3', 'PROMOTER',
                'GENEBODY']


def read_needed(path, want):
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    cols = [c for c in want if c in names]
    df = pq.read_table(path, columns=cols).to_pandas()
    if df.index.names != [None]:
        df = df.reset_index()
    return df


def h_c_max(cov_path):
    C = pd.read_csv(cov_path, index_col=0)
    Xc = np.hstack([np.ones((len(C), 1)), C.to_numpy(dtype=np.float64)])
    Q, _ = np.linalg.qr(Xc, mode='reduced')
    return float((Q * Q).sum(1).max())


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'null'
    return f'{x:.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--master', required=True)
    ap.add_argument('--boot', default=None,
                    help='separate bootstrap-merged parquet; omit if boot '
                         'columns are on --master')
    ap.add_argument('--covariates', default=None,
                    help='C.csv for h_C_max / floor-rule tables')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--fdr-column', default='fdr_est')
    ap.add_argument('--fdr-threshold', type=float, default=0.05)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    key = ['mt_id', 'gt_id']
    master_want = key + ['region', a.fdr_column, 'mt_h_max', 'mt_err',
                         'precise_mt_p', 'mt_t'] + BOOT_COLS
    df = read_needed(a.master, master_want)
    for c in key + ['mt_h_max']:
        if c not in df.columns:
            sys.exit(f'--master lacks required column {c}')
    df['mt_id'] = df['mt_id'].astype(str)
    df['gt_id'] = df['gt_id'].astype(str)

    if a.boot:
        b = read_needed(a.boot, key + BOOT_COLS + ['mt_err'])
        for c in key:
            b[c] = b[c].astype(str)
        have = [c for c in BOOT_COLS if c in b.columns]
        if not {'p_boot', 'ci_low', 'ci_high'} <= set(have):
            sys.exit(f'--boot lacks bootstrap columns; found {have}')
        # keep only pairs that actually carry a bootstrap result
        b = b.dropna(subset=['p_boot'])[key + have].drop_duplicates(key)
        df = df.drop(columns=[c for c in have if c in df.columns],
                     errors='ignore')
        df = df.merge(b, on=key, how='left', validate='many_to_one')
    if 'p_boot' not in df.columns:
        sys.exit('no bootstrap columns found on --master and no --boot given')

    hC = h_c_max(a.covariates) if a.covariates else None

    n_rows = len(df)
    df['covered'] = df['p_boot'].notna()
    sig = (df[a.fdr_column] <= a.fdr_threshold) if a.fdr_column in df.columns \
        else pd.Series(True, index=df.index)
    df['sig'] = sig.fillna(False)
    if 'region' not in df.columns:
        df['region'] = 'ALL'

    # ---- derived breakdown instruments on covered pairs -------------------
    cov = df[df['covered']].copy()
    cov['sign_frac'] = cov['p_boot'] / 2.0
    cov['ci_cross'] = (cov['ci_low'] <= 0) & (cov['ci_high'] >= 0)
    cov['break10'] = cov['sign_frac'] >= 0.10
    cov['break25'] = cov['sign_frac'] >= 0.25
    if 'mt_est_boot_std' in cov.columns and 'mt_err' in cov.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            cov['se_ratio'] = cov['mt_est_boot_std'] / cov['mt_err']
    else:
        cov['se_ratio'] = np.nan
    covsig = cov[cov['sig']].copy()

    rep, J = [], {}

    def R(line=''):
        rep.append(line)

    R('# Calibration bridge — mt_h_max vs bootstrap fragility')
    R()
    R(f'master: {a.master}  |  boot: {a.boot or "(columns on master)"}')
    R(f'rows: {n_rows:,}  |  covered (p_boot present): {int(df["covered"].sum()):,}'
      f'  |  significant ({a.fdr_column} <= {a.fdr_threshold}): {int(df["sig"].sum()):,}'
      f'  |  covered AND significant: {len(covsig):,}')
    R(f'h_C_max: {fmt(hC, 6) if hC is not None else "null (no --covariates)"}')
    B_impl = (1.0 / cov['p_boot'].min()) if len(cov) else float('nan')
    R(f'implied B from min p_boot: {B_impl:.0f}  (sign_frac floor {cov["p_boot"].min()/2:.2e})'
      if len(cov) else '')
    R()
    J['header'] = {'rows': n_rows, 'covered': int(df['covered'].sum()),
                   'significant': int(df['sig'].sum()),
                   'covered_significant': len(covsig), 'h_C_max': hC,
                   'fdr_column': a.fdr_column,
                   'fdr_threshold': a.fdr_threshold}

    # ---- 1. coverage vs leverage: is the bridge even observable? ----------
    R('## 1. Bootstrap coverage by leverage (significant pairs)')
    edges = [0, 0.25, 0.5, 0.7, 0.9, 1.01]
    labels = ['<0.25', '0.25-0.5', '0.5-0.7', '0.7-0.9', '>=0.9']
    sigall = df[df['sig']].copy()
    sigall['hbin'] = pd.cut(sigall['mt_h_max'], edges, labels=labels,
                            right=False)
    tab = sigall.groupby('hbin', observed=False).agg(
        n_sig=('covered', 'size'), n_cov=('covered', 'sum'))
    tab['coverage'] = tab['n_cov'] / tab['n_sig'].replace(0, np.nan)
    R('| mt_h_max bin | n significant | n covered | coverage |')
    R('|---|---|---|---|')
    for lb, r in tab.iterrows():
        R(f'| {lb} | {int(r.n_sig):,} | {int(r.n_cov):,} | {fmt(r.coverage)} |')
    J['coverage_by_leverage'] = tab.reset_index().astype(str).to_dict('records')
    R()
    R('Read this first: precision estimates below are conditional on coverage.')
    R('The bootstrap list was chosen by ranking, not at random, so covered pairs')
    R('are an extremity-biased sample of each bin.')
    R()

    # ---- 2. breakdown rate vs leverage (covered significant) --------------
    R('## 2. Breakdown vs leverage (covered significant pairs)')
    covsig['hbin'] = pd.cut(covsig['mt_h_max'], edges, labels=labels,
                            right=False)
    g = covsig.groupby('hbin', observed=False).agg(
        n=('ci_cross', 'size'), ci_cross=('ci_cross', 'mean'),
        break10=('break10', 'mean'), break25=('break25', 'mean'),
        med_sign_frac=('sign_frac', 'median'),
        med_se_ratio=('se_ratio', 'median'))
    R('| mt_h_max bin | n | P(ci_cross) | P(sign_frac>=0.10) | P(>=0.25) |'
      ' med sign_frac | med se_ratio |')
    R('|---|---|---|---|---|---|---|')
    for lb, r in g.iterrows():
        R(f'| {lb} | {int(r.n):,} | {fmt(r.ci_cross)} | {fmt(r.break10)} |'
          f' {fmt(r.break25)} | {fmt(r.med_sign_frac)} | {fmt(r.med_se_ratio, 2)} |')
    J['breakdown_by_leverage'] = g.reset_index().astype(str).to_dict('records')
    R()
    # same, TRANS only
    tr = covsig[covsig['region'] == 'TRANS']
    if len(tr):
        gt = tr.groupby('hbin', observed=False).agg(
            n=('ci_cross', 'size'), ci_cross=('ci_cross', 'mean'),
            break10=('break10', 'mean'), break25=('break25', 'mean'))
        R('TRANS only:')
        R('| mt_h_max bin | n | P(ci_cross) | P(>=0.10) | P(>=0.25) |')
        R('|---|---|---|---|---|')
        for lb, r in gt.iterrows():
            R(f'| {lb} | {int(r.n):,} | {fmt(r.ci_cross)} | {fmt(r.break10)} |'
              f' {fmt(r.break25)} |')
        J['breakdown_by_leverage_trans'] = \
            gt.reset_index().astype(str).to_dict('records')
    R()

    # ---- 3. precision / recall at candidate thresholds --------------------
    def threshold_table(kind, values):
        rows = []
        base = {'ci_cross': covsig['ci_cross'].mean(),
                'break10': covsig['break10'].mean(),
                'break25': covsig['break25'].mean()}
        for v in values:
            if kind == 'abs':
                flag = covsig['mt_h_max'] > v
            else:
                if hC is None:
                    return None, None
                flag = (covsig['mt_h_max'] - hC) > v
            n_f = int(flag.sum())
            row = {'threshold': v, 'n_flagged': n_f,
                   'frac_flagged': n_f / max(len(covsig), 1)}
            for inst in ('ci_cross', 'break10', 'break25'):
                brk = covsig[inst]
                p_f = brk[flag].mean() if n_f else np.nan
                p_u = brk[~flag].mean() if n_f < len(covsig) else np.nan
                rec = (brk & flag).sum() / max(brk.sum(), 1)
                row[f'{inst}|flag'] = p_f
                row[f'{inst}|unflag'] = p_u
                row[f'{inst}_recall'] = rec
            rows.append(row)
        return rows, base

    for kind, values, title in (('abs', CANDIDATE_ABS, 'abs rule (mt_h_max > tau)'),
                                ('floor', CANDIDATE_FLOOR,
                                 'floor rule (mt_h_max - h_C_max > delta)')):
        rows, base = threshold_table(kind, values)
        if rows is None:
            R(f'## 3. {title}: skipped (no --covariates)')
            R()
            continue
        R(f'## 3. Threshold sweep — {title} — covered significant pairs')
        R(f'baseline breakdown (all covered significant): '
          f'ci_cross {fmt(base["ci_cross"])}, >=0.10 {fmt(base["break10"])}, '
          f'>=0.25 {fmt(base["break25"])}')
        R('| thr | n flagged | P(break\\|flagged) ci/10/25 | '
          'P(break\\|unflagged) ci/10/25 | recall ci/10/25 |')
        R('|---|---|---|---|---|')
        for r in rows:
            R(f'| {r["threshold"]} | {r["n_flagged"]:,} |'
              f' {fmt(r["ci_cross|flag"])}/{fmt(r["break10|flag"])}/{fmt(r["break25|flag"])} |'
              f' {fmt(r["ci_cross|unflag"])}/{fmt(r["break10|unflag"])}/{fmt(r["break25|unflag"])} |'
              f' {fmt(r["ci_cross_recall"])}/{fmt(r["break10_recall"])}/{fmt(r["break25_recall"])} |')
        J[f'sweep_{kind}'] = [{k: (None if isinstance(v, float) and not
                                   np.isfinite(v) else v)
                              for k, v in r.items()} for r in rows]
        J[f'baseline_{kind}'] = base
        R()

    # ---- 4. se_ratio corroboration ----------------------------------------
    if covsig['se_ratio'].notna().any():
        R('## 4. se_ratio by leverage (covered significant; median / q90)')
        gg = covsig.groupby('hbin', observed=False)['se_ratio'] \
            .agg(median='median', q90=lambda s: s.quantile(0.9))
        R('| mt_h_max bin | median | q90 |')
        R('|---|---|---|')
        for lb, r in gg.iterrows():
            R(f'| {lb} | {fmt(r["median"], 2)} | {fmt(r["q90"], 2)} |')
        J['se_ratio_by_leverage'] = gg.reset_index().astype(str).to_dict('records')
        R()

    # ---- 5. worked examples ------------------------------------------------
    R('## 5. Highest-leverage covered significant pairs (top 15)')
    cols = ['mt_id', 'gt_id', 'region', 'mt_h_max', 'sign_frac', 'ci_cross',
            'se_ratio', 'degenerate_resamples']
    cols = [c for c in cols if c in covsig.columns]
    top = covsig.sort_values('mt_h_max', ascending=False).head(15)[cols]
    R('| ' + ' | '.join(cols) + ' |')
    R('|' + '---|' * len(cols))
    for _, r in top.iterrows():
        R('| ' + ' | '.join(
            fmt(v) if isinstance(v, (float, np.floating)) else str(v)
            for v in r.tolist()) + ' |')
    J['top15'] = top.astype(str).to_dict('records')
    R()
    R('## Caveats')
    R('- Coverage is selection-biased (ranked bootstrap list); precision is '
      'estimated on the covered stratum only.')
    R('- sign_frac is floored at 1/(2B); ci_cross detects >=2.5% minority mass '
      'only. Bootstrap perturbs all subjects jointly, not a targeted deletion; '
      'a leverage~1 pair can appear bootstrap-stable when the dominant subject '
      'is drawn into most resamples. Deletion-based mt_drop1_ratio (I-1) is the '
      'sharper instrument; this bridge is the available proxy.')
    R('- Breakdown here = sign instability of mt_est under resampling, not '
      'FDR-survival.')

    md = os.path.join(a.out_dir, 'calibration_bridge.md')
    js = os.path.join(a.out_dir, 'calibration_bridge.json')
    with open(md, 'w') as fh:
        fh.write('\n'.join(rep) + '\n')
    with open(js, 'w') as fh:
        json.dump(J, fh, indent=1, default=str)
    print(f'wrote {md}\nwrote {js}')


if __name__ == '__main__':
    main()
