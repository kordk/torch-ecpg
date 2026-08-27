#!/usr/bin/env python3
"""influence_pair_anatomy.py — show, pair by pair, what a leverage flag means.

Promotes the standalone influence runner into a committed tool. For each
selected (CpG, gene) pair it recomputes the mapping regression from M/G/C,
identifies the maximum-leverage subject, deletes that one subject in closed
form, and renders a three-panel row:

  1. covariate-adjusted scatter, with the max-leverage subject marked
  2. the same fit with that subject deleted, overlaid
  3. the effect estimate: analytic CI against the stored bootstrap CI

Pairs are chosen automatically - the most extreme flagged pairs, plus robust
pairs matched on p-value so the contrast is like-for-like - or supplied with
--pairs-file. Closed-form deletion is verified against a brute-force refit on
the first pair and the tool aborts if they disagree.

Usage:
  python3 tools/influence_pair_anatomy.py --dataset gtp \
      --data-dir data_gtp --catalog output_gtp/summarized.influence.parquet \
      --boot output_gtp/bootstrap_merged.parquet \
      --out-dir output_gtp/influence_anatomy --n-flagged 3 --n-robust 3
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
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

COL_PT = '#5b7fa6'
COL_LEV = '#c2452d'
COL_FIT = '#1f3d5c'
COL_DROP = '#c2452d'
RECON_TOL = 1e-2      # |recomputed t - stored mt_t|; guards data/model mismatch
VERIFY_TOL = 1e-8     # closed-form LOO vs brute-force refit


# ----------------------------------------------------------------- regression
def fit(X, y):
    """OLS via QR. Returns beta, residuals, leverage, XtX inverse, sigma2, df."""
    n, p = X.shape
    Q, R = np.linalg.qr(X)
    beta = np.linalg.solve(R, Q.T @ y)
    resid = y - X @ beta
    h = np.einsum('ij,ij->i', Q, Q)
    Rinv = np.linalg.inv(R)
    xtxi = Rinv @ Rinv.T
    dof = n - p
    sigma2 = float(resid @ resid) / dof
    return beta, resid, h, xtxi, sigma2, dof


def t_of(beta, xtxi, sigma2, k):
    se = np.sqrt(sigma2 * xtxi[k, k])
    return float(beta[k] / se), float(se)


def loo(X, y, i, k):
    """Closed-form leave-one-out coefficient and t for observation i, term k."""
    beta, resid, h, xtxi, sigma2, dof = fit(X, y)
    xi = X[i]
    hi = float(h[i])
    ei = float(resid[i])
    if hi >= 1 - 1e-12:
        return None
    corr = xtxi @ xi
    beta_i = beta - corr * (ei / (1.0 - hi))
    rss_i = float(resid @ resid) - ei ** 2 / (1.0 - hi)
    dof_i = dof - 1
    sigma2_i = rss_i / dof_i
    xtxi_i = xtxi + np.outer(corr, corr) / (1.0 - hi)
    se_i = np.sqrt(sigma2_i * xtxi_i[k, k])
    return float(beta_i[k]), float(beta_i[k] / se_i), float(se_i)


def brute_loo(X, y, i, k):
    m = np.ones(len(y), bool)
    m[i] = False
    beta, _, _, xtxi, sigma2, _ = fit(X[m], y[m])
    t, se = t_of(beta, xtxi, sigma2, k)
    return float(beta[k]), t, se


# --------------------------------------------------------------------- inputs
def load_matrices(data_dir, logit):
    def rd(name):
        return pd.read_csv(os.path.join(data_dir, name), index_col=0)
    M, G, C = rd('M.csv'), rd('G.csv'), rd('C.csv')
    # Subject labels may be read as int in one file and str in another (numeric
    # IDs). tecpg compares them as strings; do the same and normalise, so that
    # every subsequent .loc uses one label type.
    M.columns = [str(c) for c in M.columns]
    G.columns = [str(c) for c in G.columns]
    C.index = [str(i) for i in C.index]
    M.index = [str(i) for i in M.index]
    G.index = [str(i) for i in G.index]
    subj = list(M.columns)
    if set(G.columns) != set(subj) or set(C.index) != set(subj):
        missing_g = len(set(subj) - set(G.columns))
        missing_c = len(set(subj) - set(C.index))
        sys.exit(f'M/G/C do not share the same subject labels '
                 f'({missing_g} missing from G, {missing_c} from C).')
    if logit:
        v = M.to_numpy(dtype=float)
        v = np.clip(v, 1e-6, 1 - 1e-6)
        M = pd.DataFrame(np.log2(v / (1 - v)), index=M.index, columns=M.columns)
    return M, G, C, subj


def read_catalog(path, boot, fdr_col):
    names = set(pq.ParquetFile(path).schema_arrow.names)
    want = [c for c in ('mt_id', 'gt_id', 'region', 'mt_est', 'mt_err', 'mt_t',
                        'mt_p', fdr_col, 'mt_h_max', 'mt_influence_flag',
                        'p_boot', 'ci_low', 'ci_high') if c in names]
    df = pq.read_table(path, columns=want).to_pandas()
    if df.index.names != [None]:
        df = df.reset_index()
    for c in ('mt_id', 'gt_id'):
        df[c] = df[c].astype(str)
    if boot:
        bn = set(pq.ParquetFile(boot).schema_arrow.names)
        bw = [c for c in ('mt_id', 'gt_id', 'p_boot', 'ci_low', 'ci_high')
              if c in bn]
        b = pq.read_table(boot, columns=bw).to_pandas()
        if b.index.names != [None]:
            b = b.reset_index()
        for c in ('mt_id', 'gt_id'):
            b[c] = b[c].astype(str)
        b = b.dropna(subset=['p_boot']).drop_duplicates(['mt_id', 'gt_id'])
        df = df.drop(columns=[c for c in ('p_boot', 'ci_low', 'ci_high')
                              if c in df.columns], errors='ignore')
        df = df.merge(b, on=['mt_id', 'gt_id'], how='left',
                      validate='many_to_one')
    return df


def select_pairs(df, n_flag, n_robust, fdr_col, thr):
    """Most extreme flagged pairs, plus robust pairs matched on p-value."""
    d = df.dropna(subset=['mt_h_max', 'mt_p'])
    if fdr_col in d.columns:
        d = d[d[fdr_col] <= thr]
    if 'mt_influence_flag' in d.columns:
        flag = d[d['mt_influence_flag'] == True]                # noqa: E712
        keep = d[d['mt_influence_flag'] != True]                # noqa: E712
    else:
        cut = d['mt_h_max'].quantile(0.999)
        flag, keep = d[d['mt_h_max'] >= cut], d[d['mt_h_max'] < cut]
    chosen = flag.nlargest(n_flag, 'mt_h_max')
    picks = [(r, 'flagged') for _, r in chosen.iterrows()]
    # match each flagged pair to the robust pair closest in p
    pool = keep.copy()
    for _, r in chosen.head(n_robust).iterrows():
        if pool.empty:
            break
        # mt_p is float32 in the catalog; 1e-300 underflows to 0 there, so
        # widen to float64 before clipping or log10 warns on exact zeros.
        pool_p = pool['mt_p'].astype('float64').clip(lower=1e-300)
        target = max(float(r['mt_p']), 1e-300)
        j = (np.log10(pool_p) - np.log10(target)).abs().idxmin()
        picks.append((pool.loc[j], 'robust (p-matched)'))
        pool = pool.drop(index=j)
    return picks


# --------------------------------------------------------------------- render
def render_pair(row, kind, M, G, C, subj, out_dir, written, records):
    cpg, gene = row['mt_id'], row['gt_id']
    cpg, gene = str(cpg), str(gene)
    if cpg not in M.index or gene not in G.index:
        print(f'  skip {cpg} x {gene}: not present in M/G', file=sys.stderr)
        return
    m = M.loc[cpg, subj].to_numpy(dtype=float)
    y = G.loc[gene, subj].to_numpy(dtype=float)
    Cn = C.loc[subj].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(subj)), m, Cn])
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, y, sub = X[ok], y[ok], list(np.asarray(subj)[ok])
    beta, resid, h, xtxi, sigma2, dof = fit(X, y)
    t_full, se_full = t_of(beta, xtxi, sigma2, 1)
    istar = int(np.argmax(h))
    out = loo(X, y, istar, 1)
    if out is None:
        return
    beta_d, t_drop, se_drop = out
    ratio = abs(t_drop) / abs(t_full) if t_full else np.nan

    # partial (added-variable) coordinates: residualise m and y on the covariates
    Z = np.delete(X, 1, axis=1)
    bz, _, _, _, _, _ = fit(Z, m)
    mr = m - Z @ bz
    by, _, _, _, _, _ = fit(Z, y)
    yr = y - Z @ by

    rec = {'mt_id': cpg, 'gt_id': gene, 'kind': kind,
           'region': row.get('region'), 'n_subjects': int(len(y)),
           'mt_h_max_recomputed': float(h[istar]),
           'mt_h_max_stored': float(row.get('mt_h_max', np.nan)),
           'leverage_subject': str(sub[istar]),
           't_full': t_full, 't_drop1': t_drop, 'drop1_ratio': float(ratio),
           'beta_full': float(beta[1]), 'beta_drop1': beta_d,
           'stored_mt_t': float(row.get('mt_t', np.nan)),
           'p_boot': float(row.get('p_boot', np.nan)),
           'ci_low': float(row.get('ci_low', np.nan)),
           'ci_high': float(row.get('ci_high', np.nan))}
    records.append(rec)
    if not HAVE_MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=200,
                             gridspec_kw={'width_ratios': [1, 1, 0.85]})
    tag = 'FLAGGED' if kind == 'flagged' else 'ROBUST'
    fig.suptitle(f'{tag}   {cpg} x {gene}'
                 f'   ({row.get("region", "?")})   leverage {h[istar]:.3f}',
                 fontsize=11, y=1.02)

    # panel 1: adjusted scatter, influential subject marked
    ax = axes[0]
    ax.scatter(mr, yr, s=18, alpha=0.55, color=COL_PT, edgecolors='none')
    ax.scatter([mr[istar]], [yr[istar]], s=95, facecolors='none',
               edgecolors=COL_LEV, linewidths=2, zorder=5)
    ax.annotate(f'subject {sub[istar]}\nleverage {h[istar]:.3f}',
                (mr[istar], yr[istar]), xytext=(10, 8),
                textcoords='offset points', fontsize=8, color=COL_LEV)
    xs = np.array([mr.min(), mr.max()])
    ax.plot(xs, beta[1] * xs, color=COL_FIT, lw=2,
            label=f'all subjects: t = {t_full:.1f}')
    ax.set_xlabel('methylation, covariate-adjusted')
    ax.set_ylabel('expression, covariate-adjusted')
    ax.set_title('The association as fitted', fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='best')
    ax.spines[['top', 'right']].set_visible(False)

    # panel 2: same data, fit with that subject deleted
    ax = axes[1]
    keep = np.ones(len(mr), bool)
    keep[istar] = False
    ax.scatter(mr[keep], yr[keep], s=18, alpha=0.55, color=COL_PT,
               edgecolors='none')
    ax.scatter([mr[istar]], [yr[istar]], s=95, facecolors='none',
               edgecolors=COL_LEV, linewidths=2, alpha=0.45, zorder=5)
    ax.plot(xs, beta[1] * xs, color=COL_FIT, lw=1.4, ls=':',
            label=f'all subjects: t = {t_full:.1f}')
    ax.plot(xs, beta_d * xs, color=COL_DROP, lw=2,
            label=f'that subject removed: t = {t_drop:.1f}')
    ax.set_xlabel('methylation, covariate-adjusted')
    ax.set_title(f'Deleting one subject   (|t| ratio {ratio:.2f})', fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='best')
    ax.spines[['top', 'right']].set_visible(False)

    # panel 3: analytic vs bootstrap interval
    ax = axes[2]
    lo_a, hi_a = beta[1] - 1.96 * se_full, beta[1] + 1.96 * se_full
    items = [('analytic\n(all subjects)', beta[1], lo_a, hi_a, COL_FIT),
             ('after deleting\nthe subject', beta_d,
              beta_d - 1.96 * se_drop, beta_d + 1.96 * se_drop, COL_DROP)]
    if np.isfinite(rec['ci_low']) and np.isfinite(rec['ci_high']):
        items.append(('bootstrap\n(1000 resamples)', beta[1], rec['ci_low'],
                      rec['ci_high'], '#7a6b98'))
    ys = np.arange(len(items))[::-1]
    for yy, (lab, est, lo, hi, col) in zip(ys, items):
        ax.plot([lo, hi], [yy, yy], color=col, lw=3, solid_capstyle='round')
        ax.plot([est], [yy], 'o', color=col, ms=7)
    ax.axvline(0, color='#444', lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([i[0] for i in items], fontsize=8)
    ax.set_xlabel('effect estimate (95% interval)')
    ax.set_title('Does the interval still exclude zero?', fontsize=9)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    fig.tight_layout()
    stem = f'anatomy_{kind.split()[0]}_{cpg}_{gene}'.replace('/', '_')
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'{stem}.{ext}')
        fig.savefig(p, dpi=200, bbox_inches='tight')
        if ext == 'png':
            written.append(p)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description='Per-pair anatomy of a leverage-flagged association.')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--data-dir', required=True, help='directory with M/G/C.csv')
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--boot', help='bootstrap-merged parquet (for the CI panel)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--pairs-file', help='CSV with mt_id,gt_id to render')
    ap.add_argument('--n-flagged', type=int, default=3)
    ap.add_argument('--n-robust', type=int, default=3)
    ap.add_argument('--logit', action='store_true',
                    help='apply the logit transform to M (match the mapping)')
    ap.add_argument('--fdr-column', default='fdr_est')
    ap.add_argument('--fdr-threshold', type=float, default=0.05)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    df = read_catalog(a.catalog, a.boot, a.fdr_column)
    M, G, C, subj = load_matrices(a.data_dir, a.logit)

    if a.pairs_file:
        want = pd.read_csv(a.pairs_file, dtype=str)
        picks = [(df[(df['mt_id'] == r.mt_id) & (df['gt_id'] == r.gt_id)]
                  .iloc[0], 'requested')
                 for r in want.itertuples() if
                 ((df['mt_id'] == r.mt_id) & (df['gt_id'] == r.gt_id)).any()]
    else:
        picks = select_pairs(df, a.n_flagged, a.n_robust, a.fdr_column,
                             a.fdr_threshold)
    if not picks:
        sys.exit('no pairs selected; check --catalog and the flag column')

    # gate: closed-form deletion must match a brute-force refit on pair 1
    r0 = picks[0][0]
    if r0['mt_id'] in M.index and r0['gt_id'] in G.index:
        m = M.loc[r0['mt_id'], subj].to_numpy(dtype=float)
        y = G.loc[r0['gt_id'], subj].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(subj)), m, C.loc[subj].to_numpy(float)])
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        X, y = X[ok], y[ok]
        _, _, h, _, _, _ = fit(X, y)
        i = int(np.argmax(h))
        cf, bf = loo(X, y, i, 1), brute_loo(X, y, i, 1)
        d = max(abs(cf[0] - bf[0]), abs(cf[1] - bf[1]))
        print(f'closed-form vs brute-force refit: max |diff| = {d:.2e}')
        if d > VERIFY_TOL:
            sys.exit(f'closed-form deletion disagrees with a refit ({d:.2e}); '
                     f'refusing to emit figures.')
        b_r, _, _, xtxi_r, s2_r, _ = fit(X, y)
        t_recon, _ = t_of(b_r, xtxi_r, s2_r, 1)
        stored = float(r0.get('mt_t', np.nan))
        if np.isfinite(stored):
            print(f'reconstruction check: recomputed t = {t_recon:.4f}, '
                  f'stored mt_t = {stored:.4f}, diff = '
                  f'{abs(t_recon - stored):.2e}')
            if abs(t_recon - stored) > RECON_TOL:
                print('WARNING: recomputed t does not match the stored value. '
                      'The data directory, covariate set, or transform may not '
                      'match the mapping that produced this catalog '
                      '(try --logit).', file=sys.stderr)

    written, records = [], []
    for row, kind in picks:
        render_pair(row, kind, M, G, C, subj, a.out_dir, written, records)
    with open(os.path.join(a.out_dir, 'pair_anatomy.json'), 'w') as fh:
        json.dump({'dataset': a.dataset, 'pairs': records}, fh, indent=1,
                  default=str)
    print(f'wrote {len(written)} figure(s) (png + pdf) to {a.out_dir}')
    for p in written:
        print(f'  {p}')


if __name__ == '__main__':
    main()
