"""Tests for tools/chromatinEnrichment_parquet.py (Kennedy Fig. 6 support, chunks E2-E4).

Every 2x2 cell of both panels is checked against a brute-force count over the
synthetic universe / pair set. Guards (universe contract, coordinate drop
site, build mismatch, missing significance column) are checked at the library
level and at the CLI exit-code level. Odds ratio and CI are checked against an
independent conditional-MLE implementation (the fisher.test estimator); BH is
checked against statsmodels per panel; integer cells are fingerprinted. The
figure is smoke-tested for shape and content markers, not pixels.
"""
import json
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.special import gammaln
from scipy.stats import fisher_exact

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
import chromatin_features as cf  # noqa: E402
import chromatinEnrichment_parquet as ce  # noqa: E402

TOOL = os.path.join(REPO_ROOT, 'tools', 'chromatinEnrichment_parquet.py')
FINGERPRINT = os.path.join(REPO_ROOT, 'tests', 'fingerprint_chromatin_enrichment.json')
STAT_COLS = {'odds_ratio', 'ci_low', 'ci_high', 'sample_or', 'p', 'degenerate', 'q_bh'}


def cond_mle_or(a, b, c, d, alpha=0.05):
    """Independent conditional MLE of the odds ratio (noncentral hypergeometric
    mean inversion) with exact CI by inverting the two one-sided conditional
    tests. Mirrors R fisher.test; used as the oracle for fisher_or_ci."""
    n1, n2, m1 = a + b, c + d, a + c
    lo, hi = max(0, m1 - n2), min(n1, m1)
    ks = np.arange(lo, hi + 1)
    logw = (gammaln(n1 + 1) - gammaln(ks + 1) - gammaln(n1 - ks + 1)
            + gammaln(n2 + 1) - gammaln(m1 - ks + 1) - gammaln(n2 - m1 + ks + 1))

    def probs(psi):
        lp = logw + ks * np.log(psi)
        lp -= lp.max()
        pr = np.exp(lp)
        return pr / pr.sum()

    def mean(psi):
        return float((ks * probs(psi)).sum())

    def solve(f, target):
        return brentq(lambda x: f(np.exp(x)) - target, np.log(1e-9), np.log(1e9), xtol=1e-12)

    mle = 0.0 if a == lo else (np.inf if a == hi else np.exp(solve(mean, a)))
    ci_low = 0.0 if a == lo else np.exp(solve(lambda psi: float(probs(psi)[ks >= a].sum()), alpha / 2))
    ci_high = np.inf if a == hi else np.exp(solve(lambda psi: float(probs(psi)[ks <= a].sum()), alpha / 2))
    return mle, ci_low, ci_high


def make_bed(path, rows):
    pd.DataFrame(rows).to_csv(path, sep='\t', header=False, index=False)


@pytest.fixture
def synthetic(tmp_path):
    """200-CpG universe on chr1/chr2 (+2 ids without coordinates), 30
    significant CpGs across 3 regions, one CpG carrying 3 extra TRANS pairs,
    7 non-significant rows, CGI/shore/shelf + four bed4 states + a union.
    Seeded locally so the fixture is order-independent."""
    rng = np.random.default_rng(20180619)
    n = 200
    chrom = np.where(np.arange(n) < 120, 'chr1', 'chr2')
    pos = rng.integers(0, 100_000, n)
    ids = [f'cg{i:05d}' for i in range(n)]
    bed = tmp_path / 'M.bed6'
    make_bed(bed, [dict(c=chrom[i], s=int(pos[i]), e=int(pos[i]) + 1, n=ids[i], sc=0, st='+') for i in range(n)])
    m = tmp_path / 'M.csv'
    pd.DataFrame({'id': ids + ['cg_nocoord_a', 'cg_nocoord_b'], 's1': 0.0, 's2': 1.0}).to_csv(m, index=False)
    pairs = []
    sig_cpgs = list(rng.choice(ids[:180], 30, replace=False))
    regions = ['TRANS', 'CIS5', 'GENEBODY']
    for k, c in enumerate(sig_cpgs):
        pairs.append(dict(mt_id=c, gt_id=f'ILMN_{k}', region=regions[k % 3], fdr_est=0.01, mt_p=1e-12))
    multi = sig_cpgs[0]
    for j in range(3):
        pairs.append(dict(mt_id=multi, gt_id=f'ILMN_multi{j}', region='TRANS', fdr_est=0.001, mt_p=1e-13))
    for j in range(7):
        pairs.append(dict(mt_id=ids[190 + j], gt_id=f'ILMN_ns{j}', region='DISTAL5', fdr_est=0.5, mt_p=0.01))
    cat = tmp_path / 'summarized.parquet'
    pd.DataFrame(pairs).to_parquet(cat, index=False)
    isl = tmp_path / 'cgi.bed'
    make_bed(isl, [dict(c='chr1', s=10_000, e=12_000), dict(c='chr1', s=14_500, e=15_000),
                   dict(c='chr2', s=50_000, e=60_000)])
    hmm = tmp_path / 'hmm.bed'
    hmm_rows = []
    for c in ('chr1', 'chr2'):
        edges = np.sort(rng.choice(np.arange(0, 100_001, 500), 40, replace=False))
        for s, e in zip(edges[:-1], edges[1:]):
            hmm_rows.append(dict(c=c, s=int(s), e=int(e), n=f'{rng.integers(1, 16)}_State'))
    make_bed(hmm, hmm_rows)
    names = sorted({r['n'] for r in hmm_rows}, key=lambda x: int(x.split('_')[0]))
    man = tmp_path / 'tracks.tsv'
    lines = ['name\tkind\tpath\tbuild\tselect',
             f'CGI\tbed\t{isl}\thg19\t.',
             'CGI_shore\tderived\tCGI\thg19\tflank=1500',
             'CGI_shelf\tderived\tCGI_shore\thg19\tflank=1500;exclude=CGI']
    for nm in names[:4]:
        lines.append(f'HMM_{nm.split("_")[0]}\tbed4\t{hmm}\thg19\tname={nm}')
    lines.append(f'HMM_union\tunion\tHMM_{names[0].split("_")[0]},HMM_{names[1].split("_")[0]}\thg19\t.')
    man.write_text('\n'.join(lines) + '\n')
    return dict(tmp=tmp_path, bed=bed, m=m, cat=cat, man=man, multi=multi)


def _run(synthetic):
    tracks, _ = cf.load_tracks(synthetic['man'], 'hg19')
    ids = ce.load_universe_ids(synthetic['m'])
    universe, n_no = ce.build_universe(ids, ce.load_cpg_bed(synthetic['bed']))
    sig = ce.load_significant_pairs(synthetic['cat'], 'fdr_est', 0.05)
    ce.check_universe_contract(sig, universe)
    ov = ce.annotate_overlaps(universe, tracks)
    return ids, universe, n_no, sig, ov, ce.build_panel_a(sig, universe, ov), ce.build_panel_b(sig, universe, ov)


def _cli(synthetic, out, extra=()):
    cmd = [sys.executable, TOOL, '--catalog', str(synthetic['cat']), '--cpg-universe', str(synthetic['m']),
           '--cpg-bed', str(synthetic['bed']), '--tracks', str(synthetic['man']), '--genome-build', 'hg19',
           '--out-dir', str(out)] + list(extra)
    return subprocess.run(cmd, capture_output=True, text=True)


# -------------------------------------------------------------------- T6
def test_T6_panel_a_oracle(synthetic):
    ids, universe, n_no, sig, ov, pa, pb = _run(synthetic)
    N = len(universe)
    assert N == 200 and n_no == 2
    sig_cpgs = set(sig['mt_id'])
    for rec in pa.itertuples():
        row_set = sig_cpgs if rec.row == 'ALL' else set(sig.loc[sig['region'] == rec.row, 'mt_id'])
        a = sum(1 for u in universe.index if u in row_set and ov.at[u, rec.feature])
        b = sum(1 for u in universe.index if u in row_set and not ov.at[u, rec.feature])
        c = sum(1 for u in universe.index if u not in row_set and ov.at[u, rec.feature])
        d = N - a - b - c
        assert (rec.a, rec.b, rec.c, rec.d) == (a, b, c, d), rec
        assert rec.a + rec.b + rec.c + rec.d == N
        assert rec.n_row == len(row_set)
    # CpG with 3 extra TRANS pairs is counted once in TRANS and once in ALL
    assert pa.loc[pa.row == 'TRANS', 'n_row'].iloc[0] == sig.loc[sig.region == 'TRANS', 'mt_id'].nunique()
    assert pa.loc[pa.row == 'ALL', 'n_row'].iloc[0] == 30
    assert list(pa['row'].unique()) == ['ALL', 'GENEBODY', 'CIS5', 'TRANS']
    assert set(pa.columns) == {'panel', 'row', 'cis', 'feature', 'n_row', 'a', 'b', 'c', 'd'} | STAT_COLS
    assert pa.loc[pa.row.isin(['GENEBODY', 'CIS5']), 'cis'].eq(1).all() and pa.loc[pa.row == 'TRANS', 'cis'].eq(0).all()


# -------------------------------------------------------------------- T7
def test_T7_panel_b_oracle(synthetic):
    ids, universe, n_no, sig, ov, pa, pb = _run(synthetic)
    P = len(sig)
    assert P == 33
    for rec in pb.itertuples():
        a = sum(1 for r in sig.itertuples() if r.region == rec.row and ov.at[r.mt_id, rec.feature])
        b = sum(1 for r in sig.itertuples() if r.region == rec.row and not ov.at[r.mt_id, rec.feature])
        c = sum(1 for r in sig.itertuples() if r.region != rec.row and ov.at[r.mt_id, rec.feature])
        assert (rec.a, rec.b, rec.c, rec.d) == (a, b, c, P - a - b - c), rec
        assert rec.a + rec.b + rec.c + rec.d == P
    n_rows = pb.drop_duplicates('row').set_index('row')['n_row']
    assert n_rows.sum() == P
    assert n_rows['TRANS'] == 13  # 10 single + 3 multi
    assert 'ALL' not in set(pb['row'])


# -------------------------------------------------------------------- T8
def test_T8_universe_contract(synthetic, tmp_path):
    ids, universe, n_no, sig, ov, pa, pb = _run(synthetic)
    sig2 = pd.concat([sig, pd.DataFrame([dict(mt_id='cg_alien', gt_id='ILMN_x', region='TRANS')])], ignore_index=True)
    with pytest.raises(ce.UniverseContractError) as ei:
        ce.check_universe_contract(sig2, universe)
    assert 'cg_alien' in str(ei.value)
    # CLI: a catalog carrying a CpG outside the universe exits 2
    cat2 = tmp_path / 'alien.parquet'
    df = pd.read_parquet(synthetic['cat'])
    pd.concat([df, pd.DataFrame([dict(mt_id='cg_alien', gt_id='ILMN_x', region='TRANS', fdr_est=0.001, mt_p=1e-13)])]).to_parquet(cat2, index=False)
    r = _cli(dict(synthetic, cat=cat2), tmp_path / 'o8')
    assert r.returncode == 2 and 'cg_alien' in r.stderr


# -------------------------------------------------------------------- T9
def test_T9_coordinate_drop_site(synthetic, caplog):
    caplog.set_level(logging.INFO)
    ids, universe, n_no, sig, ov, pa, pb = _run(synthetic)
    assert len(ids) == 202 and n_no == 2 and len(universe) == 200
    assert any('Drop site chromatin_enrichment.universe[coords]: 2 of 202' in m for m in caplog.messages)


# -------------------------------------------------------------------- T10
@pytest.mark.parametrize('table', [(1200, 33318, 16000, 371498), (12, 30, 40, 200), (5, 5, 5, 5),
                                   (0, 20, 10, 100), (7, 0, 3, 50), (2, 1, 1, 2)])
def test_T10_or_ci_p_oracle(table):
    a, b, c, d = table
    got = ce.fisher_or_ci(a, b, c, d)
    mle, lo, hi = cond_mle_or(a, b, c, d)
    p_ref = fisher_exact([[a, b], [c, d]])[1]
    if np.isinf(mle):
        assert np.isinf(got['odds_ratio'])
    else:
        assert np.isclose(np.log(max(got['odds_ratio'], 1e-300)), np.log(max(mle, 1e-300)), atol=1e-5, rtol=0)
    assert np.isclose(np.log(max(got['ci_low'], 1e-300)), np.log(max(lo, 1e-300)), atol=1e-5, rtol=0)
    if np.isinf(hi):
        assert np.isinf(got['ci_high'])
    else:
        assert np.isclose(np.log(got['ci_high']), np.log(hi), atol=1e-5, rtol=0)
    assert np.isclose(got['p'], p_ref, atol=1e-12, rtol=0)
    assert np.isclose(got['sample_or'], (a * d) / (b * c), atol=1e-12, rtol=0) if b * c > 0 else True


# -------------------------------------------------------------------- T11
def test_T11_degenerate_cells():
    z = ce.fisher_or_ci(0, 20, 10, 100)
    assert z['odds_ratio'] == 0.0 and z['ci_low'] == 0.0 and np.isfinite(z['ci_high']) and z['degenerate'] == 1
    z = ce.fisher_or_ci(7, 0, 3, 50)
    assert np.isinf(z['odds_ratio']) and np.isinf(z['ci_high']) and z['degenerate'] == 1
    assert ce.fisher_or_ci(5, 5, 5, 5)['degenerate'] == 0
    z = ce.fisher_or_ci(0, 10, 0, 100)   # feature never overlaps the universe: undefined
    assert np.isnan(z['odds_ratio']) and np.isnan(z['sample_or']) and z['degenerate'] == 1 and z['p'] == 1.0


# -------------------------------------------------------------------- T12
def test_T12_bh_family_per_panel(synthetic):
    from statsmodels.stats.multitest import multipletests
    ids, universe, n_no, sig, ov, pa, pb = _run(synthetic)
    assert np.allclose(pa['q_bh'], multipletests(pa['p'], method='fdr_bh')[1], atol=1e-12, rtol=0)
    assert np.allclose(pb['q_bh'], multipletests(pb['p'], method='fdr_bh')[1], atol=1e-12, rtol=0)
    # independence: a frame's q comes from its own p-vector, not a pooled one
    fa = ce.add_bh_within_panel(pd.DataFrame({'p': [0.001, 0.02, 0.5]}))
    assert np.allclose(fa['q_bh'], multipletests([0.001, 0.02, 0.5], method='fdr_bh')[1])
    pooled = multipletests([0.001, 0.02, 0.5, 0.04, 0.9, 0.001, 0.002, 0.003], method='fdr_bh')[1]
    assert not np.allclose(pooled[:3], fa['q_bh'])


# -------------------------------------------------------------------- T13
def test_T13_fingerprint(synthetic, tmp_path):
    out = tmp_path / 'out'
    r = _cli(synthetic, out, ['--expect-n-universe', '202'])
    assert r.returncode == 0, r.stderr
    m = json.load(open(out / 'chromatin_enrichment_metrics.json'))
    got = dict(n_universe=m['n_universe'], n_universe_no_coords=m['n_universe_no_coords'],
               n_sig_pairs=m['n_sig_pairs'], n_sig_cpgs=m['n_sig_cpgs'], panel_a=m['panel_a'], panel_b=m['panel_b'])
    expected = json.load(open(FINGERPRINT))
    assert got == expected
    # float columns in the TSVs: present, finite where not degenerate
    pa = pd.read_csv(out / 'chromatin_enrichment_panelA.tsv', sep='\t')
    pb = pd.read_csv(out / 'chromatin_enrichment_panelB.tsv', sep='\t')
    assert STAT_COLS <= set(pa.columns) and STAT_COLS <= set(pb.columns)
    nd = pa[pa['degenerate'] == 0]
    assert np.isfinite(nd[['odds_ratio', 'ci_low', 'ci_high', 'p', 'q_bh']].to_numpy()).all()
    assert ((pa['p'] >= 0) & (pa['p'] <= 1)).all() and ((pa['q_bh'] >= pa['p'] - 1e-12)).all()


# -------------------------------------------------------------------- T14
def test_T14_figure_smoke(synthetic, tmp_path):
    out = tmp_path / 'out'
    r = _cli(synthetic, out, ['--plot', '--dataset-label', 'FIXTURE'])
    assert r.returncode == 0, r.stderr
    png = out / 'chromatin_enrichment_fig6.png'
    assert png.exists() and png.stat().st_size > 0
    assert 'figure written' in r.stderr and 'colour by p < 0.05' in r.stderr
    # shape: rows per panel and features come straight from the tables
    ids, universe, n_no, sig, ov, pa, pb = _run(synthetic)
    rows_a, rows_b, feats = ce.plot_fig6(pa, pb, str(tmp_path / 'direct.png'))
    assert rows_a == ['ALL', 'GENEBODY', 'CIS5', 'TRANS'] and rows_b == ['GENEBODY', 'CIS5', 'TRANS']
    assert feats == list(dict.fromkeys(pa['feature'])) and len(feats) == 8
    # a run without --plot writes no figure
    out2 = tmp_path / 'out2'
    r2 = _cli(synthetic, out2)
    assert r2.returncode == 0 and not (out2 / 'chromatin_enrichment_fig6.png').exists()
    # --color-by q_bh and --alpha are accepted and echoed
    r3 = _cli(synthetic, tmp_path / 'out3', ['--plot', '--color-by', 'q_bh', '--alpha', '0.1'])
    assert r3.returncode == 0 and 'colour by q_bh < 0.1' in r3.stderr


def test_T14b_figure_content_markers(synthetic, tmp_path, monkeypatch):
    """Cell text and shading are checked through the matplotlib objects
    rather than pixels: n/a for undefined OR, inf for infinite OR, one
    grey band per cis row, one hatch per degenerate cell."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    ids, universe, n_no, sig, ov, pa, pb = _run(synthetic)
    captured = {}
    real_savefig = plt.Figure.savefig

    def spy(self, *a, **k):
        captured['fig'] = self
        return real_savefig(self, *a, **k)
    monkeypatch.setattr(plt.Figure, 'savefig', spy)
    real_close = plt.close
    monkeypatch.setattr(plt, 'close', lambda *a, **k: None)
    try:
        ce.plot_fig6(pa, pb, str(tmp_path / 'm.png'))
        fig = captured['fig']
        for ax, df in zip(fig.axes, (pa, pb)):
            texts = [t.get_text() for t in ax.texts]
            n_nan = int(df['odds_ratio'].isna().sum()); n_inf = int(np.isinf(df['odds_ratio']).sum())
            assert texts.count('n/a') == n_nan and texts.count('inf') == n_inf
            assert 'nan' not in texts
            rects = [p for p in ax.patches if isinstance(p, Rectangle)]
            bands = [p for p in rects if p.get_fill() and p.get_alpha() == 0.12]
            hatched = [p for p in rects if p.get_hatch()]
            assert len(bands) == int(df.drop_duplicates('row')['cis'].sum())
            assert len(hatched) == int(df['degenerate'].sum())
    finally:
        real_close('all')


# ------------------------------------------------------------- guards, CLI
def test_missing_sig_column_fails_closed(synthetic):
    with pytest.raises(ce.UniverseContractError) as ei:
        ce.load_significant_pairs(synthetic['cat'], 'fdr_permute', 0.05)
    assert 'fdr_permute' in str(ei.value) and 'Refusing' in str(ei.value)


def test_duplicate_universe_ids_fail(tmp_path):
    m = tmp_path / 'M.csv'
    pd.DataFrame({'id': ['cg1', 'cg2', 'cg1'], 's1': 0.0}).to_csv(m, index=False)
    with pytest.raises(ce.UniverseContractError):
        ce.load_universe_ids(m)


def test_cli_end_to_end_and_metrics(synthetic, tmp_path):
    out = tmp_path / 'out'
    r = _cli(synthetic, out, ['--expect-n-universe', '202'])
    assert r.returncode == 0, r.stderr
    for f in ('chromatin_enrichment_panelA.tsv', 'chromatin_enrichment_panelB.tsv', 'chromatin_enrichment_metrics.json'):
        assert (out / f).exists()
    m = json.load(open(out / 'chromatin_enrichment_metrics.json'))
    assert (m['n_universe_ids'], m['n_universe_no_coords'], m['n_universe'], m['n_sig_pairs'], m['n_sig_cpgs']) == (202, 2, 200, 33, 30)
    assert m['genome_build'] == 'hg19' and m['sig_column'] == 'fdr_est' and len(m['catalog_sha256']) == 64
    assert set(m['track_provenance']) == {'CGI', 'CGI_shore', 'CGI_shelf', 'HMM_union'} | {k for k in m['track_provenance'] if k.startswith('HMM_')}
    pa = pd.read_csv(out / 'chromatin_enrichment_panelA.tsv', sep='\t')
    assert len(m['panel_a']) == len(pa) == 4 * 8 and len(m['panel_b']) == 3 * 8
    assert all(c['a'] + c['b'] + c['c'] + c['d'] == 200 for c in m['panel_a'])
    assert all(c['a'] + c['b'] + c['c'] + c['d'] == 33 for c in m['panel_b'])


def test_cli_expect_n_universe_mismatch_exits_2(synthetic, tmp_path):
    r = _cli(synthetic, tmp_path / 'o', ['--expect-n-universe', '999'])
    assert r.returncode == 2 and '999' in r.stderr


def test_cli_build_mismatch_exits_2(synthetic, tmp_path):
    bad = tmp_path / 'bad.tsv'
    bad.write_text(synthetic['man'].read_text().replace('\thg19\t', '\thg38\t', 1))
    r = _cli(dict(synthetic, man=bad), tmp_path / 'o')
    assert r.returncode == 2 and 'hg38' in r.stderr
