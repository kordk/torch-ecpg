import os
import pytest
import torch
import numpy as np
import pandas as pd
from tecpg.processing import tecpg_mlr_qr
from click.testing import CliRunner

def get_tiny_data():
    np.random.seed(42)
    torch.manual_seed(42)
    S, G, M, K = 40, 8, 6, 2
    X = pd.DataFrame(np.random.rand(S, K))
    gt = pd.DataFrame(np.random.rand(S, G))
    mt = pd.DataFrame(np.random.rand(S, M))
    mt.iloc[0, 0] = 100.0  # leverage spike
    return mt.T, gt.T, X

def get_h_max_ref(mt_data, cov_data):
    # Compute h_max locally via numpy
    # X = cat(ones, mt, ct)
    S = mt_data.shape[1]
    M = mt_data.shape[0]
    K_cov = cov_data.shape[1]
    h_max_ref = np.zeros(M)
    ones = np.ones((S, 1))
    ct = cov_data.values
    for i in range(M):
        mt_i = mt_data.iloc[i].values.reshape(-1, 1)
        X_i = np.concatenate((ones, mt_i, ct), axis=1)
        q, r = np.linalg.qr(X_i, mode='reduced')
        h_max_ref[i] = (q * q).sum(axis=1).max()
    return h_max_ref

def get_tiny_annot(mt, gt):
    """All loci on chrom 1. Genes at 1e6*j (strand +). CpGs 0..2 sit exactly at gene
    starts 0..2 (always cis); CpGs 3.. sit 500 kb past genes 3.. (never cis under a
    50 kb / 3 kb window). Columns match what region filtration consumes."""
    g_names = list(gt.index); m_names = list(mt.index)
    g_start = [1_000_000 * j for j in range(len(g_names))]
    G_annot = pd.DataFrame({'chrom': 1, 'chromStart': g_start,
                            'chromEnd': [s + 1000 for s in g_start],
                            'strand': '+', 'score': 0}, index=pd.Index(g_names, name='name'))
    m_start = [g_start[i] if i < 3 else g_start[i] + 500_000 for i in range(len(m_names))]
    M_annot = pd.DataFrame({'chrom': 1, 'chromStart': m_start,
                            'chromEnd': [s + 1 for s in m_start],
                            'strand': '+', 'score': 0}, index=pd.Index(m_names, name='name'))
    return M_annot, G_annot

def test_flag_off_no_column():
    mt, gt, x = get_tiny_data()
    res = tecpg_mlr_qr(M=mt, G=gt, C=x, p_thresh=None, region='all', compute_influence=False)
    assert 'mt_h_max' not in res.columns

def test_flag_on_column_present():
    mt, gt, x = get_tiny_data()
    res = tecpg_mlr_qr(M=mt, G=gt, C=x, p_thresh=None, region='all', compute_influence=True)
    assert 'mt_h_max' in res.columns
    assert res.columns[-1] == 'mt_h_max'

def test_h_max_matches_leverage_reference():
    mt, gt, x = get_tiny_data()
    res = tecpg_mlr_qr(M=mt, G=gt, C=x, p_thresh=None, region='all', compute_influence=True)
    ref = get_h_max_ref(mt, x)

    # Res has M*G rows. Let's isolate unique CpGs
    M = mt.shape[0]
    G = gt.shape[0]
    # In 'res', the rows are named e.g., index names.
    for i in range(M):
        mt_name = mt.index[i]
        # Filter for this CpG
        subset = res.loc[(slice(None), mt_name), :] if isinstance(res.index, pd.MultiIndex) else res[res.index.get_level_values(1) == mt_name]
        h_max_val = subset['mt_h_max'].iloc[0]
        assert np.isclose(h_max_val, ref[i], atol=5e-5)

def test_h_max_constant_across_genes():
    mt, gt, x = get_tiny_data()
    res = tecpg_mlr_qr(M=mt, G=gt, C=x, p_thresh=None, region='all', compute_influence=True)
    M = mt.shape[0]
    for i in range(M):
        mt_name = mt.index[i]
        subset = res[res.index.get_level_values(1) == mt_name]
        vals = subset['mt_h_max'].values
        assert np.allclose(vals, vals[0], atol=5e-5)

def test_h_max_masked_consistently():
    mt, gt, x = get_tiny_data()
    res_all = tecpg_mlr_qr(M=mt, G=gt, C=x, p_thresh=None, region='all', compute_influence=True)
    res_thresh = tecpg_mlr_qr(M=mt, G=gt, C=x, p_thresh=0.5, region='all', compute_influence=True)

    for idx, row in res_thresh.iterrows():
        val_thresh = row['mt_h_max']
        val_all = res_all.loc[idx, 'mt_h_max']
        assert val_thresh == val_all

def test_influence_with_ig_deep_raises_usage_error(tmp_path):
    runner = CliRunner()
    from tecpg.cli import mlr
    from tecpg.logger import Logger
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        for name in ('M.csv', 'G.csv', 'C.csv'):
            with open(name, 'w') as fh:
                fh.write('1\n2')
        result = runner.invoke(mlr, ['--compute-influence', '--compute-ig-deep', '--p-thresh', '0.05', '--mlr-method', 'qr'], obj={'logger': Logger(), 'data': {'root_path': '.', 'input_dir': '.', 'output_dir': '.', 'meth_file': 'M.csv', 'gene_file': 'G.csv', 'covar_file': 'C.csv'}})
    assert result.exit_code != 0
    assert 'Cannot use --compute-influence with --compute-ig-deep' in result.output

def _rename_tiny(mt, gt):
    mt.index = [f'cg{i}' for i in range(len(mt))]
    gt.index = [f'g{j}' for j in range(len(gt))]
    return mt, gt

def test_h_max_region_cis_matches_all(tmp_path):
    mt, gt, x = get_tiny_data()
    mt, gt = _rename_tiny(mt, gt)
    M_annot, G_annot = get_tiny_annot(mt, gt)
    res_all = tecpg_mlr_qr(M=mt, G=gt, C=x, p_thresh=None, region='all', compute_influence=True)
    res_cis = tecpg_mlr_qr(M=mt, G=gt, C=x, M_annot=M_annot, G_annot=G_annot, p_thresh=None,
                           region='cis', window_base=0, upstream=50000, downstream=3000,
                           compute_influence=True)
    # region filtration actually filtered: only the three cis CpGs survive
    cis_cpgs = set(res_cis.index.get_level_values('mt_id'))
    assert cis_cpgs == {'cg0', 'cg1', 'cg2'}, cis_cpgs
    assert res_cis.columns[-1] == 'mt_h_max'
    # the [region_mask] line: each surviving row carries its own CpG's h_max, equal to the 'all' run
    h_all = res_all.groupby(level='mt_id')['mt_h_max'].first()
    for cpg, grp in res_cis.groupby(level='mt_id'):
        assert np.allclose(grp['mt_h_max'].to_numpy(), h_all.loc[cpg], atol=5e-5)

def _read_chunks(out_dir):
    import glob
    files = sorted(glob.glob(os.path.join(str(out_dir), '*.parquet')))
    assert files, 'no chunk parquets written'
    df = pd.concat([pd.read_parquet(f) for f in files])
    df = df.reset_index() if df.index.names != [None] else df
    df['mt_id'] = df['mt_id'].astype(str)
    return df

def test_h_max_chunked_multi_meth_chunk(tmp_path):
    from tecpg.logger import Logger
    mt, gt, x = get_tiny_data()
    mt, gt = _rename_tiny(mt, gt)
    out = tmp_path / 'chunked'; out.mkdir()
    tecpg_mlr_qr(M=mt, G=gt, C=x, p_thresh=None, region='all', compute_influence=True,
                 output_dir=str(out), gene_loci_per_chunk=3, meth_loci_per_chunk=4,
                 output_format='parquet', seed=42, logger=Logger())
    df = _read_chunks(out)
    assert list(df.columns)[-1] == 'mt_h_max'
    ref = dict(zip(mt.index, get_h_max_ref(mt, x)))
    per = df.groupby('mt_id')['mt_h_max'].agg(['min', 'max'])
    assert set(per.index) == set(mt.index)                       # both meth chunks emitted
    assert float((per['max'] - per['min']).max()) <= 5e-5          # constant within CpG
    for cpg in per.index:
        assert abs(float(per.loc[cpg, 'max']) - float(ref[cpg])) <= 5e-5

def test_h_max_reservoir_carries_column(tmp_path):
    from tecpg.logger import Logger
    mt, gt, x = get_tiny_data()
    mt, gt = _rename_tiny(mt, gt)
    out = tmp_path / 'res'; out.mkdir()
    tecpg_mlr_qr(M=mt, G=gt, C=x, p_thresh=None, region='all', compute_influence=True,
                 output_dir=str(out), gene_loci_per_chunk=3, meth_loci_per_chunk=4,
                 reservoir_count=20, seed=42, logger=Logger())
    res = pd.read_csv(os.path.join(str(out), 'sample_reservoir.csv'))
    res['mt_id'] = res['mt_id'].astype(str)
    assert 'mt_h_max' in res.columns and res.columns[-1] == 'mt_h_max'
    assert len(res) == 20
    ref = dict(zip(mt.index, get_h_max_ref(mt, x)))
    for cpg, grp in res.groupby('mt_id'):
        assert np.allclose(grp['mt_h_max'].to_numpy(), float(ref[cpg]), atol=5e-5)
