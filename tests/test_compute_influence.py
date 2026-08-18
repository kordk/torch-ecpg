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

def test_influence_with_ig_deep_raises_usage_error():
    runner = CliRunner()
    from tecpg.cli import mlr
    from tecpg.logger import Logger
    import os; open('M.csv', 'w').write('1\n2'); open('G.csv', 'w').write('1\n2'); open('C.csv', 'w').write('1\n2')
    result = runner.invoke(mlr, ['--compute-influence', '--compute-ig-deep', '--p-thresh', '0.05', '--mlr-method', 'qr'], obj={'logger': Logger(), 'data': {'root_path': '.', 'input_dir': '.', 'output_dir': '.', 'meth_file': 'M.csv', 'gene_file': 'G.csv', 'covar_file': 'C.csv'}})
    assert result.exit_code != 0
    assert 'Cannot use --compute-influence with --compute-ig-deep' in result.output
