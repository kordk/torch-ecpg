import pytest
import numpy as np
import pandas as pd
import scipy.stats
import torch
import sys

from tecpg.permute import _compute_observed_statistic
from tecpg.logger import Logger

# Simple deterministic setup
@pytest.fixture
def chunking_data():
    np.random.seed(42)
    # 12 methylation loci, 7 gene loci, 3 covariates, 20 samples
    M_loci = [f"cg{i}" for i in range(12)]
    G_loci = [f"gene{i}" for i in range(7)]
    samples = [f"s{i}" for i in range(20)]

    M = pd.DataFrame(np.random.randn(12, 20), index=M_loci, columns=samples)
    G = pd.DataFrame(np.random.randn(7, 20), index=G_loci, columns=samples)
    C = pd.DataFrame(np.random.randn(20, 3), index=samples, columns=['c1', 'c2', 'c3'])

    # Full cross-product (84 pairs)
    pairs = pd.MultiIndex.from_product([M_loci, G_loci], names=['mt_id', 'gt_id']).to_frame(index=False)

    return M, G, C, pairs


def test_chunking_exact_match_and_oracle(chunking_data, capsys):
    M, G, C, pairs = chunking_data
    logger = Logger()

    # 1. Single-shot run
    t_single = _compute_observed_statistic(M, G, C, pairs, logger, pair_chunk_size=10_000)

    # 2. Chunked run (P=84, chunk=5 -> 17 chunks, remainder 4)
    t_chunked = _compute_observed_statistic(M, G, C, pairs, logger, pair_chunk_size=5)

    # Capture stdout to assert loop execution via Logger output
    out = capsys.readouterr().out
    assert 'chunks_executed=17' in out, f"expected 17 loop iterations, got:\n{out}"

    # 2.b. Progress logging run
    t_prog = _compute_observed_statistic(M, G, C, pairs, logger, pair_chunk_size=5, progress_label='test')
    out2 = capsys.readouterr().out
    assert 'test: chunk' in out2, f"expected progress lines, got:\n{out2}"
    assert 'chunks_executed=17' in out2, f"expected 17 loop iterations in progress run, got:\n{out2}"

    # Check shape & exact match
    assert len(t_single) == len(pairs)
    assert len(t_chunked) == len(pairs)
    assert np.allclose(t_single, t_chunked, atol=1e-6), "Chunked output differs from single-shot output"

    # Check equal length and elementwise equality within float limits
    # The requirement is that slices processed and written in ascending order must result in exact identical outputs.

    # 3. Oracle vs scipy.stats.linregress equivalent
    # We want to check 2-3 pairs manually via OLS
    for idx in [0, 42, 83]:
        m_id = pairs.iloc[idx]['mt_id']
        g_id = pairs.iloc[idx]['gt_id']

        y = G.loc[g_id].values
        x_m = M.loc[m_id].values
        c_vals = C.values

        # OLS design: Intercept, Methylation, Covariates
        X = np.column_stack((np.ones(len(y)), x_m, c_vals))

        # QR solve or direct lstsq
        coef, rss, rank, s = np.linalg.lstsq(X, y, rcond=None)

        # Calculate standard error
        df = len(y) - X.shape[1]
        sigma2 = rss[0] / df
        cov_matrix = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov_matrix))

        t_stat = coef[1] / se[1]

        assert np.isclose(t_chunked[idx], t_stat, atol=1e-5), f"Oracle check failed at pair {idx}"
