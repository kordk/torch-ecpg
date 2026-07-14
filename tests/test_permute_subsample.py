import os
import pytest
import numpy as np
import pandas as pd
from tecpg.permute import _select_null_population, tecpg_mlr_qr_permute
from tecpg.test_data import generate_data
from tecpg.logger import Logger

class MockLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, template, *args):
        self.warnings.append((template, args))

    def info(self, template, *args):
        pass

def test_select_null_population_subsamples():
    M, G, C = generate_data(sample_size=30, m_rows=20, g_rows=20, annotation=False)
    logger = MockLogger()
    seed = 42

    # Normal valid subsample
    null_M, null_G = _select_null_population(M, G, C, None, None, 'all', None, None, None,
                                             subsample_mt_count=5, subsample_g_count=10, seed=seed, logger=logger)

    assert len(null_M) == 5
    assert len(null_G) == 10

    # Assert proper subset
    assert null_M.index.isin(M.index).all()
    assert null_G.index.isin(G.index).all()

    # Assert no duplicates
    assert len(null_M.index.unique()) == len(null_M)
    assert len(null_G.index.unique()) == len(null_G)

def test_select_null_population_determinism():
    M, G, C = generate_data(sample_size=30, m_rows=20, g_rows=20, annotation=False)
    logger = MockLogger()
    seed = 42

    null_M_1, null_G_1 = _select_null_population(M, G, C, None, None, 'all', None, None, None,
                                             subsample_mt_count=5, subsample_g_count=10, seed=seed, logger=logger)

    null_M_2, null_G_2 = _select_null_population(M, G, C, None, None, 'all', None, None, None,
                                             subsample_mt_count=5, subsample_g_count=10, seed=seed, logger=logger)

    pd.testing.assert_frame_equal(null_M_1, null_M_2)
    pd.testing.assert_frame_equal(null_G_1, null_G_2)

def test_select_null_population_warn_and_full():
    M, G, C = generate_data(sample_size=30, m_rows=20, g_rows=20, annotation=False)
    logger = MockLogger()
    seed = 42

    null_M, null_G = _select_null_population(M, G, C, None, None, 'all', None, None, None,
                                             subsample_mt_count=50, subsample_g_count=50, seed=seed, logger=logger)

    assert len(null_M) == 20
    assert len(null_G) == 20
    assert len(logger.warnings) == 2
    assert logger.warnings[0][0] == 'Requested mt_count {0} > available {1} for null population; using full.'
    assert logger.warnings[1][0] == 'Requested g_count {0} > available {1} for null population; using full.'
    assert logger.warnings[0][1] == (50, 20)
    assert logger.warnings[1][1] == (50, 20)

def test_select_null_population_zero_count_raises():
    M, G, C = generate_data(sample_size=30, m_rows=20, g_rows=20, annotation=False)
    logger = MockLogger()
    seed = 42

    with pytest.raises(ValueError, match="qr_permute subsample mt_count must be positive; got 0"):
        _select_null_population(M, G, C, None, None, 'all', None, None, None,
                                subsample_mt_count=0, subsample_g_count=10, seed=seed, logger=logger)

    with pytest.raises(ValueError, match="qr_permute subsample g_count must be positive; got -5"):
        _select_null_population(M, G, C, None, None, 'all', None, None, None,
                                subsample_mt_count=10, subsample_g_count=-5, seed=seed, logger=logger)

def test_permute_with_subsample(tmp_path):
    M, G, C, M_annot, G_annot = generate_data(sample_size=30, m_rows=10, g_rows=10, annotation=True)
    M_annot = M_annot.set_index("name")[["chrom", "chromStart"]]
    G_annot = G_annot.set_index("name")[["chrom", "chromStart", "strand"]]
    G_annot["strand"] = G_annot["strand"].map({"+": 1, "-": -1})
    output_file = str(tmp_path / "permutation_results.csv")

    tecpg_mlr_qr_permute(
        M=M,
        G=G,
        M_annot=M_annot,
        G_annot=G_annot,
        C=C,
        output_file=output_file,
        permutations=10,
        seed=42,
        subsample_mt_count=5,
        subsample_g_count=5
    )

    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)

    expected_cols = ['mt_id', 'gt_id', 'mt_t', 'perm_mt_p', 'seed', 'n_perm']
    assert list(df.columns) == expected_cols

    expected_rows = len(M) * len(G)
    assert len(df) == expected_rows

    # assert (df['perm_mt_p'] == 0.5).all()
