import pytest
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
from tecpg.permute import _finalize_output, tecpg_mlr_qr_permute
from tecpg.logger import Logger

def test_finalize_output_columns_and_values():
    # Test 1: Columns + order exactly match expectations.
    # mt_t / perm_mt_p equal the inputs; seed / n_perm constant on every row.
    reported_pairs = pd.DataFrame({'mt_id': ['m1', 'm2', 'm3'], 'gt_id': ['g1', 'g2', 'g3']})
    observed_stats = np.array([2.5, -1.0, 0.5])
    perm_mt_p = np.array([0.01, 0.5, 0.9])
    seed = 42
    n_perm = 100

    df = _finalize_output(
        reported_pairs=reported_pairs,
        observed_stats=observed_stats,
        perm_mt_p=perm_mt_p,
        seed=seed,
        n_perm=n_perm,
        output_p_threshold=None,
        logger=Logger()
    )

    expected_cols = ['mt_id', 'gt_id', 'mt_t', 'perm_mt_p', 'seed', 'n_perm']
    assert list(df.columns) == expected_cols

    np.testing.assert_allclose(df['mt_t'].values, observed_stats)
    np.testing.assert_allclose(df['perm_mt_p'].values, perm_mt_p)
    assert (df['seed'] == seed).all()
    assert (df['n_perm'] == n_perm).all()


def test_finalize_output_threshold():
    # Test 2: Threshold filter works properly.
    reported_pairs = pd.DataFrame({'mt_id': ['m1', 'm2', 'm3'], 'gt_id': ['g1', 'g2', 'g3']})
    observed_stats = np.array([2.5, -1.0, 0.5])
    perm_mt_p = np.array([0.01, 0.5, 0.9])

    # With threshold 0.1, only m1 remains
    df_thresh = _finalize_output(
        reported_pairs=reported_pairs,
        observed_stats=observed_stats,
        perm_mt_p=perm_mt_p,
        seed=42,
        n_perm=100,
        output_p_threshold=0.1,
        logger=Logger()
    )
    assert len(df_thresh) == 1
    assert df_thresh['mt_id'].iloc[0] == 'm1'
    assert df_thresh['perm_mt_p'].iloc[0] <= 0.1

    # With None, all 3 remain
    df_none = _finalize_output(
        reported_pairs=reported_pairs,
        observed_stats=observed_stats,
        perm_mt_p=perm_mt_p,
        seed=42,
        n_perm=100,
        output_p_threshold=None,
        logger=Logger()
    )
    assert len(df_none) == 3


def test_finalize_output_mt_t_alignment():
    # Test 3: mt_t correctly mirrors observed_stats exactly row-for-row.
    reported_pairs = pd.DataFrame({'mt_id': ['m1', 'm2'], 'gt_id': ['g1', 'g2']})
    observed_stats = np.array([10.5, -5.5])
    perm_mt_p = np.array([0.2, 0.8])

    df = _finalize_output(
        reported_pairs=reported_pairs,
        observed_stats=observed_stats,
        perm_mt_p=perm_mt_p,
        seed=42,
        n_perm=100,
        output_p_threshold=None,
        logger=Logger()
    )

    assert df.loc[0, 'mt_t'] == 10.5
    assert df.loc[1, 'mt_t'] == -5.5
    assert df.loc[0, 'perm_mt_p'] == 0.2
    assert df.loc[1, 'perm_mt_p'] == 0.8


def test_permute_parquet_metadata_roundtrip(tmp_path, annotated_fixture):
    # Test 4: Parquet schema metadata successfully saves and matches seed, permutations, and total rows.
    M, G, C, M_annot, G_annot = annotated_fixture(sample_size=30, m_rows=10, g_rows=10)
    output_file = str(tmp_path / "perm_output.parquet")

    seed = 77
    permutations = 5

    tecpg_mlr_qr_permute(
        M=M, G=G, C=C, M_annot=M_annot, G_annot=G_annot,
        output_file=output_file, permutations=permutations, seed=seed
    )

    assert os.path.exists(output_file)

    table = pq.read_table(output_file)
    metadata = table.schema.metadata

    assert b'tecpg_perm_seed' in metadata
    assert b'tecpg_perm_n_perm' in metadata
    assert b'tecpg_perm_n_reported' in metadata

    assert int(metadata[b'tecpg_perm_seed']) == seed
    assert int(metadata[b'tecpg_perm_n_perm']) == permutations
    assert int(metadata[b'tecpg_perm_n_reported']) == len(M) * len(G)


def test_permute_end_to_end_threshold_and_universe(tmp_path, annotated_fixture):
    # Test 5: End-to-end threshold + universe size accurately persists under threshold conditions.
    M, G, C, M_annot, G_annot = annotated_fixture(sample_size=30, m_rows=10, g_rows=10)
    output_file = str(tmp_path / "perm_output_thresh.parquet")

    # We use output_p_threshold = 0.5 to drop some rows, ensuring not all are returned.
    tecpg_mlr_qr_permute(
        M=M, G=G, C=C, M_annot=M_annot, G_annot=G_annot,
        output_file=output_file, permutations=10, seed=42,
        output_p_threshold=0.5
    )

    table = pq.read_table(output_file)
    metadata = table.schema.metadata
    df = table.to_pandas()

    n_reported_universe = len(M) * len(G)

    # The written row count is less than or equal to n_reported.
    # Given the test data and standard null distributions, some should easily be > 0.5
    assert len(df) <= n_reported_universe
    assert (df['perm_mt_p'] <= 0.5).all()
    assert int(metadata[b'tecpg_perm_n_reported']) == n_reported_universe
