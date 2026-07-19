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
    master_df = pd.DataFrame({
        'mt_id': ['m1', 'm2', 'm3'],
        'gt_id': ['g1', 'g2', 'g3'],
        'mt_t': [2.5, -1.0, 0.5],
        'mt_p': [0.1, 0.2, 0.3]
    })
    reported_pairs = pd.DataFrame({'mt_id': ['m1', 'm2', 'm3'], 'gt_id': ['g1', 'g2', 'g3']})
    perm_mt_p = np.array([0.01, 0.5, 0.9])
    seed = 42
    n_perm = 100

    df = _finalize_output(
        master_df=master_df,
        reported_pairs=reported_pairs,
        perm_mt_p=perm_mt_p,
        seed=seed,
        n_perm=n_perm,
        output_p_threshold=None,
        logger=Logger()
    )

    expected_cols = {'mt_id', 'gt_id', 'mt_t', 'mt_p', 'perm_mt_p', 'seed', 'n_perm'}
    assert expected_cols.issubset(set(df.columns))
    assert not any(c.endswith('_x') or c.endswith('_y') for c in df.columns)

    np.testing.assert_allclose(df['mt_t'].values, master_df['mt_t'].values)
    np.testing.assert_allclose(df['mt_p'].values, master_df['mt_p'].values)
    np.testing.assert_allclose(df['perm_mt_p'].values, perm_mt_p)
    assert (df['seed'] == seed).all()
    assert (df['n_perm'] == n_perm).all()


def test_finalize_output_threshold():
    # Test 2: Threshold filter works properly.
    master_df = pd.DataFrame({
        'mt_id': ['m1', 'm2', 'm3'],
        'gt_id': ['g1', 'g2', 'g3'],
        'mt_t': [2.5, -1.0, 0.5],
        'mt_p': [0.1, 0.2, 0.3]
    })
    reported_pairs = pd.DataFrame({'mt_id': ['m1', 'm2', 'm3'], 'gt_id': ['g1', 'g2', 'g3']})
    perm_mt_p = np.array([0.01, 0.5, 0.9])

    # With threshold 0.1, only m1 remains in `res` before merge
    df_thresh = _finalize_output(
        master_df=master_df,
        reported_pairs=reported_pairs,
        perm_mt_p=perm_mt_p,
        seed=42,
        n_perm=100,
        output_p_threshold=0.1,
        logger=Logger()
    )
    assert len(df_thresh) == 3 # Master rows are untouched
    # m1 is scored and thresholded in res, so it keeps its score
    assert df_thresh.loc[df_thresh['mt_id'] == 'm1', 'perm_mt_p'].iloc[0] == 0.01
    # m2, m3 dropped by threshold, so they get merged as NaNs
    assert pd.isna(df_thresh.loc[df_thresh['mt_id'] == 'm2', 'perm_mt_p'].iloc[0])
    assert pd.isna(df_thresh.loc[df_thresh['mt_id'] == 'm3', 'perm_mt_p'].iloc[0])


def test_finalize_output_mt_t_alignment():
    # Test 3: mt_t correctly mirrors observed_stats exactly row-for-row.
    master_df = pd.DataFrame({
        'mt_id': ['m1', 'm2'],
        'gt_id': ['g1', 'g2'],
        'mt_t': [10.5, -5.5]
    })
    reported_pairs = pd.DataFrame({'mt_id': ['m1', 'm2'], 'gt_id': ['g1', 'g2']})
    perm_mt_p = np.array([0.2, 0.8])

    df = _finalize_output(
        master_df=master_df,
        reported_pairs=reported_pairs,
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


def test_permute_parquet_metadata_roundtrip(tmp_path, master_parquet_fixture):
    # Test 4: Parquet schema metadata successfully saves and matches seed, permutations, and total rows.
    master_parquet, M, G, C, M_annot, G_annot, master_df = master_parquet_fixture(sample_size=30, m_rows=10, g_rows=10)
    output_file = str(tmp_path / "perm_output.parquet")

    seed = 77
    permutations = 5

    tecpg_mlr_qr_permute(
        master_parquet=master_parquet,
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


def test_permute_end_to_end_threshold_and_universe(tmp_path, master_parquet_fixture):
    # Test 5: End-to-end threshold + universe size accurately persists under threshold conditions.
    master_parquet, M, G, C, M_annot, G_annot, master_df = master_parquet_fixture(sample_size=30, m_rows=10, g_rows=10)
    output_file = str(tmp_path / "perm_output_thresh.parquet")

    # We use output_p_threshold = 0.5 to drop some rows, ensuring not all are returned.
    tecpg_mlr_qr_permute(
        master_parquet=master_parquet,
        M=M, G=G, C=C, M_annot=M_annot, G_annot=G_annot,
        output_file=output_file, permutations=10, seed=42,
        output_p_threshold=0.5
    )

    table = pq.read_table(output_file)
    metadata = table.schema.metadata
    df = table.to_pandas()

    n_reported_universe = len(M) * len(G)

    # The merge on master always yields n_reported_universe rows.
    # The thresholding drops scores, so un-scored rows have NaN perm_mt_p.
    assert len(df) == n_reported_universe

    # Rows that *do* have a perm_mt_p should pass the threshold.
    scored = df.dropna(subset=['perm_mt_p'])
    assert (scored['perm_mt_p'] <= 0.5).all()
    # At least some should be dropped by the threshold and become NaN.
    assert len(scored) < len(df)

    assert int(metadata[b'tecpg_perm_n_reported']) == n_reported_universe
