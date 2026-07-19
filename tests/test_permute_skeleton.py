import os
import pandas as pd
import pytest
from tecpg.permute import tecpg_mlr_qr_permute
from tecpg.test_data import generate_data


def test_permute_skeleton_end_to_end(tmp_path, master_parquet_fixture):
    # Set up small fixture using annotated_fixture
    master_parquet, M, G, C, M_annot, G_annot, master_df = master_parquet_fixture(
        sample_size=30,
        m_rows=10,
        g_rows=10,
    )

    output_file = str(tmp_path / "permutation_results.csv")

    # Call the newly created permutation function directly
    tecpg_mlr_qr_permute(
        master_parquet=master_parquet,
        M=M,
        G=G,
        C=C,
        M_annot=M_annot,
        G_annot=G_annot,
        output_file=output_file,
        permutations=10,
        seed=42,
    )

    # Assert output exists
    assert os.path.exists(output_file)

    # Read output
    df = pd.read_csv(output_file)

    # Assert correct columns (schema)
    expected_cols = {'mt_id', 'gt_id', 'mt_t', 'mt_p', 'perm_mt_p', 'seed', 'n_perm'}
    assert expected_cols.issubset(set(df.columns))
    assert not any(c.endswith('_x') or c.endswith('_y') for c in df.columns)

    # Assert row count = |M| x |G|
    expected_rows = len(M) * len(G)
    assert len(df) == expected_rows

    # Assert real scoring assertions
    assert (df['perm_mt_p'] > 0).all()
    assert (df['perm_mt_p'] <= 1.0).all()
    assert df['perm_mt_p'].nunique() > 1

    # Assert additional columns
    import numpy as np
    assert np.isfinite(df['mt_t']).all()
    assert (df['seed'] == 42).all()
    assert (df['n_perm'] == 10).all()
