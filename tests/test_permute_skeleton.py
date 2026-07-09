import os
import pandas as pd
import pytest
from tecpg.permute import tecpg_mlr_qr_permute
from tecpg.test_data import generate_data


def test_permute_skeleton_end_to_end(tmp_path):
    # Set up small fixture using tecpg.test_data.generate_data
    # matching standard harness style
    M, G, C = generate_data(
        sample_size=30,
        m_rows=10,
        g_rows=10,
        annotation=False,
    )

    output_file = str(tmp_path / "permutation_results.csv")

    # Call the newly created permutation function directly
    tecpg_mlr_qr_permute(
        M=M,
        G=G,
        C=C,
        output_file=output_file,
        permutations=10,
        seed=42,
    )

    # Assert output exists
    assert os.path.exists(output_file)

    # Read output
    df = pd.read_csv(output_file)

    # Assert correct columns (schema)
    expected_cols = ['mt_id', 'gt_id', 'perm_mt_p']
    assert list(df.columns) == expected_cols

    # Assert row count = |M| x |G|
    expected_rows = len(M) * len(G)
    assert len(df) == expected_rows

    # Assert all perm_mt_p == 0.5
    assert (df['perm_mt_p'] == 0.5).all()
