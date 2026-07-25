import os
import pandas as pd
import pytest
import pyarrow.parquet as pq
from tecpg.permute import tecpg_mlr_qr_permute

def test_metadata_null_pairs(tmp_path, master_parquet_fixture):
    master_parquet, M, G, C, M_annot, G_annot, master_df = master_parquet_fixture(
        sample_size=30, m_rows=10, g_rows=10
    )
    output_file = str(tmp_path / "output.parquet")

    tecpg_mlr_qr_permute(
        master_parquet=master_parquet,
        M=M,
        G=G,
        C=C,
        M_annot=M_annot,
        G_annot=G_annot,
        output_file=output_file,
        output_format='parquet',
        permutations=10,
        seed=42,
    )

    table = pq.read_table(output_file)
    meta = table.schema.metadata

    assert b'tecpg_perm_n_null_pairs' in meta
    assert int(meta[b'tecpg_perm_n_null_pairs']) > 0
    assert b'tecpg_perm_seed' in meta
    assert meta[b'tecpg_perm_seed'] == b'42'
    assert b'tecpg_perm_n_perm' in meta
    assert meta[b'tecpg_perm_n_perm'] == b'10'
    assert b'tecpg_perm_n_reported' in meta

def test_csv_branch_unaffected(tmp_path, master_parquet_fixture):
    master_parquet, M, G, C, M_annot, G_annot, master_df = master_parquet_fixture(
        sample_size=30, m_rows=10, g_rows=10
    )
    output_file = str(tmp_path / "output.csv")

    tecpg_mlr_qr_permute(
        master_parquet=master_parquet,
        M=M,
        G=G,
        C=C,
        M_annot=M_annot,
        G_annot=G_annot,
        output_file=output_file,
        output_format='csv',
        permutations=10,
        seed=42,
    )

    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)
    assert len(df) > 0
