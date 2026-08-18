import pandas as pd
import numpy as np
import pytest
from tecpg.permute import _finalize_output
import os
import subprocess
import tempfile
import sys
from unittest.mock import MagicMock

# -- PERMUTE TESTS --
def get_master():
    return pd.DataFrame({
        'mt_id': ['M1', 'M2'],
        'gt_id': ['G1', 'G2'],
        'mt_p': [0.1, 0.01],
        'mt_t': [2.0, 3.0]
    })

def get_reported_pairs():
    return pd.DataFrame({
        'mt_id': ['M1', 'M2'],
        'gt_id': ['G1', 'G2']
    })

def get_perm_mt_p():
    return np.array([0.05, 0.005])

def get_logger():
    return MagicMock()

def test_permute_stamps_namespaced_columns():
    df = _finalize_output(get_master(), get_reported_pairs(), get_perm_mt_p(), 123, 1000, None, get_logger())
    assert 'perm_seed' in df.columns
    assert 'perm_n_perm' in df.columns
    assert (df['perm_seed'] == 123).all()
    assert (df['perm_n_perm'] == 1000).all()

def test_permute_writes_no_bare_names():
    df = _finalize_output(get_master(), get_reported_pairs(), get_perm_mt_p(), 123, 1000, None, get_logger())
    assert 'seed' not in df.columns
    assert 'n_perm' not in df.columns

def test_permute_drops_legacy_bare_columns():
    master = get_master()
    master['seed'] = 999
    master['n_perm'] = 500
    df = _finalize_output(master, get_reported_pairs(), get_perm_mt_p(), 123, 1000, None, get_logger())
    assert 'seed' not in df.columns
    assert 'n_perm' not in df.columns
    assert not any(c.endswith('_x') or c.endswith('_y') for c in df.columns)

def test_permute_idempotent_over_own_output():
    out1 = _finalize_output(get_master(), get_reported_pairs(), get_perm_mt_p(), 123, 1000, None, get_logger())
    out2 = _finalize_output(out1.copy(), get_reported_pairs(), get_perm_mt_p(), 456, 2000, None, get_logger())
    assert list(out2.columns).count('perm_seed') == 1
    assert list(out2.columns).count('perm_n_perm') == 1
    assert (out2['perm_seed'] == 456).all()
    assert (out2['perm_n_perm'] == 2000).all()
    assert not any(c.endswith('_x') or c.endswith('_y') for c in out2.columns)

def test_permute_preserves_boot_seed():
    master = get_master()
    master['boot_seed'] = 'B'
    df = _finalize_output(master, get_reported_pairs(), get_perm_mt_p(), 123, 1000, None, get_logger())
    assert 'boot_seed' in df.columns
    assert (df['boot_seed'] == 'B').all()
    assert 'perm_seed' in df.columns
    assert (df['perm_seed'] == 123).all()

def run_bootstrap_end_to_end(tmp_path, master_parquet_fixture, seed_val, extra_master_col=None):
    master_path, M, G, C, M_annot, G_annot, master_df = master_parquet_fixture()

    if extra_master_col is not None:
        for k, v in extra_master_col.items():
            master_df[k] = v
        custom_master = os.path.join(tmp_path, 'custom_master.parquet')
        master_df.to_parquet(custom_master)
        master_path = custom_master

    pairs_file = os.path.join(tmp_path, 'pairs.csv')
    pairs = master_df[['mt_id', 'gt_id']].head(1)
    pairs.to_csv(pairs_file, index=False)

    M_file = os.path.join(tmp_path, 'M.csv')
    G_file = os.path.join(tmp_path, 'G.csv')
    C_file = os.path.join(tmp_path, 'C.csv')
    M.to_csv(M_file)
    G.to_csv(G_file)
    C.to_csv(C_file)

    out_dir = os.path.join(tmp_path, 'out')

    cmd = [
        'python3', '-m', 'tecpg', '-i', tmp_path, '-o', out_dir, 'run', 'mlr', '--mlr-method', 'qr_bootstrap',
        '--pairs-file', pairs_file, '--master-parquet', master_path, '--output-format', 'parquet',
        '--bootstrap-iterations', '10', '--seed', str(seed_val)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return pd.read_parquet(os.path.join(out_dir, 'bootstrap_merged.parquet'))

def test_bootstrap_stamps_boot_seed(tmp_path, master_parquet_fixture):
    res = run_bootstrap_end_to_end(tmp_path, master_parquet_fixture, 777)
    assert 'boot_seed' in res.columns
    assert 'seed' not in res.columns
    assert (res['boot_seed'] == 777).all()

def test_bootstrap_drops_legacy_seed(tmp_path, master_parquet_fixture):
    res = run_bootstrap_end_to_end(tmp_path, master_parquet_fixture, 777, extra_master_col={'seed': 999})
    assert 'boot_seed' in res.columns
    assert 'seed' not in res.columns
    assert (res['boot_seed'] == 777).all()


# -- READER TESTS --
def test_eval_reader_resolves_new_column():
    from tools.eval_permute import _resolve_permutation_parameters
    df = pd.DataFrame({'perm_n_perm': [1000]})
    args = MagicMock()
    args.n_null_pairs = None
    args.perm_resolution_floor = None

    n_perm, n_null_pairs, floor = _resolve_permutation_parameters(args, df, {})
    assert n_perm == 1000

def test_eval_reader_resolves_legacy_column():
    from tools.eval_permute import _resolve_permutation_parameters
    df = pd.DataFrame({'n_perm': [500]})
    args = MagicMock()
    args.n_null_pairs = None
    args.perm_resolution_floor = None

    n_perm, n_null_pairs, floor = _resolve_permutation_parameters(args, df, {})
    assert n_perm == 500

def test_eval_reader_prefers_metadata():
    from tools.eval_permute import _resolve_permutation_parameters
    df = pd.DataFrame({'perm_n_perm': [1000]})
    args = MagicMock()
    args.n_null_pairs = None
    args.perm_resolution_floor = None

    n_perm, n_null_pairs, floor = _resolve_permutation_parameters(args, df, {b'tecpg_perm_n_perm': b'2000'})
    assert n_perm == 2000

def test_qc_reader_resolves_both_names():
    from tools.permute_qc_report import build_permutation_resolution_module

    # 1. New name
    df_new = pd.DataFrame({'perm_n_perm': [3000, 3000], 'perm_mt_p': [0.1, 0.2]})
    rep1 = build_permutation_resolution_module({'metadata': {}}, df=df_new)
    assert '3,000' in rep1.table_html

    # 2. Legacy name
    df_old = pd.DataFrame({'n_perm': [4000, 4000], 'perm_mt_p': [0.1, 0.2]})
    rep2 = build_permutation_resolution_module({'metadata': {}}, df=df_old)
    assert '4,000' in rep2.table_html
