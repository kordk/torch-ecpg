import pytest
import os
import pandas as pd
import numpy as np
from tecpg.permute import tecpg_mlr_qr_permute, _verify_master_consistency
from tecpg.logger import Logger

def hand_rolled_ols_oracle(M, G, C, reported_pairs):
    """
    Independent OLS t-statistic oracle for a set of pairs.
    Must not share any solver code with tecpg.
    Ensures df = n - C.shape[1] - 2 exactly matches tecpg behavior.
    """
    n = len(C)
    df = n - C.shape[1] - 2
    C_mat = C.values

    results = []
    for _, row in reported_pairs.iterrows():
        m_id, g_id = row['mt_id'], row['gt_id']
        m_vec = M.loc[m_id].values
        g_vec = G.loc[g_id].values

        # X = [ones, M_m, C]
        X = np.column_stack([np.ones(n), m_vec, C_mat])

        # B = (X'X)^-1 X'Y
        # Use lstsq for robustness
        beta, rss, _, _ = np.linalg.lstsq(X, g_vec, rcond=None)

        # Ensure we compute RSS manually if lstsq returns empty (e.g. perfect fit)
        if len(rss) == 0:
            residuals = g_vec - X @ beta
            rss_val = np.sum(residuals ** 2)
        else:
            rss_val = rss[0]

        sigma_sq = rss_val / df

        # Var(B) = sigma^2 * (X'X)^-1
        # Inverse of X'X
        XtX_inv = np.linalg.inv(X.T @ X)

        # Standard error of the methylation coefficient (index 1)
        se_m = np.sqrt(sigma_sq * XtX_inv[1, 1])

        # t-statistic
        t_stat = beta[1] / se_m
        results.append(t_stat)

    return np.array(results)

def test_master_parquet_required(master_parquet_fixture):
    """Fail-closed on missing master_parquet."""
    master_parquet, M, G, C, M_annot, G_annot, master_df = master_parquet_fixture(sample_size=30, m_rows=5, g_rows=5, seed=42)

    with pytest.raises(ValueError, match="requires --master-parquet"):
        tecpg_mlr_qr_permute(M=M, G=G, C=C, M_annot=M_annot, G_annot=G_annot)


def test_three_way_equivalence_oracle(tmp_path, master_parquet_fixture):
    """6a. Three-way equivalence oracle."""
    master_parquet, M, G, C, M_annot, G_annot, master_df = master_parquet_fixture(sample_size=30, m_rows=5, g_rows=5, seed=42)

    # Run permute
    output_file = str(tmp_path / 'out.csv')
    tecpg_mlr_qr_permute(
        master_parquet=master_parquet, M=M, G=G, C=C,
        M_annot=M_annot, G_annot=G_annot, permutations=5, seed=42,
        output_file=output_file
    )

    # 1. Read back output (observed_t via read path)
    df = pd.read_csv(output_file)
    t_read = df['mt_t'].values

    # 2. Recompute via tecpg internals (_compute_observed_statistic)
    from tecpg.permute import _compute_observed_statistic
    from tecpg.config import get_device
    reported_pairs = df[['mt_id', 'gt_id']].copy()
    t_recompute = _compute_observed_statistic(M, G, C, reported_pairs, Logger(), device=get_device())
    t_recompute = np.asarray(t_recompute, dtype=np.float64)

    # 3. Independent per-pair OLS
    t_ols = hand_rolled_ols_oracle(M, G, C, reported_pairs)

    np.testing.assert_allclose(t_read, t_recompute, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(t_read, t_ols, rtol=1e-4, atol=1e-4)

def test_consistency_guard_passes(master_parquet_fixture):
    """6b. Consistency guard - pass."""
    master_parquet, M, G, C, M_annot, G_annot, master_df = master_parquet_fixture(sample_size=30, m_rows=10, g_rows=10)

    logger = Logger()
    # Should run cleanly without raising
    _verify_master_consistency(M, G, C, master_df, None, logger, seed=42)

def test_consistency_guard_fail_design(master_parquet_fixture):
    """6b. Consistency guard - fail-design."""
    master_parquet, M, G, C, M_annot, G_annot, master_df = master_parquet_fixture(sample_size=30, m_rows=10, g_rows=10)

    # Perturb C globally: add a dummy covariate to change design/df
    C_perturbed = C.copy()
    C_perturbed['dummy'] = np.random.rand(len(C))

    with pytest.raises(ValueError, match="master mt_t is inconsistent with the provided M/G/C"):
        _verify_master_consistency(M, G, C_perturbed, master_df, None, Logger(), seed=42)

def test_consistency_guard_fail_data(master_parquet_fixture):
    """6b. Consistency guard - fail-data."""
    master_parquet, M, G, C, M_annot, G_annot, master_df = master_parquet_fixture(sample_size=30, m_rows=10, g_rows=10)

    # Use disjoint M/G IDs
    M_disjoint = M.copy()
    M_disjoint.index = [f"disjoint_m{i}" for i in range(len(M))]

    with pytest.raises(ValueError, match="master consistency check failed"):
        _verify_master_consistency(M_disjoint, G, C, master_df, None, Logger(), seed=42)

def test_pairs_file_subset(tmp_path, master_parquet_fixture):
    """6c. --pairs-file subset."""
    master_parquet, M, G, C, M_annot, G_annot, master_df = master_parquet_fixture(sample_size=30, m_rows=10, g_rows=10)

    # Create pairs file with only 2 pairs
    pairs_file = str(tmp_path / 'pairs.csv')
    subset = master_df.head(2)[['mt_id', 'gt_id']].copy()
    subset.to_csv(pairs_file, index=False)

    output_file = str(tmp_path / 'out.csv')
    tecpg_mlr_qr_permute(
        master_parquet=master_parquet, pairs_file=pairs_file,
        M=M, G=G, C=C, M_annot=M_annot, G_annot=G_annot,
        permutations=5, seed=42, output_file=output_file
    )

    df = pd.read_csv(output_file)
    assert len(df) == len(master_df)

    # Only the 2 pairs should have perm_mt_p
    scored = df.dropna(subset=['perm_mt_p'])
    assert len(scored) == 2

    # Duplicate pairs error
    dup_pairs = pd.concat([subset, subset.head(1)])
    dup_pairs_file = str(tmp_path / 'dup_pairs.csv')
    dup_pairs.to_csv(dup_pairs_file, index=False)

    with pytest.raises(ValueError, match="--pairs-file contains duplicate"):
        tecpg_mlr_qr_permute(
            master_parquet=master_parquet, pairs_file=dup_pairs_file,
            M=M, G=G, C=C, M_annot=M_annot, G_annot=G_annot
        )

    # Absent pairs error
    absent_pairs = subset.copy()
    absent_pairs.iloc[0, 0] = 'absent_m'
    absent_pairs_file = str(tmp_path / 'absent_pairs.csv')
    absent_pairs.to_csv(absent_pairs_file, index=False)

    with pytest.raises(ValueError, match="--pairs-file contains pairs absent from --master-parquet"):
        tecpg_mlr_qr_permute(
            master_parquet=master_parquet, pairs_file=absent_pairs_file,
            M=M, G=G, C=C, M_annot=M_annot, G_annot=G_annot
        )

def test_additive_merge(tmp_path, master_parquet_fixture):
    """6d. Additive merge."""
    master_parquet, M, G, C, M_annot, G_annot, master_df = master_parquet_fixture(sample_size=30, m_rows=5, g_rows=5)

    output_file = str(tmp_path / 'out.csv')
    tecpg_mlr_qr_permute(
        master_parquet=master_parquet, M=M, G=G, C=C,
        M_annot=M_annot, G_annot=G_annot, permutations=5, seed=42,
        output_file=output_file
    )

    df = pd.read_csv(output_file)

    # Row count matches master
    assert len(df) == len(master_df)

    # Invariance on mt_p
    # Align by id
    merged = df.merge(master_df, on=['mt_id', 'gt_id'], suffixes=('', '_master'))
    np.testing.assert_allclose(merged['mt_p'].values, merged['mt_p_master'].values)

    assert 'perm_mt_p' in df.columns
    assert not any(c.endswith('_x') or c.endswith('_y') for c in df.columns)
