import numpy as np
import scipy.stats
import pytest
import os
import pandas as pd
from tecpg.permute import _accumulate_null, tecpg_mlr_qr_permute, T_MAX, N_BINS, TOPK_CAPACITY
from tecpg.logger import Logger
from tecpg.test_data import generate_data

def test_accumulate_memory_boundedness():
    logger = Logger()
    P = 100

    # First test: P calls
    acc1 = None
    rng = np.random.default_rng(1)
    for _ in range(P):
        stats = rng.normal(size=500)
        acc1 = _accumulate_null(stats, acc1, logger)

    hist_size1 = acc1['hist_counts'].nbytes
    topk_len1 = len(acc1['topk_values'])
    bins_size1 = acc1['bin_edges'].nbytes

    assert acc1['total_count'] == P * 500
    assert topk_len1 <= TOPK_CAPACITY

    # Second test: 10*P calls
    acc2 = None
    rng = np.random.default_rng(2)
    for _ in range(10 * P):
        stats = rng.normal(size=500)
        acc2 = _accumulate_null(stats, acc2, logger)

    hist_size2 = acc2['hist_counts'].nbytes
    topk_len2 = len(acc2['topk_values'])
    bins_size2 = acc2['bin_edges'].nbytes

    assert acc2['total_count'] == 10 * P * 500
    assert topk_len2 <= TOPK_CAPACITY

    # Footprint should be invariant
    assert hist_size1 == hist_size2
    assert bins_size1 == bins_size2
    assert topk_len1 == topk_len2  # Since both easily exceed TOPK_CAPACITY

def test_accumulate_histogram_correctness():
    logger = Logger()
    rng = np.random.default_rng(42)
    stats = rng.normal(scale=3.0, size=20000)

    acc = _accumulate_null(stats, None, logger)

    # Total count match
    assert acc['total_count'] == acc['hist_counts'].sum() + acc['overflow_count']
    assert acc['total_count'] == 20000

    # Verify per-bin counts match direct numpy histogram on abs(stats)
    a = np.abs(stats)
    expected_counts, _ = np.histogram(a, bins=acc['bin_edges'])
    np.testing.assert_array_equal(acc['hist_counts'], expected_counts)

    # Verify overflow
    expected_overflow = (a > T_MAX).sum()
    assert acc['overflow_count'] == expected_overflow

    # Boundary regression for T_MAX exact value
    acc_edge = _accumulate_null(np.array([T_MAX, 5.0]), None, logger)
    assert acc_edge['hist_counts'].sum() + acc_edge['overflow_count'] == acc_edge['total_count']

def test_accumulate_topk_correctness():
    logger = Logger()
    rng = np.random.default_rng(42)

    acc = None
    all_stats = []

    # Push multiple batches
    for _ in range(5):
        stats = rng.normal(scale=2.0, size=5000)
        all_stats.append(stats)
        acc = _accumulate_null(stats, acc, logger)

    a = np.abs(np.concatenate(all_stats))

    # Brute-force sort
    expected_topk = np.sort(a)[-TOPK_CAPACITY:]

    actual_topk = np.sort(acc['topk_values'])

    np.testing.assert_allclose(actual_topk, expected_topk)
    assert len(actual_topk) <= TOPK_CAPACITY

def test_accumulate_determinism(tmp_path, master_parquet_fixture):
    master_parquet, M, G, C, M_annot, G_annot, master_df = master_parquet_fixture(sample_size=30, m_rows=10, g_rows=10)

    # Run 1
    out1 = str(tmp_path / "out1.csv")
    tecpg_mlr_qr_permute(master_parquet=master_parquet, M=M, G=G, C=C, M_annot=M_annot, G_annot=G_annot, permutations=10, seed=123, output_file=out1)

    # Run 2
    out2 = str(tmp_path / "out2.csv")
    tecpg_mlr_qr_permute(master_parquet=master_parquet, M=M, G=G, C=C, M_annot=M_annot, G_annot=G_annot, permutations=10, seed=123, output_file=out2)

    df1 = pd.read_csv(out1)
    df2 = pd.read_csv(out2)
    pd.testing.assert_frame_equal(df1, df2)

def test_accumulate_null_pair_stratification(tmp_path, monkeypatch, master_parquet_fixture):
    master_parquet, M, G, C, M_annot, G_annot, master_df = master_parquet_fixture(sample_size=30, m_rows=15, g_rows=15, seed=42)

    # We will patch _accumulate_null to capture the final accumulator
    captured_acc = []

    from tecpg import permute
    original_accumulate_null = permute._accumulate_null

    def mocked_accumulate_null(perm_stats, acc, logger):
        new_acc = original_accumulate_null(perm_stats, acc, logger)
        captured_acc.append(new_acc)
        return new_acc

    monkeypatch.setattr(permute, '_accumulate_null', mocked_accumulate_null)

    out = str(tmp_path / "out.csv")
    tecpg_mlr_qr_permute(master_parquet=master_parquet, M=M, G=G, C=C, M_annot=M_annot, G_annot=G_annot, permutations=10, seed=123, output_file=out)

    final_acc = captured_acc[-1]

    # Calculate expected trans pairs manually
    # null_pairs starts as cross product
    m_chrom = M_annot['chrom'].to_numpy()
    g_chrom = G_annot['chrom'].to_numpy()

    # Number of trans pairs
    expected_trans_pairs = 0
    for mc in m_chrom:
        for gc in g_chrom:
            if mc != gc:
                expected_trans_pairs += 1

    expected_total_count = 10 * expected_trans_pairs
    assert final_acc['total_count'] == expected_total_count

def test_accumulate_calibration_sanity(tmp_path, monkeypatch, master_parquet_fixture):
    master_parquet, M, G, C, M_annot, G_annot, master_df = master_parquet_fixture(sample_size=120, m_rows=25, g_rows=25, seed=42)


    captured_acc = []
    from tecpg import permute
    original_accumulate_null = permute._accumulate_null

    def mocked_accumulate_null(perm_stats, acc, logger):
        new_acc = original_accumulate_null(perm_stats, acc, logger)
        captured_acc.append(new_acc)
        return new_acc

    monkeypatch.setattr(permute, '_accumulate_null', mocked_accumulate_null)

    out = str(tmp_path / "out.csv")
    tecpg_mlr_qr_permute(master_parquet=master_parquet, M=M, G=G, C=C, M_annot=M_annot, G_annot=G_annot, permutations=60, seed=123, output_file=out)

    final_acc = captured_acc[-1]

    df = 120 - (C.shape[1] + 2) # n_samples - k

    # Check frac(|t| >= 1.96)
    threshold = 1.96
    bin_idx = np.searchsorted(final_acc['bin_edges'], threshold)
    count_above = final_acc['hist_counts'][bin_idx:].sum() + final_acc['overflow_count']
    frac_above = count_above / final_acc['total_count']

    expected_frac = 2 * scipy.stats.t.sf(threshold, df)

    assert abs(frac_above - expected_frac) < 0.01

    # Check frac(|t| >= 2.576)
    threshold = 2.576
    bin_idx = np.searchsorted(final_acc['bin_edges'], threshold)
    count_above = final_acc['hist_counts'][bin_idx:].sum() + final_acc['overflow_count']
    frac_above = count_above / final_acc['total_count']

    expected_frac = 2 * scipy.stats.t.sf(threshold, df)

    assert abs(frac_above - expected_frac) < 0.005
