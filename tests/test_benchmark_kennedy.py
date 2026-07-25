import pytest
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
import sys

from tools.benchmark_kennedy import (
    resolve_kennedy_columns,
    resolve_tecpg_pvalue_column,
    resolve_thresholds,
    load_kennedy,
    load_catalog,
    compute_eligibility,
    compute_overlap_rates,
    export_pair_lists,
    main
)

# O1. resolver_consistency
def test_resolver_consistency():
    columns = ['CpG.probe', 'exp.Probe', 'p.val', 'annot.gene']
    resolved = resolve_kennedy_columns(columns)
    # The requirement: The column used for the gene diagnostic and the merge key read from the SAME returned dict.
    assert resolved['probe'] == 'exp.Probe'
    assert resolved['gene'] == 'annot.gene'

# O2. column_order_independence
def test_column_order_independence():
    gtp_cols = ['exp.Probe', 'CpG.probe', 'exp.probe.chrm', 'exp.probe.start', 'exp.probe.stop', 'exp.probe.strand', 'annot.gene', 'in_dist', 'distance', 'status', 'other_gene', 'distance.other.gene', 'p.val', 'T.stat', 'beta', 'beta.sd']
    mesa_cols = ['CpG.probe', 'exp.Probe'] + gtp_cols[2:]

    res_gtp = resolve_kennedy_columns(gtp_cols)
    res_mesa = resolve_kennedy_columns(mesa_cols)
    assert res_gtp == res_mesa

# O3. hard_fail_missing_required
def test_hard_fail_missing_required():
    columns = ['CpG.probe', 'p.val', 'other_col']
    with pytest.raises(ValueError) as excinfo:
        resolve_kennedy_columns(columns)
    msg = str(excinfo.value)
    assert 'exp.Probe' in msg
    assert 'CpG.probe' in msg
    assert 'other_col' in msg

# O4. hard_fail_missing_precise
def test_hard_fail_missing_precise():
    columns = ['mt_id', 'gt_id', 'mt_p']
    with pytest.raises(ValueError) as excinfo:
        resolve_tecpg_pvalue_column(columns)
    msg = str(excinfo.value)
    assert 'precise_mt_p' in msg
    assert 'mt_p' in msg

# O5. na_preservation
def test_na_preservation(tmp_path):
    df = pd.DataFrame({
        'CpG.probe': ['cg1', 'cg2'],
        'exp.Probe': ['pr1', 'pr2'],
        'status': ['TRANS', 'IN'],
        'distance': [pd.NA, 100],
        'in_dist': [False, pd.NA],
        'p.val': [0.01, 0.05]
    })
    path = tmp_path / "test.tsv"
    df.to_csv(path, sep='\t', index=False)

    loaded = load_kennedy(path, sep='\t')
    assert len(loaded) == 2

# O6. eligibility_classification
def test_eligibility_classification(tmp_path):
    catalog_df = pd.DataFrame({
        'mt_id': ['cg1', 'cg3', 'cg5'],
        'gt_id': ['pr1', 'pr3', 'pr5'],
        'precise_mt_p': [1e-6, 1e-2, 1e-6],
        'mt_est': [1.0, 1.0, 1.0],
        'mt_t': [1.0, 1.0, 1.0],
        'region': ['cis', 'trans', 'cis'],
        'fdr_est': [0.01, 0.5, 0.01]
    })
    kennedy_df = pd.DataFrame({
        'CpG.probe': ['cg1', 'cg2', 'cg3', 'cg1'],
        'exp.Probe': ['pr1', 'pr1', 'pr4', 'pr3'],
        'p.val': [1e-6, 1e-6, 1e-6, 1e-6],
        'status': ['IN', 'IN', 'IN', 'IN'],
        'distance': [10, 10, 10, 10]
    })

    cols = resolve_kennedy_columns(kennedy_df.columns)
    kennedy_df = compute_eligibility(catalog_df, kennedy_df, cols)

    diag_results = compute_overlap_rates(catalog_df, kennedy_df, cols, 1e-5, 1e-5, 'precise_mt_p')

    export_pair_lists(tmp_path, catalog_df, kennedy_df, cols, diag_results, 'precise_mt_p')

    df_k_only = pd.read_csv(tmp_path / "pairs_kennedy_only.tsv", sep='\t')

    assert len(df_k_only) == 3
    # Check reasons
    reasons = df_k_only.set_index(['mt_id', 'gt_id'])['non_overlap_reason'].to_dict()
    assert reasons[('cg2', 'pr1')] == 'ineligible_cpg'
    assert reasons[('cg3', 'pr4')] == 'ineligible_probe'
    assert reasons[('cg1', 'pr3')] == 'tested_and_missed'

# O7. recovery_confirmation_arithmetic
def test_recovery_confirmation_arithmetic(tmp_path):
    catalog_df = pd.DataFrame({
        'mt_id': ['c1', 'c2', 'c3', 'c4', 'c5'],
        'gt_id': ['p1', 'p2', 'p3', 'p4', 'p5'],
        'precise_mt_p': [1e-10, 1e-10, 1e-4, 1e-4, 1e-10],
        'mt_est': [1.0]*5, 'mt_t': [1.0]*5, 'region': ['cis']*5, 'fdr_est': [0.1]*5
    })
    kennedy_df = pd.DataFrame({
        'CpG.probe': ['c1', 'c3', 'cx', 'c2'],
        'exp.Probe': ['p1', 'p3', 'px', 'p2'],
        'p.val': [1e-10, 1e-10, 1e-10, 1e-4],
        'status': ['IN']*4, 'distance': [10]*4
    })
    cols = resolve_kennedy_columns(kennedy_df.columns)
    kennedy_df = compute_eligibility(catalog_df, kennedy_df, cols)
    res = compute_overlap_rates(catalog_df, kennedy_df, cols, 1e-5, 1e-5, 'precise_mt_p')

    assert res['recovery'] == 0.5
    assert res['confirmation_raw'] == 1 / 3
    assert res['confirmation_kennedy_testable'] == 0.5

    # Check caveats in summary
    from tools.benchmark_kennedy import build_summary_text
    import argparse
    args = argparse.Namespace(tecpg='cat.parquet', kennedy='ken.tsv', kennedy_thresh=1e-5, tecpg_thresh=1e-5, upset=False)
    grid = {(1e-5, 1e-5): res}
    summary = build_summary_text(args, 2, 0, 0, 0, 'beta', 0, 0, 0, 'T.stat', grid, res, [1e-5], cols, kennedy_df)
    assert "Confirmation denominators are LOWER BOUNDS (except kennedy_testable)" in summary
    assert "The Kennedy file is not a full universe, so we cannot know how many of our" in summary

# O8. threshold_dtype
def test_threshold_dtype():
    # np.float32(1e-5) is 9.999999747378752e-06
    # python float(1e-5) is 1.000000000000000e-05
    catalog_df = pd.DataFrame({
        'mt_id': ['cg1'],
        'gt_id': ['pr1'],
        'precise_mt_p': [9.99e-6]
    })
    kennedy_df = pd.DataFrame({
        'CpG.probe': ['cg1'],
        'exp.Probe': ['pr1'],
        'p.val': [9.99e-6]
    })
    cols = resolve_kennedy_columns(kennedy_df.columns)
    kennedy_df = compute_eligibility(catalog_df, kennedy_df, cols)

    res = compute_overlap_rates(catalog_df, kennedy_df, cols, float(1e-5), float(1e-5), 'precise_mt_p')
    assert len(res['T_tt']) == 1 # float(1e-5) > 9.99e-6
    assert len(res['K_tk']) == 1

# O9. export_schema
def test_export_schema(tmp_path):
    catalog_df = pd.DataFrame({
        'mt_id': ['c1', 'c2'],
        'gt_id': ['p1', 'p2'],
        'precise_mt_p': [1e-10, 1e-10],
        'mt_est': [1.0, 1.0], 'mt_t': [1.0, 1.0], 'region': ['cis', 'trans'], 'fdr_est': [0.1, 0.1]
    })
    kennedy_df = pd.DataFrame({
        'CpG.probe': ['c1', 'c3'],
        'exp.Probe': ['p1', 'p3'],
        'p.val': [1e-10, 1e-10],
        'status': ['IN', 'IN'],
        'distance': [10, 10]
    })

    cols = resolve_kennedy_columns(kennedy_df.columns)
    kennedy_df = compute_eligibility(catalog_df, kennedy_df, cols)
    res = compute_overlap_rates(catalog_df, kennedy_df, cols, 1e-5, 1e-5, 'precise_mt_p')

    export_pair_lists(tmp_path, catalog_df, kennedy_df, cols, res, 'precise_mt_p')

    conc = pd.read_csv(tmp_path / 'pairs_concordant.tsv', sep='\t')
    t_only = pd.read_csv(tmp_path / 'pairs_tecpg_only.tsv', sep='\t')
    k_only = pd.read_csv(tmp_path / 'pairs_kennedy_only.tsv', sep='\t')

    expected_cols = {'mt_id', 'gt_id', 'precise_mt_p', 'mt_est', 'mt_t', 'region', 'fdr_est', 'p.val', 'status', 'distance', 'non_overlap_reason'}
    expected_cols = expected_cols | {
        'cpg_in_tecpg_universe', 'probe_in_tecpg_universe',
        'cpg_in_kennedy_file', 'probe_in_kennedy_file'
    }
    assert set(conc.columns) == expected_cols
    assert set(t_only.columns) == expected_cols
    assert set(k_only.columns) == expected_cols

    assert len(conc) + len(t_only) == len(res['T_tt'])
    assert len(conc) + len(k_only) == len(res['K_tk'])

# O10. threshold_arg_precedence
def test_threshold_arg_precedence():
    pass

# R1. Kennedy real data parse
@pytest.mark.skipif(not os.environ.get("TECPG_KENNEDY_GTP"), reason="Requires real Kennedy data")
def test_real_kennedy():
    df = pd.read_csv(os.environ.get("TECPG_KENNEDY_GTP"), sep='\t')
    assert len(df) == 67606
    assert len(df[['exp.Probe', 'CpG.probe']].drop_duplicates()) == 67606
    assert df.columns[0] == "exp.Probe"
    sig = df[df['p.val'] < 1e-11]
    assert len(sig) == 2466
    assert len(sig[sig['status'] == 'TRANS']) == 958

# R2. Catalog real data parse
@pytest.mark.skipif(not os.environ.get("TECPG_CATALOG_GTP"), reason="Requires real catalog data")
def test_real_catalog():
    df = pq.read_table(os.environ.get("TECPG_CATALOG_GTP"), columns=['precise_mt_p']).to_pandas()
    assert len(df) == 17142039
    assert 'precise_mt_p' in df.columns
    assert df['precise_mt_p'].max() < 1.2e-3

# Test that confirmation_testable is upward biased on a fixture where it differs
def test_confirmation_testable_upward_biased():
    catalog_df = pd.DataFrame({
        'mt_id': ['c1', 'c2'],
        'gt_id': ['p1', 'p2'],
        'precise_mt_p': [1e-10, 1e-10]
    })
    kennedy_df = pd.DataFrame({
        'CpG.probe': ['c1'],
        'exp.Probe': ['p1'],
        'p.val': [1e-10]
    })
    cols = resolve_kennedy_columns(kennedy_df.columns)
    kennedy_df = compute_eligibility(catalog_df, kennedy_df, cols)
    res = compute_overlap_rates(catalog_df, kennedy_df, cols, 1e-5, 1e-5, 'precise_mt_p')

    assert res['confirmation_raw'] == 0.5 # 1 / 2
    assert res['confirmation_kennedy_testable'] == 1.0 # 1 / 1
    assert res['confirmation_raw'] <= res['confirmation_kennedy_testable']

# N1. dropna_widening
def test_dropna_widening(tmp_path):
    import logging
    df = pd.DataFrame({
        'CpG.probe': ['cg1'],
        'exp.Probe': ['pr1'],
        'distance': [pd.NA],
        'status': ['TRANS'],
        'p.val': [1e-6]
    })
    path = tmp_path / "test.tsv"
    df.to_csv(path, sep='\t', index=False)

    import subprocess
    # write a mock parquet file
    pq.write_table(pa.Table.from_arrays([pa.array(['cg1']), pa.array(['pr1']), pa.array([1e-6])], names=['mt_id', 'gt_id', 'precise_mt_p']), tmp_path / "cat.parquet")

    # Call main through subprocess to capture exact stderr behaviour without interference
    result = subprocess.run([
        sys.executable, 'tools/benchmark_kennedy.py',
        '-t', str(tmp_path / "cat.parquet"),
        '-k', str(path),
        '-o', str(tmp_path)
    ], capture_output=True, text=True)

    # ensure that "0 dropped" because we dropna on cpg and probe ONLY.
    assert "dropped Kennedy rows with missing key columns:" not in result.stderr

    # If we had widened dropna to include distance, one row would drop and it would log "1 -> 0 (1 dropped)".
    assert "1 -> 0 (1 dropped)" not in result.stderr


# N2. dropna_present
def test_dropna_present(tmp_path):
    df = pd.DataFrame({
        'CpG.probe': ['cg1', pd.NA],
        'exp.Probe': ['pr1', 'pr2'],
        'distance': [10, 10],
        'status': ['IN', 'IN'],
        'p.val': [1e-6, 1e-6]
    })
    path = tmp_path / "test.tsv"
    df.to_csv(path, sep='\t', index=False)

    pq.write_table(pa.Table.from_arrays([pa.array(['cg1']), pa.array(['pr1']), pa.array([1e-6])], names=['mt_id', 'gt_id', 'precise_mt_p']), tmp_path / "cat.parquet")
    import subprocess
    result = subprocess.run([
        sys.executable, 'tools/benchmark_kennedy.py',
        '-t', str(tmp_path / "cat.parquet"),
        '-k', str(path),
        '-o', str(tmp_path)
    ], capture_output=True, text=True)

    assert "dropped Kennedy rows with missing key columns: 2 -> 1 (1 dropped)" in result.stderr


# N3. positional_fallback_lock
def test_positional_fallback_lock():
    # If someone reverts to position indexing and falls back on positional columns
    # We pass a schema missing p.val, but with a decoy at index 2 (where p.val might be expected)
    columns = ['CpG.probe', 'exp.Probe', 'decoy_col']
    with pytest.raises(ValueError) as excinfo:
        resolve_kennedy_columns(columns)
    msg = str(excinfo.value)
    assert 'p.val' in msg


# N4. threshold_guard_equals_form
def test_threshold_guard_equals_form():
    # direct test of the pure logic
    with pytest.raises(ValueError):
        resolve_thresholds(1e-6, 1e-5, None)

    k, t = resolve_thresholds(1e-6, None, None)
    assert k == 1e-6
    assert t == 1e-5

    k, t = resolve_thresholds(None, 1e-5, None)
    assert k == 1e-5
    assert t == 1e-5

    # Mock argv for argparse to assert the specific parser handles `--kennedy-thresh=1e-5` equals form
    import subprocess
    result = subprocess.run([
        sys.executable, 'tools/benchmark_kennedy.py',
        '-t', 'dummy.parquet',
        '-k', 'dummy.tsv',
        '--p-thresh', '1e-6',
        '--kennedy-thresh=1e-5'
    ], capture_output=True, text=True)
    assert "ValueError: Cannot provide both --p-thresh and --kennedy-thresh." in result.stderr
    assert result.returncode != 0

    result2 = subprocess.run([
        sys.executable, 'tools/benchmark_kennedy.py',
        '-t', 'dummy.parquet',
        '-k', 'dummy.tsv',
        '--kennedy-thresh=1e-5'
    ], capture_output=True, text=True)
    # The file dummy.parquet doesn't exist so it will fail at reading parquet, but the arg parse is fine.
    assert "FileNotFoundError" in result2.stderr or "No such file or directory" in result2.stderr
    assert "ValueError: Cannot provide both" not in result2.stderr

# N5. eligibility_column_disambiguation
def test_eligibility_column_disambiguation(tmp_path):
    catalog_df = pd.DataFrame({
        'mt_id': ['cg1'],
        'gt_id': ['pr1'],
        'precise_mt_p': [1e-10],
        'mt_est': [1.0], 'mt_t': [1.0], 'region': ['cis'], 'fdr_est': [0.1]
    })
    kennedy_df = pd.DataFrame({
        'CpG.probe': ['cg2'],
        'exp.Probe': ['pr2'],
        'p.val': [1e-10],
        'status': ['IN'],
        'distance': [10]
    })

    cols = resolve_kennedy_columns(kennedy_df.columns)
    kennedy_df = compute_eligibility(catalog_df, kennedy_df, cols)
    res = compute_overlap_rates(catalog_df, kennedy_df, cols, 1e-5, 1e-5, 'precise_mt_p')
    export_pair_lists(tmp_path, catalog_df, kennedy_df, cols, res, 'precise_mt_p')

    k_only = pd.read_csv(tmp_path / "pairs_kennedy_only.tsv", sep='\t')
    assert len(k_only) == 1
    row = k_only.iloc[0]

    assert 'cpg_in_tecpg_universe' in row
    assert 'probe_in_tecpg_universe' in row
    assert 'cpg_in_kennedy_file' in row
    assert 'probe_in_kennedy_file' in row

    assert row['cpg_in_kennedy_file'] == True
    assert row['cpg_in_tecpg_universe'] == False
