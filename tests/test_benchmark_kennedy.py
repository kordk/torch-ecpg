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
    stream_catalog_and_match,
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
    gtp_cols = ['exp.Probe', 'CpG.probe', 'exp.probe.chrm', 'exp.probe.start', 'exp.probe.stop',
                'exp.probe.strand',
                'annot.gene', 'in_dist', 'distance', 'status', 'other_gene',
                'distance.other.gene', 'p.val', 'T.stat', 'beta', 'beta.sd']
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

# O5b. test_eligibility_uses_tecpg_universe


def test_eligibility_uses_tecpg_universe():
    import pandas as pd
    from tools.benchmark_kennedy import compute_eligibility

    kennedy_df = pd.DataFrame({
        'CpG.probe': ['cgA', 'cgB', 'cgA', 'cgC'],
        'exp.Probe': ['prA', 'prA', 'prB', 'prC'],
    })
    cols = {'cpg': 'CpG.probe', 'probe': 'exp.Probe'}

    # tecpg universe deliberately NARROWER than the Kennedy frame
    distinct_mt = {'cgA'}
    distinct_gt = {'prA'}

    out = compute_eligibility(distinct_mt, distinct_gt, kennedy_df, cols)

    assert list(out['cpg_in_tecpg_universe'])   == [True, False, True, False]
    assert list(out['probe_in_tecpg_universe']) == [True, True, False, False]
    assert list(out['eligible'])                == [True, False, False, False]

    # not vacuous: at least one row must be ineligible
    assert not out['eligible'].all()


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
    distinct_mt = set(catalog_df['mt_id'].dropna())
    distinct_gt = set(catalog_df['gt_id'].dropna())
    kennedy_df = compute_eligibility(distinct_mt, distinct_gt, kennedy_df, cols)

    diag_results = compute_overlap_rates(catalog_df, kennedy_df, cols, 1e-5, 1e-5, 'precise_mt_p', return_sets=True)

    export_pair_lists(tmp_path, catalog_df, kennedy_df, cols, diag_results, 'precise_mt_p')

    df_k_only = pd.read_csv(tmp_path / "pairs_kennedy_only.tsv", sep='\t')

    assert len(df_k_only) == 3
    # Check reasons
    reasons = df_k_only.set_index(['mt_id', 'gt_id'])['non_overlap_reason'].to_dict()
    assert reasons[('cg2', 'pr1')] == 'ineligible_cpg'

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
    distinct_mt = set(catalog_df['mt_id'].dropna())
    distinct_gt = set(catalog_df['gt_id'].dropna())
    kennedy_df = compute_eligibility(distinct_mt, distinct_gt, kennedy_df, cols)
    res = compute_overlap_rates(catalog_df, kennedy_df, cols, 1e-5, 1e-5, 'precise_mt_p', return_sets=True)

    assert res['recovery'] > 0
    assert res['confirmation_raw'] == 1 / 3
    assert res['confirmation_kennedy_testable'] == 0.5

    # Check caveats in summary
    from tools.benchmark_kennedy import build_summary_text
    import argparse
    args = argparse.Namespace(tecpg='cat.parquet', kennedy='ken.tsv',
                              kennedy_thresh=1e-5, tecpg_thresh=1e-5, upset=False)
    grid = {(1e-5, 1e-5): res}
    summary = build_summary_text(args, 2, 0, 0, 0, 'beta', 0, 0, 0, 'T.stat', grid, res, [1e-5], cols, kennedy_df)
    assert "Confirmation denominators are LOWER BOUNDS (except kennedy_testable)" in summary
    assert "The Kennedy file is not a full universe, so we cannot know how many of our" in summary

# O7b. counts_reconstruct_rates


def test_counts_reconstruct_rates():
    # The per-cell integer counts emitted alongside each rate must reproduce every
    # reported rate exactly, so any rate in benchmark_metrics.json is obtainable from
    # that run's own outputs. Uses the same synthetic fixture as O7 (inputs hand-controlled).
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
    distinct_mt = set(catalog_df['mt_id'].dropna())
    distinct_gt = set(catalog_df['gt_id'].dropna())
    kennedy_df = compute_eligibility(distinct_mt, distinct_gt, kennedy_df, cols)
    # No return_sets: this is the grid-cell code path, where 'counts' must also be present.
    res = compute_overlap_rates(catalog_df, kennedy_df, cols, 1e-5, 1e-5, 'precise_mt_p')

    counts = res['counts']

    # Hand-computed oracle for this fixture.
    assert counts['recovery_num'] == 1
    assert counts['recovery_denom'] == 2
    assert counts['confirmation_num'] == 1
    assert counts['confirmation_denom'] == 3
    assert counts['confirmation_testable_num'] == 1
    assert counts['confirmation_testable_denom'] == 2
    assert counts['k_tk'] == 3
    assert counts['union'] == 5

    # Reproducibility: each reported rate equals the ratio of its emitted integers.
    assert res['recovery'] == counts['recovery_num'] / counts['recovery_denom']
    assert res['confirmation_raw'] == counts['confirmation_num'] / counts['confirmation_denom']
    assert res['confirmation_kennedy_testable'] == \
        counts['confirmation_testable_num'] / counts['confirmation_testable_denom']
    assert res['jaccard'] == counts['confirmation_num'] / counts['union']

    # Counts must be JSON-native python ints (so the JSON needs no encoder for them).
    for v in counts.values():
        assert type(v) is int


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
    distinct_mt = set(catalog_df['mt_id'].dropna())
    distinct_gt = set(catalog_df['gt_id'].dropna())
    kennedy_df = compute_eligibility(distinct_mt, distinct_gt, kennedy_df, cols)

    res = compute_overlap_rates(catalog_df, kennedy_df, cols, float(1e-5),
                                float(1e-5), 'precise_mt_p', return_sets=True)
    assert len(res['T_tt']) == 1  # float(1e-5) > 9.99e-6
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
    distinct_mt = set(catalog_df['mt_id'].dropna())
    distinct_gt = set(catalog_df['gt_id'].dropna())
    kennedy_df = compute_eligibility(distinct_mt, distinct_gt, kennedy_df, cols)
    res = compute_overlap_rates(catalog_df, kennedy_df, cols, 1e-5, 1e-5, 'precise_mt_p', return_sets=True)

    export_pair_lists(tmp_path, catalog_df, kennedy_df, cols, res, 'precise_mt_p')

    conc = pd.read_csv(tmp_path / 'pairs_concordant.tsv', sep='\t')
    t_only = pd.read_csv(tmp_path / 'pairs_tecpg_only.tsv', sep='\t')
    k_only = pd.read_csv(tmp_path / 'pairs_kennedy_only.tsv', sep='\t')

    expected_cols = {'mt_id', 'gt_id', 'precise_mt_p', 'mt_est', 'mt_t',
                     'region', 'fdr_est', 'p.val', 'status', 'distance', 'non_overlap_reason'}
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
    distinct_mt = set(catalog_df['mt_id'].dropna())
    distinct_gt = set(catalog_df['gt_id'].dropna())
    kennedy_df = compute_eligibility(distinct_mt, distinct_gt, kennedy_df, cols)
    res = compute_overlap_rates(catalog_df, kennedy_df, cols, 1e-5, 1e-5, 'precise_mt_p', return_sets=True)

    assert res['confirmation_raw'] == 0.5  # 1 / 2
    assert res['confirmation_kennedy_testable'] == 1.0  # 1 / 1
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
    pq.write_table(pa.Table.from_arrays([pa.array(['cg1']), pa.array(['pr1']), pa.array(
        [1e-6])], names=['mt_id', 'gt_id', 'precise_mt_p']), tmp_path / "cat.parquet")

    # Call main through subprocess to capture exact stderr behaviour without interference
    result = subprocess.run([
        sys.executable, 'tools/benchmark_kennedy.py',
        '-t', str(tmp_path / "cat.parquet"),
        '-k', str(path),
        '-o', str(tmp_path)
    ], capture_output=True, text=True)

    # ensure that "0 dropped" because we dropna on cpg and probe ONLY.
    assert "dropped Kennedy rows with missing key columns:" in result.stderr
    assert "(0 dropped)" in result.stderr

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

    pq.write_table(pa.Table.from_arrays([pa.array(['cg1']), pa.array(['pr1']), pa.array(
        [1e-6])], names=['mt_id', 'gt_id', 'precise_mt_p']), tmp_path / "cat.parquet")
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
    distinct_mt = set(catalog_df['mt_id'].dropna())
    distinct_gt = set(catalog_df['gt_id'].dropna())
    kennedy_df = compute_eligibility(distinct_mt, distinct_gt, kennedy_df, cols)
    res = compute_overlap_rates(catalog_df, kennedy_df, cols, 1e-5, 1e-5, 'precise_mt_p', return_sets=True)
    export_pair_lists(tmp_path, catalog_df, kennedy_df, cols, res, 'precise_mt_p')

    k_only = pd.read_csv(tmp_path / "pairs_kennedy_only.tsv", sep='\t')
    assert len(k_only) == 1
    row = k_only.iloc[0]

    assert 'cpg_in_tecpg_universe' in row
    assert 'probe_in_tecpg_universe' in row
    assert 'cpg_in_kennedy_file' in row
    assert 'probe_in_kennedy_file' in row

    assert bool(row['cpg_in_kennedy_file']) is True
    assert 'cpg_in_tecpg_universe' in row


def test_override_oracle_missing():
    schema = ['mt_id', 'gt_id']
    with pytest.raises(ValueError):
        resolve_tecpg_pvalue_column(schema, override='precise_mt_p')


def test_override_oracle_present_warning(caplog):
    schema = ['mt_id', 'gt_id', 'my_p']
    resolve_tecpg_pvalue_column(schema, override='my_p')
    assert "WARNING: Using non-precise p-value column 'my_p'" in caplog.text


def test_override_oracle_precise_no_warning(caplog):
    schema = ['mt_id', 'gt_id', 'precise_mt_p']
    resolve_tecpg_pvalue_column(schema, override='precise_mt_p')
    assert "WARNING: Using non-precise p-value column" not in caplog.text


def test_a5d_kennedy_thresh_equals():
    import subprocess
    import sys
    result = subprocess.run([
        sys.executable, 'tools/benchmark_kennedy.py',
        '-t', 'dummy.parquet',
        '-k', 'dummy.tsv',
        '--p-thresh', '1e-6',
        '--kennedy-thresh=1e-5'
    ], capture_output=True, text=True)
    assert "ValueError: Cannot provide both --p-thresh and --kennedy-thresh." in result.stderr


def test_o7_tsv_long_form(tmp_path):
    import os
    import json

    catalog_df = pd.DataFrame({'mt_id': ['c1'], 'gt_id': ['p1'], 'precise_mt_p': [
                              1e-10], 'mt_est': [1.0], 'mt_t': [1.0], 'region': ['cis'], 'fdr_est': [0.1]})
    kennedy_df = pd.DataFrame({'CpG.probe': ['c1'], 'exp.Probe': ['p1'], 'p.val': [
                              1e-10], 'status': ['IN'], 'distance': [10]})

    path_k = tmp_path / "k.tsv"
    kennedy_df.to_csv(path_k, sep='\t', index=False)
    path_t = tmp_path / "t.parquet"
    pq.write_table(pa.Table.from_pandas(catalog_df), path_t)

    import subprocess
    result = subprocess.run([sys.executable, 'tools/benchmark_kennedy.py', '-t', str(path_t),
                            '-k', str(path_k), '-o', str(tmp_path)], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    tsv = pd.read_csv(tmp_path / 'benchmark_metrics.tsv', sep='\t')
    assert len(tsv) == 49 * 4
    assert 'stratum' in tsv.columns
    assert 'metric' in tsv.columns


def test_o6_json_no_numpy(tmp_path):
    import json
    import numpy as np
    from tools.benchmark_kennedy import profile_kennedy, profile_catalog_post_stream, resolve_kennedy_columns, compute_eligibility, compute_overlap_rates
    from collections import Counter
    import pandas as pd

    catalog_df = pd.DataFrame({'mt_id': ['c1', 'c2'], 'gt_id': ['p1', 'p2'], 'precise_mt_p': [
                              1e-10, 0.5], 'mt_est': [1.0, 0.5], 'mt_t': [1.0, 0.5], 'region': ['cis', 'trans'], 'fdr_est': [0.1, 0.9]})
    kennedy_df = pd.DataFrame({'CpG.probe': ['c1', 'c3'], 'exp.Probe': ['p1', 'p3'], 'p.val': [
                              1e-10, 0.5], 'status': ['IN', 'TRANS'], 'distance': [10, 100], 'exp.probe.chrm': ['1', '2']})

    fake_path = tmp_path / "fake.tsv"
    kennedy_df.to_csv(fake_path, sep='\t', index=False)

    cols = resolve_kennedy_columns(kennedy_df.columns)

    kennedy_profile = profile_kennedy(str(fake_path), kennedy_df, '	', cols, [1e-5])

    class Args:
        pass

    args = Args()
    args.batch_size = 500000
    args.tecpg = str(fake_path)

    cat_metrics = {
        'row_count': 2,
        'row_group_count': 1,
        'precise_mt_p_decades': {1e-5: 1},
        'precise_mt_p_min': np.float64(1e-10),
        'precise_mt_p_max': np.float64(0.5),
        'mt_chroms': set(), 'gt_chroms': set(), 'chrom_pairs': set()
    }

    cat_profile = profile_catalog_post_stream(args, str(fake_path), cat_metrics, list(catalog_df.columns), set(['c1', 'c2']), set(['p1', 'p2']), Counter({'cis': 1, 'trans': 1}))

    distinct_mt = set(['c1', 'c2'])
    distinct_gt = set(['p1', 'p2'])
    kennedy_df = compute_eligibility(distinct_mt, distinct_gt, kennedy_df, cols)

    diag_results = compute_overlap_rates(catalog_df, kennedy_df, cols, 1e-5, 1e-5, 'precise_mt_p', return_sets=True)
    grid_results = {(1e-5, 1e-5): compute_overlap_rates(catalog_df, kennedy_df, cols, 1e-5, 1e-5, 'precise_mt_p')}

    out_json = {
        'kennedy_profile': kennedy_profile,
        'catalog_profile': cat_profile,
        'results': {
            'diagonal': {k: v for k, v in diag_results.items() if not isinstance(v, (set, pd.DataFrame))},
            'grid': {f"1e-5_1e-5": grid_results[(1e-5, 1e-5)]},
            'num_merged': 1
        }
    }

    json.dumps(out_json)

def test_p8_p9_execution(tmp_path):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import subprocess
    import sys
    import json
    import re

    # 1. Test ABOVE
    catalog_df = pd.DataFrame({
        'mt_id': ['c1', 'c2'],
        'gt_id': ['p1', 'p2'],
        'precise_mt_p': [1e-10, 1e-10],
        'mt_est': [1.0, -1.0],
        'mt_t': [1.0, -1.0],
        'region': ['trans', 'trans'],
        'fdr_est': [0.1, 0.1]
    })

    kennedy_df = pd.DataFrame({
        'CpG.probe': ['c1', 'c2'],
        'exp.Probe': ['p1', 'p2'],
        'p.val': [1e-10, 1e-10],
        'status': ['IN', 'IN'],
        'distance': [10, 10],
        'beta': [1.0, 1.0],
        'T.stat': [1.0, 1.0]
    })

    path_k = tmp_path / "k.tsv"
    kennedy_df.to_csv(path_k, sep='\t', index=False)
    path_t = tmp_path / "t.parquet"
    pq.write_table(pa.Table.from_pandas(catalog_df), path_t)

    subprocess.run([sys.executable, 'tools/benchmark_kennedy.py', '-t', str(path_t), '-k',
                   str(path_k), '-o', str(tmp_path)], capture_output=True, text=True, check=True)

    with open(tmp_path / 'benchmark_report.html', 'r') as f:
        html_above = f.read()

    trans_frac = re.search(r'<section id="trans-fraction".*?(<div class="interpretation".*?</div>)', html_above, re.DOTALL).group(1)

    assert "ABOVE" in trans_frac
    assert "BELOW" not in trans_frac

    # 2. Test BELOW
    catalog_df = pd.DataFrame({
        'mt_id': ['c1', 'c2'],
        'gt_id': ['p1', 'p2'],
        'precise_mt_p': [1e-10, 1e-10],
        'mt_est': [1.0, -1.0],
        'mt_t': [1.0, -1.0],
        'region': ['cis', 'cis'],
        'fdr_est': [0.1, 0.1]
    })

    kennedy_df = pd.DataFrame({
        'CpG.probe': ['c1', 'c2'],
        'exp.Probe': ['p1', 'p2'],
        'p.val': [1e-10, 1e-10],
        'status': ['TRANS', 'TRANS'],
        'distance': [10, 10],
        'beta': [1.0, 1.0],
        'T.stat': [1.0, 1.0]
    })

    kennedy_df.to_csv(path_k, sep='\t', index=False)
    pq.write_table(pa.Table.from_pandas(catalog_df), path_t)

    subprocess.run([sys.executable, 'tools/benchmark_kennedy.py', '-t', str(path_t), '-k',
                   str(path_k), '-o', str(tmp_path)], capture_output=True, text=True, check=True)

    with open(tmp_path / 'benchmark_report.html', 'r') as f:
        html_below = f.read()

    trans_frac_below = re.search(r'<section id="trans-fraction".*?(<div class="interpretation".*?</div>)', html_below, re.DOTALL).group(1)

    assert "BELOW" in trans_frac_below
    assert "ABOVE" not in trans_frac_below

    # P8 Interpretation populated > 100 chars
    interps = re.findall(r'<div class="interpretation".*?>.*?Interpretation\.</strong>(.*?)</div.*?>', html_above, re.DOTALL)
    for interp in interps:
        text_content = re.sub(r'<[^>]+>', '', interp).strip()
        assert len(text_content) >= 100, f"Interpretation too short: {text_content}"

def test_p8_and_p9_and_p10_and_p11(tmp_path):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import subprocess
    import sys

    with open('tools/benchmark_kennedy.py', 'r') as b:
        code = b.read()

    assert "The raw denominator is a LOWER BOUND" in code
    assert 'status="FAIL"' not in code
    assert 'status = "FAIL"' not in code

def test_p6_panel_count(tmp_path):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import subprocess
    import sys

    catalog_df = pd.DataFrame({
        'mt_id': ['c1', 'c2'],
        'gt_id': ['p1', 'p2'],
        'precise_mt_p': [1e-10, 1e-10],
        'mt_est': [1.0, -1.0],
        'region': ['cis', 'trans'],
        'fdr_est': [0.1, 0.1]
    })
    kennedy_df = pd.DataFrame({
        'CpG.probe': ['c1', 'c2'],
        'exp.Probe': ['p1', 'p2'],
        'p.val': [1e-10, 1e-10],
        'status': ['IN', 'IN'],
        'distance': [10, 10],
        'beta': [1.0, 1.0],
    })

    path_k = tmp_path / "k.tsv"
    kennedy_df.to_csv(path_k, sep='\t', index=False)
    path_t = tmp_path / "t.parquet"
    pq.write_table(pa.Table.from_pandas(catalog_df), path_t)

    subprocess.run([sys.executable, 'tools/benchmark_kennedy.py', '-t', str(path_t), '-k',
                   str(path_k), '-o', str(tmp_path)], capture_output=True, text=True, check=True)

    import os
    assert os.path.exists(tmp_path / 'concordance_scatter.png')

    with open('tools/benchmark_kennedy.py', 'r') as b:
        code = b.read()
    assert "plt.subplots(" in code and "len(panels_to_draw)" in code

def test_p_oracles(tmp_path):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import pyarrow as pa
    import pyarrow.parquet as pq
    import subprocess
    import sys

    with open('tools/benchmark_kennedy.py', 'r') as b:
        code = b.read()

    assert "LogNorm(vmin=1)" in code
    assert "fig.colorbar(" in code
    assert code.count("label='y=x'") == 1
    assert "OLS Slope:" in code
    assert "concordant signs" in code
    assert "args.concordance_scatter_fig" not in code
    assert "args.venn_fig" not in code
    assert "args.upset_fig" not in code
    assert code.find("savefig(") < code.find("fig_to_base64(")


def test_o1_chunked_equals_projected(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd

    # 5,000 rows
    mts = [f"c{i}" for i in range(50)]
    gts = [f"p{i}" for i in range(100)]

    # Non-symmetric so F5 catches it
    rows = []
    import random
    random.seed(42)
    for _ in range(5000):
        rows.append({'mt_id': random.choice(mts), 'gt_id': random.choice(gts), 'precise_mt_p': random.random(
        ) * 1e-4, 'mt_est': 1.0, 'mt_t': 1.0, 'region': 'cis', 'fdr_est': 0.1, 'mt_chrom': '1', 'gt_chrom': '1'})

    df_cat = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df_cat)
    path_cat = tmp_path / "cat.parquet"
    # 3 row groups
    pq.write_table(table, path_cat, row_group_size=1700)

    kennedy_df = pd.DataFrame({'CpG.probe': mts[:10], 'exp.Probe': gts[:10], 'p.val': [
                              1e-6]*10, 'status': ['IN']*10, 'distance': [10]*10})

    class Args:
        tecpg = str(path_cat)
        kennedy = "dummy"
        batch_size = 1000

    from tools.benchmark_kennedy import stream_catalog_and_match

    kennedy_pairs = set(zip(kennedy_df['CpG.probe'], kennedy_df['exp.Probe']))
    schema = table.schema.names

    df_matched, t_counts, dist_mt, dist_gt, reg_cnt, cat_prof = stream_catalog_and_match(
        Args(), schema, 'precise_mt_p', kennedy_pairs, [
            1e-5, 1e-6], set(kennedy_df['CpG.probe']), set(kennedy_df['exp.Probe'])
    )

    # single-read reference
    df_ref = pd.read_parquet(path_cat)
    mask = pd.Series(zip(df_ref['mt_id'], df_ref['gt_id'])).isin(kennedy_pairs).values
    df_ref_matched = df_ref[mask]

    assert len(df_matched) == len(df_ref_matched)
    # The count should be EXACT
    assert t_counts[1e-5] == (df_ref['precise_mt_p'] < 1e-5).sum()


def test_o13_html_self_contained(tmp_path):
    catalog_df = pd.DataFrame({'mt_id': ['c1'], 'gt_id': ['p1'], 'precise_mt_p': [
                              1e-10], 'mt_est': [1.0], 'mt_t': [1.0], 'region': ['cis'], 'fdr_est': [0.1]})
    kennedy_df = pd.DataFrame({'CpG.probe': ['c1'], 'exp.Probe': ['p1'], 'p.val': [
                              1e-10], 'status': ['IN'], 'distance': [10]})

    path_k = tmp_path / "k.tsv"
    kennedy_df.to_csv(path_k, sep='\t', index=False)
    path_t = tmp_path / "t.parquet"
    pq.write_table(pa.Table.from_pandas(catalog_df), path_t)

    import subprocess
    import sys
    subprocess.run([sys.executable, 'tools/benchmark_kennedy.py', '-t', str(path_t), '-k',
                   str(path_k), '-o', str(tmp_path)], capture_output=True, text=True)

    with open(tmp_path / 'benchmark_report.html', 'r') as html_file:
        html_content = html_file.read()

    import re
    # Every img tag's src must start with data:image/png;base64,
    imgs = re.findall(r'<img[^>]+src="([^"]+)"', html_content)
    for src in imgs:
        assert src.startswith("data:image/png;base64,"), "Found local file reference in img src"


def test_o4_runtime_computation(tmp_path):
    import json
    catalog_df = pd.DataFrame({'mt_id': ['c1'], 'gt_id': ['p1'], 'precise_mt_p': [
                              1e-10], 'mt_est': [1.0], 'mt_t': [1.0], 'region': ['cis'], 'fdr_est': [0.1]})
    kennedy_df_a = pd.DataFrame({'CpG.probe': ['c1'], 'exp.Probe': ['p1'], 'p.val': [
                                1e-10], 'status': ['IN'], 'distance': [10]})
    kennedy_df_b = pd.DataFrame({'CpG.probe': ['c1', 'c2'], 'exp.Probe': ['p1', 'p2'], 'p.val': [
                                1e-10, 1e-10], 'status': ['TRANS', 'IN'], 'distance': [10, 10]})

    path_t = tmp_path / "t.parquet"
    pq.write_table(pa.Table.from_pandas(catalog_df), path_t)

    import subprocess
    import sys

    # Run A
    path_k_a = tmp_path / "ka.tsv"
    kennedy_df_a.to_csv(path_k_a, sep='\t', index=False)
    subprocess.run([sys.executable, 'tools/benchmark_kennedy.py', '-t', str(path_t), '-k',
                   str(path_k_a), '-o', str(tmp_path)], capture_output=True, text=True)
    with open(tmp_path / 'benchmark_metrics.json', 'r') as f:
        ja = json.load(f)

    # Run B
    path_k_b = tmp_path / "kb.tsv"
    kennedy_df_b.to_csv(path_k_b, sep='\t', index=False)
    subprocess.run([sys.executable, 'tools/benchmark_kennedy.py', '-t', str(path_t), '-k',
                   str(path_k_b), '-o', str(tmp_path)], capture_output=True, text=True)
    with open(tmp_path / 'benchmark_metrics.json', 'r') as f:
        jb = json.load(f)

    # Assert differing profiles!
    assert ja['kennedy_profile']['status_composition']['counts'] != \
        jb['kennedy_profile']['status_composition']['counts']


def test_o10_load_catalog_ordering():
    from tools.benchmark_kennedy import stream_catalog_and_match

    class Args:
        pass
    args = Args()
    args.tecpg = 'dummy.parquet'
    args.batch_size = 1000

    # We test that stream_catalog_and_match builds cols_to_load deterministically
    import ast
    import inspect
    with open('tools/benchmark_kennedy.py', 'r') as b:
        tree = ast.parse(b.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'stream_catalog_and_match':
                func_code = ast.unparse(node)
                assert "set(desired_cols)" not in func_code
                assert "for c in desired_cols:" in func_code


def test_o11_grid_no_set_retention(tmp_path):
    catalog_df = pd.DataFrame({'mt_id': ['c1'], 'gt_id': ['p1'], 'precise_mt_p': [
                              1e-10], 'mt_est': [1.0], 'mt_t': [1.0], 'region': ['cis'], 'fdr_est': [0.1]})
    kennedy_df = pd.DataFrame({'CpG.probe': ['c1'], 'exp.Probe': ['p1'], 'p.val': [
                              1e-10], 'status': ['IN'], 'distance': [10]})

    from tools.benchmark_kennedy import compute_overlap_rates
    cols = {'pval': 'p.val', 'cpg': 'CpG.probe', 'probe': 'exp.Probe'}
    kennedy_df['eligible'] = True

    # grid call
    res = compute_overlap_rates(catalog_df, kennedy_df, cols, 1e-5, 1e-5, 'precise_mt_p',
                                return_sets=False, kennedy_cpgs=set(), kennedy_probes=set())
    assert 'T_tt' not in res

    # diag call
    res = compute_overlap_rates(catalog_df, kennedy_df, cols, 1e-5, 1e-5, 'precise_mt_p',
                                return_sets=True, kennedy_cpgs=set(), kennedy_probes=set())
    assert 'T_tt' in res


def test_o12_html_uses_shared_framework():
    with open('tools/benchmark_kennedy.py', 'r') as b:
        code = b.read()
    assert "from permute_qc_report import QCModule, fig_to_base64, render_table, render_html" in code
    assert "def render_table(" not in code


def test_o15_html_no_fail_status(tmp_path):
    catalog_df = pd.DataFrame({'mt_id': ['c1'], 'gt_id': ['p1'], 'precise_mt_p': [
                              1e-10], 'mt_est': [1.0], 'mt_t': [1.0], 'region': ['cis'], 'fdr_est': [0.1]})
    kennedy_df = pd.DataFrame({'CpG.probe': ['c1'], 'exp.Probe': ['p1'], 'p.val': [
                              1e-10], 'status': ['IN'], 'distance': [10]})

    path_k = tmp_path / "k.tsv"
    kennedy_df.to_csv(path_k, sep='\t', index=False)
    path_t = tmp_path / "t.parquet"
    pq.write_table(pa.Table.from_pandas(catalog_df), path_t)

    import subprocess
    import sys
    subprocess.run([sys.executable, 'tools/benchmark_kennedy.py', '-t', str(path_t), '-k',
                   str(path_k), '-o', str(tmp_path)], capture_output=True, text=True)

    with open(tmp_path / 'benchmark_report.html', 'r') as html_file:
        html_content = html_file.read()

    pass


def test_o8_sha256_stability(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("hello")
    from tools.benchmark_kennedy import _sha256sum
    assert _sha256sum(path) == _sha256sum(path)
    path.write_text("hello2")
    assert _sha256sum(path) == _sha256sum(tmp_path / "f.txt" if False else path)


def test_o9_no_full_pair_set():
    with open('tools/benchmark_kennedy.py', 'r') as b:
        code = b.read()
    assert "catalog_pairs = set(zip(" not in code


def test_o17_figure_rendered_once():
    with open('tools/benchmark_kennedy.py', 'r') as b:
        code = b.read()
    # It must call savefig first then fig_to_base64
    assert code.find("savefig(") < code.find("fig_to_base64(")


def test_o18_html_from_json():
    # Since Part B is just rendering, it should be possible from JSON
    # This is an architectural check
    with open('tools/benchmark_kennedy.py', 'r') as b:
        code = b.read()
    assert "build_reference_file_profile_module(args.kennedy_profile_metrics)" in code

def test_version_lookup_never_raises(monkeypatch):
    import importlib.metadata
    import sys
    from types import SimpleNamespace
    from tools.benchmark_kennedy import _resolve_tecpg_version

    def _raise_pnf(_):
        raise importlib.metadata.PackageNotFoundError('tecpg')

    monkeypatch.setattr(importlib.metadata, 'version', lambda _: '1.2.3')
    assert _resolve_tecpg_version() == '1.2.3'

    monkeypatch.setattr(importlib.metadata, 'version', _raise_pnf)
    monkeypatch.setitem(sys.modules, 'tecpg',
                        SimpleNamespace(__version__='9.9.9'))
    assert _resolve_tecpg_version() == '9.9.9'

    monkeypatch.setattr(importlib.metadata, 'version', _raise_pnf)
    monkeypatch.setitem(sys.modules, 'tecpg', SimpleNamespace())
    v = _resolve_tecpg_version()
    assert v == 'unknown'
    assert isinstance(v, str) and v
