import subprocess
import pytest
import os
import sys
import pandas as pd
import numpy as np

TOOL_PATH = "tools/summarizeOutput_parquet.py"

@pytest.fixture
def base_fixture(tmp_path):
    main_file = tmp_path / "main.parquet"
    res_file = tmp_path / "res.csv"

    # Some basic data
    # Create rows with p-values, including a null one.
    data = {
        'mt_id': [1, 2, 3, 4, 5],
        'gt_id': [1, 2, 3, 4, 5],
        'mt_t': [2.5, 3.0, 1.5, -4.0, 2.0],
        'precise_mt_p': [0.012, 0.003, 0.13, 0.0001, np.nan],
        'region': ['A', 'A', 'B', 'B', 'A']
    }
    df = pd.DataFrame(data)
    df.to_parquet(main_file)

    # Empty reservoir
    res = pd.DataFrame(columns=['mt_id', 'gt_id', 'mt_t', 'p_value'])
    res.to_csv(res_file, index=False)

    return main_file, res_file

def run_tool(args):
    cmd = [sys.executable, TOOL_PATH] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def test_default_invocation_is_unchanged(tmp_path, base_fixture):
    main_file, res_file = base_fixture
    out_file = tmp_path / "out.parquet"

    res = run_tool([
        "--main-file", str(main_file),
        "--reservoir-file", str(res_file),
        "--total-tests", "10",
        "--df", "10",
        "--calculate-fdr",
        "--output-fdr-file", str(out_file)
    ])
    assert res.returncode == 0, res.stderr

    df_in = pd.read_parquet(main_file)
    df_out = pd.read_parquet(out_file)

    # Check output columns
    expected_cols = list(df_in.columns) + ['fdr_est']
    assert list(df_out.columns) == expected_cols

    # Check original columns are equal
    pd.testing.assert_frame_equal(df_in, df_out[df_in.columns])

def test_fdr_column_write_is_additive(tmp_path, base_fixture):
    main_file, res_file = base_fixture
    out_file_1 = tmp_path / "out1.parquet"

    run_tool([
        "--main-file", str(main_file),
        "--reservoir-file", str(res_file),
        "--total-tests", "10",
        "--df", "10",
        "--calculate-fdr",
        "--output-fdr-file", str(out_file_1)
    ])

    df_out1 = pd.read_parquet(out_file_1)
    df_out1['p_permute'] = [0.01, 0.002, 0.1, 0.00005, np.nan]
    out_file_2 = tmp_path / "out2.parquet"
    df_out1.to_parquet(out_file_2)

    out_file_3 = tmp_path / "out3.parquet"
    res = run_tool([
        "--main-file", str(out_file_2),
        "--reservoir-file", str(res_file),
        "--total-tests", "10",
        "--df", "10",
        "--calculate-fdr",
        "--p-column", "p_permute",
        "--fdr-column", "fdr_permute",
        "--output-fdr-file", str(out_file_3)
    ])
    assert res.returncode == 0, res.stderr

    df_out3 = pd.read_parquet(out_file_3)
    # Assert fdr_est is byte-identical (or value identical) to first run
    pd.testing.assert_series_equal(df_out1['fdr_est'], df_out3['fdr_est'])
    assert 'fdr_permute' in df_out3.columns
    assert 'fdr_permute' != 'fdr_est'

def test_refuses_to_overwrite_existing_fdr_column(tmp_path, base_fixture):
    main_file, res_file = base_fixture
    out_file_1 = tmp_path / "out1.parquet"

    run_tool([
        "--main-file", str(main_file),
        "--reservoir-file", str(res_file),
        "--total-tests", "10",
        "--df", "10",
        "--calculate-fdr",
        "--output-fdr-file", str(out_file_1)
    ])

    out_file_2 = tmp_path / "out2.parquet"
    res = run_tool([
        "--main-file", str(out_file_1),
        "--reservoir-file", str(res_file),
        "--total-tests", "10",
        "--df", "10",
        "--calculate-fdr",
        "--fdr-column", "fdr_est",
        "--output-fdr-file", str(out_file_2)
    ])
    assert res.returncode != 0
    assert not out_file_2.exists()
    assert "already exists in" in res.stdout

def test_explicit_missing_p_column_fails_closed(tmp_path, base_fixture):
    main_file, res_file = base_fixture
    out_file = tmp_path / "out.parquet"

    res = run_tool([
        "--main-file", str(main_file),
        "--reservoir-file", str(res_file),
        "--total-tests", "10",
        "--df", "10",
        "--calculate-fdr",
        "--p-column", "p_permute",
        "--output-fdr-file", str(out_file)
    ])
    assert res.returncode != 0
    assert not out_file.exists()
    assert "Refusing to fall back" in res.stdout
    assert "Falling back to t-statistic calculation" not in res.stdout

def test_default_missing_p_column_still_falls_back(tmp_path, base_fixture):
    main_file, res_file = base_fixture
    df = pd.read_parquet(main_file)
    df = df.drop(columns=['precise_mt_p'])
    main_file_no_p = tmp_path / "main_no_p.parquet"
    df.to_parquet(main_file_no_p)

    out_file = tmp_path / "out.parquet"
    res = run_tool([
        "--main-file", str(main_file_no_p),
        "--reservoir-file", str(res_file),
        "--total-tests", "10",
        "--df", "10",
        "--calculate-fdr",
        "--output-fdr-file", str(out_file)
    ])
    assert res.returncode == 0, res.stderr
    assert "Falling back to t-statistic calculation" in res.stdout
    assert out_file.exists()

def test_null_source_p_yields_null_fdr(tmp_path, base_fixture):
    main_file, res_file = base_fixture
    out_file = tmp_path / "out.parquet"

    run_tool([
        "--main-file", str(main_file),
        "--reservoir-file", str(res_file),
        "--total-tests", "10",
        "--df", "10",
        "--calculate-fdr",
        "--output-fdr-file", str(out_file)
    ])

    df_out = pd.read_parquet(out_file)
    null_mask = df_out['precise_mt_p'].isna()

    assert df_out.loc[null_mask, 'fdr_est'].isna().all()
    assert df_out.loc[~null_mask, 'fdr_est'].notna().all()

def test_bh_matches_hand_computed_reference(tmp_path, base_fixture):
    main_file, res_file = base_fixture
    out_file = tmp_path / "out.parquet"

    run_tool([
        "--main-file", str(main_file),
        "--reservoir-file", str(res_file),
        "--total-tests", "10", # 10 tests total
        "--df", "10",
        "--calculate-fdr",
        "--output-fdr-file", str(out_file)
    ])

    df_out = pd.read_parquet(out_file)

    # Exclude nulls
    df_valid = df_out.dropna(subset=['precise_mt_p']).copy()

    # Compute BH by hand
    p_vals = df_valid['precise_mt_p'].values
    sorted_idx = np.argsort(p_vals)
    sorted_p = p_vals[sorted_idx]

    # rank 1-indexed
    ranks = np.arange(1, len(p_vals) + 1)
    total_tests = 10

    fdr_est = sorted_p * total_tests / ranks

    # Step down
    for i in range(len(fdr_est) - 2, -1, -1):
        fdr_est[i] = min(fdr_est[i], fdr_est[i+1])

    fdr_est = np.minimum(fdr_est, 1.0)

    df_valid['hand_fdr'] = 0.0
    df_valid.iloc[sorted_idx, df_valid.columns.get_loc('hand_fdr')] = fdr_est

    pd.testing.assert_series_equal(df_valid['fdr_est'], df_valid['hand_fdr'], check_names=False)

def run_fdr_tool(tmp_path, df_in, fdr_col, compare_col=None, total_tests=10):
    main_file = tmp_path / "main.parquet"
    res_file = tmp_path / "res.csv"
    out_file = tmp_path / f"out_{fdr_col}.parquet"

    df_in.to_parquet(main_file)
    pd.DataFrame(columns=['mt_id', 'gt_id', 'mt_t', 'p_value']).to_csv(res_file, index=False)

    cmd = [
        "--main-file", str(main_file),
        "--reservoir-file", str(res_file),
        "--total-tests", str(total_tests),
        "--df", "10",
        "--calculate-fdr",
        "--p-column", "p_source" if "p_source" in df_in.columns else "precise_mt_p",
        "--fdr-column", fdr_col,
        "--output-fdr-file", str(out_file)
    ]
    if compare_col:
        cmd.extend(["--compare-fdr-column", compare_col])

    return run_tool(cmd), out_file

def test_compare_fdr_identical_pools_reports_equal(tmp_path):
    df = pd.DataFrame({'mt_id': [1, 2, 3], 'gt_id': [1, 2, 3], 'precise_mt_p': [0.01, 0.05, 0.1]})
    res1, out1 = run_fdr_tool(tmp_path, df, "fdr_est")
    df2 = pd.read_parquet(out1)
    df2["p_permute"] = df2["precise_mt_p"]
    main_file2 = tmp_path / "main2.parquet"
    res_file2 = tmp_path / "res2.csv"
    out_file2 = tmp_path / "out2.parquet"
    df2.to_parquet(main_file2)
    pd.DataFrame(columns=['mt_id', 'gt_id', 'mt_t', 'p_value']).to_csv(res_file2, index=False)
    res2 = run_tool([
        "--main-file", str(main_file2), "--reservoir-file", str(res_file2), "--total-tests", "10",
        "--df", "10", "--calculate-fdr", "--p-column", "p_permute", "--fdr-column", "fdr_permute",
        "--compare-fdr-column", "fdr_est", "--output-fdr-file", str(out_file2)
    ])
    assert res2.returncode == 0
    assert "VERDICT: EQUAL (identical pools and denominator)" in res2.stdout
    assert "max|diff|            : 0.000000e+00" in res2.stdout

def test_compare_fdr_subset_pool_reports_directional(tmp_path):
    df = pd.DataFrame({'mt_id': [1, 2, 3, 4, 5], 'gt_id': [1, 2, 3, 4, 5], 'precise_mt_p': [0.001, 0.002, 0.003, 0.004, 0.005]})
    res1, out1 = run_fdr_tool(tmp_path, df, "fdr_est")
    df2 = pd.read_parquet(out1)
    df2["p_permute"] = df2["precise_mt_p"]
    df2.loc[1:2, "p_permute"] = np.nan
    main_file2 = tmp_path / "main2.parquet"
    res_file2 = tmp_path / "res2.csv"
    out_file2 = tmp_path / "out2.parquet"
    df2.to_parquet(main_file2)
    pd.DataFrame(columns=['mt_id', 'gt_id', 'mt_t', 'p_value']).to_csv(res_file2, index=False)
    res2 = run_tool([
        "--main-file", str(main_file2), "--reservoir-file", str(res_file2), "--total-tests", "10",
        "--df", "10", "--calculate-fdr", "--p-column", "p_permute", "--fdr-column", "fdr_permute",
        "--compare-fdr-column", "fdr_est", "--output-fdr-file", str(out_file2)
    ])
    assert res2.returncode == 0
    assert "VERDICT: DIRECTIONAL-OK" in res2.stdout
    assert "rows left the pool   : 2" in res2.stdout
    import re
    match = re.search(r"min diff \(signed\)\s*:\s*([^\s]+)", res2.stdout)
    assert match is not None
    assert float(match.group(1)) >= 0.0

def test_compare_fdr_smaller_denominator_is_violation(tmp_path):
    df = pd.DataFrame({'mt_id': [1, 2, 3], 'gt_id': [1, 2, 3], 'precise_mt_p': [0.01, 0.05, 0.1]})
    res1, out1 = run_fdr_tool(tmp_path, df, "fdr_est", total_tests=9)
    df2 = pd.read_parquet(out1)
    df2["p_permute"] = df2["precise_mt_p"]
    main_file2 = tmp_path / "main2.parquet"
    res_file2 = tmp_path / "res2.csv"
    out_file2 = tmp_path / "out2.parquet"
    df2.to_parquet(main_file2)
    pd.DataFrame(columns=['mt_id', 'gt_id', 'mt_t', 'p_value']).to_csv(res_file2, index=False)
    res2 = run_tool([
        "--main-file", str(main_file2), "--reservoir-file", str(res_file2), "--total-tests", "3",
        "--df", "10", "--calculate-fdr", "--p-column", "p_permute", "--fdr-column", "fdr_permute",
        "--compare-fdr-column", "fdr_est", "--output-fdr-file", str(out_file2)
    ])
    assert res2.returncode == 1
    assert "VERDICT: VIOLATION" in res2.stdout

def test_compare_fdr_column_absent_prints_notice_and_skips(tmp_path, base_fixture):
    main_file, res_file = base_fixture
    out_file = tmp_path / "out.parquet"
    res = run_tool([
        "--main-file", str(main_file), "--reservoir-file", str(res_file), "--total-tests", "10",
        "--df", "10", "--calculate-fdr", "--fdr-column", "fdr_est", "--compare-fdr-column", "missing_fdr",
        "--output-fdr-file", str(out_file)
    ])
    assert res.returncode == 0
    assert "Notice: --compare-fdr-column 'missing_fdr' not found in schema. Skipping comparison." in res.stdout
    assert "FDR comparison:" not in res.stdout
    assert out_file.exists()

def test_compare_fdr_not_requested_prints_no_verdict_block(tmp_path, base_fixture):
    main_file, res_file = base_fixture
    out_file = tmp_path / "out.parquet"
    res = run_tool([
        "--main-file", str(main_file), "--reservoir-file", str(res_file), "--total-tests", "10",
        "--df", "10", "--calculate-fdr", "--fdr-column", "fdr_est", "--output-fdr-file", str(out_file)
    ])
    assert res.returncode == 0
    assert "FDR comparison:" not in res.stdout
    assert out_file.exists()

def test_compare_fdr_disjoint_nulls_excluded_from_comparable(tmp_path):
    df = pd.DataFrame({'mt_id': [1, 2, 3, 4], 'gt_id': [1, 2, 3, 4], 'precise_mt_p': [0.01, 0.05, np.nan, 0.1]})
    res1, out1 = run_fdr_tool(tmp_path, df, "fdr_est")
    df2 = pd.read_parquet(out1)
    df2["p_permute"] = [0.01, np.nan, 0.08, 0.1]
    main_file2 = tmp_path / "main2.parquet"
    res_file2 = tmp_path / "res2.csv"
    out_file2 = tmp_path / "out2.parquet"
    df2.to_parquet(main_file2)
    pd.DataFrame(columns=['mt_id', 'gt_id', 'mt_t', 'p_value']).to_csv(res_file2, index=False)
    res2 = run_tool([
        "--main-file", str(main_file2), "--reservoir-file", str(res_file2), "--total-tests", "10",
        "--df", "10", "--calculate-fdr", "--p-column", "p_permute", "--fdr-column", "fdr_permute",
        "--compare-fdr-column", "fdr_est", "--output-fdr-file", str(out_file2)
    ])
    assert res2.returncode == 0
    assert "comparable rows      : 2" in res2.stdout
    assert "rows left the pool   : 1" in res2.stdout
    assert "rows entered the pool: 1" in res2.stdout

def test_compare_fdr_subtolerance_positive_diff_is_not_equal(tmp_path):
    df = pd.DataFrame({'mt_id': [1, 2, 3], 'gt_id': [1, 2, 3], 'precise_mt_p': [0.01, 0.05, 0.1]})
    res1, out1 = run_fdr_tool(tmp_path, df, "fdr_est")
    df2 = pd.read_parquet(out1)
    df2["p_permute"] = df2["precise_mt_p"]
    df2.loc[0, "fdr_est"] -= 5e-13
    main_file2 = tmp_path / "main2.parquet"
    res_file2 = tmp_path / "res2.csv"
    out_file2 = tmp_path / "out2.parquet"
    df2.to_parquet(main_file2)
    pd.DataFrame(columns=['mt_id', 'gt_id', 'mt_t', 'p_value']).to_csv(res_file2, index=False)
    res2 = run_tool([
        "--main-file", str(main_file2), "--reservoir-file", str(res_file2), "--total-tests", "10",
        "--df", "10", "--calculate-fdr", "--p-column", "p_permute", "--fdr-column", "fdr_permute",
        "--compare-fdr-column", "fdr_est", "--output-fdr-file", str(out_file2)
    ])
    assert res2.returncode == 0
    assert "VERDICT: DIRECTIONAL-OK" in res2.stdout
    import re
    match = re.search(r"max\|diff\|\s*:\s*([^\s]+)", res2.stdout)
    assert match is not None
    max_diff = float(match.group(1))
    assert np.isclose(max_diff, 5e-13, atol=1e-15, rtol=0)
    match2 = re.search(r"min diff \(signed\)\s*:\s*([^\s]+)", res2.stdout)
    assert match2 is not None
    assert float(match2.group(1)) >= 0.0

def test_compare_fdr_subtolerance_negative_diff_is_violation(tmp_path):
    df = pd.DataFrame({'mt_id': [1, 2, 3], 'gt_id': [1, 2, 3], 'precise_mt_p': [0.01, 0.05, 0.1]})
    res1, out1 = run_fdr_tool(tmp_path, df, "fdr_est")
    df2 = pd.read_parquet(out1)
    df2["p_permute"] = df2["precise_mt_p"]
    df2.loc[0, "fdr_est"] += 5e-13
    main_file2 = tmp_path / "main2.parquet"
    res_file2 = tmp_path / "res2.csv"
    out_file2 = tmp_path / "out2.parquet"
    df2.to_parquet(main_file2)
    pd.DataFrame(columns=['mt_id', 'gt_id', 'mt_t', 'p_value']).to_csv(res_file2, index=False)
    res2 = run_tool([
        "--main-file", str(main_file2), "--reservoir-file", str(res_file2), "--total-tests", "10",
        "--df", "10", "--calculate-fdr", "--p-column", "p_permute", "--fdr-column", "fdr_permute",
        "--compare-fdr-column", "fdr_est", "--output-fdr-file", str(out_file2)
    ])
    assert res2.returncode == 1
    assert "VERDICT: VIOLATION" in res2.stdout
