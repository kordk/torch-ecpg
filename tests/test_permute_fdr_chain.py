import json
import os
import subprocess
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np

# We ensure compatibility across python/pandas versions
def get_python_exe():
    return sys.executable

@pytest.fixture
def test_data(tmp_path):
    # Create >= 400 rows fixture
    n_rows = 405

    # 7 Canonical regions
    canonical_regions = [
        "CIS5", "DISTAL5", "TRANS", "PROMOTER", "GENEBODY", "CIS3", "DISTAL3"
    ]

    # Needs mt_id, gt_id, mt_t, mt_p, precise_mt_p, region, fdr_est
    # Need tie groups: >=10 rows at exactly 0.0, >=5 rows at identical non-zero p (e.g., 0.001)

    # Distribute the regions
    regions = np.resize(canonical_regions, n_rows)

    p_vals = np.linspace(0.01, 0.99, n_rows)
    # inject the zero group (10 rows)
    p_vals[0:10] = 0.0
    # inject the non-zero tie group (5 rows)
    p_vals[10:15] = 0.001

    # Ensure they are sorted (makes step-down easier to verify conceptually, though BH sorts internally anyway)
    # Not strictly necessary to pre-sort, BH handles it.

    df = pd.DataFrame({
        "mt_id": [f"cg{i:04d}" for i in range(n_rows)],
        "gt_id": [f"ENSG{i:04d}" for i in range(n_rows)],
        "mt_t": np.random.randn(n_rows),
        "mt_p": p_vals.astype("float32"),
        "precise_mt_p": p_vals.astype("float64"),
        "region": regions
        # fdr_est is calculated below
    })

    catalog_path = tmp_path / "summarized.parquet"
    df.to_parquet(catalog_path)

    # Also create a reservoir file
    reservoir_path = tmp_path / "sample_reservoir.csv"
    res_df = pd.DataFrame({
        "mt_id": ["cg0000", "cg0001"],
        "gt_id": ["ENSG0000", "ENSG0001"],
        "mt_t": [1.0, -1.0]
    })
    res_df.to_csv(reservoir_path, index=False)

    # Generate actual fdr_est using the tool itself to ensure consistency
    total_tests = 10000
    subprocess.run([
        get_python_exe(), os.path.join(os.getcwd(), "tools/summarizeOutput_parquet.py"),
        "--main-file", str(catalog_path),
        "--reservoir-file", str(reservoir_path),
        "--total-tests", str(total_tests),
        "--df", "100",
        "--p-column", "precise_mt_p",
        "--fdr-column", "fdr_est",
        "--calculate-fdr",
        "--output-fdr-file", str(catalog_path)
    ], cwd=tmp_path, check=True)

    return {
        "catalog": catalog_path,
        "reservoir": reservoir_path,
        "n_rows": n_rows,
        "total_tests": total_tests
    }


def create_eval_report(tmp_path, divergent_regions=None, statuses=None):
    if divergent_regions is None:
        divergent_regions = []
    if statuses is None:
        statuses = {}

    regions = [
        "CIS5", "DISTAL5", "TRANS", "PROMOTER", "GENEBODY", "CIS3", "DISTAL3"
    ]
    report = {
        "arms": {
            "calibration": {},
            "stratify_decision": {
                "mode": "per_region",
                "per_region": {
                    r: {"status": statuses.get(r, "ok")} for r in regions
                },
                "divergent_regions": divergent_regions
            }
        }
    }

    path = tmp_path / "eval_permute_report.json"
    with open(path, "w") as f:
        json.dump(report, f)
    return path


def run_chain(tmp_path, catalog_path, eval_report_path, reservoir_path, total_tests, df="100"):
    annot_tmp = tmp_path / "annot_tmp.parquet"
    final_out = tmp_path / "final.permute.parquet"

    subprocess.run([
        get_python_exe(), "-u", os.path.join(os.getcwd(), "tools/annotate_permute_p.py"),
        "--input", str(catalog_path),
        "--output", str(annot_tmp),
        "--eval-report", str(eval_report_path),
        "--p-source", "precise_mt_p",
        "--p-column", "p_permute"
    ], cwd=tmp_path, check=True)

    subprocess.run([
        get_python_exe(), "-u", os.path.join(os.getcwd(), "tools/summarizeOutput_parquet.py"),
        "--main-file", str(annot_tmp),
        "--reservoir-file", str(reservoir_path),
        "--total-tests", str(total_tests),
        "--df", df,
        "--p-column", "p_permute",
        "--fdr-column", "fdr_permute",
        "--calculate-fdr",
        "--output-fdr-file", str(final_out)
    ], cwd=tmp_path, check=True)

    return final_out

def test_fully_licensed_chain_reproduces_fdr_est(tmp_path, test_data):
    eval_report = create_eval_report(tmp_path) # All ok, none divergent

    final_out = run_chain(
        tmp_path,
        test_data["catalog"],
        eval_report,
        test_data["reservoir"],
        test_data["total_tests"]
    )

    df_in = pd.read_parquet(test_data["catalog"])
    df_out = pd.read_parquet(final_out)

    # Headline oracle: fdr_permute MUST equal fdr_est exactly
    np.testing.assert_allclose(df_out["fdr_permute"], df_in["fdr_est"], rtol=0, atol=0, equal_nan=True)

def test_denominator_change_alters_fdr_permute(tmp_path, test_data):
    eval_report = create_eval_report(tmp_path)

    smaller_total_tests = test_data["n_rows"] + 10 # still above row count but much smaller than original total_tests

    final_out = run_chain(
        tmp_path,
        test_data["catalog"],
        eval_report,
        test_data["reservoir"],
        smaller_total_tests
    )

    df_in = pd.read_parquet(test_data["catalog"])
    df_out = pd.read_parquet(final_out)

    # fdr_permute should differ
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(df_out["fdr_permute"], df_in["fdr_est"], rtol=0, atol=0, equal_nan=True)

    assert not np.array_equal(df_out["fdr_permute"].fillna(-1), df_in["fdr_est"].fillna(-1))

def test_divergent_stratum_yields_null_p_and_null_fdr(tmp_path, test_data):
    # one region in divergent_regions while its status stays 'ok'
    eval_report = create_eval_report(tmp_path, divergent_regions=["CIS5"], statuses={"CIS5": "ok"})

    final_out = run_chain(
        tmp_path,
        test_data["catalog"],
        eval_report,
        test_data["reservoir"],
        test_data["total_tests"]
    )

    df_out = pd.read_parquet(final_out)
    cis_rows = df_out[df_out["region"] == "CIS5"]

    assert cis_rows["p_permute"].isna().all(), "divergent stratum must have null p_permute"
    assert cis_rows["fdr_permute"].isna().all(), "divergent stratum must have null fdr_permute"

def test_insufficient_data_stratum_yields_null_fdr(tmp_path, test_data):
    eval_report = create_eval_report(tmp_path, statuses={"DISTAL5": "insufficient_data"})

    final_out = run_chain(
        tmp_path,
        test_data["catalog"],
        eval_report,
        test_data["reservoir"],
        test_data["total_tests"]
    )

    df_out = pd.read_parquet(final_out)
    distal_rows = df_out[df_out["region"] == "DISTAL5"]

    assert distal_rows["p_permute"].isna().all(), "insufficient_data stratum must have null p_permute"
    assert distal_rows["fdr_permute"].isna().all(), "insufficient_data stratum must have null fdr_permute"

def test_tied_p_values_receive_identical_fdr(tmp_path, test_data):
    eval_report = create_eval_report(tmp_path)

    final_out = run_chain(
        tmp_path,
        test_data["catalog"],
        eval_report,
        test_data["reservoir"],
        test_data["total_tests"]
    )

    df_out = pd.read_parquet(final_out)

    zero_group = df_out[df_out["precise_mt_p"] == 0.0]
    assert len(zero_group) >= 10
    assert zero_group["fdr_permute"].nunique() == 1

    nonzero_group = df_out[df_out["precise_mt_p"] == 0.001]
    assert len(nonzero_group) >= 5
    assert nonzero_group["fdr_permute"].nunique() == 1

    # Hand-computed reference value for the non-zero tie group (0.001 * 10000 / 15)
    expected_val = 0.6666666666666666
    np.testing.assert_allclose(nonzero_group["fdr_permute"].iloc[0], expected_val, rtol=1e-5, atol=1e-5)

def test_analytic_columns_unchanged_by_value(tmp_path, test_data):
    eval_report = create_eval_report(tmp_path, divergent_regions=["CIS5"]) # Use partial to ensure we don't accidentally drop them when masking

    final_out = run_chain(
        tmp_path,
        test_data["catalog"],
        eval_report,
        test_data["reservoir"],
        test_data["total_tests"]
    )

    df_in = pd.read_parquet(test_data["catalog"])
    df_out = pd.read_parquet(final_out)

    np.testing.assert_allclose(df_out["mt_p"], df_in["mt_p"], rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(df_out["precise_mt_p"], df_in["precise_mt_p"], rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(df_out["fdr_est"], df_in["fdr_est"], rtol=0, atol=0, equal_nan=True)

def test_partial_licensing_is_more_conservative(tmp_path, test_data):
    eval_report = create_eval_report(tmp_path, divergent_regions=["CIS5"])

    final_out = run_chain(
        tmp_path,
        test_data["catalog"],
        eval_report,
        test_data["reservoir"],
        test_data["total_tests"]
    )

    df_in = pd.read_parquet(test_data["catalog"])
    df_out = pd.read_parquet(final_out)

    licensed_mask = df_out["region"] != "CIS5"

    fdr_permute_licensed = df_out.loc[licensed_mask, "fdr_permute"]
    fdr_est_licensed = df_in.loc[licensed_mask, "fdr_est"]

    # fdr_permute should be >= fdr_est
    assert (fdr_permute_licensed >= fdr_est_licensed).all()
    # they should not be identical
    assert not (fdr_permute_licensed == fdr_est_licensed).all()
