import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

# Adjust path to import canonical regions for fixture creation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))
import eval_permute as E  # noqa: E402

TOOL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools', 'annotate_permute_p.py'))


@pytest.fixture
def eval_report_path(tmp_path):
    report = {
        "arms": {
            "stratify_decision": {
                "mode": "per_region",
                "reference": E.REGION_REFERENCE,
                "per_region": {
                    "TRANS": {"status": "reference", "n_bulk": 1000},
                    "CIS5": {"status": "ok", "n_bulk": 500},
                    "PROMOTER": {"status": "ok", "n_bulk": 200},  # Divergent
                    "GENEBODY": {"status": "insufficient_data", "n_bulk": 10},
                    "CIS3": {"status": "ok", "n_bulk": 300},
                    # DISTAL5 absent from per_region
                    # DISTAL3 absent from per_region
                },
                "divergent_regions": ["PROMOTER"],
                "recommendation": "stratification_warranted"
            }
        }
    }

    path = tmp_path / "eval_permute_report.json"
    with open(path, "w") as f:
        json.dump(report, f)
    return path


@pytest.fixture
def input_parquet_path(tmp_path):
    # Construct dataframe with ~250 rows to test chunks at size=100
    np.random.seed(42)

    regions = [
        "TRANS", "CIS5", "PROMOTER", "GENEBODY", "CIS3", "DISTAL5", "DISTAL3", None
    ]

    # 250 rows, randomly pick a region
    row_regions = np.random.choice(regions, size=250)

    df = pd.DataFrame({
        "mt_id": [f"cg{i:05d}" for i in range(250)],
        "gt_id": [f"ENSG{i:010d}" for i in range(250)],
        "region": row_regions,
        "precise_mt_p": np.random.uniform(0.0001, 1.0, size=250),
        "mt_chromStart": np.random.randint(1000, 100000, size=250),
    })

    # Inject one specific nan for precise_mt_p just to ensure handling of null source
    df.loc[10, "precise_mt_p"] = np.nan

    path = tmp_path / "input.parquet"
    df.to_parquet(path)
    return path


def run_tool(tmp_path, input_file, report_file, output_file, extra_args=None):
    cmd = [
        sys.executable, TOOL_PATH,
        "--input", str(input_file),
        "--eval-report", str(report_file),
        "--output", str(output_file)
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        capture_output=True,
        text=True
    )
    return result


def test_licensed_regions_are_populated(tmp_path, input_parquet_path, eval_report_path):
    out = tmp_path / "out.parquet"
    res = run_tool(tmp_path, input_parquet_path, eval_report_path, out)
    assert res.returncode == 0

    df = pd.read_parquet(out)

    # TRANS and CIS5 and CIS3 are licensed
    licensed = ["TRANS", "CIS5", "CIS3"]
    mask = df["region"].isin(licensed)

    # All rows in licensed should have p_permute match precise_mt_p
    # Account for nan in source! np.nan != np.nan so use np.allclose or pd.isna
    # But we can just use assert_series_equal for exact match including nan handling
    pd.testing.assert_series_equal(
        df.loc[mask, "p_permute"],
        df.loc[mask, "precise_mt_p"],
        check_names=False
    )

    # Ensure they aren't all null just in case
    assert not df.loc[mask, "p_permute"].isna().all()


def test_divergent_region_is_null_despite_ok_status(tmp_path, input_parquet_path, eval_report_path):

    out = tmp_path / "out.parquet"
    res = run_tool(tmp_path, input_parquet_path, eval_report_path, out)
    assert res.returncode == 0

    df = pd.read_parquet(out)

    # PROMOTER has status 'ok' but is divergent
    mask = df["region"] == "PROMOTER"

    assert mask.any(), "Need PROMOTER rows for the test"
    assert df.loc[mask, "p_permute"].isna().all(), "Divergent region must be null"


def test_insufficient_data_region_is_null(tmp_path, input_parquet_path, eval_report_path):
    out = tmp_path / "out.parquet"
    res = run_tool(tmp_path, input_parquet_path, eval_report_path, out)
    assert res.returncode == 0

    df = pd.read_parquet(out)

    # GENEBODY has status 'insufficient_data'
    mask = df["region"] == "GENEBODY"
    assert mask.any()
    assert df.loc[mask, "p_permute"].isna().all()


def test_region_absent_from_report_is_null(tmp_path, input_parquet_path, eval_report_path):
    out = tmp_path / "out.parquet"
    res = run_tool(tmp_path, input_parquet_path, eval_report_path, out)
    assert res.returncode == 0

    df = pd.read_parquet(out)

    # DISTAL5 and DISTAL3 are absent from the per_region dict
    mask = df["region"].isin(["DISTAL5", "DISTAL3"])
    assert mask.any()
    assert df.loc[mask, "p_permute"].isna().all()


def test_null_region_rows_are_null(tmp_path, input_parquet_path, eval_report_path):
    out = tmp_path / "out.parquet"
    res = run_tool(tmp_path, input_parquet_path, eval_report_path, out)
    assert res.returncode == 0

    df = pd.read_parquet(out)

    mask = df["region"].isna()
    assert mask.any()
    assert df.loc[mask, "p_permute"].isna().all()


def test_write_is_additive(tmp_path, input_parquet_path, eval_report_path):
    out = tmp_path / "out.parquet"
    res = run_tool(tmp_path, input_parquet_path, eval_report_path, out)
    assert res.returncode == 0

    df_in = pd.read_parquet(input_parquet_path)
    df_out = pd.read_parquet(out)

    # Check byte-identical columns order and value, one new column appended
    expected_cols = list(df_in.columns) + ["p_permute"]
    assert list(df_out.columns) == expected_cols

    # Check values for original columns
    # Re-cast Int64 back to int64 for the comparison if pyarrow upcast it per the summarizeOutput_parquet contract
    if "mt_chromStart" in df_in.columns and df_out["mt_chromStart"].dtype == "Int64":
        df_out["mt_chromStart"] = df_out["mt_chromStart"].astype("int64")

    pd.testing.assert_frame_equal(df_in, df_out[df_in.columns])


def test_refuses_to_overwrite_existing_p_column(tmp_path, input_parquet_path, eval_report_path):
    # First, modify input to already have p_permute

    df = pd.read_parquet(input_parquet_path)
    df["p_permute"] = 0.5
    modified_in = tmp_path / "mod_in.parquet"
    df.to_parquet(modified_in)

    out = tmp_path / "out.parquet"
    res = run_tool(tmp_path, modified_in, eval_report_path, out)

    assert res.returncode != 0
    assert "already present" in res.stderr
    assert not out.exists()


def test_fails_closed_when_stratify_mode_absent(tmp_path, input_parquet_path, eval_report_path):

    with open(eval_report_path, "r") as f:
        report = json.load(f)

    del report["arms"]["stratify_decision"]["mode"]

    with open(eval_report_path, "w") as f:
        json.dump(report, f)

    out = tmp_path / "out.parquet"
    res = run_tool(tmp_path, input_parquet_path, eval_report_path, out)

    assert res.returncode != 0
    assert "missing or invalid stratify mode" in res.stderr
    assert not out.exists()


def test_fails_closed_when_p_source_missing(tmp_path, input_parquet_path, eval_report_path):
    df = pd.read_parquet(input_parquet_path)
    df = df.drop(columns=["precise_mt_p"])
    modified_in = tmp_path / "mod_in.parquet"
    df.to_parquet(modified_in)

    out = tmp_path / "out.parquet"
    res = run_tool(tmp_path, modified_in, eval_report_path, out)

    assert res.returncode != 0
    assert "absent from input" in res.stderr
    assert not out.exists()


def test_fails_closed_on_unrecognized_region_label_prevents_partial_output(tmp_path, input_parquet_path, eval_report_path):
    df = pd.read_parquet(input_parquet_path)
    # Inject unknown region into a later chunk (e.g. chunk 2 at size 100)
    df.loc[150, "region"] = "UNKNOWN_REG"
    modified_in = tmp_path / "mod_in.parquet"
    df.to_parquet(modified_in)

    out = tmp_path / "out.parquet"
    res = run_tool(tmp_path, modified_in, eval_report_path, out, ["--chunk-size", "100"])

    assert res.returncode != 0
    assert "Unrecognized region label" in res.stderr
    assert not out.exists()


def test_multi_chunk_result_matches_single_chunk(tmp_path, input_parquet_path, eval_report_path):
    out_single = tmp_path / "out_single.parquet"
    out_multi = tmp_path / "out_multi.parquet"

    res_single = run_tool(tmp_path, input_parquet_path, eval_report_path, out_single, ["--chunk-size", "1000"])
    assert res_single.returncode == 0

    res_multi = run_tool(tmp_path, input_parquet_path, eval_report_path, out_multi, ["--chunk-size", "100"])
    assert res_multi.returncode == 0

    df_single = pd.read_parquet(out_single)
    df_multi = pd.read_parquet(out_multi)

    pd.testing.assert_frame_equal(df_single, df_multi)
