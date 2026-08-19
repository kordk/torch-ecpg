import json
import os
import subprocess
import pytest
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

@pytest.fixture
def run_tool():
    def _run(*args, **kwargs):
        cmd = ["python3", "tools/flagInfluence_parquet.py"] + list(args)
        return subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    return _run

@pytest.fixture
def setup_data(tmp_path):
    n_subjects = 40
    n_covariates = 2

    # 1. create covariate csv with one extreme value subject
    # C has columns c1, c2.
    np.random.seed(42)
    C_data = np.random.randn(n_subjects, n_covariates)
    C_data[0, 0] = 50.0  # extreme covariate value
    C = pd.DataFrame(C_data, columns=["c1", "c2"])
    C.index = [f"sub_{i}" for i in range(n_subjects)]
    cov_path = tmp_path / "C.csv"
    C.to_csv(cov_path)

    # calc expected h_C_max
    Xc = np.hstack([np.ones((n_subjects, 1)), C.to_numpy()])
    Q, _ = np.linalg.qr(Xc, mode='reduced')
    expected_h_c_max = float((Q * Q).sum(axis=1).max())

    # 2. create master parquet: 4 CpGs * 3 genes = 12 rows, plus one extra null mt_h_max row (5th CpG)
    cpg_h_max = {
        "cg1": 0.19,
        "cg2": 0.19,
        "cg3": 0.50,
        "cg4": 0.95
    }

    rows = []
    regions = ["TRANS", "PROMOTER", "CIS5"]
    fdr_vals = [0.01, 0.5, 0.9] # one sig, two not sig

    for cg, h in cpg_h_max.items():
        for i, (r, fdr) in enumerate(zip(regions, fdr_vals)):
            rows.append({
                "mt_id": cg,
                "mt_h_max": h,
                "region": r,
                "fdr_est": fdr,
                "other_col": f"val_{cg}_{i}"
            })

    # Add null CpG
    rows.append({
        "mt_id": "cg5_null",
        "mt_h_max": None,
        "region": "TRANS",
        "fdr_est": 0.01,
        "other_col": "val_cg5_0"
    })

    df = pd.DataFrame(rows)
    pq_path = tmp_path / "master.parquet"

    table = pa.Table.from_pandas(df)
    meta = {b"some_existing_meta": b"value"}
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, pq_path)

    return {
        "C_path": str(cov_path),
        "pq_path": str(pq_path),
        "expected_h_c_max": expected_h_c_max,
        "out_dir": str(tmp_path / "report_dir"),
        "tmp_path": tmp_path
    }

def test_report_only_writes_report_not_parquet(run_tool, setup_data):
    res = run_tool("-i", setup_data["pq_path"], "--report-dir", setup_data["out_dir"])
    assert res.returncode == 0

    assert os.path.exists(os.path.join(setup_data["out_dir"], "influence_qc.json"))
    assert os.path.exists(os.path.join(setup_data["out_dir"], "influence_qc.md"))

    with open(os.path.join(setup_data["out_dir"], "influence_qc.json")) as f:
        rep = json.load(f)
    assert isinstance(rep["pngs_written"], bool)

def test_pngs_written_when_matplotlib_available(run_tool, setup_data):
    pytest.importorskip("matplotlib")
    res = run_tool("-i", setup_data["pq_path"], "--report-dir", setup_data["out_dir"])
    assert res.returncode == 0

    assert os.path.exists(os.path.join(setup_data["out_dir"], "h_max_hist.png"))

    with open(os.path.join(setup_data["out_dir"], "influence_qc.json")) as f:
        rep = json.load(f)
    assert rep["pngs_written"] is True

def test_h_c_max_matches_numpy_reference(run_tool, setup_data):
    res = run_tool("-i", setup_data["pq_path"], "-c", setup_data["C_path"], "--report-dir", setup_data["out_dir"])
    assert res.returncode == 0

    with open(os.path.join(setup_data["out_dir"], "influence_qc.json")) as f:
        rep = json.load(f)

    assert np.isclose(rep["header"]["h_C_max"], setup_data["expected_h_c_max"], atol=1e-9)

def test_abs_rule_flags_strictly_greater(run_tool, setup_data):
    out_pq = os.path.join(setup_data["tmp_path"], "out.parquet")
    res = run_tool("-i", setup_data["pq_path"], "-o", out_pq, "--report-dir", setup_data["out_dir"],
                   "--rule", "abs", "--threshold", "0.5")
    assert res.returncode == 0

    df = pd.read_parquet(out_pq)

    cg4_flags = df[df["mt_id"] == "cg4"]["mt_influence_flag"]
    assert all(cg4_flags == True)

    cg3_flags = df[df["mt_id"] == "cg3"]["mt_influence_flag"]
    assert all(cg3_flags == False)

    cg1_flags = df[df["mt_id"] == "cg1"]["mt_influence_flag"]
    assert all(cg1_flags == False)

    cg5_flags = df[df["mt_id"] == "cg5_null"]["mt_influence_flag"]
    assert all(cg5_flags.isna()) # using isna() for nullable bool

def test_floor_rule_flags_expected(run_tool, setup_data):
    # delta chosen so only cg4 (0.95) exceeds h_C_max + delta
    # h_C_max is roughly 0.6 from extreme val,
    # cg3 = 0.5 < h_C_max + delta (0.6 + 0.1)
    # cg4 = 0.95 > 0.6 + 0.1
    delta = -0.04

    out_pq = os.path.join(setup_data["tmp_path"], "out.parquet")
    res = run_tool("-i", setup_data["pq_path"], "-c", setup_data["C_path"],
                   "-o", out_pq, "--report-dir", setup_data["out_dir"],
                   "--rule", "floor", "--threshold", str(delta))
    assert res.returncode == 0

    df = pd.read_parquet(out_pq)

    cg4_flags = df[df["mt_id"] == "cg4"]["mt_influence_flag"]
    assert all(cg4_flags == True)

    cg3_flags = df[df["mt_id"] == "cg3"]["mt_influence_flag"]
    assert all(cg3_flags == False)

def test_output_columns_additive_and_flag_last(run_tool, setup_data):
    out_pq = os.path.join(setup_data["tmp_path"], "out.parquet")
    res = run_tool("-i", setup_data["pq_path"], "-o", out_pq, "--report-dir", setup_data["out_dir"],
                   "--rule", "abs", "--threshold", "0.5")
    assert res.returncode == 0

    in_pq = pq.ParquetFile(setup_data["pq_path"])
    out_pq_file = pq.ParquetFile(out_pq)

    in_cols = in_pq.schema_arrow.names
    out_cols = out_pq_file.schema_arrow.names

    assert out_cols[:-1] == in_cols
    assert out_cols[-1] == "mt_influence_flag"
    assert in_pq.metadata.num_rows == out_pq_file.metadata.num_rows

def test_metadata_stamped(run_tool, setup_data):
    out_pq = os.path.join(setup_data["tmp_path"], "out.parquet")
    res = run_tool("-i", setup_data["pq_path"], "-c", setup_data["C_path"],
                   "-o", out_pq, "--report-dir", setup_data["out_dir"],
                   "--rule", "floor", "--threshold", "0.1")
    assert res.returncode == 0

    out_pq_file = pq.ParquetFile(out_pq)
    meta = out_pq_file.schema_arrow.metadata

    assert b'tecpg_influence_rule' in meta
    assert meta[b'tecpg_influence_rule'] == b'floor'

    assert b'tecpg_influence_threshold' in meta
    assert meta[b'tecpg_influence_threshold'] == b'0.1'

    assert b'tecpg_influence_h_c_max' in meta

    assert b'tecpg_influence_flag_column' in meta
    assert meta[b'tecpg_influence_flag_column'] == b'mt_influence_flag'

    assert b'tecpg_influence_n_cpgs' in meta
    assert meta[b'tecpg_influence_n_cpgs'] == b'5'

    assert b'tecpg_influence_n_cpgs_flagged' in meta

    assert b'tecpg_influence_source' in meta
    assert meta[b'tecpg_influence_source'] == b'master.parquet'

    assert b'some_existing_meta' in meta

def test_refuses_existing_flag_column(run_tool, setup_data):
    out_pq = os.path.join(setup_data["tmp_path"], "out.parquet")
    res = run_tool("-i", setup_data["pq_path"], "-o", out_pq, "--report-dir", setup_data["out_dir"],
                   "--rule", "abs", "--threshold", "0.5")
    assert res.returncode == 0

    res2 = run_tool("-i", out_pq, "-o", os.path.join(setup_data["tmp_path"], "out2.parquet"),
                    "--report-dir", setup_data["out_dir"], "--rule", "abs", "--threshold", "0.5")
    assert res2.returncode == 1
    assert "already present" in res2.stderr

def test_fails_closed_missing_mt_h_max(run_tool, setup_data):
    df = pd.read_parquet(setup_data["pq_path"])
    df = df.drop(columns=["mt_h_max"])
    bad_pq = os.path.join(setup_data["tmp_path"], "bad.parquet")
    df.to_parquet(bad_pq)

    res = run_tool("-i", bad_pq, "--report-dir", setup_data["out_dir"])
    assert res.returncode == 1
    assert "absent" in res.stderr

def test_fails_closed_nonconstant_h_max(run_tool, setup_data):
    df = pd.read_parquet(setup_data["pq_path"])
    df.loc[df["mt_id"] == "cg1", "mt_h_max"] = [0.19, 0.191, 0.19]
    bad_pq = os.path.join(setup_data["tmp_path"], "bad.parquet")
    df.to_parquet(bad_pq)

    res = run_tool("-i", bad_pq, "--report-dir", setup_data["out_dir"])
    assert res.returncode == 1
    assert "not constant" in res.stderr

def test_output_requires_rule_and_threshold(run_tool, setup_data):
    out_pq = os.path.join(setup_data["tmp_path"], "out.parquet")

    res = run_tool("-i", setup_data["pq_path"], "-o", out_pq, "--report-dir", setup_data["out_dir"])
    assert res.returncode == 1
    assert "requires both" in res.stderr

    res = run_tool("-i", setup_data["pq_path"], "-o", out_pq, "--report-dir", setup_data["out_dir"],
                   "--rule", "abs")
    assert res.returncode == 1
    assert "requires both" in res.stderr

def test_floor_requires_covariates(run_tool, setup_data):
    out_pq = os.path.join(setup_data["tmp_path"], "out.parquet")
    res = run_tool("-i", setup_data["pq_path"], "-o", out_pq, "--report-dir", setup_data["out_dir"],
                   "--rule", "floor", "--threshold", "0.1")
    assert res.returncode == 1
    assert "requires -c" in res.stderr

def test_multiindex_master_with_null_row(run_tool, setup_data):
    df = pd.DataFrame([
        {"gt_id": "g1", "mt_id": "cg_null", "mt_h_max": None},
        {"gt_id": "g1", "mt_id": "cg1", "mt_h_max": 0.5},
    ])
    df = df.set_index(["gt_id", "mt_id"])

    bad_pq = os.path.join(setup_data["tmp_path"], "multi.parquet")
    df.to_parquet(bad_pq)

    out_pq = os.path.join(setup_data["tmp_path"], "multi_out.parquet")

    res1 = run_tool("-i", bad_pq, "--report-dir", setup_data["out_dir"])
    assert res1.returncode == 0

    res2 = run_tool("-i", bad_pq, "-o", out_pq, "--report-dir", setup_data["out_dir"],
                    "--rule", "abs", "--threshold", "0.2")
    assert res2.returncode == 0

    out_df = pd.read_parquet(out_pq)
    assert out_df.index.names == ["gt_id", "mt_id"]

    null_flag = out_df.loc[("g1", "cg_null"), "mt_influence_flag"]
    assert pd.isna(null_flag)

def test_sweep_tables_present_and_monotone(run_tool, setup_data):
    res = run_tool("-i", setup_data["pq_path"], "-c", setup_data["C_path"], "--report-dir", setup_data["out_dir"])
    assert res.returncode == 0

    with open(os.path.join(setup_data["out_dir"], "influence_qc.json")) as f:
        rep = json.load(f)

    abs_vals = [rep["sweep_abs"][str(tau)]["frac_cpgs_flagged"] for tau in [0.3, 0.5, 0.7, 0.9, 0.95]]
    for i in range(len(abs_vals) - 1):
        assert abs_vals[i] >= abs_vals[i+1]

    floor_vals = [rep["sweep_floor"][str(delta)]["frac_cpgs_flagged"] for delta in [0.05, 0.10, 0.20, 0.30, 0.50]]
    for i in range(len(floor_vals) - 1):
        assert floor_vals[i] >= floor_vals[i+1]
