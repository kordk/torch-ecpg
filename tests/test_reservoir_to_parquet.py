import pandas as pd
import pytest
import subprocess
import sys
from pathlib import Path

def test_reservoir_to_parquet_success(tmp_path):
    csv_path = tmp_path / "sample_reservoir.csv"
    parquet_path = tmp_path / "reservoir_master.parquet"

    df = pd.DataFrame({
        "gt_id": ["g1", "g2"],
        "mt_id": ["m1", "m2"],
        "mt_t": [1.5, 2.5]
    })
    df.to_csv(csv_path, index=False)

    cmd = [
        sys.executable, "tools/reservoir_to_parquet.py",
        "--in", str(csv_path),
        "--out", str(parquet_path)
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    assert parquet_path.exists()
    df_out = pd.read_parquet(parquet_path)
    assert set(df_out.columns).issuperset({"mt_id", "gt_id", "mt_t"})
    assert df_out["mt_t"].tolist() == [1.5, 2.5]

def test_reservoir_to_parquet_missing_col(tmp_path):
    csv_path = tmp_path / "sample_reservoir_bad.csv"
    parquet_path = tmp_path / "reservoir_master_bad.parquet"

    df = pd.DataFrame({
        "gt_id": ["g1", "g2"],
        "mt_id": ["m1", "m2"]
        # mt_t is missing
    })
    df.to_csv(csv_path, index=False)

    cmd = [
        sys.executable, "tools/reservoir_to_parquet.py",
        "--in", str(csv_path),
        "--out", str(parquet_path)
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Reservoir CSV missing required columns" in res.stderr
    assert "mt_t" in res.stderr
