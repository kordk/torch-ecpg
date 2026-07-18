import pytest
import subprocess
import pandas as pd
import numpy as np

def setup_data(tmp_path, rows, cols):
    df = pd.DataFrame(np.random.rand(rows, cols))
    df.index = [f"loc_{i}" for i in range(rows)]
    df.columns = [f"sample_{j}" for j in range(cols)]
    input_file = tmp_path / "input.csv"
    df.to_csv(input_file)
    return input_file, df

def run_script(input_csv, output_csv, n_loci, seed=42):
    cmd = ["python3", "tools/subsample_loci.py", str(input_csv), str(output_csv), str(n_loci), "--seed", str(seed)]
    return subprocess.run(cmd, capture_output=True, text=True)

def test_subsample_normal(tmp_path):
    input_csv, orig_df = setup_data(tmp_path, 50, 8)
    output_csv = tmp_path / "output.csv"

    result = run_script(input_csv, output_csv, 10)
    assert result.returncode == 0

    out_df = pd.read_csv(output_csv, index_col=0)
    assert out_df.shape == (10, 8)
    # Check index subset
    assert set(out_df.index).issubset(set(orig_df.index))

def test_subsample_reproducibility(tmp_path):
    input_csv, _ = setup_data(tmp_path, 40, 8)
    out1 = tmp_path / "out1.csv"
    out2 = tmp_path / "out2.csv"

    run_script(input_csv, out1, 5, seed=123)
    run_script(input_csv, out2, 5, seed=123)

    df1 = pd.read_csv(out1, index_col=0)
    df2 = pd.read_csv(out2, index_col=0)

    pd.testing.assert_frame_equal(df1, df2)

def test_subsample_passthrough(tmp_path):
    input_csv, orig_df = setup_data(tmp_path, 20, 5)
    output_csv = tmp_path / "output.csv"

    result = run_script(input_csv, output_csv, 25)
    assert result.returncode == 0
    assert "... already <= n_loci; taking all N rows unchanged" in result.stdout

    out_df = pd.read_csv(output_csv, index_col=0)
    assert out_df.shape == (20, 5)
    pd.testing.assert_frame_equal(orig_df, out_df)

def test_subsample_passthrough_exact(tmp_path):
    input_csv, orig_df = setup_data(tmp_path, 20, 5)
    output_csv = tmp_path / "output.csv"

    result = run_script(input_csv, output_csv, 20)
    assert result.returncode == 0
    assert "... already <= n_loci; taking all N rows unchanged" in result.stdout

    out_df = pd.read_csv(output_csv, index_col=0)
    assert out_df.shape == (20, 5)
    pd.testing.assert_frame_equal(orig_df, out_df)

def test_subsample_fail_zero(tmp_path):
    input_csv, _ = setup_data(tmp_path, 10, 5)
    output_csv = tmp_path / "output.csv"

    result = run_script(input_csv, output_csv, 0)
    assert result.returncode == 1
    assert "Error: Target number of loci must be positive" in result.stdout

def test_subsample_inplace(tmp_path):
    input_csv, orig_df = setup_data(tmp_path, 30, 6)

    result = run_script(input_csv, input_csv, 15)
    assert result.returncode == 0

    out_df = pd.read_csv(input_csv, index_col=0)
    assert out_df.shape == (15, 6)
