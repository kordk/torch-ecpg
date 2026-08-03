"""Guards for the FDR output write path in tools/summarizeOutput_parquet.py.

The write loop previously caught every exception, printed to stdout, closed the
writer in a finally block and returned zero. Because the writer pointed at the
destination, a failure part way through left a valid but truncated parquet at
the path a downstream stage would read, with a success exit code. These tests
pin the replacement contract: the destination is written only after the whole
file is, and a failure is loud and non-zero.
"""
import os
import subprocess
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO_ROOT, "tools", "summarizeOutput_parquet.py")

N_ROWS = 4


def _make_inputs(d):
    schema = pa.schema([
        ("id", pa.int64()),
        ("mt_chromStart", pa.int64()),
        ("gt_chromStart", pa.int64()),
        ("precise_mt_p", pa.float64()),
    ])
    df1 = pd.DataFrame({"id": [1, 2], "mt_chromStart": [100, 200],
                        "gt_chromStart": [100, 200], "precise_mt_p": [0.01, 0.05]})
    df2 = pd.DataFrame({"id": [3, 4], "mt_chromStart": [300, 400],
                        "gt_chromStart": [300, 400], "precise_mt_p": [0.10, 0.20]})
    inp = os.path.join(d, "input.parquet")
    with pq.ParquetWriter(inp, schema) as w:
        w.write_table(pa.Table.from_pandas(df1, schema=schema))
        w.write_table(pa.Table.from_pandas(df2, schema=schema))
    res = os.path.join(d, "reservoir.csv")
    pd.DataFrame({"id": [1, 2, 3, 4],
                  "precise_mt_p": [0.01, 0.05, 0.10, 0.20]}).to_csv(res, index=False)
    return inp, res


def _run(d, out):
    inp, res = _make_inputs(d)
    return subprocess.run(
        [sys.executable, TOOL, "--main-file", inp, "--reservoir-file", res,
         "--total-tests", str(N_ROWS), "--df", "100", "--calculate-fdr",
         "--output-fdr-file", out, "--chunk-size", "2"],
        capture_output=True, text=True, cwd=d)


def test_success_writes_destination_and_leaves_no_scratch_file(tmp_path):
    """A clean run still produces the file, and clears up after itself."""
    out = str(tmp_path / "out.parquet")
    proc = _run(str(tmp_path), out)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert os.path.exists(out)
    assert pq.read_table(out).num_rows == N_ROWS
    assert not os.path.exists(out + ".tmp")


def test_scratch_path_is_used_so_the_destination_is_never_partial(tmp_path):
    """Occupying the scratch path must abort before the destination is touched.

    This fails if the writer points straight at the destination, because then
    the scratch path is irrelevant and the run succeeds.
    """
    out = str(tmp_path / "out.parquet")
    os.mkdir(out + ".tmp")
    proc = _run(str(tmp_path), out)
    assert proc.returncode == 1
    assert not os.path.exists(out)


def test_unwritable_destination_exits_non_zero(tmp_path):
    """A write that cannot start must not report success."""
    out = str(tmp_path / "no_such_dir" / "out.parquet")
    proc = _run(str(tmp_path), out)
    assert proc.returncode == 1
    assert not os.path.exists(out)


def test_write_failure_is_reported_on_stderr(tmp_path):
    """A failure printed to stdout is invisible to a shell checking for errors."""
    out = str(tmp_path / "out.parquet")
    os.mkdir(out + ".tmp")
    proc = _run(str(tmp_path), out)
    assert proc.returncode == 1
    assert "Error writing output FDR file" in proc.stderr
    assert "Error writing output FDR file" not in proc.stdout
