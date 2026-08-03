"""Guards for worker-result aggregation in tools/summarizeOutput_parquet.py.

A worker that raises was previously caught, printed to stdout, and skipped. Its
rows then never reached the pair and gene counts, the p-value histogram, the
BH-FDR pool, or the FDR-universe accounting, and the run still exited zero. The
resulting summary and FDR threshold are computed from an arbitrary subset of the
input while presenting as complete. These tests pin the replacement contract: a
worker failure aborts, names the chunk, and produces no summary and no output
file.
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
BAD_CHUNK_INDEX = 1


def _write_inputs(d, p_values, p_type):
    """Two row groups of two rows each, so chunk 0 and chunk 1 are distinct."""
    schema = pa.schema([("id", pa.int64()), ("mt_id", pa.string()),
                        ("gt_id", pa.string()), ("precise_mt_p", p_type)])
    frames = [
        pd.DataFrame({"id": [1, 2], "mt_id": ["cg1", "cg2"],
                      "gt_id": ["g1", "g2"], "precise_mt_p": p_values[:2]}),
        pd.DataFrame({"id": [3, 4], "mt_id": ["cg3", "cg4"],
                      "gt_id": ["g3", "g4"], "precise_mt_p": p_values[2:]}),
    ]
    inp = os.path.join(d, "input.parquet")
    with pq.ParquetWriter(inp, schema) as w:
        for f in frames:
            w.write_table(pa.Table.from_pandas(f, schema=schema))
    res = os.path.join(d, "reservoir.csv")
    pd.DataFrame({"id": [1, 2, 3, 4],
                  "precise_mt_p": [0.01, 0.05, 0.10, 0.20]}).to_csv(res, index=False)
    return inp, res


def _good_inputs(d):
    return _write_inputs(d, [0.01, 0.05, 0.10, 0.20], pa.float64())


def _inputs_failing_second_chunk(d):
    """A p-column typed as string whose second row group cannot be cast."""
    return _write_inputs(d, ["0.01", "0.05", "not_a_number", "0.20"], pa.string())


def _run(d, inp, res, out=None):
    cmd = [sys.executable, TOOL, "--main-file", inp, "--reservoir-file", res,
           "--total-tests", str(N_ROWS), "--df", "100", "--calculate-fdr",
           "--chunk-size", "2"]
    if out is not None:
        cmd.extend(["--output-fdr-file", out])
    return subprocess.run(cmd, capture_output=True, text=True, cwd=d)


def test_all_chunks_aggregate_when_every_worker_succeeds(tmp_path):
    """The happy path still counts every row."""
    d = str(tmp_path)
    inp, res = _good_inputs(d)
    proc = _run(d, inp, res)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert f"Total mapping pairs (eCpGs): {N_ROWS}" in proc.stdout


def test_worker_failure_exits_non_zero(tmp_path):
    """A chunk that never arrives must not read as a completed run."""
    d = str(tmp_path)
    inp, res = _inputs_failing_second_chunk(d)
    proc = _run(d, inp, res)
    assert proc.returncode == 1


def test_worker_failure_names_the_chunk_on_stderr(tmp_path):
    """Which chunk was lost is the whole diagnostic; stdout is not where a
    caller separating streams will look for it."""
    d = str(tmp_path)
    inp, res = _inputs_failing_second_chunk(d)
    proc = _run(d, inp, res)
    assert proc.returncode == 1
    assert f"chunk {BAD_CHUNK_INDEX}" in proc.stderr
    assert "Error retrieving result from worker" in proc.stderr
    assert "Error retrieving result from worker" not in proc.stdout


def test_worker_failure_suppresses_summary_and_output_file(tmp_path):
    """Nothing computed from a partial aggregate may be presented or written."""
    d = str(tmp_path)
    inp, res = _inputs_failing_second_chunk(d)
    out = os.path.join(d, "out.parquet")
    proc = _run(d, inp, res, out=out)
    assert proc.returncode == 1
    assert "Summary of Results" not in proc.stdout
    assert "Total mapping pairs" not in proc.stdout
    assert not os.path.exists(out)
    assert not os.path.exists(out + ".tmp")
