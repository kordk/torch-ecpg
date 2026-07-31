"""Guards for tools/annotate_bootstrap_concordance.py.

The tool adds four raw bootstrap-vs-analytic scores and summarizes their
distributions. It sets no thresholds, raises no flags, and removes no rows;
these tests pin that scope as much as they pin the arithmetic.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO_ROOT, "tools", "annotate_bootstrap_concordance.py")

SCORE_COLUMNS = [
    "boot_se_ratio",
    "boot_bias_ratio",
    "boot_ci_skew",
    "boot_p_floor_gap",
]

N_ROWS = 60
N_BOOTSTRAPPED = 40

# Row indices carrying deliberate edge cases, and the score each one nulls.
EDGE_CASES = {
    1: "boot_se_ratio",      # mt_err == 0
    2: "boot_bias_ratio",    # mt_est_boot_std == 0
    3: "boot_ci_skew",       # zero-width CI
    4: "boot_p_floor_gap",   # p_boot == 0
}


def _write_fixture(path):
    rng = np.random.default_rng(3)
    n = N_ROWS

    mt_est = rng.normal(0.3, 0.1, n)
    mt_err = np.abs(rng.normal(0.05, 0.01, n))
    mt_t = mt_est / mt_err
    regions = rng.choice(["TRANS", "CIS5", "PROMOTER", "GENEBODY"], n)

    b_std = mt_err * rng.normal(1.0, 0.06, n)
    b_std[5] = mt_err[5] * 1.9                       # inflated bootstrap SE
    b_mean = mt_est + b_std * rng.normal(0, 0.02, n)
    ci_low = mt_est - 1.96 * b_std
    ci_high = mt_est + 1.96 * b_std
    ci_high[7] += 3 * b_std[7]                       # skewed CI
    p_boot = np.maximum(2 * stats.norm.sf(np.abs(mt_t)), 1e-3)
    p_boot[9] = 0.02                                 # off the floor

    for arr in (b_std, b_mean, ci_low, ci_high, p_boot):
        arr[N_BOOTSTRAPPED:] = np.nan                # not bootstrapped

    mt_err[1] = 0.0
    b_std[2] = 0.0
    ci_low[3] = ci_high[3] = mt_est[3]
    p_boot[4] = 0.0
    mt_t[6] = 60.0                                   # 2*sf(|t|) underflows

    df = pd.DataFrame({
        "mt_id": [f"cg{i:04d}" for i in range(n)],
        "gt_id": [f"G{i % 7}" for i in range(n)],
        "region": regions,
        "mt_est": mt_est, "mt_err": mt_err, "mt_t": mt_t,
        "precise_mt_p": 2 * stats.t.sf(np.abs(mt_t), 321),
        "mt_est_boot_mean": b_mean, "mt_est_boot_std": b_std,
        "ci_low": ci_low, "ci_high": ci_high, "p_boot": p_boot,
        "degenerate_resamples": np.where(np.isnan(b_std), np.nan, 0.0),
    })
    pq.write_table(pa.Table.from_pandas(df), path)
    return df


def _run(inp, out, summary=None, extra=None, expect_ok=True):
    cmd = [sys.executable, TOOL, "-i", str(inp), "-o", str(out), "--chunk-size", "25"]
    if summary:
        cmd += ["-s", str(summary)]
    if extra:
        cmd += extra
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if expect_ok:
        assert proc.returncode == 0, f"tool failed:\n{proc.stdout}\n{proc.stderr}"
    return proc


@pytest.fixture
def scored(tmp_path):
    inp = tmp_path / "boot.parquet"
    out = tmp_path / "scored.parquet"
    summary = tmp_path / "summary.json"
    src = _write_fixture(inp)
    proc = _run(inp, out, summary)
    return src, pq.read_table(out).to_pandas(), json.loads(summary.read_text()), proc.stdout


def test_scores_are_additive_and_no_rows_lost(scored):
    """Load-bearing scope guard: annotate only. Every input row and column survives."""
    src, out, _, _ = scored
    assert len(out) == len(src) == N_ROWS
    for col in src.columns:
        assert col in out.columns, f"input column {col} was dropped"
    for col in SCORE_COLUMNS:
        assert col in out.columns
    assert list(out["mt_id"]) == list(src["mt_id"])


def test_se_ratio_matches_hand_computation(scored):
    """boot_se_ratio == mt_est_boot_std / mt_err on a row with no edge case."""
    src, out, _, _ = scored
    i = 10
    expected = src["mt_est_boot_std"][i] / src["mt_err"][i]
    assert out["boot_se_ratio"][i] == pytest.approx(expected, rel=1e-12)


def test_ci_skew_matches_hand_computation(scored):
    """boot_ci_skew is the normalized asymmetry of the bootstrap CI about mt_est."""
    src, out, _, _ = scored
    i = 7
    hi, lo, est = src["ci_high"][i], src["ci_low"][i], src["mt_est"][i]
    expected = ((hi - est) - (est - lo)) / (hi - lo)
    assert out["boot_ci_skew"][i] == pytest.approx(expected, rel=1e-12)
    assert out["boot_ci_skew"][i] > 0.4          # the injected skew is visible


def test_scores_null_where_bootstrap_did_not_run(scored):
    """Coverage must match p_boot's: null outside the bootstrapped block."""
    _, out, _, _ = scored
    for col in SCORE_COLUMNS:
        assert out[col][N_BOOTSTRAPPED:].isna().all(), f"{col} populated without a bootstrap"


def test_edge_cases_yield_null_not_inf(scored):
    """Zero denominators and non-positive p_boot must produce NaN, never inf."""
    _, out, _, _ = scored
    for row, col in EDGE_CASES.items():
        v = out[col][row]
        assert not np.isfinite(v), f"{col} row {row} should be null, got {v}"
    for col in SCORE_COLUMNS:
        vals = out[col].to_numpy(dtype=np.float64)
        assert not np.isinf(vals).any(), f"{col} contains inf"


def test_large_t_does_not_underflow(scored):
    """boot_p_floor_gap uses logsf, so |t|=60 stays finite where 2*sf would be 0."""
    _, out, _, _ = scored
    assert np.isfinite(out["boot_p_floor_gap"][6])
    assert 2 * stats.norm.sf(60.0) == 0.0        # the naive form really does underflow


def test_summary_reports_raw_distribution_not_cutpoints(scored):
    """The summary must carry observed percentiles and no threshold vocabulary."""
    _, _, summary, stdout = scored
    assert summary["total_rows"] == N_ROWS
    assert summary["n_bootstrapped"] == N_BOOTSTRAPPED
    for col in SCORE_COLUMNS:
        s = summary["scores"][col]
        assert s is not None and s["n_finite"] > 0
        for p in ("1", "5", "25", "50", "75", "95", "99"):
            assert p in s["percentiles"]
    # Structural scope guard: the summary may *say* it applies no thresholds, but
    # it must not carry a field that encodes one. Check keys, not prose values.
    def _keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from _keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from _keys(v)

    banned = ("threshold", "flag", "cutoff", "cut_point", "status", "badge",
              "verdict", "outlier", "problematic")
    for key in _keys(summary):
        low = key.lower()
        for word in banned:
            assert word not in low, f"summary carries a decision field: {key!r}"


def test_existing_score_column_fails_closed(tmp_path):
    """Additive-writes-only: refuse to overwrite a score column that already exists."""
    inp = tmp_path / "boot.parquet"
    _write_fixture(inp)
    df = pq.read_table(inp).to_pandas()
    df["boot_se_ratio"] = 1.0
    pq.write_table(pa.Table.from_pandas(df), inp)
    proc = _run(inp, tmp_path / "out.parquet", expect_ok=False)
    assert proc.returncode == 1
    assert "additive" in proc.stderr.lower()
    assert not (tmp_path / "out.parquet").exists()


def test_missing_required_column_fails_closed(tmp_path):
    """A missing bootstrap input must abort, not silently produce all-null scores."""
    inp = tmp_path / "boot.parquet"
    _write_fixture(inp)
    df = pq.read_table(inp).to_pandas().drop(columns=["mt_err"])
    pq.write_table(pa.Table.from_pandas(df), inp)
    proc = _run(inp, tmp_path / "out.parquet", expect_ok=False)
    assert proc.returncode == 1
    assert "mt_err" in proc.stderr
    assert not (tmp_path / "out.parquet").exists()
