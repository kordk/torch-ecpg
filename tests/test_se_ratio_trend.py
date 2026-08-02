"""Guards for tools/diagnose_se_ratio_trend.py.

The tool reports how boot_se_ratio behaves against |mt_t| within each region.
It computes no threshold and no verdict; these tests pin that scope alongside
the arithmetic. Every expected value is recomputed from the fixture at test
time rather than hardcoded, so the fixture and the assertions cannot drift
apart.
"""
import hashlib
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
TOOL = os.path.join(REPO_ROOT, "tools", "diagnose_se_ratio_trend.py")

MIN_REGION_N = 200

# Fixture composition. TRANS carries a declining r-vs-|t| relationship by
# construction, CIS5 a flat one, PROMOTER is below MIN_REGION_N, and the
# unassigned bucket receives both a NULL label and a non-canonical one.
N_TRANS = 1200
N_CIS5 = 800
N_PROMOTER = 50
N_NULL = 120
N_WEIRD = 60
N_UNSCORED = 300


def _build_frame():
    rng = np.random.default_rng(11)

    def block(n, label, slope):
        t = rng.uniform(3.4, 12.0, n)
        r = 1.35 + slope * t + rng.normal(0.0, 0.02, n)
        return pd.DataFrame({
            "region": pd.Series([label] * n, dtype=object),
            "mt_t": t * rng.choice([-1.0, 1.0], n),
            "boot_se_ratio": r,
        })

    parts = [
        block(N_TRANS, "TRANS", -0.045),
        block(N_CIS5, "CIS5", 0.0),
        block(N_PROMOTER, "PROMOTER", -0.03),
        block(N_NULL, None, -0.01),
        block(N_WEIRD, "WEIRD", -0.01),
    ]
    unscored = block(N_UNSCORED, "TRANS", -0.045)
    unscored["boot_se_ratio"] = np.nan
    parts.append(unscored)

    df = pd.concat(parts, ignore_index=True)
    df["mt_id"] = [f"cg{i:06d}" for i in range(len(df))]
    df["gt_id"] = [f"G{i % 23}" for i in range(len(df))]
    df["boot_bias_ratio"] = 0.0
    return df


def _write_fixture(path):
    df = _build_frame()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    return df


def _run(inp, out_json, expect_ok=True, extra=None):
    cmd = [sys.executable, TOOL, "-i", str(inp), "-s", str(out_json),
           "--min-region-n", str(MIN_REGION_N)]
    if extra:
        cmd.extend(extra)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if expect_ok:
        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    return proc


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


@pytest.fixture(scope="module")
def run_result(tmp_path_factory):
    d = tmp_path_factory.mktemp("se_ratio_trend")
    inp = d / "concordance.parquet"
    df = _write_fixture(inp)
    out_json = d / "summary.json"
    proc = _run(inp, out_json)
    with open(out_json) as fh:
        summary = json.load(fh)
    return df, summary, proc, d


def _scored(df, label):
    if label is None:
        m = df["region"].isna()
    else:
        m = df["region"] == label
    sub = df[m & np.isfinite(df["boot_se_ratio"])]
    return sub


def test_declining_trend_recovered(run_result):
    """TRANS is built with r falling in |t|; rho and its CI must both be negative."""
    _, summary, _, _ = run_result
    reg = summary["regions"]["TRANS"]
    assert reg["spearman_rho"] < 0
    assert reg["spearman_ci"][1] < 0


def test_flat_trend_ci_spans_zero(run_result):
    """CIS5 is built with no dependence on |t|; the CI must not exclude zero."""
    _, summary, _, _ = run_result
    reg = summary["regions"]["CIS5"]
    assert reg["spearman_ci"][0] <= 0.0 <= reg["spearman_ci"][1]


def test_per_region_median_and_mad_match_numpy(run_result):
    """Location and scale are the median and the unscaled MAD, not the mean and SD."""
    df, summary, _, _ = run_result
    for label in ("TRANS", "CIS5", "PROMOTER"):
        x = _scored(df, label)["boot_se_ratio"].to_numpy(dtype=np.float64)
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        assert summary["regions"][label]["median_se_ratio"] == pytest.approx(med, abs=1e-12)
        assert summary["regions"][label]["mad_se_ratio"] == pytest.approx(mad, abs=1e-12)


def test_bins_partition_scored_rows(run_result):
    """Bins must tile the region's scored rows exactly: no row lost, none double-counted."""
    _, summary, _, _ = run_result
    for label in ("TRANS", "CIS5"):
        reg = summary["regions"][label]
        assert reg["bins"] is not None
        assert sum(b["n"] for b in reg["bins"]) == reg["n_scored"]
        occupied = [b for b in reg["bins"] if b["n"] > 0]
        for prev, nxt in zip(occupied, occupied[1:]):
            assert prev["t_abs_hi"] <= nxt["t_abs_lo"]


def test_t_range_is_over_scored_rows_only(run_result):
    """The reported |t| range must exclude unscored rows."""
    df, summary, _, _ = run_result
    for label in ("TRANS", "CIS5", "PROMOTER"):
        a = np.abs(_scored(df, label)["mt_t"].to_numpy(dtype=np.float64))
        assert summary["regions"][label]["t_abs_min"] == pytest.approx(float(a.min()), abs=1e-12)
        assert summary["regions"][label]["t_abs_max"] == pytest.approx(float(a.max()), abs=1e-12)


def test_unassigned_bucket_collects_null_and_noncanonical(run_result):
    """No scored row is silently dropped for carrying a NULL or unrecognised region."""
    df, summary, _, _ = run_result
    reg = summary["regions"]["unassigned"]
    n_null = len(_scored(df, None))
    n_weird = len(_scored(df, "WEIRD"))
    assert reg["n_scored"] == n_null + n_weird
    assert reg["n_null_region"] == n_null
    assert reg["n_noncanonical_region"] == n_weird
    assert reg["noncanonical_labels"] == ["WEIRD"]
    total_scored = int(np.isfinite(df["boot_se_ratio"]).sum())
    assert sum(r["n_scored"] for r in summary["regions"].values()) == total_scored
    assert summary["n_scored"] == total_scored


def test_small_region_keeps_census_but_omits_trend(run_result):
    """Below --min-region-n the trend is withheld with a stated reason, not guessed."""
    _, summary, _, _ = run_result
    reg = summary["regions"]["PROMOTER"]
    assert reg["n_scored"] == N_PROMOTER
    assert reg["median_se_ratio"] is not None
    assert reg["bins"] is None
    assert reg["spearman_rho"] is None
    assert reg["spearman_ci"] is None
    assert reg["trend_omitted_reason"] == "n_scored below --min-region-n"


def test_deterministic_under_seed(run_result):
    """Same input and same seed must give a byte-identical summary."""
    _, summary, _, d = run_result
    second = d / "summary_again.json"
    _run(d / "concordance.parquet", second)
    with open(second) as fh:
        again = json.load(fh)
    again["input"] = summary["input"]
    assert json.dumps(again, sort_keys=True) == json.dumps(summary, sort_keys=True)


def test_missing_required_column_fails_closed(tmp_path):
    """A missing score column must abort, not report an empty analysis as success."""
    inp = tmp_path / "concordance.parquet"
    df = _build_frame().drop(columns=["boot_se_ratio"])
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), inp)
    out_json = tmp_path / "summary.json"
    proc = _run(inp, out_json, expect_ok=False)
    assert proc.returncode == 1
    assert "boot_se_ratio" in proc.stderr
    assert not out_json.exists()


def test_no_scored_rows_fails_closed(tmp_path):
    """An input where nothing was bootstrapped must abort rather than divide by zero."""
    inp = tmp_path / "concordance.parquet"
    df = _build_frame()
    df["boot_se_ratio"] = np.nan
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), inp)
    out_json = tmp_path / "summary.json"
    proc = _run(inp, out_json, expect_ok=False)
    assert proc.returncode == 1
    assert not out_json.exists()


def test_input_is_left_unchanged(tmp_path):
    """The tool is read-only: the input bytes must survive a run untouched."""
    inp = tmp_path / "concordance.parquet"
    _write_fixture(inp)
    before = _sha256(inp)
    _run(inp, tmp_path / "summary.json")
    assert _sha256(inp) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["concordance.parquet", "summary.json"]


def test_summary_carries_no_decision_field_and_notes_are_hedged(run_result):
    """No key may encode a decision, and every note must hedge and name its assumption."""
    _, summary, _, _ = run_result

    def _keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from _keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from _keys(v)

    banned = ("threshold", "flag", "cutoff", "cut_point", "status", "badge",
              "verdict", "outlier", "problematic", "significant", "pass", "fail")
    for key in _keys(summary):
        low = key.lower()
        for word in banned:
            assert word not in low, f"summary carries a decision field: {key!r}"

    hedges = ("may", "likely", "assumes", "assumption", "consistent with",
              "worth considering", "if ")
    assert set(summary["notes"].keys()) == {
        "mad_scaling", "range_restriction", "trend_reading", "ci_interpretation"}
    for name, text in summary["notes"].items():
        low = text.lower()
        assert any(h in low for h in hedges), f"note {name!r} is not hedged: {text!r}"
