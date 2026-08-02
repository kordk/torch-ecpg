"""Guards for tools/se_ratio_trend_report.py.

The report renders the summary written by diagnose_se_ratio_trend.py. It
echoes; it does not recompute, and it reaches no verdict. These tests pin both
properties alongside the rendering.
"""
import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO_ROOT, "tools", "se_ratio_trend_report.py")
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))


def _bins(base, step):
    out = []
    for i in range(10):
        lo = base * (1.0 + 0.12 * i)
        out.append({
            "bin": i + 1,
            "n": 100,
            "t_abs_lo": lo,
            "t_abs_hi": lo * 1.11,
            "median_se_ratio": 1.0 + step * i,
        })
    return out


def _region(n, median, mad, tmin, tmax, rho, ci, bins, reason=None):
    return {
        "n_scored": n,
        "median_se_ratio": median,
        "mad_se_ratio": mad,
        "t_abs_min": tmin,
        "t_abs_max": tmax,
        "spearman_rho": rho,
        "spearman_ci": ci,
        "spearman_ci_omitted_reason": None,
        "spearman_rho_omitted_reason": None,
        "trend_omitted_reason": reason,
        "bins": bins,
    }


def _summary():
    unassigned = _region(500, 1.0580, 0.0849, 4.2244, 16.2207,
                         0.1965, [0.1735, 0.2160], _bins(4.2244, 0.006))
    unassigned["n_null_region"] = 500
    unassigned["n_noncanonical_region"] = 0
    unassigned["noncanonical_labels"] = []
    unassigned["noncanonical_labels_truncated"] = False
    return {
        "input": "/data/output_gtp/bootstrap_concordance.parquet",
        "tool_version": "2.0.0b2.dev60",
        "n_rows_read": 1000000,
        "n_scored": 1554,
        "params": {"bins": 10, "min_region_n": 200, "ci_resamples": 1000,
                   "ci_level": 0.95, "seed": 0, "chunk_size": 100000},
        "regions": {
            "CIS5": _region(1000, 1.0206, 0.0544, 3.2915, 37.1152,
                            0.0382, [0.0076, 0.0696], _bins(3.2915, 0.001)),
            "GENEBODY": _region(54, 1.0077, 0.0570, 3.2933, 4.6735,
                                None, None, None,
                                reason="n_scored below --min-region-n"),
            "unassigned": unassigned,
        },
        "notes": {
            "mad_scaling": "reading mad_se_ratio as a standard deviation assumes normality.",
            "range_restriction": "a region selected by ranking on precise_mt_p may be truncated.",
            "trend_reading": "a flat slope may be consistent with an effect across all rows.",
            "ci_interpretation": "the interval assumes the scored rows are exchangeable.",
        },
    }


def _write_summary(path, summary=None):
    payload = _summary() if summary is None else summary
    with open(path, "w") as fh:
        json.dump(payload, fh)
    return payload


def _run(inp, out_html, expect_ok=True):
    proc = subprocess.run(
        [sys.executable, TOOL, "--trend-json", str(inp),
         "--dataset", "FIXTURE", "--out", str(out_html)],
        capture_output=True, text=True)
    if expect_ok:
        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    return proc


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    d = tmp_path_factory.mktemp("se_ratio_report")
    inp = d / "trend.json"
    summary = _write_summary(inp)
    out_html = d / "report.html"
    _run(inp, out_html)
    return summary, out_html.read_text(), d


def test_report_is_written_and_self_contained(rendered):
    """A report that reaches for the network is not an archivable artifact."""
    _, html, _ = rendered
    assert "<html" in html.lower()
    assert "base64," in html
    for token in ('src="http', "src='http", 'href="http://', 'href="https://'):
        assert token not in html, f"external reference in report: {token}"


def test_every_module_is_info(rendered):
    """The report describes a measurement whose reading is open; no badge asserts otherwise."""
    from se_ratio_trend_report import build_modules
    modules = build_modules(_summary())
    assert len(modules) == 4
    assert [m.status for m in modules] == ["INFO"] * 4
    _, html, _ = rendered
    for bad in ('class="badge warn"', 'class="badge fail"', 'class="badge pass"'):
        assert bad not in html, f"report carries a {bad!r}"


def test_module_anchors_are_present_and_ordered(rendered):
    from se_ratio_trend_report import build_modules
    anchors = [m.anchor for m in build_modules(_summary())]
    assert anchors == ["run-provenance", "region-census",
                       "trend-by-region", "interpretation-guidance"]
    _, html, _ = rendered
    for a in anchors:
        assert a in html


def test_census_echoes_summary_and_does_not_recompute(rendered):
    """A median recomputed from bins would drift from the one the tool reported."""
    summary, html, d = rendered
    assert f"{summary['regions']['CIS5']['median_se_ratio']:.4f}" in html
    assert f"{summary['regions']['CIS5']['mad_se_ratio']:.4f}" in html

    altered = copy.deepcopy(_summary())
    for b in altered["regions"]["CIS5"]["bins"]:
        b["median_se_ratio"] = 9.9
    inp2 = d / "trend2.json"
    _write_summary(inp2, altered)
    out2 = d / "report2.html"
    _run(inp2, out2)
    assert f"{summary['regions']['CIS5']['median_se_ratio']:.4f}" in out2.read_text()


def test_region_with_withheld_trend_is_reported_with_its_reason(rendered):
    """A region below the trend gate must appear, carrying why, not vanish."""
    _, html, _ = rendered
    assert "GENEBODY" in html
    assert "n_scored below --min-region-n" in html


def test_unassigned_breakdown_is_reported(rendered):
    """The null-region count is the diagnostic for an upstream annotation gap."""
    _, html, _ = rendered
    assert "n_null_region" in html or "null" in html.lower()
    assert "500" in html


def test_interval_direction_is_reported(rendered):
    _, html, _ = rendered
    assert "excludes zero" in html


def test_notes_are_rendered_verbatim(rendered):
    """Two descriptions of one quantity must not be free to drift apart."""
    summary, html, _ = rendered
    import html as html_mod
    for text in summary["notes"].values():
        assert html_mod.escape(text) in html or text in html


def test_module_interpretations_are_hedged(rendered):
    from se_ratio_trend_report import build_modules
    hedges = ("may", "likely", "assumes", "assumption", "consistent with",
              "worth considering", "if ")
    for m in build_modules(_summary()):
        low = m.interpretation.lower()
        assert any(h in low for h in hedges), f"module {m.anchor!r} is not hedged"


def test_no_recommendation_language(rendered):
    """The report reports; choosing a cut is a separate act with a separate input."""
    from se_ratio_trend_report import build_modules
    banned = ("we recommend", "you should", "should be excluded", "should use",
              "must be excluded", "optimal cut", "best cut", "recommended value")
    blob = " ".join(m.purpose + " " + m.interpretation
                    for m in build_modules(_summary())).lower()
    for word in banned:
        assert word not in blob, f"module prose recommends: {word!r}"
    _, html, _ = rendered
    low = html.lower()
    for word in banned:
        assert word not in low, f"report recommends: {word!r}"


def test_missing_required_key_fails_closed(tmp_path):
    """An incomplete summary must abort, not render a report with holes in it."""
    broken = _summary()
    del broken["regions"]
    inp = tmp_path / "trend.json"
    _write_summary(inp, broken)
    out_html = tmp_path / "report.html"
    proc = _run(inp, out_html, expect_ok=False)
    assert proc.returncode == 1
    assert "regions" in proc.stderr
    assert not out_html.exists()


def test_input_json_is_left_unchanged(tmp_path):
    inp = tmp_path / "trend.json"
    _write_summary(inp)
    before = _sha256(inp)
    _run(inp, tmp_path / "report.html")
    assert _sha256(inp) == before
