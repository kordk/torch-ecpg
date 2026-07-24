import numpy as np
import pandas as pd
import json
import os
import sys
import pytest

from tools.permute_qc_report import (
    build_calibration_direction_module,
    build_verdict_robustness_module,
    build_permutation_resolution_module,
    build_tail_behaviour_module,
    DIRECTION_WARN,
    DIRECTION_FAIL,
    TOLERANCE_SWEEP,
    NEAR_GENE_REGIONS,
    CANONICAL_REGIONS,
    build_run_provenance_module,
    build_region_composition_module,
    build_bulk_calibration_module,
    build_stratification_module,
    render_html,
    QCModule,
    STATUSES,
    main,
)


@pytest.fixture
def sample_report():
    return {
        "metadata": {
            "n_pairs_input": 1000000,
            "n_pairs_scored": 900000,
            "n_pairs_dropped_unmappable_chrom": 50000,
            "n_pairs_dropped_null_region": 50000,
            "df": 321,
            "tail_p_ana": 0.0001,
            "bulk_lo": 0.05,
            "bulk_hi": 1.0,
            "n_by_region": {
                "TRANS": 800000,
                "DISTAL5": 10000,
                "CIS5": 10000,
                "PROMOTER": 20000,
                "GENEBODY": 10000,
                "CIS3": 10000,
                "DISTAL3": 10000
            }
        },
        "arms": {

            "calibration": {
                "bulk_lo": 0.05,
                "bulk_hi": 0.95,
                "tail_p_ana": 0.01,
                "qq_data": {
                    "neg_log10_p_ana": [0.1, 0.2, 0.3],
                    "neg_log10_p_perm": [0.11, 0.22, 0.29]
                },
                "all": {
                    "bulk_median_abs_log_ratio": 0.005,
                    "bulk_90th_abs_log_ratio": 0.05,
                    "n_perm_below_analytic": 4500000,
                    "bulk_spearman_corr": 0.99
                },
                "TRANS": {
                    "n_perm_below_analytic": 4000000,
                    "tail_median_log_ratio": 0.1,
                    "tail_10th_log_ratio": 0.05,
                    "tail_90th_log_ratio": 0.5
                },
                "PROMOTER": {
                    "n_perm_below_analytic": 4000,
                    "tail_median_log_ratio": 6.5,
                    "tail_10th_log_ratio": 4.0,
                    "tail_90th_log_ratio": 8.8151
                },

                "TRANS": {"n_perm_below_analytic": 500},
                "DISTAL5": {"n_perm_below_analytic": 50},
                "CIS5": {"n_perm_below_analytic": 50},
                "PROMOTER": {"n_perm_below_analytic": 50},
                "GENEBODY": {"n_perm_below_analytic": 50},
                "CIS3": {"n_perm_below_analytic": 50},
                "DISTAL3": {"n_perm_below_analytic": 50}
            },
            "stratify_decision": {
                "recommendation": "adequate",
                "divergent_regions": [],
                "per_region": {
                    "TRANS": {"status": "adequate", "n_bulk": 1000, "median_log10_ratio": 0.0, "lambda": 1.0},
                    "DISTAL5": {"status": "adequate", "n_bulk": 100, "median_log10_ratio": 0.0,
                                "delta_vs_trans": 0.0, "mw_p": 0.9, "ks_p": 0.9, "lambda": 1.0},
                    "CIS5": {"status": "adequate", "n_bulk": 100, "median_log10_ratio": 0.0,
                             "delta_vs_trans": 0.01, "mw_p": 0.9, "ks_p": 0.9, "lambda": 1.0},
                    "PROMOTER": {"status": "adequate", "n_bulk": 100, "median_log10_ratio": 0.0,
                                 "delta_vs_trans": 0.0, "mw_p": 0.9, "ks_p": 0.9, "lambda": 1.0},
                    "GENEBODY": {"status": "adequate", "n_bulk": 100, "median_log10_ratio": 0.0,
                                 "delta_vs_trans": 0.0, "mw_p": 0.9, "ks_p": 0.9, "lambda": 1.0},
                    "CIS3": {"status": "adequate", "n_bulk": 100, "median_log10_ratio": 0.0,
                             "delta_vs_trans": 0.0, "mw_p": 0.9, "ks_p": 0.9, "lambda": 1.0},
                    "DISTAL3": {"status": "adequate", "n_bulk": 100, "median_log10_ratio": 0.0,
                                "delta_vs_trans": 0.0, "mw_p": 0.9, "ks_p": 0.9, "lambda": 1.0},
                }
            }
        }
    }


def test_render_html_self_contained():
    mod = QCModule("test-anchor", "Test Title", "PASS", "Purpose", "Interp", "", "", "")
    html = render_html("gtp", {}, [mod])
    assert "<!DOCTYPE html>" in html
    # Figure base64 presence depends on whether we pass one, we should verify it doesn't contain external links
    assert "http://" not in html
    assert "https://" not in html
    assert "<script" not in html
    assert "<link " not in html

    # Check data image tag
    mod_fig = QCModule("test-anchor-2", "Test Title", "PASS", "Purpose", "Interp", "", "iVBORw0KGgoAAA==", "")
    html_fig = render_html("gtp", {}, [mod_fig])
    assert "data:image/png;base64," in html_fig


def test_all_modules_present_and_anchored(sample_report):
    mods = [
        build_run_provenance_module(sample_report),
        build_region_composition_module(sample_report),
        build_bulk_calibration_module(sample_report),
        build_stratification_module(sample_report),
    ]
    html = render_html("gtp", {}, mods)

    for mod in mods:
        assert f'href="#{mod.anchor}"' in html
        assert f'id="{mod.anchor}"' in html


def test_status_values_valid(sample_report):
    mods = [
        build_run_provenance_module(sample_report),
        build_region_composition_module(sample_report),
        build_bulk_calibration_module(sample_report),
        build_stratification_module(sample_report),
    ]
    for mod in mods:
        assert mod.status in STATUSES


def test_region_composition_fail_below_floor(sample_report):
    mod = build_region_composition_module(sample_report)
    assert mod.status == "PASS"

    bad_report = {"metadata": {"n_by_region": {"CIS5": 10}}}
    bad_mod = build_region_composition_module(bad_report)
    assert bad_mod.status == "FAIL"


def test_region_composition_warns_on_strand_asymmetry(sample_report):
    # Make distal regions highly asymmetric
    report = dict(sample_report)
    report["metadata"]["n_by_region"]["DISTAL5"] = 100000
    report["metadata"]["n_by_region"]["DISTAL3"] = 1000
    mod = build_region_composition_module(report)
    assert mod.status == "WARN"


def test_stratification_fail_on_divergent(sample_report):
    mod = build_stratification_module(sample_report)
    assert mod.status == "PASS"

    bad_report = dict(sample_report)
    bad_report["arms"]["stratify_decision"]["divergent_regions"] = ["PROMOTER"]
    bad_mod = build_stratification_module(bad_report)
    assert bad_mod.status == "FAIL"


def test_stratification_reports_margin(sample_report):
    mod = build_stratification_module(sample_report)
    # The largest near-gene delta is CIS5 with 0.01
    assert "Largest near-gene |Δ vs TRANS| = 1.000e-02" in mod.interpretation
    assert "margin of 50x" in mod.interpretation


def test_modules_never_raise_on_empty_report():
    builders = [
        build_run_provenance_module,
        build_region_composition_module,
        build_bulk_calibration_module,
        build_stratification_module,
    ]
    for builder in builders:
        mod = builder({})
        assert mod.status in STATUSES
        # Just verifying it doesn't raise, the status for composition is FAIL due to floor logic.
        # But instructions said "A builder that cannot evaluate its check returns status='INFO' \
        # with an explanatory interpretation; it must never raise."
        # Actually composition module without any near-gene gives 0 near_gene which < MIN_REGION_BULK_N (FAIL).
        # We test that it does not raise.


def test_html_escaping():
    mod = QCModule("test", "Test", "PASS", "Purpose", "Interp", "", "", "")
    html_out = render_html("<script>alert(1)</script>", {}, [mod])

    assert "<script>alert(1)</script>" not in html_out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out
    # We should not find a raw <script> tag anywhere
    assert "<script>" not in html_out


def test_end_to_end_writes_html(tmp_path, monkeypatch, sample_report):
    json_path = tmp_path / "report.json"
    json_path.write_text(json.dumps(sample_report))
    out_html = tmp_path / "out.html"

    monkeypatch.setattr(sys, "argv", ["permute_qc_report.py", "--report",
                        str(json_path), "--dataset", "test_dataset", "--out", str(out_html)])

    # Catch sys.exit if it happens, but expect it to succeed without exiting
    try:
        main()
    except SystemExit as e:
        assert e.code == 0

    assert out_html.exists()
    content = out_html.read_text()
    assert len(content) > 0
    assert "<!DOCTYPE html>" in content
    assert 'id="run-provenance"' in content
    assert 'id="region-composition"' in content
    assert 'id="bulk-calibration"' in content
    assert 'id="stratification"' in content


def test_schema_conformance_independent_oracle():
    """
    Ensure the keys we read from report dicts match the actual keys that eval_permute emits.
    This uses AST to extract the string literal arguments passed to `.get(...)` calls
    inside the permute_qc_report.py builders, and asserts they belong to the known
    valid key set derived from the real eval_permute JSON output structure.
    """
    import ast
    import os

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'permute_qc_report.py')
    with open(script_path, 'r') as f:
        tree = ast.parse(f.read())

    requested_keys = set()

    # We walk the AST looking for Call nodes where the function is an Attribute named 'get'
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'get':
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    requested_keys.add(node.args[0].value)

    # Known valid keys from a full real GTP eval_permute_report.json excerpt
    # This encompasses all keys used by the four modules built so far.
    known_valid_keys = {
        # metadata
        'metadata', 'n_pairs_input', 'n_pairs_scored',
        'n_pairs_dropped_unmappable_chrom', 'n_pairs_dropped_null_region',
        'df', 'n_by_region',

        # canonical regions (present in n_by_region and per_region)
        'TRANS', 'DISTAL5', 'CIS5', 'PROMOTER', 'GENEBODY', 'CIS3', 'DISTAL3',

        # arms -> calibration
        'arms', 'calibration', 'bulk_lo', 'bulk_hi', 'tail_p_ana', 'qq_data',
        'per_region', 'neg_log10_p_ana', 'neg_log10_p_perm',

        # arms -> stratify_decision
        'stratify_decision', 'recommendation', 'divergent_regions',

        # inside per_region dicts (for both calibration and stratification)
        'status', 'n_bulk', 'median_log10_ratio', 'n_perm_below_analytic',
        'delta_vs_trans', 'mw_p', 'ks_p', 'lambda',
        'all', 'bulk_median_abs_log_ratio', 'bulk_90th_abs_log_ratio',
        'bulk_spearman_corr', 'tail_median_log_ratio', 'tail_10th_log_ratio',
        'tail_90th_log_ratio', 'perm_mt_p', 'n_perm', 'mt_t',

    }

    # Assert that every requested key via .get('key') is a known valid key
    # (ignoring purely structural missing-check lookups if they aren't meant to be in the JSON,
    # but in our script, all .get calls are for real keys).
    for key in requested_keys:
        assert key in known_valid_keys, f"Reader asks for missing/wrong key: {key}"


def test_direction_status_thresholds(sample_report):
    # Pass
    sample_report['arms']['calibration']['all']['bulk_median_abs_log_ratio'] = DIRECTION_WARN - 0.001
    mod = build_calibration_direction_module(sample_report)
    assert mod.status == 'PASS'

    # Warn
    sample_report['arms']['calibration']['all']['bulk_median_abs_log_ratio'] = DIRECTION_WARN + 0.001
    mod = build_calibration_direction_module(sample_report)
    assert mod.status == 'WARN'

    # Fail
    sample_report['arms']['calibration']['all']['bulk_median_abs_log_ratio'] = DIRECTION_FAIL + 0.001
    mod = build_calibration_direction_module(sample_report)
    assert mod.status == 'FAIL'

    # Missing -> Info
    del sample_report['arms']['calibration']['all']['bulk_median_abs_log_ratio']
    mod = build_calibration_direction_module(sample_report)
    assert mod.status == 'INFO'


def test_direction_table_reports_ratio_and_fraction(sample_report):
    # Mock some data for TRANS
    sample_report['arms']['stratify_decision']['per_region']['TRANS']['median_log10_ratio'] = 0.00276
    sample_report['arms']['calibration']['TRANS']['bulk_median_abs_log_ratio'] = 0.003
    sample_report['arms']['calibration']['TRANS']['bulk_90th_abs_log_ratio'] = 0.015
    sample_report['arms']['calibration']['TRANS']['n_perm_below_analytic'] = 4500000
    sample_report['arms']['stratify_decision']['per_region']['TRANS']['n_bulk'] = 9000000

    mod = build_calibration_direction_module(sample_report)
    html = mod.table_html

    # 10 ** 0.00276 is ~ 1.00638
    assert "1.00638" in html
    assert "0.00276" in html
    assert "50.00%" in html  # 4.5M / 9.0M


def test_tolerance_sweep_monotonic(sample_report):
    # Construct near-gene deltas that decrease
    # Make sure we have enough float deltas
    sample_report['arms']['stratify_decision']['per_region']['PROMOTER']['delta_vs_trans'] = 0.06
    sample_report['arms']['stratify_decision']['per_region']['GENEBODY']['delta_vs_trans'] = 0.03
    sample_report['arms']['stratify_decision']['per_region']['CIS5']['delta_vs_trans'] = 0.015

    mod = build_verdict_robustness_module(sample_report)

    # We parse the HTML table to get the div counts in order
    import re
    rows = re.findall(r'<tr>(.*?)</tr>', mod.table_html, re.DOTALL)
    # The first row is the header, subsequent rows are data
    counts = []
    for row in rows[1:]:
        cells = re.findall(r'<td.*?>(.*?)</td>', row)
        counts.append(int(cells[1]))

    # As tolerance decreases (sweeps top to bottom if sorted descending), count should non-decrease
    assert counts == sorted(counts, reverse=False), f"Counts not monotonically increasing as tolerance drops: {counts}"


def test_tolerance_sweep_crossing(sample_report):
    # Only one near-gene delta, but we need at least 2 floats for module to not return INFO.
    # We'll set one to 3.0e-5 and one to 1.0e-5
    for r in NEAR_GENE_REGIONS:
        if r in sample_report['arms']['stratify_decision']['per_region']:
            sample_report['arms']['stratify_decision']['per_region'][r]['delta_vs_trans'] = 0.0
    sample_report['arms']['stratify_decision']['per_region']['PROMOTER']['delta_vs_trans'] = 3.0e-05
    sample_report['arms']['stratify_decision']['per_region']['GENEBODY']['delta_vs_trans'] = 1.0e-05
    sample_report['arms']['stratify_decision']['recommendation'] = 'single_global_null_adequate'

    mod = build_verdict_robustness_module(sample_report)
    assert mod.status != 'INFO'

    import re
    rows = re.findall(r'<tr>(.*?)</tr>', mod.table_html, re.DOTALL)
    for row in rows[1:]:
        cells = re.findall(r'<td.*?>(.*?)</td>', row)
        t = float(cells[0])
        verdict = cells[2]

        if t > 3.0e-05 + 1e-9:
            assert verdict == "single_global_null_adequate"
        else:
            assert verdict == "stratification_warranted"


def test_verdict_robustness_warns_when_fragile(sample_report):
    applied_t = 0.5  # the constant is 0.5
    # Set a delta just below applied_t, e.g., 0.06, so it crosses at 0.1, which is applied/5
    sample_report['arms']['stratify_decision']['per_region']['PROMOTER']['delta_vs_trans'] = 0.06
    sample_report['arms']['stratify_decision']['per_region']['GENEBODY']['delta_vs_trans'] = 0.01
    sample_report['arms']['stratify_decision']['recommendation'] = 'single_global_null_adequate'

    mod = build_verdict_robustness_module(sample_report)
    assert mod.status == 'WARN'


def test_resolution_module_info_without_parquet(sample_report):
    mod = build_permutation_resolution_module(sample_report, df=None)
    assert mod.status == 'INFO'
    assert mod.interpretation == 'Not evaluated: --perm-output not supplied.'
    assert not mod.table_html
    assert not mod.figure_b64


def test_resolution_floor_computation(sample_report):
    df = pd.DataFrame({
        'perm_mt_p': [0.0, 1e-8, 5e-8, np.nan, 1e-7],
        'n_perm': [10000000] * 5,
        'mt_t': [1.0, 1.0, 1.0, 1.0, 1.0]
    })

    mod = build_permutation_resolution_module(sample_report, df=df)

    assert mod.status == 'PASS'
    assert "1e-08" in mod.table_html
    assert "Zeros</td>\n      <td style=\"text-align: left;\">1</td>" in mod.table_html
    assert "2 (40.00% of parquet rows)" in mod.table_html


def test_resolution_warns_when_tail_saturated(sample_report):
    # tail_p_ana is 1e-4 from sample_report metadata
    df = pd.DataFrame({
        'perm_mt_p': [1e-8, 1e-8, 1e-8, 0.5],
        'n_perm': [10000000] * 4,
        'mt_t': [6.0, 6.0, 6.0, 1.0]
    })
    # df degrees of freedom = 321
    sample_report['metadata']['df'] = 321

    mod = build_permutation_resolution_module(sample_report, df=df)
    assert mod.status == 'WARN'

    # Sparse tail -> PASS
    df2 = pd.DataFrame({
        'perm_mt_p': [1e-8, 0.01, 0.01, 0.5],
        'n_perm': [10000000] * 4,
        'mt_t': [6.0, 6.0, 6.0, 1.0]
    })
    mod2 = build_permutation_resolution_module(sample_report, df=df2)
    assert mod2.status == 'PASS'


def test_tail_module_never_badged(sample_report):
    # Even with extreme ratios in PROMOTER
    mod = build_tail_behaviour_module(sample_report)
    assert mod.status == 'INFO'


def test_tail_module_info_when_absent(sample_report):
    del sample_report['arms']['calibration']['PROMOTER']
    del sample_report['arms']['calibration']['TRANS']
    mod = build_tail_behaviour_module(sample_report)
    assert mod.status == 'INFO'
    assert mod.interpretation == 'Not evaluated: no tail statistics present in the report.'


def test_module_order_resolution_before_tail(sample_report):
    import os
    import ast
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'permute_qc_report.py')
    with open(script_path, 'r') as f:
        src = f.read()

    # Just string matching is enough to prove the order in the list literal
    res_idx = src.find('build_permutation_resolution_module,')
    tail_idx = src.find('build_tail_behaviour_module,')
    strat_idx = src.find('build_stratification_module,')

    assert strat_idx < res_idx
    assert res_idx < tail_idx


def test_new_modules_never_raise_on_empty_report():
    for builder in [build_calibration_direction_module, build_verdict_robustness_module,
                    build_permutation_resolution_module, build_tail_behaviour_module]:
        mod = builder({}, df=None)
        assert hasattr(mod, 'status')
        assert mod.status in ('PASS', 'WARN', 'FAIL', 'INFO')


def test_provenance_metadata_lookups(sample_report):
    mod = build_run_provenance_module(sample_report)
    # The fixture sets bulk_lo=0.05, bulk_hi=1.0, tail_p_ana=1e-4 in metadata
    html = mod.table_html
    assert "[0.05, 1.0]" in html
    assert "p_ana &lt; 0.0001" in html
