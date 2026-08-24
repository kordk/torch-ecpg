import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))
import eval_permute as E  # noqa: E402

TOL = E.TOLERANCE_MEDIAN_LOG10_RATIO_DIFF


def test_deterministic_for_same_seed():
    rng = np.random.default_rng(11)
    a = rng.normal(0.3, 1.0, 5000)
    b = rng.normal(0.0, 1.0, 5000)
    assert E.bootstrap_delta_ci(a, b) == E.bootstrap_delta_ci(a, b)


def test_ci_brackets_point_estimate():
    rng = np.random.default_rng(11)
    a = rng.normal(0.3, 1.0, 5000)
    b = rng.normal(0.0, 1.0, 5000)
    lo, hi, n = E.bootstrap_delta_ci(a, b)
    pt = float(np.median(a) - np.median(b))
    assert lo <= pt <= hi
    assert n == E.DELTA_CI_RESAMPLES


def test_ci_width_shrinks_with_n():
    rng = np.random.default_rng(12)
    small = E.bootstrap_delta_ci(rng.normal(0, 1, 150), rng.normal(0, 1, 150))
    large = E.bootstrap_delta_ci(rng.normal(0, 1, 200000),
                                 rng.normal(0, 1, 200000), n_resamples=300)
    assert (small[1] - small[0]) > (large[1] - large[0]) * 5


def test_classify_equivalent():
    assert E.classify_delta_ci(-0.1, 0.1) == 'equivalent'


def test_classify_divergent_positive():
    assert E.classify_delta_ci(TOL + 0.2, TOL + 0.4) == 'divergent'


def test_classify_divergent_negative():
    assert E.classify_delta_ci(-TOL - 0.4, -TOL - 0.2) == 'divergent'


def test_classify_inconclusive_spanning_edge():
    assert E.classify_delta_ci(TOL - 0.2, TOL + 0.3) == 'inconclusive'


def test_classify_inconclusive_spanning_zero():
    assert E.classify_delta_ci(-TOL - 0.3, TOL + 0.3) == 'inconclusive'


def test_classify_exact_band_edges_are_equivalent():
    assert E.classify_delta_ci(-TOL, TOL) == 'equivalent'


def test_classify_none_passthrough():
    assert E.classify_delta_ci(None, None) is None


def test_underpowered_region_reads_inconclusive():
    rng = np.random.default_rng(13)
    lo, hi, _ = E.bootstrap_delta_ci(rng.normal(0.03, 3.0, 120),
                                     rng.normal(0.0, 3.0, 120))
    assert E.classify_delta_ci(lo, hi) == 'inconclusive'


def test_well_powered_region_reads_equivalent():
    rng = np.random.default_rng(14)
    lo, hi, _ = E.bootstrap_delta_ci(rng.normal(0.03, 1.0, 400000),
                                     rng.normal(0.0, 1.0, 400000),
                                     n_resamples=300)
    assert E.classify_delta_ci(lo, hi) == 'equivalent'


def test_margin_distinguishes_what_label_collapses():
    rng = np.random.default_rng(15)
    noisy = E.bootstrap_delta_ci(rng.normal(0.03, 3.0, 120),
                                 rng.normal(0.0, 3.0, 120))
    tight = E.bootstrap_delta_ci(rng.normal(0.03, 1.0, 400000),
                                 rng.normal(0.0, 1.0, 400000),
                                 n_resamples=300)
    assert E.delta_ci_margin(*tight[:2]) > E.delta_ci_margin(*noisy[:2])


def test_borderline_equivalent_has_small_margin():
    assert E.delta_ci_margin(-0.489, -0.042) < 0.02


def test_ci_width_matches_analytic_oracle():
    """SE(median) ~ 1.2533*sd/sqrt(n); SE(delta) = sqrt(2)*that."""
    rng = np.random.default_rng(16)
    n, sd = 20000, 1.0
    a = rng.normal(0.0, sd, n)
    b = rng.normal(0.0, sd, n)
    lo, hi, _ = E.bootstrap_delta_ci(a, b, n_resamples=800)
    expected = 2 * 1.96 * np.sqrt(2) * (1.2533 * sd / np.sqrt(n))
    assert 0.80 < (hi - lo) / expected < 1.20


def test_alpha_maps_to_half_on_each_side():
    """Probe at alpha=0.5: a mis-mapping collapses the lower arm."""
    rng = np.random.default_rng(16)
    a = rng.normal(0.0, 1.0, 20000)
    b = rng.normal(0.0, 1.0, 20000)
    pt = float(np.median(a) - np.median(b))
    lo, hi, _ = E.bootstrap_delta_ci(a, b, n_resamples=800, alpha=0.5)
    lo_arm, hi_arm = pt - lo, hi - pt
    assert lo_arm > 0
    assert (lo_arm / hi_arm) > 0.5


def test_ci_width_strictly_positive():
    rng = np.random.default_rng(16)
    lo, hi, _ = E.bootstrap_delta_ci(rng.normal(0, 1, 20000),
                                     rng.normal(0, 1, 20000), n_resamples=800)
    assert (hi - lo) > 0


def test_subsample_cap_is_applied():
    """tb stays below every cap tested, so only rb capping can differ."""
    rng = np.random.default_rng(17)
    big_a = rng.normal(0.0, 1.0, 60000)
    small_b = rng.normal(0.0, 1.0, 800)
    capped = E.bootstrap_delta_ci(big_a, small_b, n_resamples=100, max_n=1000)
    uncapped = E.bootstrap_delta_ci(big_a, small_b, n_resamples=100,
                                    max_n=E.DELTA_CI_MAX_N)
    assert (capped[1] - capped[0]) > (uncapped[1] - uncapped[0])


def test_empty_region_arm():
    rng = np.random.default_rng(18)
    assert E.bootstrap_delta_ci(np.array([]), rng.normal(0, 1, 100)) == (None, None, 0)


def test_empty_trans_arm():
    rng = np.random.default_rng(18)
    assert E.bootstrap_delta_ci(rng.normal(0, 1, 100), np.array([])) == (None, None, 0)


def test_single_element_arms_do_not_raise():
    lo, hi, _ = E.bootstrap_delta_ci(np.array([1.0]), np.array([0.0]))
    assert lo is not None and hi is not None


def test_cost_is_bounded_by_cap():
    rng = np.random.default_rng(19)
    t0 = time.time()
    E.bootstrap_delta_ci(rng.normal(0, 1, 400000), rng.normal(0, 1, 400000),
                         n_resamples=100)
    assert (time.time() - t0) < 10


def test_new_fields_are_json_serialisable():
    rng = np.random.default_rng(20)
    lo, hi, n = E.bootstrap_delta_ci(rng.normal(0.1, 1.0, 3000),
                                     rng.normal(0.0, 1.0, 3000), n_resamples=200)
    block = {
        'delta_ci_lo': lo,
        'delta_ci_hi': hi,
        'delta_ci_width': float(hi - lo),
        'delta_ci_resamples': n,
        'delta_ci_verdict': E.classify_delta_ci(lo, hi),
        'delta_ci_margin': E.delta_ci_margin(lo, hi),
    }
    assert json.dumps(block)
