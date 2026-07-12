import numpy as np
import scipy.stats
import pytest

from tecpg.permute import _fit_gpd, _fit_tail, _score_observed, _accumulate_null
from tecpg.logger import Logger

def test_fit_gpd_parameter_recovery():
    # Test 1: GPD parameter recovery (targets `_fit_gpd`)
    rng = np.random.default_rng(0)
    # Target 1: xi = 0.2, sigma = 1.5
    exc1 = scipy.stats.genpareto.rvs(c=0.2, scale=1.5, size=20000, random_state=rng)
    xi1, sigma1 = _fit_gpd(exc1)
    assert np.isclose(xi1, 0.2, atol=0.05)
    assert np.isclose(sigma1, 1.5, atol=0.05)

    # Target 2: xi approx 0 (actually small positive)
    exc2 = scipy.stats.genpareto.rvs(c=0.01, scale=2.0, size=20000, random_state=rng)
    xi2, sigma2 = _fit_gpd(exc2)
    assert np.isclose(xi2, 0.01, atol=0.05)
    assert np.isclose(sigma2, 2.0, atol=0.05)


def test_continuity_at_threshold():
    # Test 2: Continuity at the threshold
    rng = np.random.default_rng(42)
    # Use scale=2.5 so there is a heavy tail as requested in the task description
    null_stats = rng.normal(scale=2.5, size=200000)
    logger = Logger()
    acc = _accumulate_null(null_stats, None, logger)

    u = acc['topk_values'].min()

    # Check that at `u` and `u + epsilon`, the GPD and empirical pieces match (N_u / N)
    emp_p = _score_observed([u], acc, logger)

    # Evaluate at a tiny bit above u, so we evaluate the GPD branch `abs_obs > u`
    obs_just_above = np.array([u + 1e-9])
    emp_p_just_above = _score_observed(obs_just_above, acc, logger)
    tail_p_gpd = _fit_tail(emp_p_just_above, obs_just_above, acc, logger)

    # Expected empirical value at exactly u is simply N_u / N (if no ties exactly on boundaries, this is what empirical calculates)
    # The GPD evaluation should meet it continuously.
    # We test it against exactly N_u / N, which should be what _score_observed returns if we aren't at bin boundaries, but to be sure we just test it against the math.
    expected_p_u = acc['topk_values'].size / acc['total_count']
    assert np.isclose(expected_p_u, tail_p_gpd[0], atol=1e-7)


def test_monotonicity_across_handoff():
    # Test 3: Monotonicity across the handoff
    rng = np.random.default_rng(42)
    null_stats = rng.normal(scale=2.5, size=200000)
    logger = Logger()
    acc = _accumulate_null(null_stats, None, logger)

    u = acc['topk_values'].min()

    # Sweep abs_obs from below u to well above
    obs = np.linspace(u - 1.0, u + 5.0, 100)

    emp_p = _score_observed(obs, acc, logger)
    tail_p = _fit_tail(emp_p, obs, acc, logger)

    # Should be non-increasing (monotonically decreasing or constant)
    # Diff should be <= 0, allowing for a tiny floating point epsilon
    diffs = np.diff(tail_p)
    assert np.all(diffs <= 1e-12)


def test_below_floor_extension():
    # Test 4: Below-floor extension
    rng = np.random.default_rng(42)
    null_stats = rng.normal(scale=2.5, size=200000)
    logger = Logger()
    acc = _accumulate_null(null_stats, None, logger)

    u = acc['topk_values'].min()
    exc = acc['topk_values'][acc['topk_values'] > u] - u
    xi, sigma = _fit_gpd(exc)

    extreme_obs = np.array([u + 10 * sigma])

    emp_p = _score_observed(extreme_obs, acc, logger)
    tail_p = _fit_tail(emp_p, extreme_obs, acc, logger)

    empirical_floor = 1.0 / (acc['total_count'] + 1)

    # Empirical p should be floored or hit the minimum empirical probability.
    # Note: _score_observed returns count / N floored at 1/(N+1), but since it adds overflow, it may be bounded by count.
    assert np.isclose(emp_p[0], empirical_floor) or emp_p[0] >= empirical_floor

    # GPD Tail p drops below the empirical_p
    assert tail_p[0] < empirical_floor
    assert tail_p[0] > 0


def test_bulk_passthrough():
    # Test 5: Bulk passthrough
    rng = np.random.default_rng(42)
    null_stats = rng.normal(scale=2.5, size=200000)
    logger = Logger()
    acc = _accumulate_null(null_stats, None, logger)

    u = acc['topk_values'].min()

    # Test for an observation well below the threshold
    obs = np.array([u - 2.0, u - 0.5, u])

    emp_p = _score_observed(obs, acc, logger)
    tail_p = _fit_tail(emp_p, obs, acc, logger)

    np.testing.assert_array_equal(tail_p, emp_p)


def test_fail_safe_on_degenerate():
    # Test 6: Fail-safe on degenerate fits
    rng = np.random.default_rng(42)
    logger = Logger()

    # Create a completely flat null (no spread) -> degenerate fit or too few exceedances
    null_stats = np.full(200000, 5.0)
    acc = _accumulate_null(null_stats, None, logger)

    u = acc['topk_values'].min()
    # For a completely flat null, max equals min, so exc.size will be 0 (topk > u gives empty array)

    obs = np.array([6.0])
    emp_p = _score_observed(obs, acc, logger)

    # Should not crash and should just return empirical_p
    tail_p = _fit_tail(emp_p, obs, acc, logger)

    np.testing.assert_array_equal(tail_p, emp_p)
