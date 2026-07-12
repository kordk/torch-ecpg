import numpy as np
import pytest
from tecpg.permute import _accumulate_null, _score_observed, tecpg_mlr_qr_permute
from tecpg.logger import Logger
from tecpg.test_data import generate_data

def test_score_oracle_edge_aligned():
    logger = Logger()
    rng = np.random.default_rng(42)
    null_stats = rng.normal(scale=2.5, size=50000)

    acc = _accumulate_null(null_stats, None, logger)

    obs_on_edges = acc['bin_edges'][[50, 150, 300, 500, 800]]

    abs_null = np.abs(null_stats)
    N = acc['total_count']

    ref = np.array([max((abs_null >= e).sum() / N, 1.0 / (N + 1)) for e in obs_on_edges])

    np.testing.assert_allclose(_score_observed(obs_on_edges, acc, logger), ref, atol=1e-12)

def test_score_monotonicity():
    logger = Logger()
    rng = np.random.default_rng(42)
    null_stats = rng.normal(scale=2.5, size=50000)
    acc = _accumulate_null(null_stats, None, logger)

    obs = np.linspace(0, 6, 50)
    p = _score_observed(obs, acc, logger)

    assert np.all(np.diff(p) <= 1e-12)

def test_score_floor_and_endpoints():
    logger = Logger()
    rng = np.random.default_rng(42)
    # Use a small scale so there are 0 overflow elements to strictly test the floor
    null_stats = rng.normal(scale=0.5, size=500)
    acc = _accumulate_null(null_stats, None, logger)

    N = acc['total_count']
    assert acc['overflow_count'] == 0

    assert _score_observed([100.0], acc, logger)[0] == 1.0 / (N + 1)

    # 0.0 will match all values, so p should be 1.0
    assert _score_observed([0.0], acc, logger)[0] == 1.0

def test_score_two_sided():
    logger = Logger()
    rng = np.random.default_rng(42)
    null_stats = rng.normal(scale=2.5, size=50000)
    acc = _accumulate_null(null_stats, None, logger)

    for x in [0.5, 1.0, 2.0, 5.0]:
        assert _score_observed([x], acc, logger)[0] == _score_observed([-x], acc, logger)[0]

def test_score_empty_null_fail_closed():
    logger = Logger()

    with pytest.raises(ValueError, match="Empty null accumulator; cannot score observed statistics."):
        _score_observed([1.0], None, logger)

    empty_acc = {
        'bin_edges': np.linspace(0, 10, 100),
        'hist_counts': np.zeros(99, dtype=np.int64),
        'overflow_count': 0,
        'total_count': 0,
    }

    with pytest.raises(ValueError, match="Empty null accumulator; cannot score observed statistics."):
        _score_observed([1.0], empty_acc, logger)

def test_score_requires_annotations():
    M, G, C = generate_data(sample_size=30, m_rows=10, g_rows=10, annotation=False)

    with pytest.raises(ValueError, match="qr_permute requires methylation and expression annotations"):
        tecpg_mlr_qr_permute(M, G, C)
