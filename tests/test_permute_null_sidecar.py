import json
import os
import sys

import numpy as np
import pytest
import scipy.stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))
import eval_permute as E  # noqa: E402

from tecpg.permute import (  # noqa: E402
    _accumulate_null,
    _fit_tail,
    _null_sidecar_path,
    _score_observed,
    _write_null_sidecar,
)

EXPECTED_KEYS = {
    'sidecar_version', 'bin_edges', 'hist_counts', 'topk_values',
    'overflow_count', 'total_count', 'topk_capacity', 't_max',
    'perm_seed', 'perm_n_perm', 'observed_max_abs_t', 'n_reported',
    'gpd_status', 'gpd_xi', 'gpd_sigma', 'gpd_u', 'gpd_N_u', 'gpd_N',
}

OBS = np.array([0.5, 2.0, 4.5, 6.0, 37.1])


class RecordingLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg, *a):
        self.messages.append(msg.format(*a))

    def warning(self, msg, *a):
        self.messages.append(msg.format(*a))


def build_accumulator(n_perm=20, n_pairs=50_000, df=340, seed=7):
    rng = np.random.default_rng(seed)
    log = RecordingLogger()
    acc = None
    for _ in range(n_perm):
        acc = _accumulate_null(rng.standard_t(df, size=n_pairs), acc, log)
    return acc


@pytest.fixture
def written(tmp_path):
    log = RecordingLogger()
    acc = build_accumulator()
    emp = _score_observed(OBS, acc, log)
    _fit_tail(emp, OBS, acc, log)
    out = str(tmp_path / "permutation_results.parquet")
    path = _write_null_sidecar(acc, out, 42, 20, float(np.abs(OBS).max()), 5, log)
    return acc, path


def test_sidecar_path_strips_extension():
    assert _null_sidecar_path("/x/permutation_results.parquet") == \
        "/x/permutation_results.perm_null.npz"
    assert _null_sidecar_path("/x/permutation_results.csv") == \
        "/x/permutation_results.perm_null.npz"
    assert _null_sidecar_path(None) is None


def test_sidecar_key_set_exact(written):
    _, path = written
    sc = E.load_null_sidecar(path)
    assert set(sc.keys()) == EXPECTED_KEYS


def test_topk_roundtrips_float64(written):
    acc, path = written
    sc = E.load_null_sidecar(path)
    assert np.asarray(sc['topk_values']).dtype == np.float64
    assert np.array_equal(np.asarray(sc['topk_values']), acc['topk_values'])


def test_gpd_fit_roundtrips(written):
    acc, path = written
    sc = E.load_null_sidecar(path)
    assert abs(float(sc['gpd_xi']) - acc['gpd_xi']) < 1e-12
    assert abs(float(sc['gpd_u']) - acc['gpd_u']) < 1e-12
    assert int(sc['gpd_N']) == acc['gpd_N']


def test_perm_mt_p_unchanged():
    """The sidecar change must not perturb perm_mt_p on any path."""
    log = RecordingLogger()
    acc = build_accumulator()
    emp = _score_observed(OBS, acc, log)

    topk = acc['topk_values']
    u = topk.min()
    N = acc['total_count']
    exc = topk[topk > u] - u
    xi, _, sigma = scipy.stats.genpareto.fit(exc, floc=0)
    p_gpd = (topk.size / N) * scipy.stats.genpareto.sf(
        np.abs(OBS) - u, xi, loc=0, scale=sigma)
    p_gpd = np.maximum(p_gpd, np.finfo(np.float64).tiny)
    expected = np.where(np.abs(OBS) > u, p_gpd, emp)

    assert np.array_equal(_fit_tail(emp, OBS, acc, log), expected)


def test_no_output_file_warns():
    log = RecordingLogger()
    acc = build_accumulator(n_perm=1, n_pairs=1000)
    assert _write_null_sidecar(acc, None, 1, 1, 1.0, 1, log) is None
    assert any('no output_file' in m for m in log.messages), log.messages


def test_empty_accumulator_status():
    log = RecordingLogger()
    acc = _accumulate_null(np.array([]), None, log)
    _fit_tail(np.array([1.0]), np.array([1.0]), acc, log)
    assert acc.get('gpd_status') == 'empty_accumulator'


def test_few_exceedances_status(tmp_path):
    log = RecordingLogger()
    rng = np.random.default_rng(3)
    acc = _accumulate_null(rng.standard_t(340, size=10), None, log)
    _fit_tail(np.array([1.0]), np.array([1.0]), acc, log)
    assert acc.get('gpd_status') == 'skipped_few_exceedances'
    path = _write_null_sidecar(acc, str(tmp_path / "s.parquet"), 1, 1, 1.0, 1, log)
    sc = E.load_null_sidecar(path)
    assert np.isnan(float(sc['gpd_xi']))
    assert E.summarize_sidecar(sc)['gpd_status'] == 'skipped_few_exceedances'


def test_sweep_rung_count(written):
    _, path = written
    summ = E.summarize_sidecar(E.load_null_sidecar(path))
    assert len(summ['xi_sweep']) == len(E.XI_SWEEP_QUANTILES) == 5


def test_sweep_u_monotonic(written):
    _, path = written
    us = [r['u'] for r in E.summarize_sidecar(E.load_null_sidecar(path))['xi_sweep']]
    assert all(us[i] <= us[i + 1] for i in range(len(us) - 1)), us


def test_extrapolation_gap_reported(written):
    _, path = written
    summ = E.summarize_sidecar(E.load_null_sidecar(path))
    assert summ['extrapolation_gap'] > 0
    assert summ['observed_max_abs_t'] == pytest.approx(37.1)


def test_missing_file_skips(tmp_path):
    r = E.resolve_sidecar_arm(str(tmp_path / "absent.npz"))
    assert r['status'] == 'skipped_no_sidecar'
    assert r['reason'] == 'file_absent'


def test_not_provided_skips():
    r = E.resolve_sidecar_arm(None)
    assert r['status'] == 'skipped_no_sidecar'
    assert r['reason'] == 'not_provided'


def test_version_mismatch_skips(tmp_path):
    p = str(tmp_path / "badv.npz")
    np.savez(p, sidecar_version=np.int64(99), topk_values=np.array([1.0]),
             total_count=np.int64(1), overflow_count=np.int64(0),
             hist_counts=np.zeros(3, dtype=np.int64), bin_edges=np.zeros(4))
    assert E.resolve_sidecar_arm(p)['status'] == 'skipped_unreadable'


def test_missing_keys_reason(tmp_path):
    p = str(tmp_path / "bad.npz")
    np.savez(p, sidecar_version=np.int64(1), topk_values=np.array([1.0]))
    r = E.resolve_sidecar_arm(p)
    assert r['status'] == 'skipped_unreadable'
    assert 'missing required keys' in r.get('reason', '')


def test_corrupt_file_skips(tmp_path):
    p = tmp_path / "corrupt.npz"
    p.write_bytes(b"not an npz")
    assert E.resolve_sidecar_arm(str(p))['status'] == 'skipped_unreadable'


def test_report_json_serialisable(written):
    _, path = written
    assert json.dumps(E.summarize_sidecar(E.load_null_sidecar(path)))
