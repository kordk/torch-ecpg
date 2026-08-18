"""tecpg correctness test harness (precondition for statistics fixes).

This module encodes CORRECT behavior for the numeric paths of ``tecpg``
*independently* of the current output. It is intentionally written BEFORE
the statistics fixes land so that:

* later intentional fixes flip specific ``xfail`` tests to ``pass``
  (each ``xfail`` names the audit item that will fix it, e.g. ``C1``), and
* unintended drift trips a test that should have stayed stable
  (the structural fingerprint).

NOTHING in this module fixes an audit bug. The only production change that
accompanies it is the seeding of ``tecpg/test_data.py`` so the dummy input
is byte-reproducible.

Three kinds of tests live here:

1. ORACLE / DIFFERENTIAL TESTS -- compute the answer a second, independent
   way at test time and assert agreement with an explicit tolerance
   (``numpy.testing.assert_allclose`` with named ``rtol``/``atol``).
2. STRUCTURAL FINGERPRINT -- run the dummy ``all`` pipeline once and diff a
   small committed JSON of integer / rounded aggregates
   (``fingerprint_all_pipeline.json``). That JSON is the *only* committed
   reference artifact; there are no stored output parquets and no network
   fetches anywhere in this module.
3. INVARIANT / PROPERTY TESTS -- assert properties that must hold on the
   dummy pipeline output, with no reference at all.

Regenerate the committed fingerprint with::

    python tests/test_correctness_harness.py --regenerate-fingerprint

Regeneration re-blesses the structural reference and therefore requires a
reviewed reason: it is the one place a human re-blesses structure. Do not
regenerate it to silence an unexplained diff.

Audit items referenced:

* C1 -- bootstrap ``p_boot`` has no ``1/B`` floor, so a one-sided resample
  distribution yields ``p_boot == 0`` instead of the intended ``1/B``.
* C2 -- BH-FDR extraction assumes the supplied parquet holds the top-most
  hits for the given ``--total-tests``; isolated here by fixing
  ``total_tests == len(p)`` so only the BH *math* is exercised.
* C3 -- the bootstrap resample draw is seeded through the production
  ``seed`` parameter (the path the CLI ``--seed`` plumbs); the
  bootstrap-based tests here pass that seed directly so they are
  deterministic without touching global numpy state.
* M4/M5/M6 -- per-drop-stage count logging; the row-count conservation
  invariants below assert what that logging will make observable.
"""

import argparse
import json
import math
import os
import sys
import tempfile

import numpy as np
import pytest

# Ensure we can import tecpg when run as a script from anywhere.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd  # noqa: E402

from tecpg.logger import Logger  # noqa: E402
from tecpg.regression_full import regression_full  # noqa: E402
from tecpg.test_data import generate_data  # noqa: E402

# --------------------------------------------------------------------------
# Harness configuration (a single fixed seed keeps everything reproducible).
# --------------------------------------------------------------------------
HARNESS_SEED = 1234
DUMMY_SAMPLES = 80
DUMMY_METH = 20
DUMMY_GENE = 15
# total_tests / alpha used for the fingerprint's FDR aggregate. total_tests
# equals the full pair count (the honest "all tests performed" analog for a
# tiny dummy); the BH-vs-statsmodels oracle below uses its own fixed vector.
FINGERPRINT_TOTAL_TESTS = DUMMY_METH * DUMMY_GENE  # == n_pairs == 300
FINGERPRINT_ALPHA = 0.05
P_THRESHOLD = 0.05

FINGERPRINT_PATH = os.path.join(
    os.path.dirname(__file__), 'fingerprint_all_pipeline.json'
)

# float32 carries ~7 significant decimal digits; round aggregates so the
# committed JSON stays human-diffable and stable across CPU low-bit noise.
FLOAT_ROUND = 6


# ==========================================================================
# Shared helpers
# ==========================================================================
def _quiet_logger() -> Logger:
    """A CPU-pinned logger for deterministic, GPU-free test runs."""
    return Logger(carry_data={'use_cpu': True})


def _dummy_with_annotation():
    """Seeded dummy data with annotation, indexed for region work."""
    M, G, C, M_annot, G_annot = generate_data(
        DUMMY_SAMPLES, DUMMY_METH, DUMMY_GENE, annotation=True,
        seed=HARNESS_SEED,
    )
    M_annot = M_annot.set_index('name')
    G_annot = G_annot.set_index('name')
    return M, G, C, M_annot, G_annot


def _run_all_pipeline_methylation():
    """Run the dummy ``all`` regression, keeping every pair (no p filter).

    Returns a tidy frame with columns ``gt_id, mt_id, mt_est, mt_err,
    mt_t, mt_p`` -- the methylation-only "published" columns.
    """
    M, G, C, _, _ = _dummy_with_annotation()
    out = regression_full(
        M, G, C, region='all', p_thresh=None, methylation_only=True,
        logger=_quiet_logger(),
    )
    return out.reset_index()


def _bh_q_values(p_values: np.ndarray, total_tests: int) -> np.ndarray:
    """Benjamini-Hochberg q-values, mirroring summarizeOutput_parquet.py.

    Mirrors ``tools/summarizeOutput_parquet.py`` lines 333-379:
    ``fdr_est = p * total_tests / rank`` followed by the step-down
    monotonicity pass from largest to smallest rank. Returned q-values are
    aligned to ascending p-value order.
    """
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    ranks = np.arange(1, n + 1)
    estimated_fdr = sorted_p * total_tests / ranks
    q_values = np.zeros_like(estimated_fdr, dtype=float)
    q_values[-1] = min(1.0, estimated_fdr[-1])
    for i in range(len(estimated_fdr) - 2, -1, -1):
        q_values[i] = min(q_values[i + 1], estimated_fdr[i])
    return q_values, order


def _bh_significant_count(
    p_values: np.ndarray, total_tests: int, alpha: float
) -> int:
    """BH threshold-discovery count, mirroring summarizeOutput_parquet.py.

    Mirrors ``tools/summarizeOutput_parquet.py`` lines 334-345:
    ``sorted_p <= (rank / total_tests) * alpha``; the significant count is
    the largest passing rank.
    """
    n = len(p_values)
    sorted_p = np.sort(p_values)
    ranks = np.arange(1, n + 1)
    bh_limits = (ranks / total_tests) * alpha
    valid = np.nonzero(sorted_p <= bh_limits)[0]
    return int(valid[-1] + 1) if len(valid) > 0 else 0


# ==========================================================================
# 1. ORACLE / DIFFERENTIAL TESTS
# ==========================================================================
def test_oracle_qr_regression_vs_plain_ols():
    """Batched QR regression vs. independent per-pair OLS (lstsq).

    For a small set of pairs, assert beta, t-stat, and p agree with the
    plain ``numpy.linalg.lstsq`` solution within tolerance. p uses the same
    normal-approximation that ``tecpg`` uses (``create_normal_p`` ==
    ``2 * norm.sf(|t|)``), so this isolates the regression algebra rather
    than the normal-vs-t approximation. Tolerances are explicit because the
    tensor path runs in float32 and differs from float64 numpy in low bits.
    """
    import scipy.stats

    M, G, C = generate_data(
        DUMMY_SAMPLES, DUMMY_METH, DUMMY_GENE, annotation=False,
        seed=HARNESS_SEED,
    )
    out = regression_full(
        M, G, C, region='all', p_thresh=None, methylation_only=False,
        logger=_quiet_logger(),
    )

    n_samples = C.shape[0]
    k = C.shape[1] + 2  # intercept + methylation + covariates
    df = n_samples - k

    rng = np.random.default_rng(HARNESS_SEED)
    mt_ids = list(M.index)
    gt_ids = list(G.index)
    sampled = [
        (rng.choice(gt_ids), rng.choice(mt_ids)) for _ in range(8)
    ]

    for gt_id, mt_id in sampled:
        y = G.loc[gt_id].to_numpy(dtype=float)
        x_m = M.loc[mt_id].to_numpy(dtype=float)
        X = np.column_stack(
            (np.ones(n_samples), x_m, C.to_numpy(dtype=float))
        )
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residual = y - X @ beta
        sigma2 = np.sum(residual ** 2) / df
        var_beta = sigma2 * np.linalg.inv(X.T @ X).diagonal()
        se = np.sqrt(var_beta)
        t = beta / se
        # tecpg uses the normal CDF as the Student's-t approximation.
        p = 2 * scipy.stats.norm.sf(np.abs(t))

        # methylation coefficient lives at design-matrix index 1.
        row = out.loc[(gt_id, mt_id)]
        np.testing.assert_allclose(
            row['mt_est'], beta[1], rtol=1e-3, atol=1e-4,
            err_msg=f'beta mismatch for {(gt_id, mt_id)}',
        )
        np.testing.assert_allclose(
            row['mt_t'], t[1], rtol=1e-3, atol=1e-4,
            err_msg=f't-stat mismatch for {(gt_id, mt_id)}',
        )
        np.testing.assert_allclose(
            row['mt_p'], p[1], rtol=1e-3, atol=1e-4,
            err_msg=f'p-value mismatch for {(gt_id, mt_id)}',
        )


def test_oracle_bh_fdr_vs_statsmodels():
    """BH-FDR math vs. statsmodels, with total_tests == len(p).

    Fixing ``total_tests`` to the number of supplied p-values isolates the
    BH *math* from the C2 extraction bug (the subset/top-hits assumption).
    Under that condition the in-repo ``fdr_est`` formula must equal
    ``statsmodels.stats.multitest.multipletests(method='fdr_bh')``.
    """
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(HARNESS_SEED)
    p_values = rng.uniform(0.0, 1.0, size=64)
    total_tests = len(p_values)  # isolates BH math from C2.

    q_sorted, order = _bh_q_values(p_values, total_tests)
    q_repo = np.empty_like(q_sorted)
    q_repo[order] = q_sorted  # back to original ordering

    _, q_sm, _, _ = multipletests(p_values, method='fdr_bh')

    np.testing.assert_allclose(q_repo, q_sm, rtol=1e-9, atol=1e-12)


# --- p_boot oracle (one-sided / two-sided split + degenerate resamples) ---
def _p_boot_production(estimates: np.ndarray):
    """Mirror of bootstrap.py p_boot math, on finite resamples.

    Mirrors ``tecpg/bootstrap.py`` lines 284-320: drop non-finite
    resamples, then ``p_boot = 2 * min(mean(est<=0), mean(est>=0))`` clamped
    to ``[1/finite_count, 1]``. The ``1/finite_count`` floor landed with
    audit item C1, so this mirror now matches the floored production value.
    Returns ``(p_boot, finite_count)``.
    """
    finite = estimates[np.isfinite(estimates)]
    finite_count = int(finite.size)
    if finite_count == 0:
        return float('nan'), 0
    prop_le = float(np.mean(finite <= 0))
    prop_ge = float(np.mean(finite >= 0))
    p_boot = min(prop_le, prop_ge) * 2.0
    p_boot = min(p_boot, 1.0)
    return max(p_boot, 1.0 / finite_count), finite_count


def _p_boot_oracle(estimates: np.ndarray):
    """Independent oracle p_boot calculation.

    Computes empirical two-sided p-value from first principles without
    reusing the _p_boot_production math.
    """
    finite = estimates[np.isfinite(estimates)]
    B = len(finite)
    if B == 0:
        return float('nan'), 0
    p = min(np.sum(finite <= 0), np.sum(finite >= 0)) * 2.0 / B
    return max(min(p, 1.0), 1.0 / B), B


# ==========================================================================
# p_boot via the REAL bootstrap pipeline (auto-flips when C1 lands)
# ==========================================================================
def _run_real_bootstrap(M, G, C, pairs, iterations, np_seed):
    """Run the production bootstrap on tiny data and return the result frame.

    The bootstrap resample draw is seeded through the production ``seed``
    parameter (the path the CLI ``--seed`` plumbs as of C3), so the run is
    deterministic without touching global numpy state.
    """
    from tecpg.bootstrap import tecpg_mlr_qr_bootstrap

    workdir = tempfile.mkdtemp(prefix='tecpg_boot_')
    pairs_file = os.path.join(workdir, 'pairs.csv')
    master_file = os.path.join(workdir, 'master.parquet')
    out_file = os.path.join(workdir, 'out.parquet')
    pairs_df = pd.DataFrame(pairs, columns=['mt_id', 'gt_id'])
    pairs_df.to_csv(pairs_file, index=False)
    pairs_df.to_parquet(master_file)

    tecpg_mlr_qr_bootstrap(
        M, G, C, pairs_file, master_file, out_file,
        iterations=iterations, batch_size=8, seed=np_seed,
        logger=_quiet_logger(),
    )
    return pd.read_parquet(out_file)


def _one_sided_bootstrap_fixture():
    """M/G/C with one strongly positive pair so resample estimates are >0."""
    n = 48
    subs = [f's{i}' for i in range(n)]
    rng = np.random.default_rng(HARNESS_SEED)
    m = rng.uniform(0.0, 1.0, size=n)
    g = 2.0 * m + 0.01 * rng.standard_normal(n)  # near-deterministic, >0 slope
    M = pd.DataFrame([m], index=['cg001'], columns=subs)
    G = pd.DataFrame([g], index=['ILMN_001'], columns=subs)
    C = pd.DataFrame({'age': rng.uniform(0, 1, size=n)}, index=subs)
    return M, G, C, [('cg001', 'ILMN_001')]

def _degenerate_bootstrap_fixture():
    """M/G/C with forced degenerate counts, so D is known > 0.

    A step-function M with noise-free G ensures some resamples drop to 0 variance,
    resulting in degenerate count D > 0.
    """
    n = 48
    subs = [f's{i}' for i in range(n)]
    rng = np.random.default_rng(HARNESS_SEED)
    m = np.zeros(n)
    m[0:5] = 1.0
    g = m * 2.0 + rng.standard_normal(n) * 0.01
    M = pd.DataFrame([m], index=['cg001'], columns=subs)
    G = pd.DataFrame([g], index=['ILMN_001'], columns=subs)
    C = pd.DataFrame({'age': rng.uniform(0, 1, size=n)}, index=subs)
    return M, G, C, [('cg001', 'ILMN_001')]

def test_p_boot_floor_real_pipeline():
    """Invariant ``p_boot == 1/(B-D)`` on the REAL bootstrap output for 1-sided degenerate."""
    iterations = 2000
    M, G, C, pairs = _degenerate_bootstrap_fixture()
    res = _run_real_bootstrap(M, G, C, pairs, iterations, np_seed=123)

    p_boot = res['p_boot'].to_numpy(dtype=float)[0]
    D = res['degenerate_resamples'].to_numpy(dtype=int)[0]

    assert D > 0, "Failed to force a known degenerate count D > 0"

    # Assert the floor equals 1/(B-D)
    expected_floor = 1.0 / (iterations - D)
    assert p_boot == pytest.approx(expected_floor, rel=1e-9)

    # Assert the floor does NOT equal 1/B
    assert p_boot != pytest.approx(1.0 / iterations, rel=1e-9)

def test_p_boot_oracle_real_pipeline():
    """p_boot oracle vs REAL bootstrap output, on identical resample indices.

    The bootstrap resample draw is seeded through the production ``seed``
    parameter, making `np.random.choice` deterministic. We run a two-sided
    synthetic array so we don't trivially hit the floor.
    """
    iterations = 200
    rng_fixture = np.random.default_rng(HARNESS_SEED)
    n = 48
    subs = [f's{i}' for i in range(n)]
    # Create two-sided data (estimates straddle 0)
    m = rng_fixture.uniform(0.0, 1.0, size=n)
    # weak relationship -> straddles 0
    g = 0.0 * m + 1.0 * rng_fixture.standard_normal(n)
    M = pd.DataFrame([m], index=['cg001'], columns=subs)
    G = pd.DataFrame([g], index=['ILMN_001'], columns=subs)
    C = pd.DataFrame({'age': rng_fixture.uniform(0, 1, size=n)}, index=subs)
    pairs = [('cg001', 'ILMN_001')]

    # We must construct the independent estimates by running the identical resample indices
    # The actual production uses np.random.choice(n, size=(iterations, n), replace=True)
    # immediately after setting np.random.seed(seed)
    np.random.seed(123)
    boot_indices = np.random.choice(n, size=(iterations, n), replace=True)

    res = _run_real_bootstrap(M, G, C, pairs, iterations, np_seed=123)
    p_boot_prod = res['p_boot'].to_numpy(dtype=float)[0]

    estimates = []
    m_arr = M.to_numpy()[0]
    g_arr = G.to_numpy()[0]
    c_arr = C.to_numpy()

    for i in range(iterations):
        indices = boot_indices[i]
        m_resample = m_arr[indices]
        g_resample = g_arr[indices]
        c_resample = c_arr[indices]

        X = np.column_stack([np.ones(n), m_resample, c_resample])
        y = g_resample
        if np.linalg.matrix_rank(X) < 3:
            estimates.append(np.nan)
        else:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            estimates.append(beta[1]) # m_resample coefficient is column index 1

    oracle_p_boot, _ = _p_boot_oracle(np.array(estimates))
    assert p_boot_prod == pytest.approx(oracle_p_boot, rel=1e-6)



# ==========================================================================
# 2. STRUCTURAL FINGERPRINT
# ==========================================================================
def compute_fingerprint() -> dict:
    """Compute the small structural fingerprint of the dummy ``all`` run.

    Shared by the fingerprint test and the regeneration entrypoint so the
    committed JSON and the test always use identical logic.
    """
    M, G, C, M_annot, G_annot = _dummy_with_annotation()
    out = regression_full(
        M, G, C, region='all', p_thresh=None, methylation_only=True,
        logger=_quiet_logger(),
    ).reset_index()

    n_pairs_in = int(len(out))
    p = out['mt_p'].to_numpy(dtype=float)
    finite_p = p[np.isfinite(p)]

    # cis/trans: a clean binary partition by chromosome equality, so the
    # partition invariant (n_cis + n_trans == n_pairs) holds. "cis" here is
    # the same-chromosome complement of trans (cis + distal in pipeline
    # terms); "trans" is different-chromosome, matching the pipeline filter.
    m_chrom = M_annot['chrom'].astype(int).to_dict()
    g_chrom = G_annot['chrom'].astype(int).to_dict()
    same_chrom = (
        out['mt_id'].map(m_chrom).to_numpy()
        == out['gt_id'].map(g_chrom).to_numpy()
    )
    n_cis = int(same_chrom.sum())
    n_trans = int((~same_chrom).sum())

    # p-threshold drop stage.
    p_pass = int((p <= P_THRESHOLD).sum())
    p_drop = n_pairs_in - p_pass

    # BH-FDR drop stage (fixed total_tests + alpha).
    n_fdr_sig = _bh_significant_count(
        p, FINGERPRINT_TOTAL_TESTS, FINGERPRINT_ALPHA
    )
    # Each FDR-significant mt-gt pair is one edge of the bipartite network.
    network_edges = n_fdr_sig

    return {
        'harness_seed': HARNESS_SEED,
        'dummy_dims': {
            'samples': DUMMY_SAMPLES,
            'meth': DUMMY_METH,
            'gene': DUMMY_GENE,
        },
        'fdr_total_tests': FINGERPRINT_TOTAL_TESTS,
        'fdr_alpha': FINGERPRINT_ALPHA,
        'p_threshold': P_THRESHOLD,
        'n_pairs_in': n_pairs_in,
        'n_cis': n_cis,
        'n_trans': n_trans,
        'n_fdr_sig': n_fdr_sig,
        'network_edges': network_edges,
        'p_min': round(float(finite_p.min()), FLOAT_ROUND),
        'p_max': round(float(finite_p.max()), FLOAT_ROUND),
        'drop_stages': [
            {
                'stage': 'p_threshold_filter',
                'in': n_pairs_in,
                'out': p_pass,
                'dropped': p_drop,
            },
            {
                'stage': 'fdr_bh_filter',
                'in': n_pairs_in,
                'out': n_fdr_sig,
                'dropped': n_pairs_in - n_fdr_sig,
            },
        ],
    }


def _write_fingerprint(fingerprint: dict, path: str = FINGERPRINT_PATH) -> None:
    with open(path, 'w') as handle:
        json.dump(fingerprint, handle, indent=2, sort_keys=True)
        handle.write('\n')


def test_structural_fingerprint_matches_committed():
    """Regenerate the structural fingerprint and diff against the committed JSON.

    The committed ``fingerprint_all_pipeline.json`` is the only reference
    artifact in this harness. A diff here means the structure of the dummy
    ``all`` pipeline changed; that is either an intended re-blessing (run the
    regeneration command with a reviewed reason) or unintended drift.
    """
    assert os.path.exists(FINGERPRINT_PATH), (
        'Missing committed fingerprint. Regenerate with: '
        'python tests/test_correctness_harness.py --regenerate-fingerprint'
    )
    with open(FINGERPRINT_PATH) as handle:
        committed = json.load(handle)
    current = compute_fingerprint()

    # Check floating point keys with atol
    float_keys = ['p_max', 'p_min']
    for k in float_keys:
        if k in current and k in committed:
            assert np.isclose(current[k], committed[k], atol=5e-5), (
                f"Floating point drift on {k}: {current[k]} vs {committed[k]}"
            )
            # Remove them from dicts to allow exact comparison of the rest
            del current[k]
            del committed[k]

    assert current == committed, (
        'Structural fingerprint drift detected.\n'
        f'current:   {json.dumps(current, sort_keys=True)}\n'
        f'committed: {json.dumps(committed, sort_keys=True)}\n'
        'If this change is intended, regenerate with a reviewed reason: '
        'python tests/test_correctness_harness.py --regenerate-fingerprint'
    )


# ==========================================================================
# 3. INVARIANT / PROPERTY TESTS
# ==========================================================================
def test_invariant_cis_trans_partition():
    """n_cis + n_trans == n_pairs_total (no pair unclassified/double-counted)."""
    fingerprint = compute_fingerprint()
    assert (
        fingerprint['n_cis'] + fingerprint['n_trans']
        == fingerprint['n_pairs_in']
    )


def test_invariant_no_nan_inf_in_published_columns():
    """No NaN/inf in the published regression columns (beta, t, p)."""
    out = _run_all_pipeline_methylation()
    for col in ('mt_est', 'mt_err', 'mt_t', 'mt_p'):
        values = out[col].to_numpy(dtype=float)
        assert np.all(np.isfinite(values)), f'non-finite values in {col}'


def test_invariant_fdr_est_monotonic_in_p_rank():
    """BH fdr_est is monotonic non-decreasing in ascending p-rank."""
    out = _run_all_pipeline_methylation()
    p = out['mt_p'].to_numpy(dtype=float)
    q_sorted, _ = _bh_q_values(p, FINGERPRINT_TOTAL_TESTS)
    diffs = np.diff(q_sorted)
    assert np.all(diffs >= -1e-12), 'fdr_est not monotonic in p-rank'


def test_invariant_fdr_est_finite_and_bounded():
    """Computed fdr_est values are finite and within [0, 1]."""
    out = _run_all_pipeline_methylation()
    p = out['mt_p'].to_numpy(dtype=float)
    q_sorted, _ = _bh_q_values(p, FINGERPRINT_TOTAL_TESTS)
    assert np.all(np.isfinite(q_sorted))
    assert np.all(q_sorted >= 0.0)
    assert np.all(q_sorted <= 1.0)


def test_invariant_row_count_conservation_every_drop_stage():
    """At every drop site: before == after + dropped (M4/M5/M6 logging)."""
    fingerprint = compute_fingerprint()
    for stage in fingerprint['drop_stages']:
        assert stage['in'] == stage['out'] + stage['dropped'], (
            f"row-count conservation violated at {stage['stage']}: "
            f"{stage['in']} != {stage['out']} + {stage['dropped']}"
        )


def test_invariant_p_boot_within_bounds_real_pipeline():
    """p_boot in [1/B, 1] for all rows of the REAL bootstrap output.

    The bootstrap is seeded through the production ``seed`` parameter (the
    CLI ``--seed`` path landed in C3), so the run is deterministic.
    """
    iterations = 200
    M, G, C, pairs = _one_sided_bootstrap_fixture()
    res = _run_real_bootstrap(M, G, C, pairs, iterations, np_seed=7)
    p_boot = res['p_boot'].to_numpy(dtype=float)
    floor = 1.0 / iterations
    assert np.all(p_boot >= floor - 1e-12)
    assert np.all(p_boot <= 1.0 + 1e-12)


def test_seed_round_trip():
    """Test seed persistence and output reproducibility.

    Validates that:
    1. A subprocess CLI run saves the requested seed to the parquet column and metadata.
    2. Two runs with the same seed produce byte-identical bootstrap metrics.
    """
    import subprocess
    import pyarrow.parquet as pq
    import shutil

    iterations = 50
    seed_val = 98765

    M, G, C, pairs = _one_sided_bootstrap_fixture()

    workdir1 = tempfile.mkdtemp(prefix='tecpg_seed1_')
    workdir2 = tempfile.mkdtemp(prefix='tecpg_seed2_')

    try:
        pairs_file = os.path.join(workdir1, 'pairs.csv')
        pd.DataFrame(pairs, columns=['mt_id', 'gt_id']).to_csv(pairs_file, index=False)
        master_file = os.path.join(workdir1, 'master.parquet')
        pd.DataFrame(pairs, columns=['mt_id', 'gt_id']).to_parquet(master_file)
        M.to_csv(os.path.join(workdir1, 'M.csv'))
        G.to_csv(os.path.join(workdir1, 'G.csv'))
        C.to_csv(os.path.join(workdir1, 'C.csv'))
        out_file1 = os.path.join(workdir1, 'bootstrap_merged.csv')

        subprocess.run([
            'python3', '-m', 'tecpg', '-i', workdir1, '-o', workdir1, 'run', 'mlr', '--mlr-method', 'qr_bootstrap',
            '--pairs-file', pairs_file, '--master-parquet', master_file, '--output-format', 'csv',
            '--bootstrap-iterations', str(iterations), '--seed', str(seed_val)
        ], check=True, capture_output=True)

        res1 = pd.read_csv(out_file1)

        # Assert column
        assert res1['boot_seed'].iloc[0] == seed_val

        # Assert metadata (parquet only feature, so we also need to test parquet)
        out_file1_pq = os.path.join(workdir1, 'bootstrap_merged.parquet')
        subprocess.run([
            'python3', '-m', 'tecpg', '-i', workdir1, '-o', workdir1, 'run', 'mlr', '--mlr-method', 'qr_bootstrap',
            '--pairs-file', pairs_file, '--master-parquet', master_file, '--output-format', 'parquet',
            '--bootstrap-iterations', str(iterations), '--seed', str(seed_val)
        ], check=True, capture_output=True)
        res1_pq = pd.read_parquet(out_file1_pq)

        table = pq.read_table(out_file1_pq)
        metadata = table.schema.metadata
        assert metadata[b'tecpg_bootstrap_seed'] == str(seed_val).encode()

        out_file2 = os.path.join(workdir2, 'bootstrap_merged.parquet')

        subprocess.run([
            'python3', '-m', 'tecpg', '-i', workdir1, '-o', workdir2, 'run', 'mlr', '--mlr-method', 'qr_bootstrap',
            '--pairs-file', pairs_file, '--master-parquet', master_file, '--output-format', 'parquet',
            '--bootstrap-iterations', str(iterations), '--seed', str(seed_val)
        ], check=True, capture_output=True)

        res2 = pd.read_parquet(out_file2)
        res1 = res1_pq

        pd.testing.assert_series_equal(res1['p_boot'], res2['p_boot'])
        pd.testing.assert_series_equal(res1['ci_low'], res2['ci_low'])
        pd.testing.assert_series_equal(res1['ci_high'], res2['ci_high'])
    finally:
        shutil.rmtree(workdir1, ignore_errors=True)
        shutil.rmtree(workdir2, ignore_errors=True)


# ==========================================================================
# Regeneration entrypoint
# ==========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--regenerate-fingerprint',
        action='store_true',
        help='Recompute and overwrite the committed structural fingerprint '
        'JSON. Requires a reviewed reason -- it re-blesses structure.',
    )
    args = parser.parse_args()
    if args.regenerate_fingerprint:
        fingerprint = compute_fingerprint()
        _write_fingerprint(fingerprint)
        print(f'Wrote fingerprint to {FINGERPRINT_PATH}:')
        print(json.dumps(fingerprint, indent=2, sort_keys=True))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
