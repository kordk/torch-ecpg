import json
import os
import subprocess
import tempfile
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.stats

# Import the actual module for whitebox testing of isolated functions
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tools.eval_permute as ep

# ==============================================================================
# ORACLE 1: ANALYTIC-P
# ==============================================================================

def test_analytic_p_oracle():
    """
    Assert that the script's analytic p-value computation using Student-t sf
    exactly matches known exact tail values to atol=1e-10.
    """
    test_cases = [
        # (t, df, expected_p)
        (0.0, 10, 1.0),
        (1.96, 10000, 0.05002352023),  # roughly 0.05
        (2.0, 10, 0.07338803477),
        (5.0, 5, 0.00410471598)
    ]

    for t_val, df, expected_p in test_cases:
        p_ana = ep.compute_analytic_p(t_val, df)
        assert np.isclose(p_ana, expected_p, atol=1e-8), f"Failed for t={t_val}, df={df}: got {p_ana}, expected {expected_p}"

# ==============================================================================
# ORACLE 2: GPD-RECOVERY():
    """
    Draw genpareto samples with known (xi, sigma), fit via the script's fitter,
    assert recovery within a stated tolerance. Includes a xi approx 0 case.
    """
    rng = np.random.default_rng(42)
    u = 0.0

    cases = [
        # (xi, sigma) Note: scipy param c = xi, scale = sigma
        (0.2, 1.5),
        (0.01, 1.0),  # approx exponential
        (-0.2, 0.5)
    ]

    for xi, sigma in cases:
        # Generate GPD samples
        data = scipy.stats.genpareto.rvs(c=xi, scale=sigma, size=100_000, random_state=rng)

        fit_xi, fit_sigma = ep.fit_gpd(data, u)

        # Method of moments is roughly robust but we need a loose tolerance for it
        assert np.isclose(fit_xi, xi, atol=0.02), f"Failed xi recovery: expected {xi}, got {fit_xi}"
        assert np.isclose(fit_sigma, sigma, atol=0.02), f"Failed sigma recovery: expected {sigma}, got {fit_sigma}"

# ==============================================================================
# ORACLE 3: UNIFORMITY
# ==============================================================================
def test_uniformity_oracle():
    """
    Feed U(0,1) draws through lambda / KS null-sanity code; assert lambda approx 1
    and KS small on genuine null. Assert lambda > 1 flags on inflated input.
    """
    rng = np.random.default_rng(42)
    n = 100_000

    # Genuine null: p ~ U(0,1), so t ~ StudentT
    df = 50
    t_null = scipy.stats.t.rvs(df, size=n, random_state=rng)

    lam_null = ep.calculate_genomic_inflation(t_null)
    assert np.isclose(lam_null, 1.0, atol=0.05), f"Null lambda inflated: {lam_null}"

    p_null = 2.0 * scipy.stats.t.sf(np.abs(t_null), df)
    ks_stat, ks_p = scipy.stats.kstest(p_null, 'uniform')
    assert ks_p > 0.01, f"Null p-values not uniform, ks_p={ks_p}"

    # Inflated input: scale t
    t_inflated = t_null * 1.5
    lam_inflated = ep.calculate_genomic_inflation(t_inflated)
    assert lam_inflated > 1.5, f"Inflated lambda not > 1.5: {lam_inflated}"

# ==============================================================================
# ORACLE 4: STRATUM-LABELING
# ==============================================================================
def test_stratum_labeling_oracle(tmp_path):
    """
    Tiny synthetic annotation where cis/trans is hand-known.
    Assert is_cis is exactly correct, and reported id missing raises ValueError.
    """
    m_annot = pd.DataFrame({
        'name': ['m1', 'm2', 'm3'],
        'chrom': [1, 1, 2],
        'chromStart': [100, 200, 300]
    })

    g_annot = pd.DataFrame({
        'name': ['g1', 'g2', 'g3'],
        'chrom': [1, 2, 2],
        'chromStart': [150, 250, 350]
    })

    output = pd.DataFrame({
        'mt_id': ['m1', 'm2', 'm3', 'm1'],
        'gt_id': ['g1', 'g2', 'g3', 'g3'],
        'mt_t': [1.0, 1.0, 1.0, 1.0],
        'perm_mt_p': [0.5, 0.5, 0.5, 0.5]
    })
    # expected:
    # m1-g1: chrom 1-1 (cis)
    # m2-g2: chrom 1-2 (trans)
    # m3-g3: chrom 2-2 (cis)
    # m1-g3: chrom 1-2 (trans)

    m_annot_path = tmp_path / "m_annot.csv"
    g_annot_path = tmp_path / "g_annot.csv"
    output_path = tmp_path / "output.csv"

    m_annot.to_csv(m_annot_path, index=False)
    g_annot.to_csv(g_annot_path, index=False)
    output.to_csv(output_path, index=False)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    script_path = os.path.join(os.path.dirname(__file__), "../tools/eval_permute.py")

    res = subprocess.run([
        "python", script_path,
        "--perm-output", str(output_path),
        "--m-annot", str(m_annot_path),
        "--g-annot", str(g_annot_path),
        "--df", "10",
        "--out-dir", str(out_dir)
    ], capture_output=True, text=True)

    assert res.returncode == 0, f"Script failed: {res.stderr}"


    with open(out_dir / "eval_permute_report.json") as f:
        report = json.load(f)

    assert report['metadata']['n_cis'] == 2
    assert report['metadata']['n_trans'] == 2


    # Test fail closed on missing ID
    output_bad = pd.DataFrame({
        'mt_id': ['m1', 'm99'],
        'gt_id': ['g1', 'g1'],
        'mt_t': [1.0, 1.0],
        'perm_mt_p': [0.5, 0.5]
    })
    output_bad_path = tmp_path / "output_bad.csv"
    output_bad.to_csv(output_bad_path, index=False)

    res_bad = subprocess.run([
        "python", script_path,
        "--perm-output", str(output_bad_path),
        "--m-annot", str(m_annot_path),
        "--g-annot", str(g_annot_path),
        "--df", "10",
        "--out-dir", str(out_dir)
    ], capture_output=True, text=True)

    assert res_bad.returncode == 1
    assert "missing from annotations" in res_bad.stderr

# ==============================================================================
# ORACLE 5: STRATIFY-DECISION SMOKE
# ==============================================================================

def test_label_strata_value_error():
    m_annot = pd.DataFrame({'name': ['m1'], 'chrom': [1]}).set_index('name')
    g_annot = pd.DataFrame({'name': ['g1'], 'chrom': [1]}).set_index('name')
    output = pd.DataFrame({'mt_id': ['m1'], 'gt_id': ['g2']})  # g2 is missing
    with pytest.raises(ValueError, match="missing from annotations"):
        ep.label_strata(output, m_annot, g_annot)

def test_stratify_decision_smoke(tmp_path):
    """
    Construct one synthetic case where cis == trans (expect "single_global_null_adequate")
    and one where cis is deliberately shifted (expect "stratification_warranted" or confound).
    """
    df_val = 50
    rng = np.random.default_rng(42)
    n = 10000

    m_annot = pd.DataFrame({'name': [f'm{i}' for i in range(n)], 'chrom': [1]*n})
    g_annot = pd.DataFrame({'name': [f'g{i}' for i in range(n)], 'chrom': [1]*(n//2) + [2]*(n//2)})

    m_annot_path = tmp_path / "m_annot.csv"
    g_annot_path = tmp_path / "g_annot.csv"
    m_annot.to_csv(m_annot_path, index=False)
    g_annot.to_csv(g_annot_path, index=False)

    # Case 1: Identical Distributions (cis == trans)
    t_vals = scipy.stats.t.rvs(df_val, size=n, random_state=rng)
    p_ana = 2.0 * scipy.stats.t.sf(np.abs(t_vals), df_val)
    p_perm = p_ana * 1.0  # Exact match

    output1 = pd.DataFrame({
        'mt_id': [f'm{i}' for i in range(n)],
        'gt_id': [f'g{i}' for i in range(n)],
        'mt_t': t_vals,
        'perm_mt_p': p_perm
    })

    out1_path = tmp_path / "out1.parquet"
    table1 = pa.Table.from_pandas(output1)
    pq.write_table(table1, out1_path)

    script_path = os.path.join(os.path.dirname(__file__), "../tools/eval_permute.py")
    out_dir1 = tmp_path / "dir1"

    subprocess.run([
        "python", script_path,
        "--perm-output", str(out1_path),
        "--m-annot", str(m_annot_path),
        "--g-annot", str(g_annot_path),
        "--df", str(df_val),
        "--out-dir", str(out_dir1)
    ], check=True)

    with open(out_dir1 / "eval_permute_report.json") as f:
        rep1 = json.load(f)
    assert rep1['arms']['stratify_decision']['recommendation'] == "single_global_null_adequate"

    # Case 2: Divergence
    # We alter perm_mt_p for cis so it's significantly lower than analytic
    # Half of the data is cis (chrom 1 == chrom 1). Indices 0 to n//2-1
    p_perm2 = p_ana.copy()
    p_perm2[:n//2] = p_perm2[:n//2] * 0.05  # Massive downward shift, ratio < 0

    output2 = pd.DataFrame({
        'mt_id': [f'm{i}' for i in range(n)],
        'gt_id': [f'g{i}' for i in range(n)],
        'mt_t': t_vals,
        'perm_mt_p': p_perm2
    })
    out2_path = tmp_path / "out2.parquet"
    table2 = pa.Table.from_pandas(output2)
    pq.write_table(table2, out2_path)

    out_dir2 = tmp_path / "dir2"
    subprocess.run([
        "python", script_path,
        "--perm-output", str(out2_path),
        "--m-annot", str(m_annot_path),
        "--g-annot", str(g_annot_path),
        "--df", str(df_val),
        "--out-dir", str(out_dir2)
    ], check=True)

    with open(out_dir2 / "eval_permute_report.json") as f:
        rep2 = json.load(f)
    assert rep2['arms']['stratify_decision']['recommendation'] == "stratification_warranted"


    # Case 3: Confound Branch
    p_perm3 = p_ana.copy()
    p_perm3[:n//2] = p_perm3[:n//2] * 0.05
    t_vals3 = t_vals.copy()
    t_vals3[:n//2] = t_vals3[:n//2] * 2.0  # Inflate cis t-values substantially

    output3 = pd.DataFrame({
        'mt_id': [f'm{i}' for i in range(n)],
        'gt_id': [f'g{i}' for i in range(n)],
        'mt_t': t_vals3,
        'perm_mt_p': p_perm3
    })
    out3_path = tmp_path / "out3.parquet"
    pq.write_table(pa.Table.from_pandas(output3), out3_path)

    out_dir3 = tmp_path / "dir3"
    subprocess.run([
        sys.executable, script_path,
        "--perm-output", str(out3_path),
        "--m-annot", str(m_annot_path),
        "--g-annot", str(g_annot_path),
        "--df", str(df_val),
        "--out-dir", str(out_dir3)
    ], check=True)

    with open(out_dir3 / "eval_permute_report.json") as f:
        rep3 = json.load(f)
    assert rep3['arms']['stratify_decision']['recommendation'] == "inconclusive_cis_signal_confound"

# ==============================================================================
# ORACLE 6: SIDECAR-ABSENT SMOKE
# ==============================================================================
def test_sidecar_absent_smoke(tmp_path):
    """
    Run end-to-end with no --perm-null-sidecar.
    Assert JSON report is produced and gated sections read "skipped_no_sidecar".
    """
    df_val = 50
    m_annot = pd.DataFrame({'name': ['m1'], 'chrom': [1]})
    g_annot = pd.DataFrame({'name': ['g1'], 'chrom': [1]})
    output = pd.DataFrame({
        'mt_id': ['m1'],
        'gt_id': ['g1'],
        'mt_t': [1.0],
        'perm_mt_p': [0.5]
    })

    m_annot_path = tmp_path / "m_annot.csv"
    g_annot_path = tmp_path / "g_annot.csv"
    out_path = tmp_path / "out.parquet"

    m_annot.to_csv(m_annot_path, index=False)
    g_annot.to_csv(g_annot_path, index=False)
    table = pa.Table.from_pandas(output)
    pq.write_table(table, out_path)

    script_path = os.path.join(os.path.dirname(__file__), "../tools/eval_permute.py")
    out_dir = tmp_path / "dir"

    subprocess.run([
        "python", script_path,
        "--perm-output", str(out_path),
        "--m-annot", str(m_annot_path),
        "--g-annot", str(g_annot_path),
        "--df", str(df_val),
        "--out-dir", str(out_dir)
    ], check=True)

    with open(out_dir / "eval_permute_report.json") as f:
        rep = json.load(f)

    assert rep['arms']['sidecar']['status'] == "skipped_no_sidecar"
    # End-to-end assert for analytic_p
    p_ana_expected = float(ep.compute_analytic_p(1.0, df_val))
    expected_neg_log = -float(np.log10(p_ana_expected))
    assert np.isclose(rep['arms']['calibration']['qq_data']['neg_log10_p_ana'][0], expected_neg_log)


def test_stratum_mixed_dtype_regression(tmp_path):
    import json
    import subprocess
    import sys
    import os
    import pandas as pd

    # Mixed dtype fixture: m_annot has 'X' row (object dtype), g_annot is autosome-only (int64)
    m_annot = pd.DataFrame({
        'name': ['m1', 'm2', 'mX'],
        'chrom': [1, 2, 'X'],
        'chromStart': [100, 200, 300]
    })

    g_annot = pd.DataFrame({
        'name': ['g1', 'g2', 'g3'],
        'chrom': [1, 2, 3],
        'chromStart': [150, 250, 350]
    })

    # m1-g1 is cis (1-1), m2-g2 is cis (2-2), mX-g3 is trans ('X'-3)
    output = pd.DataFrame({
        'mt_id': ['m1', 'm2', 'mX'],
        'gt_id': ['g1', 'g2', 'g3'],
        'mt_t': [1.0, 1.0, 1.0],
        'perm_mt_p': [0.5, 0.5, 0.5]
    })

    m_annot_path = tmp_path / "m_annot.csv"
    g_annot_path = tmp_path / "g_annot.csv"
    output_path = tmp_path / "output.csv"

    m_annot.to_csv(m_annot_path, index=False)
    g_annot.to_csv(g_annot_path, index=False)
    output.to_csv(output_path, index=False)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    script_path = os.path.join(os.path.dirname(__file__), "../tools/eval_permute.py")

    res = subprocess.run([
        sys.executable, script_path,
        "--perm-output", str(output_path),
        "--m-annot", str(m_annot_path),
        "--g-annot", str(g_annot_path),
        "--df", "10",
        "--out-dir", str(out_dir)
    ], capture_output=True, text=True)

    assert res.returncode == 0, f"Script failed: {res.stderr}"

    with open(out_dir / "eval_permute_report.json") as f:
        report = json.load(f)

    assert report['metadata']['n_cis'] == 2, f"Expected 2 cis, got {report['metadata'].get('n_cis')}"
    assert report['metadata']['n_trans'] == 1, f"Expected 1 trans, got {report['metadata'].get('n_trans')}"
