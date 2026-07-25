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
# ORACLE 2: GPD-RECOVERY
# ==============================================================================
def test_gpd_recovery_oracle():
    """
    Draw genpareto samples with known (xi, sigma), fit via the script's fitter,
    assert recovery within a stated tolerance. Includes a xi approx 0 case.
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


def test_label_strata_nan_chrom():
    m_annot = pd.DataFrame({'name': ['m1', 'm2', 'm3', 'm4'], 'chrom': [1, 2, None, 4]}).set_index('name')
    g_annot = pd.DataFrame({'name': ['g1', 'g2', 'g3', 'g4'], 'chrom': [1, 3, 3, None]}).set_index('name')
    output = pd.DataFrame({'mt_id': ['m1', 'm2', 'm3', 'm4'], 'gt_id': ['g1', 'g2', 'g3', 'g4']})
    keep, is_cis, is_trans, n_cis, n_trans, n_dropped = ep.label_strata(output, m_annot, g_annot)

    assert n_dropped == 2
    assert keep.tolist() == [True, True, False, False]
    assert n_cis == 1
    assert n_trans == 1

def test_load_annotation_real_bed6_shape(tmp_path):
    annot_path = tmp_path / "real_bed6.bed"
    with open(annot_path, "w") as f:
        f.write("chrom\tchromStart\tchromEnd\tname\tscore\tstrand\n")
        f.write("chr1\t1\t100\tm1\t0\t+\n")
        f.write("chr2\t101\t200\tm2\t0\t+\n")
        f.write("chr3\t201\t300\tm3\t0\t+\n")
        f.write("chr4\t301\t400\tm4\t0\t+\n")
        f.write("chr5\t401\t500\tm5\t0\t+\n")
        f.write("\t501\t600\tm6\t0\t+\n") # Empty chrom

    annot = ep._load_annotation(str(annot_path), "test")
    assert annot.shape == (6, 5)
    assert list(annot.columns) == ['chrom', 'chromStart', 'chromEnd', 'score', 'strand']
    assert annot.index.name == 'name'

def test_load_annotation_bare_int_fallback(tmp_path):
    annot_path = tmp_path / "bare_int.bed"
    with open(annot_path, "w") as f:
        f.write("chrom\tchromStart\tchromEnd\tname\tscore\tstrand\n")
        f.write("1\t1\t100\tm1\t0\t+\n")
        f.write("2\t101\t200\tm2\t0\t+\n")
        f.write("3\t201\t300\tm3\t0\t+\n")
        f.write("4\t301\t400\tm4\t0\t+\n")
        f.write("5\t401\t500\tm5\t0\t+\n")
        f.write("\t501\t600\tm6\t0\t+\n") # Empty chrom

    annot = ep._load_annotation(str(annot_path), "test")
    assert annot.shape == (6, 5)
    assert list(annot.columns) == ['chrom', 'chromStart', 'chromEnd', 'score', 'strand']
    assert annot.index.name == 'name'

def test_load_annotation_missing_name_column_fails_closed(tmp_path):
    annot_path = tmp_path / "bad.csv"
    with open(annot_path, "w") as f:
        f.write("chrom,chromStart,chromEnd,score,strand\n")
        f.write("chr1,1,100,0,+\n")

    with pytest.raises(ValueError, match="has no 'name' column") as excinfo:
        ep._load_annotation(str(annot_path), "test")
    assert "separator mismatch" in str(excinfo.value)

def test_load_annotation_duplicate_names_fails_closed(tmp_path):
    annot_path = tmp_path / "dup.bed"
    with open(annot_path, "w") as f:
        f.write("chrom\tchromStart\tchromEnd\tname\tscore\tstrand\n")
        f.write("chr1\t1\t100\tm1\t0\t+\n")
        f.write("chr2\t101\t200\tm1\t0\t+\n")

    with pytest.raises(ValueError, match="is not unique") as excinfo:
        ep._load_annotation(str(annot_path), "test")
    assert "(1 duplicated names)" in str(excinfo.value)

def test_canon_chrom_preserves_nan():
    res = ep._canon_chrom(['chr1', np.nan, '2.0', 'X'], "test")
    res_list = res.tolist()
    assert res_list[0] == '1'
    assert pd.isna(res_list[1])
    assert res_list[2] == '2'
    assert res_list[3] == 'X'
    # pd.NA != 'nan' evaluates to pd.NA, which raises TypeError when asserted.
    # We just want to ensure it is not the string 'nan'.
    assert not (isinstance(res[1], str) and res[1] == 'nan')

def test_label_strata_all_dropped_fails_closed():
    m_annot = pd.DataFrame({'name': ['m1', 'm2'], 'chrom': [None, 1]}).set_index('name')
    g_annot = pd.DataFrame({'name': ['g1', 'g2'], 'chrom': [1, None]}).set_index('name')
    output = pd.DataFrame({'mt_id': ['m1', 'm2'], 'gt_id': ['g1', 'g2']})
    with pytest.raises(ValueError, match="All reported pairs dropped"):
        ep.label_strata(output, m_annot, g_annot)

def test_main_alignment_after_drop(tmp_path):
    df_val = 50
    m_annot = pd.DataFrame({'name': ['m1', 'm2', 'm3', 'm4'], 'chrom': [1, 2, None, 4]})
    g_annot = pd.DataFrame({'name': ['g1', 'g2', 'g3', 'g4'], 'chrom': [1, 3, 3, None]})

    m_annot_path = tmp_path / "m_annot.bed"
    g_annot_path = tmp_path / "g_annot.bed"
    m_annot.to_csv(m_annot_path, sep="\t", index=False)
    g_annot.to_csv(g_annot_path, sep="\t", index=False)

    t_vals = [2.0, -1.5, 999.0, 999.0]
    p_perm = [0.05, 0.15, 0.0, 0.0]

    output = pd.DataFrame({
        'mt_id': ['m1', 'm2', 'm3', 'm4'],
        'gt_id': ['g1', 'g2', 'g3', 'g4'],
        'mt_t': t_vals,
        'perm_mt_p': p_perm
    })

    out_path = tmp_path / "out.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pandas(output), out_path)

    out_dir = tmp_path / "dir"
    script_path = os.path.join(os.path.dirname(__file__), "../tools/eval_permute.py")

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

    assert rep['metadata']['n_pairs_input'] == 4
    assert rep['metadata']['n_pairs_dropped_unmappable_chrom'] == 2
    assert rep['metadata']['n_pairs_scored'] == 2
    assert rep['metadata']['n_pairs_scored'] == rep['metadata']['n_cis'] + rep['metadata']['n_trans']

    import scipy.stats
    expected_p_ana = 2.0 * scipy.stats.t.sf(np.abs([2.0, -1.5]), df_val)
    expected_neg_log = -np.log10(expected_p_ana)
    actual_neg_log = rep['arms']['calibration']['qq_data']['neg_log10_p_ana']

    assert np.allclose(actual_neg_log[:2], expected_neg_log[:2])

def test_report_carries_drop_counts(tmp_path):
    df_val = 50
    m_annot = pd.DataFrame({'name': ['m1', 'm2'], 'chrom': [1, 2]})
    g_annot = pd.DataFrame({'name': ['g1', 'g2'], 'chrom': [1, 3]})

    m_annot_path = tmp_path / "m_annot.bed"
    g_annot_path = tmp_path / "g_annot.bed"
    m_annot.to_csv(m_annot_path, sep="\t", index=False)
    g_annot.to_csv(g_annot_path, sep="\t", index=False)

    t_vals = [2.0, -1.5]
    p_perm = [0.05, 0.15]

    output = pd.DataFrame({
        'mt_id': ['m1', 'm2'],
        'gt_id': ['g1', 'g2'],
        'mt_t': t_vals,
        'perm_mt_p': p_perm
    })

    out_path = tmp_path / "out.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pandas(output), out_path)

    out_dir = tmp_path / "dir"
    script_path = os.path.join(os.path.dirname(__file__), "../tools/eval_permute.py")

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

    assert 'n_pairs_input' in rep['metadata']
    assert 'n_pairs_dropped_unmappable_chrom' in rep['metadata']
    assert 'n_pairs_scored' in rep['metadata']
    assert 'n_pairs' not in rep['metadata']

    assert rep['metadata']['n_pairs_input'] == 2
    assert rep['metadata']['n_pairs_dropped_unmappable_chrom'] == 0
    assert rep['metadata']['n_pairs_scored'] == 2

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
    # lambda_excess no longer gates the verdict; divergence causes a warrant.
    assert rep3['arms']['stratify_decision']['recommendation'] == "stratification_warranted"

def test_lambda_excess_does_not_gate(tmp_path):
    """
    Demonstrates the demotion of lambda_excess from gating factor.
    We construct a case where lambda_excess > 0.2 and delta >= 0.5.
    The verdict should be "stratification_warranted", not an inconclusive confound.
    """
    script_path = os.path.join(os.path.dirname(__file__), "..", "tools", "eval_permute.py")
    df_val = 50
    rng = np.random.default_rng(42)
    n = 10000

    m_annot = pd.DataFrame({'name': [f'm{i}' for i in range(n)], 'chrom': [1]*n})
    g_annot = pd.DataFrame({'name': [f'g{i}' for i in range(n)], 'chrom': [1]*(n//2) + [2]*(n//2)})

    m_annot_path = tmp_path / "m_annot.csv"
    g_annot_path = tmp_path / "g_annot.csv"
    m_annot.to_csv(m_annot_path, index=False)
    g_annot.to_csv(g_annot_path, index=False)

    t_vals = scipy.stats.t.rvs(df_val, size=n, random_state=rng)
    p_ana = 2.0 * scipy.stats.t.sf(np.abs(t_vals), df_val)

    p_perm = p_ana.copy()
    p_perm[:n//2] = p_perm[:n//2] * 0.05
    t_vals_inflated = t_vals.copy()
    t_vals_inflated[:n//2] = t_vals_inflated[:n//2] * 2.0  # Inflate cis t-values substantially

    output = pd.DataFrame({
        'mt_id': [f'm{i}' for i in range(n)],
        'gt_id': [f'g{i}' for i in range(n)],
        'mt_t': t_vals_inflated,
        'perm_mt_p': p_perm
    })
    out_path = tmp_path / "out.parquet"
    pq.write_table(pa.Table.from_pandas(output), out_path)

    out_dir = tmp_path / "dir_lambda_gate"
    subprocess.run([
        sys.executable, script_path,
        "--perm-output", str(out_path),
        "--m-annot", str(m_annot_path),
        "--g-annot", str(g_annot_path),
        "--df", str(df_val),
        "--out-dir", str(out_dir)
    ], check=True)

    with open(out_dir / "eval_permute_report.json") as f:
        rep = json.load(f)

    strat = rep['arms']['stratify_decision']
    assert 'recommendation' in strat                         # block executed, not skipped
    assert abs(strat['delta_median_log10_ratio']) >= 0.5     # divergence present
    assert strat['lambda_excess'] > 0.2                      # λ still high AND still reported ...
    assert strat['recommendation'] == "stratification_warranted"   # ... but no longer gates to inconclusive

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



@pytest.mark.parametrize("m_chrom_col, g_chrom_col", [
    ([1, 2, 'X'], [1, 2, 3]),  # object-vs-int (existing)
    ([1.0, 2.0, None], [1, 2, 3]),  # float-vs-int (None on unused forces float64)
    (['chr1', 'chr2', 'chr3'], [1, 2, 3])  # 'chr1'-vs-1
])
def test_stratum_mixed_dtype_regression(tmp_path, m_chrom_col, g_chrom_col):
    import json
    import subprocess
    import sys
    import os
    import pandas as pd

    m_annot = pd.DataFrame({
        'name': ['m1', 'm2', 'm3'],
        'chrom': m_chrom_col,
        'chromStart': [100, 200, 300]
    })

    g_annot = pd.DataFrame({
        'name': ['g1', 'g2', 'g3'],
        'chrom': g_chrom_col,
        'chromStart': [150, 250, 350]
    })

    # m1-g1 is cis (1-1), m2-g2 is cis (2-2)
    output = pd.DataFrame({
        'mt_id': ['m1', 'm2'],
        'gt_id': ['g1', 'g2'],
        'mt_t': [1.0, 1.0],
        'perm_mt_p': [0.5, 0.5]
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
    assert report['metadata']['n_trans'] == 0, f"Expected 0 trans, got {report['metadata'].get('n_trans')}"

def test_has_region_oracle(tmp_path):
    df_val = 50
    rng = np.random.default_rng(42)
    n = 1000

    m_annot = pd.DataFrame({'name': [f'm{i}' for i in range(n)], 'chrom': [1]*n})
    g_annot = pd.DataFrame({'name': [f'g{i}' for i in range(n)], 'chrom': [1]*(n//2) + [2]*(n//2)})

    m_annot_path = tmp_path / "m_annot.csv"
    g_annot_path = tmp_path / "g_annot.csv"
    m_annot.to_csv(m_annot_path, index=False)
    g_annot.to_csv(g_annot_path, index=False)

    regions = ep.CANONICAL_REGIONS
    assigned_regions = [regions[i % len(regions)] for i in range(n)]

    # Introduce some dropped regions
    assigned_regions[10] = None
    assigned_regions[20] = np.nan

    t_vals = scipy.stats.t.rvs(df_val, size=n, random_state=rng)
    p_ana = 2.0 * scipy.stats.t.sf(np.abs(t_vals), df_val)

    output = pd.DataFrame({
        'mt_id': [f'm{i}' for i in range(n)],
        'gt_id': [f'g{i}' for i in range(n)],
        'mt_t': t_vals,
        'perm_mt_p': p_ana,
        'region': assigned_regions
    })

    out_path = tmp_path / "out.parquet"
    table = pa.Table.from_pandas(output)
    pq.write_table(table, out_path)

    script_path = os.path.join(os.path.dirname(__file__), "../tools/eval_permute.py")
    out_dir = tmp_path / "dir"

    subprocess.run([
        sys.executable, script_path,
        "--perm-output", str(out_path),
        "--m-annot", str(m_annot_path),
        "--g-annot", str(g_annot_path),
        "--df", str(df_val),
        "--out-dir", str(out_dir)
    ], check=True)

    with open(out_dir / "eval_permute_report.json") as f:
        rep = json.load(f)

    # Oracle assertion: n_pairs_scored + dropped = total
    assert rep['metadata']['n_pairs_scored'] + rep['metadata']['n_pairs_dropped_null_region'] == n
    assert rep['metadata']['n_pairs_dropped_null_region'] == 2
    assert rep['metadata']['n_pairs_dropped_unmappable_chrom'] == 0

    # Ensure n_by_region adds up to n_pairs_scored
    assert sum(rep['metadata']['n_by_region'].values()) == rep['metadata']['n_pairs_scored']

    strat = rep['arms']['stratify_decision']
    assert strat['mode'] == 'per_region'
    assert strat['reference'] == 'TRANS'

def test_forced_fail_promoter(tmp_path):
    df_val = 50
    rng = np.random.default_rng(42)
    n = 2000

    m_annot = pd.DataFrame({'name': [f'm{i}' for i in range(n)], 'chrom': [1]*n})
    g_annot = pd.DataFrame({'name': [f'g{i}' for i in range(n)], 'chrom': [1]*(n//2) + [2]*(n//2)})

    m_annot_path = tmp_path / "m_annot.csv"
    g_annot_path = tmp_path / "g_annot.csv"
    m_annot.to_csv(m_annot_path, index=False)
    g_annot.to_csv(g_annot_path, index=False)

    regions = ep.CANONICAL_REGIONS
    assigned_regions = [regions[i % len(regions)] for i in range(n)]

    t_vals = scipy.stats.t.rvs(df_val, size=n, random_state=rng)
    p_ana = 2.0 * scipy.stats.t.sf(np.abs(t_vals), df_val)
    p_perm = p_ana.copy()

    # Step 1: Prove it passes initially (adequate)
    output1 = pd.DataFrame({
        'mt_id': [f'm{i}' for i in range(n)],
        'gt_id': [f'g{i}' for i in range(n)],
        'mt_t': t_vals,
        'perm_mt_p': p_perm,
        'region': assigned_regions
    })

    out1_path = tmp_path / "out1.parquet"
    pq.write_table(pa.Table.from_pandas(output1), out1_path)

    script_path = os.path.join(os.path.dirname(__file__), "../tools/eval_permute.py")
    out_dir1 = tmp_path / "dir1"

    subprocess.run([
        sys.executable, script_path,
        "--perm-output", str(out1_path),
        "--m-annot", str(m_annot_path),
        "--g-annot", str(g_annot_path),
        "--df", str(df_val),
        "--out-dir", str(out_dir1)
    ], check=True)

    with open(out_dir1 / "eval_permute_report.json") as f:
        rep1 = json.load(f)

    assert rep1['arms']['stratify_decision']['recommendation'] == "single_global_null_adequate"
    assert rep1['arms']['stratify_decision']['divergent_regions'] == []

    # Step 2: Push PROMOTER out of tolerance
    p_perm2 = p_perm.copy()
    promoter_mask = np.array(assigned_regions) == 'PROMOTER'
    p_perm2[promoter_mask] *= 0.05

    output2 = pd.DataFrame({
        'mt_id': [f'm{i}' for i in range(n)],
        'gt_id': [f'g{i}' for i in range(n)],
        'mt_t': t_vals,
        'perm_mt_p': p_perm2,
        'region': assigned_regions
    })

    out2_path = tmp_path / "out2.parquet"
    pq.write_table(pa.Table.from_pandas(output2), out2_path)

    out_dir2 = tmp_path / "dir2"

    subprocess.run([
        sys.executable, script_path,
        "--perm-output", str(out2_path),
        "--m-annot", str(m_annot_path),
        "--g-annot", str(g_annot_path),
        "--df", str(df_val),
        "--out-dir", str(out_dir2)
    ], check=True)

    with open(out_dir2 / "eval_permute_report.json") as f:
        rep2 = json.load(f)

    assert rep2['arms']['stratify_decision']['recommendation'] == "stratification_warranted"
    assert "PROMOTER" in rep2['arms']['stratify_decision']['divergent_regions']

def test_condition_b_insufficient_near_gene(tmp_path):
    df_val = 50
    rng = np.random.default_rng(42)

    # 200 TRANS (plenty), but only 10 CIS5 (insufficient pooled near gene)
    n_trans = 200
    n_cis5 = 10
    n = n_trans + n_cis5

    m_annot = pd.DataFrame({'name': [f'm{i}' for i in range(n)], 'chrom': [1]*n})
    g_annot = pd.DataFrame({'name': [f'g{i}' for i in range(n)], 'chrom': [1]*n})

    m_annot_path = tmp_path / "m_annot.csv"
    g_annot_path = tmp_path / "g_annot.csv"
    m_annot.to_csv(m_annot_path, index=False)
    g_annot.to_csv(g_annot_path, index=False)

    assigned_regions = ['TRANS'] * n_trans + ['CIS5'] * n_cis5

    t_vals = scipy.stats.t.rvs(df_val, size=n, random_state=rng)
    p_ana = 2.0 * scipy.stats.t.sf(np.abs(t_vals), df_val)

    output = pd.DataFrame({
        'mt_id': [f'm{i}' for i in range(n)],
        'gt_id': [f'g{i}' for i in range(n)],
        'mt_t': t_vals,
        'perm_mt_p': p_ana,
        'region': assigned_regions
    })

    out_path = tmp_path / "out.parquet"
    pq.write_table(pa.Table.from_pandas(output), out_path)

    script_path = os.path.join(os.path.dirname(__file__), "../tools/eval_permute.py")
    out_dir = tmp_path / "dir"

    subprocess.run([
        sys.executable, script_path,
        "--perm-output", str(out_path),
        "--m-annot", str(m_annot_path),
        "--g-annot", str(g_annot_path),
        "--df", str(df_val),
        "--out-dir", str(out_dir)
    ], check=True)

    with open(out_dir / "eval_permute_report.json") as f:
        rep = json.load(f)

    strat = rep['arms']['stratify_decision']
    assert strat['recommendation'] == "insufficient_near_gene_coverage"
    assert strat['median_log10_ratio_cis'] is None
    assert strat['delta_median_log10_ratio'] is None
    assert strat['test_stat'] is None
    assert strat['test_p'] is None
    assert strat['ks_stat'] is None
    assert strat['ks_p'] is None
    # lambda_excess should still be populated
    assert strat['lambda_excess'] is not None

def test_unexpected_region_fails_closed(tmp_path):
    output = pd.DataFrame({
        'mt_id': ['m1'], 'gt_id': ['g1'], 'mt_t': [1.0], 'perm_mt_p': [0.5], 'region': ['JUNK']
    })
    out_path = tmp_path / "out.parquet"
    pq.write_table(pa.Table.from_pandas(output), out_path)

    m_annot = pd.DataFrame({'name': ['m1'], 'chrom': [1]})
    g_annot = pd.DataFrame({'name': ['g1'], 'chrom': [1]})
    m_annot_path = tmp_path / "m_annot.csv"
    g_annot_path = tmp_path / "g_annot.csv"
    m_annot.to_csv(m_annot_path, index=False)
    g_annot.to_csv(g_annot_path, index=False)

    script_path = os.path.join(os.path.dirname(__file__), "../tools/eval_permute.py")
    out_dir = tmp_path / "dir"

    res = subprocess.run([
        sys.executable, script_path,
        "--perm-output", str(out_path),
        "--m-annot", str(m_annot_path),
        "--g-annot", str(g_annot_path),
        "--df", "50",
        "--out-dir", str(out_dir)
    ], capture_output=True, text=True)

    assert res.returncode != 0
    assert "unexpected region labels" in res.stderr

def test_fallback_byte_identity(tmp_path):
    """
    Oracle-based fallback-equivalence test: evaluates a region-less fixture whose numbers are analytically known.
    Ensures that values have not drifted and schema shape remains byte-identical to the original legacy path.
    """
    df_val = 50
    rng = np.random.default_rng(42)
    n = 1000

    m_annot = pd.DataFrame({'name': [f'm{i}' for i in range(n)], 'chrom': [1]*n})
    g_annot = pd.DataFrame({'name': [f'g{i}' for i in range(n)], 'chrom': [1]*(n//2) + [2]*(n//2)})

    m_annot_path = tmp_path / "m_annot.csv"
    g_annot_path = tmp_path / "g_annot.csv"
    m_annot.to_csv(m_annot_path, index=False)
    g_annot.to_csv(g_annot_path, index=False)

    t_vals = scipy.stats.t.rvs(df_val, size=n, random_state=rng)
    p_ana = 2.0 * scipy.stats.t.sf(np.abs(t_vals), df_val)

    output = pd.DataFrame({
        'mt_id': [f'm{i}' for i in range(n)],
        'gt_id': [f'g{i}' for i in range(n)],
        'mt_t': t_vals,
        'perm_mt_p': p_ana
    })

    out_path = tmp_path / "out.parquet"
    pq.write_table(pa.Table.from_pandas(output), out_path)

    script_path = os.path.join(os.path.dirname(__file__), "../tools/eval_permute.py")
    out_dir = tmp_path / "dir"

    subprocess.run([
        sys.executable, script_path,
        "--perm-output", str(out_path),
        "--m-annot", str(m_annot_path),
        "--g-annot", str(g_annot_path),
        "--df", str(df_val),
        "--out-dir", str(out_dir)
    ], check=True)

    with open(out_dir / "eval_permute_report.json") as f:
        rep = json.load(f)

    # Assert schema
    assert 'n_by_region' not in rep['metadata']
    assert 'n_pairs_dropped_null_region' not in rep['metadata']
    assert 'per_region' not in rep['arms']['stratify_decision']
    assert 'mode' not in rep['arms']['stratify_decision']

    # Assert labeling oracle
    assert rep['metadata']['n_cis'] == n // 2
    assert rep['metadata']['n_trans'] == n // 2
    assert rep['metadata']['n_pairs_scored'] == n

    # Assert value oracle (all should be ~0 as perm_mt_p == p_ana)
    calib = rep['arms']['calibration']
    stratify = rep['arms']['stratify_decision']

    assert calib['cis']['bulk_median_abs_log_ratio'] == pytest.approx(0.0, abs=1e-9)
    assert calib['trans']['bulk_median_abs_log_ratio'] == pytest.approx(0.0, abs=1e-9)
    assert calib['all']['bulk_median_abs_log_ratio'] == pytest.approx(0.0, abs=1e-9)

    assert stratify['median_log10_ratio_cis'] == pytest.approx(0.0, abs=1e-9)
    assert stratify['median_log10_ratio_trans'] == pytest.approx(0.0, abs=1e-9)
    assert stratify['delta_median_log10_ratio'] == pytest.approx(0.0, abs=1e-9)
    assert stratify['recommendation'] == 'single_global_null_adequate'

def test_eval_permute_metadata_floor_resolved(tmp_path, master_parquet_fixture, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import sys
    sys.path.insert(0, 'tools')
    from eval_permute import main

    df = pd.DataFrame({
        'mt_id': ['cg001'],
        'gt_id': ['ILMN_001'],
        'perm_mt_p': [0.5],
        'mt_t': [2.0],
        'n_perm': [100]
    })
    table = pa.Table.from_pandas(df)
    meta = {
        b'tecpg_perm_n_perm': b'100',
        b'tecpg_perm_n_null_pairs': b'5000'
    }
    table = table.replace_schema_metadata(meta)

    perm_file = tmp_path / "perm.parquet"
    pq.write_table(table, str(perm_file))

    m_annot = tmp_path / "m.csv"
    g_annot = tmp_path / "g.csv"
    pd.DataFrame({'chrom': ['chr1'], 'chromStart': [0], 'chromEnd': [1], 'name': ['cg001'], 'score': [0], 'strand': ['+']}).to_csv(m_annot, index=False, sep='\t')
    pd.DataFrame({'chrom': ['chr1'], 'chromStart': [0], 'chromEnd': [1], 'name': ['ILMN_001'], 'score': [0], 'strand': ['+']}).to_csv(g_annot, index=False, sep='\t')

    out_dir = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["eval_permute.py", "--perm-output", str(perm_file), "--m-annot", str(m_annot), "--g-annot", str(g_annot), "--out-dir", str(out_dir), "--df", "321"])

    try:
        main()
    except SystemExit as e:
        assert e.code == 0

    import json
    with open(out_dir / "eval_permute_report.json", "r") as f:
        report = json.load(f)

    assert report['metadata']['n_perm'] == 100
    assert report['metadata']['n_null_pairs'] == 5000
    assert report['metadata']['perm_resolution_floor'] == 1.0 / (100 * 5000)

def test_eval_permute_metadata_missing_no_cli(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import sys
    sys.path.insert(0, 'tools')
    from eval_permute import main

    df = pd.DataFrame({
        'mt_id': ['cg001'],
        'gt_id': ['ILMN_001'],
        'perm_mt_p': [0.5],
        'mt_t': [2.0]
    })
    table = pa.Table.from_pandas(df)

    perm_file = tmp_path / "perm.parquet"
    pq.write_table(table, str(perm_file))

    m_annot = tmp_path / "m.csv"
    g_annot = tmp_path / "g.csv"
    pd.DataFrame({'chrom': ['chr1'], 'chromStart': [0], 'chromEnd': [1], 'name': ['cg001'], 'score': [0], 'strand': ['+']}).to_csv(m_annot, index=False, sep='\t')
    pd.DataFrame({'chrom': ['chr1'], 'chromStart': [0], 'chromEnd': [1], 'name': ['ILMN_001'], 'score': [0], 'strand': ['+']}).to_csv(g_annot, index=False, sep='\t')

    out_dir = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["eval_permute.py", "--perm-output", str(perm_file), "--m-annot", str(m_annot), "--g-annot", str(g_annot), "--out-dir", str(out_dir), "--df", "321"])

    try:
        main()
    except SystemExit as e:
        assert e.code == 0

    import json
    with open(out_dir / "eval_permute_report.json", "r") as f:
        report = json.load(f)

    assert 'n_perm' in report['metadata']
    assert report['metadata']['n_perm'] is None
    assert 'n_null_pairs' in report['metadata']
    assert report['metadata']['n_null_pairs'] is None
    assert 'perm_resolution_floor' in report['metadata']
    assert report['metadata']['perm_resolution_floor'] is None

def test_eval_permute_cli_flag_wins(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import sys
    sys.path.insert(0, 'tools')
    from eval_permute import main

    df = pd.DataFrame({
        'mt_id': ['cg001'],
        'gt_id': ['ILMN_001'],
        'perm_mt_p': [0.5],
        'mt_t': [2.0]
    })
    table = pa.Table.from_pandas(df)
    meta = {
        b'tecpg_perm_n_perm': b'100',
        b'tecpg_perm_n_null_pairs': b'5000'
    }
    table = table.replace_schema_metadata(meta)

    perm_file = tmp_path / "perm.parquet"
    pq.write_table(table, str(perm_file))

    m_annot = tmp_path / "m.csv"
    g_annot = tmp_path / "g.csv"
    pd.DataFrame({'chrom': ['chr1'], 'chromStart': [0], 'chromEnd': [1], 'name': ['cg001'], 'score': [0], 'strand': ['+']}).to_csv(m_annot, index=False, sep='\t')
    pd.DataFrame({'chrom': ['chr1'], 'chromStart': [0], 'chromEnd': [1], 'name': ['ILMN_001'], 'score': [0], 'strand': ['+']}).to_csv(g_annot, index=False, sep='\t')

    out_dir = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["eval_permute.py", "--perm-output", str(perm_file), "--m-annot", str(m_annot), "--g-annot", str(g_annot), "--out-dir", str(out_dir), "--n-null-pairs", "2000", "--df", "321"])

    try:
        main()
    except SystemExit as e:
        assert e.code == 0

    import json
    with open(out_dir / "eval_permute_report.json", "r") as f:
        report = json.load(f)

    assert report['metadata']['n_perm'] == 100
    assert report['metadata']['n_null_pairs'] == 2000
    assert report['metadata']['perm_resolution_floor'] == 1.0 / (100 * 2000)

def test_eval_permute_n_perm_fallback(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import sys
    sys.path.insert(0, 'tools')
    from eval_permute import main

    df = pd.DataFrame({
        'mt_id': ['cg001'],
        'gt_id': ['ILMN_001'],
        'perm_mt_p': [0.5],
        'mt_t': [2.0],
        'n_perm': [100]
    })
    table = pa.Table.from_pandas(df)

    perm_file = tmp_path / "perm.parquet"
    pq.write_table(table, str(perm_file))

    m_annot = tmp_path / "m.csv"
    g_annot = tmp_path / "g.csv"
    pd.DataFrame({'chrom': ['chr1'], 'chromStart': [0], 'chromEnd': [1], 'name': ['cg001'], 'score': [0], 'strand': ['+']}).to_csv(m_annot, index=False, sep='\t')
    pd.DataFrame({'chrom': ['chr1'], 'chromStart': [0], 'chromEnd': [1], 'name': ['ILMN_001'], 'score': [0], 'strand': ['+']}).to_csv(g_annot, index=False, sep='\t')

    out_dir = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["eval_permute.py", "--perm-output", str(perm_file), "--m-annot", str(m_annot), "--g-annot", str(g_annot), "--out-dir", str(out_dir), "--n-null-pairs", "2000", "--df", "321"])

    try:
        main()
    except SystemExit as e:
        assert e.code == 0

    import json
    with open(out_dir / "eval_permute_report.json", "r") as f:
        report = json.load(f)

    assert report['metadata']['n_perm'] == 100
