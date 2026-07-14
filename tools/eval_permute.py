#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.stats

# ==============================================================================
# CONSTANTS & THRESHOLDS
# ==============================================================================
# Provisional — calibrate from eval output on GTP/MESA before blessing.
BULK_LO = 0.05
BULK_HI = 1.0  # mostly-null band for divergence
TAIL_P_ANA = 1e-4  # tail band for arm (a)
TOLERANCE_MEDIAN_LOG10_RATIO_DIFF = 0.5  # |Δ| below this => strata agree
LAMBDA_EXCESS_CONFOUND_FLAG = 0.2  # (λ_cis − λ_trans) above this => cis signal-contaminated

# Frozen Sidecar Contract: The sidecar .npz is expected to have these exactly.
# Arrays: bin_edges, hist_counts, overflow_count, total_count, topk_values
# Scalars: gpd_xi, gpd_sigma, gpd_u, gpd_N_u, gpd_N

# ==============================================================================
# GPD RECOVERY (Standalone Fitter for Sidecar Arm)
# ==============================================================================
def fit_gpd(data, u):
    """
    Fits Generalized Pareto Distribution to data > u using method of moments or MLE.
    This is a standalone stub/fitter used by the sidecar-gated arm.
    """
    exceedances = data[data > u] - u
    if len(exceedances) < 5:
        return np.nan, np.nan

    # Simple Method of Moments (MoM) as a robust fallback/fitter
    mean_ex = np.mean(exceedances)
    var_ex = np.var(exceedances, ddof=1)

    # MoM estimators for GPD
    xi = 0.5 * (1 - (mean_ex**2) / var_ex)
    sigma = 0.5 * mean_ex * (1 + (mean_ex**2) / var_ex)

    return xi, sigma

# ==============================================================================
# DIAGNOSTICS & ARMS
# ==============================================================================
def compute_calibration_stats(t_vals, p_perm, df_val, p_ana, is_bulk, is_tail):
    """Arm A.a: Calibration"""
    stats = {}

    # Bulk agreement
    if np.any(is_bulk):
        p_perm_bulk = p_perm[is_bulk]
        p_ana_bulk = p_ana[is_bulk]

        # Spearman correlation on -log10
        # Prevent log(0)
        p_perm_bulk_safe = np.clip(p_perm_bulk, a_min=1e-300, a_max=1.0)
        p_ana_bulk_safe = np.clip(p_ana_bulk, a_min=1e-300, a_max=1.0)

        corr, _ = scipy.stats.spearmanr(-np.log10(p_perm_bulk_safe), -np.log10(p_ana_bulk_safe))
        stats['bulk_spearman_corr'] = float(corr)

        log_ratio = np.log10(p_perm_bulk_safe / p_ana_bulk_safe)
        stats['bulk_median_abs_log_ratio'] = float(np.median(np.abs(log_ratio)))
        stats['bulk_90th_abs_log_ratio'] = float(np.percentile(np.abs(log_ratio), 90))
    else:
        stats['bulk_spearman_corr'] = None
        stats['bulk_median_abs_log_ratio'] = None
        stats['bulk_90th_abs_log_ratio'] = None

    # Tail behavior
    n_perm_below_analytic = int(np.sum(p_perm < p_ana))
    stats['n_perm_below_analytic'] = n_perm_below_analytic

    if np.any(is_tail):
        p_perm_tail = p_perm[is_tail]
        p_ana_tail = p_ana[is_tail]

        p_perm_tail_safe = np.clip(p_perm_tail, a_min=1e-300, a_max=1.0)
        p_ana_tail_safe = np.clip(p_ana_tail, a_min=1e-300, a_max=1.0)

        log_ratio_tail = np.log10(p_perm_tail_safe / p_ana_tail_safe)

        stats['tail_median_log_ratio'] = float(np.median(log_ratio_tail))
        stats['tail_10th_log_ratio'] = float(np.percentile(log_ratio_tail, 10))
        stats['tail_90th_log_ratio'] = float(np.percentile(log_ratio_tail, 90))
    else:
        stats['tail_median_log_ratio'] = None
        stats['tail_10th_log_ratio'] = None
        stats['tail_90th_log_ratio'] = None

    return stats

def calculate_genomic_inflation(t_vals):
    """Arm A.c: Genomic Inflation Lambda"""
    median_t_sq = np.median(t_vals**2)
    expected_median_chi2 = scipy.stats.chi2.ppf(0.5, 1)
    return float(median_t_sq / expected_median_chi2)

def main():
    parser = argparse.ArgumentParser(description="Phase 3 standalone read-only permutation-evaluation diagnostic")
    parser.add_argument("--perm-output", required=True, help="Path to qr_permute output, .parquet or .csv")
    parser.add_argument("--m-annot", required=True, help="Path to methylation annotation (mt_id -> chrom,...)")
    parser.add_argument("--g-annot", required=True, help="Path to expression annotation (gt_id -> chrom,...)")
    parser.add_argument("--df", required=True, type=int, help="Run-level df = n_samples - n_covars - 2. MUST equal the df that produced mt_t.")
    parser.add_argument("--perm-null-sidecar", help="Path to null-accumulator sidecar .npz. When absent, sidecar-gated arms are skipped.")
    parser.add_argument("--out-dir", required=True, help="Directory for the JSON report.")
    parser.add_argument("--bulk-lo", type=float, default=BULK_LO, help="Lower bound of p_ana for bulk band.")
    parser.add_argument("--bulk-hi", type=float, default=BULK_HI, help="Upper bound of p_ana for bulk band.")

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Setup & Validation
    # -------------------------------------------------------------------------
    if args.df <= 0 or not np.isfinite(args.df):
        print(f"Error: Non-finite or invalid df ({args.df}).", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "eval_permute_report.json")

    # Load data
    try:
        if args.perm_output.endswith(".parquet"):
            output = pq.read_table(args.perm_output).to_pandas()
            if output.index.names != [None]:
                output = output.reset_index()
        else:
            output = pd.read_csv(args.perm_output)

        m_annot = pd.read_csv(args.m_annot)
        g_annot = pd.read_csv(args.g_annot)

        # Replicate annotated_fixture logic
        if 'name' in m_annot.columns:
            m_annot = m_annot.set_index('name')
        if 'name' in g_annot.columns:
            g_annot = g_annot.set_index('name')

    except Exception as e:
        print(f"Error loading inputs: {e}", file=sys.stderr)
        sys.exit(1)

    if len(output) == 0:
        print("Error: Empty permutation output.", file=sys.stderr)
        sys.exit(1)

    required_cols = {'mt_id', 'gt_id', 'mt_t', 'perm_mt_p'}
    if not required_cols.issubset(output.columns):
        print(f"Error: Missing columns in permutation output. Requires {required_cols}", file=sys.stderr)
        sys.exit(1)

    # Map Chromosomes
    m_mapped = m_annot.index.astype(str).get_indexer(output['mt_id'].astype(str))
    g_mapped = g_annot.index.astype(str).get_indexer(output['gt_id'].astype(str))

    if (m_mapped == -1).any() or (g_mapped == -1).any():
        print("Error: Reported mt_id/gt_id missing from annotations.", file=sys.stderr)
        sys.exit(1)

    m_chrom = m_annot.iloc[m_mapped]['chrom'].to_numpy()
    g_chrom = g_annot.iloc[g_mapped]['chrom'].to_numpy()

    # -------------------------------------------------------------------------
    # Core Data Extraction
    # -------------------------------------------------------------------------
    t = output['mt_t'].to_numpy(dtype=np.float64)
    p_perm = output['perm_mt_p'].to_numpy(dtype=np.float64)
    df_val = args.df

    # Precise analytic ref
    p_ana = 2.0 * scipy.stats.t.sf(np.abs(t), df_val)

    # Strata
    is_cis = (m_chrom == g_chrom)
    is_trans = ~is_cis

    # Bands
    is_bulk = (p_ana >= args.bulk_lo) & (p_ana <= args.bulk_hi)
    is_tail = p_ana < TAIL_P_ANA

    report = {
        "metadata": {
            "n_pairs": len(t),
            "df": df_val,
            "bulk_lo": args.bulk_lo,
            "bulk_hi": args.bulk_hi,
            "tail_p_ana": TAIL_P_ANA
        },
        "arms": {}
    }

    # -------------------------------------------------------------------------
    # Arm A.a: Calibration
    # -------------------------------------------------------------------------
    calib = {}

    # all
    calib['all'] = compute_calibration_stats(t, p_perm, df_val, p_ana, is_bulk, is_tail)
    # cis
    calib['cis'] = compute_calibration_stats(t[is_cis], p_perm[is_cis], df_val, p_ana[is_cis], is_bulk[is_cis], is_tail[is_cis])
    # trans
    calib['trans'] = compute_calibration_stats(t[is_trans], p_perm[is_trans], df_val, p_ana[is_trans], is_bulk[is_trans], is_tail[is_trans])

    # Downsampled QQ Data (All)
    n_qq = min(len(t), 5000)
    rng = np.random.default_rng(42)
    qq_idx = rng.choice(len(t), size=n_qq, replace=False)

    p_ana_safe_qq = np.clip(p_ana[qq_idx], a_min=1e-300, a_max=1.0)
    p_perm_safe_qq = np.clip(p_perm[qq_idx], a_min=1e-300, a_max=1.0)

    calib['qq_data'] = {
        "neg_log10_p_ana": (-np.log10(p_ana_safe_qq)).tolist(),
        "neg_log10_p_perm": (-np.log10(p_perm_safe_qq)).tolist()
    }

    report["arms"]["calibration"] = calib

    # -------------------------------------------------------------------------
    # Arm A.c: Null-Sanity
    # -------------------------------------------------------------------------
    null_sanity = {}

    lambda_trans = calculate_genomic_inflation(t[is_trans]) if np.any(is_trans) else None
    lambda_cis = calculate_genomic_inflation(t[is_cis]) if np.any(is_cis) else None

    null_sanity['lambda_trans'] = lambda_trans
    null_sanity['lambda_cis'] = lambda_cis
    null_sanity['lambda_cis_label'] = "expected > 1 under real cis signal; NOT a miscalibration flag"

    trans_bulk_mask = is_trans & is_bulk
    if np.any(trans_bulk_mask):
        ks_stat, ks_p = scipy.stats.kstest(p_ana[trans_bulk_mask], 'uniform')
        null_sanity['ks_trans_bulk_vs_uniform'] = {"stat": float(ks_stat), "p": float(ks_p)}
    else:
        null_sanity['ks_trans_bulk_vs_uniform'] = None

    report["arms"]["null_sanity"] = null_sanity

    # -------------------------------------------------------------------------
    # Arm A.d: Stratify-or-not
    # -------------------------------------------------------------------------
    stratify = {}

    cis_bulk_mask = is_cis & is_bulk

    if np.any(trans_bulk_mask) and np.any(cis_bulk_mask):
        p_perm_trans_safe = np.clip(p_perm[trans_bulk_mask], a_min=1e-300, a_max=1.0)
        p_ana_trans_safe = np.clip(p_ana[trans_bulk_mask], a_min=1e-300, a_max=1.0)

        p_perm_cis_safe = np.clip(p_perm[cis_bulk_mask], a_min=1e-300, a_max=1.0)
        p_ana_cis_safe = np.clip(p_ana[cis_bulk_mask], a_min=1e-300, a_max=1.0)

        log_ratio_trans = np.log10(p_perm_trans_safe / p_ana_trans_safe)
        log_ratio_cis = np.log10(p_perm_cis_safe / p_ana_cis_safe)

        median_trans = float(np.median(log_ratio_trans))
        median_cis = float(np.median(log_ratio_cis))

        delta = median_cis - median_trans

        mw_stat, mw_p = scipy.stats.mannwhitneyu(log_ratio_cis, log_ratio_trans, alternative='two-sided')
        ks_stat2, ks_p2 = scipy.stats.kstest(log_ratio_cis, log_ratio_trans)

        stratify['median_log10_ratio_trans'] = median_trans
        stratify['median_log10_ratio_cis'] = median_cis
        stratify['delta_median_log10_ratio'] = delta

        stratify['test_name'] = "mann_whitney_u"
        stratify['test_stat'] = float(mw_stat)
        stratify['test_p'] = float(mw_p)
        stratify['ks_stat'] = float(ks_stat2)
        stratify['ks_p'] = float(ks_p2)

        lambda_excess = (lambda_cis - lambda_trans) if (lambda_cis is not None and lambda_trans is not None) else 0.0
        stratify['lambda_excess'] = float(lambda_excess)

        if abs(delta) < TOLERANCE_MEDIAN_LOG10_RATIO_DIFF:
            rec = "single_global_null_adequate"
        elif lambda_excess > LAMBDA_EXCESS_CONFOUND_FLAG:
            rec = "inconclusive_cis_signal_confound"
        else:
            rec = "stratification_warranted"

        stratify['recommendation'] = rec
    else:
        stratify['status'] = "skipped_insufficient_data"

    report["arms"]["stratify_decision"] = stratify

    # -------------------------------------------------------------------------
    # Arm B: Sidecar-gated
    # -------------------------------------------------------------------------
    sidecar_arm = {}

    if args.perm_null_sidecar:
        if os.path.exists(args.perm_null_sidecar):
            # Future sidecar integration
            sidecar_arm['status'] = "loaded"
            # TODO: Literal null-flatness / rigorous null-shape stratify
        else:
            warnings.warn(f"Sidecar file specified but not found: {args.perm_null_sidecar}")
            sidecar_arm['status'] = "skipped_no_sidecar"
    else:
        print("Warning: --perm-null-sidecar not provided. Skipping sidecar-gated arms.")
        sidecar_arm['status'] = "skipped_no_sidecar"

    report["arms"]["sidecar"] = sidecar_arm

    # Write JSON report
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report written to {report_path}")

if __name__ == "__main__":
    main()
