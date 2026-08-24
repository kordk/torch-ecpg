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
MIN_REGION_BULK_N = 100  # provisional: min bulk pairs for a region to be testable
CANONICAL_REGIONS = ['TRANS', 'DISTAL5', 'CIS5', 'PROMOTER', 'GENEBODY', 'CIS3', 'DISTAL3']
NEAR_GENE_REGIONS = ['CIS5', 'PROMOTER', 'GENEBODY', 'CIS3']
REGION_REFERENCE = 'TRANS'

# Frozen Sidecar Contract: The sidecar .npz is expected to have these exactly.
# Arrays: bin_edges, hist_counts, overflow_count, total_count, topk_values
# Scalars: gpd_xi, gpd_sigma, gpd_u, gpd_N_u, gpd_N

# ==============================================================================
# GPD RECOVERY (Standalone Fitter for Sidecar Arm)
# ==============================================================================
SIDECAR_VERSION_SUPPORTED = 1

# Quantiles of the retained topk buffer at which to refit the GPD.
XI_SWEEP_QUANTILES = (0.0, 0.25, 0.50, 0.75, 0.90)


def load_null_sidecar(path):
    """Load a permute null sidecar. Raises ValueError with a specific reason."""
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        required = {'sidecar_version', 'topk_values', 'total_count',
                    'overflow_count', 'hist_counts', 'bin_edges'}
        missing = required - keys
        if missing:
            raise ValueError(
                "sidecar missing required keys: {0}".format(sorted(missing)))
        version = int(z['sidecar_version'])
        if version != SIDECAR_VERSION_SUPPORTED:
            raise ValueError(
                "sidecar version {0} unsupported (expected {1})".format(
                    version, SIDECAR_VERSION_SUPPORTED))
        out = {k: z[k] for k in z.files}
    return out


def xi_convergence_sweep(topk_values, total_count, quantiles=XI_SWEEP_QUANTILES):
    """Refit the GPD at a ladder of thresholds; report xi stability."""
    topk = np.asarray(topk_values, dtype=np.float64)
    rows = []
    if topk.size == 0:
        return rows
    for q in quantiles:
        u = float(np.quantile(topk, q))
        n_exc = int((topk > u).sum())
        xi, sigma = fit_gpd(topk, u)
        rows.append({
            'quantile': float(q),
            'u': u,
            'n_exceedances': n_exc,
            'xi': xi,
            'sigma': sigma,
            'tail_frac': (float(topk.size) / float(total_count)
                          if total_count else None),
        })
    return rows


def summarize_sidecar(sc):
    """Build the eval-report block from a loaded sidecar."""
    topk = np.asarray(sc['topk_values'], dtype=np.float64)
    total = int(sc['total_count'])
    overflow = int(sc['overflow_count'])
    obs_max = float(sc['observed_max_abs_t']) if 'observed_max_abs_t' in sc else None

    sweep = xi_convergence_sweep(topk, total)
    xis = [r['xi'] for r in sweep if r['xi'] is not None]
    xi_spread = (max(xis) - min(xis)) if len(xis) >= 2 else None

    null_max = float(topk.max()) if topk.size else None
    extrapolation_gap = ((obs_max - null_max)
                         if (obs_max is not None and null_max is not None)
                         else None)

    return {
        'status': 'loaded',
        'sidecar_version': int(sc['sidecar_version']),
        'perm_seed': int(sc['perm_seed']) if 'perm_seed' in sc else None,
        'perm_n_perm': int(sc['perm_n_perm']) if 'perm_n_perm' in sc else None,
        'total_null_draws': total,
        'overflow_count': overflow,
        'overflow_frac': (overflow / total) if total else None,
        'empirical_floor': 1.0 / (total + 1) if total else None,
        'null_max_abs_t': null_max,
        'observed_max_abs_t': obs_max,
        'extrapolation_gap': extrapolation_gap,
        'gpd_status': str(sc['gpd_status']) if 'gpd_status' in sc else 'unknown',
        'gpd_xi_run': float(sc['gpd_xi']) if 'gpd_xi' in sc else None,
        'gpd_u_run': float(sc['gpd_u']) if 'gpd_u' in sc else None,
        'xi_sweep': sweep,
        'xi_spread': xi_spread,
    }


def resolve_sidecar_arm(perm_null_sidecar):
    """Resolve the sidecar-gated eval arm. Never raises.

    The 'skipped_no_sidecar' status is preserved for both the not-provided and
    file-absent cases; the two are distinguished by the additive 'reason' key.
    """
    if not perm_null_sidecar:
        print("Warning: --perm-null-sidecar not provided. Skipping sidecar-gated arms.")
        return {'status': "skipped_no_sidecar", 'reason': "not_provided"}
    if not os.path.exists(perm_null_sidecar):
        warnings.warn(f"Sidecar file specified but not found: {perm_null_sidecar}")
        return {'status': "skipped_no_sidecar", 'reason': "file_absent"}
    try:
        return summarize_sidecar(load_null_sidecar(perm_null_sidecar))
    except Exception as exc:
        warnings.warn(f"Sidecar unreadable ({exc}); skipping sidecar-gated arms.")
        return {'status': "skipped_unreadable", 'reason': str(exc)}


def fit_gpd(data, u):
    """
    Fits Generalized Pareto Distribution to data > u using method of moments or MLE.
    This is a standalone stub/fitter used by the sidecar-gated arm.
    """
    exc = data[data > u] - u
    if exc.size < 50:
        return np.nan, np.nan

    xi, _, sigma = scipy.stats.genpareto.fit(exc, floc=0)
    return float(xi), float(sigma)

# ==============================================================================
# DIAGNOSTICS & ARMS
# ==============================================================================
def compute_analytic_p(t, df):
    return 2.0 * scipy.stats.t.sf(np.abs(np.asarray(t, dtype=np.float64)), df)

def compute_calibration_stats(t_vals, p_perm, p_ana, is_bulk, is_tail):
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

def _canon_chrom(arr, which):
    s = pd.Series(arr)
    isna = s.isna()                # BEFORE astype(str) — afterwards NaN is 'nan' and isna() is blind
    s = s.astype(str).str.strip()
    s = s.str.replace(r'\.0$', '', regex=True)              # 1.0 -> 1  (pandas NA inference -> float64)
    s = s.str.replace(r'^chr', '', regex=True, case=False)  # chr1 -> 1
    s = s.str.upper()                                       # x -> X
    s = s.mask(isna, other=pd.NA)                           # restore NaN; never the string 'nan'
    return s.to_numpy(dtype=object)

def label_strata(output, m_annot, g_annot):
    m_mapped = m_annot.index.astype(str).get_indexer(output['mt_id'].astype(str))
    g_mapped = g_annot.index.astype(str).get_indexer(output['gt_id'].astype(str))

    if (m_mapped == -1).any() or (g_mapped == -1).any():
        raise ValueError("Reported mt_id/gt_id missing from annotations.")

    m_chrom = _canon_chrom(m_annot.iloc[m_mapped]['chrom'].to_numpy(), 'm_annot')
    g_chrom = _canon_chrom(g_annot.iloc[g_mapped]['chrom'].to_numpy(), 'g_annot')

    keep = ~(pd.isna(m_chrom) | pd.isna(g_chrom))
    n_dropped = int((~keep).sum())

    if not keep.any():
        raise ValueError(
            "All reported pairs dropped: every pair has an unmappable chromosome "
            "on at least one axis. Check that --m-annot/--g-annot match the run."
        )

    is_cis = np.zeros(len(output), dtype=bool)
    is_cis[keep] = (m_chrom[keep] == g_chrom[keep])
    is_trans = keep & ~is_cis

    return keep, is_cis, is_trans, int(is_cis.sum()), int(is_trans.sum()), n_dropped

def calculate_genomic_inflation(t_vals):
    """Arm A.c: Genomic Inflation Lambda"""
    median_t_sq = np.median(t_vals**2)
    expected_median_chi2 = scipy.stats.chi2.ppf(0.5, 1)
    return float(median_t_sq / expected_median_chi2)

def _load_annotation(path, which):
    # tecpg convention: sniff the separator. BED6 is TAB; fallbacks may differ.
    annot = pd.read_csv(path, sep=None, engine='python')

    if 'name' not in annot.columns:
        raise ValueError(
            "{0}: annotation at {1} has no 'name' column; found columns {2}. "
            "Expected a BED6 with header (chrom chromStart chromEnd name score strand). "
            "A single mashed column indicates a separator mismatch."
            .format(which, path, list(annot.columns))
        )
    if 'chrom' not in annot.columns:
        raise ValueError(
            "{0}: annotation at {1} has no 'chrom' column; found columns {2}. "
            "Expected a BED6 with header (chrom chromStart chromEnd name score strand). "
            "A single mashed column indicates a separator mismatch."
            .format(which, path, list(annot.columns))
        )

    annot = annot.set_index('name')

    if not annot.index.is_unique:
        n_dup = int(annot.index.duplicated().sum())
        raise ValueError(
            "{0}: annotation index is not unique ({1} duplicated names); "
            "get_indexer requires a unique index.".format(which, n_dup)
        )
    return annot


def _resolve_permutation_parameters(args, df, meta):
    n_perm = None
    n_null_pairs = None
    perm_resolution_floor = None

    if b'tecpg_perm_n_perm' in meta:
        n_perm = int(meta[b'tecpg_perm_n_perm'])
    else:
        # Namespaced column first; the bare name is still read so artifacts
        # written before the rename keep resolving rather than silently
        # falling through to None.
        for _col in ('perm_n_perm', 'n_perm'):
            if _col in df.columns:
                unique_perms = df[_col].dropna().unique()
                if len(unique_perms) == 1:
                    n_perm = int(unique_perms[0])
                break

    if args.n_null_pairs is not None:
        n_null_pairs = args.n_null_pairs
        if b'tecpg_perm_n_null_pairs' in meta and int(meta[b'tecpg_perm_n_null_pairs']) != args.n_null_pairs:
            print(f"WARNING: --n-null-pairs CLI flag ({args.n_null_pairs}) overrides "
                  f"the parquet metadata value ({int(meta[b'tecpg_perm_n_null_pairs'])}).", file=sys.stderr)
    elif b'tecpg_perm_n_null_pairs' in meta:
        n_null_pairs = int(meta[b'tecpg_perm_n_null_pairs'])

    if n_perm is not None and n_perm > 0 and n_null_pairs is not None and n_null_pairs > 0:
        perm_resolution_floor = 1.0 / (n_perm * n_null_pairs)
        print(f"INFO: Permutation resolution floor resolved to {perm_resolution_floor:.3g} "
              f"({n_perm} perms * {n_null_pairs} null pairs).", file=sys.stderr)
    else:
        print("WARNING: Permutation resolution floor could not be resolved. "
              "If reading an older parquet, supply --n-null-pairs.", file=sys.stderr)

    return n_perm, n_null_pairs, perm_resolution_floor


def main():
    parser = argparse.ArgumentParser(description="Phase 3 standalone read-only permutation-evaluation diagnostic")
    parser.add_argument("--perm-output", required=True, help="Path to qr_permute output, .parquet or .csv")
    parser.add_argument("--m-annot", required=True, help="Path to methylation annotation (mt_id -> chrom,...)")
    parser.add_argument("--g-annot", required=True, help="Path to expression annotation (gt_id -> chrom,...)")
    parser.add_argument("--df", required=True, type=int, help="Run-level df = n_samples - n_covars - 2. MUST equal the df that produced mt_t.")
    parser.add_argument("--perm-null-sidecar", help="Path to null-accumulator sidecar .npz. When absent, sidecar-gated arms are skipped.")
    parser.add_argument('--n-null-pairs', type=int, default=None,
                        help="Number of null pairs used to build the permutation null. "
                             "Only needed for outputs written before this value was "
                             "recorded in the parquet metadata.")
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
    perm_file_meta = {}
    try:
        if args.perm_output.endswith(".parquet"):
            _table = pq.read_table(args.perm_output)
            perm_file_meta = _table.schema.metadata or {}
            output = _table.to_pandas()
            if output.index.names != [None]:
                output = output.reset_index()
        else:
            output = pd.read_csv(args.perm_output)

        m_annot = _load_annotation(args.m_annot, '--m-annot')
        g_annot = _load_annotation(args.g_annot, '--g-annot')

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

    n_pairs_input = len(output)

    has_region = 'region' in output.columns
    if has_region:
        region_col = output['region']
        keep = ~region_col.isna()
        n_dropped = int((~keep).sum())

        if n_dropped:
            print("Drop site eval_permute.region[reported_pairs]: dropped pairs with "
                  "null region: {0} -> {1} ({2} dropped)"
                  .format(n_pairs_input, int(keep.sum()), n_dropped), file=sys.stderr)

        output = output.loc[keep].reset_index(drop=True)

        region_col = output['region']
        invalid_regions = set(region_col.unique()) - set(CANONICAL_REGIONS)
        if invalid_regions:
            raise ValueError(f"eval_permute: unexpected region labels: {sorted(list(invalid_regions))}")

        masks_R = {R: (region_col == R).to_numpy() for R in CANONICAL_REGIONS}

        is_trans = masks_R['TRANS']
        # Pooled near-gene replaces is_cis for bulk logic
        is_cis = np.zeros(len(output), dtype=bool)
        for R in NEAR_GENE_REGIONS:
            is_cis |= masks_R[R]

        n_by_region = {R: int(masks_R[R].sum()) for R in CANONICAL_REGIONS}
        n_cis = int(is_cis.sum())
        n_trans = int(is_trans.sum())
        n_dropped_unmappable_chrom = 0
        n_dropped_null_region = n_dropped
    else:
        try:
            keep, is_cis, is_trans, n_cis, n_trans, n_dropped = label_strata(output, m_annot, g_annot)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if n_dropped:
            print("Drop site eval_permute.label_strata[reported_pairs]: dropped pairs with "
                  "unmappable chromosome: {0} -> {1} ({2} dropped)"
                  .format(n_pairs_input, int(keep.sum()), n_dropped), file=sys.stderr)
            output = output.loc[keep].reset_index(drop=True)
            is_cis = is_cis[keep]
            is_trans = is_trans[keep]

        n_dropped_unmappable_chrom = n_dropped
        n_dropped_null_region = 0

    # -------------------------------------------------------------------------
    # Core Data Extraction
    # -------------------------------------------------------------------------
    t = output['mt_t'].to_numpy(dtype=np.float64)
    p_perm = output['perm_mt_p'].to_numpy(dtype=np.float64)
    df_val = args.df

    # Precise analytic ref
    p_ana = compute_analytic_p(t, df_val)


    # Bands
    is_bulk = (p_ana >= args.bulk_lo) & (p_ana <= args.bulk_hi)
    is_tail = p_ana < TAIL_P_ANA

    n_perm, n_null_pairs, perm_resolution_floor = _resolve_permutation_parameters(args, output, perm_file_meta)

    report = {
        "metadata": {
            "n_pairs_input": n_pairs_input,
            "n_pairs_dropped_unmappable_chrom": n_dropped_unmappable_chrom,
            "n_pairs_scored": len(t),
            "n_cis": n_cis,
            "n_trans": n_trans,
            "df": df_val,
            "bulk_lo": args.bulk_lo,
            "bulk_hi": args.bulk_hi,
            "tail_p_ana": TAIL_P_ANA,
            "n_perm": n_perm,
            "n_null_pairs": n_null_pairs,
            "perm_resolution_floor": perm_resolution_floor
        },
        "arms": {}
    }

    if has_region:
        report["metadata"]["n_by_region"] = n_by_region
        report["metadata"]["n_pairs_dropped_null_region"] = n_dropped_null_region

    # -------------------------------------------------------------------------
    # Arm A.a: Calibration
    # -------------------------------------------------------------------------
    calib = {}

    # all
    calib['all'] = compute_calibration_stats(t, p_perm, p_ana, is_bulk, is_tail)
    if has_region:
        for R in CANONICAL_REGIONS:
            mask_R = masks_R[R]
            if np.any(mask_R):
                calib[R] = compute_calibration_stats(t[mask_R], p_perm[mask_R], p_ana[mask_R], is_bulk[mask_R], is_tail[mask_R])
        # Keep legacy 'cis' / 'trans' for summarize_permute
        calib['cis'] = compute_calibration_stats(t[is_cis], p_perm[is_cis], p_ana[is_cis], is_bulk[is_cis], is_tail[is_cis])
        calib['trans'] = compute_calibration_stats(t[is_trans], p_perm[is_trans], p_ana[is_trans], is_bulk[is_trans], is_tail[is_trans])
    else:
        # cis
        calib['cis'] = compute_calibration_stats(t[is_cis], p_perm[is_cis], p_ana[is_cis], is_bulk[is_cis], is_tail[is_cis])
        # trans
        calib['trans'] = compute_calibration_stats(t[is_trans], p_perm[is_trans], p_ana[is_trans], is_bulk[is_trans], is_tail[is_trans])

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

    def bulk_log_ratios(mask):
        mask_bulk = mask & is_bulk
        if not np.any(mask_bulk):
            return np.array([])
        p_perm_safe = np.clip(p_perm[mask_bulk], a_min=1e-300, a_max=1.0)
        p_ana_safe = np.clip(p_ana[mask_bulk], a_min=1e-300, a_max=1.0)
        return np.log10(p_perm_safe / p_ana_safe)

    if has_region:
        trans_bulk = bulk_log_ratios(masks_R[REGION_REFERENCE])
        lambda_excess = (lambda_cis - lambda_trans) if (lambda_cis is not None and lambda_trans is not None) else None

        if len(trans_bulk) < MIN_REGION_BULK_N:
            stratify['status'] = "skipped_insufficient_data"
        else:
            median_trans = float(np.median(trans_bulk))
            per_region = {}
            divergent = []

            per_region[REGION_REFERENCE] = {
                'status': 'reference',
                'n_bulk': len(trans_bulk),
                'median_log10_ratio': median_trans,
                'lambda': float(calculate_genomic_inflation(t[masks_R[REGION_REFERENCE]])) if np.any(masks_R[REGION_REFERENCE]) else None
            }

            for R in CANONICAL_REGIONS:
                if R == REGION_REFERENCE:
                    continue
                rb = bulk_log_ratios(masks_R[R])
                if len(rb) < MIN_REGION_BULK_N:
                    per_region[R] = {'status': 'insufficient_data', 'n_bulk': len(rb)}
                else:
                    median_R = float(np.median(rb))
                    delta = median_R - median_trans
                    mw_stat, mw_p = scipy.stats.mannwhitneyu(rb, trans_bulk, alternative='two-sided')
                    ks_stat2, ks_p2 = scipy.stats.kstest(rb, trans_bulk)
                    lambda_R = calculate_genomic_inflation(t[masks_R[R]])
                    per_region[R] = {
                        'status': 'ok',
                        'n_bulk': len(rb),
                        'median_log10_ratio': median_R,
                        'delta_vs_trans': delta,
                        'mw_p': float(mw_p),
                        'ks_p': float(ks_p2),
                        'lambda': float(lambda_R)
                    }
                    if R in NEAR_GENE_REGIONS and abs(delta) >= TOLERANCE_MEDIAN_LOG10_RATIO_DIFF:
                        divergent.append(R)

            stratify['mode'] = 'per_region'
            stratify['reference'] = REGION_REFERENCE
            stratify['per_region'] = per_region
            stratify['divergent_regions'] = divergent

            pooled_near_gene = bulk_log_ratios(is_cis)

            stratify['median_log10_ratio_trans'] = median_trans
            stratify['lambda_excess'] = float(lambda_excess) if lambda_excess is not None else None
            stratify['test_name'] = "mann_whitney_u"

            if len(pooled_near_gene) < MIN_REGION_BULK_N:
                stratify['median_log10_ratio_cis'] = None
                stratify['delta_median_log10_ratio'] = None
                stratify['test_stat'] = None
                stratify['test_p'] = None
                stratify['ks_stat'] = None
                stratify['ks_p'] = None
                stratify['recommendation'] = "insufficient_near_gene_coverage"
            else:
                median_cis = float(np.median(pooled_near_gene))
                delta = median_cis - median_trans
                mw_stat, mw_p = scipy.stats.mannwhitneyu(pooled_near_gene, trans_bulk, alternative='two-sided')
                ks_stat2, ks_p2 = scipy.stats.kstest(pooled_near_gene, trans_bulk)

                stratify['median_log10_ratio_cis'] = median_cis
                stratify['delta_median_log10_ratio'] = delta
                stratify['test_stat'] = float(mw_stat)
                stratify['test_p'] = float(mw_p)
                stratify['ks_stat'] = float(ks_stat2)
                stratify['ks_p'] = float(ks_p2)

                if not divergent:
                    stratify['recommendation'] = "single_global_null_adequate"
                else:
                    stratify['recommendation'] = "stratification_warranted"

    else:
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

            # lambda_excess is reported as a DESCRIPTIVE diagnostic only. Genomic
            # inflation (lambda_GC) presumes a mostly-null test space, which eQTM —
            # and cis in particular — violates: high lambda_cis is expected biology,
            # not miscalibration, so it must not gate the verdict. The stratify
            # decision keys on the calibration-divergence effect size (delta) alone.
            lambda_excess = (lambda_cis - lambda_trans) if (lambda_cis is not None and lambda_trans is not None) else None
            stratify['lambda_excess'] = float(lambda_excess) if lambda_excess is not None else None

            if abs(delta) < TOLERANCE_MEDIAN_LOG10_RATIO_DIFF:
                rec = "single_global_null_adequate"
            else:
                rec = "stratification_warranted"

            stratify['recommendation'] = rec
        else:
            stratify['status'] = "skipped_insufficient_data"

    report["arms"]["stratify_decision"] = stratify

    # -------------------------------------------------------------------------
    # Arm B: Sidecar-gated
    # -------------------------------------------------------------------------
    sidecar_arm = resolve_sidecar_arm(args.perm_null_sidecar)

    report["arms"]["sidecar"] = sidecar_arm

    # Write JSON report
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report written to {report_path}")

if __name__ == "__main__":
    main()
