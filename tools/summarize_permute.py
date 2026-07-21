#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import pyarrow.parquet as pq

# Add tools directory to path to allow importing eval_permute
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval_permute import CANONICAL_REGIONS, NEAR_GENE_REGIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def build_summary_text(report_dict: dict) -> str:
    lines = []
    lines.append("# Permutation vs Analytic p-value Summary\n")

    meta = report_dict.get("metadata", {})
    strat_decision = report_dict.get("arms", {}).get("stratify_decision", {})
    per_region = strat_decision.get("per_region")

    lines.append(f"**Pairs Input:** {meta.get('n_pairs_input', 'N/A')}")
    lines.append(f"**Pairs Dropped (Unmappable):** {meta.get('n_pairs_dropped_unmappable_chrom', 'N/A')}")

    if per_region is not None:
        n_by_reg = meta.get("n_by_region", {})
        n_near = sum(n_by_reg.get(r, 0) for r in NEAR_GENE_REGIONS)
        n_dist = sum(n_by_reg.get(r, 0) for r in ["DISTAL5", "DISTAL3"])
        n_trans = n_by_reg.get("TRANS", 0)
        lines.append(f"**Pairs Scored:** {meta.get('n_pairs_scored', 'N/A')} (near-gene: {n_near}, distal: {n_dist}, trans: {n_trans})")
    else:
        lines.append(f"**Pairs Scored:** {meta.get('n_pairs_scored', 'N/A')} (Cis: {meta.get('n_cis', 'N/A')}, Trans: {meta.get('n_trans', 'N/A')})")

    lines.append("\n## Calibration Agreement")
    calib = report_dict.get("arms", {}).get("calibration", {})

    for stratum in ["all", "cis", "trans"]:
        c = calib.get(stratum)
        if c:
            spearman = c.get("bulk_spearman_corr")
            median_diff = c.get("bulk_median_abs_log_ratio")
            if spearman is not None and median_diff is not None:
                lines.append(f"- **{stratum.capitalize()}:** in the null bulk, permutation and analytic p agree well (median |log10 ratio| = {median_diff:.4f}); Spearman = {spearman:.4f}")
            else:
                lines.append(f"- **{stratum.capitalize()}:** bulk not computed (master appears threshold-filtered)")
        else:
            lines.append(f"- **{stratum.capitalize()}:** bulk not computed (master appears threshold-filtered)")

    if per_region is not None:
        lines.append("\n## Regional Stratification")
        lines.append("| region | n | n_bulk | status | median|log10 ratio| | Δ vs TRANS | MW p | λ |")
        lines.append("|---|---|---|---|---|---|---|---|")

        n_by_reg = meta.get("n_by_region", {})
        for R in CANONICAL_REGIONS:
            r_data = per_region.get(R, {})
            n_pairs = n_by_reg.get(R, 0)
            status = r_data.get("status", "N/A")
            n_bulk = r_data.get("n_bulk", "N/A")

            if status in ("insufficient_data", "N/A"):
                lines.append(f"| {R} | {n_pairs} | {n_bulk} | {status} | — | — | — | — |")
            else:
                med = r_data.get("median_log10_ratio")
                delta = r_data.get("delta_vs_trans")
                mwp = r_data.get("mw_p")
                lam = r_data.get("lambda")

                med_str = f"{med:.4f}" if isinstance(med, float) else med
                delta_str = f"{delta:.4f}" if isinstance(delta, float) else delta
                mwp_str = f"{mwp:.2e}" if isinstance(mwp, float) else mwp
                lam_str = f"{lam:.4f}" if isinstance(lam, float) else lam

                lines.append(f"| {R} | {n_pairs} | {n_bulk} | {status} | {med_str} | {delta_str} | {mwp_str} | {lam_str} |")

        rec = strat_decision.get("recommendation", "N/A")
        div_regions = strat_decision.get("divergent_regions", [])
        div_str = ", ".join(div_regions) if div_regions else "none"
        lines.append(f"\n**Verdict:** `{rec}`")
        lines.append(f"Divergent regions: {div_str}")

        lines.append("\n## Caveat")
        if rec == "insufficient_near_gene_coverage":
            lines.append("> The near-gene regions fell below the coverage floor in the reservoir, so the cis-window verdict is deferred to a gene-anchored enrichment run. TRANS and both DISTAL strata serve as the powered, calibrated reference.")
        else:
            lines.append("> A `single_global_null_adequate` verdict is not a substitute for a gene-anchored cis-window run.")
    else:
        lines.append("\n## Null Sanity & Stratification")
        sanity = report_dict.get("arms", {}).get("null_sanity", {})
        lines.append(f"- **λ_trans:** {sanity.get('lambda_trans', 'N/A')}")
        lines.append(f"- **λ_cis:** {sanity.get('lambda_cis', 'N/A')}")

        strat = report_dict.get("arms", {}).get("stratify_decision", {})
        if strat.get("status") == "skipped_insufficient_data":
            lines.append("- **Verdict:** skipped due to insufficient data (master appears threshold-filtered)")
        else:
            rec = strat.get("recommendation", "N/A")
            delta = strat.get("delta_median_log10_ratio", "N/A")
            test_p = strat.get("test_p", "N/A")
            ks_p = strat.get("ks_p", "N/A")

            delta_str = f"{delta:.4f}" if isinstance(delta, float) else delta
            test_p_str = f"{test_p:.2e}" if isinstance(test_p, float) else test_p
            ks_p_str = f"{ks_p:.2e}" if isinstance(ks_p, float) else ks_p

            lines.append(f"- **Verdict:** `{rec}`")
            lines.append(f"  - Delta median log10 ratio: {delta_str}")
            lines.append(f"  - Mann-Whitney p: {test_p_str}")
            lines.append(f"  - KS p: {ks_p_str}")

        lines.append("\n## Caveat")
        lines.append("> cis here = same-chromosome (uniform sample), a weak proxy for the cis window; a `single_global_null_adequate` verdict is not a substitute for a gene-anchored cis-window run.")

    lines.append("\n## Figures")
    lines.append("- `qq_perm_vs_analytic.png`: QQ scatter of analytic vs permuted p-values.")
    lines.append("- `dist_overlap_p.png`: Overlapping distributions of analytic and permuted p-values.")
    lines.append("- `dist_tstat.png`: Distribution of the observed t-statistic.")

    return "\n".join(lines)

def _family_of(region):
    if region == 'TRANS':
        return 'trans'
    if region in NEAR_GENE_REGIONS:
        return 'near_gene'
    return 'distal'

def assign_family(df, m_annot_path=None, g_annot_path=None):
    if 'region' in df.columns:
        keep = ~df['region'].isna()
        n_drop = int((~keep).sum())
        if n_drop:
            logger.info("Drop site summarize.region: dropped null-region pairs: {0} -> {1} ({2} dropped)"
                        .format(len(df), int(keep.sum()), n_drop))
        df = df.loc[keep].reset_index(drop=True)
        invalid = set(df['region'].unique()) - set(CANONICAL_REGIONS)
        if invalid:
            raise ValueError("summarize_permute: unexpected region labels: {0}".format(sorted(invalid)))
        df['family'] = df['region'].map(_family_of)
        return df, 'region'

    if m_annot_path and g_annot_path:
        from eval_permute import label_strata as eval_label_strata, _load_annotation
        m_annot = _load_annotation(m_annot_path, 'M')
        g_annot = _load_annotation(g_annot_path, 'G')

        m_ok = m_annot.index.astype(str).get_indexer(df['mt_id'].astype(str)) != -1
        g_ok = g_annot.index.astype(str).get_indexer(df['gt_id'].astype(str)) != -1
        ann_keep = m_ok & g_ok
        n_unann = int((~ann_keep).sum())
        if n_unann:
            logger.info("Drop site summarize.annot: dropped pairs missing from annotations: {0} -> {1} ({2} dropped)"
                        .format(len(df), int(ann_keep.sum()), n_unann))
        df = df.loc[ann_keep].reset_index(drop=True)

        _keep, is_cis, is_trans, _nc, _nt, _nd = eval_label_strata(df, m_annot, g_annot)
        df['family'] = np.where(is_cis, 'near_gene', np.where(is_trans, 'trans', 'dropped'))

        # Also drop unmappable-chrom dropped by label_strata's own keep mask
        valid = (df['family'] != 'dropped') & _keep
        df = df.loc[valid].reset_index(drop=True)
        return df, 'family2'

    df['family'] = 'all'
    return df, 'all'

def downsample_preserving_tail(df, n_points=15000, seed=42):
    if len(df) <= n_points:
        return df

    np.random.seed(seed)

    # Define tail points: top 1% by -log10 p_ana
    tail_n = max(int(n_points * 0.1), 1)
    bulk_n = n_points - tail_n

    df_sorted = df.sort_values(by='p_ana', ascending=True)

    tail_actual = min(tail_n, len(df_sorted))
    df_tail = df_sorted.head(tail_actual)

    df_bulk_pool = df_sorted.iloc[tail_actual:]
    if len(df_bulk_pool) > bulk_n:
        df_bulk = df_bulk_pool.sample(n=bulk_n, random_state=seed)
    else:
        df_bulk = df_bulk_pool

    return pd.concat([df_tail, df_bulk])

def main():
    parser = argparse.ArgumentParser(description="Summarize Permutation vs Analytic P-values")
    parser.add_argument("--perm-output", required=True, help="Path to permutation_results.parquet")
    parser.add_argument("--report", required=True, help="Path to eval_permute_report.json")
    parser.add_argument("--df", type=int, required=True, help="Degrees of freedom")
    parser.add_argument("--m-annot", help="Optional path to M.bed6")
    parser.add_argument("--g-annot", help="Optional path to G.bed6")
    parser.add_argument("--out-dir", required=True, help="Output directory")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.report, 'r') as f:
        report = json.load(f)

    summary_text = build_summary_text(report)
    with open(os.path.join(args.out_dir, "permute_vs_analytic_summary.md"), "w") as f:
        f.write(summary_text)

    logger.info("Reading permutation results...")
    df = pq.read_table(args.perm_output).to_pandas()

    df['p_ana'] = 2.0 * scipy.stats.t.sf(np.abs(df['mt_t'].astype(np.float64)), args.df)

    df, strat_mode = assign_family(df, args.m_annot, args.g_annot)

    df_plot = downsample_preserving_tail(df)

    df_plot['neg_log10_p_ana'] = -np.log10(df_plot['p_ana'].clip(lower=1e-300))
    df_plot['neg_log10_p_perm'] = -np.log10(df_plot['perm_mt_p'].clip(lower=1e-300))

    logger.info("Plotting qq_perm_vs_analytic.png...")
    plt.figure(figsize=(8, 8))

    if strat_mode in ('region', 'family2'):
        trans_mask = df_plot['family'] == 'trans'
        distal_mask = df_plot['family'] == 'distal'
        near_gene_mask = df_plot['family'] == 'near_gene'

        plt.scatter(df_plot.loc[trans_mask, 'neg_log10_p_ana'], df_plot.loc[trans_mask, 'neg_log10_p_perm'],
                    alpha=0.5, label='Trans', color='gray', s=10, rasterized=True)
        plt.scatter(df_plot.loc[distal_mask, 'neg_log10_p_ana'], df_plot.loc[distal_mask, 'neg_log10_p_perm'],
                    alpha=0.5, label='Distal', color='green', s=10, rasterized=True)
        plt.scatter(df_plot.loc[near_gene_mask, 'neg_log10_p_ana'], df_plot.loc[near_gene_mask, 'neg_log10_p_perm'],
                    alpha=0.5, label='Near Gene', color='orange', s=10, rasterized=True)
        plt.legend()
    else:
        plt.scatter(df_plot['neg_log10_p_ana'], df_plot['neg_log10_p_perm'],
                    alpha=0.5, color='gray', s=10, rasterized=True)

    max_val = max(df_plot['neg_log10_p_ana'].max(), df_plot['neg_log10_p_perm'].max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.8)

    plt.xlabel('-log10(Analytic p-value)')
    plt.ylabel('-log10(Permutation p-value)')
    plt.title('QQ Plot: Permutation vs Analytic')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "qq_perm_vs_analytic.png"), dpi=300)
    plt.close()

    logger.info("Plotting dist_overlap_p.png...")
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 100)
    plt.hist(df['p_ana'], bins=bins, alpha=0.5, label='Analytic', density=True, color='blue')
    plt.hist(df['perm_mt_p'].dropna(), bins=bins, alpha=0.5, label='Permutation', density=True, color='orange')
    plt.xlabel('p-value')
    plt.ylabel('Density')
    plt.title('P-value Distribution Overlap')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "dist_overlap_p.png"), dpi=300)
    plt.close()

    logger.info("Plotting dist_tstat.png...")
    plt.figure(figsize=(10, 6))
    plt.hist(df['mt_t'], bins=100, alpha=0.7, color='purple')
    plt.xlabel('t-statistic')
    plt.ylabel('Count')
    plt.title('Distribution of Observed t-statistic')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "dist_tstat.png"), dpi=300)
    plt.close()

    logger.info("Summary complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
