#!/usr/bin/env python3
import argparse
import os
import sys
import logging
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import upsetplot
from scipy import stats


def resolve_kennedy_columns(columns):
    required = {'cpg': 'CpG.probe', 'probe': 'exp.Probe', 'pval': 'p.val'}
    missing = [name for name in required.values() if name not in columns]
    if missing:
        raise ValueError(f"Missing required Kennedy columns: {missing}. Observed columns: {list(columns)}")

    resolved = {
        'cpg': 'CpG.probe',
        'probe': 'exp.Probe',
        'pval': 'p.val',
        'status': 'status' if 'status' in columns else None,
        'distance': 'distance' if 'distance' in columns else None,
        'in_dist': 'in_dist' if 'in_dist' in columns else None,
        'tstat': 'T.stat' if 'T.stat' in columns else None,
        'beta': 'beta' if 'beta' in columns else None,
        'beta_sd': 'beta.sd' if 'beta.sd' in columns else None,
        'gene': 'annot.gene' if 'annot.gene' in columns else None,
        'probe_chrom': 'exp.probe.chrm' if 'exp.probe.chrm' in columns else None,
        'probe_start': 'exp.probe.start' if 'exp.probe.start' in columns else None,
        'probe_stop': 'exp.probe.stop' if 'exp.probe.stop' in columns else None,
        'probe_strand': 'exp.probe.strand' if 'exp.probe.strand' in columns else None,
        'other_gene': 'other_gene' if 'other_gene' in columns else None,
        'other_gene_distance': 'distance.other.gene' if 'distance.other.gene' in columns else None
    }
    return resolved


def resolve_tecpg_pvalue_column(columns):
    if 'precise_mt_p' not in columns:
        raise ValueError(f"Missing precise_mt_p in tecpg catalog. Observed columns: {list(columns)}")
    return 'precise_mt_p'



def load_kennedy(path, sep):
    df = pd.read_csv(path, sep=sep)
    # We must dropna on key columns, but we don't know the resolved column names here yet!
    # Wait, the old code didn't dropna on load_kennedy. It did it during merge or set operations.
    return df


def load_catalog(path, columns_to_read=None):
    schema = pq.ParquetFile(path).schema_arrow.names
    desired_cols = [
        'mt_id', 'gt_id', 'precise_mt_p', 'mt_est', 'mt_t',
        'region', 'fdr_est', 'mt_chrom', 'gt_chrom'
    ]
    if columns_to_read:
        desired_cols.append(columns_to_read)

    cols_to_load = [c for c in set(desired_cols) if c in schema]
    df = pq.read_table(path, columns=cols_to_load).to_pandas()
    if df.index.names != [None]:
        df = df.reset_index()
    return df


def compute_eligibility(catalog_df, kennedy_df, cols):
    catalog_cpgs = set(catalog_df['mt_id'].dropna())
    catalog_probes = set(catalog_df['gt_id'].dropna())

    df = kennedy_df.copy()
    df['eligible_cpg'] = df[cols['cpg']].isin(catalog_cpgs)
    df['eligible_probe'] = df[cols['probe']].isin(catalog_probes)
    df['eligible'] = df['eligible_cpg'] & df['eligible_probe']

    return df


def compute_overlap_rates(catalog_df, kennedy_df, cols, kennedy_thresh, tecpg_thresh, tecpg_p_col):
    K_tk_mask = kennedy_df[cols['pval']] < kennedy_thresh
    K_tk_df = kennedy_df[K_tk_mask]
    K_tk = set(zip(K_tk_df[cols['cpg']], K_tk_df[cols['probe']]))

    K_tk_E_df = kennedy_df[K_tk_mask & kennedy_df['eligible']]
    K_tk_E = set(zip(K_tk_E_df[cols['cpg']], K_tk_E_df[cols['probe']]))

    T_tt_df = catalog_df[catalog_df[tecpg_p_col] < tecpg_thresh]
    T_tt = set(zip(T_tt_df['mt_id'], T_tt_df['gt_id']))

    recovery_num = len(K_tk_E.intersection(T_tt))
    recovery_denom = len(K_tk_E)
    recovery = recovery_num / recovery_denom if recovery_denom > 0 else 0.0

    confirmation_num = len(T_tt.intersection(K_tk))
    confirmation_denom = len(T_tt)
    confirmation_raw = confirmation_num / confirmation_denom if confirmation_denom > 0 else 0.0

    kennedy_cpgs = set(kennedy_df[cols['cpg']].dropna())
    kennedy_probes = set(kennedy_df[cols['probe']].dropna())

    T_tt_kennedy_testable = {(c, p) for c, p in T_tt if c in kennedy_cpgs and p in kennedy_probes}
    confirmation_testable_num = len(T_tt_kennedy_testable.intersection(K_tk))
    confirmation_testable_denom = len(T_tt_kennedy_testable)
    testable_val = confirmation_testable_num / confirmation_testable_denom if confirmation_testable_denom > 0 else 0.0
    confirmation_kennedy_testable = testable_val

    union = K_tk.union(T_tt)
    jaccard = len(T_tt.intersection(K_tk)) / len(union) if len(union) > 0 else 0.0

    return {
        'recovery': recovery,
        'confirmation_raw': confirmation_raw,
        'confirmation_kennedy_testable': confirmation_kennedy_testable,
        'jaccard': jaccard,
        'T_tt': T_tt,
        'K_tk': K_tk,
        'K_tk_E': K_tk_E,
        'K_tk_E_df': K_tk_E_df,
        'T_tt_df': T_tt_df,
        'kennedy_cpgs': kennedy_cpgs,
        'kennedy_probes': kennedy_probes
    }


def export_pair_lists(outdir, catalog_df, kennedy_df, cols, diag_results, tecpg_p_col):
    T_tt = diag_results['T_tt']
    K_tk = diag_results['K_tk']
    kennedy_cpgs = diag_results['kennedy_cpgs']
    kennedy_probes = diag_results['kennedy_probes']

    overlap = T_tt.intersection(K_tk)
    tecpg_only = T_tt - K_tk
    kennedy_only = K_tk - T_tt

    conc_mt, conc_gt = zip(*overlap) if overlap else ([], [])
    df_conc = pd.DataFrame({'mt_id': conc_mt, 'gt_id': conc_gt})

    t_only_mt, t_only_gt = zip(*tecpg_only) if tecpg_only else ([], [])
    df_t_only = pd.DataFrame({'mt_id': t_only_mt, 'gt_id': t_only_gt})

    k_only_mt, k_only_gt = zip(*kennedy_only) if kennedy_only else ([], [])
    df_k_only = pd.DataFrame({'mt_id': k_only_mt, 'gt_id': k_only_gt})

    def annotate_reasons(df, reason='concordant'):
        if len(df) == 0:
            df['non_overlap_reason'] = []
            df['eligible_cpg'] = []
            df['eligible_probe'] = []
            return df

        cpg_in_k = df['mt_id'].isin(kennedy_cpgs)
        probe_in_k = df['gt_id'].isin(kennedy_probes)
        df['eligible_cpg'] = cpg_in_k
        df['eligible_probe'] = probe_in_k

        if reason == 'tecpg_only':
            reasons = []
            for c, p in zip(cpg_in_k, probe_in_k):
                if c and p:
                    reasons.append('kennedy_tested_and_missed')
                else:
                    reasons.append('kennedy_universe_unknown')
            df['non_overlap_reason'] = reasons
        else:
            df['non_overlap_reason'] = reason

        return df

    df_conc = annotate_reasons(df_conc, 'concordant')
    df_t_only = annotate_reasons(df_t_only, 'tecpg_only')

    if len(df_k_only) > 0:
        k_merge = pd.merge(
            df_k_only, kennedy_df, left_on=['mt_id', 'gt_id'],
            right_on=[cols['cpg'], cols['probe']], how='left'
        )
        catalog_pairs = set(zip(catalog_df['mt_id'], catalog_df['gt_id']))
        in_catalog = [p in catalog_pairs for p in zip(k_merge['mt_id'], k_merge['gt_id'])]

        reasons = []
        for i, row in k_merge.iterrows():
            if in_catalog[i]:
                reasons.append('tested_and_missed')
            elif not row['eligible_cpg']:
                reasons.append('ineligible_cpg')
            elif not row['eligible_probe']:
                reasons.append('ineligible_probe')
            else:
                reasons.append('tested_and_missed')
        k_merge['non_overlap_reason'] = reasons
        df_k_only = k_merge[['mt_id', 'gt_id', 'non_overlap_reason', 'eligible_cpg', 'eligible_probe']]
    else:
        df_k_only['non_overlap_reason'] = []
        df_k_only['eligible_cpg'] = []
        df_k_only['eligible_probe'] = []

    tecpg_cols = ['mt_id', 'gt_id', tecpg_p_col, 'mt_est', 'mt_t', 'region', 'fdr_est']
    tecpg_cols_to_merge = [c for c in tecpg_cols if c in catalog_df.columns]

    k_cols_base = [
        cols['pval'], cols['tstat'], cols['beta'],
        cols['beta_sd'], cols['status'], cols['distance']
    ]
    k_cols_valid = [c for c in k_cols_base if c and c in kennedy_df.columns]

    def merge_all(df):
        if len(df) == 0:
            for c in tecpg_cols_to_merge[2:] + k_cols_valid:
                df[c] = []
            return df
        df = pd.merge(df, catalog_df[tecpg_cols_to_merge], on=['mt_id', 'gt_id'], how='left')
        df = pd.merge(
            df, kennedy_df[[cols['cpg'], cols['probe']] + k_cols_valid],
            left_on=['mt_id', 'gt_id'], right_on=[cols['cpg'], cols['probe']], how='left'
        )
        if cols['cpg'] in df.columns and cols['cpg'] != 'mt_id':
            df = df.drop(columns=[cols['cpg']])
        if cols['probe'] in df.columns and cols['probe'] != 'gt_id':
            df = df.drop(columns=[cols['probe']])
        return df

    df_conc = merge_all(df_conc)
    df_t_only = merge_all(df_t_only)
    df_k_only = merge_all(df_k_only)

    df_conc.to_csv(os.path.join(outdir, 'pairs_concordant.tsv'), sep='\t', index=False)
    df_t_only.to_csv(os.path.join(outdir, 'pairs_tecpg_only.tsv'), sep='\t', index=False)
    df_k_only.to_csv(os.path.join(outdir, 'pairs_kennedy_only.tsv'), sep='\t', index=False)


def build_summary_text(
    args, num_merged, pearson_r_beta, spearman_r_beta, r2_beta, beta_col,
    pearson_r_t, spearman_r_t, r2_t, tstat_col,
    grid_results, diag_results, thresholds, cols, kennedy_df
):

    summary = f"""Benchmark Summary
=================
Input tecpg file: {args.tecpg}
Input Kennedy file: {args.kennedy}
Kennedy p-value threshold: {args.kennedy_thresh}
tecpg p-value threshold: {args.tecpg_thresh}

Mapping & Merging
-----------------
Overlapping Pairs Mapped: {num_merged}
"""
    if 'status' in cols and cols['status']:
        statuses = kennedy_df[cols['status']].unique()
        summary += "\nEligibility by Status:\n"
        for s in statuses:
            sub = kennedy_df[kennedy_df[cols['status']] == s]
            el = sub['eligible'].sum()
            summary += f"  {s}: {el} / {len(sub)} eligible\n"
    else:
        summary += f"\nOverall Eligibility: {kennedy_df['eligible'].sum()} / {len(kennedy_df)}\n"

    summary += f"""
Comparison A: Statistical Concordance
-------------------------------------
Effect Size (mt_est vs {beta_col}):
  Pearson r:  {pearson_r_beta:.4f}
  Spearman rho: {spearman_r_beta:.4f}
  R^2:        {r2_beta:.4f}

Test Statistic (mt_t vs {tstat_col}):
  Pearson r:  {pearson_r_t:.4f}
  Spearman rho: {spearman_r_t:.4f}
  R^2:        {r2_t:.4f}

Comparison B: Hit Overlap (Directional Rates)
---------------------------------------------
Note: Confirmation denominators are LOWER BOUNDS (except kennedy_testable).
The Kennedy file is not a full universe, so we cannot know how many of our
hits they actually tested and found insignificant vs never tested.

--- Recovery Matrix (tecpg_thresh \\ kennedy_thresh) ---
"""
    header = "          " + "".join([f"{t:10.1e}" for t in thresholds])
    summary += header + "\n"
    for tt in thresholds:
        row_str = f"{tt:10.1e}"
        for tk in thresholds:
            res = grid_results[(tk, tt)]
            row_str += f"{res['recovery']:10.3f}"
        summary += row_str + "\n"

    summary += "\n--- Confirmation (Raw Lower Bound) Matrix ---\n"
    summary += header + "\n"
    for tt in thresholds:
        row_str = f"{tt:10.1e}"
        for tk in thresholds:
            res = grid_results[(tk, tt)]
            row_str += f"{res['confirmation_raw']:10.3f}"
        summary += row_str + "\n"

    summary += f"""
--- Diagonal (Like-for-like at {args.tecpg_thresh}) ---
  Recovery:                      {diag_results['recovery']:.4f}
  Confirmation (Raw Lower Bound): {diag_results['confirmation_raw']:.4f}
  Confirmation (Kennedy Testable):{diag_results['confirmation_kennedy_testable']:.4f}
  Jaccard Index:                 {diag_results['jaccard']:.4f}

Outputs
-------
Summary saved to: benchmark_summary.txt
Plots saved to: concordance_scatter.png, overlap_venn_diagonal.png
Pair lists saved to: pairs_*.tsv
"""
    if args.upset:
        summary += "UpSet plot saved to: overlap_upset_diagonal.png\n"

    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark tecpg output against Kennedy eQTL summary statistics.")
    parser.add_argument('tecpg', help="Path to tecpg output parquet file")
    parser.add_argument('kennedy', help="Path to Kennedy supplementary CSV/TSV file")
    parser.add_argument('--outdir', default='.', help="Directory to save output plots and summary")

    parser.add_argument('--tecpg-thresh', type=float, default=1e-5, help="p-value threshold for tecpg hits")
    parser.add_argument('--kennedy-thresh', type=float, default=1e-5, help="p-value threshold for Kennedy hits")
    parser.add_argument('--p-thresh', type=float, default=None, help="Deprecated. Use --kennedy-thresh")

    parser.add_argument('--kennedy-sep', default='\t', help="Separator for Kennedy file")
    parser.add_argument('--tecpg-p-col', default=None, help="Override for tecpg precise_mt_p column")
    parser.add_argument('--upset', action='store_true', help="Generate UpSet plot")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.p_thresh is not None:
        if '--p-thresh' in sys.argv and '--kennedy-thresh' in sys.argv:
            raise ValueError("Cannot provide both --p-thresh and --kennedy-thresh.")
        logging.warning("DeprecationWarning: --p-thresh is deprecated. Please use --kennedy-thresh instead.")
        args.kennedy_thresh = float(args.p_thresh)

    args.kennedy_thresh = float(args.kennedy_thresh)
    args.tecpg_thresh = float(args.tecpg_thresh)

    os.makedirs(args.outdir, exist_ok=True)

    logging.info(f"Loading Kennedy file: {args.kennedy}")
    df_kennedy = load_kennedy(args.kennedy, args.kennedy_sep)
    cols = resolve_kennedy_columns(df_kennedy.columns)

    cpg_col = cols['cpg']
    query_col = cols['probe']

    df_kennedy = df_kennedy.dropna(subset=[cpg_col, query_col])

    if cols['gene']:
        logging.info(f"Using column '{cols['gene']}' for Gene cardinality diagnostic.")
        genes = len(df_kennedy[cols['gene']].dropna().unique())
        logging.info(f"Kennedy unique genes: {genes}")

    logging.info(f"Loading tecpg catalog: {args.tecpg}")
    schema = pq.ParquetFile(args.tecpg).schema_arrow.names
    tecpg_p_col = args.tecpg_p_col if args.tecpg_p_col else resolve_tecpg_pvalue_column(schema)

    df_tecpg = load_catalog(args.tecpg, tecpg_p_col)

    if 'mt_chrom' in df_tecpg.columns and 'gt_chrom' in df_tecpg.columns:
        mt_chroms = df_tecpg['mt_chrom'].dropna().unique()
        gt_chroms = df_tecpg['gt_chrom'].dropna().unique()
        expected_combos = len(mt_chroms) * len(gt_chroms)
        actual_combos = len(df_tecpg[['mt_chrom', 'gt_chrom']].drop_duplicates())
        if actual_combos < expected_combos:
            logging.warning(
                f"STRUCTURAL WARNING: Catalog does not span all chromosome pairings "
                f"({actual_combos} observed vs {expected_combos} expected). "
                f"Eligibility may not factorize; recovery is an upper bound."
            )

    df_kennedy = compute_eligibility(df_tecpg, df_kennedy, cols)

    if 'mt_id' in df_tecpg.columns and cpg_col in df_kennedy.columns:
        loci_overlap = len(set(df_tecpg['mt_id'].dropna()).intersection(set(df_kennedy[cpg_col].dropna())))
        logging.info(f"Overlapping distinct CpG loci: {loci_overlap}")

    if 'gt_id' in df_tecpg.columns and query_col in df_kennedy.columns:
        genes_overlap = len(set(df_tecpg['gt_id'].dropna()).intersection(set(df_kennedy[query_col].dropna())))
        logging.info(f"Overlapping distinct genes: {genes_overlap}")

    logging.info("Merging datasets (inner join)...")
    df_merged = pd.merge(
        df_tecpg,
        df_kennedy,
        left_on=['mt_id', 'gt_id'],
        right_on=[cpg_col, query_col],
        how='inner',
        suffixes=('_tecpg', '_kennedy')
    )

    num_merged = len(df_merged)
    logging.info(f"Successfully mapped and merged {num_merged} pairs.")

    if num_merged == 0:
        logging.error("No overlapping pairs found between datasets. Please check the ID mapping.")
        sys.exit(1)

    beta_col = cols['beta']
    tstat_col = cols['tstat']

    if beta_col is None or tstat_col is None:
        logging.warning("Could not automatically identify beta or T.stat columns in Kennedy data. Will try to infer.")

    pearson_r_beta = spearman_r_beta = r2_beta = 0.0
    pearson_r_t = spearman_r_t = r2_t = 0.0

    if beta_col and 'mt_est' in df_merged.columns:
        valid_beta = df_merged[['mt_est', beta_col]].dropna()
        dropped = len(df_merged) - len(valid_beta)
        logging.info(
            f"Drop site benchmark_kennedy.valid_beta[mt_est,{beta_col}]: "
            f"dropped merged pairs with missing effect-size values: {len(df_merged)} -> "
            f"{len(valid_beta)} ({dropped} dropped)"
        )
        if len(valid_beta) > 1:
            pearson_r_beta, _ = stats.pearsonr(valid_beta['mt_est'], valid_beta[beta_col])
            spearman_r_beta, _ = stats.spearmanr(valid_beta['mt_est'], valid_beta[beta_col])
            r2_beta = pearson_r_beta**2

            logging.info("Generating scatter plots...")
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            ax1 = axes[0]
            ax1.hexbin(valid_beta['mt_est'], valid_beta[beta_col], gridsize=50, cmap='Blues', mincnt=1)
            min_val_b = min(valid_beta['mt_est'].min(), valid_beta[beta_col].min())
            max_val_b = max(valid_beta['mt_est'].max(), valid_beta[beta_col].max())
            ax1.plot([min_val_b, max_val_b], [min_val_b, max_val_b], 'r--', lw=2, label='y=x')
            ax1.set_xlabel('tecpg Effect Size (mt_est)')
            ax1.set_ylabel(f'Kennedy Effect Size ({beta_col})')
            ax1.set_title('Effect Size Concordance')
            text_b = f"Pearson r: {pearson_r_beta:.3f}\nSpearman $\\rho$: {spearman_r_beta:.3f}\n$R^2$: {r2_beta:.3f}"
            ax1.text(0.05, 0.95, text_b, transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if tstat_col and 'mt_t' in df_merged.columns:
        valid_t = df_merged[['mt_t', tstat_col]].dropna()
        dropped = len(df_merged) - len(valid_t)
        logging.info(
            f"Drop site benchmark_kennedy.valid_t[mt_t,{tstat_col}]: "
            f"dropped merged pairs with missing test-statistic values: {len(df_merged)} -> "
            f"{len(valid_t)} ({dropped} dropped)"
        )
        if len(valid_t) > 1:
            pearson_r_t, _ = stats.pearsonr(valid_t['mt_t'], valid_t[tstat_col])
            spearman_r_t, _ = stats.spearmanr(valid_t['mt_t'], valid_t[tstat_col])
            r2_t = pearson_r_t**2

            if 'ax1' in locals():
                ax2 = axes[1]
                ax2.hexbin(valid_t['mt_t'], valid_t[tstat_col], gridsize=50, cmap='Blues', mincnt=1)
                min_val_t = min(valid_t['mt_t'].min(), valid_t[tstat_col].min())
                max_val_t = max(valid_t['mt_t'].max(), valid_t[tstat_col].max())
                ax2.plot([min_val_t, max_val_t], [min_val_t, max_val_t], 'r--', lw=2, label='y=x')
                ax2.set_xlabel('tecpg Test Statistic (mt_t)')
                ax2.set_ylabel(f'Kennedy Test Statistic ({tstat_col})')
                ax2.set_title('Test Statistic Concordance')
                text_t = f"Pearson r: {pearson_r_t:.3f}\nSpearman $\\rho$: {spearman_r_t:.3f}\n$R^2$: {r2_t:.3f}"
                ax2.text(0.05, 0.95, text_t, transform=ax2.transAxes, fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                plt.tight_layout()
                plt.savefig(os.path.join(args.outdir, 'concordance_scatter.png'), dpi=300)
                plt.close()

    logging.info("Calculating directional overlap rates (sweep)...")
    thresholds = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]

    grid_results = {}
    for tt in thresholds:
        for tk in thresholds:
            grid_results[(tk, tt)] = compute_overlap_rates(df_tecpg, df_kennedy, cols, tk, tt, tecpg_p_col)

    logging.info("Calculating directional overlap rates (diagonal)...")
    diag_results = compute_overlap_rates(
        df_tecpg, df_kennedy, cols, args.kennedy_thresh, args.tecpg_thresh, tecpg_p_col
    )

    logging.info("Exporting pair lists...")
    export_pair_lists(args.outdir, df_tecpg, df_kennedy, cols, diag_results, tecpg_p_col)

    T_tt = diag_results['T_tt']
    K_tk_E = diag_results['K_tk_E']

    ineligible_count = len(diag_results['K_tk']) - len(K_tk_E)

    plt.figure(figsize=(8, 6))
    v = venn2([T_tt, K_tk_E], set_labels=('tecpg hits', 'Kennedy hits (Eligible)'))
    plt.title(f'Diagonal Overlap\n({ineligible_count} ineligible Kennedy hits omitted)')
    plt.savefig(os.path.join(args.outdir, 'overlap_venn_diagonal.png'), dpi=300)
    plt.close()

    if args.upset and (len(T_tt) > 0 or len(K_tk_E) > 0):
        upset_data = upsetplot.from_contents({
            'tecpg': T_tt,
            'Kennedy (Eligible)': K_tk_E
        })
        plt.figure(figsize=(8, 6))
        upsetplot.plot(upset_data)
        plt.title('UpSet Plot (Eligible)')
        plt.savefig(os.path.join(args.outdir, 'overlap_upset_diagonal.png'), dpi=300)
        plt.close()

    logging.info("Building summary...")
    summary = build_summary_text(
        args, num_merged, pearson_r_beta, spearman_r_beta, r2_beta, beta_col,
        pearson_r_t, spearman_r_t, r2_t, tstat_col,
        grid_results, diag_results, thresholds, cols, df_kennedy
    )

    print(summary)
    with open(os.path.join(args.outdir, 'benchmark_summary.txt'), 'w') as f:
        f.write(summary)


if __name__ == '__main__':
    main()
