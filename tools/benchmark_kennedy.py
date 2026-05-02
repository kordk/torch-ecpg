import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pyarrow.parquet as pq
from matplotlib_venn import venn2
import upsetplot
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Benchmark tecpg output against Kennedy 2018 baseline.")
    parser.add_argument('-t', '--tecpg', required=True, help="Path to tecpg Parquet output file")
    parser.add_argument('-k', '--kennedy', required=True, help="Path to Kennedy 2018 txt file")
    parser.add_argument('--p-thresh', type=float, default=1e-6, help="P-value threshold for Kennedy hits (default: 1e-6)")
    parser.add_argument('-o', '--outdir', default='.', help="Directory to save output files (default: current directory)")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    logging.info(f"Loading tecpg data from {args.tecpg}...")
    try:
        df_tecpg = pq.read_table(args.tecpg).to_pandas()
        logging.info(f"tecpg data loaded. Shape: {df_tecpg.shape}")
        logging.info(f"tecpg columns: {list(df_tecpg.columns)}")
        logging.info(f"tecpg distinct genes (gt_id): {df_tecpg.get('gt_id', pd.Series()).nunique()}")
        logging.info(f"tecpg distinct CpG loci (mt_id): {df_tecpg.get('mt_id', pd.Series()).nunique()}")
    except Exception as e:
        logging.error(f"Failed to read tecpg file: {e}")
        sys.exit(1)

    logging.info(f"Loading Kennedy data from {args.kennedy}...")
    try:
        # Kennedy format is likely tab or space separated
        df_kennedy = pd.read_csv(args.kennedy, sep=None, engine='python')
        logging.info(f"Kennedy data loaded. Shape: {df_kennedy.shape}")
        logging.info(f"Kennedy columns: {list(df_kennedy.columns)}")
        # Check potential cpg and gene columns
        cpg_col_initial = 'CpG.probe' if 'CpG.probe' in df_kennedy.columns else df_kennedy.columns[0]
        gene_col_initial = 'annot.gene' if 'annot.gene' in df_kennedy.columns else ('exp.Probe' if 'exp.Probe' in df_kennedy.columns else df_kennedy.columns[1])
        logging.info(f"Kennedy distinct genes ({gene_col_initial}): {df_kennedy.get(gene_col_initial, pd.Series()).nunique()}")
        logging.info(f"Kennedy distinct CpG loci ({cpg_col_initial}): {df_kennedy.get(cpg_col_initial, pd.Series()).nunique()}")
    except Exception as e:
        logging.error(f"Failed to read Kennedy file: {e}")
        sys.exit(1)

    logging.info("Preprocessing and mapping IDs...")

    # Standardize column names for merge
    # Kennedy CpG usually 'CpG.probe', tecpg 'mt_id'
    cpg_col = 'CpG.probe' if 'CpG.probe' in df_kennedy.columns else df_kennedy.columns[0]
    query_col = 'exp.Probe' if 'exp.Probe' in df_kennedy.columns else ('annot.gene' if 'annot.gene' in df_kennedy.columns else df_kennedy.columns[1])

    df_kennedy = df_kennedy.dropna(subset=[query_col, cpg_col])

    # Log a sample of the key columns before merging
    if 'mt_id' in df_tecpg.columns and 'gt_id' in df_tecpg.columns:
        logging.info(f"Sample of tecpg key columns (first 5 rows):\n{df_tecpg[['mt_id', 'gt_id']].head().to_string()}")

    if cpg_col in df_kennedy.columns and query_col in df_kennedy.columns:
        logging.info(f"Sample of Kennedy key columns before merge (first 5 rows):\n{df_kennedy[[cpg_col, query_col]].head().to_string()}")

    # Calculate overlaps before merging
    if 'mt_id' in df_tecpg.columns and cpg_col in df_kennedy.columns:
        loci_overlap = len(set(df_tecpg['mt_id'].dropna()).intersection(set(df_kennedy[cpg_col].dropna())))
        logging.info(f"Overlapping distinct CpG loci: {loci_overlap}")

    if 'gt_id' in df_tecpg.columns and query_col in df_kennedy.columns:
        genes_overlap = len(set(df_tecpg['gt_id'].dropna()).intersection(set(df_kennedy[query_col].dropna())))
        logging.info(f"Overlapping distinct genes: {genes_overlap}")

    logging.info("Merging datasets...")
    # Inner join on CpG and Gene
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

    # Variables for plotting
    # Effect Size: tecpg mt_est vs Kennedy beta
    # Test Statistic: tecpg mt_t vs Kennedy T.stat

    beta_col = 'beta' if 'beta' in df_kennedy.columns else 'estimate' if 'estimate' in df_kennedy.columns else None
    tstat_col = 'T.stat' if 'T.stat' in df_kennedy.columns else 't.stat' if 't.stat' in df_kennedy.columns else None
    pval_col = 'p.val' if 'p.val' in df_kennedy.columns else 'p.value' if 'p.value' in df_kennedy.columns else None

    if beta_col is None or tstat_col is None:
        # Try to infer
        logging.warning("Could not automatically identify beta or T.stat columns in Kennedy data. Will try to infer.")
        beta_col = df_kennedy.columns[2] if len(df_kennedy.columns) > 2 else df_kennedy.columns[-2]
        tstat_col = df_kennedy.columns[3] if len(df_kennedy.columns) > 3 else df_kennedy.columns[-1]

    # --- Comparison A: Statistical Concordance ---
    logging.info("Calculating concordance metrics...")

    # Filter out NaNs if any
    valid_beta = df_merged[['mt_est', beta_col]].dropna()
    valid_t = df_merged[['mt_t', tstat_col]].dropna()

    pearson_r_beta, _ = stats.pearsonr(valid_beta['mt_est'], valid_beta[beta_col])
    spearman_r_beta, _ = stats.spearmanr(valid_beta['mt_est'], valid_beta[beta_col])
    r2_beta = pearson_r_beta**2

    pearson_r_t, _ = stats.pearsonr(valid_t['mt_t'], valid_t[tstat_col])
    spearman_r_t, _ = stats.spearmanr(valid_t['mt_t'], valid_t[tstat_col])
    r2_t = pearson_r_t**2

    # Plotting
    logging.info("Generating scatter plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Beta vs mt_est
    ax1 = axes[0]
    ax1.hexbin(valid_beta['mt_est'], valid_beta[beta_col], gridsize=50, cmap='Blues', mincnt=1)
    # Overlay identity line
    min_val_b = min(valid_beta['mt_est'].min(), valid_beta[beta_col].min())
    max_val_b = max(valid_beta['mt_est'].max(), valid_beta[beta_col].max())
    ax1.plot([min_val_b, max_val_b], [min_val_b, max_val_b], 'r--', lw=2, label='y=x')
    ax1.set_xlabel('tecpg Effect Size (mt_est)')
    ax1.set_ylabel('Kennedy Effect Size (beta)')
    ax1.set_title('Effect Size Concordance')
    # Text box
    text_b = f"Pearson r: {pearson_r_beta:.3f}\nSpearman $\\rho$: {spearman_r_beta:.3f}\n$R^2$: {r2_beta:.3f}"
    ax1.text(0.05, 0.95, text_b, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 2: T.stat vs mt_t
    ax2 = axes[1]
    ax2.hexbin(valid_t['mt_t'], valid_t[tstat_col], gridsize=50, cmap='Blues', mincnt=1)
    min_val_t = min(valid_t['mt_t'].min(), valid_t[tstat_col].min())
    max_val_t = max(valid_t['mt_t'].max(), valid_t[tstat_col].max())
    ax2.plot([min_val_t, max_val_t], [min_val_t, max_val_t], 'r--', lw=2, label='y=x')
    ax2.set_xlabel('tecpg Test Statistic (mt_t)')
    ax2.set_ylabel('Kennedy Test Statistic (T.stat)')
    ax2.set_title('Test Statistic Concordance')
    text_t = f"Pearson r: {pearson_r_t:.3f}\nSpearman $\\rho$: {spearman_r_t:.3f}\n$R^2$: {r2_t:.3f}"
    ax2.text(0.05, 0.95, text_t, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    scatter_path = os.path.join(args.outdir, 'concordance_scatter.png')
    plt.savefig(scatter_path, dpi=300)
    plt.close()

    # --- Comparison B: Hit Overlap and FDR Sensitivity ---
    logging.info("Calculating overlap sets...")

    # Calculate Sets for ALL
    tecpg_all_mappings = set(zip(df_tecpg['mt_id'], df_tecpg['gt_id']))
    tecpg_all_genes = set(df_tecpg['gt_id'].dropna())
    tecpg_all_loci = set(df_tecpg['mt_id'].dropna())

    kennedy_all_mappings = set(zip(df_kennedy[cpg_col], df_kennedy[query_col]))
    kennedy_all_genes = set(df_kennedy[query_col].dropna())
    kennedy_all_loci = set(df_kennedy[cpg_col].dropna())

    # Calculate Sets for SIGNIFICANT
    tecpg_fdr_col = 'fdr_est' if 'fdr_est' in df_tecpg.columns else 'fdr_bh' if 'fdr_bh' in df_tecpg.columns else 'mt_p' if 'mt_p' in df_tecpg.columns else None

    if pval_col is None:
        pval_col = [c for c in df_kennedy.columns if 'p' in c.lower()][-1]

    kennedy_sig_df = df_kennedy[df_kennedy[pval_col] < args.p_thresh]
    kennedy_sig_mappings = set(zip(kennedy_sig_df[cpg_col], kennedy_sig_df[query_col]))
    kennedy_sig_genes = set(kennedy_sig_df[query_col].dropna())
    kennedy_sig_loci = set(kennedy_sig_df[cpg_col].dropna())

    if tecpg_fdr_col:
        # if using FDR, < 0.05
        tecpg_thresh = 0.05 if 'fdr' in tecpg_fdr_col.lower() else args.p_thresh
        tecpg_sig_df = df_tecpg[df_tecpg[tecpg_fdr_col] < tecpg_thresh]
    else:
        logging.warning("Could not find FDR or p-value column in tecpg data. Using all merged pairs as tecpg hits.")
        tecpg_sig_df = df_tecpg

    tecpg_sig_mappings = set(zip(tecpg_sig_df['mt_id'], tecpg_sig_df['gt_id']))
    tecpg_sig_genes = set(tecpg_sig_df['gt_id'].dropna())
    tecpg_sig_loci = set(tecpg_sig_df['mt_id'].dropna())

    def create_plots(tecpg_set, kennedy_set, title_prefix, filename_prefix):
        overlap = kennedy_set.intersection(tecpg_set)
        union = kennedy_set.union(tecpg_set)
        jaccard = len(overlap) / len(union) if len(union) > 0 else 0

        # Venn diagram
        plt.figure(figsize=(8, 6))
        venn2([tecpg_set, kennedy_set], set_labels=('tecpg', 'Kennedy'))
        plt.title(f'{title_prefix} Overlap')
        venn_path = os.path.join(args.outdir, f'overlap_venn_{filename_prefix}.png')
        plt.savefig(venn_path, dpi=300)
        plt.close()

        # UpSet plot
        if len(tecpg_set) > 0 or len(kennedy_set) > 0:
            upset_data = upsetplot.from_contents({
                'tecpg': tecpg_set,
                'Kennedy': kennedy_set
            })
            plt.figure(figsize=(8, 6))
            upsetplot.plot(upset_data)
            plt.title(f'UpSet Plot of {title_prefix}')
            upset_path = os.path.join(args.outdir, f'overlap_upset_{filename_prefix}.png')
            plt.savefig(upset_path, dpi=300)
            plt.close()

        only_tecpg = len(tecpg_set - kennedy_set)
        only_kennedy = len(kennedy_set - tecpg_set)
        return len(tecpg_set), len(kennedy_set), len(overlap), only_tecpg, only_kennedy, jaccard

    logging.info("Generating overlap visualizations...")

    comparisons = [
        (tecpg_all_mappings, kennedy_all_mappings, 'All Mappings', 'all_mappings'),
        (tecpg_all_genes, kennedy_all_genes, 'All Genes', 'all_genes'),
        (tecpg_all_loci, kennedy_all_loci, 'All Loci', 'all_loci'),
        (tecpg_sig_mappings, kennedy_sig_mappings, 'Significant Mappings', 'sig_mappings'),
        (tecpg_sig_genes, kennedy_sig_genes, 'Significant Genes', 'sig_genes'),
        (tecpg_sig_loci, kennedy_sig_loci, 'Significant Loci', 'sig_loci'),
    ]

    results = {}
    for t_set, k_set, title, fname in comparisons:
        results[title] = create_plots(t_set, k_set, title, fname)

    # Summary Output
    def format_overlap_stats(title, stats_tuple):
        t_len, k_len, o_len, only_t, only_k, jaccard = stats_tuple
        return f"{title}:\n  tecpg Count:   {t_len}\n  Kennedy Count: {k_len}\n  Overlap:       {o_len}\n  Only in tecpg: {only_t}\n  Only in Kennedy: {only_k}\n  Jaccard Index: {jaccard:.4f}\n"

    summary = f"""Benchmark Summary
=================
Input tecpg file: {args.tecpg}
Input Kennedy file: {args.kennedy}
Kennedy p-value threshold: {args.p_thresh}

Mapping & Merging
-----------------
Overlapping Pairs Mapped: {num_merged}

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

Comparison B: Hit Overlap
-------------------------
"""
    for title, fname in [comp[2:4] for comp in comparisons]:
        summary += format_overlap_stats(title, results[title])

    summary += f"""
Outputs
-------
Summary saved to: benchmark_summary.txt
Plots saved to: concordance_scatter.png, overlap_venn_*.png, overlap_upset_*.png
"""
    print(summary)
    summary_path = os.path.join(args.outdir, 'benchmark_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)

if __name__ == '__main__':
    main()
