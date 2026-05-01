import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pyarrow.parquet as pq
import mygene
from matplotlib_venn import venn2
import upsetplot
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

def map_kennedy_ids(df_kennedy):
    # Kennedy format usually has CpG.probe, annot.gene, and/or exp.Probe
    # We will try to map annot.gene (Gene Symbol) or exp.Probe (Illumina ID) to Ensembl ID

    # We first need to check what columns exist
    if 'exp.Probe' in df_kennedy.columns:
        query_col = 'exp.Probe'
        scopes = 'reporter'
    elif 'annot.gene' in df_kennedy.columns:
        query_col = 'annot.gene'
        scopes = 'symbol'
    else:
        logging.warning("Could not find 'exp.Probe' or 'annot.gene' in Kennedy data to map. Will use whatever transcript column is available.")
        # fallback
        query_col = df_kennedy.columns[1] # assuming 2nd column is transcript
        scopes = 'symbol,reporter'

    logging.info(f"Querying mygene to translate {len(df_kennedy[query_col].unique())} unique IDs from '{query_col}'...")

    mg = mygene.MyGeneInfo()
    try:
        # Use a large step or just query many
        results = mg.querymany(df_kennedy[query_col].unique().tolist(), scopes=scopes, fields='ensembl.gene', species='human', as_dataframe=True)

        # results index is the queried ID, columns include 'ensembl.gene' or 'ensembl'
        mapping_dict = {}
        for query_id, row in results.iterrows():
            if isinstance(row, pd.DataFrame):
                # Multiple hits, take the first one
                row = row.iloc[0]

            if 'ensembl' in row and isinstance(row['ensembl'], list):
                # If ensembl is a list of dicts
                mapping_dict[query_id] = row['ensembl'][0]['gene']
            elif 'ensembl' in row and isinstance(row['ensembl'], dict):
                mapping_dict[query_id] = row['ensembl']['gene']
            elif 'ensembl.gene' in row and not pd.isna(row['ensembl.gene']):
                mapping_dict[query_id] = row['ensembl.gene']

        logging.info(f"Successfully mapped {len(mapping_dict)} IDs.")
        df_kennedy['mapped_gt_id'] = df_kennedy[query_col].map(mapping_dict)

        # Fallback to original if mapping fails?
        # Often tecpg output has version suffixes like ENSG000001.1
        # tecpg's summarizeOutput_parquet strips suffixes from gt_id, we will do the same on tecpg data.

    except Exception as e:
        logging.error(f"Error mapping IDs: {e}")
        df_kennedy['mapped_gt_id'] = df_kennedy[query_col]

    return df_kennedy

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
    except Exception as e:
        logging.error(f"Failed to read tecpg file: {e}")
        sys.exit(1)

    logging.info(f"Loading Kennedy data from {args.kennedy}...")
    try:
        # Kennedy format is likely tab or space separated
        df_kennedy = pd.read_csv(args.kennedy, sep=None, engine='python')
    except Exception as e:
        logging.error(f"Failed to read Kennedy file: {e}")
        sys.exit(1)

    logging.info("Preprocessing and mapping IDs...")

    # Strip version suffix from tecpg gt_id if present
    df_tecpg['gt_id_base'] = df_tecpg['gt_id'].astype(str).str.split('.').str[0]

    df_kennedy = map_kennedy_ids(df_kennedy)

    # Standardize column names for merge
    # Kennedy CpG usually 'CpG.probe', tecpg 'mt_id'
    cpg_col = 'CpG.probe' if 'CpG.probe' in df_kennedy.columns else df_kennedy.columns[0]

    df_kennedy = df_kennedy.dropna(subset=['mapped_gt_id', cpg_col])

    logging.info("Merging datasets...")
    # Inner join on CpG and Gene
    df_merged = pd.merge(
        df_tecpg,
        df_kennedy,
        left_on=['mt_id', 'gt_id_base'],
        right_on=[cpg_col, 'mapped_gt_id'],
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
    logging.info("Calculating hit overlap...")
    # Assume tecpg FDR column is 'fdr_est', 'fdr_bh' or 'mt_fdr'
    tecpg_fdr_col = 'fdr_est' if 'fdr_est' in df_tecpg.columns else 'fdr_bh' if 'fdr_bh' in df_tecpg.columns else 'mt_p' if 'mt_p' in df_tecpg.columns else None

    if pval_col is None:
        pval_col = [c for c in df_kennedy.columns if 'p' in c.lower()][-1]

    kennedy_sig = set(df_kennedy[df_kennedy[pval_col] < args.p_thresh].apply(lambda row: (row[cpg_col], row['mapped_gt_id']), axis=1))

    if tecpg_fdr_col:
        # if using FDR, < 0.05
        tecpg_thresh = 0.05 if 'fdr' in tecpg_fdr_col.lower() else args.p_thresh
        tecpg_sig = set(df_tecpg[df_tecpg[tecpg_fdr_col] < tecpg_thresh].apply(lambda row: (row['mt_id'], row['gt_id_base']), axis=1))
    else:
        logging.warning("Could not find FDR or p-value column in tecpg data. Using all merged pairs as tecpg hits.")
        tecpg_sig = set(df_tecpg.apply(lambda row: (row['mt_id'], row['gt_id_base']), axis=1))

    overlap = kennedy_sig.intersection(tecpg_sig)
    union = kennedy_sig.union(tecpg_sig)
    jaccard = len(overlap) / len(union) if len(union) > 0 else 0

    logging.info("Generating overlap visualizations...")
    # Venn diagram
    plt.figure(figsize=(8, 6))
    venn2([tecpg_sig, kennedy_sig], set_labels=('tecpg Hits', 'Kennedy Hits'))
    plt.title('Significant Hits Overlap')
    venn_path = os.path.join(args.outdir, 'overlap_venn.png')
    plt.savefig(venn_path, dpi=300)
    plt.close()

    # UpSet plot
    upset_data = upsetplot.from_contents({
        'tecpg Hits': tecpg_sig,
        'Kennedy Hits': kennedy_sig
    })
    plt.figure(figsize=(8, 6))
    upsetplot.plot(upset_data)
    plt.title('UpSet Plot of Significant Hits')
    upset_path = os.path.join(args.outdir, 'overlap_upset.png')
    plt.savefig(upset_path, dpi=300)
    plt.close()

    # Summary Output
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
tecpg Significant Hits:   {len(tecpg_sig)}
Kennedy Significant Hits: {len(kennedy_sig)}
Overlapping Hits:         {len(overlap)}
Jaccard Index:            {jaccard:.4f}

Outputs
-------
Summary saved to: benchmark_summary.txt
Plots saved to: concordance_scatter.png, overlap_venn.png, overlap_upset.png
"""
    print(summary)
    summary_path = os.path.join(args.outdir, 'benchmark_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)

if __name__ == '__main__':
    main()
