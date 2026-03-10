import argparse
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for stratified subsampling
HITS_P_THRESH = 1e-5
SIGNAL_P_THRESH = 0.05
SIGNAL_FRAC = 0.05
NOISE_FRAC = 0.005

class Plotter:
    def __init__(self, parquet_path: str, output_dir: str):
        self.parquet_path = parquet_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = None

    def load_and_subsample_data(self):
        """Loads parquet data using stratified subsampling to manage memory and visual density."""
        logger.info(f"Loading data from {self.parquet_path}...")

        # Read the file using pyarrow iteratively to save memory
        parquet_file = pq.ParquetFile(self.parquet_path)

        # Check available columns
        schema = parquet_file.schema_arrow
        col_names = schema.names

        p_col = 'precise_mt_p' if 'precise_mt_p' in col_names else 'mt_p'
        if p_col not in col_names:
            raise ValueError(f"Neither 'precise_mt_p' nor 'mt_p' found in {self.parquet_path}")

        logger.info(f"Using {p_col} as the p-value column.")

        required_cols = ['mt_est', p_col, 'region', 'gt_id', 'mt_id', 'mt_chr', 'mt_pos', 'fdr']
        cols_to_load = [c for c in required_cols if c in col_names]

        chunks = []

        for batch in parquet_file.iter_batches(columns=cols_to_load):
            df_batch = batch.to_pandas()

            # Extract p-values
            p_vals = df_batch[p_col]

            # Stratified subsampling
            hits_mask = p_vals < HITS_P_THRESH
            signal_mask = (p_vals >= HITS_P_THRESH) & (p_vals < SIGNAL_P_THRESH)
            noise_mask = p_vals >= SIGNAL_P_THRESH

            # Sample
            sampled_signal = df_batch[signal_mask].sample(frac=SIGNAL_FRAC, random_state=42)
            sampled_noise = df_batch[noise_mask].sample(frac=NOISE_FRAC, random_state=42)

            chunks.append(pd.concat([df_batch[hits_mask], sampled_signal, sampled_noise]))

        self.df = pd.concat(chunks, ignore_index=True)
        self.p_col = p_col

        # Calculate -log10(p)
        self.df['neg_log10_p'] = -np.log10(self.df[self.p_col].clip(lower=1e-300))

        logger.info(f"Loaded and subsampled {len(self.df)} rows for plotting.")

    def plot_volcano(self):
        logger.info("Generating Volcano Plot...")
        plt.figure(figsize=(10, 8))

        # Determine unique regions and colors
        regions = self.df['region'].dropna().unique()
        palette = sns.color_palette('husl', n_colors=len(regions))

        for idx, region in enumerate(regions):
            region_df = self.df[self.df['region'] == region]

            # Split into significant and non-significant for alpha blending
            sig_mask = region_df[self.p_col] < HITS_P_THRESH

            # Plot non-significant (noise + signal) with low alpha
            plt.scatter(region_df.loc[~sig_mask, 'mt_est'],
                        region_df.loc[~sig_mask, 'neg_log10_p'],
                        color=palette[idx], alpha=0.3, s=10, label=region)

            # Plot significant hits with full alpha
            plt.scatter(region_df.loc[sig_mask, 'mt_est'],
                        region_df.loc[sig_mask, 'neg_log10_p'],
                        color=palette[idx], alpha=1.0, s=20)

        plt.axhline(-np.log10(HITS_P_THRESH), color='black', linestyle='--', lw=1, alpha=0.5)

        # Label top 10 most significant genes
        top10 = self.df.nsmallest(10, self.p_col)
        for _, row in top10.iterrows():
            plt.annotate(row['gt_id'], (row['mt_est'], row['neg_log10_p']),
                         xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

        plt.xlabel('Effect Size (Coefficient / Beta)')
        plt.ylabel(f'-log10({self.p_col})')
        plt.title('eQTM Volcano Plot')
        plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, 'volcano_plot.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Volcano Plot to {out_path}")

    def plot_manhattan(self):
        logger.info("Generating Manhattan Plot...")

        if 'mt_chr' not in self.df.columns or 'mt_pos' not in self.df.columns:
            logger.warning("Missing mt_chr or mt_pos columns. Skipping Manhattan Plot.")
            return

        # Prepare chromosome data
        df_mh = self.df.copy()

        # Clean chromosome names to just numbers/X/Y
        df_mh['mt_chr_clean'] = df_mh['mt_chr'].astype(str).str.replace('chr', '', case=False)

        # Map chromosomes to integers for sorting
        def map_chr(c):
            if c.upper() == 'X': return 23
            if c.upper() == 'Y': return 24
            if c.upper() == 'MT': return 25
            try:
                return int(c)
            except:
                return 99

        df_mh['chr_num'] = df_mh['mt_chr_clean'].apply(map_chr)
        df_mh = df_mh[df_mh['chr_num'] < 99] # Filter out weird contigs
        df_mh = df_mh.sort_values(['chr_num', 'mt_pos'])

        # Calculate absolute positions for continuous x-axis
        df_mh['pos_abs'] = 0
        last_max = 0
        xticks = []
        xlabels = []

        for chr_num, group in df_mh.groupby('chr_num', sort=True):
            if len(group) == 0: continue

            group_min = group['mt_pos'].min()
            group_max = group['mt_pos'].max()

            # Shift positions
            df_mh.loc[group.index, 'pos_abs'] = group['mt_pos'] - group_min + last_max

            # Midpoint for label
            midpoint = (df_mh.loc[group.index, 'pos_abs'].min() + df_mh.loc[group.index, 'pos_abs'].max()) / 2
            xticks.append(midpoint)

            # Original chr name for label
            chr_name = group['mt_chr_clean'].iloc[0]
            xlabels.append(chr_name)

            # Update last_max (add a small gap between chromosomes)
            last_max = df_mh.loc[group.index, 'pos_abs'].max() + 10_000_000

        plt.figure(figsize=(15, 6))

        # Alternating colors
        colors = ['#4c72b0', '#55a868']

        for i, (chr_num, group) in enumerate(df_mh.groupby('chr_num', sort=True)):
            color = colors[i % 2]

            sig_mask = group[self.p_col] < HITS_P_THRESH

            # Non-significant
            plt.scatter(group.loc[~sig_mask, 'pos_abs'], group.loc[~sig_mask, 'neg_log10_p'],
                        color=color, alpha=0.3, s=10)

            # Significant
            plt.scatter(group.loc[sig_mask, 'pos_abs'], group.loc[sig_mask, 'neg_log10_p'],
                        color=color, alpha=1.0, s=20)

        plt.axhline(-np.log10(HITS_P_THRESH), color='red', linestyle='--', lw=1, alpha=0.8, label=f'Threshold (P={HITS_P_THRESH})')

        plt.xticks(xticks, xlabels, rotation=45, fontsize=8)
        plt.xlabel('Chromosome')
        plt.ylabel(f'-log10({self.p_col})')
        plt.title('Genomic Distribution of eQTMs (Manhattan Plot)')
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, 'manhattan_plot.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Manhattan Plot to {out_path}")

    def plot_region_breakdown(self):
        logger.info("Generating Region Breakdown Plot (Significant Hits Only)...")

        if 'region' not in self.df.columns:
            logger.warning("Missing region column. Skipping Region Breakdown.")
            return

        # Filter for significant hits
        sig_df = self.df[self.df[self.p_col] < HITS_P_THRESH]
        region_counts = sig_df['region'].value_counts()

        plt.figure(figsize=(8, 6))
        sns.barplot(x=region_counts.index, y=region_counts.values, hue=region_counts.index, palette='viridis', legend=False)

        plt.xlabel('Region Type')
        plt.ylabel('Count (Subsampled)')
        plt.title('Region-Type Breakdown')

        # Add labels on top of bars
        for i, v in enumerate(region_counts.values):
            plt.text(i, v, str(v), ha='center', va='bottom')

        plt.tight_layout()

        out_path = os.path.join(self.output_dir, 'region_breakdown.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Region Breakdown to {out_path}")


def plot_comparative_eqtm(cpg_id: str, gene_id: str, row: pd.Series, p_col: str, M: pd.DataFrame, G: pd.DataFrame, C: pd.DataFrame, output_dir: str, region: str = "Unknown"):
    logger.info(f"Generating Comparative Scatter Plot for {cpg_id} vs {gene_id}...")

    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        logger.error("scikit-learn is required for adjusted scatter plots. Please install it.")
        return

    if cpg_id not in M.index:
        logger.error(f"CpG ID {cpg_id} not found in methylation matrix.")
        return
    if gene_id not in G.index:
        logger.error(f"Gene ID {gene_id} not found in gene expression matrix.")
        return

    # Extract series and align
    meth_vals = M.loc[cpg_id]
    gene_vals = G.loc[gene_id]

    # Ensure subjects align
    common_subjects = meth_vals.index.intersection(gene_vals.index).intersection(C.index)
    if len(common_subjects) == 0:
        logger.error("No common subjects found between matrices.")
        return

    meth_vals = meth_vals[common_subjects]
    gene_vals = gene_vals[common_subjects]
    C_aligned = C.loc[common_subjects]

    # Drop NAs
    valid_mask = ~meth_vals.isna() & ~gene_vals.isna() & ~C_aligned.isna().any(axis=1)
    meth_vals = meth_vals[valid_mask]
    gene_vals = gene_vals[valid_mask]
    C_aligned = C_aligned[valid_mask]

    if len(meth_vals) == 0:
        logger.error("No valid data points after dropping NAs.")
        return

    # Fit models to get residuals
    X = C_aligned.values

    model_meth = LinearRegression()
    model_meth.fit(X, meth_vals.values)
    meth_residuals = meth_vals.values - model_meth.predict(X)

    model_gene = LinearRegression()
    model_gene.fit(X, gene_vals.values)
    gene_residuals = gene_vals.values - model_gene.predict(X)

    # Setup the side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)

    # --- Panel A: Unadjusted ---
    ax_raw = axes[0]
    sns.regplot(x=meth_vals.values, y=gene_vals.values, scatter_kws={'alpha':0.5}, line_kws={'color': 'red'}, ax=ax_raw)

    # Calculate unadjusted stats
    slope_raw, intercept_raw, r_value_raw, p_value_raw, std_err_raw = stats.linregress(meth_vals.values, gene_vals.values)

    # Annotate unadjusted plot
    ax_raw.annotate(f"R = {r_value_raw:.3f}\nP = {p_value_raw:.2e}", xy=(0.05, 0.90), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    ax_raw.set_xlabel('Raw DNA Methylation (Beta)')
    ax_raw.set_ylabel('Raw Log2 Gene Expression')
    ax_raw.set_title(f'Unadjusted eQTM Association\n{cpg_id} vs {gene_id}')
    ax_raw.grid(True, alpha=0.3)

    # --- Panel B: Adjusted ---
    ax_adj = axes[1]

    # Plot residuals
    ax_adj.scatter(meth_residuals, gene_residuals, alpha=0.5)

    # Calculate partial correlation (r-value) for residuals
    r_value_adj, _ = stats.pearsonr(meth_residuals, gene_residuals)

    # Use mt_est slope from parquet and parquet's p-value
    mt_est = row['mt_est']
    pq_p_val = row[p_col]

    # Draw line through origin using mt_est slope
    x_vals = np.array([meth_residuals.min(), meth_residuals.max()])
    y_vals = mt_est * x_vals
    ax_adj.plot(x_vals, y_vals, color='red', label=f'mt_est slope: {mt_est:.3f}')

    # Annotate adjusted plot
    ax_adj.annotate(f"R = {r_value_adj:.3f}\nP = {pq_p_val:.2e}", xy=(0.05, 0.90), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    ax_adj.legend(loc='lower right')

    ax_adj.set_xlabel('Covariate-Adjusted DNA Methylation')
    ax_adj.set_ylabel('Covariate-Adjusted Log2 Gene Expression')
    ax_adj.set_title(f'Adjusted eQTM Association (Residuals)\n{cpg_id} vs {gene_id} ({region})')
    ax_adj.grid(True, alpha=0.3)

    plt.tight_layout()

    safe_region = region.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(output_dir, f'comparative_scatter_{safe_region}_{cpg_id}_{gene_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Comparative Scatter Plot to {out_path}")


def plot_adjusted_eqtm(cpg_id: str, gene_id: str, M: pd.DataFrame, G: pd.DataFrame, C: pd.DataFrame, output_dir: str, region: str = "Unknown"):
    logger.info(f"Generating Adjusted Scatter Plot for {cpg_id} vs {gene_id}...")

    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        logger.error("scikit-learn is required for adjusted scatter plots. Please install it.")
        return

    if cpg_id not in M.index:
        logger.error(f"CpG ID {cpg_id} not found in methylation matrix.")
        return
    if gene_id not in G.index:
        logger.error(f"Gene ID {gene_id} not found in gene expression matrix.")
        return

    # Extract series and align
    meth_vals = M.loc[cpg_id]
    gene_vals = G.loc[gene_id]

    # Ensure subjects align
    common_subjects = meth_vals.index.intersection(gene_vals.index).intersection(C.index)
    if len(common_subjects) == 0:
        logger.error("No common subjects found between matrices.")
        return

    meth_vals = meth_vals[common_subjects]
    gene_vals = gene_vals[common_subjects]
    C_aligned = C.loc[common_subjects]

    # Drop NAs
    valid_mask = ~meth_vals.isna() & ~gene_vals.isna() & ~C_aligned.isna().any(axis=1)
    meth_vals = meth_vals[valid_mask]
    gene_vals = gene_vals[valid_mask]
    C_aligned = C_aligned[valid_mask]

    if len(meth_vals) == 0:
        logger.error("No valid data points after dropping NAs.")
        return

    # Fit models to get residuals
    X = C_aligned.values

    model_meth = LinearRegression()
    model_meth.fit(X, meth_vals.values)
    meth_residuals = meth_vals.values - model_meth.predict(X)

    model_gene = LinearRegression()
    model_gene.fit(X, gene_vals.values)
    gene_residuals = gene_vals.values - model_gene.predict(X)

    # Plot
    plt.figure(figsize=(8, 8))
    sns.regplot(x=meth_residuals, y=gene_residuals, scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})

    # Add correlation info
    correlation = np.corrcoef(meth_residuals, gene_residuals)[0, 1]
    plt.annotate(f"r = {correlation:.3f}", xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.xlabel('Covariate-Adjusted DNA Methylation')
    plt.ylabel('Covariate-Adjusted Log2 Gene Expression')
    plt.title(f'Adjusted eQTM Association\n{cpg_id} vs {gene_id} ({region})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_region = region.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(output_dir, f'scatter_{safe_region}_{cpg_id}_{gene_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Scatter Plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Multi-Omic Diagnostic Plots")
    parser.add_argument("parquet_file", help="Path to the annotated Parquet results file")

    parser.add_argument("--out-dir", default="plots", help="Directory to save plots")

    # Selection of plots
    parser.add_argument("--volcano", action="store_true", help="Generate Volcano Plot")
    parser.add_argument("--manhattan", action="store_true", help="Generate Manhattan Plot")
    parser.add_argument("--region-breakdown", action="store_true", help="Generate Region Breakdown Plot")
    parser.add_argument("--scatter", action="store_true", help="Generate Scatter Plots for top hits")
    parser.add_argument("--comparative-scatter", action="store_true", help="Generate Comparative (Unadjusted vs Adjusted) Scatter Plots for top hits")
    parser.add_argument("--all", action="store_true", help="Generate all plots (default if none specified)")

    # Tecpg paths for scatter plots
    parser.add_argument("-m", "--meth-file", required=True, help="Full path to the methylation matrix file (e.g., M.csv)")
    parser.add_argument("-g", "--gene-file", required=True, help="Full path to the gene expression matrix file (e.g., G.csv)")
    parser.add_argument("-c", "--covar-file", required=True, help="Full path to the covariate matrix file (e.g., C.csv)")

    args = parser.parse_args()

    # If no specific plot is requested, do all
    if not (args.volcano or args.manhattan or args.region_breakdown or args.scatter or args.comparative_scatter):
        args.all = True

    plotter = Plotter(args.parquet_file, args.out_dir)
    plotter.load_and_subsample_data()

    if args.all or args.volcano:
        plotter.plot_volcano()

    if args.all or args.manhattan:
        plotter.plot_manhattan()

    if args.all or args.region_breakdown:
        plotter.plot_region_breakdown()

    if args.all or args.scatter or args.comparative_scatter:
        logger.info("Preparing to generate scatter plots for top hits per region...")

        if 'fdr' not in plotter.df.columns and plotter.p_col not in plotter.df.columns:
            logger.warning("FDR or p-value column missing, cannot determine top hits.")
            return

        sort_col = 'fdr' if 'fdr' in plotter.df.columns else plotter.p_col
        logger.info(f"Sorting by {sort_col} to find top hits.")

        meth_path = args.meth_file
        gene_path = args.gene_file
        covar_path = args.covar_file

        if not (os.path.exists(meth_path) and os.path.exists(gene_path) and os.path.exists(covar_path)):
            logger.error(f"Missing one or more data files for scatter plots: {meth_path}, {gene_path}, {covar_path}")
            logger.error("Please provide the correct paths using -m, -g, -c")
            return

        # Load matrices once to save memory and I/O time
        try:
            logger.info("Loading methylation, gene, and covariate matrices into memory...")
            M_df = pd.read_csv(meth_path, index_col=0)
            G_df = pd.read_csv(gene_path, index_col=0)
            C_df = pd.read_csv(covar_path, index_col=0)
        except Exception as e:
            logger.error(f"Failed to read data matrices: {e}")
            return

        regions = plotter.df['region'].dropna().unique() if 'region' in plotter.df.columns else ["All"]

        for region in regions:
            if 'region' in plotter.df.columns:
                region_df = plotter.df[plotter.df['region'] == region]
            else:
                region_df = plotter.df

            top_hits = region_df.nsmallest(10, sort_col)

            for _, row in top_hits.iterrows():
                cpg_id = row['mt_id']
                gene_id = row['gt_id']

                if args.all or args.comparative_scatter:
                    plot_comparative_eqtm(cpg_id, gene_id, row, plotter.p_col, M_df, G_df, C_df, args.out_dir, region)
                elif args.scatter:
                    plot_adjusted_eqtm(cpg_id, gene_id, M_df, G_df, C_df, args.out_dir, region)

if __name__ == "__main__":
    main()
