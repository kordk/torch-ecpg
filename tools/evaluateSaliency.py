import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_WINDOWS = ["1-50", "200-250", "1000-1050"]

def parse_rank_windows(windows_str_list):
    windows = []
    for w in windows_str_list:
        try:
            start, end = map(int, w.split('-'))
            if start > end or start < 1:
                raise ValueError
            windows.append((start, end))
        except ValueError:
            logger.error(f"Invalid window format: {w}. Must be START-END with START <= END and START >= 1.")
            sys.exit(1)
    return windows

def detect_inflection(y, method='auto'):
    n = len(y)
    if n < 3:
        return None, "Not enough points"

    x = np.arange(n)

    use_kneed = False
    if method in ['auto', 'kneed']:
        try:
            from kneed import KneeLocator
            use_kneed = True
        except ImportError:
            if method == 'kneed':
                logger.error("kneed package not found but required by --inflection-method kneed.")
                sys.exit(1)
            use_kneed = False

    if use_kneed:
        kl = KneeLocator(x, y, curve='convex', direction='decreasing')
        knee = kl.knee
        if knee is not None:
            return knee, "kneed"

    # Fallback: max distance to chord
    # Normalize x and y to [0, 1] so that differing scales don't distort distance
    x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
    y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y

    # chord line connects (x_norm[0], y_norm[0]) to (x_norm[-1], y_norm[-1])
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])

    # Update p_vec to use normalized values
    norm_points = np.column_stack((x_norm, y_norm))

    # Calculate distance from each point to the line
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return 0, "chord"

    line_unitvec = line_vec / line_len
    p_vec = norm_points - p1

    # t is the scalar projection of p_vec onto the line
    t = np.dot(p_vec, line_unitvec)

    # nearest points on the line
    nearest = p1 + np.outer(t, line_unitvec)

    # distance
    dist = np.linalg.norm(norm_points - nearest, axis=1)

    knee = np.argmax(dist)
    return knee, "chord"

def load_data(filepath, rank_by):
    logger.info(f"Loading data from {filepath}...")
    try:
        parquet_file = pq.ParquetFile(filepath)
        schema = parquet_file.schema
        columns = schema.names

        # We need ranking metric, mt_ig, all _ig columns, region, mt_est/mt_t if available, mt_id, gt_id
        cols_to_load = ['mt_id', 'gt_id', 'region']
        if rank_by in columns:
            cols_to_load.append(rank_by)
        else:
            logger.error(f"Ranking metric '{rank_by}' not found in Parquet schema.")
            sys.exit(1)

        ig_cols = [c for c in columns if c.endswith('_ig')]
        cols_to_load.extend(ig_cols)

        if 'mt_est' in columns and 'mt_est' not in cols_to_load:
            cols_to_load.append('mt_est')
        if 'mt_t' in columns and 'mt_t' not in cols_to_load:
            cols_to_load.append('mt_t')

        # Optional precise p
        if 'p_boot' in columns and 'p_boot' not in cols_to_load:
            cols_to_load.append('p_boot')

        cols_to_load = list(set(cols_to_load).intersection(set(columns)))

        # Read the file in chunks to handle very large datasets without OOM
        batches = []
        for batch in parquet_file.iter_batches(columns=cols_to_load, batch_size=500000):
            batch_df = batch.to_pandas()
            if batch_df.index.names != [None]:
                batch_df = batch_df.reset_index()
            batches.append(batch_df)

        df = pd.concat(batches, ignore_index=True)

        # Fill NA region
        if 'region' in df.columns:
            df['region'] = df['region'].fillna('UNKNOWN')
        else:
            df['region'] = 'UNKNOWN'

        return df, ig_cols
    except Exception as e:
        logger.error(f"Failed to read Parquet file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Integrated Gradients Saliency distribution.")
    parser.add_argument("-i", "--input", required=True, help="Input Parquet file (e.g., bootstrap_merged.parquet).")
    parser.add_argument("-o", "--out-dir", default=".", help="Output directory for plots.")
    parser.add_argument("--rank-by", default="mt_ig", choices=["mt_ig", "p_boot", "mt_t", "mt_est", "abs_t"], help="Metric to rank pairs by.")
    parser.add_argument("--rank-windows", nargs="+", default=DEFAULT_WINDOWS, help="Rank windows for summaries (e.g., 1-50 200-250).")
    parser.add_argument("--inflection-method", default="auto", choices=["auto", "kneed", "chord"], help="Method to detect inflection point.")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    windows = parse_rank_windows(args.rank_windows)

    df, ig_cols = load_data(args.input, args.rank_by)

    if 'mt_ig' not in ig_cols:
        logger.error("No 'mt_ig' column found in the dataset. Saliency evaluation requires Integrated Gradients output.")
        sys.exit(1)

    has_covar_ig = len(ig_cols) > 1

    # Absolute magnitudes
    for col in ig_cols:
        df[col] = df[col].abs()

    if has_covar_ig:
        df['total_ig'] = df[ig_cols].sum(axis=1)
        df['mt_ig_frac'] = df['mt_ig'] / df['total_ig']
        # Handle zero division
        df['mt_ig_frac'] = df['mt_ig_frac'].fillna(0)
    else:
        logger.warning("Only scalar 'mt_ig' is available. Per-feature IG analysis will be skipped, and fraction defaults to 1.0.")
        df['total_ig'] = df['mt_ig']
        df['mt_ig_frac'] = 1.0

    # Sort
    asc = False
    if args.rank_by == 'p_boot':
        asc = True
    elif args.rank_by in ['mt_t', 'mt_est']:
        df['abs_rank'] = df[args.rank_by].abs()
        sort_col = 'abs_rank'
    else:
        sort_col = args.rank_by

    df = df.sort_values(by=sort_col, ascending=asc).reset_index(drop=True)

    # A. Console Summary Report
    print("=" * 60)
    print("SALIENCY EVALUATION REPORT")
    print("=" * 60)
    print(f"Total pairs analyzed: {len(df)}")

    if has_covar_ig:
        print("\n--- Methylation Saliency Fraction Distribution ---")
        fracs = df['mt_ig_frac']
        print(f"Min: {fracs.min():.4f}")
        print(f"Max: {fracs.max():.4f}")
        print(f"Mean: {fracs.mean():.4f}")
        print(f"Median: {fracs.median():.4f}")
        print(f"Q1 (25%): {fracs.quantile(0.25):.4f}")
        print(f"Q3 (75%): {fracs.quantile(0.75):.4f}")
    else:
        print("\n--- Saliency Magnitude Distribution (|mt_ig|) ---")
        mags = df['mt_ig']
        print(f"Min: {mags.min():.4f}")
        print(f"Max: {mags.max():.4f}")
        print(f"Mean: {mags.mean():.4f}")
        print(f"Median: {mags.median():.4f}")

    print("\n--- Methylation Saliency Fraction by Rank Bands ---")
    overall_mean = df['mt_ig_frac'].mean()
    print(f"Overall Mean Fraction: {overall_mean:.4f}")

    for start, end in windows:
        if start > len(df):
            logger.warning(f"Window {start}-{end} exceeds total pairs ({len(df)}). Skipping.")
            continue
        actual_end = min(end, len(df))
        band = df.iloc[start-1:actual_end]
        band_mean = band['mt_ig_frac'].mean()
        print(f"Rank {start}-{actual_end}: {band_mean:.4f}")

    print("\n--- Breakdown by Region ---")
    region_means = df.groupby('region')['mt_ig_frac'].mean()
    region_counts = df.groupby('region').size()
    for r in region_means.index:
        print(f"{r}: {region_means[r]:.4f} (n={region_counts[r]})")

    if has_covar_ig:
        print("\n--- Mean Saliency Proportion per Feature Class ---")
        for col in ig_cols:
            prop = (df[col] / df['total_ig']).fillna(0).mean()
            print(f"{col}: {prop:.4f}")
        print("Note: If covariate saliency is low, consider verifying if input scales are comparable.")

    print("\n--- Inflection Point Analysis ---")
    knee, method_used = detect_inflection(df['mt_ig'].values, args.inflection_method)
    if knee is not None:
        print(f"Magnitude curve (|mt_ig|) inflection point at rank {knee + 1} (method: {method_used})")
    else:
        print("Magnitude curve inflection point could not be detected.")

    if has_covar_ig:
        knee_frac, method_used_frac = detect_inflection(df['mt_ig_frac'].values, args.inflection_method)
        if knee_frac is not None:
            print(f"Fraction curve (mt_ig_frac) inflection point at rank {knee_frac + 1} (method: {method_used_frac})")
        else:
            print("Fraction curve inflection point could not be detected.")

    # B & C. Plots
    # Create the stacked proportional chart for rank windows
    if has_covar_ig:
        print("\nGenerating Rank-Windowed Stacked Proportional Saliency Plots...")
        for start, end in windows:
            if start > len(df):
                continue
            actual_end = min(end, len(df))
            band = df.iloc[start-1:actual_end].copy()

            # Recalculate proportions locally for plotting
            prop_columns = []
            for col in ig_cols:
                prop_col = f"{col}_prop"
                band[prop_col] = band[col] / band['total_ig']
                prop_columns.append(prop_col)

            band['locus_pair'] = band['mt_id'] + " - " + band['gt_id']
            plot_df = band.set_index('locus_pair')[prop_columns]
            plot_df = plot_df.iloc[::-1] # Reverse to match #1 at top

            import matplotlib as mpl
            colors = []
            cmap = mpl.colormaps.get_cmap('Pastel1') if hasattr(mpl.colormaps, 'get_cmap') else plt.cm.Pastel1
            covar_idx = 0
            for col in prop_columns:
                if col == 'mt_ig_prop':
                    colors.append('darkblue')
                else:
                    colors.append(cmap(covar_idx % 9))
                    covar_idx += 1

            fig, ax = plt.subplots(figsize=(12, max(6, (actual_end - start) * 0.2)))
            plot_df.plot.barh(stacked=True, color=colors, width=0.8, ax=ax)

            plt.title(f"Stacked Proportional Saliency Profile (Ranks {start}-{actual_end})")
            plt.xlabel("Proportion of Total Saliency")
            plt.ylabel("Locus Pair (CpG - Gene)")

            handles, labels = ax.get_legend_handles_labels()
            cleaned_labels = [label.replace('_ig_prop', '') for label in labels]
            ax.legend(handles, cleaned_labels, title="Features", loc='center left', bbox_to_anchor=(1.0, 0.5))
            plt.tight_layout()
            out_file = os.path.join(args.out_dir, f"saliency_profile_ranks_{start}_{actual_end}.png")
            plt.savefig(out_file)
            plt.close()

    # Histogram
    if has_covar_ig:
        plt.figure(figsize=(8, 6))
        sns.histplot(df['mt_ig_frac'], bins=50)
        plt.title("Distribution of Methylation Saliency Fraction")
        plt.xlabel("Methylation Saliency Fraction")
        plt.ylabel("Count")
        plt.savefig(os.path.join(args.out_dir, "saliency_fraction_hist.png"))
        plt.close()

    # Scatter vs effect size
    if has_covar_ig and ('mt_est' in df.columns or 'mt_t' in df.columns):
        effect_col = 'mt_est' if 'mt_est' in df.columns else 'mt_t'
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[effect_col].abs(), y=df['mt_ig_frac'], alpha=0.3)
        plt.title(f"Methylation Saliency Fraction vs |{effect_col}|")
        plt.xlabel(f"|{effect_col}|")
        plt.ylabel("Methylation Saliency Fraction")
        plt.savefig(os.path.join(args.out_dir, f"saliency_vs_{effect_col}.png"))
        plt.close()

    # Box plot by region
    if has_covar_ig:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='region', y='mt_ig_frac', data=df)
        plt.title("Methylation Saliency Fraction by Region")
        plt.xlabel("Region")
        plt.ylabel("Methylation Saliency Fraction")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "saliency_fraction_by_region.png"))
        plt.close()

    # Scree-style decay curves
    # Curve 1: Fraction vs Rank
    x_ranks = np.arange(1, len(df) + 1)

    if has_covar_ig:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x_ranks, y=df['mt_ig_frac'], hue=df['region'], alpha=0.5, edgecolor=None, s=10)
        if knee_frac is not None:
            plt.axvline(x=knee_frac + 1, color='r', linestyle='--', label=f'Inflection ({method_used_frac}): {knee_frac + 1}')
        plt.xscale('log')
        plt.title("Methylation Saliency Fraction vs Rank")
        plt.xlabel("Rank (Log Scale)")
        plt.ylabel("Methylation Saliency Fraction")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.figtext(0.01, 0.01, "Note: Structure describes candidate set, not genome-wide distribution.", fontsize=8)
        plt.savefig(os.path.join(args.out_dir, "saliency_fraction_decay_curve.png"))
        plt.close()

    # Curve 2: Magnitude vs Rank
    plt.figure(figsize=(10, 6))
    plt.plot(x_ranks, df['mt_ig'].values, color='darkblue')
    if knee is not None:
        plt.axvline(x=knee + 1, color='r', linestyle='--', label=f'Inflection ({method_used}): {knee + 1}')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Saliency Magnitude (|mt_ig|) vs Rank")
    plt.xlabel("Rank (Log Scale)")
    plt.ylabel("Magnitude (|mt_ig|) (Log Scale)")
    plt.legend()
    plt.tight_layout()
    plt.figtext(0.01, 0.01, "Note: Structure describes candidate set, not genome-wide distribution.", fontsize=8)
    plt.savefig(os.path.join(args.out_dir, "saliency_magnitude_decay_curve.png"))
    plt.close()

    print("\nEvaluation completed. Plots saved to:", args.out_dir)

if __name__ == "__main__":
    main()
