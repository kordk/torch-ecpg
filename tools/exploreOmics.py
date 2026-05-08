#!/usr/bin/env python3
"""
Explore omics data (methylation, expression) and generate summary reports.
Provides quality control metrics and visualizations.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import random

def setup_logger(name, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(dir_path):
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_omics_data(filepath, transpose=False):
    # Load data with first column as index (Sample IDs)
    df = pd.read_csv(filepath, index_col=0)
    if transpose:
        df = df.T
    return df

def log_dataframe_info(logger, df, name):
    logger.info(f"{name} info:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")



def calculate_qc_metrics(df: pd.DataFrame, logger) -> dict:
    """
    Calculate quality control metrics for omics data.

    Args:
        df: Omics data DataFrame (samples × features)
        logger: Logger instance

    Returns:
        Dictionary with QC metrics
    """
    metrics = {}

    # Basic statistics
    metrics['n_samples'] = df.shape[0]
    metrics['n_features'] = df.shape[1]
    metrics['missing_rate'] = df.isna().sum().sum() / (df.shape[0] * df.shape[1])

    # Per-sample statistics
    metrics['samples_mean'] = df.mean(axis=1).mean()
    metrics['samples_std'] = df.mean(axis=1).std()
    metrics['samples_missing_mean'] = df.isna().sum(axis=1).mean()

    # Per-feature statistics
    metrics['features_mean'] = df.mean(axis=0).mean()
    metrics['features_std'] = df.mean(axis=0).std()
    metrics['features_missing_mean'] = df.isna().sum(axis=0).mean()

    logger.info("\nQuality Control Metrics:")
    logger.info(f"  Samples: {metrics['n_samples']}")
    logger.info(f"  Features: {metrics['n_features']}")
    logger.info(f"  Overall missing rate: {metrics['missing_rate']:.4f}")
    logger.info(f"  Sample mean (avg): {metrics['samples_mean']:.4f}")
    logger.info(f"  Feature mean (avg): {metrics['features_mean']:.4f}")

    return metrics


def plot_sample_distributions(df: pd.DataFrame, output_dir: Path,
                             data_type: str, logger) -> None:
    """
    Plot distributions across samples.

    Args:
        df: Omics data DataFrame
        output_dir: Output directory
        data_type: Type of data (for plot title)
        logger: Logger instance
    """
    logger.info("Creating sample distribution plots...")

    # Sample means
    sample_means = df.mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(sample_means, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Mean Value per Sample')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{data_type} - Sample Mean Distribution')

    # Box plot
    axes[1].boxplot([sample_means])
    axes[1].set_ylabel('Mean Value')
    axes[1].set_title(f'{data_type} - Sample Mean Box Plot')
    axes[1].set_xticklabels(['All Samples'])

    plt.tight_layout()
    plot_path = output_dir / f'{data_type}_sample_distributions.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved sample distribution plot: {plot_path}")


def plot_feature_distributions(df: pd.DataFrame, output_dir: Path,
                              data_type: str, logger, n_features: int = 100) -> None:
    """
    Plot distributions of top variable features.

    Args:
        df: Omics data DataFrame
        output_dir: Output directory
        data_type: Type of data (for plot title)
        logger: Logger instance
        n_features: Number of top features to plot
    """
    logger.info("Creating feature distribution plots...")

    # Calculate feature variance
    feature_var = df.var(axis=0).sort_values(ascending=False)
    top_features = feature_var.head(n_features).index

    # Plot distribution of variances
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(np.log10(feature_var + 1e-10), bins=min(50, len(feature_var)), edgecolor='black', alpha=0.7)
    ax.set_xlabel('log10(Variance)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{data_type} - Feature Variance Distribution')

    plt.tight_layout()
    plot_path = output_dir / f'{data_type}_feature_variances.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved feature variance plot: {plot_path}")


def plot_missing_data(df: pd.DataFrame, output_dir: Path,
                     data_type: str, logger) -> None:
    """
    Plot missing data patterns.

    Args:
        df: Omics data DataFrame
        output_dir: Output directory
        data_type: Type of data (for plot title)
        logger: Logger instance
    """
    logger.info("Analyzing missing data patterns...")

    # Missing per sample
    missing_per_sample = df.isna().sum(axis=1)

    # Missing per feature
    missing_per_feature = df.isna().sum(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sample missing
    axes[0].hist(missing_per_sample, bins=min(50, len(missing_per_sample.unique())), edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of Missing Features')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title(f'{data_type} - Missing Data per Sample')

    # Feature missing
    axes[1].hist(missing_per_feature, bins=min(50, len(missing_per_feature.unique())), edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Number of Missing Samples')
    axes[1].set_ylabel('Number of Features')
    axes[1].set_title(f'{data_type} - Missing Data per Feature')

    plt.tight_layout()
    plot_path = output_dir / f'{data_type}_missing_data.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved missing data plot: {plot_path}")


def generate_html_report(output_dir: Path, data_type: str, metrics_processed: dict, metrics_orig: dict, logger) -> None:
    """
    Generate side-by-side HTML comparison report.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{data_type.capitalize()} Omics Report Comparison</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            .container {{ display: flex; }}
            .column {{ flex: 50%; padding: 10px; }}
            img {{ max-width: 100%; height: auto; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>{data_type.capitalize()} Omics Report Comparison</h1>

        <div class="container">
            <div class="column">
                <h2>Original Data</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Samples</td><td>{metrics_orig['n_samples']}</td></tr>
                    <tr><td>Features</td><td>{metrics_orig['n_features']}</td></tr>
                    <tr><td>Overall missing rate</td><td>{metrics_orig['missing_rate']:.4f}</td></tr>
                    <tr><td>Average sample mean</td><td>{metrics_orig['samples_mean']:.4f}</td></tr>
                    <tr><td>Average feature mean</td><td>{metrics_orig['features_mean']:.4f}</td></tr>
                </table>
                <h3>Sample Distributions</h3>
                <img src="{data_type}_orig_sample_distributions.png" alt="Sample Distributions (Orig)">
                <h3>Feature Distributions</h3>
                <img src="{data_type}_orig_feature_variances.png" alt="Feature Distributions (Orig)">
                <h3>Missing Data</h3>
                <img src="{data_type}_orig_missing_data.png" alt="Missing Data (Orig)">
            </div>
            <div class="column">
                <h2>Processed Data</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Samples</td><td>{metrics_processed['n_samples']}</td></tr>
                    <tr><td>Features</td><td>{metrics_processed['n_features']}</td></tr>
                    <tr><td>Overall missing rate</td><td>{metrics_processed['missing_rate']:.4f}</td></tr>
                    <tr><td>Average sample mean</td><td>{metrics_processed['samples_mean']:.4f}</td></tr>
                    <tr><td>Average feature mean</td><td>{metrics_processed['features_mean']:.4f}</td></tr>
                </table>
                <h3>Sample Distributions</h3>
                <img src="{data_type}_processed_sample_distributions.png" alt="Sample Distributions (Processed)">
                <h3>Feature Distributions</h3>
                <img src="{data_type}_processed_feature_variances.png" alt="Feature Distributions (Processed)">
                <h3>Missing Data</h3>
                <img src="{data_type}_processed_missing_data.png" alt="Missing Data (Processed)">
            </div>
        </div>
    </body>
    </html>
    """

    report_path = output_dir / f'{data_type}_comparison_report.html'
    with open(report_path, 'w') as f:
        f.write(html_content)

    logger.info(f"Saved HTML comparison report: {report_path}")

def generate_summary_report(df: pd.DataFrame, metrics: dict,
                           output_dir: Path, data_type: str, logger) -> None:
    """
    Generate text summary report.

    Args:
        df: Omics data DataFrame
        metrics: QC metrics dictionary
        output_dir: Output directory
        data_type: Type of data
        logger: Logger instance
    """
    report_path = output_dir / f'{data_type}_summary_report.txt'

    with open(report_path, 'w') as f:
        f.write(f"{data_type} Data Summary Report\n")
        f.write("=" * 80 + "\n\n")

        f.write("Data Dimensions:\n")
        f.write(f"  Samples: {metrics['n_samples']}\n")
        f.write(f"  Features: {metrics['n_features']}\n\n")

        f.write("Quality Control Metrics:\n")
        f.write(f"  Overall missing rate: {metrics['missing_rate']:.4f}\n")
        f.write(f"  Average sample mean: {metrics['samples_mean']:.4f}\n")
        f.write(f"  Sample mean std dev: {metrics['samples_std']:.4f}\n")
        f.write(f"  Average feature mean: {metrics['features_mean']:.4f}\n")
        f.write(f"  Feature mean std dev: {metrics['features_std']:.4f}\n\n")

        f.write("Missing Data:\n")
        f.write(f"  Avg missing per sample: {metrics['samples_missing_mean']:.2f}\n")
        f.write(f"  Avg missing per feature: {metrics['features_missing_mean']:.2f}\n\n")

        # Top variable features
        feature_var = df.var(axis=0).sort_values(ascending=False)
        f.write("Top 20 Most Variable Features:\n")
        for idx, (feat, var) in enumerate(feature_var.head(20).items(), 1):
            f.write(f"  {idx}. {feat}: {var:.6f}\n")

    logger.info(f"Saved summary report: {report_path}")


def main():
    """Main function to explore omics data."""
    parser = argparse.ArgumentParser(
        description='Explore omics data and generate QC reports'
    )
    parser.add_argument('--input-processed', type=str, required=True,
                       help='Input processed omics data file')
    parser.add_argument('--input-orig', type=str, required=True,
                       help='Input original omics data file')
    parser.add_argument('--data-type', type=str, default='omics',
                       choices=['methylation', 'expression', 'omics'],
                       help='Type of omics data')
    parser.add_argument('--transpose', action='store_true',
                       help='Transpose data (if features are in rows)')
    parser.add_argument('--output-dir', type=str,
                       default='output/omics_exploration',
                       help='Output directory for reports and plots')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Setup
    set_random_seed(args.seed)
    logger = setup_logger('explore_omics', level='INFO')

    output_dir = ensure_dir(args.output_dir)
    logger.info(f"Output directory: {output_dir}")

    # Load data
    logger.info(f"Loading {args.data_type} processed data from: {args.input_processed}")
    df_processed = load_omics_data(args.input_processed, transpose=args.transpose)
    log_dataframe_info(logger, df_processed, f"{args.data_type} processed data")

    logger.info(f"Loading {args.data_type} original data from: {args.input_orig}")
    df_orig = load_omics_data(args.input_orig, transpose=args.transpose)
    log_dataframe_info(logger, df_orig, f"{args.data_type} original data")

    # Calculate QC metrics
    metrics_processed = calculate_qc_metrics(df_processed, logger)
    metrics_orig = calculate_qc_metrics(df_orig, logger)

    # Generate summary report
    generate_summary_report(df_processed, metrics_processed, output_dir, f"{args.data_type}_processed", logger)
    generate_summary_report(df_orig, metrics_orig, output_dir, f"{args.data_type}_orig", logger)

    # Generate plots
    if not args.skip_plots:
        logger.info("Generating visualizations...")

        plot_sample_distributions(df_processed, output_dir, f"{args.data_type}_processed", logger)
        plot_feature_distributions(df_processed, output_dir, f"{args.data_type}_processed", logger)
        plot_missing_data(df_processed, output_dir, f"{args.data_type}_processed", logger)

        plot_sample_distributions(df_orig, output_dir, f"{args.data_type}_orig", logger)
        plot_feature_distributions(df_orig, output_dir, f"{args.data_type}_orig", logger)
        plot_missing_data(df_orig, output_dir, f"{args.data_type}_orig", logger)

    # Generate HTML report
    generate_html_report(output_dir, args.data_type, metrics_processed, metrics_orig, logger)

    logger.info(f"\n{args.data_type} data exploration complete!")
    logger.info(f"All outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
