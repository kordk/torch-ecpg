#!/usr/bin/env python3

"""
preprocessPcaCovariates.py
Standalone Python preprocessing script to generate a combined covariate matrix.
Calculates Principal Component Analysis (PCA) on gene expression data and appends
the top principal components to a set of known fixed effects.
"""

import os
import argparse
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate a combined covariate matrix by appending PCA components from gene expression data.")

    parser.add_argument("-g", "--gene-file", required=True, help="Path to the gene expression data file (CSV). Default format: loci in rows, samples in columns.")
    parser.add_argument("-c", "--covar-file", required=True, help="Path to the base covariates file (CSV). Expected format: samples in rows.")
    parser.add_argument("-o", "--output-file", required=True, help="Output filepath for the combined covariate CSV.")
    parser.add_argument("-n", "--n-components", type=int, default=5, help="Number of top principal components to extract. Default is 5.")
    parser.add_argument("--transpose", action="store_true", help="Transpose the gene data before processing. Use this if your input gene data already has samples as rows. By default, the script assumes samples are columns and will transpose it automatically.")
    parser.add_argument("-D", "--debug", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(message)s')

    logger.debug(f"Arguments parsed: {args}")

    if not os.path.exists(args.gene_file):
        logger.error(f"Gene expression file not found: {args.gene_file}")
        return
    if not os.path.exists(args.covar_file):
        logger.error(f"Base covariates file not found: {args.covar_file}")
        return

    logger.info(f"Loading gene expression data from {args.gene_file}")
    # Load gene expression data. Assume index is the first column.
    G = pd.read_csv(args.gene_file, index_col=0)

    # Drop loci with missing data
    # G format: Default is loci in rows, samples in columns.
    # If we need to drop loci, we drop rows.
    initial_loci = G.shape[0] if not args.transpose else G.shape[1]

    if args.transpose:
        # If transposed, samples are rows, loci are columns. We drop columns with missing data.
        G = G.dropna(axis=1)
        final_loci = G.shape[1]
    else:
        # Default: loci are rows, samples are columns. We drop rows with missing data.
        G = G.dropna(axis=0)
        final_loci = G.shape[0]

    dropped_loci = initial_loci - final_loci
    logger.info(f"Dropped {dropped_loci} loci with missing data. Retaining {final_loci} loci.")

    if not args.transpose:
        logger.info("Transposing gene expression data (assuming samples are columns).")
        G = G.transpose()
    else:
        logger.info("Skipping transposition (--transpose flag used).")

    logger.info(f"Loading base covariates from {args.covar_file}")
    # Load base covariates. Assume index is the first column (sample IDs).
    C = pd.read_csv(args.covar_file, index_col=0)

    # Ensure sample IDs align correctly using inner join on index
    initial_samples_G = G.shape[0]
    initial_samples_C = C.shape[0]

    # The inner join on index
    G.index = G.index.astype(str)
    C.index = C.index.astype(str)

    merged_indices = G.index.intersection(C.index)

    if len(merged_indices) == 0:
        logger.error("No matching samples found between gene expression data and base covariates.")
        return

    G_aligned = G.loc[merged_indices]
    C_aligned = C.loc[merged_indices]

    logger.info(f"Aligned {len(merged_indices)} samples (G had {initial_samples_G}, C had {initial_samples_C}).")

    logger.info("Standardizing molecular features...")
    scaler = StandardScaler()
    G_scaled = scaler.fit_transform(G_aligned)

    n_components = min(args.n_components, G_scaled.shape[0], G_scaled.shape[1])
    if n_components < args.n_components:
        logger.warning(f"Requested {args.n_components} components, but only {n_components} can be extracted.")

    logger.info(f"Running PCA to extract top {n_components} components...")
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(G_scaled)

    explained_variance = sum(pca.explained_variance_ratio_) * 100
    logger.info(f"Explained variance by top {n_components} PCs: {explained_variance:.2f}%")

    # Create column names PC1, PC2, ...
    pc_columns = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(pca_features, index=merged_indices, columns=pc_columns)

    logger.info("Merging PCA components with base covariates...")
    C_combined = pd.concat([C_aligned, pca_df], axis=1)

    logger.info(f"Saving combined covariates to {args.output_file}")
    C_combined.to_csv(args.output_file)

    # Summary Report
    logger.info("-" * 40)
    logger.info("SUMMARY REPORT")
    logger.info("-" * 40)
    logger.info(f"Initial Loci:   {initial_loci}")
    logger.info(f"Dropped Loci:   {dropped_loci}")
    logger.info(f"Retained Loci:  {final_loci}")
    logger.info(f"Initial Samples (Gene):   {initial_samples_G}")
    logger.info(f"Initial Samples (Covars): {initial_samples_C}")
    logger.info(f"Aligned Samples:          {len(merged_indices)}")
    logger.info(f"Extracted PCs:            {n_components}")
    logger.info(f"Total Explained Variance: {explained_variance:.2f}%")
    logger.info(f"Output saved to:          {args.output_file}")
    logger.info("-" * 40)

if __name__ == "__main__":
    main()
