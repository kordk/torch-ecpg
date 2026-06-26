#!/usr/bin/env python3

import os
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import patsy

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Residualize matrix on covariates and extract PCA components.")

    parser.add_argument("-i", "--input-matrix", required=True, help="Path to input matrix (M.csv or G.csv). Expected format: loci in rows, samples in columns by default.")
    parser.add_argument("-c", "--covar-file", required=True, help="Path to the base covariates file (CSV). Expected format: samples in rows.")
    parser.add_argument("-o", "--output-file", required=True, help="Output filepath for the PCA CSV.")
    parser.add_argument("-p", "--prefix", required=True, help="Prefix for the PCA columns (e.g. Exp_PC, Meth_PC).")
    parser.add_argument("-n", "--n-components", type=int, default=5, help="Number of top principal components to extract. Default is 5.")
    parser.add_argument("--transpose", action="store_true", help="Transpose input matrix before processing. Use if input has samples as rows.")
    parser.add_argument("-D", "--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--log2-transform", action="store_true", help="Apply log2(x + 1) transform to the input matrix before residualization.")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(message)s')

    if not os.path.exists(args.input_matrix):
        logger.error(f"Input matrix file not found: {args.input_matrix}")
        return
    if not os.path.exists(args.covar_file):
        logger.error(f"Base covariates file not found: {args.covar_file}")
        return

    logger.info(f"Loading input matrix from {args.input_matrix}")
    # Read without index_col first, then cast to string, then set index to preserve leading zeros
    M_raw = pd.read_csv(args.input_matrix, dtype={0: str})
    M_raw.set_index(M_raw.columns[0], inplace=True)

    # Drop loci with missing data
    _resid_before = M_raw.shape
    if args.transpose:
        M_raw = M_raw.dropna(axis=1)
    else:
        M_raw = M_raw.dropna(axis=0)
    logger.info(
        f"Drop site residualize_pca.dropna_missing: dropped loci with missing "
        f"data (axis={'columns' if args.transpose else 'rows'}): "
        f"{_resid_before} -> {M_raw.shape}"
    )

    if not args.transpose:
        M = M_raw.transpose()
    else:
        M = M_raw

    if args.log2_transform:
        logger.info("Applying log2(x + 1) transformation to the input matrix.")
        float_M = M.astype(float)
        if (float_M <= -1).any().any():
            logger.error("Negative values <= -1 found in input matrix before log2 transform. Cannot apply log2(x + 1) to values <= -1.")
            import sys
            sys.exit(1)
        M = np.log2(float_M + 1)

    logger.info(f"Loading base covariates from {args.covar_file}")
    C = pd.read_csv(args.covar_file, dtype={0: str})
    C.set_index(C.columns[0], inplace=True)

    # Ensure sample IDs align correctly using inner join on index
    M.index = M.index.astype(str)
    C.index = C.index.astype(str)
    merged_indices = M.index.intersection(C.index)

    if len(merged_indices) == 0:
        logger.error("No matching samples found between input matrix and base covariates.")
        return

    M_aligned = M.loc[merged_indices]
    C_aligned = C.loc[merged_indices]

    logger.info(f"Aligned {len(merged_indices)} samples.")

    # Generate design matrix for covariates using patsy
    # All columns in C_aligned are treated as fixed effects.
    # Convert column names to valid python identifiers for patsy, or use Q()
    formula = "~ " + " + ".join([f"Q('{c}')" for c in C_aligned.columns])
    logger.info(f"Using formula: {formula}")

    design_matrix = patsy.dmatrix(formula, data=C_aligned, return_type='dataframe')

    # Residualization
    logger.info("Running residualization (Linear Regression)...")
    reg = LinearRegression(fit_intercept=False) # patsy adds intercept
    reg.fit(design_matrix, M_aligned)
    preds = reg.predict(design_matrix)
    residuals = M_aligned - preds

    logger.info("Running PCA on residuals...")
    n_components = min(args.n_components, residuals.shape[0], residuals.shape[1])
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(residuals)

    explained_variance = sum(pca.explained_variance_ratio_) * 100
    logger.info(f"Explained variance by top {n_components} PCs: {explained_variance:.2f}%")

    pc_columns = [f"{args.prefix}{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(pca_features, index=merged_indices, columns=pc_columns)

    logger.info(f"Saving PCA components to {args.output_file}")
    pca_df.to_csv(args.output_file)

if __name__ == "__main__":
    main()
