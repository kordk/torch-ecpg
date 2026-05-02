#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript generatePeerFactors.R <gene_expression_file.csv> <covariates_file.csv> <output_file.csv>")
}

expr_file <- args[1]
cov_file <- args[2]
out_file <- args[3]

# Install peer if not present
if (!requireNamespace("peer", quietly = TRUE)) {
    cat("peer package not found. Attempting to install...\n")
    # BiocManager might not have r-peer directly, but we can instruct the user
    # or rely on conda to install it.
    stop("The 'peer' package is required. Please install it using your environment manager (e.g., conda install -c bioconda r-peer).")
}

library(peer)

# Load data
cat(paste("Loading gene expression data from", expr_file, "\n"))
# Assume the first column is the identifier
expr_data <- read.csv(expr_file, row.names=1, check.names=FALSE)

cat(paste("Loading covariate data from", cov_file, "\n"))
cov_data <- read.csv(cov_file, row.names=1, check.names=FALSE)

# Handle drop NA and transposition
# By default, genes are usually rows, samples are columns in our G.csv
# Let's remove rows with missing values
expr_data <- na.omit(expr_data)

# Transpose to make Samples x Features (Genes)
cat("Transposing gene expression data to Samples (rows) x Genes (columns)...\n")
expr_data <- t(expr_data)

# Ensure sample IDs align correctly
common_samples <- intersect(rownames(expr_data), rownames(cov_data))
if (length(common_samples) == 0) {
    stop("No matching samples found between gene expression data and base covariates.")
}
cat(paste("Aligned", length(common_samples), "samples.\n"))

expr_data <- expr_data[common_samples, , drop=FALSE]
cov_data <- cov_data[common_samples, , drop=FALSE]

# Apply log2(x + 1) transformation for normalization
cat("Applying log2(x + 1) transformation to gene expression data...\n")
expr_data <- log2(expr_data + 1)

# Standardize the gene expression data (Normalization/Scaling)
cat("Standardizing gene expression features (scaling)...\n")
expr_data_scaled <- scale(expr_data)

# Initialize PEER
cat("Initializing PEER model...\n")
model <- PEER()

# Set data and parameters
PEER_setPhenoMean(model, as.matrix(expr_data_scaled))
PEER_setCovariates(model, as.matrix(cov_data))

# Extract 30 hidden factors
K <- 30
PEER_setNk(model, K)

# Run Inference
cat("Running PEER update... this may take a few minutes depending on gene count.\n")
PEER_update(model)

# Extract the final matrix
# PEER_getX returns a matrix where the first columns are our known covariates,
# and the remaining K columns are the inferred hidden factors.
final_covariates <- PEER_getX(model)
rownames(final_covariates) <- rownames(expr_data_scaled)

# Name the columns properly
cov_names <- colnames(cov_data)
peer_names <- paste0("PEER_Factor", 1:K)
colnames(final_covariates) <- c(cov_names, peer_names)

# Save for torch-eCpG
cat(paste("Saving final covariate matrix to", out_file, "\n"))
write.csv(final_covariates, out_file, quote=FALSE, row.names=TRUE)
cat("Saved final covariate matrix for tecpg!\n")
