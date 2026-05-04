#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript estimateCellProportions.R <methylation_file.csv> <covariates_file.csv> <output_file.csv>")
}

meth_file <- args[1]
cov_file <- args[2]
out_file <- args[3]

# Install BiocManager if not present
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    cat("Installing BiocManager...\n")
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
}

# Install EpiDISH if not present
if (!requireNamespace("EpiDISH", quietly = TRUE)) {
    cat("Installing EpiDISH...\n")
    BiocManager::install("EpiDISH", ask = FALSE, update = FALSE)
}

# Install pheatmap if not present
if (!requireNamespace("pheatmap", quietly = TRUE)) {
    cat("Installing pheatmap...\n")
    install.packages("pheatmap", repos = "https://cloud.r-project.org")
}

library(EpiDISH)
library(pheatmap)

# Load data
cat(paste("Loading methylation data from", meth_file, "\n"))
# Assuming CpGs as rows, samples as columns for EpiDISH based on standard input
beta_matrix <- read.csv(meth_file, row.names=1, check.names=FALSE)
cat("Methylation matrix first 5 column names (sample IDs):\n")
print(head(colnames(beta_matrix), 5))

cat(paste("Loading covariate data from", cov_file, "\n"))
# Assuming samples as rows for covariate matrix
cov_matrix <- read.csv(cov_file, row.names=1, check.names=FALSE)
cat("Covariate matrix first 5 row names (sample IDs):\n")
print(head(rownames(cov_matrix), 5))

# Load reference panel
data(centDHSbloodDMC.m)

# Run EpiDISH
cat("Running EpiDISH (RPC method)...\n")
out.l <- epidish(beta.m = as.matrix(beta_matrix), ref.m = centDHSbloodDMC.m, method = "RPC")

# Extract fractions
cell_fractions <- as.data.frame(out.l$estF)

# Print a brief summary report of cell type proportions
cat("\nSummary report of overall cell type proportions:\n")
print(summary(cell_fractions))
cat("\n")

# Generate Heatmaps
out_basename <- tools::file_path_sans_ext(out_file)
heatmap_fully_clustered_file <- paste0(out_basename, "_heatmap_fully_clustered.png")
heatmap_celltype_clustered_file <- paste0(out_basename, "_heatmap_celltype_clustered.png")

cat("Generating heatmaps...\n")
# Calculate an appropriate height based on the number of samples (rows)
# Minimum 400px height or 10px per row, width can be fixed or scaling
n_samples <- nrow(cell_fractions)

# Fully clustered heatmap
pheatmap(
    as.matrix(cell_fractions),
    cluster_rows = TRUE,
    cluster_cols = TRUE,
    cellheight = 10,
    cellwidth = 40,
    filename = heatmap_fully_clustered_file,
    main = "Cell Type Proportions (Fully Clustered)"
)
cat(paste("Saved fully clustered heatmap to", heatmap_fully_clustered_file, "\n"))

# Cell type clustered heatmap (samples in original order)
pheatmap(
    as.matrix(cell_fractions),
    cluster_rows = FALSE,
    cluster_cols = TRUE,
    cellheight = 10,
    cellwidth = 40,
    filename = heatmap_celltype_clustered_file,
    main = "Cell Type Proportions (Cell Type Clustered)"
)
cat(paste("Saved cell type clustered heatmap to", heatmap_celltype_clustered_file, "\n"))

cat("Cell fractions first 5 row names (sample IDs) before merge:\n")
print(head(rownames(cell_fractions), 5))

# Merge based on row names
# EpiDISH returns sample IDs as row names in cell_fractions
cat("Merging cell fractions with covariates...\n")
merged_cov <- merge(cov_matrix, cell_fractions, by="row.names", all.x=TRUE)

# Restore row names and remove the temporary 'Row.names' column
rownames(merged_cov) <- merged_cov$Row.names
merged_cov$Row.names <- NULL

cat(paste("Saving updated covariates to", out_file, "\n"))
write.csv(merged_cov, out_file, row.names=TRUE)
cat("Done.\n")
