#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 3 || length(args) > 4) {
  stop("Usage: Rscript estimateCellProportions.R <methylation_file.csv> <covariates_file.csv> <output_file.csv> [cohort_name]")
}

meth_file <- args[1]
cov_file <- args[2]
out_file <- args[3]
cohort_name <- if (length(args) == 4) args[4] else ""

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

# Check if data are M-values (contains negative values)
if (any(beta_matrix < 0, na.rm = TRUE)) {
    cat("Negative values detected. Assuming input data are M-values.\n")
    cat("Converting M-values to Beta-values for EpiDISH...\n")
    beta_matrix <- (2^beta_matrix) / (2^beta_matrix + 1)
} else {
    cat("Input data appears to be Beta-values. No conversion needed.\n")
}

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

# Print a brief summary report of cell type proportions before filtering
cat("\nSummary report of overall cell type proportions (Before Filtering):\n")
print(summary(cell_fractions))
cat("\n")

# MESA cohort specific filtering
if (tolower(cohort_name) == "mesa") {
    cat("MESA cohort detected. Applying strict pre-filtering (Mono >= 0.80)...\n")
    if ("Mono" %in% colnames(cell_fractions)) {
        excluded_samples <- rownames(cell_fractions)[cell_fractions$Mono < 0.80]
        if (length(excluded_samples) > 0) {
            cat(paste("Excluding", length(excluded_samples), "samples with Mono < 80%:\n"))
            print(excluded_samples)
            cell_fractions <- cell_fractions[cell_fractions$Mono >= 0.80, , drop=FALSE]
            cat("\nSummary report of overall cell type proportions (After Filtering):\n")
            print(summary(cell_fractions))
            cat("\n")
        } else {
            cat("No samples met the exclusion criteria (Mono < 80%).\n")
        }
    } else {
        cat("Warning: 'Mono' column not found in cell fractions. Skipping filtering.\n")
    }
}

# Generate Heatmaps
out_basename <- tools::file_path_sans_ext(out_file)
heatmap_fully_clustered_file <- paste0(out_basename, "_heatmap_fully_clustered.png")
heatmap_celltype_clustered_file <- paste0(out_basename, "_heatmap_celltype_clustered.png")

cat("Generating heatmaps...\n")
n_samples <- nrow(cell_fractions)

# Dynamically calculate cellheight to avoid Cairo max limit
# Cairo generally crashes around 32767 pixels (approx 109 inches at 300dpi).
# We want total height = cellheight * n_samples to be safe. Let's aim for max 30000 pixels.
# pheatmap height unit is pt (1/72 inch). Actually, cellheight is in pt.
# 32000 pt is very large. Let's cap cellheight such that it never exceeds a very safe value.
# Also if cellheight is too small (< 1), pheatmap might just look bad or throw warnings.
# In those cases, we omit cellheight and let it automatically determine based on fixed overall height.
if (n_samples > 1000) {
    # Drop cellheight/cellwidth, rely on automatic scaling to fit the page
    pheatmap_args <- list(
        mat = as.matrix(cell_fractions),
        cluster_cols = TRUE
    )
    pheatmap_args_fully <- c(pheatmap_args, list(cluster_rows = TRUE, filename = heatmap_fully_clustered_file, main = "Cell Type Proportions (Fully Clustered)"))
    pheatmap_args_celltype <- c(pheatmap_args, list(cluster_rows = FALSE, filename = heatmap_celltype_clustered_file, main = "Cell Type Proportions (Cell Type Clustered)"))

    do.call(pheatmap, pheatmap_args_fully)
    cat(paste("Saved fully clustered heatmap to", heatmap_fully_clustered_file, "\n"))

    do.call(pheatmap, pheatmap_args_celltype)
    cat(paste("Saved cell type clustered heatmap to", heatmap_celltype_clustered_file, "\n"))

} else {
    # For smaller sample sizes, use the explicit cell sizes
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
}

cat("Cell fractions first 5 row names (sample IDs) before merge:\n")
print(head(rownames(cell_fractions), 5))

# Merge based on row names
# EpiDISH returns sample IDs as row names in cell_fractions
cat("Merging cell fractions with covariates...\n")
merged_cov <- merge(cov_matrix, cell_fractions, by="row.names", all.x=FALSE) # only keep samples present in filtered fractions

# Restore row names and remove the temporary 'Row.names' column
rownames(merged_cov) <- merged_cov$Row.names
merged_cov$Row.names <- NULL

if (tolower(cohort_name) == "mesa") {
    cat("MESA cohort detected. Omitting cell fraction columns from output.\n")
    # Remove all columns that came from cell_fractions
    cols_to_remove <- colnames(cell_fractions)
    merged_cov <- merged_cov[, !(colnames(merged_cov) %in% cols_to_remove), drop=FALSE]
}

cat(paste("Saving updated covariates to", out_file, "\n"))
write.csv(merged_cov, out_file, row.names=TRUE)
cat("Done.\n")
