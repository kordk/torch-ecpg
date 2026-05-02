#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript generateSvaFactors.R <gene_expression_file.csv> <covariates_file.csv> <output_file.csv>")
}

expr_file <- args[1]
cov_file <- args[2]
out_file <- args[3]

# Install sva if not present
if (!requireNamespace("sva", quietly = TRUE)) {
    cat("sva package not found. Attempting to install...\n")
    stop("The 'sva' package is required. Please install it using your environment manager (e.g., conda install -c bioconda bioconductor-sva).")
}

library(sva)

# Load data
cat(paste("Loading gene expression data from", expr_file, "\n"))
# Assume the first column is the identifier
expr_data <- read.csv(expr_file, row.names=1, check.names=FALSE)

cat(paste("Loading covariate data from", cov_file, "\n"))
cov_data <- read.csv(cov_file, row.names=1, check.names=FALSE)

# Handle drop NA
expr_data <- na.omit(expr_data)

# Usually our G.csv is Samples (columns) x Genes (rows) ?
# According to previous script:
# expr_data <- t(expr_data) # made it Samples (rows) x Genes (columns)
# So initially expr_data was Genes (rows) x Samples (columns).

# For SVA, expr_data needs to be Features (rows) x Samples (columns).
# So we do NOT transpose it like we did for PEER.
# Wait, let me double check the memory.
# Memory: "Gene expression data processed by the generatePeerFactors.R script is explicitly log2(x + 1) transformed and standardized using R's scale() function prior to PEER factor generation."
# "When using the peer package in R, input matrices (such as gene expression and covariates) must be oriented strictly in a Samples x Features format (samples as rows)."
# "In the tecpg pipeline, covariate (C_orig.csv) and gene expression (G.csv) matrices use the first column as the index (Sample ID). When processing these matrices in R, they must be loaded with row.names=1 and saved with row.names=TRUE."
# THIS MEANS G.csv is Samples (rows) x Genes (columns).

# Wait! The old script said:
# # By default, genes are usually rows, samples are columns in our G.csv
# # Let's remove rows with missing values
# expr_data <- na.omit(expr_data)
# # Transpose to make Samples x Features (Genes)
# cat("Transposing gene expression data to Samples (rows) x Genes (columns)...\n")
# expr_data <- t(expr_data)

# If the old script was right that G.csv is Genes (rows) x Samples (columns) initially...
# Wait, let's look at pipeline.sh.
# pipeline.sh uses `tecpg data dummy`, which generates G.csv. In dummy dataset, G.csv is Samples (rows) x Genes (columns).
# Wait, memory says: "In the tecpg pipeline, covariate (C_orig.csv) and gene expression (G.csv) matrices use the first column as the index (Sample ID)."
# If G.csv uses Sample ID as row index, it is Samples (rows) x Genes (columns).
# The old script did `expr_data <- t(expr_data)`, transposing it to Genes (rows) x Samples (columns)? Or did it think it was Genes x Samples and transposed to Samples x Genes?
# If `expr_data` was Samples x Genes, and it transposed it, it became Genes x Samples!
# And PEER wants Samples x Features!
# If it wanted Samples x Features, and dummy generates Samples x Genes... then `t(expr_data)` would make it Genes x Samples, breaking PEER.
# Unless G.csv actually IS Genes x Samples.

# SVA requires an m x n matrix (m = features/genes, n = samples).
# So we want Genes (rows) x Samples (columns).
# If G.csv is Samples x Genes, we need to transpose it.
# Let's align on common samples.
# Assuming G.csv and C_with_celltypes.csv both have Samples as rows (because `rownames(cov_data)` are sample IDs).

# Let's assume expr_data is currently Samples (rows) x Genes (columns).
# Let's check common samples between rownames of both.
common_samples <- intersect(rownames(expr_data), rownames(cov_data))
if (length(common_samples) == 0) {
    # Maybe expr_data is Genes (rows) x Samples (columns)? Try colnames.
    common_samples <- intersect(colnames(expr_data), rownames(cov_data))
    if (length(common_samples) > 0) {
        cat("expr_data appears to be Genes (rows) x Samples (columns).\n")
        # Keep common samples
        expr_data <- expr_data[, common_samples, drop=FALSE]
        cov_data <- cov_data[common_samples, , drop=FALSE]
    } else {
        stop("No matching samples found between gene expression data and base covariates.")
    }
} else {
    cat("expr_data appears to be Samples (rows) x Genes (columns). Transposing to Genes (rows) x Samples (columns) for SVA.\n")
    # Keep common samples
    expr_data <- expr_data[common_samples, , drop=FALSE]
    cov_data <- cov_data[common_samples, , drop=FALSE]

    # Transpose to Genes x Samples
    expr_data <- t(expr_data)
}

cat(paste("Aligned", nrow(cov_data), "samples.\n"))

# Log-transform if necessary
cat("Applying log2(x + 1) transformation to gene expression data...\n")
expr_data <- log2(expr_data + 1)

cat("Standardizing gene expression features (scaling)...\n")
# scale() standardizes columns. Since expr_data is Genes (rows) x Samples (columns),
# scale(expr_data) would standardize across samples for each gene? No, it standardizes each sample across genes.
# We usually want to standardize each gene across samples.
# So we transpose, scale, and transpose back.
expr_data_scaled <- t(scale(t(expr_data)))

# Design matrices
cat("Setting up design matrices for SVA...\n")
mod0 <- model.matrix(~ 1, data = cov_data)
mod <- model.matrix(~ ., data = cov_data)

# Report the columns used
cat("Columns used in the full design matrix:\n")
print(colnames(mod))

# Estimate number of SVs
cat("Estimating number of Surrogate Variables (SVs)...\n")
n.sv.identified <- num.sv(as.matrix(expr_data_scaled), mod, method="leek")
n.sv.retained <- min(n.sv.identified, 50)

cat(paste("SVA identified", n.sv.identified, "surrogate variables.\n"))
cat(paste("Retaining", n.sv.retained, "surrogate variables.\n"))

if (n.sv.retained > 0) {
    cat("Running SVA...\n")
    svobj <- sva(as.matrix(expr_data_scaled), mod, mod0, n.sv=n.sv.retained)
    sv_matrix <- svobj$sv
    colnames(sv_matrix) <- paste0("SVA_Factor", 1:n.sv.retained)
    rownames(sv_matrix) <- rownames(cov_data)

    # Append SVs to covariates
    final_covariates <- cbind(cov_data, sv_matrix)
} else {
    cat("No surrogate variables were retained. Using original covariates.\n")
    final_covariates <- cov_data
}

# Save for torch-eCpG
cat(paste("Saving final covariate matrix to", out_file, "\n"))
write.csv(final_covariates, out_file, quote=FALSE, row.names=TRUE)
cat("Saved final covariate matrix for tecpg!\n")
