#!/usr/bin/env Rscript

# estimateCellProportions.R
#
# Estimates blood cell-type proportions with EpiDISH (RPC) and appends them to
# the covariate matrix.
#
# IMPORTANT (compositional fix): EpiDISH RPC returns fractions that sum to ~1.0
# across cell types per sample. A set of predictors that sums to a constant is
# exactly collinear with the model intercept, making the regression design
# rank-deficient. Downstream, that smears one shared coefficient across the
# cell-type columns and makes their individual coefficients / IG attributions
# uninterpretable. To break the closure, this script drops one reference cell
# type before appending the proportions to the covariates, and (by default)
# applies a rank-based inverse-normal transform (INT) to the remaining columns
# so the covariates share a common scale. The raw fractions are still used for
# the heatmaps and the closure check; only the appended copy is transformed.
#
# Usage:
#   Rscript estimateCellProportions.R <methylation.csv> <covariates.csv> <out.csv> [cohort] [flags]
#
# Optional flags (use --key=value form; order-independent; positional args
# unaffected, so existing pipeline calls keep working):
#   --reference=<CellType>   Drop this specific cell type as the reference
#                            instead of the most-abundant one (recommended when
#                            cross-cohort reproducibility matters, so the
#                            reference is fixed rather than data-dependent).
#   --no-int                 Skip the inverse-normal transform; append the
#                            reference-dropped raw proportions as-is.
#   --int                    Force the inverse-normal transform (default ON).
#   --min-nonzero-frac=<f>   Drop near-degenerate cell types whose fraction of
#                            samples with a nonzero proportion is below <f>
#                            (default 0.5). Catches cells the RPC reference
#                            panel cannot resolve (e.g. eosinophils -> mostly
#                            zeros), which INT would otherwise collapse to a
#                            near-constant column. Set to 0 to disable.

# ---------------------------------------------------------------------------
# Argument parsing: separate --flags from positional args so the historical
# positional interface (<meth> <cov> <out> [cohort]) is preserved verbatim.
# ---------------------------------------------------------------------------
raw_args <- commandArgs(trailingOnly = TRUE)
is_flag  <- grepl("^--", raw_args)
flags    <- raw_args[is_flag]
pos      <- raw_args[!is_flag]

if (length(pos) < 3 || length(pos) > 4) {
  stop(paste0("Usage: Rscript estimateCellProportions.R <methylation_file.csv> ",
              "<covariates_file.csv> <output_file.csv> [cohort_name] ",
              "[--reference=<CellType>] [--no-int|--int] [--min-nonzero-frac=<f>]"))
}

meth_file   <- pos[1]
cov_file    <- pos[2]
out_file    <- pos[3]
cohort_name <- if (length(pos) == 4) pos[4] else ""

# Flag defaults: reference drop is always applied (it is the fix); the only
# choice is WHICH cell type. INT defaults ON to match prior cohort preprocessing.
ref_cell         <- NA_character_   # NA => auto-select most abundant
do_int           <- TRUE
min_nonzero_frac <- 0.5             # drop cells nonzero in fewer than this fraction of samples
for (f in flags) {
  if (grepl("^--reference=", f)) {
    ref_cell <- sub("^--reference=", "", f)
  } else if (f == "--no-int") {
    do_int <- FALSE
  } else if (f == "--int") {
    do_int <- TRUE
  } else if (grepl("^--min-nonzero-frac=", f)) {
    min_nonzero_frac <- as.numeric(sub("^--min-nonzero-frac=", "", f))
    if (is.na(min_nonzero_frac) || min_nonzero_frac < 0 || min_nonzero_frac > 1) {
      stop("--min-nonzero-frac must be a number in [0, 1].")
    }
  } else {
    warning(paste("Ignoring unknown flag:", f))
  }
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Rank-based inverse-normal transform (Blom, c = 3/8), NA-safe, column-wise.
inverse_normal_transform <- function(x, c = 3/8) {
  out  <- rep(NA_real_, length(x))
  mask <- !is.na(x)
  n    <- sum(mask)
  if (n == 0) return(out)
  r <- rank(x[mask], ties.method = "average")
  out[mask] <- qnorm((r - c) / (n - 2 * c + 1))
  out
}

# Closure check on the RAW fractions: confirms the compositional constraint that
# motivates the reference drop. This is the empirical gate we agreed on.
report_closure <- function(cells) {
  rs <- rowSums(cells, na.rm = TRUE)
  cat(sprintf("Cell-fraction row sums: mean=%.4f sd=%.4g min=%.4f max=%.4f\n",
              mean(rs), sd(rs), min(rs), max(rs)))
  if (abs(mean(rs) - 1) < 0.01 && sd(rs) < 0.01) {
    cat("Closure confirmed: fractions sum to ~1.0 (compositional). The cell-type ",
        "columns are collinear with the intercept; a reference-cell drop is ",
        "required and will be applied.\n", sep = "")
  } else {
    cat("WARNING: row sums are not ~1.0. Verify these are EpiDISH fractions ",
        "before relying on the reference-drop rationale; the collinearity fix ",
        "assumes a closed composition.\n", sep = "")
  }
  invisible(rs)
}

# Drop near-degenerate cell types: those whose proportion is nonzero in fewer
# than `min_nonzero_frac` of samples. The RPC reference panel sometimes fails to
# resolve a cell type (e.g. eosinophils -> mostly exact zeros). Such a column is
# near-constant; after INT it collapses to a single tie-rank value for most
# samples, carries no usable signal, and is weakly collinear with the intercept.
# Removing it is cleaner than appending a degenerate covariate.
drop_degenerate_cells <- function(cells, min_nonzero_frac = 0.5, eps = 1e-6) {
  if (min_nonzero_frac <= 0) {
    cat("Near-degenerate cell drop disabled (--min-nonzero-frac=0).\n")
    return(cells)
  }
  nz_frac <- vapply(cells, function(x) mean(abs(x) > eps, na.rm = TRUE), numeric(1))
  degenerate <- names(nz_frac)[nz_frac < min_nonzero_frac]
  if (length(degenerate) > 0) {
    for (d in degenerate) {
      cat(sprintf("Dropping near-degenerate cell type '%s': nonzero in only %.1f%% of ",
                  d, 100 * nz_frac[[d]]))
      cat(sprintf("samples (> %g), below the %.0f%% threshold.\n",
                  eps, 100 * min_nonzero_frac))
    }
    cells <- cells[, setdiff(colnames(cells), degenerate), drop = FALSE]
  } else {
    cat("No near-degenerate cell types detected.\n")
  }
  cells
}

# Drop near-degenerate cells, then one reference cell type (breaks closure),
# then optionally INT-transform the remaining columns. Returns the processed
# cell block to append.
process_cells <- function(cells, ref_cell, do_int, min_nonzero_frac = 0.5) {
  cells <- drop_degenerate_cells(cells, min_nonzero_frac = min_nonzero_frac)
  if (ncol(cells) < 2) {
    stop(sprintf(paste0("After dropping near-degenerate cell types, only %d column(s) ",
                        "remain; need at least 2 (one is dropped as the reference)."),
                 ncol(cells)))
  }

  means <- colMeans(cells, na.rm = TRUE)
  # If a fixed reference was requested but it was removed as degenerate, fall
  # back to the most-abundant remaining type (a reference drop is still required
  # to break the closure among the surviving columns).
  if (!is.na(ref_cell) && ref_cell != "" && !(ref_cell %in% colnames(cells))) {
    cat(sprintf("Requested --reference=%s is no longer present (dropped as ",
                ref_cell))
    cat("near-degenerate); falling back to most-abundant remaining type.\n")
    ref_cell <- NA_character_
  }

  if (is.na(ref_cell) || ref_cell == "") {
    ref_cell <- names(means)[which.max(means)]
    cat(sprintf("Reference cell type (auto, most abundant): %s (mean=%.4f)\n",
                ref_cell, max(means)))
  } else {
    if (!(ref_cell %in% colnames(cells))) {
      stop(sprintf("Requested --reference=%s not found among cell types: %s",
                   ref_cell, paste(colnames(cells), collapse = ", ")))
    }
    cat(sprintf("Reference cell type (fixed): %s (mean=%.4f)\n",
                ref_cell, means[[ref_cell]]))
  }

  kept <- setdiff(colnames(cells), ref_cell)
  out  <- cells[, kept, drop = FALSE]
  cat(sprintf("Dropped reference '%s'; %d cell-type column(s) remain: %s\n",
              ref_cell, ncol(out), paste(kept, collapse = ", ")))

  if (do_int) {
    cat("Applying rank-based inverse-normal transform (Blom c=3/8), column-wise...\n")
    int_df <- as.data.frame(lapply(out, inverse_normal_transform),
                            check.names = FALSE)
    rownames(int_df) <- rownames(out)
    out <- int_df
  } else {
    cat("INT disabled (--no-int): appending reference-dropped raw proportions.\n")
  }
  out
}

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    cat("Installing BiocManager...\n")
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
}
if (!requireNamespace("EpiDISH", quietly = TRUE)) {
    cat("Installing EpiDISH...\n")
    BiocManager::install("EpiDISH", ask = FALSE, update = FALSE)
}
if (!requireNamespace("pheatmap", quietly = TRUE)) {
    cat("Installing pheatmap...\n")
    install.packages("pheatmap", repos = "https://cloud.r-project.org")
}

library(EpiDISH)
library(pheatmap)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
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

# Generate Heatmaps (on RAW, interpretable proportions)
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

# ---------------------------------------------------------------------------
# Compositional fix: closure check (raw) -> reference drop -> optional INT.
# Skipped for MESA, where cell columns are omitted from the output entirely.
# ---------------------------------------------------------------------------
cat("\n--- Cell-fraction composition handling ---\n")
report_closure(cell_fractions)

if (tolower(cohort_name) == "mesa") {
    cat("MESA cohort: cell fractions are omitted from the output, so no ",
        "reference-drop / INT is applied.\n", sep = "")
    cells_for_merge <- cell_fractions
} else {
    cells_for_merge <- process_cells(cell_fractions, ref_cell, do_int, min_nonzero_frac)
    cat("\nSummary report of appended cell columns (after processing):\n")
    print(summary(cells_for_merge))
    cat("\n")
}

cat("Cell columns first 5 row names (sample IDs) before merge:\n")
print(head(rownames(cells_for_merge), 5))

# Merge based on row names
# EpiDISH returns sample IDs as row names in cell_fractions
cat("Merging cell columns with covariates...\n")
merged_cov <- merge(cov_matrix, cells_for_merge, by="row.names", all.x=FALSE) # only keep samples present in filtered fractions

# Restore row names and remove the temporary 'Row.names' column
rownames(merged_cov) <- merged_cov$Row.names
merged_cov$Row.names <- NULL

if (tolower(cohort_name) == "mesa") {
    cat("MESA cohort detected. Omitting cell fraction columns from output.\n")
    # Remove all columns that came from cell_fractions
    cols_to_remove <- colnames(cells_for_merge)
    merged_cov <- merged_cov[, !(colnames(merged_cov) %in% cols_to_remove), drop=FALSE]
}

cat(paste("Saving updated covariates to", out_file, "\n"))
write.csv(merged_cov, out_file, row.names=TRUE)
cat("Done.\n")
