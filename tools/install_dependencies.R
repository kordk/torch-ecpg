#!/usr/bin/env Rscript

# install_dependencies.R
# This script installs all necessary R packages required by the tecpg pipeline tools.

# Set CRAN mirror if not set to avoid interactive prompts
local({
  r <- getOption("repos")
  if (identical(r["CRAN"], "@CRAN@") || is.null(r["CRAN"]) || is.na(r["CRAN"])) {
    r["CRAN"] <- "https://cloud.r-project.org"
    options(repos = r)
  }
})

# Install BiocManager if not already installed
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  cat("Installing BiocManager...\n")
  install.packages("BiocManager")
}

# CRAN packages
cran_packages <- c(
  "pheatmap"
)

# Bioconductor packages
bioc_packages <- c(
  "EpiDISH",
  "sva",
  "IlluminaHumanMethylationEPICanno.ilm10b4.hg19",
  "ExperimentHub"
)

cat("Installing CRAN dependencies...\n")
for (pkg in cran_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    install.packages(pkg)
  } else {
    cat(sprintf("%s is already installed.\n", pkg))
  }
}

cat("Installing Bioconductor dependencies...\n")
for (pkg in bioc_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    BiocManager::install(pkg, update = FALSE, ask = FALSE)
  } else {
    cat(sprintf("%s is already installed.\n", pkg))
  }
}

cat("All dependencies installed successfully.\n")
