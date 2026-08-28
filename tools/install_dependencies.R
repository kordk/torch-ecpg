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
  "pheatmap"                                       # estimateCellProportions.R
)

# Bioconductor packages. The comment on each line names the tool that loads it,
# so this list can be checked against the scripts when either changes.
bioc_packages <- c(
  "EpiDISH",                                       # estimateCellProportions.R
  "sva",                                           # generateSvaFactors.R
  "ExperimentHub",                                 # generateProbeBlacklist.R,
                                                   # generateEpicProbeBlacklist_v2.R
  # generateProbeBlacklist.R calls minfi::getAnnotation() directly. It arrives
  # as a dependency of the annotation packages below, but is named here because
  # the script uses it by name.
  "minfi",                                         # generateProbeBlacklist.R
  # Array manifests. generateProbeBlacklist.R defaults to --array=450k, so the
  # 450K manifest is required for a default `pipelinePre.sh` run; `epic` and
  # `both` need the EPICv1 manifest as well.
  "IlluminaHumanMethylation450kanno.ilmn12.hg19",  # generateProbeBlacklist.R (450k, default)
  "IlluminaHumanMethylationEPICanno.ilm10b4.hg19"  # generateProbeBlacklist.R (epic),
                                                   # generateEpicProbeBlacklist_v2.R
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

# Verify, rather than assume. install.packages() and BiocManager::install()
# report failures as warnings and return normally, so without this check the
# script exits 0 after a failed install and the missing package only surfaces
# later as a `library()` error inside a pipeline stage.
cat("Verifying installed packages...\n")
missing <- Filter(
  function(pkg) !requireNamespace(pkg, quietly = TRUE),
  c(cran_packages, bioc_packages)
)

if (length(missing) > 0) {
  cat(sprintf("FAILED: %d package(s) could not be loaded:\n", length(missing)))
  for (pkg in missing) cat(sprintf("  - %s\n", pkg))
  quit(status = 1)
}

cat("All dependencies installed successfully.\n")
