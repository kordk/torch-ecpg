# Generate EPIC Probe Blacklist
#
# Produces epic_probes_blacklist.csv containing Illumina EPIC (v1) probes that
# should be excluded from downstream analysis:
#   - SNP-affected probes   (distance <= 2 bp, MAF > 0.05)
#   - Cross-reactive probes (cross-hybridising to other genomic locations)
#   - Sex-chromosome probes (chrX, chrY)
#
# Probe lists are obtained directly from the DMRcatedata ExperimentHub
# resources (EH3130, EH3129) and the EPIC annotation package, avoiding the
# dimension-handling issues in DMRcate::rmSNPandCH().

# Install necessary packages if not already present
#if (!require("BiocManager", quietly = TRUE))
#    install.packages("BiocManager")
#BiocManager::install(c("IlluminaHumanMethylationEPICanno.ilm10b4.hg19",
#                        "ExperimentHub"))

library(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)
library(ExperimentHub)

# 1. Get the EPIC manifest
data("IlluminaHumanMethylationEPICanno.ilm10b4.hg19")
full_ann <- getAnnotation(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)

# 2. Extract X and Y probes directly (Sex Chromosome Scrubbing)
# This aligns with the Taylor/MESA study protocol.
xy_probes <- rownames(full_ann[full_ann$chr %in% c("chrX", "chrY"), ])

# 3. Identify problematic probes using DMRcatedata resources from ExperimentHub.
#    These are the same authoritative lists used internally by
#    DMRcate::rmSNPandCH() for EPICv1 data.
eh <- ExperimentHub()

# 3a. SNP-affected probes to exclude: within 2 bp of a SNP with MAF > 5 %
snpsall <- eh[["EH3130"]]
snp_distances <- as.integer(snpsall$distances)
snp_probes <- as.character(
  snpsall$probe[(snp_distances >= -1) & (snp_distances <= 2) & (snpsall$mafs > 0.05)]
)

# 3b. Cross-reactive (cross-hybridising) probes
crosshyb <- eh[["EH3129"]]
crosshyb_probes <- as.character(crosshyb)

# 4. Combine all blacklisted probes
final_blacklist <- unique(c(snp_probes, crosshyb_probes, xy_probes))

# 5. Export for prepare_data.py
write.csv(data.frame(Probe_ID = final_blacklist),
          "epic_probes_blacklist.csv",
          row.names = FALSE)

cat("Success: Blacklist generated.\n")
cat(" - Probes to exclude:", length(final_blacklist), "\n")
