# Generate Illumina probe blacklist (450K by default; EPICv1 optional)
#
# Produces a CSV of probes that should be excluded from downstream analysis:
#   - SNP-affected probes   (variant within -1..2 bp of the CpG, MAF > 0.05)
#   - Cross-reactive probes (cross-hybridising to other genomic locations)
#   - Sex-chromosome probes (chrX, chrY)
#
# Probe lists come from the DMRcatedata ExperimentHub resources used internally
# by DMRcate::rmSNPandCH(). IMPORTANT: per the DMRcatedata vignette these three
# resources ALREADY span both 450K and EPICv1 probe IDs -- they are not
# EPIC-only:
#   EH3130 snpsall    - 450K + EPICv1 probes confounded by a SNP/indel
#   EH3129 crosshyb   - 450K + EPICv1 cross-hybridising probes
#   EH3131 XY.probes  - 450K + EPICv1 probes targeting sex chromosomes
#
# The --array argument selects which array manifest to SCOPE the result to, so
# the emitted list contains only probes that exist on the array in use. Probes
# from the other array are inert when filtering, but they inflate the list size
# and distort any audit that uses it as a denominator.
#
# Supersedes generateEpicProbeBlacklist_v2.R, which derived sex-chromosome
# probes from the EPIC manifest alone and therefore MISSED 450K probes with no
# EPIC counterpart (EPIC covers >90% of 450K CpGs, so ~10% were unreachable).

suppressPackageStartupMessages({
  library(ExperimentHub)
})

# ---- arguments -------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  hit <- grep(paste0("^", flag, "="), args, value = TRUE)
  if (length(hit) == 0) return(default)
  sub(paste0("^", flag, "="), "", hit[1])
}
array_type <- tolower(get_arg("--array", "450k"))
out_file   <- get_arg("--out", "probes_blacklist.csv")

if (!array_type %in% c("450k", "epic", "both")) {
  stop("--array must be one of: 450k (default), epic, both")
}
cat("Array scope:", array_type, "\n")

if (any(args == "--selftest")) {
  # Exercise the reason-labelling and CSV-writing path with synthetic input,
  # without touching ExperimentHub or the annotation packages. Confirms the R
  # environment and the output contract in seconds, before the real slow run.
  snp_probes <- c("cg01", "cg02"); crosshyb_probes <- c("cg02", "cg03")
  xy_probes <- c("cg03", "cg04")
  all_probes <- unique(c(snp_probes, crosshyb_probes, xy_probes))
  in_snp <- all_probes %in% snp_probes
  in_ch  <- all_probes %in% crosshyb_probes
  in_xy  <- all_probes %in% xy_probes
  reason <- paste(ifelse(in_snp, "SNP", ""), ifelse(in_ch, "CROSSREACTIVE", ""),
                  ifelse(in_xy, "SEXCHROM", ""), sep = ";")
  reason <- gsub("^;+|;+$", "", gsub(";{2,}", ";", reason))
  out <- data.frame(Probe_ID = all_probes, Reason = reason, stringsAsFactors = FALSE)
  out <- out[order(out$Probe_ID), ]
  write.csv(out, out_file, row.names = FALSE)
  stopifnot(nrow(out) == 4,
            !any(out$Reason == ""),
            names(out)[1] == "Probe_ID",
            out$Reason[out$Probe_ID == "cg02"] == "SNP;CROSSREACTIVE",
            out$Reason[out$Probe_ID == "cg03"] == "CROSSREACTIVE;SEXCHROM")
  cat("SELFTEST PASS: reason logic and CSV contract OK ->", out_file, "\n")
  quit(status = 0)
}

# ---- 1. array manifest, used only to scope the result ----------------------
manifest_probes <- function(which_array) {
  if (which_array == "450k") {
    suppressPackageStartupMessages(
      library(IlluminaHumanMethylation450kanno.ilmn12.hg19))
    ann <- minfi::getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)
  } else {
    suppressPackageStartupMessages(
      library(IlluminaHumanMethylationEPICanno.ilm10b4.hg19))
    ann <- minfi::getAnnotation(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)
  }
  rownames(ann)
}

if (array_type == "both") {
  on_array <- union(manifest_probes("450k"), manifest_probes("epic"))
} else {
  on_array <- manifest_probes(array_type)
}
cat(" - probes on array manifest:", length(on_array), "\n")

# ---- 2. DMRcatedata exclusion lists (already 450K + EPICv1) ----------------
eh <- ExperimentHub()

snpsall <- eh[["EH3130"]]
snp_distances <- as.integer(snpsall$distances)
snp_probes <- as.character(
  snpsall$probe[(snp_distances >= -1) & (snp_distances <= 2) & (snpsall$mafs > 0.05)]
)

crosshyb_probes <- as.character(eh[["EH3129"]])

# Sex-chromosome probes from DMRcate rather than a single array's manifest, so
# 450K-only XY probes are not silently missed.
xy_probes <- as.character(eh[["EH3131"]])

# ---- 3. scope to the array, then label by reason ---------------------------
snp_probes      <- intersect(unique(snp_probes),      on_array)
crosshyb_probes <- intersect(unique(crosshyb_probes), on_array)
xy_probes       <- intersect(unique(xy_probes),       on_array)

cat(" - SNP-affected   :", length(snp_probes), "\n")
cat(" - cross-reactive :", length(crosshyb_probes), "\n")
cat(" - sex chromosome :", length(xy_probes), "\n")

# A probe may qualify under several criteria; keep every reason, joined by ';',
# so the list stays one row per probe while remaining decomposable.
all_probes <- unique(c(snp_probes, crosshyb_probes, xy_probes))

# Vectorised membership: three hashed `%in%` calls over the whole vector.
# Do NOT rewrite this as a per-probe loop. `p %in% vec` inside vapply is a
# linear scan per probe: at real list sizes (~82k probes against ~94k list
# entries) that is ~7.7 billion comparisons and the script appears to hang.
in_snp <- all_probes %in% snp_probes
in_ch  <- all_probes %in% crosshyb_probes
in_xy  <- all_probes %in% xy_probes

reason <- paste(ifelse(in_snp, "SNP", ""),
                ifelse(in_ch,  "CROSSREACTIVE", ""),
                ifelse(in_xy,  "SEXCHROM", ""),
                sep = ";")
reason <- gsub("^;+|;+$", "", gsub(";{2,}", ";", reason))

if (any(reason == "")) {
  stop("Internal error: probe(s) with no exclusion reason; set logic is wrong.")
}

# Probe_ID MUST remain the FIRST column: tools/exclude_blacklisted_probes.py
# reads the blacklist by position (iloc[:, 0]).
out <- data.frame(Probe_ID = all_probes, Reason = reason,
                  stringsAsFactors = FALSE)
out <- out[order(out$Probe_ID), ]

write.csv(out, out_file, row.names = FALSE)

cat("Success: blacklist generated.\n")
cat(" - array           :", array_type, "\n")
cat(" - probes excluded :", nrow(out), "\n")
cat(" - written to      :", out_file, "\n")
