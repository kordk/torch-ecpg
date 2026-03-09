#!/usr/bin/env python3

## kord.kober@ucsf.edu
## github.com/kordk/torch-ecpg

import os
import sys
import argparse
import logging
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

## DEFAULTS - Kennedy et al. BMC Genomics (2018) 19:476

#PVALCUTOFF=0.00001                   ## 10-5 is "suggestive" in Kennedy 2018
#PVALCUTOFF=0.00000000001             ## 10-11 is "significant" in Kennedy 2018
PVALCUTOFF = np.float32(0.000001)     ## 10-6 is our "exploratory" cutoff

## DISTAL >50Kb TSS
DISTAL_OFFSET = 50000

## CIS <50Kb TSS
CIS_OFFSET = 0
CIS_UPSTREAM_DISTANCE = 50000

## PROMOTER +/- 2500 bp TSS
PROMOTER_OFFSET = 0
PROMOTER_UPSTREAM_DISTANCE = 2500
PROMOTER_DOWNSTREAM_DISTANCE = 2500

logger = logging.getLogger(__name__)

def _normalize_chrom(chrom_str):
    """Normalize chromosome string by stripping 'chr' prefix for consistent comparison."""
    s = str(chrom_str).strip()
    if s.lower().startswith("chr"):
        return s[3:]
    return s

def _strip_id_version(identifier):
    """Strip version suffix from Ensembl-style IDs (e.g., ENSG00000000003.15_3 -> ENSG00000000003)."""
    return identifier.split('.')[0]

#### Read in the annotation file to a dictionary (supports BED6 and GFF) #######################################
def readAnnotationFileToDict(my_annotFile):
    my_lociH = {}

    with open(my_annotFile, "r") as fp:
        ng = 0 ## number of genes/loci processed
        nskip = 0 ## number of loci with missing data
        for line in fp:
            if line.startswith("#"):
                continue

            logger.debug(f"[readAnnotationFileToDict] line: {line.strip()}")
            dataA = line.strip('\n').split('\t')

            # Skip empty lines
            if not dataA or (len(dataA) == 1 and dataA[0] == ""):
                continue

            # Check if this is a header line for BED file
            if dataA[0] == "chrom":
                continue

            if dataA[0] == "NA":
                nskip += 1
                continue

            num_cols = len(dataA)

            if num_cols >= 9:
                # GFF/GTF format
                # Extract gene ID from attributes column
                # Supports featureCounts flat format (Geneid) and standard GTF (gene_id)
                attributes = dataA[8]
                gene_id = None
                for attr in attributes.split(";"):
                    attr = attr.strip()
                    if attr.startswith("Geneid "):
                        # featureCounts flat format: Geneid "ENSG00000000003"
                        gene_id = attr[len("Geneid "):].strip('"').strip()
                        break
                    elif attr.startswith("gene_id "):
                        # Standard GTF format: gene_id "ENSG00000000003"
                        gene_id = attr[len("gene_id "):].strip('"').strip()
                        break
                    elif attr.startswith("gene_id="):
                        # GFF3 format: gene_id=ENSG00000000003
                        gene_id = attr[len("gene_id="):].strip('"').strip()
                        break

                if not gene_id:
                    logger.error(f"[readAnnotationFileToDict] Missing gene ID in GFF/GTF line: {line.strip()}")
                    sys.exit(1)

                my_name = _strip_id_version(gene_id)

                # GFF is 1-based, inclusive. BED is 0-based, half-open
                chromStart = int(dataA[3]) - 1
                chromEnd = int(dataA[4])

                my_lociH[my_name] = {}
                my_lociH[my_name]["chrom"]      = _normalize_chrom(dataA[0])
                my_lociH[my_name]["chromStart"] = chromStart
                my_lociH[my_name]["chromEnd"]   = chromEnd
                my_lociH[my_name]["strand"]     = str(dataA[6])

            elif num_cols >= 6:
                # BED6 format
                my_name = _strip_id_version(dataA[3])
                my_lociH[my_name] = {}

                my_lociH[my_name]["chrom"]      = _normalize_chrom(dataA[0])
                my_lociH[my_name]["chromStart"] = int(dataA[1])
                my_lociH[my_name]["chromEnd"]   = int(dataA[2])
                my_lociH[my_name]["strand"]     = str(dataA[5])
            else:
                logger.warning(f"[readAnnotationFileToDict] Unsupported number of columns ({num_cols}) in line: {line.strip()}")
                continue

            if logger.isEnabledFor(logging.DEBUG):
                if my_name == "cg13191808":
                    logger.debug(f"[readAnnotationFileToDict] cg13191808: {my_lociH[my_name]}")

            ng += 1

    logger.info(f"[readAnnotationFileToDict] Skipped (NA) {nskip} loci from {my_annotFile}")
    logger.info(f"[readAnnotationFileToDict] Processed {len(my_lociH)} loci from {my_annotFile}")
    return my_lociH

#### Read through file and report on p-values #######################################
def reportPvalues(my_ecpgDataFile, pval_col, chunk_size):
    parquet_file = pq.ParquetFile(my_ecpgDataFile)

    nskip = 0
    total_read = 0
    p_lt_1e6 = 0
    p_lt_1e7 = 0
    p_lt_1e8 = 0
    p_lt_1e9 = 0

    for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=[pval_col]):
        df = batch.to_pandas()

        # Count missing
        missing_mask = df[pval_col].isna()
        nskip += missing_mask.sum()

        # Valid p-values
        valid_pvals = df.loc[~missing_mask, pval_col].values
        total_read += len(valid_pvals)

        p_lt_1e6 += np.sum(valid_pvals < 0.000001)
        p_lt_1e7 += np.sum(valid_pvals < 0.0000001)
        p_lt_1e8 += np.sum(valid_pvals < 0.00000001)
        p_lt_1e9 += np.sum(valid_pvals < 0.000000001)

    logger.info(f"[reportPvalues] P-values skipped (missing data): {nskip}")
    logger.info(f"[reportPvalues] P-values read: {total_read}")
    logger.info(f"[reportPvalues] P < 0.000001: {p_lt_1e6}")
    logger.info(f"[reportPvalues] P < 0.0000001: {p_lt_1e7}")
    logger.info(f"[reportPvalues] P < 0.00000001: {p_lt_1e8}")
    logger.info(f"[reportPvalues] P < 0.000000001: {p_lt_1e9}")

#### Assign region for each eCpG #######################################
def assignRegion(my_ecpgDataFile, gH, mH, pval_col, outFileName, chunk_size):
    my_typeCountH = {
        "trans": 0,
        "distal": 0,
        "cis": 0,
        "promoter": 0,
        "genebody": 0
    }

    nlp = 0 ## number loci processed
    ne = 0 ## number loci excluded
    npvalx = 0
    negx = 0
    nemt = 0
    npskip = 0
    assigned_total = 0

    parquet_file = pq.ParquetFile(my_ecpgDataFile)
    writer = None

    schema = pa.schema([
        ('mt_id', pa.string()),
        ('mt_chrom', pa.string()),
        ('mt_chromStart', pa.int64()),
        ('mt_strand', pa.string()),
        ('gt_id', pa.string()),
        ('gt_chrom', pa.string()),
        ('gt_chromStart', pa.int64()),
        ('gt_strand', pa.string()),
        ('region', pa.string())
    ])

    for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=['gt_id', 'mt_id', pval_col]):
        df = batch.to_pandas()
        my_eqtmA = []

        for index, row in df.iterrows():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[assignRegion] line: {row.to_dict()}")

            gt_id = _strip_id_version(str(row['gt_id']))
            mt_id = _strip_id_version(str(row['mt_id']))
            mt_p = row[pval_col]

            if pd.isna(mt_p):
                logger.info(f"[assignRegion] P-value missing. Excluding loci {gt_id} {mt_id} {mt_p}")
                npskip += 1
                continue

            nlp += 1

            if mt_p > 0.000001:
                logger.info(f"[assignRegion] P-value too large. Excluding loci {gt_id} {mt_id} {mt_p}")
                ne += 1
                npvalx += 1
                continue

            if mt_id not in mH:
                logger.info(f"[assignRegion] Annotation missing - methylation: {nlp} {mt_id}")
                ne += 1
                nemt += 1
                continue

            if gt_id not in gH:
                logger.info(f"[assignRegion] Annotation missing - gene expression: {nlp} {gt_id}")
                ne += 1
                negx += 1
                continue

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[assignRegion] {nlp} mt: {mt_id} {mH[mt_id]['chrom']} gx: {gt_id} {gH[gt_id]['chrom']}")

            if "chrom" not in mH[mt_id]:
                logger.info(f"[assignRegion] Annotation missing - methylation [chrom]: {nlp} {mt_id}")
                ne += 1
                nemt += 1
                continue

            if "chrom" not in gH[gt_id]:
                logger.info(f"[assignRegion] Annotation missing - gene expression [chrom]: {nlp} {gt_id}")
                ne += 1
                negx += 1
                continue

            if mH[mt_id]["chrom"] != gH[gt_id]["chrom"]:
                cpgA = {
                    'mt_id': mt_id,
                    'mt_chrom': mH[mt_id]["chrom"],
                    'mt_chromStart': mH[mt_id]["chromStart"],
                    'mt_strand': mH[mt_id]["strand"],
                    'gt_id': gt_id,
                    'gt_chrom': gH[gt_id]["chrom"],
                    'gt_chromStart': gH[gt_id]["chromStart"],
                    'gt_strand': gH[gt_id]["strand"],
                    'region': "TRANS"
                }
                my_typeCountH["trans"] += 1

                if logger.isEnabledFor(logging.DEBUG): logger.debug(f"[assignRegion] {nlp} {list(cpgA.values())}")
                my_eqtmA.append(cpgA)

                if logger.isEnabledFor(logging.DEBUG) and my_typeCountH["trans"] < 5:
                    logger.debug(f"[assignRegion] {nlp} {len(my_eqtmA)} {my_eqtmA}")

                continue   ## Move on to the next. CpGs cannot be TRANS && another region

            ## check for DISTAL - positive strand
            if gH[gt_id]["strand"] == "+":
                cpg_pos = mH[mt_id]["chromStart"]
                geneStart_pos = gH[gt_id]["chromStart"]
                regionRef_pos = geneStart_pos - DISTAL_OFFSET
                if cpg_pos < regionRef_pos:
                    cpgA = {
                        'mt_id': mt_id,
                        'mt_chrom': mH[mt_id]["chrom"],
                        'mt_chromStart': mH[mt_id]["chromStart"],
                        'mt_strand': mH[mt_id]["strand"],
                        'gt_id': gt_id,
                        'gt_chrom': gH[gt_id]["chrom"],
                        'gt_chromStart': gH[gt_id]["chromStart"],
                        'gt_strand': gH[gt_id]["strand"],
                        'region': "DISTAL"
                    }
                    my_typeCountH["distal"] += 1
                    if logger.isEnabledFor(logging.DEBUG): logger.debug(f"[assignRegion] {nlp} {list(cpgA.values())}")
                    my_eqtmA.append(cpgA)

            ## check for CIS - positive strand
            if gH[gt_id]["strand"] == "+":
                cpg_pos = mH[mt_id]["chromStart"]
                geneStart_pos = gH[gt_id]["chromStart"]
                regionRef_pos = geneStart_pos - CIS_OFFSET
                regionUpStreamRange = geneStart_pos - CIS_UPSTREAM_DISTANCE
                if (regionUpStreamRange < cpg_pos) and (cpg_pos < geneStart_pos):
                    cpgA = {
                        'mt_id': mt_id,
                        'mt_chrom': mH[mt_id]["chrom"],
                        'mt_chromStart': mH[mt_id]["chromStart"],
                        'mt_strand': mH[mt_id]["strand"],
                        'gt_id': gt_id,
                        'gt_chrom': gH[gt_id]["chrom"],
                        'gt_chromStart': gH[gt_id]["chromStart"],
                        'gt_strand': gH[gt_id]["strand"],
                        'region': "CIS"
                    }
                    my_typeCountH["cis"] += 1
                    if logger.isEnabledFor(logging.DEBUG): logger.debug(f"[assignRegion] {nlp} {list(cpgA.values())}")
                    my_eqtmA.append(cpgA)

            ## check for PROMOTER - positive strand
            if gH[gt_id]["strand"] == "+":
                cpg_pos = mH[mt_id]["chromStart"]
                geneStart_pos = gH[gt_id]["chromStart"]
                regionRef_pos = geneStart_pos - PROMOTER_OFFSET
                regionUpStreamRange = regionRef_pos - PROMOTER_UPSTREAM_DISTANCE
                regionDnStreamRange = regionRef_pos + PROMOTER_DOWNSTREAM_DISTANCE
                if (regionUpStreamRange < cpg_pos) and (cpg_pos < regionDnStreamRange):
                    cpgA = {
                        'mt_id': mt_id,
                        'mt_chrom': mH[mt_id]["chrom"],
                        'mt_chromStart': mH[mt_id]["chromStart"],
                        'mt_strand': mH[mt_id]["strand"],
                        'gt_id': gt_id,
                        'gt_chrom': gH[gt_id]["chrom"],
                        'gt_chromStart': gH[gt_id]["chromStart"],
                        'gt_strand': gH[gt_id]["strand"],
                        'region': "PROMOTER"
                    }
                    my_typeCountH["promoter"] += 1
                    if logger.isEnabledFor(logging.DEBUG): logger.debug(f"[assignRegion] {nlp} {list(cpgA.values())}")
                    my_eqtmA.append(cpgA)

            ## check for GENE BODY - positive strand
            if gH[gt_id]["strand"] == "+":
                cpg_pos = mH[mt_id]["chromStart"]
                geneStart_pos = gH[gt_id]["chromStart"]
                geneEnd_pos = gH[gt_id]["chromEnd"]
                if (geneStart_pos < cpg_pos) and (cpg_pos < geneEnd_pos):
                    cpgA = {
                        'mt_id': mt_id,
                        'mt_chrom': mH[mt_id]["chrom"],
                        'mt_chromStart': mH[mt_id]["chromStart"],
                        'mt_strand': mH[mt_id]["strand"],
                        'gt_id': gt_id,
                        'gt_chrom': gH[gt_id]["chrom"],
                        'gt_chromStart': gH[gt_id]["chromStart"],
                        'gt_strand': gH[gt_id]["strand"],
                        'region': "GENEBODY"
                    }
                    my_typeCountH["genebody"] += 1
                    if logger.isEnabledFor(logging.DEBUG): logger.debug(f"[assignRegion] {nlp} {list(cpgA.values())}")
                    my_eqtmA.append(cpgA)

            ## check for DISTAL - negative strand
            if gH[gt_id]["strand"] == "-":
                cpg_pos = mH[mt_id]["chromStart"]
                tss = max(gH[gt_id]["chromStart"], gH[gt_id]["chromEnd"])
                regionRef_pos = tss + DISTAL_OFFSET
                if regionRef_pos < cpg_pos:
                    cpgA = {
                        'mt_id': mt_id,
                        'mt_chrom': mH[mt_id]["chrom"],
                        'mt_chromStart': mH[mt_id]["chromStart"],
                        'mt_strand': mH[mt_id]["strand"],
                        'gt_id': gt_id,
                        'gt_chrom': gH[gt_id]["chrom"],
                        'gt_chromStart': gH[gt_id]["chromStart"],
                        'gt_strand': gH[gt_id]["strand"],
                        'region': "DISTAL"
                    }
                    my_typeCountH["distal"] += 1
                    if logger.isEnabledFor(logging.DEBUG): logger.debug(f"[assignRegion] {nlp} {list(cpgA.values())}")
                    my_eqtmA.append(cpgA)

            ## check for CIS - negative strand
            if gH[gt_id]["strand"] == "-":
                cpg_pos = mH[mt_id]["chromStart"]
                tss = max(gH[gt_id]["chromStart"], gH[gt_id]["chromEnd"])
                regionUpStreamRange = tss + CIS_UPSTREAM_DISTANCE
                if (tss < cpg_pos) and (cpg_pos < regionUpStreamRange):
                    cpgA = {
                        'mt_id': mt_id,
                        'mt_chrom': mH[mt_id]["chrom"],
                        'mt_chromStart': mH[mt_id]["chromStart"],
                        'mt_strand': mH[mt_id]["strand"],
                        'gt_id': gt_id,
                        'gt_chrom': gH[gt_id]["chrom"],
                        'gt_chromStart': gH[gt_id]["chromStart"],
                        'gt_strand': gH[gt_id]["strand"],
                        'region': "CIS"
                    }
                    my_typeCountH["cis"] += 1
                    if logger.isEnabledFor(logging.DEBUG): logger.debug(f"[assignRegion] {nlp} {list(cpgA.values())}")
                    my_eqtmA.append(cpgA)

            ## check for PROMOTER - negative strand
            if gH[gt_id]["strand"] == "-":
                cpg_pos = mH[mt_id]["chromStart"]
                tss = max(gH[gt_id]["chromStart"], gH[gt_id]["chromEnd"])
                regionRef_pos = tss + PROMOTER_OFFSET
                regionDnStreamRange = regionRef_pos - PROMOTER_UPSTREAM_DISTANCE
                regionUpStreamRange = regionRef_pos + PROMOTER_DOWNSTREAM_DISTANCE
                if (regionDnStreamRange < cpg_pos) and (cpg_pos < regionUpStreamRange):
                    cpgA = {
                        'mt_id': mt_id,
                        'mt_chrom': mH[mt_id]["chrom"],
                        'mt_chromStart': mH[mt_id]["chromStart"],
                        'mt_strand': mH[mt_id]["strand"],
                        'gt_id': gt_id,
                        'gt_chrom': gH[gt_id]["chrom"],
                        'gt_chromStart': gH[gt_id]["chromStart"],
                        'gt_strand': gH[gt_id]["strand"],
                        'region': "PROMOTER"
                    }
                    my_typeCountH["promoter"] += 1
                    if logger.isEnabledFor(logging.DEBUG): logger.debug(f"[assignRegion] {nlp} {list(cpgA.values())}")
                    my_eqtmA.append(cpgA)

            ## check for GENE BODY - negative strand
            if gH[gt_id]["strand"] == "-":
                cpg_pos = mH[mt_id]["chromStart"]
                gene_low = min(gH[gt_id]["chromStart"], gH[gt_id]["chromEnd"])
                gene_high = max(gH[gt_id]["chromStart"], gH[gt_id]["chromEnd"])
                if (gene_low < cpg_pos) and (cpg_pos < gene_high):
                    cpgA = {
                        'mt_id': mt_id,
                        'mt_chrom': mH[mt_id]["chrom"],
                        'mt_chromStart': mH[mt_id]["chromStart"],
                        'mt_strand': mH[mt_id]["strand"],
                        'gt_id': gt_id,
                        'gt_chrom': gH[gt_id]["chrom"],
                        'gt_chromStart': gH[gt_id]["chromStart"],
                        'gt_strand': gH[gt_id]["strand"],
                        'region': "GENEBODY"
                    }
                    my_typeCountH["genebody"] += 1
                    if logger.isEnabledFor(logging.DEBUG): logger.debug(f"[assignRegion] {nlp} {list(cpgA.values())}")
                    my_eqtmA.append(cpgA)

        if my_eqtmA:
            assigned_total += len(my_eqtmA)
            out_df = pd.DataFrame(my_eqtmA)
            # Make sure types are correct for Parquet
            out_df['mt_chromStart'] = out_df['mt_chromStart'].astype(np.int64)
            out_df['gt_chromStart'] = out_df['gt_chromStart'].astype(np.int64)
            table = pa.Table.from_pandas(out_df, schema=schema, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(outFileName, schema)
            writer.write_table(table)

    if writer is not None:
        writer.close()
    else:
        # If no results matched, we still want to create an empty parquet file
        writer = pq.ParquetWriter(outFileName, schema)
        writer.close()

    logger.info(f"[assignRegion] eCpgs Processed: {nlp} Assigned: {assigned_total} Excluded (any): {ne}")
    logger.info(f"[assignRegion] eCpgs Excluded: p-value filter: {npvalx} p-value missing: {npskip} gx annotation: {negx} mt annotation: {nemt}")
    logger.info(f"[assignRegion] eCpgs Counts by Region: {my_typeCountH}")
    return assigned_total

def main():
    parser = argparse.ArgumentParser(description="assignRegionToEcpg_parquet.py - assign a region class to eCpGs")
    parser.add_argument("-d", "--ecpgDataFile", required=True, help="<tecpg eQTM output parquet file>")
    parser.add_argument("-g", "--geneAnnotFile", required=True, help="<gene annotation file>")
    parser.add_argument("-m", "--methylAnnotFile", required=True, help="<methylation annotation file>")
    parser.add_argument("-o", "--outFileName", required=True, help="<outfile name parquet>")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Number of rows to process per chunk. Default is 100,000.")
    parser.add_argument("-D", "--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(message)s')

    ecpgDataFile = args.ecpgDataFile
    geneAnnotFile = args.geneAnnotFile
    methylAnnotFile = args.methylAnnotFile
    outFileName = args.outFileName
    chunk_size = args.chunk_size

    if not os.path.exists(ecpgDataFile):
        logger.error(f"[MAIN] eCpG data file not found: {ecpgDataFile}")
        sys.exit(202)
    logger.info(f"[MAIN] eCpG datafile: {ecpgDataFile}")

    if not os.path.exists(geneAnnotFile):
        logger.error(f"[MAIN] gene annotation file not found: {geneAnnotFile}")
        sys.exit(203)
    logger.info(f"[MAIN] gene annotation file: {geneAnnotFile}")

    if not os.path.exists(methylAnnotFile):
        logger.error(f"[MAIN] methylation annotation file not found: {methylAnnotFile}")
        sys.exit(204)
    logger.info(f"[MAIN] methylation annotation file: {methylAnnotFile}")

    logger.info(f"[MAIN] output file name: {outFileName}")


    ## Read in the gene annotation file to a dictionary
    geneH = readAnnotationFileToDict(geneAnnotFile)

    ## Read in the methylation annotation file to a dictionary
    methylH = readAnnotationFileToDict(methylAnnotFile)

    ## Determine which p-value column to use
    try:
        schema = pq.read_schema(ecpgDataFile)
        columns = schema.names
        if "precise_mt_p" in columns:
            pval_col = "precise_mt_p"
        elif "mt_p" in columns:
            pval_col = "mt_p"
        else:
            logger.error("[MAIN] Neither 'precise_mt_p' nor 'mt_p' found in input Parquet file.")
            sys.exit(1)
        logger.info(f"[MAIN] Using p-value column: {pval_col}")
    except Exception as e:
        logger.error(f"[MAIN] Error reading schema from {ecpgDataFile}: {e}")
        sys.exit(1)

    ## summarize the p-values
    logger.info(f"[MAIN] Using default p-value cutoff of {PVALCUTOFF}")
    reportPvalues(ecpgDataFile, pval_col, chunk_size)

    ## Annotate the pairs
    assigned_total = assignRegion(ecpgDataFile, geneH, methylH, pval_col, outFileName, chunk_size)

    logger.info(f"[MAIN] Saving annotated data to: {outFileName}")
    logger.info("[MAIN] Done.")

if __name__ == "__main__":
    main()
