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
                # GFF format
                # Only process 'gene' features to get full boundaries
                if dataA[2] != "gene":
                    continue

                # Extract Geneid or gene_id from attributes column
                attributes = dataA[8]
                gene_id = None
                for attr in attributes.split(";"):
                    attr = attr.strip()
                    if attr.startswith("Geneid "):
                        # e.g., Geneid "ENSG00000000003"
                        gene_id = attr[len("Geneid "):].strip('"')
                        break
                    elif attr.startswith("gene_id "):
                        # e.g., gene_id "ENSG00000223972.4"
                        gene_id = attr[len("gene_id "):].strip('"')
                        break

                if not gene_id:
                    logger.error(f"[readAnnotationFileToDict] Missing Geneid or gene_id in GFF line: {line.strip()}")
                    sys.exit(1)

                my_name = gene_id.split('.')[0]

                # GFF is 1-based, inclusive. BED is 0-based, half-open
                chromStart = int(dataA[3]) - 1
                chromEnd = int(dataA[4])

                my_lociH[my_name] = {}
                my_lociH[my_name]["chrom"]      = str(dataA[0])
                my_lociH[my_name]["chromStart"] = chromStart
                my_lociH[my_name]["chromEnd"]   = chromEnd
                my_lociH[my_name]["strand"]     = str(dataA[6])

            elif num_cols >= 6:
                # BED6 format
                my_name = dataA[3]

                # Read + validate coordinates FIRST, before creating the entry.
                chrom_val = dataA[0].strip()
                start_val = dataA[1].strip()
                end_val   = dataA[2].strip()

                # NA / unmapped probe: no usable coordinates. Skip WITHOUT
                # creating a dict entry, so downstream lookup treats the probe
                # as absent -> NULL coordinates -> excluded for missing coords.
                if (not chrom_val or chrom_val.upper() in ("NA", "<NA>", "NAN")
                        or not start_val or not end_val):
                    nskip += 1
                    continue

                my_lociH[my_name] = {}
                my_lociH[my_name]["chrom"]      = str(chrom_val)
                my_lociH[my_name]["chromStart"] = int(start_val)
                my_lociH[my_name]["chromEnd"]   = int(end_val)
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
        if df.index.names != [None]:
            df = df.reset_index()

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
        "distal5": 0,
        "cis5": 0,
        "promoter": 0,
        "genebody": 0,
        "cis3": 0,
        "distal3": 0
    }

    nlp = 0 ## number loci processed
    ne = 0 ## number loci excluded
    npvalx = 0
    negx = 0
    nemt = 0
    npskip = 0
    assigned_total = 0

    missing_gene_ids = {}
    missing_meth_ids = {}

    parquet_file = pq.ParquetFile(my_ecpgDataFile)
    writer = None

    input_schema = parquet_file.schema.to_arrow_schema()

    # Create new fields for the annotation columns
    new_fields = [
        pa.field('mt_chrom', pa.string()),
        pa.field('mt_chromStart', pa.int64()),
        pa.field('mt_strand', pa.string()),
        pa.field('gt_chrom', pa.string()),
        pa.field('gt_chromStart', pa.int64()),
        pa.field('gt_strand', pa.string()),
        pa.field('region', pa.string())
    ]

    # Append the new fields to the input schema
    schema = pa.schema(list(input_schema) + new_fields)

    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()
        if df.index.names != [None]:
            df = df.reset_index()
        my_eqtmA = []

        for index, row in df.iterrows():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[assignRegion] line: {row.to_dict()}")

            gt_id = str(row['gt_id'])
            mt_id = str(row['mt_id'])
            mt_p = row[pval_col]

            # Start with the original row as a dict
            base_row = row.to_dict()

            nlp += 1

            # Determine if we have annotation info
            has_mt_annot = mt_id in mH and "chrom" in mH[mt_id]
            has_gt_annot = gt_id in gH and "chrom" in gH[gt_id]

            if not has_mt_annot:
                if mt_id not in mH:
                    if mt_id not in missing_meth_ids:
                        missing_meth_ids[mt_id] = "missing_id"
                else:
                    if mt_id not in missing_meth_ids:
                        missing_meth_ids[mt_id] = "missing_chrom"
                nemt += 1
            if not has_gt_annot:
                if gt_id not in gH:
                    if gt_id not in missing_gene_ids:
                        missing_gene_ids[gt_id] = "missing_id"
                else:
                    if gt_id not in missing_gene_ids:
                        missing_gene_ids[gt_id] = "missing_chrom"
                negx += 1

            if not has_mt_annot or not has_gt_annot:
                # Missing annotation -> just append the row with nulls for the new columns
                cpgA = dict(base_row)
                cpgA.update({
                    'mt_chrom': None,
                    'mt_chromStart': None,
                    'mt_strand': None,
                    'gt_chrom': None,
                    'gt_chromStart': None,
                    'gt_strand': None,
                    'region': None
                })
                my_eqtmA.append(cpgA)
                ne += 1
                continue

            # We have annotations
            mt_chrom = mH[mt_id]["chrom"]
            mt_chromStart = mH[mt_id]["chromStart"]
            mt_strand = mH[mt_id]["strand"]

            gt_chrom = gH[gt_id]["chrom"]
            gt_chromStart = gH[gt_id]["chromStart"]
            gt_strand = gH[gt_id]["strand"]

            annot_base = {
                'mt_chrom': mt_chrom,
                'mt_chromStart': mt_chromStart,
                'mt_strand': mt_strand,
                'gt_chrom': gt_chrom,
                'gt_chromStart': gt_chromStart,
                'gt_strand': gt_strand
            }

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[assignRegion] {nlp} mt: {mt_id} {mt_chrom} gx: {gt_id} {gt_chrom}")

            if mt_chrom != gt_chrom:
                cpgA = dict(base_row)
                cpgA.update(annot_base)
                cpgA['region'] = "TRANS"
                my_typeCountH["trans"] += 1

                if logger.isEnabledFor(logging.DEBUG): logger.debug(f"[assignRegion] {nlp} {list(cpgA.values())}")
                my_eqtmA.append(cpgA)

                if logger.isEnabledFor(logging.DEBUG) and my_typeCountH["trans"] < 5:
                    logger.debug(f"[assignRegion] {nlp} {len(my_eqtmA)} {my_eqtmA}")

                continue   ## Move on to the next. CpGs cannot be TRANS && another region

            # Keep track of regions assigned for this pair
            assigned_regions = 0

            ## Region assignment - positive strand
            if gt_strand == "+":
                gt_chromEnd = gH[gt_id]["chromEnd"]
                if mt_chromStart < gt_chromStart - 50000:
                    region = 'DISTAL5'
                    my_typeCountH["distal5"] += 1
                elif gt_chromStart - 50000 <= mt_chromStart < gt_chromStart - 2500:
                    region = 'CIS5'
                    my_typeCountH["cis5"] += 1
                elif gt_chromStart - 2500 <= mt_chromStart <= gt_chromStart + 2500:
                    region = 'PROMOTER'
                    my_typeCountH["promoter"] += 1
                elif gt_chromStart + 2500 < mt_chromStart < gt_chromEnd:
                    region = 'GENEBODY'
                    my_typeCountH["genebody"] += 1
                elif gt_chromEnd <= mt_chromStart <= gt_chromEnd + 50000:
                    region = 'CIS3'
                    my_typeCountH["cis3"] += 1
                elif mt_chromStart > gt_chromEnd + 50000:
                    region = 'DISTAL3'
                    my_typeCountH["distal3"] += 1
                else:
                    region = None

                if region:
                    cpgA = dict(base_row)
                    cpgA.update(annot_base)
                    cpgA['region'] = region
                    if logger.isEnabledFor(logging.DEBUG): logger.debug(f"[assignRegion] {nlp} {list(cpgA.values())}")
                    my_eqtmA.append(cpgA)
                    assigned_regions += 1

            ## Region assignment - negative strand
            if gt_strand == "-":
                gt_chromEnd = gH[gt_id]["chromEnd"]
                if mt_chromStart > gt_chromEnd + 50000:
                    region = 'DISTAL5'
                    my_typeCountH["distal5"] += 1
                elif gt_chromEnd + 2500 < mt_chromStart <= gt_chromEnd + 50000:
                    region = 'CIS5'
                    my_typeCountH["cis5"] += 1
                elif gt_chromEnd - 2500 <= mt_chromStart <= gt_chromEnd + 2500:
                    region = 'PROMOTER'
                    my_typeCountH["promoter"] += 1
                elif gt_chromStart < mt_chromStart < gt_chromEnd - 2500:
                    region = 'GENEBODY'
                    my_typeCountH["genebody"] += 1
                elif gt_chromStart - 50000 <= mt_chromStart <= gt_chromStart:
                    region = 'CIS3'
                    my_typeCountH["cis3"] += 1
                elif mt_chromStart < gt_chromStart - 50000:
                    region = 'DISTAL3'
                    my_typeCountH["distal3"] += 1
                else:
                    region = None

                if region:
                    cpgA = dict(base_row)
                    cpgA.update(annot_base)
                    cpgA['region'] = region
                    if logger.isEnabledFor(logging.DEBUG): logger.debug(f"[assignRegion] {nlp} {list(cpgA.values())}")
                    my_eqtmA.append(cpgA)
                    assigned_regions += 1

            # If no region matched, still append the row with region=None but with existing annotation base
            if assigned_regions == 0:
                cpgA = dict(base_row)
                cpgA.update(annot_base)
                cpgA['region'] = None
                my_eqtmA.append(cpgA)

        if my_eqtmA:
            assigned_total += len(my_eqtmA)
            out_df = pd.DataFrame(my_eqtmA)

            # Make sure types are correct for Parquet, handling nulls with Pandas nullable integer extension type if needed
            # For pa.int64() in pyarrow, pd.Series with NaNs should be either float64 (old way) or Int64 (new pandas way).
            # PyArrow handles Int64 properly when casting.
            if 'mt_chromStart' in out_df.columns:
                out_df['mt_chromStart'] = pd.to_numeric(out_df['mt_chromStart'], errors='coerce').astype('Int64')
            if 'gt_chromStart' in out_df.columns:
                out_df['gt_chromStart'] = pd.to_numeric(out_df['gt_chromStart'], errors='coerce').astype('Int64')

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

    # Write missing annotation IDs to sidecar file
    out_dir = os.path.dirname(outFileName)
    if not out_dir:
        out_dir = "."
    sidecar_path = os.path.join(out_dir, "annotation_missing_ids.txt")

    with open(sidecar_path, "w") as f:
        f.write("type\tid\treason\n")
        for g_id in sorted(missing_gene_ids.keys()):
            f.write(f"gene\t{g_id}\t{missing_gene_ids[g_id]}\n")
        for m_id in sorted(missing_meth_ids.keys()):
            f.write(f"meth\t{m_id}\t{missing_meth_ids[m_id]}\n")

    N_gene = len(missing_gene_ids)
    N_meth = len(missing_meth_ids)
    logger.info(f"[assignRegion] Annotation missing: {N_gene} unique genes, {N_meth} unique CpGs (full list -> {os.path.abspath(sidecar_path)})")

    logger.info(f"[assignRegion] eCpgs Processed: {nlp} Assigned: {assigned_total} Excluded (any): {ne}")
    logger.info(f"[assignRegion] eCpgs Excluded: p-value filter: {npvalx} p-value missing: {npskip} gx annotation: {negx} mt annotation: {nemt}")
    logger.info(f"[assignRegion] eCpgs Counts by Region: {my_typeCountH}")
    return assigned_total


#### Verify alignment of annotation files #######################################
def verify_alignment(geneH, methylH, ecpgDataFile):
    logger.info("[verify_alignment] Starting alignment verification...")

    # 1. Chromosome Nomenclature Test
    gene_chroms = [str(info["chrom"]) for key, info in list(geneH.items())[:50]]
    methyl_chroms = [str(info["chrom"]) for key, info in list(methylH.items())[:50]]

    gene_has_chr = any(c.startswith("chr") for c in gene_chroms)
    methyl_has_chr = any(c.startswith("chr") for c in methyl_chroms)

    if gene_has_chr != methyl_has_chr:
        logger.critical("[CRITICAL WARNING] Chromosome nomenclature mismatch detected between gene and methylation annotations. One uses 'chr' prefix and the other does not.")
        if gene_has_chr:
            logger.info("[verify_alignment] Normalizing methylation annotations to include 'chr' prefix.")
            for key, info in methylH.items():
                if not str(info["chrom"]).startswith("chr"):
                    info["chrom"] = "chr" + str(info["chrom"])
        else:
            logger.info("[verify_alignment] Normalizing methylation annotations to strip 'chr' prefix.")
            for key, info in methylH.items():
                if str(info["chrom"]).startswith("chr"):
                    info["chrom"] = str(info["chrom"])[3:]
            logger.info("[verify_alignment] Normalizing gene annotations to strip 'chr' prefix.")
            for key, info in geneH.items():
                if str(info["chrom"]).startswith("chr"):
                    info["chrom"] = str(info["chrom"])[3:]

    # Read first chunk of parquet file to check Ensembl ID and top 10 associations
    try:
        parquet_file = pq.ParquetFile(ecpgDataFile)
        schema = pq.read_schema(ecpgDataFile)
        columns = schema.names
        if "precise_mt_p" in columns:
            pval_col = "precise_mt_p"
        elif "mt_p" in columns:
            pval_col = "mt_p"
        else:
            logger.error("[verify_alignment] Neither 'precise_mt_p' nor 'mt_p' found in input Parquet file.")
            sys.exit(1)

        batch = next(parquet_file.iter_batches(batch_size=1000, columns=['gt_id', 'mt_id', pval_col]))
        df = batch.to_pandas()
        if df.index.names != [None]:
            df = df.reset_index()
    except Exception as e:
        logger.error(f"[verify_alignment] Error reading parquet file for verification: {e}")
        sys.exit(1)

    # 2. Ensembl ID Suffix Check
    first_few_genes = df['gt_id'].dropna().head(50).astype(str).tolist()
    versioned_in_geneH = any('.' in k for k in list(geneH.keys())[:100])
    versionless_in_parquet = first_few_genes and not any('.' in g for g in first_few_genes)

    if versioned_in_geneH and versionless_in_parquet:
        logger.info("[verify_alignment] Ensembl ID version mismatch detected. GTF uses versioned IDs while Parquet uses versionless. Normalizing GTF IDs by stripping version (.split('.')[0]).")
        new_geneH = {}
        for key, value in geneH.items():
            new_key = key.split('.')[0]
            # Avoid overwriting if multiple transcripts map to same versionless gene
            if new_key not in new_geneH:
                new_geneH[new_key] = value
        geneH.clear()
        geneH.update(new_geneH)


    # 2.5 Data Type Verification
    logger.info("[verify_alignment] Verifying and enforcing integer types for all coordinates...")
    for key, info in methylH.items():
        if not isinstance(info["chromStart"], int):
            try:
                info["chromStart"] = int(info["chromStart"])
            except ValueError:
                logger.error(f"[verify_alignment] Failed to cast chromStart '{info['chromStart']}' to int for {key}")

    for key, info in geneH.items():
        if not isinstance(info["chromStart"], int):
            try:
                info["chromStart"] = int(info["chromStart"])
            except ValueError:
                logger.error(f"[verify_alignment] Failed to cast chromStart '{info['chromStart']}' to int for {key}")
        if not isinstance(info["chromEnd"], int):
            try:
                info["chromEnd"] = int(info["chromEnd"])
            except ValueError:
                logger.error(f"[verify_alignment] Failed to cast chromEnd '{info['chromEnd']}' to int for {key}")

    # 3. Coordinate "Sanity" Sample
    logger.info("[verify_alignment] Top 10 Coordinate Sanity Sample (Log Dump):")
    df_sorted = df.sort_values(by=pval_col)
    top_10 = df_sorted.head(10)

    for index, row in top_10.iterrows():
        gt_id = str(row['gt_id'])
        mt_id = str(row['mt_id'])

        cpg_chr, cpg_pos = "N/A", "N/A"
        gene_chr, gene_start, gene_end = "N/A", "N/A", "N/A"

        if mt_id in methylH:
            cpg_chr = methylH[mt_id]["chrom"]
            cpg_pos = methylH[mt_id]["chromStart"]

        if gt_id in geneH:
            gene_chr = geneH[gt_id]["chrom"]
            gene_start = geneH[gt_id]["chromStart"]
            gene_end = geneH[gt_id]["chromEnd"]

        logger.info(f"[verify_alignment] Pair Sample -> cpg_id: {mt_id}, cpg_chr: {cpg_chr}, cpg_pos: {cpg_pos} | gene_id: {gt_id}, gene_chr: {gene_chr}, gene_start: {gene_start}")

    logger.info("[verify_alignment] Alignment verification complete.")

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

        ## Verify alignment before processing
    verify_alignment(geneH, methylH, ecpgDataFile)

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
