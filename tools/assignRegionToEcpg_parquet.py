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

#### Shared annotation parser lives in tools/annotation_io.py #################
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotation_io import readAnnotationFileToDict, readProbeGeneModel  # noqa: E402

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
def assignRegion(my_ecpgDataFile, gH, mH, gmH, geneModelHeader, pval_col, outFileName, chunk_size):
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
    n_coord_ok = 0    # pairs with both probe coordinates
    n_model_ok = 0    # pairs whose gene probe resolved to a gene model
    n_no_model = 0    # same-chromosome pairs with no gene model

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
        pa.field('region', pa.string()),
        pa.field('gtf_gene_model', pa.string()),
        pa.field('gtf_gene_symbol', pa.string())
    ]

    # Append the new fields to the input schema
    schema = pa.schema(list(input_schema) + new_fields)

    # Stamp the probe->gene map provenance (deriver, GTF path, sha) into the
    # Parquet key-value metadata so the annotation source travels with the data.
    if geneModelHeader:
        md = dict(input_schema.metadata or {})
        for k, v in geneModelHeader.items():
            md[("tecpg_pgm_" + str(k)).encode()] = str(v).encode()
        schema = schema.with_metadata(md)

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

            # Determine if we have annotation info. mH/gH are the PROBE BEDs and
            # supply the coordinate columns; gmH is the probe->gene map and is used
            # only to place the CpG against a gene span. A probe absent from gmH
            # loses its region label, never its coordinates.
            has_mt_annot = mt_id in mH and "chrom" in mH[mt_id]
            has_gt_annot = gt_id in gH and "chrom" in gH[gt_id]
            has_gene_model = gt_id in gmH

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

            # Coordinates are written from the probe BEDs whenever the id is
            # present, independent of whether a gene model resolved.
            mt_chrom       = mH[mt_id]["chrom"]       if has_mt_annot else None
            mt_chromStart  = mH[mt_id]["chromStart"]  if has_mt_annot else None
            mt_strand      = mH[mt_id]["strand"]      if has_mt_annot else None

            gt_chrom       = gH[gt_id]["chrom"]       if has_gt_annot else None
            gt_chromStart  = gH[gt_id]["chromStart"]  if has_gt_annot else None
            gt_strand      = gH[gt_id]["strand"]      if has_gt_annot else None

            # Gene identity is a property of the probe, so it is written whenever
            # the probe resolved -- including on TRANS rows, where no gene model
            # was consulted to produce the label.
            annot_base = {
                'mt_chrom': mt_chrom,
                'mt_chromStart': mt_chromStart,
                'mt_strand': mt_strand,
                'gt_chrom': gt_chrom,
                'gt_chromStart': gt_chromStart,
                'gt_strand': gt_strand,
                'gtf_gene_model':  gmH[gt_id]["gtf_gene_model"]  if has_gene_model else None,
                'gtf_gene_symbol': gmH[gt_id]["gtf_gene_symbol"] if has_gene_model else None,
            }

            if has_mt_annot and has_gt_annot:
                n_coord_ok += 1
            if has_gene_model:
                n_model_ok += 1

            if not has_mt_annot or not has_gt_annot:
                # No usable position for one side: the CpG cannot be placed at all.
                cpgA = dict(base_row)
                cpgA.update(annot_base)
                cpgA['region'] = None
                my_eqtmA.append(cpgA)
                ne += 1
                continue

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

            if not has_gene_model:
                # Same chromosome, but no gene span to measure against.
                cpgA = dict(base_row)
                cpgA.update(annot_base)
                cpgA['region'] = None
                my_eqtmA.append(cpgA)
                n_no_model += 1
                continue

            # The region windows are measured against the GENE MODEL span, not the
            # expression probe footprint.
            gm_chromStart = gmH[gt_id]["chromStart"]
            gm_chromEnd   = gmH[gt_id]["chromEnd"]
            gm_strand     = gmH[gt_id]["strand"]

            # Keep track of regions assigned for this pair
            assigned_regions = 0

            ## Region assignment - positive strand
            if gm_strand == "+":
                if mt_chromStart < gm_chromStart - 50000:
                    region = 'DISTAL5'
                    my_typeCountH["distal5"] += 1
                elif gm_chromStart - 50000 <= mt_chromStart < gm_chromStart - 2500:
                    region = 'CIS5'
                    my_typeCountH["cis5"] += 1
                elif gm_chromStart - 2500 <= mt_chromStart <= gm_chromStart + 2500:
                    region = 'PROMOTER'
                    my_typeCountH["promoter"] += 1
                elif gm_chromStart + 2500 < mt_chromStart < gm_chromEnd:
                    region = 'GENEBODY'
                    my_typeCountH["genebody"] += 1
                elif gm_chromEnd <= mt_chromStart <= gm_chromEnd + 50000:
                    region = 'CIS3'
                    my_typeCountH["cis3"] += 1
                elif mt_chromStart > gm_chromEnd + 50000:
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
            if gm_strand == "-":
                if mt_chromStart > gm_chromEnd + 50000:
                    region = 'DISTAL5'
                    my_typeCountH["distal5"] += 1
                elif gm_chromEnd + 2500 < mt_chromStart <= gm_chromEnd + 50000:
                    region = 'CIS5'
                    my_typeCountH["cis5"] += 1
                elif gm_chromEnd - 2500 <= mt_chromStart <= gm_chromEnd + 2500:
                    region = 'PROMOTER'
                    my_typeCountH["promoter"] += 1
                elif gm_chromStart < mt_chromStart < gm_chromEnd - 2500:
                    region = 'GENEBODY'
                    my_typeCountH["genebody"] += 1
                elif gm_chromStart - 50000 <= mt_chromStart <= gm_chromStart:
                    region = 'CIS3'
                    my_typeCountH["cis3"] += 1
                elif mt_chromStart < gm_chromStart - 50000:
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

    pct = (lambda n: (100.0 * n / nlp) if nlp else 0.0)
    logger.info(f"[assignRegion] coverage: coordinate {n_coord_ok} ({pct(n_coord_ok):.1f}%) of {nlp} pairs")
    logger.info(f"[assignRegion] coverage: gene-annotation {n_model_ok} ({pct(n_model_ok):.1f}%) of {nlp} pairs")
    logger.info(f"[assignRegion] same-chromosome pairs with no gene model: {n_no_model}")
    logger.info(f"[assignRegion] eCpgs Processed: {nlp} Assigned: {assigned_total} Excluded (any): {ne}")
    logger.info(f"[assignRegion] eCpgs Excluded: p-value filter: {npvalx} p-value missing: {npskip} gx annotation: {negx} mt annotation: {nemt}")
    logger.info(f"[assignRegion] eCpgs Counts by Region: {my_typeCountH}")
    return assigned_total


#### Verify alignment of annotation files #######################################
def verify_alignment(geneH, methylH, geneModelH, ecpgDataFile):
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

    # The probe->gene map is a third, independently derived annotation. Normalize
    # it onto whatever convention the probe BEDs settled on above.
    if geneModelH:
        probe_has_chr = any(str(i["chrom"]).startswith("chr")
                            for i in list(geneH.values())[:50])
        for key, info in geneModelH.items():
            c = str(info["chrom"])
            if probe_has_chr and not c.startswith("chr"):
                info["chrom"] = "chr" + c
            elif not probe_has_chr and c.startswith("chr"):
                info["chrom"] = c[3:]

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

    for key, info in geneModelH.items():
        for f in ("chromStart", "chromEnd"):
            if not isinstance(info[f], int):
                try:
                    info[f] = int(info[f])
                except ValueError:
                    logger.error(f"[verify_alignment] Failed to cast {f} '{info[f]}' to int for {key}")

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
    parser.add_argument("-g", "--geneAnnotFile", required=True, help="<expression PROBE annotation BED; supplies the gt_* coordinate columns>")
    parser.add_argument("--gene-model", dest="geneModelFile", required=True, help="<probe->gene map TSV from build_probe_gene_model.py; supplies the gene span used for the region windows>")
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
    geneModelFile = args.geneModelFile
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

    if not os.path.exists(geneModelFile):
        logger.error(f"[MAIN] probe->gene map not found: {geneModelFile}")
        sys.exit(205)
    logger.info(f"[MAIN] probe->gene map: {geneModelFile}")

    if not os.path.exists(methylAnnotFile):
        logger.error(f"[MAIN] methylation annotation file not found: {methylAnnotFile}")
        sys.exit(204)
    logger.info(f"[MAIN] methylation annotation file: {methylAnnotFile}")

    logger.info(f"[MAIN] output file name: {outFileName}")


    ## Read in the gene annotation file to a dictionary
    geneH = readAnnotationFileToDict(geneAnnotFile)

    ## Read in the methylation annotation file to a dictionary
    methylH = readAnnotationFileToDict(methylAnnotFile)

    ## Read in the probe->gene map (region windows only; never written verbatim)
    geneModelH, geneModelHeader = readProbeGeneModel(geneModelFile)

        ## Verify alignment before processing
    verify_alignment(geneH, methylH, geneModelH, ecpgDataFile)

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
    reportPvalues(ecpgDataFile, pval_col, chunk_size)

    ## Annotate the pairs
    assigned_total = assignRegion(ecpgDataFile, geneH, methylH, geneModelH, geneModelHeader, pval_col, outFileName, chunk_size)

    logger.info(f"[MAIN] Saving annotated data to: {outFileName}")
    logger.info("[MAIN] Done.")

if __name__ == "__main__":
    main()
