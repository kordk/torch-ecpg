#!/usr/bin/env python3

## kord.kober@ucsf.edu
## github.com/kordk/torch-ecpg

## Shared annotation parser used by the region-assignment tool and the
## expression gene-model annotation build. Deliberately stdlib-only so that
## importing it never pulls in pyarrow/torch.

import sys
import logging

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

                # Extract gene_name (symbol) for probe->gene resolution. Absent
                # -> None; the symbol resolver skips entries without one.
                gene_name = None
                for attr in attributes.split(";"):
                    attr = attr.strip()
                    if attr.startswith("gene_name "):
                        # e.g., gene_name "DDX11L2"
                        gene_name = attr[len("gene_name "):].strip('"')
                        break

                # GFF is 1-based, inclusive. BED is 0-based, half-open
                chromStart = int(dataA[3]) - 1
                chromEnd = int(dataA[4])

                my_lociH[my_name] = {}
                my_lociH[my_name]["chrom"]      = str(dataA[0])
                my_lociH[my_name]["chromStart"] = chromStart
                my_lociH[my_name]["chromEnd"]   = chromEnd
                my_lociH[my_name]["strand"]     = str(dataA[6])
                my_lociH[my_name]["gene_name"]  = gene_name


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
