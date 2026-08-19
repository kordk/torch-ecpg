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


#### Invert a parsed GFF loci dict into a symbol -> gene-model map #############
def build_symbol_to_model(loci_dict):
    """Return {gene_name: {'chrom','chromStart','chromEnd','strand'}} from a
    parsed GFF loci dict (keyed by Ensembl gene id, each entry carrying a
    'gene_name' from readAnnotationFileToDict's GFF branch).

    Precedence rule: drop-if-ambiguous. A symbol that resolves to two or more
    DISTINCT gene models -- differing in any of chrom/chromStart/chromEnd/strand
    -- is omitted from the map, so a probe carrying that symbol finds nothing and
    falls to region=None. Multiple entries with identical coordinates collapse to
    the single shared model and are kept. Entries lacking a 'gene_name' (e.g.
    BED6-parsed) are skipped.
    """
    by_symbol = {}
    for entry in loci_dict.values():
        symbol = entry.get("gene_name")
        if not symbol:
            continue
        model = {
            "chrom": entry["chrom"],
            "chromStart": entry["chromStart"],
            "chromEnd": entry["chromEnd"],
            "strand": entry["strand"],
        }
        by_symbol.setdefault(symbol, []).append(model)

    resolved = {}
    n_dropped = 0
    for symbol, models in by_symbol.items():
        distinct = {
            (m["chrom"], m["chromStart"], m["chromEnd"], m["strand"])
            for m in models
        }
        if len(distinct) == 1:
            resolved[symbol] = models[0]
        else:
            # Ambiguous symbol: >=2 distinct gene models. Dropped by policy.
            n_dropped += 1

    logger.info(
        f"[build_symbol_to_model] Resolved {len(resolved)} symbols; "
        f"dropped {n_dropped} ambiguous (>=2 distinct models)."
    )
    return resolved


#### Resolve probes to gene models via their Re-Annotator symbol ##############
def build_probe_to_model(probe_symbol_pairs, symbol_to_model):
    """Return {probe_id: {'chrom','chromStart','chromEnd','strand'}} by resolving
    each probe's Re-Annotator Gene_symbol against a symbol_to_model map (from
    build_symbol_to_model). Drop-if-ambiguous is applied at the probe level:

      * no symbol                              -> dropped
      * two or more symbols (comma/semicolon)  -> dropped
      * one symbol, unmatched in symbol_to_model -> dropped
      * one symbol, matched                     -> emitted with that gene model

    A dropped probe is simply absent from the result, so downstream lookup treats
    it as unannotated (region=None). probe_symbol_pairs is an iterable of
    (probe_id, gene_symbol_field); gene_symbol_field is the raw Re-Annotator
    string (possibly '', 'NA', or 'A,B'). No pandas here, so the module stays
    stdlib-only and importable without side effects.
    """
    resolved = {}
    n_no_symbol = 0
    n_multi = 0
    n_unmatched = 0
    for probe_id, symbol_field in probe_symbol_pairs:
        raw = (symbol_field or "").replace(";", ",")
        tokens = [t.strip() for t in raw.split(",")]
        tokens = [t for t in tokens if t and t.upper() not in ("NA", ".", "-")]

        if len(tokens) == 0:
            n_no_symbol += 1
            continue
        if len(tokens) > 1:
            n_multi += 1
            continue

        model = symbol_to_model.get(tokens[0])
        if model is None:
            n_unmatched += 1
            continue

        resolved[probe_id] = model

    logger.info(
        f"[build_probe_to_model] Resolved {len(resolved)} probes; dropped "
        f"{n_no_symbol} no-symbol, {n_multi} multi-symbol, {n_unmatched} unmatched."
    )
    return resolved
