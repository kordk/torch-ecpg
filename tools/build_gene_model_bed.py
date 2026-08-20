#!/usr/bin/env python3

## kord.kober@ucsf.edu
## github.com/kordk/torch-ecpg

## Derive an ILMN-keyed GENE-MODEL BED for region assignment, from a GENCODE GTF
## and the Re-Annotator probe->symbol table. Stdlib-only and import-safe so it can
## be unit-tested and invoked from pipelinePre.sh without triggering any downloads.
## The probe-coordinate BEDs are a separate, unrelated annotation.

import os
import sys
import csv
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotation_io import (  # noqa: E402
    readAnnotationFileToDict,
    build_symbol_to_model,
    build_probe_to_model,
)

logger = logging.getLogger(__name__)


def read_probe_symbols(reannotator_path):
    """Return [(probe_id, gene_symbol_field), ...] from a Re-Annotator TSV,
    one row per probe (first occurrence wins), reading only X.PROBE_ID and
    Gene_symbol."""
    pairs = []
    seen = set()
    with open(reannotator_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            probe = (row.get("X.PROBE_ID") or "").strip()
            if not probe or probe in seen:
                continue
            seen.add(probe)
            pairs.append((probe, row.get("Gene_symbol") or ""))
    return pairs


def build_rows(probe_to_model):
    """Return BED6 tuples (chrom, chromStart, chromEnd, name, score, strand),
    sorted by probe id. chromStart is converted 0-based -> 1-based to match
    M.bed6 and the probe BEDs; readAnnotationFileToDict's GFF branch stores it
    0-based (start-1), so add 1 back here."""
    rows = []
    for probe_id in sorted(probe_to_model):
        m = probe_to_model[probe_id]
        rows.append((
            m["chrom"],
            int(m["chromStart"]) + 1,   # 0-based -> 1-based
            int(m["chromEnd"]),
            probe_id,
            0,
            m["strand"],
        ))
    return rows


def write_bed6(rows, output_path):
    """Write BED6 rows with a header row (matching the demo BEDs)."""
    with open(output_path, "w") as out:
        out.write("chrom\tchromStart\tchromEnd\tname\tscore\tstrand\n")
        for chrom, start, end, name, score, strand in rows:
            out.write(f"{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Derive an ILMN-keyed gene-model BED (1-based) for region assignment."
    )
    ap.add_argument("--gtf", required=True, help="GENCODE GTF (.gtf or .gtf.gz)")
    ap.add_argument("--reannotator", required=True,
                    help="Re-Annotator TSV with X.PROBE_ID and Gene_symbol columns")
    ap.add_argument("--output", required=True, help="Output BED6 path")
    args = ap.parse_args(argv)

    gene_loci = readAnnotationFileToDict(args.gtf)
    symbol_to_model = build_symbol_to_model(gene_loci)
    pairs = read_probe_symbols(args.reannotator)
    probe_to_model = build_probe_to_model(pairs, symbol_to_model)
    rows = build_rows(probe_to_model)
    write_bed6(rows, args.output)

    msg = (f"[build_gene_model_bed] Wrote {len(rows)} probes (1-based) to "
           f"{args.output} from {len(pairs)} probe rows.")
    logger.info(msg)
    print(msg)
    return len(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
