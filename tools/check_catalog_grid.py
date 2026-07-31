#!/usr/bin/env python3
import sys
import argparse
import pyarrow.parquet as pq
import pyarrow.compute as pc

def main():
    parser = argparse.ArgumentParser(description="Check catalog distinct feature counts against bounds.")
    parser.add_argument('--catalog', required=True, help="Path to the parquet catalog to check.")
    parser.add_argument('--max-genes', type=int, default=None, help="Upper bound on distinct gene count.")
    parser.add_argument('--max-loci', type=int, default=None, help="Upper bound on distinct locus count.")
    parser.add_argument('--gene-column', default='gt_id', help="Name of the gene column (default: gt_id).")
    parser.add_argument('--locus-column', default='mt_id', help="Name of the locus column (default: mt_id).")
    args = parser.parse_args()

    try:
        schema = pq.read_schema(args.catalog)
    except Exception as e:
        print(f"Error reading schema from {args.catalog}: {e}", file=sys.stderr)
        sys.exit(1)

    names = schema.names
    if args.gene_column not in names:
        print(f"Missing column '{args.gene_column}'. Available columns: {', '.join(names)}", file=sys.stderr)
        sys.exit(1)
    if args.locus_column not in names:
        print(f"Missing column '{args.locus_column}'. Available columns: {', '.join(names)}", file=sys.stderr)
        sys.exit(1)

    try:
        # Read only the two required columns
        table = pq.read_table(args.catalog, columns=[args.gene_column, args.locus_column])
    except Exception as e:
        print(f"Error reading table from {args.catalog}: {e}", file=sys.stderr)
        sys.exit(1)

    n_rows = table.num_rows

    try:
        u_genes = pc.count_distinct(table[args.gene_column]).as_py()
        u_loci = pc.count_distinct(table[args.locus_column]).as_py()
    except Exception as e:
        print(f"Error computing distinct counts: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"catalog grid: {n_rows} rows, {u_genes} distinct {args.gene_column}, {u_loci} distinct {args.locus_column}")

    if args.max_genes is not None and u_genes > args.max_genes:
        print(f"Validation failed: distinct '{args.gene_column}' ({u_genes}) exceeds bound ({args.max_genes}).", file=sys.stderr)
        sys.exit(1)

    if args.max_loci is not None and u_loci > args.max_loci:
        print(f"Validation failed: distinct '{args.locus_column}' ({u_loci}) exceeds bound ({args.max_loci}).", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)

if __name__ == '__main__':
    main()
