#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

P_COLUMN_PREFERENCE = ['precise_mt_p', 'mt_p']


def main():
    parser = argparse.ArgumentParser(description="Export eQTM Parquet file to Cytoscape Node and Edge tables.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input Parquet file (e.g., results.precise_p.annot.fdr.parquet).")
    parser.add_argument("-o", "--out-prefix", default="cytoscape", help="Output prefix for generated CSV files (default: 'cytoscape').")
    parser.add_argument("--out-dir", default=".", help="Directory to save output files (default: current directory).")
    parser.add_argument("--top-k", type=int, default=10000, help="Number of top hits to include (default: 10000).")
    parser.add_argument("--min-effect", type=float, default=None, help="Retain only edges where abs(mt_est) >= min-effect.")
    parser.add_argument("--max-boot-p", type=float, default=None, help="Retain only edges where p_boot <= max-boot-p.")
    parser.add_argument("--max-fdr", type=float, default=None, help="Retain only edges where fdr_est <= max-fdr. Rows with missing (NaN) fdr_est are excluded when this filter is active.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)

    logging.info(f"Loading data from {args.input}...")
    try:
        df = pd.read_parquet(args.input)
    except Exception as e:
        logging.error(f"Failed to read Parquet file: {e}")
        sys.exit(1)

    # 1. Check required node columns
    required_node_cols = ['mt_id', 'mt_chrom', 'mt_chromStart', 'mt_strand',
                          'gt_id', 'gt_chrom', 'gt_chromStart', 'gt_strand']
    missing_node_cols = [col for col in required_node_cols if col not in df.columns]
    if missing_node_cols:
        logging.error(f"Missing required node columns: {missing_node_cols}. Expected: {required_node_cols}")
        sys.exit(1)

    # 2. Check required edge columns (statistical)
    # precise_mt_p is float64. mt_p is float32 and is computed by a subtraction
    # in which values below about 5.96e-08 (2**-24) are lost to cancellation, so
    # it can read as exactly zero across the range the top-ranked edges occupy.
    p_column = next((c for c in P_COLUMN_PREFERENCE if c in df.columns), None)
    missing_edge_cols = [c for c in ['mt_est'] if c not in df.columns]
    if p_column is None:
        missing_edge_cols.append(' or '.join(P_COLUMN_PREFERENCE))
    if missing_edge_cols:
        logging.error(f"Missing required edge columns: {missing_edge_cols}. Expected 'mt_est' and one of: {P_COLUMN_PREFERENCE}")
        sys.exit(1)
    if p_column == 'mt_p':
        logging.warning(
            "Using 'mt_p' for edge p-values because 'precise_mt_p' is absent. "
            "mt_p is float32 and loses values below about 5.96e-08 to "
            "cancellation, so the most significant edges may carry a p-value "
            "of exactly zero."
        )
    else:
        logging.info(f"Using '{p_column}' for edge p-values.")

    # 3. Handle region column
    if 'region' not in df.columns:
        logging.info("Column 'region' not found. Defaulting Interaction to 'Undefined'.")
        df['region'] = 'Undefined'

    # 4. Apply Filters (Order: --min-effect, --max-boot-p, --max-fdr, --top-k)
    # The --top-k filter selects the top K from whatever survives the threshold filters.
    # It ranks by mt_ig (Integrated Gradients score) descending, with fallback ranking by abs(mt_t).

    total_edges = len(df)
    logging.info(f"Total edges before filtering: {total_edges}")

    if args.min_effect is not None:
        df = df[df['mt_est'].abs() >= args.min_effect]
        logging.info(f"Edges surviving --min-effect >= {args.min_effect}: {len(df)}")

    if args.max_boot_p is not None:
        if 'p_boot' in df.columns:
            df = df[df['p_boot'] <= args.max_boot_p]
            logging.info(f"Edges surviving --max-boot-p <= {args.max_boot_p}: {len(df)}")
        else:
            logging.warning("Column 'p_boot' not found in Parquet file. Skipping --max-boot-p filter.")

    if args.max_fdr is not None:
        if 'fdr_est' in df.columns:
            df = df[df['fdr_est'] <= args.max_fdr]
            logging.info(f"Edges surviving --max-fdr <= {args.max_fdr}: {len(df)}")
        else:
            logging.warning("Column 'fdr_est' not found in Parquet file. Skipping --max-fdr filter.")

    if 'mt_ig' in df.columns:
        logging.info(f"Sorting by mt_ig (Saliency) in descending order and taking top {args.top_k}.")
        df_top = df.sort_values(by='mt_ig', ascending=False).head(args.top_k)
        used_abs_t = False
    elif 'mt_t' in df.columns:
        logging.info(f"Column 'mt_ig' not found. Falling back to sorting by absolute mt_t and taking top {args.top_k}.")
        df['abs_t'] = df['mt_t'].abs()
        df_top = df.sort_values(by='abs_t', ascending=False).head(args.top_k)
        used_abs_t = True
    else:
        logging.error("Neither 'mt_ig' nor 'mt_t' column found in the Parquet file. Cannot perform top-K ranking.")
        sys.exit(1)

    logging.info(f"Edges surviving --top-k filter: {len(df_top)}")

    # 5. Build Edge Table
    logging.info("Building Edge table...")
    base_edge_cols = ['mt_id', 'gt_id', 'region', 'mt_est', p_column]
    if 'mt_t' in df_top.columns:
        base_edge_cols.append('mt_t')
    if used_abs_t:
        base_edge_cols.append('abs_t')
    if 'fdr_est' in df_top.columns:
        base_edge_cols.append('fdr_est')

    ig_cols = [c for c in df_top.columns if c.endswith('_ig')]

    edge_cols = base_edge_cols + ig_cols
    edges = df_top[edge_cols].copy()
    edges.rename(columns={'mt_id': 'Source', 'gt_id': 'Target', 'region': 'Interaction'}, inplace=True)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    out_edges = os.path.join(args.out_dir, f"{args.out_prefix}_edges.csv")
    edges.to_csv(out_edges, index=False)

    # 6. Build Node Table
    logging.info("Building Node table...")
    # Extract CpGs
    cpgs = df_top[['mt_id', 'mt_chrom', 'mt_chromStart', 'mt_strand', 'region']].copy()
    cpgs.columns = ['Node_ID', 'Chrom', 'Start', 'Strand', 'Region']
    cpgs['Node_Type'] = 'CpG'

    # Extract Genes
    genes = df_top[['gt_id', 'gt_chrom', 'gt_chromStart', 'gt_strand']].copy()
    genes.columns = ['Node_ID', 'Chrom', 'Start', 'Strand']
    genes['Node_Type'] = 'Gene'
    genes['Region'] = 'Undefined'

    # Stack and Deduplicate
    nodes = pd.concat([cpgs, genes]).drop_duplicates(subset=['Node_ID'])

    out_nodes = os.path.join(args.out_dir, f"{args.out_prefix}_nodes.csv")
    nodes.to_csv(out_nodes, index=False)

    logging.info(f"Final edge count written to output: {len(edges)}")
    logging.info(f"Exported {len(nodes)} nodes to {out_nodes} and {len(edges)} edges to {out_edges} for Cytoscape.")

if __name__ == "__main__":
    main()
