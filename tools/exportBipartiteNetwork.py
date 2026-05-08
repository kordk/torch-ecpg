#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Export eQTM Parquet file to Cytoscape Node and Edge tables.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input Parquet file (e.g., results.precise_p.annot.fdr.parquet).")
    parser.add_argument("-o", "--out-prefix", default="cytoscape", help="Output prefix for generated CSV files (default: 'cytoscape').")
    parser.add_argument("--top-k", type=int, default=10000, help="Number of top hits to include (default: 10000).")
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
    required_edge_cols = ['mt_est', 'mt_p']
    missing_edge_cols = [col for col in required_edge_cols if col not in df.columns]
    if missing_edge_cols:
        logging.error(f"Missing required edge columns: {missing_edge_cols}. Expected at least: {required_edge_cols}")
        sys.exit(1)

    # 3. Handle region column
    if 'region' not in df.columns:
        logging.info("Column 'region' not found. Defaulting Interaction to 'Undefined'.")
        df['region'] = 'Undefined'

    # 4. Filter and sort by Saliency / mt_t
    if 'mt_ig' in df.columns:
        logging.info("Sorting by mt_ig (Saliency) in descending order.")
        df_top = df.sort_values(by='mt_ig', ascending=False).head(args.top_k)
        used_abs_t = False
    elif 'mt_t' in df.columns:
        logging.info("Column 'mt_ig' not found. Falling back to sorting by absolute mt_t.")
        df['abs_t'] = df['mt_t'].abs()
        df_top = df.sort_values(by='abs_t', ascending=False).head(args.top_k)
        used_abs_t = True
    else:
        logging.error("Neither 'mt_ig' nor 'mt_t' column found in the Parquet file. Cannot perform top-K ranking.")
        sys.exit(1)

    # 5. Build Edge Table
    logging.info("Building Edge table...")
    base_edge_cols = ['mt_id', 'gt_id', 'region', 'mt_est', 'mt_p']
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

    out_edges = f"{args.out_prefix}_edges.csv"
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

    out_nodes = f"{args.out_prefix}_nodes.csv"
    nodes.to_csv(out_nodes, index=False)

    logging.info(f"Exported {len(nodes)} nodes to {out_nodes} and {len(edges)} edges to {out_edges} for Cytoscape.")

if __name__ == "__main__":
    main()
