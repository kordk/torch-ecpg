#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import numpy as np
import logging
import networkx as nx
from fa2 import ForceAtlas2
import umap
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Bipartite Network Visualizations.")
    parser.add_argument("--edges", required=True, help="Path to the edges CSV file.")
    parser.add_argument("--nodes", required=True, help="Path to the nodes CSV file.")
    parser.add_argument("--out-dir", default=".", help="Directory to save output figures (default: current directory).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Edge weight threshold for filtering (default: 0.5).")
    return parser.parse_args()

def load_data(args):
    logging.info(f"Loading edges from {args.edges}...")
    edges = pd.read_csv(args.edges)
    logging.info(f"Loading nodes from {args.nodes}...")
    nodes = pd.read_csv(args.nodes)

    # Determine weight column
    if 'mt_ig' in edges.columns:
        weight_col = 'mt_ig'
        logging.info("Using 'mt_ig' as edge weight.")
    elif 'abs_t' in edges.columns:
        weight_col = 'abs_t'
        logging.warning("Column 'mt_ig' not found. Falling back to 'abs_t' for edge weights.")
    else:
        logging.error("Neither 'mt_ig' nor 'abs_t' found in edges. Cannot proceed.")
        sys.exit(1)

    return edges, nodes, weight_col

def prepare_network(edges, nodes, weight_col, threshold):
    logging.info(f"Filtering edges with {weight_col} >= {threshold}...")
    filtered_edges = edges[edges[weight_col] >= threshold].copy()

    if filtered_edges.empty:
        logging.error(f"No edges left after filtering with threshold {threshold}.")
        sys.exit(1)

    logging.info(f"Remaining edges: {len(filtered_edges)}")

    G = nx.Graph()

    # Add Nodes
    cpg_nodes = nodes[nodes['Node_Type'] == 'CpG']
    gene_nodes = nodes[nodes['Node_Type'] == 'Gene']

    # We need to map nodes to their regions for coloring
    region_map = dict(zip(cpg_nodes['Node_ID'], cpg_nodes['Region']))

    # Keep track of which nodes actually appear in filtered edges
    active_sources = set(filtered_edges['Source'])
    active_targets = set(filtered_edges['Target'])
    active_nodes = active_sources.union(active_targets)

    for _, row in cpg_nodes.iterrows():
        node_id = row['Node_ID']
        if node_id in active_nodes:
            region = row.get('Region', 'Undefined')
            G.add_node(node_id, bipartite=0, type='CpG', region=region)

    for _, row in gene_nodes.iterrows():
        node_id = row['Node_ID']
        if node_id in active_nodes:
            G.add_node(node_id, bipartite=1, type='Gene', region='Undefined')

    # Add Edges
    for _, row in filtered_edges.iterrows():
        source = row['Source']
        target = row['Target']
        weight = row[weight_col]
        G.add_edge(source, target, weight=weight)

    return G, filtered_edges

def plot_network(G, out_dir):
    logging.info("Generating Figure 1: Energy-Minimized Bipartite Network...")

    forceatlas2 = ForceAtlas2(
        # Behavior alternatives
        outboundAttractionDistribution=True,  # Dissuade hubs
        linLogMode=False,  # NOT IMPLEMENTED
        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
        edgeWeightInfluence=1.0,

        # Performance
        jitterTolerance=1.0,  # Tolerance
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        multiThreaded=False,

        # Tuning
        scalingRatio=2.0,
        strongGravityMode=False,
        gravity=1.0,

        # Log
        verbose=False
    )

    positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=100)

    fig, ax = plt.subplots(figsize=(12, 12))

    # Node styling
    cpg_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'CpG']
    gene_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'Gene']

    # Map regions to colors
    regions = sorted(list(set([G.nodes[n].get('region', 'Undefined') for n in cpg_nodes])))
    palette = sns.color_palette("Set2", len(regions))
    color_map = dict(zip(regions, palette))

    cpg_colors = [color_map[G.nodes[n].get('region', 'Undefined')] for n in cpg_nodes]

    # Draw Gene Nodes (Light Grey, Larger)
    nx.draw_networkx_nodes(G, positions, nodelist=gene_nodes, node_size=50, node_color='lightgray', ax=ax)

    # Draw CpG Nodes (Colored by Region, Smaller)
    nx.draw_networkx_nodes(G, positions, nodelist=cpg_nodes, node_size=20, node_color=cpg_colors, ax=ax)

    # Edge styling
    edges = G.edges(data=True)
    weights = [d['weight'] for u, v, d in edges]
    if weights:
        max_w = max(weights)
        min_w = min(weights)
        if max_w > min_w:
            # Scale edge width between 0.1 and 2.0 based on weight
            edge_widths = [0.1 + 1.9 * (w - min_w) / (max_w - min_w) for w in weights]
        else:
            edge_widths = [1.0] * len(weights)
    else:
        edge_widths = []

    nx.draw_networkx_edges(G, positions, width=edge_widths, alpha=0.3, ax=ax)

    # Legend for regions
    import matplotlib.lines as mlines
    legend_handles = [mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[reg], markersize=8, label=reg) for reg in regions]
    legend_handles.append(mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=10, label='Gene'))
    ax.legend(handles=legend_handles, title="Node Type / Region", loc='upper right')

    ax.set_title("Energy-Minimized Bipartite Network")
    ax.axis('off')

    out_path = os.path.join(out_dir, "EnergyMinimizedBipartiteNetwork.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved Figure 1 to {out_path}")

def plot_umap(filtered_edges, nodes, weight_col, out_dir):
    logging.info("Generating Figure 2: UMAP of Regulatory Beta-Diversity...")

    # Pivot to CpG (rows) x Gene (cols) matrix
    matrix = filtered_edges.pivot(index='Source', columns='Target', values=weight_col).fillna(0)

    if matrix.shape[0] < 2:
        logging.warning("Not enough CpG nodes to run UMAP. Skipping Figure 2.")
        return

    # Run UMAP
    reducer = umap.UMAP(metric='braycurtis', random_state=42)
    embeddings = reducer.fit_transform(matrix)

    # Create DataFrame for plotting
    umap_df = pd.DataFrame(embeddings, columns=['UMAP1', 'UMAP2'], index=matrix.index)

    # Merge region info
    cpg_nodes = nodes[nodes['Node_Type'] == 'CpG'][['Node_ID', 'Region']].drop_duplicates()
    umap_df = umap_df.merge(cpg_nodes, left_index=True, right_on='Node_ID', how='left')
    umap_df['Region'] = umap_df['Region'].fillna('Undefined')

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Region', palette='Set2', s=30, alpha=0.8, ax=ax)

    ax.set_title("UMAP of Regulatory Beta-Diversity (CpG Nodes)")

    out_path = os.path.join(out_dir, "UMAPofRegulatoryBetaDiversity.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved Figure 2 to {out_path}")

def plot_degree_distribution(G, nodes, out_dir):
    logging.info("Generating Figure 3: Regulatory Degree Distribution...")

    cpg_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'CpG']
    if not cpg_nodes:
        logging.warning("No CpG nodes found for degree distribution. Skipping Figure 3.")
        return

    degrees = []
    regions = []

    for n in cpg_nodes:
        degrees.append(G.degree(n))
        regions.append(G.nodes[n].get('region', 'Undefined'))

    degree_df = pd.DataFrame({'Degree': degrees, 'Region': regions})

    if degree_df.empty:
        logging.warning("Degree dataframe empty. Skipping Figure 3.")
        return

    # Clip to 99th percentile
    p99 = np.percentile(degree_df['Degree'], 99)
    if p99 > 0:
        clipped_df = degree_df[degree_df['Degree'] <= p99]
    else:
        clipped_df = degree_df

    fig, ax = plt.subplots(figsize=(10, 6))

    # Ensure common_norm=False so each region's density integrates to 1
    sns.kdeplot(data=clipped_df, x='Degree', hue='Region', fill=True, common_norm=False, palette='Set2', alpha=0.5, ax=ax)

    ax.set_title("Regulatory Degree Distribution by Region (Clipped to 99th Percentile)")
    ax.set_xlabel("Degree (Number of Target Genes)")
    ax.set_ylabel("Density")

    out_path = os.path.join(out_dir, "RegulatoryDegreeDistribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved Figure 3 to {out_path}")

def main():
    args = parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    edges, nodes, weight_col = load_data(args)

    G, filtered_edges = prepare_network(edges, nodes, weight_col, args.threshold)

    plot_network(G, args.out_dir)
    plot_umap(filtered_edges, nodes, weight_col, args.out_dir)
    plot_degree_distribution(G, nodes, args.out_dir)

    logging.info("All visualizations complete.")

if __name__ == "__main__":
    main()
