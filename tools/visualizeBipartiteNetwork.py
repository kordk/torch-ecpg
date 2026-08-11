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

    num_edges = len(edges)
    num_cpg_nodes = len(nodes[nodes['Node_Type'] == 'CpG']) if 'Node_Type' in nodes.columns else 0
    num_gene_nodes = len(nodes[nodes['Node_Type'] == 'Gene']) if 'Node_Type' in nodes.columns else 0
    logging.info(f"Network input: {num_edges} edges, {num_cpg_nodes} CpG nodes, {num_gene_nodes} gene nodes")

    if 'Region' not in nodes.columns:
        logging.error("Column 'Region' not found in nodes. Cannot proceed.")
        sys.exit(1)

    nodes['Region'] = nodes['Region'].fillna('Undefined')

    # Log summary of counts for each region
    region_counts = nodes['Region'].value_counts()
    logging.info("Region counts:")
    for region, count in region_counts.items():
        logging.info(f"  {region}: {count}")

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

    # Check for inf / -inf values in the weight column
    if np.isinf(edges[weight_col]).any():
        logging.error(f"The weight column '{weight_col}' contains infinite values (inf or -inf). "
                      f"This typically occurs due to preprocessing artifacts like -log10(p) conversions of exact zero p-values. "
                      f"Please clean your data to remove or cap these infinite values before running the script.")
        sys.exit(1)

    return edges, nodes, weight_col

def prepare_network(edges, nodes, weight_col, threshold, out_dir):
    logging.info(f"Filtering edges with {weight_col} >= {threshold}...")
    filtered_edges = edges[edges[weight_col] >= threshold].copy()

    if filtered_edges.empty:
        logging.error(f"No edges left after filtering with threshold {threshold}.")
        sys.exit(1)

    # Check for duplicate edges
    duplicates_mask = filtered_edges.duplicated(subset=['Source', 'Target'], keep=False)
    if duplicates_mask.any():
        duplicate_edges = filtered_edges[duplicates_mask]

        # Keep the edge with the maximum weight
        filtered_edges = filtered_edges.sort_values(by=weight_col, ascending=False).drop_duplicates(subset=['Source', 'Target'], keep='first')

        # Identify the exact dropped edges by filtering out the ones we kept
        dropped_edges = duplicate_edges.loc[~duplicate_edges.index.isin(filtered_edges.index)]

        num_dropped = len(dropped_edges)
        logging.warning(f"Found duplicate edges. Dropped {num_dropped} duplicate(s), keeping the maximum weight.")

        dropped_out_path = os.path.join(out_dir, "dropped_duplicate_edges.csv")
        dropped_edges.to_csv(dropped_out_path, index=False)
        logging.info(f"Saved dropped duplicate edges to {dropped_out_path}")

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

    if matrix.shape[0] < 4:
        logging.warning("Not enough CpG nodes to run UMAP (requires >= 4). Skipping Figure 2.")
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

def plot_bi_adjacency_heatmap(df, cpg_col='Source', gene_col='Target', weight_col='weight', out_dir='.', figsize=(10, 8),
                              out_name='BiclusteredBiAdjacencyHeatmap.png', title='Biclustered Bi-Adjacency Heatmap'):
    logging.info(f"Generating {title} (weight_col='{weight_col}')...")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logging.error("Required visualization libraries not found. Please run 'pip install torch-ecpg[viz]'.")
        sys.exit(1)

    # Pivot to wide form and fill NaNs with 0
    matrix = df.pivot(index=cpg_col, columns=gene_col, values=weight_col).fillna(0.0)

    has_negative = (matrix.values < 0).any()

    if has_negative:
        cmap = 'RdBu_r'
        center = 0
        mask = None
    else:
        cmap = 'viridis'
        center = None
        mask = matrix == 0.0

    try:
        g = sns.clustermap(
            matrix,
            cmap=cmap,
            center=center,
            mask=mask,
            figsize=figsize,
            cbar_pos=(0.02, 0.8, 0.05, 0.18),
            xticklabels=False,
            yticklabels=False,
            method='average' # Hierarchical clustering method
        )

        g.fig.suptitle(title, y=1.05)

        out_path = os.path.join(out_dir, out_name)
        g.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(g.fig)
        logging.info(f"Saved Figure 4 to {out_path}")
    except Exception as e:
        logging.error(f"Failed to generate Biclustered Bi-Adjacency Heatmap: {e}")

def plot_arc_diagram(df, cpg_col='Source', gene_col='Target', weight_col='weight', out_dir='.', figsize=(12, 6)):
    logging.info("Generating Figure 5: Arc Diagram...")
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        logging.error("Required visualization libraries not found. Please run 'pip install torch-ecpg[viz]'.")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=figsize)

    # Get unique CpGs and Genes
    unique_cpgs = df[cpg_col].unique()
    unique_genes = df[gene_col].unique()

    n_cpg = len(unique_cpgs)
    n_gene = len(unique_genes)

    # Map to coordinates
    # CpGs from 0 to n_cpg - 1
    # Genes from n_cpg to n_cpg + n_gene - 1
    cpg_pos = {cpg: i for i, cpg in enumerate(unique_cpgs)}
    gene_pos = {gene: i + n_cpg for i, gene in enumerate(unique_genes)}

    # Plot nodes
    ax.scatter(list(cpg_pos.values()), [0] * n_cpg, color='tab:blue', s=20, label='CpG', zorder=5)
    ax.scatter(list(gene_pos.values()), [0] * n_gene, color='tab:orange', s=20, label='Gene', zorder=5)

    # Normalize weights for widths/alphas
    weights = df[weight_col].abs()
    if len(weights) > 0:
        max_w = weights.max()
        min_w = weights.min()
    else:
        max_w, min_w = 1, 0

    # Colormap based on weight sign if applicable
    cmap_pos = plt.get_cmap('Reds')
    cmap_neg = plt.get_cmap('Blues')

    # Draw arcs
    for _, row in df.iterrows():
        cpg = row[cpg_col]
        gene = row[gene_col]
        w = row[weight_col]
        abs_w = abs(w)

        x1 = cpg_pos[cpg]
        x2 = gene_pos[gene]

        center = ((x1 + x2) / 2.0, 0)
        width = abs(x2 - x1)
        # Height can be proportional to width or a constant. Let's make it proportional to width
        height = width * 0.5

        # Line width scaling
        if max_w > min_w:
            lw = 0.5 + 2.0 * (abs_w - min_w) / (max_w - min_w)
            alpha = 0.3 + 0.5 * (abs_w - min_w) / (max_w - min_w)
        else:
            lw = 1.0
            alpha = 0.5

        color = cmap_pos(alpha) if w >= 0 else cmap_neg(alpha)

        arc = patches.Arc(center, width, height, theta1=0, theta2=180,
                          linewidth=lw, color=color, alpha=alpha, zorder=1)
        ax.add_patch(arc)

    ax.set_xlim(-1, n_cpg + n_gene)
    # The max height is roughly half the max width
    max_width = (n_cpg + n_gene)
    ax.set_ylim(-max_width * 0.05, max_width * 0.3)

    ax.axis('off')
    ax.legend(loc='upper right')
    ax.set_title("Arc Diagram")

    out_path = os.path.join(out_dir, "ArcDiagram.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved Figure 5 to {out_path}")

def project_bipartite_to_unipartite(df, cpg_col='Source', gene_col='Target', weight_col='weight', target='gene', weight_method='count'):
    logging.info(f"Generating Bipartite Projection (1-Mode Network) for {target} using method: {weight_method}...")
    try:
        import networkx as nx
        from networkx.algorithms import bipartite
    except ImportError:
        logging.error("Required visualization libraries not found. Please run 'pip install torch-ecpg[viz]'.")
        sys.exit(1)

    B = nx.Graph()

    unique_cpgs = df[cpg_col].unique()
    unique_genes = df[gene_col].unique()

    B.add_nodes_from(unique_cpgs, bipartite=0)
    B.add_nodes_from(unique_genes, bipartite=1)

    edges = [(row[cpg_col], row[gene_col], {weight_col: row[weight_col]}) for _, row in df.iterrows()]
    B.add_edges_from(edges)

    target_nodes = unique_genes if target == 'gene' else unique_cpgs

    if weight_method == 'count':
        # Default networkx behavior (returns count of shared neighbors)
        G_proj = bipartite.weighted_projected_graph(B, target_nodes, ratio=False)
    elif weight_method == 'sum':
        # Manually compute 'sum' of products of shared edges weights
        G_proj = nx.Graph()
        G_proj.add_nodes_from(target_nodes)

        target_nodes_list = list(target_nodes)
        for i, u in enumerate(target_nodes_list):
            for j in range(i + 1, len(target_nodes_list)):
                v = target_nodes_list[j]

                shared_neighbors = set(B[u]) & set(B[v])
                if shared_neighbors:
                    total_weight = sum(B[u][k][weight_col] * B[v][k][weight_col] for k in shared_neighbors)
                    G_proj.add_edge(u, v, weight=total_weight)
    elif weight_method == 'hypergeometric':
        # Compute hypergeom p-value as weight.
        from scipy.stats import hypergeom
        G_proj = nx.Graph()
        G_proj.add_nodes_from(target_nodes)

        total_other_nodes = len(unique_cpgs) if target == 'gene' else len(unique_genes)

        target_nodes_list = list(target_nodes)
        for i, u in enumerate(target_nodes_list):
            for j in range(i + 1, len(target_nodes_list)):
                v = target_nodes_list[j]

                neighbors_u = set(B[u])
                neighbors_v = set(B[v])
                shared_neighbors = neighbors_u & neighbors_v

                if shared_neighbors:
                    k = len(shared_neighbors)
                    M = total_other_nodes
                    n = len(neighbors_u)
                    N = len(neighbors_v)
                    # Survival function: P(X >= k)
                    pval = hypergeom.sf(k - 1, M, n, N)
                    # For visualization/edges, -log10(p) is often better to use as weight
                    import math
                    weight_val = -math.log10(pval) if pval > 0 else 100 # cap to 100 for exact 0
                    G_proj.add_edge(u, v, weight=weight_val)
    else:
        logging.error(f"Unknown weight_method: {weight_method}")
        sys.exit(1)

    # Convert to DataFrame
    edge_list = []
    for u, v, data in G_proj.edges(data=True):
        edge_list.append({
            'Node1': u,
            'Node2': v,
            'weight': data.get('weight', 1.0)
        })
    proj_df = pd.DataFrame(edge_list)

    return G_proj, proj_df

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

    G, filtered_edges = prepare_network(edges, nodes, weight_col, args.threshold, args.out_dir)

    plot_network(G, args.out_dir)
    plot_umap(filtered_edges, nodes, weight_col, args.out_dir)
    plot_degree_distribution(G, nodes, args.out_dir)

    # Note: Source and Target defaults match the output of prepare_network/filtered_edges
    plot_bi_adjacency_heatmap(filtered_edges, cpg_col='Source', gene_col='Target', weight_col=weight_col, out_dir=args.out_dir)

    # Signed biclustered heatmap (mt_est): additive second figure; never replaces
    # the mt_ig/abs_t heatmap above. Writes SignedBiclusteredBiAdjacencyHeatmap.png.
    if 'mt_est' in filtered_edges.columns:
        plot_bi_adjacency_heatmap(
            filtered_edges,
            cpg_col='Source',
            gene_col='Target',
            weight_col='mt_est',
            out_dir=args.out_dir,
            out_name='SignedBiclusteredBiAdjacencyHeatmap.png',
            title='Signed Biclustered Bi-Adjacency Heatmap (mt_est)',
        )
    else:
        logging.warning("Column 'mt_est' not found in edges. Skipping signed biclustered heatmap (mt_est).")

    plot_arc_diagram(filtered_edges, cpg_col='Source', gene_col='Target', weight_col=weight_col, out_dir=args.out_dir)

    # 1-Mode Projection
    G_proj, proj_df = project_bipartite_to_unipartite(filtered_edges, cpg_col='Source', gene_col='Target', weight_col=weight_col, target='gene', weight_method='count')
    # Save the projection edge list
    proj_out_path = os.path.join(args.out_dir, "UnipartiteProjection_Edges.csv")
    proj_df.to_csv(proj_out_path, index=False)
    logging.info(f"Saved Unipartite Projection edge list to {proj_out_path}")

    # Generate a simple visualization for the unipartite network (Figure 6)
    logging.info("Generating Figure 6: Unipartite Projection Network...")
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        fig, ax = plt.subplots(figsize=(10, 10))
        pos = nx.spring_layout(G_proj, seed=42)

        edges_proj = G_proj.edges(data=True)
        weights = [d['weight'] for u, v, d in edges_proj]
        if weights:
            max_w = max(weights)
            min_w = min(weights)
            if max_w > min_w:
                edge_widths = [0.5 + 2.0 * (w - min_w) / (max_w - min_w) for w in weights]
            else:
                edge_widths = [1.0] * len(weights)
        else:
            edge_widths = []

        nx.draw_networkx_nodes(G_proj, pos, node_size=50, node_color='tab:orange', alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G_proj, pos, width=edge_widths, alpha=0.5, ax=ax)

        ax.set_title("Unipartite Projection (Genes)")
        ax.axis('off')

        fig_out_path = os.path.join(args.out_dir, "UnipartiteProjection.png")
        plt.savefig(fig_out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved Figure 6 to {fig_out_path}")
    except Exception as e:
        logging.error(f"Failed to generate Unipartite Projection visualization: {e}")

    logging.info("All visualizations complete.")

if __name__ == "__main__":
    main()
