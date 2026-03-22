import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a publication-quality Circos plot visualizing global eQTM architecture.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input Parquet file (e.g., results.precise_p.annot.fdr.parquet).\n"
             "Must contain: mt_chrom, mt_chromStart, gt_chrom, gt_chromStart, mt_est, mt_ig."
    )

    parser.add_argument(
        "--cytoband", "-c",
        required=True,
        help="Path to the cytoband (karyotype) file for drawing the genome perimeter.\n"
             "You can download this from UCSC Genome Browser, for example:\n"
             "  curl -O http://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/cytoBand.txt.gz\n"
             "  gunzip cytoBand.txt.gz\n"
             "Then provide the extracted cytoBand.txt file here."
    )

    parser.add_argument(
        "--top-n", "-n",
        type=int,
        default=5000,
        help="Number of top edges (by mt_ig) to plot in the Top Saliency plot. Default: 5000."
    )

    parser.add_argument(
        "--top-n-trans", "-t",
        type=int,
        default=2000,
        help="Number of top edges (by mt_ig) to plot in the Trans-Only Focus plot. Default: 2000."
    )

    parser.add_argument(
        "--out-dir", "-o",
        default="plots/",
        help="Directory to save the generated plots. Default: 'plots/'."
    )

    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Save plots in PDF format instead of the default PNG."
    )

    return parser.parse_args()

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import pycircos

def load_and_validate_data(filepath):
    print(f"Loading data from {filepath}...")

    # We use pq.ParquetFile to inspect schema without loading full file
    try:
        parquet_file = pq.ParquetFile(filepath)
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        sys.exit(1)

    required_cols = ['mt_chrom', 'mt_chromStart', 'gt_chrom', 'gt_chromStart', 'mt_est', 'mt_ig']
    schema_cols = parquet_file.schema.names

    missing_cols = [col for col in required_cols if col not in schema_cols]
    if missing_cols:
        print(f"Error: Missing required columns in input Parquet file: {', '.join(missing_cols)}")
        print(f"The input must contain all of: {', '.join(required_cols)}")
        sys.exit(1)

    print(f"Validation passed. All required columns present.")

    # Load required columns + mt_id for density calculations
    # checking if mt_id is present
    cols_to_load = required_cols.copy()
    if 'mt_id' in schema_cols:
        cols_to_load.append('mt_id')

    # Load into pandas dataframe
    df = pd.read_parquet(filepath, columns=cols_to_load)

    # We strictly cast chroms to str, and coords to int
    df['mt_chrom'] = df['mt_chrom'].astype(str)
    df['gt_chrom'] = df['gt_chrom'].astype(str)

    # Handle 'chr' prefix formatting
    df['mt_chrom'] = df['mt_chrom'].apply(lambda x: f"chr{x}" if not x.startswith('chr') else x)
    df['gt_chrom'] = df['gt_chrom'].apply(lambda x: f"chr{x}" if not x.startswith('chr') else x)

    # Strip decimals if present, then cast to int
    df['mt_chromStart'] = df['mt_chromStart'].astype(float).astype(int)
    df['gt_chromStart'] = df['gt_chromStart'].astype(float).astype(int)

    return df

def filter_data(df, top_n, top_n_trans):
    print(f"Applying filters...")

    # 1. Top Saliency Filter
    # Sort by mt_ig descending
    df_sorted = df.sort_values(by='mt_ig', ascending=False)

    df_top_saliency = df_sorted.head(top_n).copy()
    print(f"Top Saliency dataset filtered to top {len(df_top_saliency)} rows by mt_ig.")

    # 2. Trans-Only Focus Filter
    # Filter for trans-chromosomal links (mt_chrom != gt_chrom)
    df_trans = df_sorted[df_sorted['mt_chrom'] != df_sorted['gt_chrom']]
    df_trans_only = df_trans.head(top_n_trans).copy()
    print(f"Trans-Only Focus dataset filtered to top {len(df_trans_only)} trans rows by mt_ig.")

    return df_top_saliency, df_trans_only

def create_circos_plot(df_all, df_filtered, cytoband_file, out_path, title):
    print(f"Generating Circos plot: {title}")

    # Read cytoband data
    try:
        # Standard UCSC cytoBand.txt format: chrom, chromStart, chromEnd, name, gieStain
        cytoband_df = pd.read_csv(cytoband_file, sep='\t', header=None,
                                  names=['chrom', 'chromStart', 'chromEnd', 'name', 'gieStain'])
    except Exception as e:
        print(f"Error reading cytoband file: {e}")
        sys.exit(1)

    # Standard chromosomes to include
    standard_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    cytoband_df = cytoband_df[cytoband_df['chrom'].isin(standard_chroms)]

    # Prepare lengths for pycircos.Circos
    chrom_lengths_list = []
    chrom_lengths_dict = {}
    for chrom in standard_chroms:
        chrom_data = cytoband_df[cytoband_df['chrom'] == chrom]
        if not chrom_data.empty:
            max_len = chrom_data['chromEnd'].max()
            chrom_lengths_list.append({'id': chrom, 'length': max_len})
            chrom_lengths_dict[chrom] = max_len

    if not chrom_lengths_list:
        print("Error: No standard chromosomes found in the cytoband file.")
        sys.exit(1)

    chrom_df = pd.DataFrame(chrom_lengths_list).set_index('id')

    # 1. Initialize Circos
    circle = pycircos.Circos(chrom_df)

    # Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'polar': True})

    # In pycircos, you associate ax with circle
    circle.pax = ax

    # To avoid Hatch error with newer matplotlib, we explicitly pass hatch='' to bar or fill=True
    circle.draw_scaffold(rad=0.98, width=0.04, fill=False, hatch='')

    circle.draw_cytobands(rad=0.98, width=0.04, cbfile=cytoband_file)
    circle.draw_scaffold_ids(rad=1.05, fontsize=12)

    # 2. Add Density Tracks (Histograms)
    # 1MB bins
    bin_size = 1000000

    def add_density_track(data_df, r, color, track_name, height=0.08):
        if data_df.empty:
            return

        for chrom in standard_chroms:
            if chrom not in chrom_lengths_dict:
                continue

            chrom_cpgs = data_df[data_df['mt_chrom'] == chrom]
            if chrom_cpgs.empty:
                continue

            if 'mt_id' in chrom_cpgs.columns:
                chrom_cpgs = chrom_cpgs.drop_duplicates(subset=['mt_id'])

            chrom_len = chrom_lengths_dict[chrom]
            bins = np.arange(0, chrom_len + bin_size, bin_size)
            counts, _ = np.histogram(chrom_cpgs['mt_chromStart'], bins=bins)

            if counts.max() > 0:
                # Normalize counts relative to maximum to fit in the 'height'
                norm_counts = counts / counts.max() * height

                # Positions in theta
                # pycircos get_theta expects lists of equal length for iterable pos
                gids = [chrom] * len(bins[:-1])
                theta_starts = circle.get_theta(gids, bins[:-1])
                theta_ends = circle.get_theta(gids, bins[1:])

                # Format data for fill_between
                fill_data = []
                for i in range(len(counts)):
                    if norm_counts[i] > 0:
                        fill_data.append({
                            'chrom': chrom,
                            'start': bins[i],
                            'end': bins[i+1],
                            'score': norm_counts[i]
                        })

                if fill_data:
                    # In newer matplotlib + polar plots, fill_between on discrete segments can trigger autoscale bugs.
                    # We will draw these bins manually as polygons using matplotlib patches to ensure safety and control.
                    for item in fill_data:
                        t1, t2 = circle.get_theta([item['chrom'], item['chrom']], [item['start'], item['end']])
                        h = item['score']

                        # A bin is a polygon in polar coordinates from r to r+h between t1 and t2
                        # To approximate the curve, we can just use a simple polygon for narrow bins
                        poly = mpatches.Polygon(
                            [[t1, r], [t2, r], [t2, r + h], [t1, r + h]],
                            closed=True,
                            facecolor=color,
                            edgecolor='none'
                        )
                        ax.add_patch(poly)

    # Outer Density (All CpGs)
    print("Computing global density track...")
    add_density_track(df_all, r=0.88, color='#888888', track_name="Global Density", height=0.08)

    # Inner Density (Filtered CpGs)
    print("Computing filtered density track...")
    add_density_track(df_filtered, r=0.78, color='#E69F00', track_name="Filtered Density", height=0.08)

    # 3. Add Link Tracks (The Connections)
    print("Drawing connection links...")
    # Scale linewidths linearly by mt_ig
    min_ig = df_filtered['mt_ig'].min()
    max_ig = df_filtered['mt_ig'].max()

    if max_ig == min_ig:
        max_ig = min_ig + 1

    for _, row in df_filtered.iterrows():
        source_chrom = row['mt_chrom']
        source_start = row['mt_chromStart']
        target_chrom = row['gt_chrom']
        target_start = row['gt_chromStart']

        if source_chrom not in chrom_lengths_dict or target_chrom not in chrom_lengths_dict:
            continue

        if row['mt_est'] > 0:
            link_color = "#FF0000"
            alpha = 0.3
        else:
            link_color = "#0000FF"
            alpha = 0.3

        norm_val = (row['mt_ig'] - min_ig) / (max_ig - min_ig)
        lw = 0.1 + (2.9 * norm_val)

        # draw_link method in pycircos 1.0.2 doesn't explicitly accept linewidth
        # However, we can add it directly to the patch since it uses matplotlib PathPatch
        ets = circle.get_theta([source_chrom, target_chrom], [source_start, target_start])
        ete = circle.get_theta([source_chrom, target_chrom], [source_start + 1, target_start + 1])
        rad = 0.75
        points = [(ets[0], rad),  # start1
                  ((ets[0]+ete[0])/2, rad),  # through point
                  (ete[0], rad),  # end 1
                  (0, 0),  # through point
                  (ets[1], rad),  # start2
                  ((ets[1]+ete[1])/2, rad),  # through point
                  (ete[1], rad),  # end2
                  (0, 0),  # through point
                  (ets[0], rad)]

        codes = [mpath.Path.CURVE3]*len(points)
        codes[0] = mpath.Path.MOVETO
        path = mpath.Path(points, codes)

        # Setting linewidth using `lw` argument
        patch = mpatches.PathPatch(path, facecolor=link_color, edgecolor=link_color, lw=lw, alpha=alpha)
        circle.pax.add_patch(patch)

    # 4. Save Plot
    print(f"Saving plot to {out_path}...")

    ax.set_title(title, y=1.05, fontsize=16)

    # Turn off axis cleanly for polar plot
    ax.set_axis_off()

    # Ensure axes limits are set to avoid autoscale crashes in newer matplotlib versions
    # Sometimes setting xlim directly triggers the autoscale error, so we will disable autoscaling completely.
    try:
        ax.set_rlim(0, 1.2)
    except Exception as e:
        pass

    # Optional tight_layout removed, manual limits should handle spacing
    try:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        # If bbox_inches tight crashes due to autoscale, fallback to standard save
        plt.savefig(out_path, dpi=300)
    plt.close(fig)


import os

def main():
    args = parse_args()
    print(f"Arguments parsed successfully: {args}")

    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_and_validate_data(args.input)
    print(f"Loaded {len(df)} records.")

    df_top_saliency, df_trans_only = filter_data(df, args.top_n, args.top_n_trans)

    ext = "pdf" if args.pdf else "png"

    # Generate Top Saliency Plot
    out_path_saliency = os.path.join(args.out_dir, f"circos_top_saliency.{ext}")
    create_circos_plot(
        df_all=df,
        df_filtered=df_top_saliency,
        cytoband_file=args.cytoband,
        out_path=out_path_saliency,
        title=f"Global eQTM Architecture: Top {len(df_top_saliency)} Saliency Links"
    )

    # Generate Trans-Only Plot
    out_path_trans = os.path.join(args.out_dir, f"circos_trans_only.{ext}")
    create_circos_plot(
        df_all=df,
        df_filtered=df_trans_only,
        cytoband_file=args.cytoband,
        out_path=out_path_trans,
        title=f"Global eQTM Architecture: Top {len(df_trans_only)} Trans-Chromosomal Links"
    )

    print("Done!")

if __name__ == "__main__":
    main()
