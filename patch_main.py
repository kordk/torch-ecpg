import sys

def patch_file():
    with open('tools/visualizeFindings.py', 'r') as f:
        content = f.read()

    main_start = content.find('def main():')
    old_main_body = content[main_start:]

    new_main = """def main():
    parser = argparse.ArgumentParser(description="Generate Multi-Omic Diagnostic Plots")
    parser.add_argument("parquet_file", help="Path to the annotated Parquet results file")

    parser.add_argument("--out-dir", default="plots", help="Directory to save plots")

    # Selection of plots
    parser.add_argument("--volcano", action="store_true", help="Generate Volcano Plot")
    parser.add_argument("--manhattan", action="store_true", help="Generate Manhattan Plot")
    parser.add_argument("--region-breakdown", action="store_true", help="Generate Region Breakdown Plot")
    parser.add_argument("--scatter", action="store_true", help="Generate Scatter Plots for top hits")
    parser.add_argument("--comparative-scatter", action="store_true", help="Generate Comparative (Unadjusted vs Adjusted) Scatter Plots for top hits")
    parser.add_argument("--all", action="store_true", help="Generate all plots (default if none specified)")

    # Tecpg paths for scatter plots
    parser.add_argument("-m", "--meth-file", required=False, help="Full path to the methylation matrix file (e.g., M.csv)")
    parser.add_argument("-g", "--gene-file", required=False, help="Full path to the gene expression matrix file (e.g., G.csv)")
    parser.add_argument("-c", "--covar-file", required=False, help="Full path to the covariate matrix file (e.g., C.csv)")

    args = parser.parse_args()

    # If no specific plot is requested, do all
    if not (args.volcano or args.manhattan or args.region_breakdown or args.scatter or args.comparative_scatter):
        args.all = True

    logger.info(f"Inspecting schema of {args.parquet_file}...")
    try:
        parquet_file = pq.ParquetFile(args.parquet_file)
        col_names = parquet_file.schema_arrow.names
    except Exception as e:
        logger.error(f"Failed to read parquet file schema: {e}")
        return

    p_configs = []

    has_boot = 'p_boot' in col_names
    has_precise = 'precise_mt_p' in col_names
    has_mt = 'mt_p' in col_names

    if has_boot:
        p_configs.append(('p_boot', 'bootstrapP_'))
    else:
        logger.info("p_boot column not found in parquet file. Bootstrap plots will not be generated.")

    if has_precise:
        p_configs.append(('precise_mt_p', 'preciseP_'))
    else:
        logger.info("precise_mt_p column not found in parquet file. Precise p-value plots will not be generated.")

    if not has_boot and not has_precise:
        if has_mt:
            logger.info("Fallback to mt_p since neither p_boot nor precise_mt_p were found.")
            p_configs.append(('mt_p', 'mtP_'))
        else:
            logger.error("No valid p-value columns (p_boot, precise_mt_p, mt_p) found. Exiting.")
            return

    # Load scatter plot matrices once if needed
    M_df, G_df, C_df = None, None, None
    if args.all or args.scatter or args.comparative_scatter:
        meth_path = args.meth_file
        gene_path = args.gene_file
        covar_path = args.covar_file

        if not meth_path or not gene_path or not covar_path:
            logger.error("Scatter plots requested but one or more data file paths (-m, -g, -c) are missing. Please provide them.")
            return

        if not (os.path.exists(meth_path) and os.path.exists(gene_path) and os.path.exists(covar_path)):
            logger.error(f"Missing one or more data files for scatter plots: {meth_path}, {gene_path}, {covar_path}")
            return

        try:
            logger.info("Loading methylation, gene, and covariate matrices into memory...")
            M_df = pd.read_csv(meth_path, index_col=0)
            G_df = pd.read_csv(gene_path, index_col=0)
            C_df = pd.read_csv(covar_path, index_col=0)

            # Ensure subject IDs align by casting all to string
            M_df.columns = M_df.columns.astype(str)
            G_df.columns = G_df.columns.astype(str)
            C_df.index = C_df.index.astype(str)

            # Log sample info for visual confirmation of overlap
            logger.info(f"M_df subjects (first 5): {list(M_df.columns[:5])}")
            logger.info(f"G_df subjects (first 5): {list(G_df.columns[:5])}")
            logger.info(f"C_df subjects (first 5): {list(C_df.index[:5])}")

        except Exception as e:
            logger.error(f"Failed to read data matrices: {e}")
            return

    for p_col, prefix in p_configs:
        logger.info(f"=== Generating plots for {p_col} with prefix '{prefix}' ===")
        plotter = Plotter(args.parquet_file, args.out_dir, p_col, prefix)

        try:
            plotter.load_and_subsample_data()
        except Exception as e:
            logger.error(f"Failed to load data for {p_col}: {e}")
            continue

        if args.all or args.volcano:
            plotter.plot_volcano()

        if args.all or args.manhattan:
            plotter.plot_manhattan()

        if args.all or args.region_breakdown:
            plotter.plot_region_breakdown()

        if args.all or args.scatter or args.comparative_scatter:
            logger.info(f"Preparing to generate scatter plots for top hits per region using {p_col}...")

            if 'fdr' not in plotter.df.columns and plotter.p_col not in plotter.df.columns:
                logger.warning("FDR or p-value column missing, cannot determine top hits.")
                continue

            sort_col = 'fdr' if 'fdr' in plotter.df.columns else plotter.p_col
            logger.info(f"Sorting by {sort_col} to find top hits.")

            regions = plotter.df['region'].dropna().unique() if 'region' in plotter.df.columns else ["All"]

            for region in regions:
                if 'region' in plotter.df.columns:
                    region_df = plotter.df[plotter.df['region'] == region]
                else:
                    region_df = plotter.df

                top_hits = region_df.nsmallest(10, sort_col)

                for _, row in top_hits.iterrows():
                    cpg_id = row['mt_id']
                    gene_id = row['gt_id']

                    if args.all or args.comparative_scatter:
                        plot_comparative_eqtm(cpg_id, gene_id, row, plotter.p_col, prefix, M_df, G_df, C_df, args.out_dir, region)
                    elif args.scatter:
                        plot_adjusted_eqtm(cpg_id, gene_id, row, plotter.p_col, prefix, M_df, G_df, C_df, args.out_dir, region)

if __name__ == "__main__":
    main()
"""

    content = content[:main_start] + new_main

    with open('tools/visualizeFindings.py', 'w') as f:
        f.write(content)

patch_file()
