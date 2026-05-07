import sys

def patch_file():
    with open('tools/visualizeFindings.py', 'r') as f:
        content = f.read()

    # Apply changes to Plotter.__init__
    old_init = """    def __init__(self, parquet_path: str, output_dir: str):
        self.parquet_path = parquet_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = None"""
    new_init = """    def __init__(self, parquet_path: str, output_dir: str, p_col: str, prefix: str):
        self.parquet_path = parquet_path
        self.output_dir = output_dir
        self.p_col = p_col
        self.prefix = prefix
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = None"""
    content = content.replace(old_init, new_init)

    # Apply changes to load_and_subsample_data
    old_load = """        if 'p_boot' in col_names:
            p_col = 'p_boot'
        elif 'precise_mt_p' in col_names:
            p_col = 'precise_mt_p'
        else:
            p_col = 'mt_p'

        if p_col not in col_names:
            raise ValueError(f"No valid p-value column ('p_boot', 'precise_mt_p', 'mt_p') found in {self.parquet_path}")

        logger.info(f"Using {p_col} as the p-value column.")

        required_cols = ['mt_est', p_col, 'region', 'gt_id', 'mt_id', 'mt_chrom', 'mt_chromStart', 'fdr']
        cols_to_load = [c for c in required_cols if c in col_names]"""

    new_load = """        if self.p_col not in col_names:
            raise ValueError(f"Specified p-value column '{self.p_col}' not found in {self.parquet_path}")

        logger.info(f"Using {self.p_col} as the p-value column.")

        required_cols = ['mt_est', self.p_col, 'region', 'gt_id', 'mt_id', 'mt_chrom', 'mt_chromStart', 'fdr']
        cols_to_load = [c for c in required_cols if c in col_names]"""
    content = content.replace(old_load, new_load)

    # Apply changes to p_vals definition inside iteration
    old_p_vals = """            # Extract p-values
            p_vals = df_batch[p_col]"""
    new_p_vals = """            # Extract p-values
            p_vals = df_batch[self.p_col]"""
    content = content.replace(old_p_vals, new_p_vals)

    # Remove self.p_col = p_col since it's already set
    old_set_p = """        self.df = pd.concat(chunks, ignore_index=True)
        self.p_col = p_col"""
    new_set_p = """        self.df = pd.concat(chunks, ignore_index=True)"""
    content = content.replace(old_set_p, new_set_p)

    # Add prefix to filenames in plot functions
    content = content.replace("""out_path = os.path.join(self.output_dir, 'volcano_plot.png')""", """out_path = os.path.join(self.output_dir, f'{self.prefix}volcano_plot.png')""")
    content = content.replace("""out_path = os.path.join(self.output_dir, 'manhattan_plot.png')""", """out_path = os.path.join(self.output_dir, f'{self.prefix}manhattan_plot.png')""")
    content = content.replace("""out_path = os.path.join(self.output_dir, 'region_breakdown.png')""", """out_path = os.path.join(self.output_dir, f'{self.prefix}region_breakdown.png')""")

    with open('tools/visualizeFindings.py', 'w') as f:
        f.write(content)

patch_file()
