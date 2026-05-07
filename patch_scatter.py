import sys

def patch_file():
    with open('tools/visualizeFindings.py', 'r') as f:
        content = f.read()

    # Update plot_comparative_eqtm signature
    old_comp_sig = """def plot_comparative_eqtm(cpg_id: str, gene_id: str, row: pd.Series, p_col: str, M: pd.DataFrame, G: pd.DataFrame, C: pd.DataFrame, output_dir: str, region: str = "Unknown"):"""
    new_comp_sig = """def plot_comparative_eqtm(cpg_id: str, gene_id: str, row: pd.Series, p_col: str, prefix: str, M: pd.DataFrame, G: pd.DataFrame, C: pd.DataFrame, output_dir: str, region: str = "Unknown"):"""
    content = content.replace(old_comp_sig, new_comp_sig)

    # Update plot_comparative_eqtm output path
    old_comp_out = """    out_path = os.path.join(output_dir, f'comparative_scatter_{safe_region}_{cpg_id}_{gene_id}.png')"""
    new_comp_out = """    out_path = os.path.join(output_dir, f'{prefix}comparative_scatter_{safe_region}_{cpg_id}_{gene_id}.png')"""
    content = content.replace(old_comp_out, new_comp_out)

    # Update plot_adjusted_eqtm signature
    old_adj_sig = """def plot_adjusted_eqtm(cpg_id: str, gene_id: str, row: pd.Series, p_col: str, M: pd.DataFrame, G: pd.DataFrame, C: pd.DataFrame, output_dir: str, region: str = "Unknown"):"""
    new_adj_sig = """def plot_adjusted_eqtm(cpg_id: str, gene_id: str, row: pd.Series, p_col: str, prefix: str, M: pd.DataFrame, G: pd.DataFrame, C: pd.DataFrame, output_dir: str, region: str = "Unknown"):"""
    content = content.replace(old_adj_sig, new_adj_sig)

    # Update plot_adjusted_eqtm output path
    old_adj_out = """    out_path = os.path.join(output_dir, f'scatter_{safe_region}_{cpg_id}_{gene_id}.png')"""
    new_adj_out = """    out_path = os.path.join(output_dir, f'{prefix}scatter_{safe_region}_{cpg_id}_{gene_id}.png')"""
    content = content.replace(old_adj_out, new_adj_out)

    with open('tools/visualizeFindings.py', 'w') as f:
        f.write(content)

patch_file()
