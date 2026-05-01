import pandas as pd

df_tecpg = pd.DataFrame({
    'mt_id': ['cg1', 'cg2', 'cg3'],
    'gt_id': ['g1', 'g2', 'g1'],
    'fdr_est': [0.01, 0.04, 0.1]
})
df_kennedy = pd.DataFrame({
    'cpg': ['cg1', 'cg2', 'cg4'],
    'gene': ['g1', 'g2', 'g4'],
    'p.val': [0.001, 0.2, 0.0001]
})

tecpg_fdr_col = 'fdr_est'
tecpg_thresh = 0.05
pval_col = 'p.val'
args_p_thresh = 0.05

cpg_col = 'cpg'
query_col = 'gene'

tecpg_all_mappings = set(zip(df_tecpg['mt_id'], df_tecpg['gt_id']))
tecpg_all_genes = set(df_tecpg['gt_id'].dropna())
tecpg_all_loci = set(df_tecpg['mt_id'].dropna())

kennedy_all_mappings = set(zip(df_kennedy[cpg_col], df_kennedy[query_col]))
kennedy_all_genes = set(df_kennedy[query_col].dropna())
kennedy_all_loci = set(df_kennedy[cpg_col].dropna())

tecpg_sig_df = df_tecpg[df_tecpg[tecpg_fdr_col] < tecpg_thresh] if tecpg_fdr_col else df_tecpg
tecpg_sig_mappings = set(zip(tecpg_sig_df['mt_id'], tecpg_sig_df['gt_id']))
tecpg_sig_genes = set(tecpg_sig_df['gt_id'].dropna())
tecpg_sig_loci = set(tecpg_sig_df['mt_id'].dropna())

kennedy_sig_df = df_kennedy[df_kennedy[pval_col] < args_p_thresh]
kennedy_sig_mappings = set(zip(kennedy_sig_df[cpg_col], kennedy_sig_df[query_col]))
kennedy_sig_genes = set(kennedy_sig_df[query_col].dropna())
kennedy_sig_loci = set(kennedy_sig_df[cpg_col].dropna())

print("All Mappings:", len(tecpg_all_mappings), len(kennedy_all_mappings))
print("Sig Mappings:", len(tecpg_sig_mappings), len(kennedy_sig_mappings))
