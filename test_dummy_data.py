import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

df_tecpg = pd.DataFrame({
    'mt_id': ['cg1', 'cg2', 'cg3', 'cg5'],
    'gt_id': ['g1', 'g2', 'g1', 'g3'],
    'mt_est': [0.5, -0.2, 0.1, 0.4],
    'mt_t': [2.5, -1.2, 0.8, 2.0],
    'fdr_est': [0.01, 0.04, 0.1, 0.02]
})
table = pa.Table.from_pandas(df_tecpg)
pq.write_table(table, 'test_tecpg.parquet')

df_kennedy = pd.DataFrame({
    'CpG.probe': ['cg1', 'cg2', 'cg4', 'cg5'],
    'annot.gene': ['g1', 'g2', 'g4', 'g3'],
    'beta': [0.45, -0.15, 0.6, 0.35],
    'T.stat': [2.4, -1.0, 3.1, 1.9],
    'p.val': [0.001, 0.2, 0.0001, 0.03]
})
df_kennedy.to_csv('test_kennedy.txt', sep='\t', index=False)
