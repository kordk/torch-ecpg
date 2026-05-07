import pandas as pd
import numpy as np
import os

os.makedirs('test_data', exist_ok=True)
pd.DataFrame({
    'mt_est': [0.1, 0.2, 0.3],
    'p_boot': [1e-6, 1e-3, 0.5],
    'precise_mt_p': [1e-7, 1e-4, 0.6],
    'region': ['promoter', 'body', 'enhancer'],
    'gt_id': ['geneA', 'geneB', 'geneC'],
    'mt_id': ['cpg1', 'cpg2', 'cpg3'],
    'mt_chrom': ['chr1', 'chr2', 'chr3'],
    'mt_chromStart': [100, 200, 300],
    'fdr': [1e-5, 1e-2, 0.8]
}).to_parquet('test_data/test.parquet')

# dummy matrices
M = pd.DataFrame(np.random.rand(3, 2), index=['cpg1', 'cpg2', 'cpg3'], columns=['sub1', 'sub2'])
G = pd.DataFrame(np.random.rand(3, 2), index=['geneA', 'geneB', 'geneC'], columns=['sub1', 'sub2'])
C = pd.DataFrame(np.random.rand(2, 2), index=['sub1', 'sub2'], columns=['cov1', 'cov2'])

M.to_csv('test_data/M.csv')
G.to_csv('test_data/G.csv')
C.to_csv('test_data/C.csv')
