import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

df = pd.DataFrame({
    'mt_chrom': ['chr1', '2', 'chr3', '4', 'chr5'],
    'mt_chromStart': [100, 200, np.nan, 400, 500],
    'gt_chrom': ['1', 'chr2', '3', 'chr4', '5'],
    'gt_chromStart': [150, 250, 350, np.nan, 550],
    'mt_est': [0.1, 0.2, 0.3, 0.4, 0.5],
    'mt_ig': [1.1, 1.2, 1.3, 1.4, 1.5],
    'mt_id': ['cg00000001', 'ch.123', 'rs456', 'cg00000002', 'cg00000003'],
    'gt_id': ['ILMN_123', 'ENSG000001', 'ENSG000002', 'ILMN_456', 'ENSG000003']
})

table = pa.Table.from_pandas(df)
pq.write_table(table, 'test_data.parquet')

with open("test_cytoband.txt", "w") as f:
    f.write("chr1\t0\t249250621\tp36.33\tgneg\n")
    f.write("chr2\t0\t243199373\tp25.3\tgneg\n")
    f.write("chr3\t0\t198022430\tp26.3\tgneg\n")
    f.write("chr4\t0\t191154276\tp16.3\tgneg\n")
    f.write("chr5\t0\t180915260\tp15.33\tgneg\n")
