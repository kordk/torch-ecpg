#!/usr/bin/env python3
"""diagnose_overlap.py -- characterise the assemble overlap disagreement.

    python3 diagnose_overlap.py output_mesa/cis_map_write_all.parquet \
                                output_mesa/sample_reservoir.csv

Replicates build_gene_anchored_master.py's guard (group by (mt_id, gt_id),
max(mt_t) - min(mt_t)) and reports the full distribution rather than one
example, so --mt-t-atol can be chosen from the data instead of guessed.
"""
import sys
import numpy as np
import pandas as pd

cis_path, res_path = sys.argv[1], sys.argv[2]

cis = pd.read_parquet(cis_path, columns=['mt_id', 'gt_id', 'mt_t'])
res = (pd.read_csv(res_path) if res_path.endswith('.csv')
       else pd.read_parquet(res_path))[['mt_id', 'gt_id', 'mt_t']]

print("cis-map   rows: {:,}".format(len(cis)))
print("reservoir rows: {:,}".format(len(res)))

m = cis.merge(res, on=['mt_id', 'gt_id'], suffixes=('_cis', '_res'))
print("overlapping pairs: {:,}".format(len(m)))
if len(m) == 0:
    print("No overlap -- the guard cannot fire. Investigate why.")
    sys.exit(0)

d = (m.mt_t_cis - m.mt_t_res).abs()
rel = d / m.mt_t_cis.abs().clip(lower=1e-12)

print("\n=== |delta mt_t| ===")
for q in (0.50, 0.90, 0.99, 0.999, 1.00):
    print("  q{:<6} {:.3e}".format(q, float(d.quantile(q))))
print("  MAX     {:.6e}".format(float(d.max())))
print("\n=== relative |delta| / |t_cis| ===")
print("  median  {:.3e}".format(float(rel.median())))
print("  MAX     {:.3e}".format(float(rel.max())))

corr = float(np.corrcoef(m.mt_t_cis, m.mt_t_res)[0, 1])
print("\ncorrelation: {:.12f}  (1 - corr = {:.2e})".format(corr, 1 - corr))

bias = float((m.mt_t_cis - m.mt_t_res).mean())
print("mean signed delta: {:+.3e}  (near zero => unbiased noise, not a shift)".format(bias))

for atol in (1e-3, 2e-3, 5e-3, 1e-2):
    n = int((d > atol).sum())
    print("  pairs exceeding {:.0e}: {:,}".format(atol, n))

mx = float(d.max())
rec = float("%.1g" % (mx * 3))
print("\nSuggested --mt-t-atol: {:g}  (3x the observed max {:.3e})".format(rec, mx))
print("\nRead this before acting:")
print("  corr ~ 1 and mean signed delta ~ 0  => float32 noise, benign.")
print("  A systematic shift or corr well below 1 => different covariates or")
print("  sample order. That is NOT a tolerance problem; stop and investigate.")
