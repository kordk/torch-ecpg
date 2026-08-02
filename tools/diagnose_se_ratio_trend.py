#!/usr/bin/env python3
"""Diagnose SE ratio trend across regions."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import eval_permute as E
    CANONICAL_REGIONS = E.CANONICAL_REGIONS
except ImportError:
    CANONICAL_REGIONS = ['TRANS', 'DISTAL5', 'CIS5', 'PROMOTER', 'GENEBODY', 'CIS3', 'DISTAL3']

try:
    import tecpg
    TEC_VERSION = tecpg.__version__
except ImportError:
    TEC_VERSION = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="path to the concordance parquet")
    parser.add_argument("-s", "--summary-json", help="path to write the JSON summary")
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--min-region-n", type=int, default=200)
    parser.add_argument("--ci-resamples", type=int, default=1000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    required_cols = ["mt_t", "boot_se_ratio"]
    try:
        parquet_file = pq.ParquetFile(args.input)
    except Exception as e:
        print(f"Error opening parquet file: {e}", file=sys.stderr)
        sys.exit(1)

    schema_names = parquet_file.schema.names
    missing = [c for c in required_cols if c not in schema_names]
    if missing:
        print(f"Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    has_region = "region" in schema_names
    read_cols = required_cols + (["region"] if has_region else [])

    n_rows_read = 0
    scored_rows = []

    for batch in parquet_file.iter_batches(batch_size=args.chunk_size, columns=read_cols):
        n_rows_read += batch.num_rows
        df = batch.to_pandas()

        valid = np.isfinite(df["boot_se_ratio"]) & np.isfinite(df["mt_t"])
        if not valid.any():
            continue

        sub = df[valid].copy()
        sub["mt_t_abs"] = np.abs(sub["mt_t"])
        scored_rows.append(sub)

    if n_rows_read == 0:
        print("Input has zero rows", file=sys.stderr)
        sys.exit(1)

    if not scored_rows:
        print("Input has zero rows with finite boot_se_ratio and finite mt_t", file=sys.stderr)
        sys.exit(1)

    df_scored = pd.concat(scored_rows, ignore_index=True)
    total_scored = len(df_scored)

    regions_data = {}

    if not has_region:
        df_scored["assigned_region"] = "unassigned"
    else:
        # Route to region
        def map_region(r):
            if pd.isna(r):
                return "unassigned"
            if r in CANONICAL_REGIONS:
                return r
            return "unassigned"

        df_scored["assigned_region"] = df_scored["region"].apply(map_region)

    rng = np.random.default_rng(args.seed)

    out_regions = {}
    for region_name, group in df_scored.groupby("assigned_region"):
        n_scored = len(group)
        t_abs = group["mt_t_abs"].to_numpy(dtype=np.float64)
        r = group["boot_se_ratio"].to_numpy(dtype=np.float64)

        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
        t_abs_min = float(np.min(t_abs))
        t_abs_max = float(np.max(t_abs))

        reg_dict = {
            "n_scored": n_scored,
            "median_se_ratio": med,
            "mad_se_ratio": mad,
            "t_abs_min": t_abs_min,
            "t_abs_max": t_abs_max,
            "spearman_rho": None,
            "spearman_ci": None,
            "spearman_ci_omitted_reason": None,
            "trend_omitted_reason": None,
            "bins": None
        }

        if region_name == "unassigned":
            if has_region:
                null_mask = group["region"].isna()
                n_null = int(null_mask.sum())
                n_noncan = n_scored - n_null
                noncan_labels = group.loc[~null_mask, "region"].unique()
                noncan_labels_sorted = sorted([str(x) for x in noncan_labels])
                reg_dict["n_null_region"] = n_null
                reg_dict["n_noncanonical_region"] = n_noncan
                reg_dict["noncanonical_labels"] = noncan_labels_sorted[:20]
                reg_dict["noncanonical_labels_truncated"] = len(noncan_labels_sorted) > 20
            else:
                reg_dict["n_null_region"] = n_scored
                reg_dict["n_noncanonical_region"] = 0
                reg_dict["noncanonical_labels"] = []
                reg_dict["noncanonical_labels_truncated"] = False

        if n_scored < args.min_region_n:
            reg_dict["trend_omitted_reason"] = "n_scored below --min-region-n"
        else:
            q = np.linspace(0, 1, args.bins + 1)
            edges = np.quantile(t_abs, q)
            # Ensure bins partition exactly. searchsorted gives index 1 for first bin
            # interior edges: edges[1:-1]
            bin_idx = np.searchsorted(edges[1:-1], t_abs, side='right')

            bins = []
            for b in range(args.bins):
                b_mask = (bin_idx == b)
                n_b = int(np.sum(b_mask))
                if n_b > 0:
                    t_b = t_abs[b_mask]
                    r_b = r[b_mask]
                    bins.append({
                        "bin": b + 1,
                        "n": n_b,
                        "t_abs_lo": float(np.min(t_b)),
                        "t_abs_hi": float(np.max(t_b)),
                        "median_se_ratio": float(np.median(r_b))
                    })
                else:
                    bins.append({
                        "bin": b + 1,
                        "n": 0,
                        "t_abs_lo": None,
                        "t_abs_hi": None,
                        "median_se_ratio": None
                    })
            reg_dict["bins"] = bins

            rho_val = float(stats.spearmanr(t_abs, r).statistic)
            reg_dict["spearman_rho"] = rho_val

            # Bootstrap CI
            resampled_rhos = []
            indices = np.arange(n_scored)
            for _ in range(args.ci_resamples):
                idx = rng.choice(indices, size=n_scored, replace=True)
                rho = stats.spearmanr(t_abs[idx], r[idx]).statistic
                resampled_rhos.append(rho)

            resampled_rhos = np.array(resampled_rhos, dtype=np.float64)
            finite_rhos = resampled_rhos[np.isfinite(resampled_rhos)]

            if len(finite_rhos) < 50:
                reg_dict["spearman_ci_omitted_reason"] = "fewer than 50 finite resample statistics"
            else:
                alpha = (1 - args.ci_level) / 2
                ci_lo = np.percentile(finite_rhos, alpha * 100)
                ci_hi = np.percentile(finite_rhos, (1 - alpha) * 100)
                reg_dict["spearman_ci"] = [float(ci_lo), float(ci_hi)]

        out_regions[region_name] = reg_dict

    out = {
        "input": os.path.abspath(args.input),
        "tool_version": TEC_VERSION,
        "n_rows_read": n_rows_read,
        "n_scored": total_scored,
        "params": {
            "bins": args.bins,
            "min_region_n": args.min_region_n,
            "ci_resamples": args.ci_resamples,
            "ci_level": args.ci_level,
            "seed": args.seed,
            "chunk_size": args.chunk_size
        },
        "regions": out_regions,
        "notes": {
            "mad_scaling": "reading mad_se_ratio as a standard deviation assumes approximate normality of the region's boot_se_ratio; under that assumption the scale factor is about 1.4826, and the assumption may not hold where the distribution is skewed.",
            "range_restriction": "a region whose scored rows were selected by ranking on precise_mt_p will have its |mt_t| range truncated from below, so a trend measured within that region is measured over a restricted range and may not extend to the region's unscored rows.",
            "trend_reading": "under the assumption that the bootstrap and analytic standard errors estimate the same quantity, a slope that is flat in |mt_t| may be consistent with a specification effect that applies to all rows, whereas a slope that declines as |mt_t| grows may be consistent with an effect confined to the selected subset; the tool computes no verdict and both readings are worth considering against other evidence.",
            "ci_interpretation": "the interval is a percentile bootstrap over rows and assumes the scored rows within a region are exchangeable; if they are not, the interval is likely optimistic."
        }
    }

    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(out, f, indent=2)
    else:
        print(json.dumps(out, indent=2))

    print(f"n_rows_read: {n_rows_read}")
    print(f"n_scored: {total_scored}")
    print(f"params: {out['params']}")
    print()
    print(f"{'region':<15} {'n_scored':<10} {'median_r':<10} {'mad_r':<10} {'t_min':<10} {'t_max':<10} {'rho':<10} {'ci'}")
    for k, v in out_regions.items():
        ci_str = f"[{v['spearman_ci'][0]:.3f}, {v['spearman_ci'][1]:.3f}]" if v['spearman_ci'] else "None"
        rho_str = f"{v['spearman_rho']:.3f}" if v['spearman_rho'] is not None else "None"
        print(f"{k:<15} {v['n_scored']:<10} {v['median_se_ratio']:<10.3f} {v['mad_se_ratio']:<10.3f} {v['t_abs_min']:<10.3f} {v['t_abs_max']:<10.3f} {rho_str:<10} {ci_str}")
        if v["bins"]:
            print(f"  Bins:")
            for b in v["bins"]:
                if b["n"] > 0:
                    print(f"    bin {b['bin']}: n={b['n']}, t_abs in [{b['t_abs_lo']:.3f}, {b['t_abs_hi']:.3f}], med_r={b['median_se_ratio']:.3f}")
                else:
                    print(f"    bin {b['bin']}: n=0")
        print()

if __name__ == "__main__":
    main()
