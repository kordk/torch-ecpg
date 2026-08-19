#!/usr/bin/env python3
import argparse
import json
import os
import sys
import collections

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_permute as E  # noqa: E402

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-c", "--covariates")
    parser.add_argument("-o", "--output")
    parser.add_argument("--rule", choices=["abs", "floor"])
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--flag-column", default="mt_influence_flag")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--fdr-column", default="fdr_est")
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=100000)
    return parser.parse_args()

def main():
    args = parse_args()

    if args.output:
        if not args.rule or args.threshold is None:
            sys.stderr.write("ERROR: -o/--output requires both --rule and --threshold\n")
            sys.exit(1)
    elif args.rule or args.threshold is not None:
        sys.stderr.write("ERROR: --rule and --threshold require -o/--output\n")
        sys.exit(1)

    if args.rule == "floor" and not args.covariates:
        sys.stderr.write("ERROR: --rule floor requires -c/--covariates\n")
        sys.exit(1)

    h_C_max = np.nan
    n_subjects = np.nan
    n_covariates = np.nan
    p_design = np.nan
    p_over_n = np.nan

    if args.covariates:
        C = pd.read_csv(args.covariates, index_col=0)
        n_subjects = len(C)
        n_covariates = len(C.columns)
        p_design = n_covariates + 2
        p_over_n = p_design / n_subjects if n_subjects > 0 else np.nan

        Xc = np.hstack([np.ones((n_subjects, 1), dtype=np.float64), C.to_numpy(dtype=np.float64)])
        Q, _ = np.linalg.qr(Xc, mode='reduced')
        h_C_max = float((Q * Q).sum(axis=1).max())

    pq_file = pq.ParquetFile(args.input)
    schema = pq_file.schema_arrow
    col_names = schema.names

    if "mt_id" not in col_names or "mt_h_max" not in col_names:
        sys.stderr.write("ERROR: mt_id or mt_h_max absent from input\n")
        sys.exit(1)

    if args.output and args.flag_column in col_names:
        sys.stderr.write("ERROR: flag column already present\n")
        sys.exit(1)

    has_fdr = args.fdr_column in col_names
    has_region = "region" in col_names

    cpg_min = {}
    cpg_max = {}
    cpg_has_null = set()
    cpg_n_rows = collections.defaultdict(int)
    cpg_n_sig_rows = collections.defaultdict(int)
    cpg_region_rows = collections.defaultdict(lambda: collections.defaultdict(int))
    cpg_region_sig_rows = collections.defaultdict(lambda: collections.defaultdict(int))

    n_rows_total = 0
    n_rows_null_h = 0

    for batch in pq_file.iter_batches(batch_size=args.chunk_size):
        df = batch.to_pandas()
        if df.index.names != [None]:
            df = df.reset_index()
        n_rows_total += len(df)

        null_mask = df["mt_h_max"].isna()
        n_rows_null_h += int(null_mask.sum())

        valid_df = df[~null_mask]
        null_df = df[null_mask]

        if not valid_df.empty:
            grouped = valid_df.groupby("mt_id")
            b_min = grouped["mt_h_max"].min()
            b_max = grouped["mt_h_max"].max()
            for mt_id, val in b_min.items():
                if mt_id not in cpg_min:
                    cpg_min[mt_id] = val
                    cpg_max[mt_id] = b_max[mt_id]
                else:
                    cpg_min[mt_id] = min(cpg_min[mt_id], val)
                    cpg_max[mt_id] = max(cpg_max[mt_id], b_max[mt_id])

        grouped_all = df.groupby("mt_id")
        b_counts = grouped_all.size()
        for mt_id, val in b_counts.items():
            cpg_n_rows[mt_id] += val

        if not null_df.empty:
            for mt_id in null_df["mt_id"].unique():
                cpg_has_null.add(mt_id)

        if has_fdr:
            sig_mask = df[args.fdr_column] <= args.fdr_threshold
            b_sig_counts = df[sig_mask].groupby("mt_id").size()
            for mt_id, val in b_sig_counts.items():
                cpg_n_sig_rows[mt_id] += val

        if has_region:
            b_reg_counts = df.groupby(["mt_id", "region"]).size()
            for (mt_id, reg), val in b_reg_counts.items():
                cpg_region_rows[mt_id][reg] += val

            if has_fdr:
                b_reg_sig_counts = df[sig_mask].groupby(["mt_id", "region"]).size()
                for (mt_id, reg), val in b_reg_sig_counts.items():
                    cpg_region_sig_rows[mt_id][reg] += val
        else:
            for mt_id, val in b_counts.items():
                cpg_region_rows[mt_id]["ALL"] += val

            if has_fdr:
                b_reg_sig_counts = df[sig_mask].groupby("mt_id").size()
                for mt_id, val in b_reg_sig_counts.items():
                    cpg_region_sig_rows[mt_id]["ALL"] += val

    for mt_id in cpg_min:
        if mt_id in cpg_has_null:
            sys.stderr.write(f"ERROR: mt_h_max not constant for mt_id {mt_id}\n")
            sys.exit(1)
        if cpg_max[mt_id] - cpg_min[mt_id] > 1e-6:
            sys.stderr.write(f"ERROR: mt_h_max not constant for mt_id {mt_id}\n")
            sys.exit(1)

    cpg_ids = list(cpg_n_rows.keys())
    n_cpgs = len(cpg_ids)

    cpg_df = pd.DataFrame({
        "mt_id": cpg_ids,
        "mt_h_max": [cpg_min.get(k, np.nan) for k in cpg_ids],
        "n_rows": [cpg_n_rows[k] for k in cpg_ids],
        "n_sig_rows": [cpg_n_sig_rows[k] for k in cpg_ids],
    })

    cpg_df["h_excess"] = cpg_df["mt_h_max"] - h_C_max if not np.isnan(h_C_max) else np.nan

    if has_region:
        recognized_regions = set(E.CANONICAL_REGIONS)
        all_regions = set()
        for mt_id in cpg_region_rows:
            all_regions.update(cpg_region_rows[mt_id].keys())

        unrecognized = all_regions - recognized_regions
        if unrecognized:
            sys.stderr.write(f"WARNING: Unrecognized regions found: {unrecognized}\n")

        region_list = [r for r in E.CANONICAL_REGIONS if r in all_regions] + list(unrecognized)
    else:
        region_list = ["ALL"]

    def get_dist(series):
        if series.empty:
            return {}
        quants = series.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999]).to_dict()
        return {
            "min": float(series.min()),
            "q01": float(quants[0.01]),
            "q05": float(quants[0.05]),
            "q25": float(quants[0.25]),
            "q50": float(quants[0.50]),
            "q75": float(quants[0.75]),
            "q95": float(quants[0.95]),
            "q99": float(quants[0.99]),
            "q999": float(quants[0.999]),
            "max": float(series.max()),
        }

    h_max_dist = get_dist(cpg_df["mt_h_max"]) if not cpg_df.empty else {}
    h_excess_dist = get_dist(cpg_df["h_excess"]) if not np.isnan(h_C_max) and not cpg_df.empty else None

    frac_cpgs_at_floor = float((cpg_df["h_excess"] <= 1e-3).mean()) if not np.isnan(h_C_max) and not cpg_df.empty else None

    per_region_stats = {}
    for reg in region_list:
        reg_rows = sum(cpg_region_rows[mt_id].get(reg, 0) for mt_id in cpg_ids)
        reg_cpgs = sum(1 for mt_id in cpg_ids if reg in cpg_region_rows[mt_id])

        reg_mt_ids = [mt_id for mt_id in cpg_ids if reg in cpg_region_rows[mt_id] and mt_id in cpg_min]
        reg_median = float(np.median([cpg_min[mt_id] for mt_id in reg_mt_ids])) if reg_mt_ids else None

        stats = {
            "n_rows": int(reg_rows),
            "n_cpgs": int(reg_cpgs),
            "median_mt_h_max": reg_median,
        }

        if has_fdr:
            reg_sig_rows = sum(cpg_region_sig_rows[mt_id].get(reg, 0) for mt_id in cpg_ids)
            reg_sig_cpgs = sum(1 for mt_id in cpg_ids if cpg_region_sig_rows[mt_id].get(reg, 0) > 0)
            stats["n_sig_rows"] = int(reg_sig_rows)
            stats["n_sig_cpgs"] = int(reg_sig_cpgs)

        per_region_stats[reg] = stats

    sweep_abs = {}
    tau_list = [0.3, 0.5, 0.7, 0.9, 0.95]
    for tau in tau_list:
        if cpg_df.empty:
            sweep_abs[str(tau)] = {"frac_cpgs_flagged": 0.0}
            continue
        flagged_cpgs = set(cpg_df[cpg_df["mt_h_max"] > tau]["mt_id"])
        frac_cpgs = len(flagged_cpgs) / n_cpgs if n_cpgs > 0 else 0.0

        res = {"frac_cpgs_flagged": float(frac_cpgs)}
        if has_fdr and has_region:
            reg_res = {}
            for reg in region_list:
                total_sig = per_region_stats[reg].get("n_sig_rows", 0)
                flagged_sig = sum(cpg_region_sig_rows[mt_id].get(reg, 0) for mt_id in flagged_cpgs)
                reg_res[reg] = flagged_sig / total_sig if total_sig > 0 else 0.0
            res["frac_sig_rows_flagged"] = reg_res
        sweep_abs[str(tau)] = res

    sweep_floor = None
    if not np.isnan(h_C_max):
        sweep_floor = {}
        delta_list = [0.05, 0.10, 0.20, 0.30, 0.50]
        for delta in delta_list:
            if cpg_df.empty:
                sweep_floor[str(delta)] = {"frac_cpgs_flagged": 0.0}
                continue
            flagged_cpgs = set(cpg_df[cpg_df["h_excess"] > delta]["mt_id"])
            frac_cpgs = len(flagged_cpgs) / n_cpgs if n_cpgs > 0 else 0.0

            res = {"frac_cpgs_flagged": float(frac_cpgs)}
            if has_fdr and has_region:
                reg_res = {}
                for reg in region_list:
                    total_sig = per_region_stats[reg].get("n_sig_rows", 0)
                    flagged_sig = sum(cpg_region_sig_rows[mt_id].get(reg, 0) for mt_id in flagged_cpgs)
                    reg_res[reg] = flagged_sig / total_sig if total_sig > 0 else 0.0
                res["frac_sig_rows_flagged"] = reg_res
            sweep_floor[str(delta)] = res

    chosen_stats = None
    n_cpgs_flagged = 0
    if args.output:
        if cpg_df.empty:
            flagged_mt_ids = set()
        else:
            flag_mask = (cpg_df["mt_h_max"] > args.threshold) if args.rule == "abs" else (cpg_df["h_excess"] > args.threshold)
            flagged_mt_ids = set(cpg_df[flag_mask]["mt_id"])

        n_cpgs_flagged = len(flagged_mt_ids)

        chosen_stats = {
            "n_cpgs_flagged": int(n_cpgs_flagged),
            "per_region": {}
        }
        for reg in region_list:
            reg_total_rows = per_region_stats[reg]["n_rows"]
            reg_flagged_rows = sum(cpg_region_rows[mt_id].get(reg, 0) for mt_id in flagged_mt_ids)
            frac_rows = reg_flagged_rows / reg_total_rows if reg_total_rows > 0 else 0.0

            c = {
                "n_rows_flagged": int(reg_flagged_rows),
                "frac_rows_flagged": float(frac_rows),
                "n_cpgs_flagged": sum(1 for mt_id in flagged_mt_ids if reg in cpg_region_rows[mt_id])
            }
            if has_fdr:
                total_sig = per_region_stats[reg].get("n_sig_rows", 0)
                flagged_sig = sum(cpg_region_sig_rows[mt_id].get(reg, 0) for mt_id in flagged_mt_ids)
                c["frac_sig_rows_flagged"] = flagged_sig / total_sig if total_sig > 0 else 0.0
            chosen_stats["per_region"][reg] = c

    top25 = []
    if not cpg_df.empty:
        top25_df = cpg_df.sort_values("mt_h_max", ascending=False).head(25)
        for _, row in top25_df.iterrows():
            mt_id = row["mt_id"]
            r = {
                "mt_id": str(mt_id),
                "mt_h_max": float(row["mt_h_max"]),
                "h_excess": float(row["h_excess"]) if not np.isnan(row["h_excess"]) else None,
                "n_rows": int(row["n_rows"]),
                "n_sig_rows": int(row["n_sig_rows"]) if has_fdr else None,
            }
            if args.output:
                if args.rule == "abs":
                    r["flagged"] = bool(row["mt_h_max"] > args.threshold)
                else:
                    r["flagged"] = bool(row["h_excess"] > args.threshold)
            top25.append(r)

    mode = "flag" if args.output else "report-only"

    def safe_nan(val):
        if val is None:
            return None
        return None if np.isnan(val) else val

    report = {
        "header": {
            "input_path": args.input,
            "n_rows": int(n_rows_total),
            "n_cpgs": int(n_cpgs),
            "n_rows_null_h": int(n_rows_null_h),
            "n_subjects": safe_nan(n_subjects),
            "n_covariates": safe_nan(n_covariates),
            "p_design": safe_nan(p_design),
            "p_over_n": safe_nan(p_over_n),
            "h_C_max": safe_nan(h_C_max),
            "mode": mode,
            "rule": args.rule,
            "threshold": float(args.threshold) if args.threshold is not None else None,
        },
        "mt_h_max_dist": h_max_dist,
        "h_excess_dist": h_excess_dist if h_excess_dist is not None else None,
        "frac_cpgs_at_floor": safe_nan(frac_cpgs_at_floor),
        "per_region": per_region_stats,
        "sweep_abs": sweep_abs,
        "sweep_floor": sweep_floor,
        "chosen_rule_stats": chosen_stats,
        "top25": top25,
    }

    def clean_nan(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        elif isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan(v) for v in obj]
        return obj

    report = clean_nan(report)

    pngs_written = True
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    os.makedirs(args.report_dir, exist_ok=True)

    if plt is None:
        sys.stderr.write("WARNING: matplotlib unavailable; histograms not written\n")
        pngs_written = False
    else:
        if not cpg_df.empty:
            plt.figure()
            cpg_df["mt_h_max"].hist(bins=50)
            plt.title("Per-CpG mt_h_max")
            if not np.isnan(h_C_max):
                plt.axvline(h_C_max, color="red", linestyle="--", label="h_C_max")
            if args.output and args.rule == "abs":
                plt.axvline(args.threshold, color="black", linestyle="-", label="Threshold")
            plt.legend()
            plt.savefig(os.path.join(args.report_dir, "h_max_hist.png"))
            plt.close()

            if not np.isnan(h_C_max):
                plt.figure()
                cpg_df["h_excess"].hist(bins=50, log=True)
                plt.title("Per-CpG h_excess")
                if args.output and args.rule == "floor":
                    plt.axvline(args.threshold, color="black", linestyle="-", label="Threshold")
                plt.legend()
                plt.savefig(os.path.join(args.report_dir, "h_excess_hist.png"))
                plt.close()

    report["pngs_written"] = pngs_written

    # Python's json handles nan as NaN unless configured, strict compliance is required
    with open(os.path.join(args.report_dir, "influence_qc.json"), "w") as f:
        # allow_nan=False to strictly adhere to JSON spec
        json.dump(report, f, indent=2, allow_nan=False)

    md_lines = ["# Influence QC Report\n"]
    md_lines.append(f"**Mode**: {report['header']['mode']}")
    md_lines.append(f"**Input**: {report['header']['input_path']}")
    md_lines.append(f"**Rule**: {report['header']['rule']}")
    if report['header']['threshold'] is not None:
         md_lines.append(f"**Threshold**: {report['header']['threshold']}")
    md_lines.append(f"**h_C_max**: {report['header']['h_C_max']}")
    if report['frac_cpgs_at_floor'] is not None:
        md_lines.append(f"**frac_cpgs_at_floor**: {report['frac_cpgs_at_floor']:.4f}")
    md_lines.append("")

    md_lines.append("## Distributions")
    if h_max_dist:
        md_lines.append("### mt_h_max")
        md_lines.append("| min | q01 | q05 | q25 | q50 | q75 | q95 | q99 | q999 | max |")
        md_lines.append("|---|---|---|---|---|---|---|---|---|---|")
        d = h_max_dist
        md_lines.append(f"| {d['min']:.4f} | {d['q01']:.4f} | {d['q05']:.4f} | {d['q25']:.4f} | {d['q50']:.4f} | {d['q75']:.4f} | {d['q95']:.4f} | {d['q99']:.4f} | {d['q999']:.4f} | {d['max']:.4f} |")
        md_lines.append("")

    if h_excess_dist:
        md_lines.append("### h_excess")
        md_lines.append("| min | q01 | q05 | q25 | q50 | q75 | q95 | q99 | q999 | max |")
        md_lines.append("|---|---|---|---|---|---|---|---|---|---|")
        d = h_excess_dist
        md_lines.append(f"| {d['min']:.4f} | {d['q01']:.4f} | {d['q05']:.4f} | {d['q25']:.4f} | {d['q50']:.4f} | {d['q75']:.4f} | {d['q95']:.4f} | {d['q99']:.4f} | {d['q999']:.4f} | {d['max']:.4f} |")
        md_lines.append("")

    md_lines.append("## Per-Region")
    cols = ["Region", "n_rows", "n_cpgs", "median_mt_h_max"]
    if has_fdr:
        cols.extend(["n_sig_rows", "n_sig_cpgs"])
    md_lines.append("| " + " | ".join(cols) + " |")
    md_lines.append("|---" * len(cols) + "|")
    for reg in region_list:
        rs = per_region_stats[reg]
        row = f"| {reg} | {rs['n_rows']} | {rs['n_cpgs']} | {rs.get('median_mt_h_max') or 'None'} |"
        if has_fdr:
            row += f" {rs['n_sig_rows']} | {rs['n_sig_cpgs']} |"
        md_lines.append(row)
    md_lines.append("")

    if chosen_stats:
        md_lines.append("## Chosen Rule Stats")
        md_lines.append(f"**n_cpgs_flagged**: {chosen_stats['n_cpgs_flagged']}")
        md_lines.append("")
        ccols = ["Region", "n_rows_flagged", "frac_rows_flagged", "n_cpgs_flagged"]
        if has_fdr:
            ccols.append("frac_sig_rows_flagged")
        md_lines.append("| " + " | ".join(ccols) + " |")
        md_lines.append("|---" * len(ccols) + "|")
        for reg in region_list:
            cs = chosen_stats["per_region"][reg]
            row = f"| {reg} | {cs['n_rows_flagged']} | {cs['frac_rows_flagged']:.4f} | {cs['n_cpgs_flagged']} |"
            if has_fdr:
                row += f" {cs.get('frac_sig_rows_flagged', 0.0):.4f} |"
            md_lines.append(row)
        md_lines.append("")

    md_lines.append("## Sweep Abs")
    if sweep_abs:
        md_lines.append("| tau | frac_cpgs_flagged | " + " | ".join(region_list) + " |")
        md_lines.append("|---" * (2 + len(region_list)) + "|")
        for tau_str, val in sweep_abs.items():
            row = f"| {tau_str} | {val['frac_cpgs_flagged']:.4f} |"
            if "frac_sig_rows_flagged" in val:
                for reg in region_list:
                    row += f" {val['frac_sig_rows_flagged'].get(reg, 0.0):.4f} |"
            else:
                for _ in region_list:
                    row += " - |"
            md_lines.append(row)
    md_lines.append("")

    md_lines.append("## Sweep Floor")
    if sweep_floor:
        md_lines.append("| delta | frac_cpgs_flagged | " + " | ".join(region_list) + " |")
        md_lines.append("|---" * (2 + len(region_list)) + "|")
        for delta_str, val in sweep_floor.items():
            row = f"| {delta_str} | {val['frac_cpgs_flagged']:.4f} |"
            if "frac_sig_rows_flagged" in val:
                for reg in region_list:
                    row += f" {val['frac_sig_rows_flagged'].get(reg, 0.0):.4f} |"
            else:
                for _ in region_list:
                    row += " - |"
            md_lines.append(row)
    md_lines.append("")

    md_lines.append("## Top 25 CpGs by mt_h_max")
    if top25:
        tcols = ["mt_id", "mt_h_max", "h_excess", "n_rows"]
        if has_fdr:
            tcols.append("n_sig_rows")
        if args.output:
            tcols.append("flagged")
        md_lines.append("| " + " | ".join(tcols) + " |")
        md_lines.append("|---" * len(tcols) + "|")
        for r in top25:
            row = f"| {r['mt_id']} | {r['mt_h_max']:.4f} | {r['h_excess'] if r['h_excess'] is not None else 'None'} | {r['n_rows']} |"
            if has_fdr:
                row += f" {r['n_sig_rows']} |"
            if args.output:
                row += f" {r['flagged']} |"
            md_lines.append(row)
    md_lines.append("")

    with open(os.path.join(args.report_dir, "influence_qc.md"), "w") as f:
        f.write("\n".join(md_lines) + "\n")

    if args.output:
        flag_field = pa.field(args.flag_column, pa.bool_(), nullable=True)
        new_schema = schema.append(flag_field)

        new_meta = {
            **(schema.metadata or {}),
            b'tecpg_influence_rule': str(args.rule).encode(),
            b'tecpg_influence_threshold': str(args.threshold).encode(),
            b'tecpg_influence_h_c_max': str(h_C_max).encode() if not np.isnan(h_C_max) else b'nan',
            b'tecpg_influence_flag_column': str(args.flag_column).encode(),
            b'tecpg_influence_n_cpgs': str(n_cpgs).encode(),
            b'tecpg_influence_n_cpgs_flagged': str(n_cpgs_flagged).encode(),
            b'tecpg_influence_source': os.path.basename(args.input).encode(),
        }
        new_schema = new_schema.with_metadata(new_meta)

        tmp_out = args.output + ".tmp"
        try:
            with pq.ParquetWriter(tmp_out, new_schema, compression="snappy") as writer:
                for batch in pq_file.iter_batches(batch_size=args.chunk_size):
                    df = batch.to_pandas()
                    if df.index.names != [None]:
                        df = df.reset_index()

                    flag_series = df["mt_id"].isin(flagged_mt_ids)

                    flag_series = flag_series.astype(object)
                    flag_series[df["mt_h_max"].isna()] = None

                    flag_array = pa.array(flag_series, type=pa.bool_(), from_pandas=True)

                    new_batch = batch.append_column(flag_field, flag_array)
                    writer.write_batch(new_batch)

            os.replace(tmp_out, args.output)
        except Exception:
            if os.path.isfile(tmp_out):
                try:
                    os.remove(tmp_out)
                except Exception:
                    pass
            raise

if __name__ == "__main__":
    main()
