#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from permute_qc_report import QCModule, fig_to_base64, render_table, render_html  # noqa: E402

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def build_modules(summary: dict) -> list:
    """Return the ordered list of QCModule instances for this summary."""
    modules = []

    # 1. run-provenance
    prov_headers = ["Key", "Value"]
    prov_rows = [
        ["input", summary["input"]],
        ["tool_version", summary["tool_version"]],
        ["n_rows_read", summary["n_rows_read"]],
        ["n_scored", summary["n_scored"]],
        ["scored fraction", f'{(summary["n_scored"] / summary["n_rows_read"]) * 100:.2f}%'],
    ]
    for k, v in summary["params"].items():
        prov_rows.append([f"params.{k}", str(v)])

    modules.append(QCModule(
        anchor="run-provenance",
        title="Run provenance",
        status="INFO",
        purpose="Records the parameters and data sizes that produced these statistics.",
        interpretation="The scored rows are a subset selected upstream, so the statistics that follow describe that subset and may not extend to the rows not scored.",
        table_html=render_table(prov_headers, prov_rows),
    ))

    # 2. region-census
    regions = summary["regions"]
    sorted_regions = sorted(regions.items(), key=lambda x: x[1]["n_scored"], reverse=True)

    census_headers = ["region", "n_scored", "median_se_ratio", "mad_se_ratio", "t_abs_min", "t_abs_max", "trend_status"]
    census_rows = []

    for rname, rdata in sorted_regions:
        trend_status = rdata.get("trend_omitted_reason")
        if trend_status is None:
            trend_status = "reported"

        median = rdata.get("median_se_ratio")
        mad = rdata.get("mad_se_ratio")
        t_abs_min = rdata.get("t_abs_min")
        t_abs_max = rdata.get("t_abs_max")

        row = [
            rname,
            str(rdata["n_scored"]),
            f"{median:.4f}" if median is not None else "None",
            f"{mad:.4f}" if mad is not None else "None",
            f"{t_abs_min:.4f}" if t_abs_min is not None else "None",
            f"{t_abs_max:.4f}" if t_abs_max is not None else "None",
            trend_status
        ]
        census_rows.append(row)

    census_table_html = render_table(census_headers, census_rows)

    null_table_html = ""
    for rname, rdata in sorted_regions:
        if "n_null_region" in rdata:
            null_headers = ["n_null_region", "n_noncanonical_region", "noncanonical_labels", "noncanonical_labels_truncated"]
            labels = rdata.get("noncanonical_labels", [])
            labels_str = ", ".join(labels) if labels else "none"
            null_rows = [[
                str(rdata["n_null_region"]),
                str(rdata.get("n_noncanonical_region", 0)),
                labels_str,
                str(rdata.get("noncanonical_labels_truncated", False))
            ]]
            null_table_html = "<br><br>" + render_table(null_headers, null_rows)
            break

    modules.append(QCModule(
        anchor="region-census",
        title="Region census",
        status="INFO",
        purpose="Shows the median se-ratio and observed |t| range for each region.",
        interpretation="Comparing medians across regions assumes their scored rows were selected comparably; where t_abs_min differs markedly between regions that assumption likely does not hold.",
        table_html=census_table_html + null_table_html,
    ))

    # 3. trend-by-region
    trend_headers = ["region", "spearman_rho", "interval", "direction"]
    trend_rows = []

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_added = False

    for rname, rdata in sorted_regions:
        rho = rdata.get("spearman_rho")
        if rho is None:
            trend_rows.append([rname, rdata.get("trend_omitted_reason", "None"), "None", "None"])
        else:
            ci = rdata.get("spearman_ci", [0.0, 0.0])
            lo, hi = ci
            interval = f"[{lo:.4f}, {hi:.4f}]"
            direction = "excludes zero" if (lo > 0 or hi < 0) else "includes zero"
            trend_rows.append([rname, f"{rho:.4f}", interval, direction])

        bins = rdata.get("bins")
        if bins:
            plot_added = True
            t_lo = [b["t_abs_lo"] for b in bins]
            med = [b["median_se_ratio"] for b in bins]
            ax.plot(t_lo, med, marker='o', label=rname)

    if plot_added:
        ax.axhline(1.0, color='black', linestyle='--')
        ax.set_xscale('log')
        ax.set_xlabel('t_abs_lo')
        ax.set_ylabel('median_se_ratio')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        figure_b64 = fig_to_base64(fig)
    else:
        plt.close(fig)
        figure_b64 = ""

    modules.append(QCModule(
        anchor="trend-by-region",
        title="Trend within region",
        status="INFO",
        purpose="Shows whether the se-ratio changes with |t|.",
        interpretation="Under the assumption that the bootstrap and analytic standard errors estimate the same quantity, a flat slope may be consistent with an effect across all rows. The interval assumes the scored rows are exchangeable.",
        table_html=render_table(trend_headers, trend_rows),
        figure_b64=figure_b64
    ))

    # 4. interpretation-guidance
    notes = summary["notes"]
    guidance_headers = ["Key", "Note"]
    guidance_rows = [[k, v] for k, v in notes.items()]

    modules.append(QCModule(
        anchor="interpretation-guidance",
        title="Interpretation guidance",
        status="INFO",
        purpose="Provides context for the statistics above.",
        interpretation="These notes are emitted by the tool that computed the statistics and are reproduced unchanged, so that any later revision to them reaches this report without a second edit. It assumes they are still relevant.",
        table_html=render_table(guidance_headers, guidance_rows),
    ))

    return modules


def main():
    parser = argparse.ArgumentParser(description="Render the se-ratio trend summary as HTML.")
    parser.add_argument("--trend-json", required=True, help="path to the JSON from diagnose_se_ratio_trend.py")
    parser.add_argument("--dataset", required=True, help="dataset name for the report title")
    parser.add_argument("--out", required=True, help="path to the output HTML")
    args = parser.parse_args()

    try:
        with open(args.trend_json, "r") as f:
            summary = json.load(f)
    except Exception as e:
        sys.stderr.write(f"Failed to read or parse trend JSON: {e}\n")
        sys.exit(1)

    missing = []
    for key in ("regions", "notes", "params", "n_rows_read", "n_scored"):
        if key not in summary:
            missing.append(key)

    if missing:
        sys.stderr.write(f"Missing required keys in JSON: {', '.join(missing)}\n")
        sys.exit(1)

    modules = build_modules(summary)

    meta = {
        "input": args.trend_json,
        "tool_version": summary.get("tool_version", "unknown")
    }

    html_out = render_html(args.dataset, meta, modules)

    with open(args.out, "w") as f:
        f.write(html_out)


if __name__ == "__main__":
    main()
