#!/usr/bin/env python3
import argparse
import base64
import dataclasses
import datetime
import html
import io
import json
import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval_permute import (  # noqa: E402
    CANONICAL_REGIONS,
    NEAR_GENE_REGIONS,
    MIN_REGION_BULK_N,
    TOLERANCE_MEDIAN_LOG10_RATIO_DIFF,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Provisional QC thresholds -------------------------------------------
# Set from the first full-GTP enrichment run. Re-derive from MESA and the
# oncology cohort before treating any of these as settled.
STRAND_ASYMMETRY_WARN = 0.05     # |DISTAL5-DISTAL3| / mean above this warns
DIRECTION_WARN = 0.01            # |median log10(p_perm/p_ana)| in the bulk
DIRECTION_FAIL = 0.05
DELTA_WARN_FRACTION = 0.10       # near-gene |delta| above tolerance*this warns
TOLERANCE_SWEEP = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 1e-3, 1e-4, 1e-5)
# -------------------------------------------------------------------------

STATUSES = ('PASS', 'WARN', 'FAIL', 'INFO')


@dataclasses.dataclass
class QCModule:
    """One standalone QC item, rendered as a self-contained section."""
    anchor: str              # url fragment, e.g. "region-composition"
    title: str               # display name in nav and heading
    status: str              # one of PASS / WARN / FAIL / INFO
    purpose: str             # why this check exists; 1-3 sentences, plain text
    interpretation: str      # what to look for and what the observed values mean
    table_html: str = ""     # pre-rendered <table>...</table>, or ""
    figure_b64: str = ""     # base64 PNG payload, or ""
    figure_alt: str = ""     # alt text for the figure


def fig_to_base64(fig, dpi=110) -> str:
    """Render a matplotlib figure to a base64 PNG payload and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def render_table(headers: list, rows: list, aligns: list = None) -> str:
    """Render an HTML table. All cell content is HTML-escaped."""
    if not aligns:
        aligns = ['left'] * len(headers)

    html_out = ['<table>', '  <thead>', '    <tr>']
    for h, align in zip(headers, aligns):
        html_out.append(f'      <th style="text-align: {align};">{html.escape(str(h))}</th>')
    html_out.append('    </tr>')
    html_out.append('  </thead>')
    html_out.append('  <tbody>')

    for row in rows:
        html_out.append('    <tr>')
        for cell, align in zip(row, aligns):
            html_out.append(f'      <td style="text-align: {align};">{html.escape(str(cell))}</td>')
        html_out.append('    </tr>')

    html_out.append('  </tbody>')
    html_out.append('</table>')
    return '\n'.join(html_out)


def render_html(dataset: str, meta: dict, modules: list) -> str:
    """Assemble the full self-contained document."""
    escaped_dataset = html.escape(str(dataset))
    timestamp = html.escape(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    css = """
    body { font-family: sans-serif; margin: 0; padding: 0; display: flex; color: #333; }
    #qc-nav { width: 250px; background: #f5f5f5; padding: 20px; height: 100vh; position: fixed; \
              overflow-y: auto; border-right: 1px solid #ddd; box-sizing: border-box; }
    #qc-nav h1 { font-size: 1.2em; margin-top: 0; }
    #qc-nav .dataset { font-size: 0.9em; color: #666; margin-bottom: 20px; \
                       word-wrap: break-word; }
    #qc-nav ul { list-style: none; padding: 0; margin: 0; }
    #qc-nav li { margin-bottom: 10px; }
    #qc-nav a { text-decoration: none; color: #333; display: flex; \
                align-items: center; font-size: 0.9em; }
    #qc-nav a:hover { color: #000; text-decoration: underline; }
    main { margin-left: 250px; padding: 40px; flex: 1; \
           max-width: 1000px; box-sizing: border-box; }
    section { margin-bottom: 60px; }
    h2 { border-bottom: 1px solid #ddd; padding-bottom: 10px; display: flex; align-items: center; }

    .badge { display: inline-block; padding: 3px 8px; border-radius: 4px; color: white; \
             font-size: 0.8em; font-weight: bold; margin-right: 10px; margin-left: 10px; }
    .pass { background-color: #2e7d32; }
    .warn { background-color: #ef6c00; }
    .fail { background-color: #c62828; }
    .info { background-color: #546e7a; }

    .purpose, .interpretation { padding: 15px; margin: 20px 0; border-left: 5px solid #ccc; \
                                background-color: #f9f9f9; line-height: 1.5; }
    .purpose { border-left-color: #0277bd; background-color: #e1f5fe; }
    .interpretation { border-left-color: #558b2f; background-color: #f1f8e9; }

    table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.95em; }
    th, td { border: 1px solid #ddd; padding: 8px 12px; }
    th { background-color: #f5f5f5; }
    img { max-width: 100%; height: auto; border: 1px solid #eee; margin: 20px 0; }
    footer { margin-top: 60px; padding-top: 20px; border-top: 1px solid #ddd; \
             font-size: 0.8em; color: #777; text-align: center; }
    """

    nav_items = []
    for m in modules:
        status_lower = m.status.lower()
        title_esc = html.escape(m.title)
        nav_items.append(
            f'    <li><a href="#{m.anchor}"><span class="badge {status_lower}">{m.status}</span>{title_esc}</a></li>'
        )

    body_sections = []
    for m in modules:
        status_lower = m.status.lower()
        title_esc = html.escape(m.title)

        section_html = [f'  <section id="{m.anchor}">']
        section_html.append(f'    <h2>{title_esc} <span class="badge {status_lower}">{m.status}</span></h2>')
        section_html.append(f'    <div class="purpose"><strong>Purpose.</strong> {m.purpose}</div>')

        if m.figure_b64:
            section_html.append(
                f'    <img src="data:image/png;base64,{m.figure_b64}" alt="{html.escape(m.figure_alt)}">')

        if m.table_html:
            # table_html is already escaped in render_table
            section_html.append(f'    {m.table_html}')

        section_html.append(
            f'    <div class="interpretation"><strong>Interpretation.</strong> {m.interpretation}</div>')
        section_html.append('  </section>')

        body_sections.append('\n'.join(section_html))

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>qr_permute QC Report \u2014 {escaped_dataset}</title>
<style>
{css}
</style>
</head>
<body>
<nav id="qc-nav">
  <h1>qr_permute QC</h1>
  <p class="dataset">{escaped_dataset}</p>
  <ul>
{chr(10).join(nav_items)}
  </ul>
</nav>
<main>
{chr(10).join(body_sections)}
  <footer>Generated {timestamp} by tools/permute_qc_report.py</footer>
</main>
</body>
</html>"""

    return html_doc


def build_run_provenance_module(report: dict, df=None) -> QCModule:
    purpose = (
        "Records the inputs the rest of this report is computed from, so that any "
        "figure or table here can be traced back to a specific run. Every other "
        "module is conditioned on these counts."
    )
    interpretation = (
        "Pairs scored should equal pairs input minus the two drop counts. Pairs "
        "dropped for a null region are pairs whose locus lost its annotation during "
        "normalisation; they are excluded consistently by the evaluation and the "
        "summary, and a non-zero count is expected. Degrees of freedom must match "
        "the covariate design the master parquet was mapped under - a mismatch here "
        "invalidates every analytic p-value in the report."
    )

    meta = report.get('metadata', {})

    headers = ['Metric', 'Value']

    # bulk band as [bulk_lo, bulk_hi]
    # tail band as p_ana < tail_p_ana
    bulk_lo = report.get('metadata', {}).get('bulk_lo', 'N/A')
    bulk_hi = report.get('metadata', {}).get('bulk_hi', 'N/A')
    bulk_band = f"[{bulk_lo}, {bulk_hi}]" if bulk_lo != 'N/A' and bulk_hi != 'N/A' else "N/A"

    tail_p = report.get('metadata', {}).get('tail_p_ana', 'N/A')
    tail_band = f"p_ana < {tail_p}" if tail_p != 'N/A' else "N/A"

    rows = [
        ['Pairs input', meta.get('n_pairs_input', 'N/A')],
        ['Pairs scored', meta.get('n_pairs_scored', 'N/A')],
        ['Pairs dropped (unmappable chrom)', meta.get('n_pairs_dropped_unmappable_chrom', 'N/A')],
        ['Pairs dropped (null region)', meta.get('n_pairs_dropped_null_region', 'N/A')],
        ['Degrees of freedom', meta.get('df', 'N/A')],
        ['Bulk band', bulk_band],
        ['Tail band', tail_band],
    ]

    table_html = render_table(headers, rows)

    return QCModule(
        anchor="run-provenance",
        title="Run Provenance",
        status="INFO",
        purpose=purpose,
        interpretation=interpretation,
        table_html=table_html
    )


def build_region_composition_module(report: dict, df=None) -> QCModule:
    purpose = (
        "Confirms the cis-window enrichment produced the near-gene coverage the "
        "per-region calibration needs, and that the two distal strata are balanced. "
        "Near-gene pairs are a small fraction of the full test grid, so a uniform "
        "sample cannot power the cis check on its own."
    )
    interpretation = (
        "The near-gene total must clear the coverage floor with headroom; below it, "
        "the per-region verdict is not computed and the run reports "
        "insufficient_near_gene_coverage. A large trans fraction is expected and is "
        "not a defect - the trans test space is far larger than the cis one. The "
        "5-prime and 3-prime distal strata should be close to equal: a symmetric "
        "window applied to both strands produces a symmetric split, so a large "
        "asymmetry points at a strand-handling defect in the window predicate rather "
        "than at biology. Region labels come from the canonical region assignment "
        "tool and are not re-derived here."
    )

    meta = report.get('metadata', {})
    n_by_region = meta.get('n_by_region', {})

    # Fill in zeros for missing canonical regions
    counts = {r: n_by_region.get(r, 0) for r in CANONICAL_REGIONS}
    denom = sum(counts.values())
    if denom == 0:
        denom = 1  # prevent div by zero

    near_gene_sum = sum(counts[r] for r in NEAR_GENE_REGIONS)
    distal_sum = counts.get('DISTAL5', 0) + counts.get('DISTAL3', 0)
    trans_sum = counts.get('TRANS', 0)

    # Determine status
    distal5 = counts.get('DISTAL5', 0)
    distal3 = counts.get('DISTAL3', 0)
    mean_distal = (distal5 + distal3) / 2.0

    status = "PASS"
    if near_gene_sum < MIN_REGION_BULK_N:
        status = "FAIL"
    elif mean_distal > 0 and abs(distal5 - distal3) / mean_distal > STRAND_ASYMMETRY_WARN:
        status = "WARN"

    # Table rows
    headers = ['Region', 'n', '% of scored']
    aligns = ['left', 'right', 'right']
    rows = []
    for r in CANONICAL_REGIONS:
        val = counts[r]
        pct = f"{(val / denom) * 100:.2f}%"
        rows.append([r, f"{val:,}", pct])

    # Rollups
    rows.append(["Near-gene (sum)", f"{near_gene_sum:,}", f"{(near_gene_sum / denom) * 100:.2f}%"])
    rows.append(["Distal (sum)", f"{distal_sum:,}", f"{(distal_sum / denom) * 100:.2f}%"])
    rows.append(["Trans (sum)", f"{trans_sum:,}", f"{(trans_sum / denom) * 100:.2f}%"])

    table_html = render_table(headers, rows, aligns)

    # Figure
    fig_b64 = ""
    # We always plot, even if empty, as per instructions "always render all seven rows, in canonical order"
    # Actually, if all counts are 0, log axis will fail. But denom=1 avoids 0 div, and we can handle log zero gracefully
    fig, ax = plt.subplots(figsize=(8, 4))

    # Reverse order for horizontal bar chart to match list order top-to-bottom
    y_pos = range(len(CANONICAL_REGIONS))
    x_vals = [max(1, counts[r]) for r in CANONICAL_REGIONS]  # replace 0 with 1 for log scale

    colors = ['#1f77b4' if r in NEAR_GENE_REGIONS else '#ff7f0e' for r in CANONICAL_REGIONS]

    bars = ax.barh(y_pos, x_vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(CANONICAL_REGIONS)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xscale('log')
    ax.set_xlabel('Count')
    ax.set_title('Region Composition')

    ax.axvline(MIN_REGION_BULK_N, color='red', linestyle='--', label='coverage floor')
    ax.legend(loc='lower right')

    # Create custom legend for colors
    import matplotlib.patches as mpatches
    near_patch = mpatches.Patch(color='#1f77b4', label='Near-gene')
    dist_patch = mpatches.Patch(color='#ff7f0e', label='Distal / Trans')
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([near_patch, dist_patch])
    ax.legend(handles=handles, loc='lower right')

    # Use tighter layout
    fig.tight_layout()
    fig_b64 = fig_to_base64(fig)

    return QCModule(
        anchor="region-composition",
        title="Region Composition",
        status=status,
        purpose=purpose,
        interpretation=interpretation,
        table_html=table_html,
        figure_b64=fig_b64,
        figure_alt="Bar chart of region counts on a log scale"
    )


def build_bulk_calibration_module(report: dict, df=None) -> QCModule:
    purpose = (
        "Compares the permutation p-value against the analytic p-value across the "
        "mostly-null bulk band, where the two should agree if the parametric null "
        "holds. This is the arm that licenses using the analytic p downstream."
    )
    interpretation = (
        "Points should track the diagonal through the bulk. A systematic offset in "
        "one direction is reported separately in the calibration direction module "
        "and is meaningful even when small. Departures at the extreme right of this "
        "plot are expected and are addressed in the tail module: the permutation "
        "p-value cannot resolve below the reciprocal of the number of null draws, so "
        "the most significant pairs are compared against a floored value rather than "
        "against a genuine null probability. Read this plot for the bulk and defer "
        "to the tail module for the extreme."
    )

    calibration = report.get('arms', {}).get('calibration', {})
    per_region = report.get('arms', {}).get('stratify_decision', {}).get('per_region', {})

    headers = ['Region', 'n_bulk', 'Median log10 ratio', 'Median ratio', '% p_perm < p_ana']
    aligns = ['left', 'right', 'right', 'right', 'right']
    rows = []

    for r in CANONICAL_REGIONS:
        r_data = per_region.get(r, {})
        r_cal = calibration.get(r, {})
        r_status = r_data.get('status', 'not_reported')

        if r_status == 'insufficient_data' or r_status == 'not_reported' or not r_data:
            rows.append([r, "\u2014", "\u2014", "\u2014", "\u2014"])
            continue

        n_bulk = r_data.get('n_bulk')
        median_log10_ratio = r_data.get('median_log10_ratio')
        n_below = r_cal.get('n_perm_below_analytic')

        if n_bulk is None or median_log10_ratio is None or n_below is None:
            rows.append([r, "\u2014", "\u2014", "\u2014", "\u2014"])
            continue

        median_ratio = 10 ** median_log10_ratio
        pct_below = (n_below / n_bulk) * 100 if n_bulk > 0 else 0

        rows.append([
            r,
            f"{n_bulk:,}",
            f"{median_log10_ratio:.5f}",
            f"{median_ratio:.5f}",
            f"{pct_below:.2f}%"
        ])

    table_html = render_table(headers, rows, aligns)

    qq_data = calibration.get('qq_data')
    fig_b64 = ""
    if qq_data and qq_data.get('neg_log10_p_ana') and qq_data.get('neg_log10_p_perm'):
        fig, ax = plt.subplots(figsize=(6, 6))

        ana = qq_data['neg_log10_p_ana']
        perm = qq_data['neg_log10_p_perm']

        ax.scatter(ana, perm, alpha=0.4, rasterized=True, s=10)

        max_val = max(max(ana), max(perm)) if ana and perm else 1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')

        ax.set_xlabel(r'$-\log_{10}(p_{ana})$')
        ax.set_ylabel(r'$-\log_{10}(p_{perm})$')
        ax.set_title('Bulk Calibration')
        ax.legend()

        fig.tight_layout()
        fig_b64 = fig_to_base64(fig)
    else:
        interpretation += " (No qq_data was found in the report, so the figure is omitted.)"

    return QCModule(
        anchor="bulk-calibration",
        title="Bulk Calibration",
        status="INFO",
        purpose=purpose,
        interpretation=interpretation,
        table_html=table_html,
        figure_b64=fig_b64,
        figure_alt="Scatter plot of permutation vs analytic p-values" if fig_b64 else ""
    )


def build_calibration_direction_module(report: dict, df=None) -> QCModule:
    purpose = (
        "Reports the direction and magnitude of disagreement between the permutation "
        "and analytic p-values across the null bulk. The bulk calibration module shows "
        "whether the two agree; this module shows which way they disagree when they do, "
        "and how consistent that disagreement is across pairs."
    )
    interpretation = (
        "A median ratio above one means the permutation p-value is larger than the "
        "analytic p-value, i.e. the analytic p-value is anti-conservative and would "
        "call marginally more pairs significant than the permutation null supports. A "
        "ratio below one means the reverse. The direction matters even when the "
        "magnitude is small, because it is systematic rather than random: a consistent "
        "offset in one direction across millions of pairs is a property of the null "
        "model, not sampling noise. Read the fraction of pairs with p_perm below p_ana "
        "alongside the median - a median near zero combined with a fraction near fifty "
        "percent indicates genuinely symmetric disagreement, whereas a median near zero "
        "with a lopsided fraction indicates a small consistent shift. The Spearman "
        "correlation reports rank agreement only and is insensitive to any systematic "
        "scale offset, so a value of one is compatible with a substantial median ratio "
        "and is not on its own evidence of agreement. The 90th percentile column shows "
        "whether a tight median conceals a heavy shoulder."
    )

    calibration = report.get('arms', {}).get('calibration', {})
    per_region = report.get('arms', {}).get('stratify_decision', {}).get('per_region', {})

    headers = [
        'Stratum', 'n_bulk', 'Median log10 ratio', 'Median ratio',
        'Median |log10 ratio|', '90th pct |log10 ratio|', 'p_perm < p_ana %', 'Spearman ρ'
    ]
    aligns = ['left'] + ['right'] * 7
    rows = []

    def get_row(stratum, is_all=False):
        c_data = calibration.get(stratum, {})
        r_data = per_region.get(stratum, {}) if not is_all else {}

        n_bulk = c_data.get('n_bulk') if is_all else r_data.get('n_bulk')
        if n_bulk is None and not is_all:
            # Fallback if per_region is missing it but it might be somewhere else;
            # actually, if missing, skip. For 'all', n_bulk is sum.
            pass

        # For 'all', calculate n_bulk if not present explicitly
        if is_all and n_bulk is None:
            n_bulk = sum([per_region.get(r, {}).get('n_bulk', 0) for r in CANONICAL_REGIONS])
            if n_bulk == 0:
                n_bulk = None

        if n_bulk is None:
            return [stratum, "\u2014", "\u2014", "\u2014", "\u2014", "\u2014", "\u2014", "\u2014"]

        # Signed median
        signed_med = r_data.get('median_log10_ratio') if not is_all else None

        # Absolute median
        abs_med = c_data.get('bulk_median_abs_log_ratio')

        # 90th pct
        pct_90 = c_data.get('bulk_90th_abs_log_ratio')

        # p_perm < p_ana fraction
        n_below = c_data.get('n_perm_below_analytic')
        pct_below = (n_below / n_bulk) * 100 if n_below is not None and n_bulk > 0 else None

        # Spearman
        rho = c_data.get('bulk_spearman_corr')

        row = [stratum, f"{n_bulk:,}"]

        # Signed median and ratio
        if signed_med is not None:
            row.extend([f"{signed_med:.5f}", f"{10**signed_med:.5f}"])
        else:
            row.extend(["\u2014", "\u2014"])

        # Abs median
        if abs_med is not None:
            row.append(f"{abs_med:.5f}")
        else:
            row.append("\u2014")

        # 90th pct
        if pct_90 is not None:
            row.append(f"{pct_90:.5f}")
        else:
            row.append("\u2014")

        # pct below
        if pct_below is not None:
            row.append(f"{pct_below:.2f}%")
        else:
            row.append("\u2014")

        # Spearman
        if rho is not None:
            row.append(f"{rho:.4f}")
        else:
            row.append("\u2014")

        return row

    rows.append(get_row('all', is_all=True))
    for r in CANONICAL_REGIONS:
        rows.append(get_row(r))

    table_html = render_table(headers, rows, aligns)

    # Figure
    fig_b64 = ""
    y_pos = range(len(CANONICAL_REGIONS))
    x_vals = []
    labels_present = []
    for r in CANONICAL_REGIONS:
        med = per_region.get(r, {}).get('median_log10_ratio')
        x_vals.append(med if med is not None else 0.0)
        labels_present.append(med is not None)

    if any(labels_present):
        fig, ax = plt.subplots(figsize=(8, 4))
        # Colors: matching composition
        colors = ['#1f77b4' if r in NEAR_GENE_REGIONS else '#ff7f0e' for r in CANONICAL_REGIONS]

        ax.barh(y_pos, x_vals, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(CANONICAL_REGIONS)
        ax.invert_yaxis()

        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(DIRECTION_WARN, color='red', linestyle='--', alpha=0.7)
        ax.axvline(-DIRECTION_WARN, color='red', linestyle='--', alpha=0.7)

        ax.set_xlabel(r'median $\log_{10}(p_{perm} / p_{ana})$')
        ax.set_title('Calibration Direction')

        # Legend
        import matplotlib.patches as mpatches
        near_patch = mpatches.Patch(color='#1f77b4', label='Near-gene')
        dist_patch = mpatches.Patch(color='#ff7f0e', label='Distal / Trans')
        ax.legend(handles=[near_patch, dist_patch], loc='lower right')

        fig.tight_layout()
        fig_b64 = fig_to_base64(fig)

    # Status gated on 'all' stratum magnitude
    status = "PASS"
    all_cal = calibration.get('all', {})
    all_abs_med = all_cal.get('bulk_median_abs_log_ratio')

    if all_abs_med is None:
        status = "INFO"
    elif all_abs_med >= DIRECTION_FAIL:
        status = "FAIL"
    elif all_abs_med >= DIRECTION_WARN:
        status = "WARN"

    return QCModule(
        anchor="calibration-direction",
        title="Calibration Direction",
        status=status,
        purpose=purpose,
        interpretation=interpretation,
        table_html=table_html,
        figure_b64=fig_b64,
        figure_alt="Horizontal bar chart of median log10 ratios for canonical regions" if fig_b64 else ""
    )


def build_verdict_robustness_module(report: dict, df=None) -> QCModule:
    purpose = (
        "Asks how far the applied tolerance would have to move before the "
        "stratification verdict changed. A verdict that holds across several orders of "
        "magnitude of tolerance is a materially stronger result than one that clears "
        "the threshold narrowly, and the table makes the difference explicit rather "
        "than leaving it implicit in a single pass or fail."
    )
    interpretation = (
        "The applied tolerance is provisional and was set before this cohort was run. "
        "Read the row at which the verdict first changes: if that row is many orders of "
        "magnitude below the applied tolerance, the verdict does not depend on the "
        "specific value chosen and the choice can be deferred. If the verdict changes "
        "within about one order of magnitude of the applied value, the threshold is "
        "doing real work and should be set deliberately from evidence before the "
        "verdict is relied upon. This sweep re-applies the same rule the evaluation "
        "uses and does not introduce a second decision procedure; it reports "
        "sensitivity, and it does not change the verdict or the applied tolerance. "
        "Note that a very wide margin can also mean the test is insensitive rather "
        "than that the result is strong - a tolerance no realistic departure could "
        "ever exceed would produce the same wide margin, so read this together with "
        "the calibration direction module rather than on its own."
    )

    strat = report.get('arms', {}).get('stratify_decision', {})
    per_region = strat.get('per_region', {})
    actual_verdict = strat.get('recommendation', 'unknown')

    near_deltas = {}
    float_count = 0
    for r in NEAR_GENE_REGIONS:
        r_data = per_region.get(r, {})
        d = r_data.get('delta_vs_trans')
        if isinstance(d, (int, float)):
            near_deltas[r] = abs(d)
            float_count += 1

    headers = ['Tolerance', 'Divergent near-gene regions', 'Resulting verdict', 'Applied']
    aligns = ['left', 'right', 'left', 'center']
    rows = []

    applied_t = TOLERANCE_MEDIAN_LOG10_RATIO_DIFF

    warn_triggered = False

    for t in TOLERANCE_SWEEP:
        div_count = sum(1 for val in near_deltas.values() if val >= t)
        verdict = "single_global_null_adequate" if div_count == 0 else "stratification_warranted"

        is_applied = abs(t - applied_t) < 1e-12
        marker = "<- applied" if is_applied else ""

        rows.append([
            f"{t:g}",
            str(div_count),
            verdict,
            marker
        ])

        # Check if the verdict differs from the applied tolerance verdict within one order of magnitude
        # We need to know what the verdict at the applied tolerance is.
        # But we can calculate what it should be.
        applied_div_count = sum(1 for val in near_deltas.values() if val >= applied_t)
        applied_verdict = "single_global_null_adequate" if applied_div_count == 0 else "stratification_warranted"

        if verdict != applied_verdict and t >= applied_t / 10.0:
            warn_triggered = True

    table_html = render_table(headers, rows, aligns)

    # Figure
    fig_b64 = ""
    if float_count > 0:
        fig, ax = plt.subplots(figsize=(7, 4))

        x_vals = []
        y_vals = []
        # Sort tolerances for plot (small to large)
        for t in sorted(TOLERANCE_SWEEP):
            div_count = sum(1 for val in near_deltas.values() if val >= t)
            x_vals.append(t)
            y_vals.append(div_count)

        ax.step(x_vals, y_vals, where='post', color='#2e7d32', linewidth=2)
        ax.set_xscale('log')

        # Mark applied tolerance
        ax.axvline(applied_t, color='black', linestyle='--', label=f'Applied ({applied_t:g})')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

        # Annotate largest observed |delta|
        if near_deltas:
            max_delta = max(near_deltas.values())
            ax.axvline(max_delta, color='red', linestyle='-.', alpha=0.7,
                       label=f'Max |Δ| ({max_delta:.2e})')

        ax.set_xlabel('Candidate Tolerance')
        ax.set_ylabel('Divergent near-gene regions')
        ax.set_title('Verdict Robustness Sweep')

        # reverse x axis so it goes large to small? No, log scale standard is small to large left to right.
        # But wait, tolerances usually sweep from big to small in the table. Standard log x is fine.
        ax.invert_xaxis()  # larger tolerances on the left to match the table visually
        ax.legend()

        fig.tight_layout()
        fig_b64 = fig_to_base64(fig)

    status = "PASS"
    if float_count < 2 or actual_verdict == 'insufficient_near_gene_coverage':
        status = "INFO"
    elif warn_triggered:
        status = "WARN"

    return QCModule(
        anchor="verdict-robustness",
        title="Verdict Robustness",
        status=status,
        purpose=purpose,
        interpretation=interpretation,
        table_html=table_html,
        figure_b64=fig_b64,
        figure_alt="Step plot of candidate tolerance vs divergent regions" if fig_b64 else ""
    )


def build_permutation_resolution_module(report: dict, df=None) -> QCModule:
    purpose = (
        "Establishes the smallest p-value the permutation null in this run is capable "
        "of resolving, and how much of the significant tail is sitting at that limit. "
        "Every ratio reported in the tail behaviour module is bounded below by this "
        "floor, so this module must be read before that one."
    )
    interpretation = (
        "An empirical permutation p-value cannot fall below roughly the reciprocal of "
        "the total number of null draws, so any pair whose true p-value is far past "
        "that limit receives a floored permutation p-value rather than a resolved one. "
        "Where that happens, the ratio of permutation to analytic p-value measures the "
        "distance between the analytic p-value and the floor, and reports nothing "
        "about whether the analytic p-value is correct. A large fraction of the tail "
        "sitting at the floor therefore means the tail is unmeasured at this number of "
        "permutations - it does not mean the analytic tail has been validated, and it "
        "does not mean it has been contradicted. Raising the permutation count lowers "
        "this floor roughly in proportion. The implied effective null draws figure is "
        "derived by inverting the observed floor and depends on the empirical p-value "
        "convention in use; treat it as an order-of-magnitude cross-check on the run "
        "configuration rather than an exact count."
    )

    if df is None or 'perm_mt_p' not in df.columns or df['perm_mt_p'].isna().all():
        return QCModule(
            anchor="permutation-resolution",
            title="Permutation Resolution",
            status="INFO",
            purpose=purpose,
            interpretation="Not evaluated: --perm-output not supplied.",
            table_html="",
            figure_b64="",
            figure_alt=""
        )

    # Compute stats
    import numpy as np

    perm_p = df['perm_mt_p']
    positive_p = perm_p[perm_p > 0]

    if len(positive_p) == 0:
        return QCModule(
            anchor="permutation-resolution",
            title="Permutation Resolution",
            status="INFO",
            purpose=purpose,
            interpretation="Not evaluated: no strictly positive permutation p-values found.",
            table_html="",
            figure_b64="",
            figure_alt=""
        )

    floor = positive_p.min()
    n_at_floor = (perm_p <= floor).sum()
    n_zero = (perm_p == 0).sum()

    n_perm_val = df['n_perm'].unique()[0] if 'n_perm' in df.columns and len(df['n_perm'].unique()) == 1 else None

    # Tail stats
    metadata = report.get('metadata', {})
    tail_p_ana = metadata.get('tail_p_ana')
    df_value = metadata.get('df')

    n_tail = "\u2014"
    n_tail_at_floor = "\u2014"
    frac_tail_at_floor_str = "\u2014"
    frac_tail_at_floor = None

    if tail_p_ana is not None and df_value is not None and 'mt_t' in df.columns:
        import scipy.stats
        mt_t = df['mt_t']

        # p_ana = 2 * scipy.stats.t.sf(abs(mt_t), df_value)
        p_ana = 2 * scipy.stats.t.sf(np.abs(mt_t), df_value)

        tail_mask = p_ana < tail_p_ana
        n_tail_val = tail_mask.sum()

        n_tail = f"{n_tail_val:,}"

        if n_tail_val > 0:
            tail_at_floor = ((perm_p <= floor) & tail_mask).sum()
            n_tail_at_floor = f"{tail_at_floor:,}"
            frac_tail_at_floor = tail_at_floor / n_tail_val
            frac_tail_at_floor_str = f"{frac_tail_at_floor * 100:.2f}%"
        else:
            n_tail_at_floor = "0"
            frac_tail_at_floor = 0.0
            frac_tail_at_floor_str = "0.00%"

    n_scored = len(df)
    frac_scored = (n_at_floor / n_scored) * 100 if n_scored > 0 else 0

    headers = ['Metric', 'Value']
    rows = [
        ['Smallest non-zero permutation p', f"{floor:.3g}"],
        ['Pairs at floor', f"{n_at_floor:,} ({frac_scored:.2f}% of parquet rows)"],
        ['Zeros', f"{n_zero:,}"],
        ['Permutations performed', f"{n_perm_val:,}" if n_perm_val is not None else "\u2014"],
        ['Pairs in tail band', n_tail],
        ['Tail pairs at floor', f"{n_tail_at_floor} ({frac_tail_at_floor_str})"],
        ['Implied effective null draws ≈ 1 / floor', f"{1/floor:.3g}" if floor > 0 else "\u2014"]
    ]

    table_html = render_table(headers, rows)

    # Figure
    fig_b64 = ""
    fig, ax = plt.subplots(figsize=(7, 4))

    neg_log10_p = -np.log10(positive_p)
    ax.hist(neg_log10_p, bins=50, color='#607d8b', edgecolor='black', alpha=0.7)

    floor_val = -np.log10(floor)
    ax.axvline(floor_val, color='red', linestyle='--', linewidth=2, label='resolution floor')

    ax.set_yscale('log')
    ax.set_xlabel(r'$-\log_{10}(p_{perm})$')
    ax.set_ylabel('Count')
    ax.set_title('Permutation P-value Distribution (strictly positive)')
    ax.legend()

    fig.tight_layout()
    fig_b64 = fig_to_base64(fig)

    status = "PASS"
    if frac_tail_at_floor is not None and frac_tail_at_floor >= 0.5:
        status = "WARN"

    return QCModule(
        anchor="permutation-resolution",
        title="Permutation Resolution",
        status=status,
        purpose=purpose,
        interpretation=interpretation,
        table_html=table_html,
        figure_b64=fig_b64,
        figure_alt="Histogram of strictly positive permutation p-values on a log scale"
    )


def build_tail_behaviour_module(report: dict, df=None) -> QCModule:
    purpose = (
        "Reports how the permutation and analytic p-values diverge in the significant "
        "tail, which is the regime where a permutation null is expected to add the "
        "most value over a parametric one. It is presented for inspection and is "
        "deliberately not given a pass or fail status."
    )
    interpretation = (
        "Read this table only after the permutation resolution module. Ratios here are "
        "bounded below by the resolution floor, so where much of a region's tail sits "
        "at that floor the ratio is an artefact of the permutation count rather than a "
        "statement about the null. Ratios are also inflated by genuine signal: a region "
        "carrying more true associations places more pairs deep in the tail and "
        "therefore more pairs at the floor, which raises its ratio without any "
        "miscalibration being present. For that reason an ordering across regions that "
        "tracks where signal is expected is the expected result and is not evidence of "
        "regional divergence; the stratification verdict is decided on the null bulk, "
        "not here. This module cannot at present distinguish resolution limits from "
        "model misspecification. Separating them requires a run with a substantially "
        "larger permutation count, and until then no conclusion about the analytic "
        "tail should be drawn from this table in either direction."
    )

    calibration = report.get('arms', {}).get('calibration', {})

    headers = ['Region', 'Tail median log10 ratio', 'Median ratio',
               '10th percentile', '90th percentile', 'Ratio at 90th']
    aligns = ['left'] + ['right'] * 5
    rows = []

    any_tail_stats = False

    for r in CANONICAL_REGIONS:
        r_cal = calibration.get(r, {})

        tail_med = r_cal.get('tail_median_log_ratio')
        tail_10 = r_cal.get('tail_10th_log_ratio')
        tail_90 = r_cal.get('tail_90th_log_ratio')

        row = [r]
        if tail_med is not None:
            any_tail_stats = True
            row.append(f"{tail_med:.3f}")
            row.append(f"{10**tail_med:.2f}")
        else:
            row.extend(["\u2014", "\u2014"])

        if tail_10 is not None:
            row.append(f"{tail_10:.3f}")
        else:
            row.append("\u2014")

        if tail_90 is not None:
            row.append(f"{tail_90:.3f}")
            row.append(f"{10**tail_90:.3g}")
        else:
            row.extend(["\u2014", "\u2014"])

        rows.append(row)

    if not any_tail_stats:
        return QCModule(
            anchor="tail-behaviour",
            title="Tail Behaviour",
            status="INFO",
            purpose=purpose,
            interpretation="Not evaluated: no tail statistics present in the report.",
            table_html="",
            figure_b64="",
            figure_alt=""
        )

    table_html = render_table(headers, rows, aligns)

    # Figure
    fig_b64 = ""
    y_pos = range(len(CANONICAL_REGIONS))
    x_vals = []

    for r in CANONICAL_REGIONS:
        tail_med = calibration.get(r, {}).get('tail_median_log_ratio')
        x_vals.append(tail_med if tail_med is not None else 0.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#1f77b4' if r in NEAR_GENE_REGIONS else '#ff7f0e' for r in CANONICAL_REGIONS]

    ax.barh(y_pos, x_vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(CANONICAL_REGIONS)
    ax.invert_yaxis()

    ax.axvline(0, color='black', linewidth=1)

    ax.set_xlabel('Tail median log10 ratio')
    ax.set_title('Tail Behaviour')

    import matplotlib.patches as mpatches
    near_patch = mpatches.Patch(color='#1f77b4', label='Near-gene')
    dist_patch = mpatches.Patch(color='#ff7f0e', label='Distal / Trans')
    ax.legend(handles=[near_patch, dist_patch], loc='lower right')

    fig.tight_layout()
    fig_b64 = fig_to_base64(fig)

    return QCModule(
        anchor="tail-behaviour",
        title="Tail Behaviour",
        status="INFO",
        purpose=purpose,
        interpretation=interpretation,
        table_html=table_html,
        figure_b64=fig_b64,
        figure_alt="Horizontal bar chart of tail median log10 ratios per region"
    )


def build_stratification_module(report: dict, df=None) -> QCModule:
    purpose = (
        "Asks whether each region's null bulk behaves like the trans reference. If "
        "the near-gene regions calibrate the same way trans does, a single global "
        "null is adequate and no per-region approximation is needed; if they "
        "diverge, the cis strata need their own treatment."
    )
    interpretation = (
        "The gate is the absolute difference in median log10 ratio against the trans "
        "reference, compared to the tolerance. The Mann-Whitney and KS p-values are "
        "reported for completeness but do not gate anything: at bulk sizes in the "
        "millions they resolve differences far too small to matter, so a very small "
        "p next to a negligible delta is expected and is not a finding. The genomic "
        "inflation factor is advisory only and is discussed in its own module. Read "
        "the margin line below: a verdict that holds across several orders of "
        "magnitude of tolerance is a stronger result than one that clears the "
        "threshold narrowly. This verdict is dataset-specific and is never inherited "
        "by another cohort."
    )

    strat = report.get('arms', {}).get('stratify_decision', {})
    per_region = strat.get('per_region', {})

    headers = ['Region', 'n_bulk', 'Status', 'Median log10 ratio', 'Δ vs TRANS', 'Mann-Whitney p', 'KS p', 'λ']
    aligns = ['left', 'right', 'center', 'right', 'right', 'right', 'right', 'right']
    rows = []

    for r in CANONICAL_REGIONS:
        r_data = per_region.get(r, {})
        r_status = r_data.get('status', 'not_reported')

        if r_status == 'insufficient_data' or r_status == 'not_reported' or not r_data:
            rows.append([r, "\u2014", r_status, "\u2014", "\u2014", "\u2014", "\u2014", "\u2014"])
            continue

        n_bulk = r_data.get('n_bulk')
        median_log10 = r_data.get('median_log10_ratio')
        delta = r_data.get('delta_vs_trans')
        mw_p = r_data.get('mw_p')
        ks_p = r_data.get('ks_p')
        lam = r_data.get('lambda')

        n_bulk_str = f"{n_bulk:,}" if n_bulk is not None else "\u2014"
        med_str = f"{median_log10:.5f}" if median_log10 is not None else "\u2014"
        lam_str = f"{lam:.5f}" if lam is not None else "\u2014"

        if r == 'TRANS':
            rows.append([r, n_bulk_str, r_status, med_str, "\u2014", "\u2014", "\u2014", lam_str])
        else:
            delta_str = f"{delta:.3e}" if delta is not None else "\u2014"
            mw_str = f"{mw_p:.3e}" if mw_p is not None else "\u2014"
            ks_str = f"{ks_p:.3e}" if ks_p is not None else "\u2014"
            rows.append([r, n_bulk_str, r_status, med_str, delta_str, mw_str, ks_str, lam_str])

    table_html = render_table(headers, rows, aligns)

    # Determine status & interpretation logic
    divergent_regions = strat.get('divergent_regions', [])
    recommendation = html.escape(strat.get('recommendation', 'unknown'))

    # Append verdict line
    div_str = html.escape(", ".join(divergent_regions)) if divergent_regions else "none"
    interpretation += f"<br><br>Verdict: `{recommendation}`. Divergent regions: {div_str}."

    # Find max near-gene delta
    near_deltas = {}
    for r in NEAR_GENE_REGIONS:
        r_data = per_region.get(r, {})
        d = r_data.get('delta_vs_trans')
        if isinstance(d, (int, float)):
            near_deltas[r] = abs(d)

    max_delta_val = 0
    max_delta_region = None
    if near_deltas:
        max_delta_region = max(near_deltas, key=near_deltas.get)
        max_delta_val = near_deltas[max_delta_region]

    status = "PASS"
    if divergent_regions:
        status = "FAIL"
    elif max_delta_val >= TOLERANCE_MEDIAN_LOG10_RATIO_DIFF * DELTA_WARN_FRACTION:
        status = "WARN"
    elif recommendation == 'insufficient_near_gene_coverage':
        status = "INFO"

    if near_deltas and max_delta_val > 0:
        margin = TOLERANCE_MEDIAN_LOG10_RATIO_DIFF / max_delta_val
        margin_line = (
            f"Largest near-gene |Δ vs TRANS| = {max_delta_val:.3e} ({max_delta_region}); "
            f"applied tolerance = {TOLERANCE_MEDIAN_LOG10_RATIO_DIFF}; "
            f"the verdict is unchanged for any tolerance above {max_delta_val:.3e}, a margin of {margin:,.0f}x."
        )
        if len(near_deltas) < len(NEAR_GENE_REGIONS):
            margin_line += (
                f" Based on {len(near_deltas)} of {len(NEAR_GENE_REGIONS)} "
                "near-gene regions; the remainder lacked a reported delta."
            )
        interpretation += f"<br>{margin_line}"

    return QCModule(
        anchor="stratification",
        title="Stratification",
        status=status,
        purpose=purpose,
        interpretation=interpretation,
        table_html=table_html
    )


def main():
    parser = argparse.ArgumentParser(description="Generate self-contained HTML QC report for qr_permute")
    parser.add_argument('--report', required=True, help="Path to eval_permute_report.json")
    parser.add_argument(
        '--perm-output', help="Path to permutation_results.parquet (optional)"
    )
    parser.add_argument('--df', type=int, help="Degrees of freedom (optional)")
    parser.add_argument('--dataset', required=True, help="Dataset name for the report title")
    parser.add_argument('--out', required=True, help="Path to output HTML file")

    args = parser.parse_args()

    try:
        with open(args.report, 'r') as f:
            report_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read report JSON {args.report}: {e}")
        sys.exit(1)

    df = None
    if args.perm_output and args.df is not None:
        try:
            # We don't actually need to read the parquet for Chunk 1, but we satisfy the contract
            # If pyarrow isn't installed, we might fail, but it was installed.
            import pyarrow.parquet as pq
            df = pq.read_table(args.perm_output).to_pandas()
            logger.info(f"Loaded parquet {args.perm_output}")
        except Exception as e:
            logger.warning(f"Failed to load parquet {args.perm_output}: {e}")
            df = None

    modules = []

    builders = [
        build_run_provenance_module,
        build_region_composition_module,
        build_bulk_calibration_module,
        build_calibration_direction_module,
        build_stratification_module,
        build_verdict_robustness_module,
        build_permutation_resolution_module,
        build_tail_behaviour_module,
    ]

    for build_func in builders:
        try:
            mod = build_func(report_data, df)
            modules.append(mod)

        except Exception as e:
            logger.error(f"Module builder {build_func.__name__} failed: {e}")
            # Do not raise as per instructions, but wait, the instructions say:
            # "A builder that cannot evaluate its check returns status='INFO' \
            # with an explanatory interpretation; it must never raise."
            # Our builders don't catch their own exceptions if missing keys fail hard, \
            # but we used .get() heavily.
            # To be absolutely safe and meet "never raise":
            modules.append(QCModule(
                anchor=build_func.__name__.replace('build_', '').replace('_module', '').replace(
                    '_', '-'),
                title="Error",
                status="INFO",
                purpose="Error occurred",
                interpretation=f"Builder raised an exception: {e}"
            ))

    html_content = render_html(args.dataset, report_data.get('metadata', {}), modules)

    try:
        with open(args.out, 'w') as f:
            f.write(html_content)
        logger.info(f"Wrote HTML QC report to {args.out}")
    except Exception as e:
        logger.error(f"Failed to write output HTML to {args.out}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception in permute_qc_report: {e}")
        sys.exit(1)
