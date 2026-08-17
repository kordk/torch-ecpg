#!/usr/bin/env python3
"""Summarize tools/runEnrichment.py outputs as a single self-contained HTML report.

Inputs (all optional-presence, all read-only):
  <enrichment-dir>/enrichment_results/<REGION>_<fdr|ig>_<LIBRARY>_enrichment.csv
  <enrichment-dir>/encode_enrichment_results.csv

For every enrichment CSV ("analysis" = region x method x library) the report
contains a bar figure of -log10(Adjusted P-value) for the top-N terms and a
table of the top-N terms. An overview table and overview figure summarize the
number of significant terms per analysis. Figures are embedded as base64 PNG so
the HTML is a single portable file. When no results are present the report is
still written and carries an explicit "no results" banner; this tool never
fabricates content and never exits non-zero on an empty (but well-formed) input.
"""
import argparse
import base64
import datetime as _dt
import glob
import html
import io
import logging
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# <REGION>_<fdr|ig>_<LIBRARY>_enrichment.csv ; LIBRARY may itself contain underscores.
ENRICH_FILE_RE = re.compile(r'^(?P<region>.+?)_(?P<method>fdr|ig)_(?P<library>.+)_enrichment\.csv$')

ENCODE_FILENAME = "encode_enrichment_results.csv"
NO_RESULTS_BANNER = "No significant enrichment results were found under the supplied directory."
ADJ_P_COL = "Adjusted P-value"


def parse_enrichment_filename(basename):
    """Return (region, method, library) or None if the name is not an enrichment CSV."""
    m = ENRICH_FILE_RE.match(basename)
    if not m:
        return None
    return m.group('region'), m.group('method'), m.group('library')


def discover_analyses(enrichment_dir):
    """Return a list of dicts: {region, method, library, path} sorted by (region, method, library)."""
    results_dir = os.path.join(enrichment_dir, "enrichment_results")
    found = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_enrichment.csv"))):
        parsed = parse_enrichment_filename(os.path.basename(path))
        if parsed is None:
            logger.warning(f"Skipping unrecognised file name in enrichment_results: {os.path.basename(path)}")
            continue
        region, method, library = parsed
        found.append({'region': region, 'method': method, 'library': library, 'path': path})
    found.sort(key=lambda d: (d['region'], d['method'], d['library']))
    return found


def load_top_terms(path, top_n):
    """Load an enrichment CSV, sort ascending by adjusted p, return the top_n rows."""
    df = pd.read_csv(path)
    if ADJ_P_COL not in df.columns:
        raise ValueError(f"{os.path.basename(path)} lacks required column '{ADJ_P_COL}'")
    df = df.sort_values(ADJ_P_COL, ascending=True, kind='mergesort').reset_index(drop=True)
    return df, df.head(top_n).copy()


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def make_terms_figure(top_df, title):
    """Horizontal bar of -log10(adjusted p) for the top terms; returns base64 PNG."""
    n = len(top_df)
    fig_h = max(2.5, 0.32 * n + 1.2)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    if n == 0:
        ax.text(0.5, 0.5, "no terms", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return _fig_to_base64(fig)
    adj = top_df[ADJ_P_COL].astype(float).clip(lower=1e-300)
    y = -np.log10(adj.values)
    labels = [str(t) if len(str(t)) <= 70 else str(t)[:67] + '...' for t in top_df['Term']]
    ax.barh(range(n)[::-1], y, color='#4C72B0')
    ax.set_yticks(range(n)[::-1])
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('-log10(Adjusted P-value)')
    ax.set_title(title, fontsize=10)
    ax.axvline(-np.log10(0.05), color='grey', linestyle='--', linewidth=0.8)
    return _fig_to_base64(fig)


def make_overview_figure(overview_df):
    """Bar of number of significant terms per analysis; returns base64 PNG."""
    n = len(overview_df)
    fig_h = max(2.5, 0.3 * n + 1.2)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    if n == 0:
        ax.text(0.5, 0.5, "no analyses", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return _fig_to_base64(fig)
    labels = [f"{r} | {m} | {l}" for r, m, l in zip(overview_df['Region'], overview_df['Method'], overview_df['Library'])]
    ax.barh(range(n)[::-1], overview_df['Significant terms'].values, color='#55A868')
    ax.set_yticks(range(n)[::-1])
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Number of significant terms (Adjusted P-value < 0.05)')
    ax.set_title('Overview: significant terms per analysis', fontsize=10)
    return _fig_to_base64(fig)


def _truncate_genes(s, max_genes=10):
    if not isinstance(s, str):
        return s
    parts = [p for p in s.split(';') if p]
    if len(parts) <= max_genes:
        return s
    return ';'.join(parts[:max_genes]) + f' ... (+{len(parts) - max_genes})'


def table_html(top_df):
    cols = [c for c in ['Term', 'Overlap', 'P-value', ADJ_P_COL, 'Genes'] if c in top_df.columns]
    df = top_df[cols].copy()
    if 'Genes' in df.columns:
        df['Genes'] = df['Genes'].map(_truncate_genes)
    for c in ('P-value', ADJ_P_COL):
        if c in df.columns:
            df[c] = df[c].map(lambda v: f"{float(v):.3g}")
    return df.to_html(index=False, escape=True, classes='terms', border=0)


def load_encode_summary(enrichment_dir, top_n):
    path = os.path.join(enrichment_dir, ENCODE_FILENAME)
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    sort_col = 'Adj P-value' if 'Adj P-value' in df.columns else ('P-value' if 'P-value' in df.columns else None)
    if sort_col is None:
        return path, df.head(top_n)
    df = df.sort_values(sort_col, ascending=True, kind='mergesort').reset_index(drop=True)
    return path, df.head(top_n)


def render_report(enrichment_dir, analyses, top_n, encode_path, encode_top):
    esc = html.escape
    now = _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    parts = []
    parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    parts.append("<title>tecpg enrichment summary</title>")
    parts.append("<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;max-width:1200px}"
                 "table.terms,table.overview{border-collapse:collapse;font-size:12px}"
                 "table.terms td,table.terms th,table.overview td,table.overview th{border:1px solid #ccc;padding:3px 6px;text-align:left;vertical-align:top}"
                 "table.terms th,table.overview th{background:#f0f0f0}"
                 ".banner{background:#fff3cd;border:1px solid #ffeeba;padding:12px;margin:12px 0}"
                 ".analysis{margin-top:36px;border-top:2px solid #ddd;padding-top:12px}"
                 "img{max-width:100%}</style></head><body>")
    parts.append("<h1>tecpg enrichment summary</h1>")
    parts.append(f"<p><b>Source directory:</b> {esc(os.path.abspath(enrichment_dir))}<br>"
                 f"<b>Generated:</b> {esc(now)}<br>"
                 f"<b>Top-N per analysis:</b> {top_n}<br>"
                 f"<b>Analyses found:</b> {len(analyses)}</p>")

    if not analyses and encode_path is None:
        parts.append(f"<div class='banner'>{esc(NO_RESULTS_BANNER)}</div>")

    # Overview
    rows = []
    for a in analyses:
        full, _ = a['data']
        min_adj = float(full[ADJ_P_COL].min()) if len(full) else float('nan')
        rows.append({'Region': a['region'], 'Method': a['method'], 'Library': a['library'],
                     'Significant terms': int(len(full)), 'Min Adjusted P-value': min_adj})
    overview_df = pd.DataFrame(rows, columns=['Region', 'Method', 'Library', 'Significant terms', 'Min Adjusted P-value'])
    parts.append("<h2 id='overview'>Overview</h2>")
    if len(overview_df):
        ov = overview_df.copy()
        ov['Min Adjusted P-value'] = ov['Min Adjusted P-value'].map(lambda v: f"{v:.3g}")
        parts.append(ov.to_html(index=False, escape=True, classes='overview', border=0))
    parts.append(f"<p><img alt='overview' src='data:image/png;base64,{make_overview_figure(overview_df)}'></p>")

    # TOC
    if analyses:
        parts.append("<h2>Contents</h2><ul>")
        for i, a in enumerate(analyses):
            parts.append(f"<li><a href='#a{i}'>{esc(a['region'])} | {esc(a['method'])} | {esc(a['library'])}</a></li>")
        if encode_path is not None:
            parts.append("<li><a href='#encode'>ENCODE ChromHMM enrichment</a></li>")
        parts.append("</ul>")

    # Per-analysis sections
    for i, a in enumerate(analyses):
        full, top = a['data']
        title = f"{a['region']} | {a['method']} | {a['library']}"
        parts.append(f"<div class='analysis' id='a{i}'><h2>{esc(title)}</h2>")
        parts.append(f"<p>{len(full)} significant term(s); showing top {len(top)} by adjusted p-value. "
                     f"Source: <code>{esc(os.path.basename(a['path']))}</code></p>")
        parts.append(f"<p><img alt='{esc(title)}' src='data:image/png;base64,{make_terms_figure(top, title)}'></p>")
        parts.append(table_html(top))
        parts.append("</div>")

    if encode_path is not None:
        parts.append("<div class='analysis' id='encode'><h2>ENCODE ChromHMM enrichment</h2>")
        parts.append(f"<p>Top {len(encode_top)} rows by adjusted p-value. Source: <code>{esc(os.path.basename(encode_path))}</code></p>")
        parts.append(encode_top.to_html(index=False, escape=True, classes='terms', border=0))
        parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Render a self-contained HTML summary of runEnrichment.py outputs.")
    parser.add_argument("--enrichment-dir", required=True, help="Directory passed as --out-dir to runEnrichment.py.")
    parser.add_argument("--out", required=True, help="Path of the HTML report to write.")
    parser.add_argument("--top-n", type=int, default=25, help="Number of top terms per analysis (default: 25).")
    args = parser.parse_args()

    if not os.path.isdir(args.enrichment_dir):
        logger.error(f"Enrichment directory not found: {args.enrichment_dir}")
        sys.exit(1)

    analyses = discover_analyses(args.enrichment_dir)
    for a in analyses:
        a['data'] = load_top_terms(a['path'], args.top_n)
        logger.info(f"Loaded {a['region']} | {a['method']} | {a['library']}: {len(a['data'][0])} significant term(s)")

    encode_path, encode_top = load_encode_summary(args.enrichment_dir, args.top_n)
    if not analyses and encode_path is None:
        logger.warning(NO_RESULTS_BANNER)

    report = render_report(args.enrichment_dir, analyses, args.top_n, encode_path, encode_top)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(args.out, 'w', encoding='utf-8') as fh:
        fh.write(report)
    logger.info(f"Wrote enrichment summary ({len(analyses)} analyses) to {args.out}")


if __name__ == "__main__":
    main()
