#!/usr/bin/env python3
"""influence_qc_report.py — FastQC-style HTML report for the influence diagnostic.

Consumes JSON artifacts produced elsewhere in the pipeline and renders them as
self-contained QC modules, each with a plain-language Purpose, a PASS/WARN/FAIL/
INFO badge, and an Interpretation written against the observed numbers:

  --influence-qc      influence_qc.json          (tools/flagInfluence_parquet.py)
  --bridge            calibration_bridge.json    (tools/calibration_bridge.py)
  --kennedy-influence influence_stratified.json  (tools/benchmark_kennedy.py)
  --flagged-parquet   summarized.influence.parquet (metadata: rule in force)

Every input is optional; a module whose input is absent renders as INFO with the
reason, so the report is useful at any stage. Styling and layout are shared with
tools/permute_qc_report.py (QCModule / render_html / render_table).

Usage:
  python3 tools/influence_qc_report.py --dataset gtp \
      --influence-qc      output_gtp/influence_qc/influence_qc.json \
      --bridge            output_gtp/calibration_bridge/calibration_bridge.json \
      --kennedy-influence output_gtp/kennedy/influence_stratified.json \
      --flagged-parquet   output_gtp/summarized.influence.parquet \
      --out               output_gtp/influence_qc_report.html
"""
import argparse
import base64
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from permute_qc_report import QCModule, render_html, render_table  # noqa: E402

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ---- verdict thresholds (documented, not tuned to any dataset) -------------
FLOOR_DOMINANCE_INFO = 0.5     # frac of CpGs at the covariate floor: context only
CONCENTRATION_WARN = 2.0       # significant-row enrichment of flagged CpGs
CONCENTRATION_FAIL = 5.0
RESIDUAL_WARN = 0.05           # P(break | unflagged) above this warns
RESIDUAL_FAIL = 0.15
DOSE_MONOTONIC_MIN = 3         # bins needed to judge a trend
COVERAGE_MIN = 0.05            # per-bin bootstrap coverage below this is unreliable
KENNEDY_TREND_WARN = -0.5      # Spearman recovery-vs-decile at or below this: signal

COL_ALL = '#3b6ea5'
COL_TR = '#c2452d'
COL_MUTE = '#888888'
REGION_ORDER = ['TRANS', 'DISTAL5', 'DISTAL3', 'CIS5', 'CIS3', 'PROMOTER',
                'GENEBODY']


# ------------------------------------------------------------------ helpers
def load_json(path, flag):
    if not path:
        return None, f'--{flag} not supplied'
    if not os.path.exists(path):
        return None, f'file not found: {path}'
    try:
        with open(path) as fh:
            return json.load(fh), None
    except Exception as exc:                                   # noqa: BLE001
        return None, f'could not parse {path}: {exc}'


def read_flag_metadata(path):
    if not path or not os.path.exists(path):
        return None
    try:
        import pyarrow.parquet as pq
        md = pq.ParquetFile(path).schema_arrow.metadata or {}
        out = {k.decode().replace('tecpg_influence_', ''): v.decode()
               for k, v in md.items() if k.startswith(b'tecpg_influence_')}
        return out or None
    except Exception:                                          # noqa: BLE001
        return None


def num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def fmt(v, nd=4, pct=False):
    f = num(v)
    if f is None:
        return '-'
    if pct:
        return f'{f * 100:.1f}%'
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return f'{int(v):,}'
    return f'{f:.{nd}f}'


def fmt_int(v):
    f = num(v)
    return '-' if f is None else f'{int(f):,}'


def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def ordered_regions(d):
    keys = list(d.keys())
    return [r for r in REGION_ORDER if r in keys] + \
           sorted(k for k in keys if k not in REGION_ORDER)


def not_evaluated(anchor, title, purpose, reason):
    return QCModule(anchor=anchor, title=title, status='INFO', purpose=purpose,
                    interpretation=f'Not evaluated: {reason}. This module is '
                                   'skipped; the rest of the report is '
                                   'unaffected.')


# ------------------------------------------------------------------ modules
def mod_provenance(dataset, args, qc, bridge, ki, meta):
    purpose = (
        'Records exactly which files this report was computed from, and which '
        'flagging rule (if any) was in force. Influence numbers are only '
        'comparable between runs that share a mapping, an annotation, and a '
        'covariate matrix, so every figure below should be read against this '
        'table rather than against a previous run from memory.')
    rows = [['Dataset', dataset]]
    h = (qc or {}).get('header') or {}
    if h:
        rows += [
            ['Catalog analysed', str(h.get('input_path', '-'))],
            ['Rows / CpGs', f"{fmt_int(h.get('n_rows'))} / {fmt_int(h.get('n_cpgs'))}"],
            ['Rows with no leverage value', fmt_int(h.get('n_rows_null_h'))],
            ['Subjects / covariates / model terms',
             f"{fmt_int(h.get('n_subjects'))} / {fmt_int(h.get('n_covariates'))} / "
             f"{fmt_int(h.get('p_design'))}"],
            ['Covariate leverage floor (h_C_max)', fmt(h.get('h_C_max'), 6)],
        ]
    rows.append(['influence_qc.json',
                 args.influence_qc if qc else 'not supplied'])
    rows.append(['calibration_bridge.json',
                 args.bridge if bridge else 'not supplied'])
    rows.append(['influence_stratified.json',
                 args.kennedy_influence if ki else 'not supplied'])
    if meta:
        rows.append(['Flag in force',
                     ', '.join(f'{k}={v}' for k, v in sorted(meta.items())
                               if k in ('rule', 'threshold', 'n_cpgs_flagged'))])
    else:
        rows.append(['Flag in force', 'no flagged catalog supplied'])
    interpretation = (
        'Confirm the catalog path is the run you intend to describe. If the '
        '"Flag in force" row is empty, no flag has been stamped yet: the '
        'landscape and guidance modules still apply, but nothing downstream is '
        'being filtered.')
    return QCModule(anchor='provenance', title='Run provenance', status='INFO',
                    purpose=purpose, interpretation=interpretation,
                    table_html=render_table(['Item', 'Value'], rows,
                                            ['left', 'left']))


def mod_landscape(qc, note):
    purpose = (
        'Leverage measures how much a single subject can pull the fit for one '
        'CpG. It depends only on the design matrix - the CpG\'s own methylation '
        'values plus the covariates - and not on any gene, so it is a property '
        'of the locus. This module shows two things: the floor that the '
        'covariates impose on every CpG in this cohort (h_C_max), and how far '
        'individual CpGs rise above it.')
    if not qc:
        return not_evaluated('landscape', 'Leverage landscape', purpose, note)
    h = qc.get('header') or {}
    hC = num(h.get('h_C_max'))
    pn = num(h.get('p_over_n'))
    frac_floor = num(qc.get('frac_cpgs_at_floor'))
    labels = ['min', 'q01', 'q05', 'q25', 'q50', 'q75', 'q95', 'q99', 'q999',
              'max']
    rows = [['Covariate leverage floor (h_C_max)', fmt(hC, 6)],
            ['Textbook mean leverage (p/n)', fmt(pn, 4)],
            ['Textbook cutoffs 2p/n, 3p/n',
             f'{fmt(2 * pn) if pn else "-"}, {fmt(3 * pn) if pn else "-"}'],
            ['CpGs sitting at the floor', fmt(frac_floor, pct=True)]]
    tables = [render_table(['Quantity', 'Value'], rows, ['left', 'right'])]
    for key, name in (('mt_h_max_dist', 'Leverage (mt_h_max) by quantile'),
                      ('h_excess_dist', 'Excess over the floor (h_excess)')):
        d = qc.get(key)
        if d:
            tables.append(f'<p><strong>{name}</strong></p>' + render_table(
                labels, [[fmt(d.get(k)) for k in labels]], ['right'] * 10))
    fig = ''
    if HAVE_MPL and qc.get('mt_h_max_dist'):
        d, e = qc['mt_h_max_dist'], qc.get('h_excess_dist') or {}
        xs = np.arange(len(labels))
        f, ax = plt.subplots(figsize=(7.2, 3.4), dpi=110)
        ax.plot(xs, [num(d.get(k)) for k in labels], 'o-', color=COL_ALL,
                ms=5, label='mt_h_max')
        if e:
            ax.plot(xs, [num(e.get(k)) for k in labels], 's--', color=COL_TR,
                    ms=4, label='h_excess')
        if hC is not None:
            ax.axhline(hC, color=COL_MUTE, ls=':', lw=1)
            ax.text(0, hC, f'  covariate floor = {hC:.4f}', fontsize=8,
                    va='bottom', color='#555')
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlabel('quantile of the per-CpG distribution')
        ax.set_ylabel('leverage')
        ax.legend(frameon=False, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        fig = fig_b64(f)

    status = 'INFO'
    bits = []
    if hC is not None and pn is not None:
        if 3 * pn < hC:
            bits.append(
                f'The covariate design alone gives one subject a leverage of '
                f'{hC:.3f}, which is above the textbook cutoffs '
                f'(2p/n = {2 * pn:.3f}, 3p/n = {3 * pn:.3f}). Those rules of '
                f'thumb would flag essentially every CpG here and must not be '
                f'used; thresholds have to be stated relative to the floor.')
        else:
            bits.append(
                f'The covariate floor ({hC:.3f}) sits below the textbook '
                f'cutoff 3p/n ({3 * pn:.3f}), so conventional leverage rules '
                f'are not obviously broken for this cohort - unusual for this '
                f'design, and worth checking the covariate matrix.')
    if frac_floor is not None:
        bits.append(
            f'{frac_floor * 100:.0f}% of CpGs sit at that floor, meaning the '
            f'most influential subject for them is the one the covariates '
            f'already single out, not anything about the methylation values.')
        status = 'PASS' if frac_floor >= FLOOR_DOMINANCE_INFO else 'INFO'
    e = qc.get('h_excess_dist') or {}
    if num(e.get('q95')) is not None and num(e.get('max')) is not None:
        bits.append(
            f'The excess distribution is heavily right-skewed: 95% of CpGs add '
            f'less than {num(e["q95"]):.3f} over the floor, while the most '
            f'extreme adds {num(e["max"]):.3f}. That tail is the population '
            f'this diagnostic exists to find.')
    return QCModule(anchor='landscape', title='Leverage landscape',
                    status=status, purpose=purpose,
                    interpretation=' '.join(bits) or 'No distribution summary '
                                                     'was present in the input.',
                    table_html='\n'.join(tables), figure_b64=fig,
                    figure_alt='leverage quantiles')


def mod_region(qc, note):
    purpose = (
        'Shows how the tested pairs and the significant hits break down by '
        'genomic region, together with the median leverage of each region. '
        'Leverage is a property of the CpG, so the medians should look alike '
        'across regions; large differences would point at an annotation or '
        'assignment problem rather than at biology.')
    if not qc:
        return not_evaluated('region', 'Region composition', purpose, note)
    per = qc.get('per_region') or {}
    if not per:
        return not_evaluated('region', 'Region composition', purpose,
                             'no per-region block in the QC input')
    regs = ordered_regions(per)
    has_sig = any('n_sig_rows' in per[r] for r in regs)
    hdr = ['Region', 'Rows tested', 'CpGs', 'Median leverage']
    if has_sig:
        hdr += ['Significant rows', 'Significant CpGs']
    rows = []
    for r in regs:
        s = per[r]
        row = [r, fmt_int(s.get('n_rows')), fmt_int(s.get('n_cpgs')),
               fmt(s.get('median_mt_h_max'))]
        if has_sig:
            row += [fmt_int(s.get('n_sig_rows')), fmt_int(s.get('n_sig_cpgs'))]
        rows.append(row)
    meds = [num(per[r].get('median_mt_h_max')) for r in regs]
    meds = [m for m in meds if m is not None]
    spread = (max(meds) - min(meds)) if meds else None
    total = sum(num(per[r].get('n_rows')) or 0 for r in regs)
    trans_share = (num(per.get('TRANS', {}).get('n_rows')) or 0) / total if total else 0
    bits = []
    if spread is not None:
        bits.append(
            f'Median leverage varies by only {spread:.4f} across regions, as '
            f'expected: the diagnostic does not know where a CpG sits relative '
            f'to a gene.' if spread < 0.01 else
            f'Median leverage differs by {spread:.4f} across regions, which is '
            f'more than a CpG-level property should vary. Check the region '
            f'assignment before reading anything into region-specific results.')
    if total:
        bits.append(
            f'TRANS accounts for {trans_share * 100:.1f}% of all tested pairs, '
            f'so genome-wide totals are dominated by it; near-gene regions are '
            f'small enough that their percentages move on few pairs.')
    bits.append('Region labels come from the upstream annotation stage. A rerun '
                'that changes annotation changes this table without any change '
                'to the diagnostic itself.')
    status = 'PASS' if (spread is not None and spread < 0.01) else 'WARN'
    return QCModule(anchor='region', title='Region composition', status=status,
                    purpose=purpose, interpretation=' '.join(bits),
                    table_html=render_table(
                        hdr, rows, ['left'] + ['right'] * (len(hdr) - 1)))


def mod_flag(qc, note, meta):
    purpose = (
        'Reports what the flagging rule actually removed. The key number is not '
        'how many CpGs were flagged but how concentrated they are among the '
        'significant results: a small set of loci carrying a large share of the '
        'hits is the signature this screen is designed to catch.')
    if not qc:
        return not_evaluated('flag', 'Applied flag', purpose, note)
    chosen = qc.get('chosen_rule_stats')
    h = qc.get('header') or {}
    if not chosen:
        return QCModule(
            anchor='flag', title='Applied flag', status='INFO', purpose=purpose,
            interpretation='No rule was applied: the input was produced in '
                           'report-only mode, so nothing is flagged and '
                           'nothing downstream is filtered. The threshold '
                           'guidance module below is the place to choose a '
                           'rule.')
    per = chosen.get('per_region') or {}
    regs = ordered_regions(per)
    has_sig = any('frac_sig_rows_flagged' in per[r] for r in regs)
    hdr = ['Region', 'Rows flagged', 'Share of rows', 'CpGs flagged']
    if has_sig:
        hdr.append('Share of significant rows')
    rows = []
    for r in regs:
        c = per[r]
        row = [r, fmt_int(c.get('n_rows_flagged')),
               fmt(c.get('frac_rows_flagged'), pct=True),
               fmt_int(c.get('n_cpgs_flagged'))]
        if has_sig:
            row.append(fmt(c.get('frac_sig_rows_flagged'), pct=True))
        rows.append(row)
    n_flag = num(chosen.get('n_cpgs_flagged'))
    n_cpg = num(h.get('n_cpgs'))
    frac_cpgs = (n_flag / n_cpg) if (n_flag and n_cpg) else None
    tr = per.get('TRANS') or {}
    tr_rows = num(tr.get('frac_rows_flagged'))
    tr_sig = num(tr.get('frac_sig_rows_flagged'))
    conc = (tr_sig / tr_rows) if (tr_rows and tr_sig) else None
    bits = [f'Rule in force: {h.get("rule", "-")} at threshold '
            f'{h.get("threshold", "-")}.']
    if frac_cpgs is not None:
        bits.append(f'{n_flag:,.0f} CpGs are flagged ({frac_cpgs * 100:.1f}% of '
                    f'those tested).')
    if conc is not None:
        bits.append(
            f'In TRANS, flagged CpGs contribute {tr_rows * 100:.1f}% of the '
            f'tests but {tr_sig * 100:.1f}% of the significant pairs - a '
            f'{conc:.1f}-fold concentration among hits. Loci that are '
            f'over-represented among significant results relative to how often '
            f'they are tested are behaving like artefacts, not like biology.')
    near = [r for r in ('CIS5', 'CIS3', 'PROMOTER') if r in per]
    if near:
        worst = max(num(per[r].get('frac_sig_rows_flagged')) or 0 for r in near)
        bits.append(
            f'Near-gene regions are comparatively clean (at most '
            f'{worst * 100:.1f}% of their significant rows flagged), so the '
            f'screen is removing distal associations, not cis signal.')
    status = 'PASS'
    if conc is not None and conc >= CONCENTRATION_FAIL:
        status = 'FAIL'
    elif conc is not None and conc >= CONCENTRATION_WARN:
        status = 'WARN'
    if status != 'PASS':
        bits.append('The badge reflects how much of the significant catalog '
                    'depends on high-leverage loci, not a defect in the run: a '
                    'strong concentration means the unfiltered catalog should '
                    'not be interpreted as-is.')
    return QCModule(anchor='flag', title='Applied flag', status=status,
                    purpose=purpose, interpretation=' '.join(bits),
                    table_html=render_table(
                        hdr, rows, ['left'] + ['right'] * (len(hdr) - 1)))


def mod_coverage(bridge, note):
    purpose = (
        'The fragility checks that follow are computed only on pairs that were '
        'bootstrapped. This module shows what fraction of significant pairs in '
        'each leverage band actually have bootstrap results, because every rate '
        'below is conditional on that coverage.')
    if not bridge:
        return not_evaluated('coverage', 'Bootstrap coverage', purpose, note)
    cov = bridge.get('coverage_by_leverage') or []
    if not cov:
        return not_evaluated('coverage', 'Bootstrap coverage', purpose,
                             'no coverage block in the bridge input')
    rows = [[c.get('hbin'), fmt_int(c.get('n_sig')), fmt_int(c.get('n_cov')),
             fmt(c.get('coverage'), pct=True)] for c in cov]
    fracs = [num(c.get('coverage')) for c in cov]
    fracs = [f for f in fracs if f is not None]
    lowest = min(fracs) if fracs else None
    hdr = bridge.get('header') or {}
    bits = [f'{fmt_int(hdr.get("covered_significant"))} pairs are both '
            f'significant and bootstrapped, out of '
            f'{fmt_int(hdr.get("significant"))} significant pairs.']
    if lowest is not None:
        bits.append(
            f'The thinnest leverage band has {lowest * 100:.1f}% coverage.'
            + (' That is enough to estimate a rate, but the bootstrap list was '
               'chosen by ranking rather than at random, so covered pairs are '
               'an extremity-biased sample of each band and these rates should '
               'not be extrapolated to uncovered pairs without assumption.'
               if lowest >= COVERAGE_MIN else
               ' That is too thin to estimate a rate reliably; treat the '
               'corresponding row in the next module as indicative only.'))
    status = 'PASS' if (lowest is not None and lowest >= COVERAGE_MIN) else 'WARN'
    return QCModule(anchor='coverage', title='Bootstrap coverage',
                    status=status, purpose=purpose,
                    interpretation=' '.join(bits),
                    table_html=render_table(
                        ['Leverage band', 'Significant pairs', 'Bootstrapped',
                         'Coverage'], rows, ['left', 'right', 'right', 'right']))


def mod_dose(bridge, note):
    purpose = (
        'Tests the premise of the whole diagnostic: if high leverage really '
        'means a result rests on one subject, then resampling the subjects '
        'should destabilise those results more often. "Sign instability" here '
        'means the bootstrap confidence interval for the effect spans zero, so '
        'the direction of the association is not resolved.')
    if not bridge:
        return not_evaluated('dose', 'Leverage vs fragility', purpose, note)
    rows_j = bridge.get('breakdown_by_leverage') or []
    if not rows_j:
        return not_evaluated('dose', 'Leverage vs fragility', purpose,
                             'no breakdown block in the bridge input')
    bins = [r.get('hbin') for r in rows_j]
    ci = [num(r.get('ci_cross')) for r in rows_j]
    n = [num(r.get('n')) for r in rows_j]
    tr = {r.get('hbin'): num(r.get('ci_cross'))
          for r in (bridge.get('breakdown_by_leverage_trans') or [])}
    table_rows = [[r.get('hbin'), fmt_int(r.get('n')),
                   fmt(r.get('ci_cross'), pct=True),
                   fmt(r.get('break10'), pct=True),
                   fmt(r.get('break25'), pct=True),
                   fmt(r.get('med_se_ratio'), 2)] for r in rows_j]
    fig = ''
    if HAVE_MPL:
        x = np.arange(len(bins))
        w = 0.38
        f, ax = plt.subplots(figsize=(7.2, 4.0), dpi=110)
        ax.bar(x - w / 2, [(c or 0) * 100 for c in ci], w, label='All regions',
               color=COL_ALL)
        if tr:
            ax.bar(x + w / 2, [(tr.get(b) or np.nan) * 100 for b in bins], w,
                   label='TRANS only', color=COL_TR)
        for xi, (c, ni) in enumerate(zip(ci, n)):
            top = max((c or 0) * 100, (tr.get(bins[xi]) or 0) * 100)
            ax.text(xi, top + 2.5, f'n={int(ni):,}' if ni else '', ha='center',
                    va='bottom', fontsize=8, color='#444')
        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.set_xlabel('per-CpG max sample leverage')
        ax.set_ylabel('% of pairs with sign instability')
        ax.set_ylim(0, 108)
        ax.legend(frameon=False, fontsize=9, loc='upper left')
        ax.spines[['top', 'right']].set_visible(False)
        fig = fig_b64(f)
    clean = [c for c in ci if c is not None]
    monotone = (len(clean) >= DOSE_MONOTONIC_MIN
                and all(b >= a - 1e-9 for a, b in zip(clean, clean[1:])))
    bits = []
    if len(clean) >= 2:
        bits.append(
            f'Sign instability rises from {clean[0] * 100:.1f}% in the lowest '
            f'leverage band to {clean[-1] * 100:.1f}% in the highest.')
    if monotone:
        bits.append(
            'The increase is monotonic across every band, which is what makes '
            'leverage usable as a screen: it is not merely correlated with '
            'fragility at the extreme, it tracks it throughout.')
        status = 'PASS'
    elif len(clean) >= DOSE_MONOTONIC_MIN:
        bits.append(
            'The increase is not monotonic across bands. Leverage still '
            'separates the extremes, but a non-monotone profile weakens the '
            'case for a single threshold and is worth investigating before '
            'the flag is relied on.')
        status = 'WARN'
    else:
        status = 'INFO'
    se_lo = num(rows_j[0].get('med_se_ratio'))
    se_hi = num(rows_j[-1].get('med_se_ratio'))
    if se_lo and se_hi:
        bits.append(
            f'The SE ratio column is an independent check on the same claim: it '
            f'compares the spread seen across resamples with the standard error '
            f'the model reports. It moves from {se_lo:.2f} (analytic error '
            f'about right) to {se_hi:.2f} (analytic error understating the true '
            f'variability several-fold), agreeing with the instability rates '
            f'without reusing them.')
    bits.append(
        'Two cautions on reading these rates. The bootstrap resamples all '
        'subjects jointly rather than deleting the influential one, so a '
        'leverage-1 pair can look stable when that subject lands in most '
        'resamples - these rates are a floor on true single-deletion '
        'fragility. And instability here is about the sign of the effect, not '
        'about whether the pair would survive multiple-testing correction.')
    return QCModule(anchor='dose', title='Leverage vs fragility', status=status,
                    purpose=purpose, interpretation=' '.join(bits),
                    table_html=render_table(
                        ['Leverage band', 'n', 'Sign instability',
                         'Unstable in >=10% of resamples', 'In >=25%',
                         'Median SE ratio'], table_rows,
                        ['left'] + ['right'] * 5),
                    figure_b64=fig, figure_alt='instability by leverage band')


def mod_kennedy(ki, note):
    purpose = (
        'An external check that does not use the bootstrap at all. Pairs are '
        'compared against an independently published catalog (Kennedy et al.) '
        'and grouped by leverage. If high-leverage results were real biology, '
        'agreement with an independent study should not depend on leverage.')
    if not ki:
        return not_evaluated('kennedy', 'External catalog agreement', purpose,
                             note)
    if ki.get('skipped'):
        return not_evaluated('kennedy', 'External catalog agreement', purpose,
                             f"analysis skipped ({ki.get('reason', 'no reason given')})")
    dec = ki.get('recovery_by_decile') or []
    trend = num(ki.get('recovery_trend_spearman'))
    tables = []
    if dec:
        tables.append(render_table(
            ['Decile', 'Leverage range', 'Catalog-significant', 'Concordant',
             'Recovery'],
            [[d.get('decile'),
              f"{fmt(d.get('h_max_lo'))} - {fmt(d.get('h_max_hi'))}",
              fmt_int(d.get('n_kennedy_sig')), fmt_int(d.get('n_concordant')),
              fmt(d.get('recovery'), pct=True)] for d in dec],
            ['right', 'left', 'right', 'right', 'right']))
    c = ki.get('concordance_low_high') or {}
    if c:
        tables.append('<p><strong>Agreement in the low- and high-leverage '
                      'halves</strong></p>' + render_table(
            ['Measure', 'Low leverage', 'High leverage', 'Difference'],
            [['Effect-size correlation', fmt(c.get('effect_spearman_low')),
              fmt(c.get('effect_spearman_high')),
              fmt(c.get('effect_delta_low_minus_high'))],
             ['t-statistic correlation', fmt(c.get('t_spearman_low')),
              fmt(c.get('t_spearman_high')),
              fmt(c.get('t_delta_low_minus_high'))],
             ['Pairs', fmt_int(c.get('n_low')), fmt_int(c.get('n_high')), '-']],
            ['left', 'right', 'right', 'right']))
    fig = ''
    if HAVE_MPL and dec:
        pts = [(d.get('decile'), num(d.get('recovery')), num(d.get('n_kennedy_sig')))
               for d in dec if num(d.get('recovery')) is not None]
        if pts:
            f, ax = plt.subplots(figsize=(7.2, 3.4), dpi=110)
            ax.plot([p[0] for p in pts], [p[1] * 100 for p in pts], 'o-',
                    color=COL_ALL, ms=6)
            for xi, yi, ni in pts:
                ax.annotate(f'n={int(ni)}' if ni else '', (xi, yi * 100),
                            xytext=(0, 8), textcoords='offset points',
                            ha='center', fontsize=7, color='#555')
            ax.set_xticks([p[0] for p in pts])
            ax.set_xlabel('leverage decile (0 = lowest)')
            ax.set_ylabel('recovery of catalog-significant pairs (%)')
            ax.spines[['top', 'right']].set_visible(False)
            fig = fig_b64(f)
    bits = []
    if dec:
        rec = [num(d.get('recovery')) for d in dec if num(d.get('recovery')) is not None]
        if len(rec) >= 2:
            bits.append(f'Recovery falls from {rec[0] * 100:.0f}% in the '
                        f'lowest-leverage decile to {rec[-1] * 100:.0f}% in the '
                        f'highest.')
    status = 'INFO'
    if trend is not None:
        if trend <= KENNEDY_TREND_WARN:
            bits.append(
                f'The trend across deciles is strongly negative '
                f'(Spearman {trend:.2f}): agreement with an independent study '
                f'degrades as leverage rises. Because this evidence is external '
                f'to both the bootstrap and the flag, it is the strongest '
                f'available argument that high-leverage associations are not '
                f'reproducible biology.')
            status = 'PASS'
        else:
            bits.append(
                f'The trend across deciles is {trend:.2f}, which does not show '
                f'the degradation seen elsewhere. Either this cohort behaves '
                f'differently or the matched set is too small to resolve it - '
                f'check the pair counts per decile before drawing a conclusion.')
            status = 'WARN'
    if c and num(c.get('effect_delta_low_minus_high')) is not None:
        d = num(c['effect_delta_low_minus_high'])
        bits.append(
            f'Splitting at the median leverage tells the same story: effect '
            f'sizes agree {d:+.2f} better in the low-leverage half.')
    bits.append('Note that this comparison is limited to pairs present in both '
                'catalogs, and that the two studies may differ in tissue, '
                'preprocessing, and covariates; it is corroboration, not a '
                'gold standard.')
    return QCModule(anchor='kennedy', title='External catalog agreement',
                    status=status, purpose=purpose,
                    interpretation=' '.join(bits),
                    table_html='\n'.join(tables), figure_b64=fig,
                    figure_alt='recovery by leverage decile')


def mod_guidance(qc, qc_note, bridge, bridge_note):
    purpose = (
        'Choosing a threshold is a research decision, not a tool default, so '
        'this module lays out the trade-off rather than recommending a value. '
        'Two quantities move in opposite directions: how much of the catalog a '
        'threshold removes, and how fragile the part you keep still is.')
    if not qc and not bridge:
        return not_evaluated('guidance', 'Threshold guidance', purpose,
                             f'{qc_note or ""} {bridge_note or ""}'.strip())
    tables = []
    if qc:
        for key, name, param in (
                ('sweep_floor', 'Floor rule: flag when leverage exceeds the '
                                'covariate floor by more than delta', 'delta'),
                ('sweep_abs', 'Absolute rule: flag when leverage exceeds tau',
                 'tau')):
            sw = qc.get(key)
            if not sw:
                continue
            keys = sorted(sw.keys(), key=float)
            regs = ordered_regions(next(
                (sw[k].get('frac_sig_rows_flagged', {}) for k in keys
                 if sw[k].get('frac_sig_rows_flagged')), {}))
            rows = []
            for k in keys:
                per = sw[k].get('frac_sig_rows_flagged') or {}
                rows.append([f'{float(k):g}',
                             fmt(sw[k].get('frac_cpgs_flagged'), pct=True)]
                            + [fmt(per.get(r), pct=True) for r in regs])
            tables.append(f'<p><strong>{name}</strong></p>' + render_table(
                [param, 'CpGs flagged'] + regs, rows,
                ['right'] * (2 + len(regs))))
    if bridge:
        for key, name in (('sweep_floor', 'Floor rule: what the flag catches'),
                          ('sweep_abs', 'Absolute rule: what the flag catches')):
            rows_j = bridge.get(key) or []
            if not rows_j:
                continue
            tables.append(f'<p><strong>{name}</strong></p>' + render_table(
                ['Threshold', 'Pairs flagged', 'Unstable among flagged',
                 'Unstable among kept', 'Share of unstable pairs caught'],
                [[f"{float(r['threshold']):g}", fmt_int(r.get('n_flagged')),
                  fmt(r.get('ci_cross|flag'), pct=True),
                  fmt(r.get('ci_cross|unflag'), pct=True),
                  fmt(r.get('ci_cross_recall'), pct=True)] for r in rows_j],
                ['right'] * 5))
    bits = [
        'Read the columns in this order. "Unstable among kept" is the residual '
        'fragility of the catalog you would retain, and is usually the number '
        'that should drive the decision. "Unstable among flagged" says how '
        'often the flag is right when it fires. "Share of unstable pairs '
        'caught" says how much fragility the screen would miss. Lower '
        'thresholds flag more loci and leave a cleaner retained set, at the '
        'cost of discarding more of the catalog.']
    status = 'INFO'
    sweep = (bridge or {}).get('sweep_floor') or (bridge or {}).get('sweep_abs')
    if sweep:
        best = min(sweep, key=lambda r: num(r.get('ci_cross|unflag')) or 1)
        resid = num(best.get('ci_cross|unflag'))
        if resid is not None:
            bits.append(
                f'In this run the lowest achievable residual instability is '
                f'{resid * 100:.1f}% at threshold {float(best["threshold"]):g}, '
                f'flagging {fmt_int(best.get("n_flagged"))} covered pairs.')
            status = ('PASS' if resid <= RESIDUAL_WARN else
                      'WARN' if resid <= RESIDUAL_FAIL else 'FAIL')
    bits.append(
        'Whatever value is chosen, state it relative to this cohort\'s floor '
        'and recompute the floor for every new dataset. A rule transfers '
        'between cohorts; a threshold value does not.')
    return QCModule(anchor='guidance', title='Threshold guidance',
                    status=status, purpose=purpose,
                    interpretation=' '.join(bits),
                    table_html='\n'.join(tables))


def mod_top(qc, note):
    purpose = (
        'Lists the individual loci with the highest leverage, so that specific '
        'CpGs can be looked up, cross-checked against the raw methylation '
        'values, or excluded by name. Entries combining near-unit leverage with '
        'many significant partners are the clearest candidates for a single '
        'subject driving a large set of associations.')
    if not qc:
        return not_evaluated('top-loci', 'Highest-leverage loci', purpose, note)
    rows_j = qc.get('top25') or []
    if not rows_j:
        return not_evaluated('top-loci', 'Highest-leverage loci', purpose,
                             'no top-25 block in the QC input')
    has_flag = any('flagged' in r for r in rows_j)
    hdr = ['CpG', 'Leverage', 'Excess over floor', 'Rows', 'Significant rows']
    if has_flag:
        hdr.append('Flagged')
    body = []
    for r in rows_j:
        row = [str(r.get('mt_id')), fmt(r.get('mt_h_max')),
               fmt(r.get('h_excess')), fmt_int(r.get('n_rows')),
               fmt_int(r.get('n_sig_rows'))]
        if has_flag:
            f = r.get('flagged')
            row.append('-' if f is None else ('yes' if f else 'no'))
        body.append(row)
    top = num(rows_j[0].get('mt_h_max'))
    many = [r for r in rows_j if (num(r.get('n_sig_rows')) or 0) >= 5]
    bits = []
    if top is not None:
        bits.append(
            f'The most extreme locus reaches a leverage of {top:.4f}. A value '
            f'approaching 1 means the fitted line for that CpG passes through '
            f'one subject almost exactly, so its effect estimate is close to '
            f'being determined by that individual alone.')
    if many:
        bits.append(
            f'{len(many)} of the listed loci have five or more significant '
            f'partners, so a single subject would be propagating into many '
            f'reported associations at once.')
    bits.append('These are the loci to inspect first if a downstream result '
                'depends on one of them.')
    return QCModule(anchor='top-loci', title='Highest-leverage loci',
                    status='INFO', purpose=purpose,
                    interpretation=' '.join(bits),
                    table_html=render_table(
                        hdr, body, ['left'] + ['right'] * (len(hdr) - 1)))


def mod_caveats():
    purpose = (
        'What this analysis does and does not establish. These limits apply to '
        'every number above and should travel with any result derived from '
        'them.')
    items = [
        ('The floor is specific to this cohort', 'h_C_max comes from this '
         'covariate matrix. Recompute it for every dataset; never carry a '
         'threshold value across cohorts, only the rule.'),
        ('Leverage is necessary, not sufficient', 'High leverage means one '
         'subject <em>can</em> determine the fit, not that it does. '
         'Deletion-based confirmation is the sharper test and is not yet part '
         'of the mapping.'),
        ('The bootstrap understates deletion fragility', 'Resampling perturbs '
         'all subjects jointly, so the instability rates here are a floor on '
         'the true single-deletion effect.'),
        ('Coverage is selection-biased', 'The bootstrapped pairs were chosen by '
         'ranking, not at random, so fragility rates describe the covered '
         'stratum and do not automatically generalise.'),
        ('Instability is not the same as non-replication', 'A sign-stable pair '
         'can still fail to replicate, and a sign-unstable one can remain '
         'nominally significant. These are different questions.'),
        ('Region labels are inherited', 'Per-region tables depend on the '
         'upstream annotation stage; changing it changes those tables without '
         'changing the diagnostic.'),
        ('Transforms must match', 'Leverage is computed on the methylation '
         'values as fed to the mapping. Cohorts run with different transforms '
         'are not directly comparable.'),
    ]
    return QCModule(
        anchor='caveats', title='Caveats and scope', status='INFO',
        purpose=purpose,
        interpretation='These are limitations of the method as currently '
                       'implemented, not problems with this particular run.',
        table_html=render_table(['Limitation', 'What it means'],
                                [[t, b] for t, b in items], ['left', 'left']))


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description='FastQC-style HTML report for the influence diagnostic.')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--influence-qc', help='influence_qc.json')
    ap.add_argument('--bridge', help='calibration_bridge.json')
    ap.add_argument('--kennedy-influence', help='influence_stratified.json')
    ap.add_argument('--flagged-parquet',
                    help='parquet carrying tecpg_influence_* metadata')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    qc, qc_note = load_json(args.influence_qc, 'influence-qc')
    bridge, bridge_note = load_json(args.bridge, 'bridge')
    ki, ki_note = load_json(args.kennedy_influence, 'kennedy-influence')
    meta = read_flag_metadata(args.flagged_parquet)

    if not any((qc, bridge, ki)):
        sys.exit('No usable inputs: '
                 + '; '.join(str(n) for n in (qc_note, bridge_note, ki_note) if n))
    if not HAVE_MPL:
        print('WARNING: matplotlib unavailable; tables only, no figures.',
              file=sys.stderr)

    modules = [
        mod_provenance(args.dataset, args, qc, bridge, ki, meta),
        mod_landscape(qc, qc_note),
        mod_region(qc, qc_note),
        mod_flag(qc, qc_note, meta),
        mod_coverage(bridge, bridge_note),
        mod_dose(bridge, bridge_note),
        mod_kennedy(ki, ki_note),
        mod_guidance(qc, qc_note, bridge, bridge_note),
        mod_top(qc, qc_note),
        mod_caveats(),
    ]
    doc = render_html(dataset=args.dataset, meta={}, modules=modules,
                      report_title='Influence QC',
                      generator='tools/influence_qc_report.py')
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w') as fh:
        fh.write(doc)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
