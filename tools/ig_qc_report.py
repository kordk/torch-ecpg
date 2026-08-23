#!/usr/bin/env python3
"""ig_qc_report.py — FastQC-style HTML report for Integrated Gradients (IG) output.

Unlike the permute / influence reports, tools/evaluateSaliency.py emits figures
and CSVs rather than a JSON summary, so this tool computes its own statistics
from the catalog parquet and embeds the existing figures alongside them.

  --catalog     parquet carrying mt_ig (and mt_t / mt_est / mt_err / region)
  --plots-dir   directory of evaluateSaliency.py PNGs to embed (optional)
  --top-csv     top50_by_mt_saliency.csv (optional)

Styling and layout are shared with tools/permute_qc_report.py.

Usage:
  python3 tools/ig_qc_report.py --dataset gtp \
      --catalog   output_gtp/bootstrap_merged.parquet \
      --plots-dir output_gtp/plots \
      --top-csv   output_gtp/plots/top50_by_mt_saliency.csv \
      --out       output_gtp/ig_qc_report.html
"""
import argparse
import base64
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from permute_qc_report import QCModule, render_html, render_table  # noqa: E402

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ---- verdict thresholds (documented; not tuned to any dataset) -------------
REDUNDANCY_FAIL = 0.98    # |Spearman(mt_ig, |t|)| at/above this: no new ranking info
REDUNDANCY_WARN = 0.90
TOPK_OVERLAP_FAIL = 0.90  # top-k set overlap with |t| at/above this: same shortlist
TOPK_OVERLAP_WARN = 0.75
TOPK = 1000
SAMPLE_N = 2_000_000      # cap rows used for correlations, deterministic
COL_A = '#3b6ea5'
COL_B = '#c2452d'
COL_MUTE = '#888888'
REGION_ORDER = ['TRANS', 'DISTAL5', 'DISTAL3', 'CIS5', 'CIS3', 'PROMOTER',
                'GENEBODY']
FIGURES = [
    ('saliency_magnitude_decay_curve.png', 'Saliency magnitude vs rank',
     'How quickly |mt_ig| falls away from the top-ranked pair. A sharp knee '
     'means a small candidate set carries most of the attributed magnitude; a '
     'shallow curve means the ranking has no natural cut point.'),
    ('saliency_fraction_decay_curve.png', 'Saliency fraction vs rank',
     'The share of a pair\'s total attribution that lands on methylation '
     'rather than on the covariates, ordered by that share.'),
    ('saliency_fraction_hist.png', 'Distribution of the methylation share',
     'How the methylation share of attribution is distributed across pairs.'),
    ('saliency_fraction_by_region.png', 'Methylation share by region',
     'Whether the share of attribution carried by methylation depends on where '
     'the CpG sits relative to the gene.'),
    ('effect_vs_mad.png', 'Effect size vs methylation spread',
     'The two quantities that mechanically determine IG magnitude, plotted '
     'against each other.'),
    ('saliency_fraction_vs_effect_mad.png',
     'Methylation share vs effect size and spread',
     'How the attribution share tracks effect size once methylation spread is '
     'accounted for.'),
    ('saliency_fraction_vs_standardized_effect.png',
     'Methylation share vs standardized effect',
     'The attribution share against the standardized effect (t-like) scale.'),
    ('saliency_vs_mt_est.png', 'Saliency vs effect estimate',
     'Attribution magnitude against the raw regression coefficient.'),
    ('input_scale_vs_ig.png', 'Input scale vs attribution',
     'Attribution magnitude against the scale of the methylation input.'),
]


# ------------------------------------------------------------------ helpers
def fmt(v, nd=4, pct=False):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return '-'
    if not np.isfinite(f):
        return '-'
    return f'{f * 100:.1f}%' if pct else f'{f:.{nd}f}'


def fmt_int(v):
    try:
        return f'{int(v):,}'
    except (TypeError, ValueError):
        return '-'


def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def png_b64(path):
    try:
        with open(path, 'rb') as fh:
            return base64.b64encode(fh.read()).decode('ascii')
    except OSError:
        return ''


def ordered_regions(names):
    return [r for r in REGION_ORDER if r in names] + \
           sorted(n for n in names if n not in REGION_ORDER)


def not_evaluated(anchor, title, purpose, reason):
    return QCModule(anchor=anchor, title=title, status='INFO', purpose=purpose,
                    interpretation=f'Not evaluated: {reason}. This module is '
                                   'skipped; the rest of the report is '
                                   'unaffected.')


def spearman(a, b):
    """Rank correlation without a scipy dependency."""
    if len(a) < 3:
        return None
    ra = pd.Series(a).rank().to_numpy(dtype=float, copy=True)
    rb = pd.Series(b).rank().to_numpy(dtype=float, copy=True)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else None


# ------------------------------------------------------------------ modules
def mod_introduction(cols):
    purpose = (
        '<strong>Question: what does the Integrated Gradients (IG) column add '
        'to a catalog that already reports an effect size and a p-value?</strong> '
        'IG is an attribution method: for each tested pair it distributes the '
        'model\'s prediction across the input features, so that '
        '<code>mt_ig</code> is the portion attributed to methylation and the '
        'remaining <code>*_ig</code> columns the portions attributed to each '
        'covariate. It is reported to show <em>where a prediction came from</em>, '
        'which a coefficient alone does not.')
    interpretation = (
        '<strong>How it is computed here.</strong> The mapping model is a '
        'linear regression, so the gradient of the prediction with respect to '
        'each input is constant along the integration path. The path integral '
        'therefore collapses to a closed form: the attribution for a feature is '
        'its coefficient multiplied by the average distance of that feature '
        'from the baseline, '
        '<code>|beta_k| &times; mean|x_k &minus; baseline_k|</code>. No neural '
        'network is involved and nothing is approximated numerically.'
        '<br><br>'
        '<strong>What follows from that.</strong> Because IG magnitude is a '
        'deterministic function of the coefficient and the input spread, it '
        'cannot contain evidence that the coefficient and its standard error do '
        'not already carry. It is a <em>decomposition</em>, not an independent '
        'line of evidence, and the modules below test exactly that: how much '
        'independent ranking information <code>mt_ig</code> adds relative to '
        'the t statistic, what drives its magnitude, and what the per-feature '
        'shares do and do not tell you.'
        '<br><br>'
        '<strong>Where it is genuinely useful.</strong> The per-feature '
        'breakdown answers a question the coefficient cannot: of everything '
        'driving this prediction, how much is methylation rather than the '
        'covariates? A pair with a respectable coefficient whose attribution is '
        'dominated by covariates is a different object from one where '
        'methylation carries the prediction, even when their p-values match. '
        'That share, not the raw magnitude, is where the added value lies.'
        '<br><br>'
        f'<strong>Columns present in this catalog:</strong> '
        f'{", ".join(f"<code>{c}</code>" for c in cols) or "none"}.')
    return QCModule(anchor='introduction',
                    title='Introduction: what IG measures here',
                    status='INFO', purpose=purpose,
                    interpretation=interpretation)


def mod_coverage(df, n_total, ig_cols, catalog_path):
    purpose = (
        '<strong>Question: which pairs carry IG at all, and is that set '
        'representative of the catalog?</strong> IG is computed during mapping '
        'only for pairs that survive the p-value filter, and further stages may '
        'subset again, so the attributions describe a candidate set rather than '
        'the genome-wide test space.')
    n_ig = int(df['mt_ig'].notna().sum())
    rows = [['Catalog', catalog_path],
            ['Rows in catalog', fmt_int(n_total)],
            ['Rows carrying mt_ig', fmt_int(n_ig)],
            ['IG coverage', fmt(n_ig / n_total if n_total else None, pct=True)],
            ['Per-feature IG columns', fmt_int(max(len(ig_cols) - 1, 0))]]
    bits = [f'{fmt_int(n_ig)} of {fmt_int(n_total)} rows carry an attribution.']
    if len(ig_cols) > 1:
        bits.append(
            f'Per-feature attributions are present for '
            f'{len(ig_cols) - 1} covariate feature(s), so the methylation '
            f'share of attribution can be computed.')
        status = 'PASS'
    else:
        bits.append(
            'Only the scalar <code>mt_ig</code> is present, with no covariate '
            'attributions. The share modules below cannot be computed: rerun '
            'the mapping with per-feature IG enabled if that comparison is '
            'wanted.')
        status = 'WARN'
    bits.append(
        'Every statistic in this report is conditional on this covered set. '
        'Rank positions and decay curves describe the candidate list, not the '
        'genome-wide distribution.')
    return QCModule(anchor='coverage', title='IG coverage', status=status,
                    purpose=purpose, interpretation=' '.join(bits),
                    table_html=render_table(['Item', 'Value'], rows,
                                            ['left', 'right']))


def mod_redundancy(df):
    purpose = (
        '<strong>Question: does ranking pairs by IG magnitude produce a '
        'different shortlist than ranking them by the t statistic?</strong> '
        'This is the decisive test of whether <code>mt_ig</code> can be treated '
        'as an independent axis for prioritisation. If the two rankings agree, '
        'IG magnitude adds no selection information, however intuitive the '
        'attribution framing may be.')
    need = {'mt_ig', 'mt_t'}
    if not need <= set(df.columns):
        return not_evaluated('redundancy', 'Is IG an independent ranking axis?',
                             purpose, 'mt_ig and mt_t are both required')
    w = df.dropna(subset=['mt_ig', 'mt_t'])
    if len(w) < 100:
        return not_evaluated('redundancy', 'Is IG an independent ranking axis?',
                             purpose, 'fewer than 100 pairs with both columns')
    if len(w) > SAMPLE_N:
        w = w.sample(SAMPLE_N, random_state=0)
    ig = w['mt_ig'].abs().to_numpy()
    at = w['mt_t'].abs().to_numpy()
    rho = spearman(ig, at)
    k = min(TOPK, len(w))
    top_ig = set(np.argsort(-ig)[:k])
    top_t = set(np.argsort(-at)[:k])
    overlap = len(top_ig & top_t) / k
    rows = [['Pairs compared', fmt_int(len(w))],
            ['Spearman correlation, |mt_ig| vs |t|', fmt(rho, 4)],
            [f'Top-{k} shortlist overlap', fmt(overlap, pct=True)],
            [f'Pairs in the IG top-{k} but not the |t| top-{k}',
             fmt_int(k - len(top_ig & top_t))]]
    fig = ''
    if HAVE_MPL:
        s = w.sample(min(20000, len(w)), random_state=0)
        f, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110)
        ax.scatter(s['mt_t'].abs(), s['mt_ig'].abs(), s=4, alpha=0.25,
                   color=COL_A, edgecolors='none')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('|t statistic|')
        ax.set_ylabel('|mt_ig|')
        ax.set_title(f'IG magnitude against |t|  (Spearman {fmt(rho, 3)})',
                     fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        fig = fig_b64(f)
    bits = []
    if rho is not None:
        bits.append(f'The two rankings correlate at Spearman {rho:.3f}, and '
                    f'{overlap * 100:.0f}% of the top-{k} pairs are shared.')
        if abs(rho) >= REDUNDANCY_FAIL or overlap >= TOPK_OVERLAP_FAIL:
            status = 'FAIL'
            bits.append(
                'At this level of agreement the two axes are interchangeable '
                'for selection purposes. Ranking by IG magnitude is ranking by '
                'the t statistic in different units, so IG must not be '
                'presented as independent corroboration of a hit, and a '
                'candidate list built from it should not be described as '
                'derived from a separate method. This is expected for a linear '
                'model - it is the closed form doing what it must - and it is '
                'a statement about the magnitude only, not about the '
                'per-feature shares.')
        elif abs(rho) >= REDUNDANCY_WARN or overlap >= TOPK_OVERLAP_WARN:
            status = 'WARN'
            bits.append(
                'The axes largely agree. IG magnitude may reorder pairs at the '
                'margin, but a shortlist drawn from it will mostly reproduce '
                'the one drawn from the t statistic; treat any difference as '
                'reordering rather than as new evidence.')
        else:
            status = 'PASS'
            bits.append(
                'The rankings differ more than the closed form alone would '
                'predict, which is worth investigating: for a linear model the '
                'two should be tightly coupled, so a low correlation suggests '
                'the spread term (methylation variability) is doing most of '
                'the reordering. Check the drivers module before treating the '
                'difference as signal.')
    else:
        status = 'INFO'
        bits.append('The correlation could not be computed.')
    return QCModule(anchor='redundancy',
                    title='Is IG an independent ranking axis?', status=status,
                    purpose=purpose, interpretation=' '.join(bits),
                    table_html=render_table(['Quantity', 'Value'], rows,
                                            ['left', 'right']),
                    figure_b64=fig, figure_alt='IG magnitude vs |t|')


def mod_drivers(df):
    purpose = (
        '<strong>Question: what actually determines IG magnitude - the strength '
        'of the association, or the spread of the methylation values?</strong> '
        'The closed form is coefficient times average distance from baseline, '
        'so a pair can rank highly either because the effect is large or simply '
        'because that CpG varies a lot across subjects. Separating the two '
        'matters when IG is used to shortlist loci.')
    if 'mt_ig' not in df.columns or 'mt_est' not in df.columns:
        return not_evaluated('drivers', 'What drives IG magnitude', purpose,
                             'mt_ig and mt_est are both required')
    w = df.dropna(subset=['mt_ig', 'mt_est'])
    if len(w) < 100:
        return not_evaluated('drivers', 'What drives IG magnitude', purpose,
                             'too few pairs with both columns')
    if len(w) > SAMPLE_N:
        w = w.sample(SAMPLE_N, random_state=0)
    ig = w['mt_ig'].abs().to_numpy()
    est = w['mt_est'].abs().to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        implied = np.where(est > 0, ig / est, np.nan)
    rho_est = spearman(ig, est)
    ok = np.isfinite(implied)
    rows = [['Pairs compared', fmt_int(len(w))],
            ['Spearman, |mt_ig| vs |effect size|', fmt(rho_est, 4)],
            ['Implied methylation spread (|mt_ig| / |effect|), median',
             fmt(np.nanmedian(implied[ok]) if ok.any() else None, 5)],
            ['Implied spread, 5th - 95th percentile',
             (f'{fmt(np.nanpercentile(implied[ok], 5), 5)} - '
              f'{fmt(np.nanpercentile(implied[ok], 95), 5)}') if ok.any() else '-']]
    fig = ''
    if HAVE_MPL and ok.any():
        f, ax = plt.subplots(figsize=(6.4, 3.4), dpi=110)
        v = implied[ok]
        v = v[(v > 0) & np.isfinite(v)]
        if len(v):
            ax.hist(np.log10(v), bins=60, color=COL_A)
            ax.set_xlabel('log10 of implied methylation spread '
                          '(|mt_ig| / |effect|)')
            ax.set_ylabel('pairs')
            ax.spines[['top', 'right']].set_visible(False)
            fig = fig_b64(f)
        else:
            plt.close(f)
    spread_range = None
    if ok.any():
        lo = np.nanpercentile(implied[ok], 5)
        hi = np.nanpercentile(implied[ok], 95)
        spread_range = (hi / lo) if lo and lo > 0 else None
    bits = []
    if rho_est is not None:
        bits.append(f'IG magnitude tracks the raw effect size at Spearman '
                    f'{rho_est:.3f}.')
    if spread_range:
        bits.append(
            f'Dividing IG by the effect size recovers the other factor - the '
            f'average distance of the methylation values from the baseline. '
            f'Across the middle 90% of pairs that factor varies '
            f'{spread_range:.0f}-fold, so two pairs with identical effect sizes '
            f'can differ in IG magnitude by that much purely because one CpG is '
            f'more variable across subjects.')
        bits.append(
            'This is the practical consequence: a high IG magnitude is not a '
            'statement that the association is strong. Highly variable probes '
            'are promoted and stable ones demoted regardless of evidence '
            'strength. Where a variability-independent ranking is wanted, use '
            'the t statistic or the FDR; where variability is itself of '
            'interest, IG magnitude captures it - but say so explicitly.')
    status = 'INFO'
    return QCModule(anchor='drivers', title='What drives IG magnitude',
                    status=status, purpose=purpose,
                    interpretation=' '.join(bits),
                    table_html=render_table(['Quantity', 'Value'], rows,
                                            ['left', 'right']),
                    figure_b64=fig, figure_alt='implied spread distribution')


def mod_share(df, ig_cols):
    purpose = (
        '<strong>Question: of everything driving a prediction, how much is '
        'attributed to methylation rather than to the covariates?</strong> '
        'This is the part of IG that the coefficient genuinely cannot supply, '
        'and the reason to compute attributions at all.')
    if len(ig_cols) <= 1:
        return not_evaluated('share', 'Methylation share of attribution',
                             purpose,
                             'the catalog carries only the scalar mt_ig, with '
                             'no per-feature covariate attributions')
    w = df.dropna(subset=['mt_ig']).copy()
    denom = w[ig_cols].abs().sum(axis=1)
    share = (w['mt_ig'].abs() / denom).replace([np.inf, -np.inf], np.nan)
    share = share.dropna()
    if share.empty:
        return not_evaluated('share', 'Methylation share of attribution',
                             purpose, 'the attribution total is zero for every '
                                      'row')
    q = share.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    rows = [['Pairs with a computable share', fmt_int(len(share))],
            ['Median methylation share', fmt(q[0.5], pct=True)],
            ['5th - 95th percentile',
             f'{fmt(q[0.05], pct=True)} - {fmt(q[0.95], pct=True)}'],
            ['Pairs where methylation carries under 10%',
             fmt((share < 0.10).mean(), pct=True)],
            ['Pairs where methylation carries over 50%',
             fmt((share > 0.50).mean(), pct=True)]]
    fig = ''
    if HAVE_MPL:
        f, ax = plt.subplots(figsize=(6.4, 3.4), dpi=110)
        ax.hist(share.to_numpy(), bins=60, color=COL_A)
        ax.axvline(float(q[0.5]), color=COL_B, ls='--', lw=1,
                   label=f'median {q[0.5] * 100:.1f}%')
        ax.set_xlabel('methylation share of total attribution')
        ax.set_ylabel('pairs')
        ax.legend(frameon=False, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        fig = fig_b64(f)
    bits = [
        f'Across pairs the median methylation share is '
        f'{q[0.5] * 100:.1f}%, with the middle 90% spanning '
        f'{q[0.05] * 100:.1f}% to {q[0.95] * 100:.1f}%.']
    bits.append(
        'Read a low share as a warning about interpretation rather than about '
        'correctness: the association may be real, but the prediction for that '
        'pair is mostly carried by covariates, so describing it as a '
        'methylation-driven relationship overstates what the model is using.')
    bits.append(
        'One caution on the denominator. Expression principal components, if '
        'they are among the covariates, are near-proxies for the outcome and '
        'will absorb most of the attribution, pushing every methylation share '
        'down. If those features are present, recompute the share with them '
        'excluded (evaluateSaliency.py --frac-exclude) before comparing pairs.')
    return QCModule(anchor='share', title='Methylation share of attribution',
                    status='INFO', purpose=purpose,
                    interpretation=' '.join(bits),
                    table_html=render_table(['Quantity', 'Value'], rows,
                                            ['left', 'right']),
                    figure_b64=fig, figure_alt='methylation share histogram')


def mod_region(df):
    purpose = (
        '<strong>Question: does IG magnitude differ systematically by genomic '
        'region?</strong> Since IG magnitude depends on effect size and probe '
        'variability, a strong regional pattern usually reflects the '
        'composition of each region rather than a difference in attribution '
        'behaviour.')
    if 'region' not in df.columns or 'mt_ig' not in df.columns:
        return not_evaluated('region', 'IG by region', purpose,
                             'region and mt_ig are both required')
    w = df.dropna(subset=['mt_ig'])
    if w.empty:
        return not_evaluated('region', 'IG by region', purpose,
                             'no rows carry mt_ig')
    g = w.groupby('region')['mt_ig'].agg(
        n='size', median=lambda s: s.abs().median(),
        q95=lambda s: s.abs().quantile(0.95))
    regs = ordered_regions(list(g.index))
    rows = [[r, fmt_int(g.loc[r, 'n']), fmt(g.loc[r, 'median'], 5),
             fmt(g.loc[r, 'q95'], 5)] for r in regs]
    meds = [float(g.loc[r, 'median']) for r in regs
            if np.isfinite(g.loc[r, 'median'])]
    ratio = (max(meds) / min(meds)) if meds and min(meds) > 0 else None
    bits = []
    if ratio:
        bits.append(
            f'Median IG magnitude varies {ratio:.1f}-fold across regions.'
            + (' That is modest, and consistent with regions differing in the '
               'effect sizes they contain rather than in how attribution '
               'behaves.' if ratio < 3 else
               ' That is a large spread; check whether the regions differ in '
               'probe variability before reading it as a difference in signal '
               'strength.'))
    bits.append(
        'Region counts here reflect the IG-covered subset, which is filtered by '
        'p-value, so they are not the genome-wide region composition.')
    return QCModule(anchor='region', title='IG by region', status='INFO',
                    purpose=purpose, interpretation=' '.join(bits),
                    table_html=render_table(
                        ['Region', 'Pairs with IG', 'Median |mt_ig|',
                         '95th percentile'], rows,
                        ['left', 'right', 'right', 'right']))


def mod_figures(plots_dir):
    purpose = (
        '<strong>Question: does the attribution ranking have a natural cut '
        'point, and how do the attributions relate to effect size and probe '
        'variability?</strong> These are the figures produced by '
        'evaluateSaliency.py, reproduced here so the diagnostics and their '
        'interpretation sit in one document.')
    if not plots_dir or not os.path.isdir(plots_dir):
        return not_evaluated('figures', 'Saliency diagnostic figures', purpose,
                             'no --plots-dir supplied or the directory does '
                             'not exist')
    found = []
    for name, title, blurb in FIGURES:
        b64 = png_b64(os.path.join(plots_dir, name))
        if b64:
            found.append((name, title, blurb, b64))
    if not found:
        return not_evaluated('figures', 'Saliency diagnostic figures', purpose,
                             f'no known saliency figures found in {plots_dir}')
    blocks = []
    for name, title, blurb, b64 in found:
        blocks.append(
            f'<p><strong>{title}</strong><br>{blurb}<br>'
            f'<code>{name}</code></p>'
            f'<img src="data:image/png;base64,{b64}" alt="{title}">')
    bits = [
        f'{len(found)} of {len(FIGURES)} known saliency figures were found and '
        f'embedded.']
    bits.append(
        'On the decay curves specifically: an inflection point identifies where '
        'the ranked magnitudes stop falling steeply, which is useful for '
        'choosing how many candidates to carry forward. It is a property of '
        'this candidate list, not a significance threshold, and it will move '
        'when the p-value filter or the cohort changes.')
    bits.append(
        'The effect-size and spread scatter plots are the visual form of the '
        'closed form: pairs sit high because the coefficient is large, because '
        'the probe is variable, or both, and these figures show which.')
    return QCModule(anchor='figures', title='Saliency diagnostic figures',
                    status='INFO', purpose=purpose,
                    interpretation=' '.join(bits),
                    table_html='\n'.join(blocks))


def mod_top(top_csv):
    purpose = (
        '<strong>Question: which specific pairs rank highest by attribution, '
        'and do they look like strong associations or like variable '
        'probes?</strong> Named pairs allow the ranking to be checked against '
        'the effect sizes and statistics in the same row.')
    if not top_csv or not os.path.exists(top_csv):
        return not_evaluated('top-pairs', 'Highest-attribution pairs', purpose,
                             'no --top-csv supplied or the file does not exist')
    try:
        t = pd.read_csv(top_csv)
    except Exception as exc:                                   # noqa: BLE001
        return not_evaluated('top-pairs', 'Highest-attribution pairs', purpose,
                             f'could not read {top_csv}: {exc}')
    if t.empty:
        return not_evaluated('top-pairs', 'Highest-attribution pairs', purpose,
                             'the top-pairs file is empty')
    t = t.head(25)
    hdr = list(t.columns)
    rows = [[('' if pd.isna(v) else
              (f'{v:.4g}' if isinstance(v, (float, np.floating)) else str(v)))
             for v in r] for r in t.to_numpy()]
    bits = [f'Showing the top {len(t)} rows from '
            f'<code>{os.path.basename(top_csv)}</code>.']
    bits.append(
        'Check these against the t statistic in the same row. Pairs that rank '
        'highly by attribution but only modestly by t are being promoted by '
        'probe variability, which is the expected behaviour of the closed form '
        'and not a sign of stronger evidence.')
    return QCModule(anchor='top-pairs', title='Highest-attribution pairs',
                    status='INFO', purpose=purpose,
                    interpretation=' '.join(bits),
                    table_html=render_table(hdr, rows,
                                            ['left'] + ['right'] * (len(hdr) - 1)))


def mod_caveats():
    purpose = (
        '<strong>Question: what conclusions are not licensed by IG output?</strong> '
        'These limits apply to every number above.')
    items = [
        ('IG magnitude is not independent evidence',
         'For a linear model it is a closed-form transform of the coefficient '
         'and the input spread. It cannot corroborate a result that the '
         'coefficient and standard error do not already support.'),
        ('Magnitude rewards variable probes',
         'Two pairs with the same effect size differ in IG magnitude purely by '
         'how variable their methylation is across subjects.'),
        ('The share depends on the covariate set',
         'Adding, removing, or residualising covariates changes every '
         'methylation share. Shares are comparable only within one covariate '
         'specification.'),
        ('Attribution is not causal',
         'IG describes how a fitted model distributes its prediction across '
         'inputs. It says nothing about mechanism or direction of effect.'),
        ('Coverage is a candidate set',
         'IG is computed on pairs surviving the p-value filter, so ranks, '
         'decay curves, and inflection points describe that list rather than '
         'the genome-wide test space.'),
        ('Deep IG is not in use',
         'The reported attributions come from the closed form for the linear '
         'model. Any path-integral or neural-network framing describes '
         'architectural intent, not what produced these numbers.'),
    ]
    return QCModule(
        anchor='caveats', title='Caveats and scope', status='INFO',
        purpose=purpose,
        interpretation='These are properties of the method as implemented, not '
                       'defects in this particular run.',
        table_html=render_table(['Limitation', 'What it means'],
                                [[a, b] for a, b in items], ['left', 'left']))


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description='FastQC-style HTML report for Integrated Gradients output.')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--catalog', required=True,
                    help='parquet carrying mt_ig (e.g. bootstrap_merged.parquet)')
    ap.add_argument('--plots-dir', help='directory of evaluateSaliency.py PNGs')
    ap.add_argument('--top-csv', help='top50_by_mt_saliency.csv')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if not os.path.exists(args.catalog):
        sys.exit(f'catalog not found: {args.catalog}')
    import pyarrow.parquet as pq
    names = pq.ParquetFile(args.catalog).schema_arrow.names
    if 'mt_ig' not in names:
        sys.exit(f'{args.catalog} has no mt_ig column; nothing to report. '
                 f'Run the mapping with --compute-ig.')
    ig_cols = [c for c in names if c.endswith('_ig')]
    want = [c for c in ('mt_id', 'gt_id', 'mt_est', 'mt_err', 'mt_t', 'mt_p',
                        'region', 'fdr_est') if c in names] + ig_cols
    df = pq.read_table(args.catalog, columns=want).to_pandas()
    if df.index.names != [None]:
        df = df.reset_index()
    n_total = len(df)
    if not HAVE_MPL:
        print('WARNING: matplotlib unavailable; tables only, no computed '
              'figures.', file=sys.stderr)

    modules = [
        mod_introduction(ig_cols),
        mod_coverage(df, n_total, ig_cols, args.catalog),
        mod_redundancy(df),
        mod_drivers(df),
        mod_share(df, ig_cols),
        mod_region(df),
        mod_figures(args.plots_dir),
        mod_top(args.top_csv),
        mod_caveats(),
    ]
    doc = render_html(dataset=args.dataset, meta={}, modules=modules,
                      report_title='IG QC',
                      generator='tools/ig_qc_report.py')
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w') as fh:
        fh.write(doc)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
