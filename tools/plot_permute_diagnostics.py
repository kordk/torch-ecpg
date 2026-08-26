#!/usr/bin/env python3
"""plot_permute_diagnostics.py -- diagnostic figures for qr_permute output.

Renders three diagnostics comparing permutation-derived p-values against the
analytic ones, for one or more runs side by side:

  tail_divergence   Median log10(p_perm / p_analytic) as a function of |t|.
                    Resolves agreement as a function of depth rather than
                    collapsing it to a single number. Values near 0 indicate
                    the two methods agree; positive values indicate the
                    permutation assigns larger p-values than the parametric
                    null, i.e. the analytic p is anti-conservative there.

  xi_convergence    GPD shape parameter across the threshold ladder recorded in
                    the eval report's xi_sweep, optionally against the 1/df a
                    t-distributed null would predict. A shape parameter that
                    varies with the threshold indicates the fit is not in its
                    asymptotic regime and the provisional threshold warrants
                    revisiting.

  ranking_recovery  Distribution of permutation p-values among pairs whose
                    analytic p underflowed to exactly zero. Where the analytic
                    p is stored in float32, values below 2^-24 collapse to a
                    single tied value and cannot be ranked; this panel shows
                    what range the permutation resolves in their place.

Usage:

    python3 plot_permute_diagnostics.py OUTDIR[:LABEL[:DF]] ... [--outdir DIR]

Each positional argument names a run directory. Artifacts are discovered inside
it by the pipeline's naming convention, and any that are absent are skipped
rather than failing:

    permutation_results.parquet          (required)
    permutation_results.perm_null.npz    (optional: threshold and shape markers)
    eval_permute_report.json             (optional: the xi sweep)

LABEL defaults to the directory name. DF is the residual degrees of freedom of
the run, used only to draw the t-null reference in the shape figure; omit it and
that reference is left out.

Examples:

    python3 plot_permute_diagnostics.py output_gtp output_mesa
    python3 plot_permute_diagnostics.py output_a:CohortA:330 output_b:CohortB:1171
    python3 plot_permute_diagnostics.py output_gtp --bin-max 8 --outdir figures

Read-only with respect to run outputs. Writes PDF (vector) and PNG.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Artifact names, as written by pipelinePermute.sh.
# Preferred first: the joined file carries float64 precise_mt_p from the
# mainline catalog. The permute chain branches upstream of pipeline.sh stage
# [6/9], so its own output has only float32 mt_p.
PRECISE_PARQUET_NAME = 'permutation_results.precise.parquet'
PARQUET_NAME = 'permutation_results.parquet'
SIDECAR_NAME = 'permutation_results.perm_null.npz'
REPORT_NAME = 'eval_permute_report.json'
JOIN_TOOL = 'tools/join_precise_p_permute.py'

# Analytic p stored in float32 underflows to exactly zero below this value.
FLOAT32_MIN_NORMAL_P = 2.0 ** -24

# Colourblind-safe qualitative palette (Okabe-Ito subset), cycled per run.
PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
GREY = '#6c757d'

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.6,
    'legend.frameon': False,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})


# --------------------------------------------------------------------- io
def parse_run_spec(spec):
    """DIR[:LABEL[:DF]] -> (dir, label, df_or_None)."""
    parts = spec.split(':')
    d = parts[0]
    label = parts[1] if len(parts) > 1 and parts[1] else os.path.basename(d.rstrip('/'))
    df = None
    if len(parts) > 2 and parts[2]:
        try:
            df = float(parts[2])
        except ValueError:
            raise SystemExit("bad DF in run spec {!r}".format(spec))
    if len(parts) > 3:
        raise SystemExit("too many fields in run spec {!r}".format(spec))
    return d, label, df


def _join_hint(d):
    """Instructions for producing the joined, float64 input."""
    return (
        "  Produce it with:\n"
        "    python3 {tool} \\\n"
        "        --permute {d}/{pq} \\\n"
        "        --catalog {d}/summarized.parquet \\\n"
        "        --out     {d}/{px}\n"
        "  (adjust --catalog if the mainline catalog lives elsewhere)"
    ).format(tool=JOIN_TOOL, d=d.rstrip('/'),
             pq=PARQUET_NAME, px=PRECISE_PARQUET_NAME)


def load_run(d, label, df, require_precise):
    precise = os.path.join(d, PRECISE_PARQUET_NAME)
    plain = os.path.join(d, PARQUET_NAME)

    if os.path.exists(precise):
        pq = precise
        source = 'joined'
    elif os.path.exists(plain):
        if require_precise:
            raise SystemExit(
                "{}: {} not found.\n"
                "  Only {} is present, whose analytic p is float32 and\n"
                "  underflows to exactly 0 below 2^-24 (5.96e-08). Comparisons\n"
                "  above that point would be undefined.\n{}\n"
                "  Or pass --allow-float32 to proceed with the truncated column."
                .format(label, precise, PARQUET_NAME, _join_hint(d)))
        pq = plain
        source = 'permute-only'
    else:
        raise SystemExit(
            "{}: neither {} nor {} found in {}".format(
                label, PRECISE_PARQUET_NAME, PARQUET_NAME, d))

    frame = pd.read_parquet(pq)
    for need in ('mt_t', 'perm_mt_p'):
        if need not in frame.columns:
            raise SystemExit("{}: {} has no '{}' column".format(label, pq, need))
    acol = next((c for c in ('precise_mt_p', 'mt_p') if c in frame.columns), None)
    if acol is None:
        raise SystemExit("{}: no analytic p column (precise_mt_p or mt_p)".format(label))

    run = {
        'label': label,
        'dir': d,
        'df': df,
        'source': source,
        'parquet': pq,
        'analytic_col': acol,
        't': np.abs(frame['mt_t'].to_numpy(np.float64)),
        'pa': frame[acol].to_numpy(np.float64),
        'pp': frame['perm_mt_p'].to_numpy(np.float64),
        'u': None,
        'xi': None,
        'sweep': None,
        'h': (frame['mt_h_max'].to_numpy(np.float64)
              if 'mt_h_max' in frame.columns else None),
        'mt_id': (frame['mt_id'].to_numpy()
                  if 'mt_id' in frame.columns else None),
    }

    sc = os.path.join(d, SIDECAR_NAME)
    if os.path.exists(sc):
        try:
            z = np.load(sc, allow_pickle=False)
            if 'gpd_u' in z and np.isfinite(float(z['gpd_u'])):
                run['u'] = float(z['gpd_u'])
            if 'gpd_xi' in z and np.isfinite(float(z['gpd_xi'])):
                run['xi'] = float(z['gpd_xi'])
        except Exception as exc:
            print("  {}: sidecar unreadable ({}); markers omitted".format(label, exc))

    rp = os.path.join(d, REPORT_NAME)
    if os.path.exists(rp):
        try:
            r = json.load(open(rp))
            run['sweep'] = r.get('arms', {}).get('sidecar', {}).get('xi_sweep')
        except Exception as exc:
            print("  {}: report unreadable ({}); xi sweep omitted".format(label, exc))

    n_zero = int((run['pa'] == 0).sum())
    print("  {}: {:,} pairs | {} | analytic column '{}' | analytic p == 0: {:,}".format(
        label, len(run['t']), os.path.basename(pq), acol, n_zero))
    if acol != 'precise_mt_p':
        print("    WARNING: using float32 '{}'. It underflows to exactly 0 below\n"
              "    5.96e-08, so {:,} pairs have no defined ratio and the curve\n"
              "    will truncate there.".format(acol, n_zero))
        if source == 'joined':
            print("    {} exists but carries no precise_mt_p column -- the join\n"
                  "    likely did not find it in the catalog. Check that the\n"
                  "    mainline catalog passed to --catalog has been through\n"
                  "    pipeline.sh stage [6/9]."
                  .format(os.path.basename(pq)))
        else:
            print(_join_hint(d))
    return run


def save(fig, outdir, name, formats):
    for ext in formats:
        p = os.path.join(outdir, '{}.{}'.format(name, ext))
        fig.savefig(p, dpi=300 if ext == 'png' else None)
        print("  wrote {}".format(p))
    plt.close(fig)


# --------------------------------------------------------------- figure 1
def ratio_curve(run, lo, hi, step, min_n):
    edges = np.arange(lo, hi + step, step)
    ok = (run['pa'] > 0) & (run['pp'] > 0) & np.isfinite(run['t'])
    lr = np.log10(run['pp'][ok] / run['pa'][ok])
    tt = run['t'][ok]
    x, med, q25, q75 = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (tt >= a) & (tt < b)
        if int(m.sum()) < min_n:
            continue
        v = lr[m]
        x.append((a + b) / 2.0)
        med.append(np.median(v))
        q25.append(np.percentile(v, 25))
        q75.append(np.percentile(v, 75))
    return map(np.array, (x, med, q25, q75))


def fig_tail_divergence(runs, args, outdir):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    drew = False
    for run in runs:
        x, med, q25, q75 = ratio_curve(run, args.bin_min, args.bin_max,
                                       args.bin_step, args.min_bin_n)
        if x.size == 0:
            continue
        drew = True
        c = run['colour']
        lab = run['label']
        if run['df'] is not None:
            lab += '  ($\\nu$={:g})'.format(run['df'])
        if run['xi'] is not None:
            lab += '  ($\\xi$={:.3f})'.format(run['xi'])
        ax.fill_between(x, q25, q75, color=c, alpha=0.13, linewidth=0)
        ax.plot(x, med, color=c, lw=2.2, label=lab)
        if run['u'] is not None:
            ax.axvline(run['u'], color=c, ls=':', lw=1.2, alpha=0.7)
    if not drew:
        plt.close(fig)
        print("  tail_divergence: no bin met --min-bin-n; skipped")
        return

    ax.axhline(0, color='k', lw=1.0, alpha=0.6)
    ymin, ymax = ax.get_ylim()

    # A fold-change axis is only legible over a modest span. Past a couple of
    # decades the ticks pile up and the left axis already says it in decades.
    if (ymax - ymin) <= 3.0:
        ax2 = ax.twinx()
        ax2.set_ylim(ymin, ymax)
        fold = [0.5, 1, 1.5, 2, 3, 5, 10, 20, 50, 100, 300, 1000]
        ticks = [np.log10(f) for f in fold if ymin <= np.log10(f) <= ymax]
        ax2.set_yticks(ticks)
        ax2.set_yticklabels(['{:g}x'.format(10 ** t) for t in ticks])
        ax2.set_ylabel('$p_{\\mathrm{perm}}$ / $p_{\\mathrm{analytic}}$')
        ax2.grid(False)
        ax2.spines['top'].set_visible(False)
    else:
        ax.set_ylabel('median $\\log_{10}(p_{\\mathrm{perm}} / p_{\\mathrm{analytic}})$'
                      '\n(decades; higher = analytic $p$ more anti-conservative)')

    # Where the analytic p underflows, the ratio is undefined and the curve ends.
    cutoffs = [float(r['t'][r['pa'] > 0].max())
               for r in runs if (r['pa'] > 0).any()]
    if cutoffs and min(cutoffs) < ax.get_xlim()[1]:
        cut = min(cutoffs)
        ax.axvline(cut, color=GREY, lw=1.4, alpha=0.55)
        ax.annotate('analytic $p$ underflows to 0\nbeyond here; ratio undefined',
                    xy=(cut, ymin + 0.62 * (ymax - ymin)), xytext=(-8, 0),
                    textcoords='offset points', ha='right', va='center',
                    fontsize=9, color=GREY, style='italic')

    ax.set_xlabel('|t|')
    ax.set_ylabel('median $\\log_{10}(p_{\\mathrm{perm}} / p_{\\mathrm{analytic}})$')
    ax.set_title('Permutation vs analytic $p$-values across the |t| range')
    ax.legend(loc='upper left')
    fig.text(0.005, -0.055,
             'Shaded band: interquartile range.   Dotted vertical: GPD threshold $u$.   '
             'Values above 0: permutation $p$ exceeds analytic $p$.',
             fontsize=8, color=GREY, ha='left')
    save(fig, outdir, 'tail_divergence', args.formats)


# --------------------------------------------------------------- figure 2
def fig_xi_convergence(runs, args, outdir):
    with_sweep = [r for r in runs if r.get('sweep')]
    if not with_sweep:
        print("  xi_convergence: no xi_sweep in any report; skipped")
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    refs = []
    for run in with_sweep:
        sw = [w for w in run['sweep'] if w.get('xi') is not None]
        if not sw:
            continue
        u = [w['u'] for w in sw]
        xi = [w['xi'] for w in sw]
        ax.plot(u, xi, 'o-', color=run['colour'], lw=2.0, ms=6,
                label='{}  ($\\xi$ {:.3f}$\\rightarrow${:.3f})'.format(
                    run['label'], xi[0], xi[-1]))
        if run['df']:
            refs.append(1.0 / run['df'])

    ax.axhline(0, color='k', lw=0.9, alpha=0.5)
    if refs:
        ax.axhspan(0, max(refs), color=GREY, alpha=0.30, linewidth=0)
        txt = ('$t$-null predicts $\\xi \\approx 1/\\nu$ = {:.4f}'.format(refs[0])
               if len(set(refs)) == 1 else
               '$t$-null predicts $\\xi \\approx 1/\\nu$ ({:.4f}\u2013{:.4f})'.format(
                   min(refs), max(refs)))
        ax.annotate(txt + ' \u2014 shaded',
                    xy=(0.5, max(refs)), xycoords=('axes fraction', 'data'),
                    xytext=(0, 26), textcoords='offset points',
                    ha='center', fontsize=9, color=GREY,
                    arrowprops=dict(arrowstyle='-|>', color=GREY, lw=1.0))

    ax.set_xlabel('GPD threshold $u$  (|t|)')
    ax.set_ylabel('GPD shape parameter $\\xi$')
    ax.set_title('GPD shape parameter across the threshold ladder')
    ax.legend(loc='best')
    fig.text(0.005, -0.055,
             '$\\xi$ varying with $u$ indicates the fit is not in its asymptotic '
             'regime and the threshold warrants revisiting.',
             fontsize=8, color=GREY, ha='left')
    save(fig, outdir, 'xi_convergence', args.formats)


# --------------------------------------------------------------- figure 3
def fig_ranking_recovery(runs, args, outdir):
    counts = {r['label']: int((r['pa'] == 0).sum()) for r in runs}
    usable = [r for r in runs
              if counts[r['label']] >= args.min_recovery_n
              and (r['pp'][r['pa'] == 0] > 0).any()]
    if not usable:
        print("  ranking_recovery: skipped. Pairs with analytic p == 0 per run: "
              + ", ".join("{}={:,}".format(k, v) for k, v in counts.items())
              + " (need >= {}).".format(args.min_recovery_n))
        if any(r['analytic_col'] == 'precise_mt_p' for r in runs):
            print("    This is the expected outcome when the analytic p is\n"
                  "    float64: only values below the subnormal limit (~4.9e-324)\n"
                  "    reach exactly 0, so there is essentially no ordering for\n"
                  "    the permutation to recover. A large count here under\n"
                  "    float32 'mt_p' is a storage artifact of the permute chain,\n"
                  "    not a property of the analytic method.")
        return
    skipped = [r['label'] for r in runs if r not in usable]
    if skipped:
        print("  ranking_recovery: omitting {} (fewer than {} pairs with "
              "analytic p == 0)".format(", ".join(skipped), args.min_recovery_n))
    fig, axes = plt.subplots(1, len(usable), figsize=(4.3 * len(usable), 4.3),
                             squeeze=False)
    for ax, run in zip(axes[0], usable):
        z = run['pa'] == 0
        pp0 = run['pp'][z]
        pp0 = pp0[pp0 > 0]
        lp = -np.log10(pp0)
        ax.hist(lp, bins=45, color=run['colour'], alpha=0.82,
                edgecolor='white', linewidth=0.4)
        pad = 0.04 * max(lp.max() - lp.min(), 1e-9)
        ax.set_xlim(lp.min() - pad, lp.max() + pad)
        floor = -np.log10(FLOAT32_MIN_NORMAL_P)
        if lp.min() - pad <= floor <= lp.max() + pad:
            ax.axvline(floor, color='k', ls='--', lw=1.2, alpha=0.6)
            ax.annotate('float32 limit', xy=(floor, ax.get_ylim()[1] * 0.55),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=8, color=GREY, rotation=90, va='center')
        ax.set_title('{}\n{:,} pairs with analytic $p$ = 0'.format(
            run['label'], int(z.sum())), fontsize=11)
        ax.set_xlabel('$-\\log_{10}(p_{\\mathrm{perm}})$')
        ax.set_ylabel('pairs' if ax is axes[0][0] else '')
        ax.text(0.97, 0.95,
                'analytic: 1 tied value\npermutation: spans\n{:.0f} orders of magnitude'.format(
                    lp.max() - lp.min()),
                transform=ax.transAxes, ha='right', va='top', fontsize=9)
    fig.suptitle('Permutation $p$-values where the analytic $p$ underflows\n'
                 '(float32 $p$ collapses to 0 below $2^{-24}$ = 5.96e-08)',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, outdir, 'ranking_recovery', args.formats)


# --------------------------------------------------------------- figure 4
def fig_leverage_enrichment(runs, args, outdir):
    """Are the extreme statistics leverage-driven?

    The permuted null is far heavier-tailed than a t-null predicts. Single-
    sample leverage is a candidate mechanism: a high-leverage sample inflates
    |t| regardless of how labels are assigned, which is exactly the structure
    the t-distribution assumes away.

    Note this cannot be posed as "does the p-ratio vary with leverage" --
    perm_mt_p is a function of |t| alone through the pooled null, so at fixed
    |t| the ratio is fixed. The well-posed question is whether the pairs that
    reach extreme |t| are themselves enriched for high-leverage CpGs.
    """
    usable = [r for r in runs if r.get('h') is not None]
    if not usable:
        print("  leverage_enrichment: no run carries mt_h_max; skipped "
              "(join it from the mainline catalog, or map with --compute-influence)")
        return

    ladder = [t for t in (4, 5, 6, 8, 10, 15, 20, 30, 50)]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    axL, axR = axes

    for run in usable:
        h = run['h']
        t = run['t']
        ok = np.isfinite(h) & np.isfinite(t)
        ids = run.get('mt_id')
        ids = ids[ok] if ids is not None else None
        h, t = h[ok], t[ok]
        if h.size == 0:
            continue
        cut = float(np.quantile(h, 1.0 - args.leverage_top_frac))
        base = float((h >= cut).mean())          # == leverage_top_frac by construction
        floor = float(h.min())

        # CpG-level baseline: pairs sharing a CpG are not independent, so the
        # pair-level rate over-weights prolific CpGs. Recompute over distinct
        # CpGs as a robustness overlay.
        cpg_base = None
        if ids is not None:
            uid, first = np.unique(ids, return_index=True)
            cpg_base = float((h[first] >= cut).mean())

        xs, enr, meds, enr_cpg = [], [], [], []
        for thr in ladder:
            m = t >= thr
            n = int(m.sum())
            if n < args.min_bin_n:
                continue
            xs.append(thr)
            enr.append(float((h[m] >= cut).mean()) / base if base > 0 else np.nan)
            meds.append(float(np.median(h[m])))
            if cpg_base:
                sid, sfirst = np.unique(ids[m], return_index=True)
                hm = h[m]
                enr_cpg.append(float((hm[sfirst] >= cut).mean()) / cpg_base)

        if not xs:
            continue
        c = run['colour']
        axL.plot(xs, enr, 'o-', color=c, lw=2.0, ms=6, label=run['label'])
        if enr_cpg:
            axL.plot(xs, enr_cpg, 'o--', color=c, lw=1.4, ms=5,
                     mfc='white', alpha=0.85,
                     label='{} (distinct CpGs)'.format(run['label']))
        axR.plot(xs, meds, 'o-', color=c, lw=2.0, ms=6, label=run['label'])
        axR.axhline(float(np.median(h)), color=c, ls=':', lw=1.2, alpha=0.7)
        axR.axhline(floor, color=c, ls='--', lw=1.0, alpha=0.45)
        print("    {}: leverage top-{:.0%} cutoff h={:.4f}, catalog median "
              "h={:.4f}, floor h={:.4f}".format(
                  run['label'], args.leverage_top_frac, cut,
                  float(np.median(h)), floor))

    axL.axhline(1.0, color='k', lw=1.0, alpha=0.6)
    for _ax in (axL, axR):
        _ax.set_xscale('log')
        _ax.set_xticks(ladder)
        _ax.set_xticklabels([str(v) for v in ladder])
        _ax.minorticks_off()
    axL.set_xlabel('|t| threshold')
    axL.set_ylabel('enrichment for top-{:.0%} leverage CpGs'.format(
        args.leverage_top_frac))
    axL.set_title('Enrichment among extreme statistics')
    axL.legend(loc='best')

    axR.set_xlabel('|t| threshold')
    axR.set_ylabel('median $h_{\\max}$ of pairs above threshold')
    axR.set_title('Leverage of the pairs above each threshold')
    axR.legend(loc='best')

    fig.suptitle('Is the heavy tail leverage-driven?', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.text(0.005, -0.045,
             'Left: 1.0 means no enrichment. Right: dotted = catalog median, '
             'dashed = observed minimum (the covariate-only leverage floor).   '
             'Left, open markers: recomputed over distinct CpGs, since pairs '
             'sharing a CpG are not independent.',
             fontsize=8, color=GREY, ha='left')
    save(fig, outdir, 'leverage_enrichment', args.formats)

# --------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', metavar='OUTDIR[:LABEL[:DF]]',
                    help='run directory, optional display label, optional df')
    ap.add_argument('--outdir', default='figures')
    ap.add_argument('--formats', default='pdf,png',
                    help='comma-separated output formats (default: pdf,png)')
    ap.add_argument('--bin-min', type=float, default=None,
                    help='lower |t| edge. Defaults to 4.0 when reading joined '
                         'files, because the mainline catalog is p-thresholded '
                         'and its coverage below |t|~4 is a biased remnant '
                         'rather than a sample; 1.0 otherwise.')
    ap.add_argument('--bin-max', type=float, default=None,
                    help='upper |t| edge; default: the largest |t| with a '
                         'defined ratio across all runs')
    ap.add_argument('--bin-step', type=float, default=0.25)
    ap.add_argument('--min-bin-n', type=int, default=30,
                    help='bins with fewer pairs are dropped (default: 30)')
    ap.add_argument('--leverage-top-frac', type=float, default=0.10,
                    help='fraction of the catalog defining "high leverage", '
                         'taken from each run own mt_h_max distribution '
                         '(default: 0.10)')
    ap.add_argument('--min-recovery-n', type=int, default=500,
                    help='minimum pairs with analytic p == 0 for the '
                         'ranking_recovery panel to be drawn (default: 500). '
                         'Below this the histogram carries no information.')
    ap.add_argument('--allow-float32', action='store_true',
                    help='proceed when only the un-joined permute output is '
                         'present; its float32 analytic p underflows to 0 and '
                         'the tail comparison will truncate')
    ap.add_argument('--only', default=None,
                    help='comma-separated subset of: tail_divergence,'
                         'xi_convergence,ranking_recovery,leverage_enrichment')
    args = ap.parse_args(argv)
    args.formats = [f.strip() for f in args.formats.split(',') if f.strip()]

    os.makedirs(args.outdir, exist_ok=True)

    print("loading:")
    runs = []
    for i, spec in enumerate(args.runs):
        d, label, df = parse_run_spec(spec)
        run = load_run(d, label, df, require_precise=not args.allow_float32)
        run['colour'] = PALETTE[i % len(PALETTE)]
        runs.append(run)

    if args.bin_min is None:
        joined = any(r['source'] == 'joined' for r in runs)
        args.bin_min = 4.0 if joined else 1.0
        if joined:
            print("\n--bin-min not given; using 4.0. The joined catalog is\n"
                  "p-thresholded, so coverage below |t|~4 is a biased remnant.\n"
                  "For the bulk, plot the un-joined output with --allow-float32.")
        else:
            print("\n--bin-min not given; using 1.0")

    if args.bin_max is None:
        cutoffs = [float(r['t'][(r['pa'] > 0) & (r['pp'] > 0)].max())
                   for r in runs if ((r['pa'] > 0) & (r['pp'] > 0)).any()]
        args.bin_max = (max(cutoffs) + args.bin_step) if cutoffs else 8.0
        print("\n--bin-max not given; using {:.2f}".format(args.bin_max))

    want = ({s.strip() for s in args.only.split(',')} if args.only else
            {'tail_divergence', 'xi_convergence', 'ranking_recovery',
             'leverage_enrichment'})

    print("\nfigures:")
    if 'tail_divergence' in want:
        fig_tail_divergence(runs, args, args.outdir)
    if 'xi_convergence' in want:
        fig_xi_convergence(runs, args, args.outdir)
    if 'ranking_recovery' in want:
        fig_ranking_recovery(runs, args, args.outdir)
    if 'leverage_enrichment' in want:
        fig_leverage_enrichment(runs, args, args.outdir)
    print("\nDone. PDF output is vector; prefer it for print.")


if __name__ == '__main__':
    main()
