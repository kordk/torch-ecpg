#!/usr/bin/env python3
"""ancestry_probes_report.py — evaluate methylation-derived ancestry instruments.

Standalone exploratory tool. NOT wired into any pipeline stage: it reads
M_orig.csv (and optionally a covariate file and a probe blacklist), runs two to
four candidate ancestry-axis constructions side by side, and renders a
FastQC-style HTML report plus a JSON sidecar.

The question it exists to answer: the mapping model currently carries Meth_PC1-5,
which are PCs of ALL probes and therefore encode ancestry, batch, cell
composition and any broad methylation module indiscriminately. Genotype-like
probes (the rs control probes, and CpG probes whose beta distribution is
trimodal because a SNP sits under the interrogated base) give an ancestry axis
that cannot encode methylation biology, because the per-probe value used is a
discrete cluster index rather than a methylation level. This report characterises
how those instruments compare, so a covariate choice can be made from evidence.

Methods evaluated (each yields sample x k score matrices):

  A  rs         65 (450K) / 59 (EPIC) rs* control probes. Genotype calls.
                Zero methylation leakage, low resolution.
  B  gap        Data-driven trimodal/gapped CpG probes (gaphunter-style).
                Genotype calls. Near-zero leakage, higher resolution.
  C  blacklist  PCA restricted to probes on a supplied blacklist (SNP-affected
                and cross-reactive). Betas, not calls; runs only with --blacklist.
  D  allcpg     PCA over a probe subsample. The status quo comparator; this is
                what Meth_PC currently is.

Nothing here is prescriptive. Every module reports what was observed; the
verdict badges flag only conditions that would make a number unreliable
(too few probes, degenerate calls), never "use this method".

Usage:
  python3 tools/ancestry_probes_report.py \
      --dataset mesa \
      --methylation data_mesa/M_orig.csv \
      --covariates data_mesa/C_orig.csv \
      --group-column racegendersite \
      --blacklist data_mesa/probes_blacklist.csv \
      --out output_mesa/ancestry_probes_report.html \
      --json output_mesa/ancestry_probes.json \
      --scores-out output_mesa/ancestry_scores.csv
"""
import argparse
import datetime
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from permute_qc_report import (  # noqa: E402
    QCModule,
    fig_to_base64,
    render_html,
    render_table,
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---- thresholds -----------------------------------------------------------
# Documented defaults, not tuned to any cohort. Every one is a CLI flag.
GAP_THRESHOLD = 0.10        # min beta gap between clusters to call a probe gapped
MIN_GROUP_FRAC = 0.01       # min fraction of samples on each side of a gap
MIN_GROUP_ABS = 5           # ...and at least this many samples
RS_MIN_PROBES = 20          # fewer rs probes than this: axis is unreliable
GAP_MIN_PROBES = 200        # fewer gapped probes than this: axis is unreliable
LEAK_HIGH = 0.70            # |r| with an all-CpG PC at/above this: heavy overlap
LEAK_MOD = 0.40
N_PC = 5                    # PCs retained per method
SUBSAMPLE_PROBES = 50_000   # probes sampled for the all-CpG comparator
CHUNK_ROWS = 20_000

COL_A = '#3b6ea5'
COL_B = '#2e7d32'
COL_C = '#b8860b'
COL_D = '#c2452d'
METHOD_COLOURS = {'rs': COL_A, 'gap': COL_B, 'blacklist': COL_C, 'allcpg': COL_D}
METHOD_LABELS = {
    'rs': 'A. rs control probes',
    'gap': 'B. Gapped CpG probes',
    'blacklist': 'C. Blacklist probes',
    'allcpg': 'D. All-CpG PCs (status quo)',
}


# ---- input handling -------------------------------------------------------

def looks_like_mvalues(sample_block: np.ndarray) -> bool:
    """M-values are unbounded and routinely negative; betas live in [0, 1]."""
    finite = sample_block[np.isfinite(sample_block)]
    if finite.size == 0:
        return False
    return bool(np.nanmin(finite) < -0.01 or np.nanmax(finite) > 1.01)


def m_to_beta(x: np.ndarray) -> np.ndarray:
    """beta = 2^M / (2^M + 1), computed stably."""
    return 1.0 / (1.0 + np.power(2.0, -x))


def beta_to_m(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """M = log2(beta / (1 - beta)), clipped so 0 and 1 do not produce infinities."""
    b = np.clip(x, eps, 1.0 - eps)
    return np.log2(b / (1.0 - b))


def find_gaps(sorted_row: np.ndarray, gap_threshold: float, min_group: int):
    """Return the cut values splitting one probe's sorted betas into clusters.

    A gap qualifies when the jump between adjacent sorted values is at least
    gap_threshold and both sides carry at least min_group samples. The cut is
    placed at the midpoint of the gap.
    """
    n = sorted_row.size
    if n < 2 * min_group:
        return []
    diffs = np.diff(sorted_row)
    idx = np.nonzero(diffs >= gap_threshold)[0]
    cuts = []
    for i in idx:
        left = i + 1
        if left >= min_group and (n - left) >= min_group:
            cuts.append(0.5 * (sorted_row[i] + sorted_row[i + 1]))
    return cuts


def scan_methylation(path, gap_threshold, min_group_frac, min_group_abs,
                     subsample_probes, chunk_rows, seed,
                     exclude_ids=None, pca_scale='beta'):
    """Single pass over the methylation matrix.

    Returns rs-probe betas, gapped-probe betas, gap cut points, a probe
    subsample for the all-CpG comparator, and scan counters. Reads in chunks so
    a 450K x 1200 matrix never lands in memory at once.

    Gap and rs detection always operate on beta, since cluster separation is
    defined on the bounded scale. The PCA comparator uses pca_scale, so it can
    be matched to whatever tools/residualize_pca.py consumed: an M-value
    comparator is the like-for-like reference for the Meth_PC covariates.

    exclude_ids removes probes from every method before any of them run, so an
    exclusion (chrX, a blacklist) is applied consistently across instruments.
    """
    exclude_ids = exclude_ids or set()
    rng = np.random.default_rng(seed)
    rs_rows, rs_ids = [], []
    gap_rows, gap_ids, gap_cuts = [], [], []
    sub_rows, sub_ids = [], []
    n_probes = 0
    n_dropped_na = 0
    n_excluded = 0
    scale_is_m = None
    columns = None
    seen_for_subsample = 0

    reader = pd.read_csv(path, index_col=0, chunksize=chunk_rows)
    for chunk in reader:
        if columns is None:
            columns = [str(c) for c in chunk.columns]
        raw = chunk.to_numpy(dtype=np.float64, na_value=np.nan)
        if scale_is_m is None:
            scale_is_m = looks_like_mvalues(raw)
            logger.info("Detected input scale: %s",
                        "M-values (converting to beta)" if scale_is_m else "beta")

        # beta drives cluster detection; pca_values drives the PCA comparator.
        values = m_to_beta(raw) if scale_is_m else raw
        if pca_scale == 'mvalue':
            pca_values = raw if scale_is_m else beta_to_m(raw)
        else:
            pca_values = values

        n_probes += values.shape[0]
        ids = np.asarray([str(i) for i in chunk.index])

        if exclude_ids:
            excl = np.asarray([i in exclude_ids for i in ids])
            n_excluded += int(excl.sum())
        else:
            excl = np.zeros(ids.shape[0], dtype=bool)

        complete = np.isfinite(values).all(axis=1) & ~excl
        n_dropped_na += int((~np.isfinite(values).all(axis=1)).sum())

        is_rs = np.char.startswith(np.char.lower(ids.astype(str)), 'rs')

        min_group = max(min_group_abs,
                        int(np.ceil(min_group_frac * values.shape[1])))

        # rs control probes: kept whether or not they are gapped.
        keep_rs = is_rs & complete
        if keep_rs.any():
            rs_rows.append(values[keep_rs])
            rs_ids.extend(ids[keep_rs].tolist())

        # Gapped CpG probes: cg* only, so the rs axis and the gap axis stay
        # independent instruments rather than sharing probes.
        cand = complete & ~is_rs
        cand_idx = np.nonzero(cand)[0]
        if cand_idx.size:
            block = values[cand_idx]
            srt = np.sort(block, axis=1)
            diffs = np.diff(srt, axis=1)
            # cheap prefilter: any gap at all above threshold
            has_big = (diffs >= gap_threshold).any(axis=1)
            for local in np.nonzero(has_big)[0]:
                cuts = find_gaps(srt[local], gap_threshold, min_group)
                if cuts:
                    gap_rows.append(block[local])
                    gap_ids.append(ids[cand_idx[local]])
                    gap_cuts.append(cuts)

        # Reservoir subsample for the all-CpG comparator.
        comp_idx = np.nonzero(complete)[0]
        for local in comp_idx:
            seen_for_subsample += 1
            if len(sub_rows) < subsample_probes:
                sub_rows.append(pca_values[local])
                sub_ids.append(ids[local])
            else:
                j = rng.integers(0, seen_for_subsample)
                if j < subsample_probes:
                    sub_rows[j] = pca_values[local]
                    sub_ids[j] = ids[local]

    rs_mat = np.vstack(rs_rows) if rs_rows else np.empty((0, 0))
    gap_mat = np.vstack([r[None, :] for r in gap_rows]) if gap_rows else np.empty((0, 0))
    sub_mat = np.vstack([r[None, :] for r in sub_rows]) if sub_rows else np.empty((0, 0))

    return {
        'columns': columns or [],
        'rs': (rs_mat, rs_ids),
        'gap': (gap_mat, gap_ids, gap_cuts),
        'subsample': (sub_mat, sub_ids),
        'n_probes': n_probes,
        'n_dropped_na': n_dropped_na,
        'n_excluded': n_excluded,
        'scale_is_m': bool(scale_is_m),
        'pca_scale': pca_scale,
        'n_complete_seen': seen_for_subsample,
    }


# ---- genotype calling and PCA --------------------------------------------

def call_dosage(mat, cuts_list=None, gap_threshold=GAP_THRESHOLD,
                min_group=MIN_GROUP_ABS):
    """Convert beta rows to integer cluster indices (0, 1, 2, ...).

    With cuts supplied (gapped probes) the cuts are reused. Without them (rs
    probes) fixed genotype thresholds at 0.25 / 0.75 are applied, which is the
    conventional call for trimodal control probes.
    """
    n_probes, n_samples = mat.shape
    dosage = np.empty((n_probes, n_samples), dtype=np.float64)
    n_clusters = np.empty(n_probes, dtype=int)
    for i in range(n_probes):
        if cuts_list is not None:
            cuts = cuts_list[i]
        else:
            cuts = [0.25, 0.75]
        d = np.digitize(mat[i], np.asarray(cuts, dtype=float))
        dosage[i] = d
        n_clusters[i] = int(np.unique(d).size)
    return dosage, n_clusters


def standardize_rows(mat, min_sd=1e-8):
    """Center and scale each probe; drop probes with no variance."""
    mu = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, ddof=0, keepdims=True)
    keep = (sd.ravel() > min_sd)
    if not keep.any():
        return np.empty((0, mat.shape[1])), keep
    return (mat[keep] - mu[keep]) / sd[keep], keep


def pca_scores(mat_probes_by_samples, n_pc=N_PC):
    """PCA over samples. Input is probes x samples, already standardized.

    Returns (scores samples x k, explained variance ratio for k components).
    Uses the SVD of the sample covariance implied by the probe matrix, which is
    the standard genotype-PCA construction.
    """
    if mat_probes_by_samples.size == 0:
        return np.empty((0, 0)), np.empty(0)
    X = mat_probes_by_samples.T          # samples x probes
    X = X - X.mean(axis=0, keepdims=True)
    k = int(min(n_pc, min(X.shape) - 1))
    if k < 1:
        return np.empty((X.shape[0], 0)), np.empty(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    total = float((S ** 2).sum())
    scores = U[:, :k] * S[:k]
    evr = (S[:k] ** 2) / total if total > 0 else np.zeros(k)
    return scores, evr


def abs_corr_matrix(A, B):
    """|Pearson r| between every column of A and every column of B."""
    if A.size == 0 or B.size == 0:
        return np.empty((0, 0))
    out = np.zeros((A.shape[1], B.shape[1]))
    for i in range(A.shape[1]):
        for j in range(B.shape[1]):
            a, b = A[:, i], B[:, j]
            if a.std() < 1e-12 or b.std() < 1e-12:
                out[i, j] = np.nan
            else:
                out[i, j] = abs(float(np.corrcoef(a, b)[0, 1]))
    return out


def subspace_overlap(A, B):
    """Mean squared canonical correlation between two score matrices.

    1.0 means the two methods span the same sample-space directions; 0.0 means
    they are orthogonal. Reported as a single number so methods can be compared
    without reading a full correlation grid.
    """
    if A.size == 0 or B.size == 0:
        return float('nan')
    Qa, _ = np.linalg.qr(A - A.mean(axis=0, keepdims=True))
    Qb, _ = np.linalg.qr(B - B.mean(axis=0, keepdims=True))
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float((s ** 2).mean())


def eta_squared(scores, groups):
    """Fraction of each PC's variance explained by a categorical grouping."""
    out = []
    g = np.asarray(groups)
    for j in range(scores.shape[1]):
        y = scores[:, j]
        grand = y.mean()
        ss_total = float(((y - grand) ** 2).sum())
        if ss_total <= 0:
            out.append(float('nan'))
            continue
        ss_between = 0.0
        for level in np.unique(g):
            sel = (g == level)
            if sel.sum() == 0:
                continue
            ss_between += sel.sum() * (y[sel].mean() - grand) ** 2
        out.append(float(ss_between / ss_total))
    return out


# ---- report modules -------------------------------------------------------

def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'n/a'
    return f'{x:.{nd}f}'


def build_inputs_module(scan, args, methods):
    rows = [
        ['Methylation matrix', os.path.basename(args.methylation)],
        ['Probes read', f"{scan['n_probes']:,}"],
        ['Samples', f"{len(scan['columns']):,}"],
        ['Probes with missing values (excluded)', f"{scan['n_dropped_na']:,}"],
        ['Input scale', 'M-values, converted to beta' if scan['scale_is_m'] else 'beta'],
        ['PCA comparator scale', args.pca_scale],
        ['Probes excluded by --exclude-probes', f"{scan.get('n_excluded', 0):,}"],
        ['Gap threshold', f"{args.gap_threshold}"],
        ['Min samples per cluster', f"{args.min_group_abs} or {args.min_group_frac:.1%} of samples"],
        ['PCs retained per method', f"{args.n_pc}"],
        ['Methods evaluated', ', '.join(METHOD_LABELS[m] for m in methods)],
    ]
    if args.covariates:
        rows.append(['Covariates', os.path.basename(args.covariates)])
    if args.blacklist:
        rows.append(['Blacklist', os.path.basename(args.blacklist)])

    return QCModule(
        anchor='inputs',
        title='Inputs and scan',
        status='INFO',
        purpose=('Records what was read and how it was interpreted, so a report can be '
                 'traced back to a specific matrix and parameter set. The scale '
                 'detection matters: gap detection operates on beta values, so an '
                 'M-value input is converted before anything else happens.'),
        interpretation=(
            f"Read {scan['n_probes']:,} probes across {len(scan['columns']):,} samples. "
            f"{scan['n_dropped_na']:,} probes had at least one missing value and were "
            'excluded from every method, so all methods below run on the same sample set. '
            'This tool reads M_orig.csv deliberately: the probes it needs are the ones '
            'the blacklist stage removes, so it wants the pre-filter matrix.'),
        table_html=render_table(['Item', 'Value'], rows),
    )


def build_probe_module(name, n_found, n_used, extra_rows, status, purpose,
                       interpretation, figure_b64='', figure_alt=''):
    rows = [['Probes found', f'{n_found:,}'], ['Probes used (variable)', f'{n_used:,}']]
    rows.extend(extra_rows)
    return QCModule(
        anchor=f'method-{name}',
        title=METHOD_LABELS[name],
        status=status,
        purpose=purpose,
        interpretation=interpretation,
        table_html=render_table(['Item', 'Value'], rows),
        figure_b64=figure_b64,
        figure_alt=figure_alt,
    )


def scree_figure(results, methods):
    if not HAVE_MPL:
        return '', ''
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in methods:
        evr = results[m].get('evr')
        if evr is None or len(evr) == 0:
            continue
        ax.plot(range(1, len(evr) + 1), 100 * np.asarray(evr), marker='o',
                color=METHOD_COLOURS[m], label=METHOD_LABELS[m])
    ax.set_xlabel('Component')
    ax.set_ylabel('Variance explained (%)')
    ax.set_title('Scree by method')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig_to_base64(fig), 'Variance explained per component for each method'


def scatter_figure(results, methods, groups=None, group_name=''):
    if not HAVE_MPL:
        return '', ''
    usable = [m for m in methods if results[m].get('scores') is not None
              and results[m]['scores'].shape[1] >= 2]
    if not usable:
        return '', ''
    fig, axes = plt.subplots(1, len(usable), figsize=(4.2 * len(usable), 4),
                             squeeze=False)
    for ax, m in zip(axes[0], usable):
        s = results[m]['scores']
        if groups is not None:
            levels = np.unique(groups)
            cmap = plt.get_cmap('tab20')
            for i, lv in enumerate(levels):
                sel = (groups == lv)
                ax.scatter(s[sel, 0], s[sel, 1], s=8, alpha=0.7,
                           color=cmap(i % 20), label=str(lv))
            if len(levels) <= 12:
                ax.legend(fontsize=6, title=group_name, title_fontsize=6)
        else:
            ax.scatter(s[:, 0], s[:, 1], s=8, alpha=0.7,
                       color=METHOD_COLOURS[m])
        ax.set_title(METHOD_LABELS[m], fontsize=9)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    alt = 'PC1 vs PC2 per method'
    if groups is not None:
        alt += f', coloured by {group_name}'
    return fig_to_base64(fig), alt


def build_overlap_module(results, methods):
    rows = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            sa, sb = results[a].get('scores'), results[b].get('scores')
            if sa is None or sb is None or sa.size == 0 or sb.size == 0:
                continue
            ov = subspace_overlap(sa, sb)
            cm = abs_corr_matrix(sa[:, :1], sb[:, :1])
            r11 = cm[0, 0] if cm.size else float('nan')
            rows.append([METHOD_LABELS[a], METHOD_LABELS[b], fmt(ov), fmt(r11)])

    if not rows:
        return QCModule(
            anchor='overlap', title='Cross-method agreement', status='INFO',
            purpose='Compares the sample-space directions recovered by each method.',
            interpretation='Not evaluated: fewer than two methods produced scores.')

    return QCModule(
        anchor='overlap',
        title='Cross-method agreement',
        status='INFO',
        purpose=('Two instruments that recover the same population structure should span '
                 'the same directions in sample space, whatever their per-probe inputs. '
                 'Subspace overlap is the mean squared canonical correlation between two '
                 "methods' retained components: 1.0 means the same subspace, 0.0 "
                 'orthogonal.'),
        interpretation=(
            'High overlap between the genotype-based methods (A, B) and the all-CpG PCs '
            '(D) would indicate the current Meth_PCs are largely tracking ancestry, in '
            'which case a genotype-based axis carries the same information without the '
            'capacity to encode methylation biology. Low overlap indicates the all-CpG '
            'PCs are dominated by something else (batch, composition, broad methylation '
            'modules), which is the case where swapping instruments changes the model.'),
        table_html=render_table(
            ['Method', 'Method', 'Subspace overlap', '|r| PC1 vs PC1'], rows,
            aligns=['left', 'left', 'right', 'right']),
    )


def build_leakage_module(results, methods):
    if 'allcpg' not in results or results['allcpg'].get('scores') is None:
        return QCModule(
            anchor='leakage', title='Methylation leakage', status='INFO',
            purpose='Quantifies how much each candidate axis reproduces the all-CpG PCs.',
            interpretation='Not evaluated: the all-CpG comparator produced no scores.')

    ref = results['allcpg']['scores']
    rows, worst = [], 0.0
    for m in methods:
        if m == 'allcpg':
            continue
        s = results[m].get('scores')
        if s is None or s.size == 0:
            continue
        cm = abs_corr_matrix(s, ref)
        mx = float(np.nanmax(cm)) if cm.size else float('nan')
        if np.isfinite(mx):
            worst = max(worst, mx)
        loc = np.unravel_index(np.nanargmax(cm), cm.shape) if cm.size else (0, 0)
        rows.append([METHOD_LABELS[m], fmt(mx),
                     f'PC{loc[0] + 1} vs all-CpG PC{loc[1] + 1}',
                     fmt(float(np.nanmean(cm)))])

    status = 'INFO'
    if worst >= LEAK_HIGH:
        status = 'WARN'

    return QCModule(
        anchor='leakage',
        title='Methylation leakage',
        status=status,
        purpose=('The concern motivating this report is that an ancestry proxy built from '
                 'methylation can carry methylation biology into the covariate set, where '
                 'it would be removed from every CpG in the mapping model. Methods A and '
                 'B reduce each probe to a discrete cluster index, which discards the '
                 'within-cluster methylation variation that biology would live in. This '
                 'module measures how much of the all-CpG structure each method still '
                 'reproduces.'),
        interpretation=(
            f'Highest |r| between any candidate component and any all-CpG component is '
            f'{fmt(worst)}. A high value is not by itself evidence of leakage: if '
            'ancestry is the dominant axis of the methylation matrix, an ancestry '
            'instrument and the all-CpG PCs will agree precisely because both are '
            'measuring ancestry. Read this against the cross-method agreement above and '
            'the group separation below. The interpretable case for leakage is a '
            'genotype-based axis correlating with an all-CpG component that does NOT '
            'separate the ancestry grouping.'),
        table_html=render_table(
            ['Method', 'Max |r| with all-CpG PCs', 'Where', 'Mean |r|'], rows,
            aligns=['left', 'right', 'left', 'right']),
    )


def build_group_module(results, methods, groups, group_name, n_pc):
    if groups is None:
        return QCModule(
            anchor='group', title='Group separation', status='INFO',
            purpose='Checks whether each axis separates a known grouping.',
            interpretation=('Not evaluated: no --group-column supplied. With a '
                            'self-reported race or composite demographic column, this '
                            'module reports how much of each component is explained by '
                            'that grouping, which is the closest available check that an '
                            'axis is tracking ancestry rather than something else.'))

    headers = ['Method'] + [f'PC{i + 1}' for i in range(n_pc)] + ['Max']
    rows = []
    for m in methods:
        s = results[m].get('scores')
        if s is None or s.size == 0:
            continue
        eta = eta_squared(s, groups)
        padded = eta + [float('nan')] * (n_pc - len(eta))
        mx = float(np.nanmax(eta)) if eta else float('nan')
        rows.append([METHOD_LABELS[m]] + [fmt(e) for e in padded[:n_pc]] + [fmt(mx)])

    n_levels = int(np.unique(groups).size)
    return QCModule(
        anchor='group',
        title='Group separation',
        status='INFO',
        purpose=(f'Reports eta-squared: the fraction of each component\'s variance '
                 f'explained by {group_name} ({n_levels} levels). An axis that tracks '
                 'ancestry should separate a self-reported demographic grouping, without '
                 'being reducible to it, since genotype-derived axes also carry '
                 'continuous within-group admixture that self-report cannot.'),
        interpretation=(
            'Compare methods on the maximum column. A genotype-based axis (A, B) with '
            'separation comparable to the all-CpG PCs (D) is recovering the same '
            'population structure from probes that cannot encode methylation biology. '
            'Values near 1.0 on any method indicate the axis is close to a relabelling '
            'of the grouping itself, which adds little beyond including the grouping as '
            'a covariate directly. Intermediate values are the informative case: '
            'structure beyond the self-reported labels.'),
        table_html=render_table(headers, rows,
                                aligns=['left'] + ['right'] * (n_pc + 1)),
    )


def build_covariate_module(results, methods, covar_df, n_pc):
    if covar_df is None or covar_df.shape[1] == 0:
        return QCModule(
            anchor='covariates', title='Correlation with known covariates',
            status='INFO',
            purpose='Checks each axis against the covariates already in the model.',
            interpretation='Not evaluated: no --covariates supplied.')

    numeric = covar_df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return QCModule(
            anchor='covariates', title='Correlation with known covariates',
            status='INFO',
            purpose='Checks each axis against the covariates already in the model.',
            interpretation='Not evaluated: no numeric covariate columns found.')

    headers = ['Method', 'Covariate', 'Max |r| over PCs', 'Component']
    rows = []
    for m in methods:
        s = results[m].get('scores')
        if s is None or s.size == 0:
            continue
        for col in numeric.columns:
            v = numeric[col].to_numpy(dtype=float)[:, None]
            cm = abs_corr_matrix(s, v)
            if cm.size == 0 or not np.isfinite(cm).any():
                continue
            mx = float(np.nanmax(cm))
            which = int(np.nanargmax(cm[:, 0])) + 1
            rows.append([METHOD_LABELS[m], str(col), fmt(mx), f'PC{which}'])

    return QCModule(
        anchor='covariates',
        title='Correlation with known covariates',
        status='INFO',
        purpose=('An ancestry axis that is strongly correlated with a covariate already '
                 'in the model is partly redundant with it. This is context for the '
                 'covariate ladder rather than a pass/fail: some overlap is expected '
                 'wherever demographic composition differs across the cohort.'),
        interpretation=(
            'Read the largest values first. A candidate axis correlating with age or a '
            'technical covariate rather than with the demographic grouping would suggest '
            'the probes selected are not behaving as genotypes. Correlation with a '
            'demographic covariate is expected and is not a problem.'),
        table_html=render_table(headers, rows,
                                aligns=['left', 'left', 'right', 'left']),
    )


# ---- main -----------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, help='Label for the report header')
    p.add_argument('--methylation', required=True,
                   help='M_orig.csv: probes in rows, samples in columns')
    p.add_argument('--covariates', default=None,
                   help='Covariate CSV, samples in rows (e.g. C_orig.csv)')
    p.add_argument('--group-column', default=None,
                   help='Categorical covariate column for the separation module')
    p.add_argument('--blacklist', default=None,
                   help='probes_blacklist.csv; enables method C')
    p.add_argument('--out', required=True, help='Output HTML path')
    p.add_argument('--json', default=None, help='Optional JSON sidecar path')
    p.add_argument('--scores-out', default=None,
                   help='Optional CSV of all component scores, samples in rows')
    p.add_argument('--probes-out', default=None,
                   help='Optional TSV of the probes selected by methods A and B, '
                        'with cluster count and allele frequency. Join against '
                        'annot_<dataset>/M.bed6 to get chromosomes.')
    p.add_argument('--exclude-probes', default=None,
                   help='Optional file of probe IDs to drop from every method '
                        '(one per line, or a CSV whose first column is the ID). '
                        'Use to test a chrX or blacklist exclusion.')
    p.add_argument('--pca-scale', choices=('beta', 'mvalue'), default='beta',
                   help='Scale for the PCA comparator (methods C and D). Cluster '
                        'detection always uses beta. Set mvalue to match '
                        'tools/residualize_pca.py, which computes Meth_PC on '
                        'M-values; the leading component differs by scale.')
    p.add_argument('--n-pc', type=int, default=N_PC)
    p.add_argument('--gap-threshold', type=float, default=GAP_THRESHOLD)
    p.add_argument('--min-group-frac', type=float, default=MIN_GROUP_FRAC)
    p.add_argument('--min-group-abs', type=int, default=MIN_GROUP_ABS)
    p.add_argument('--subsample-probes', type=int, default=SUBSAMPLE_PROBES)
    p.add_argument('--chunk-rows', type=int, default=CHUNK_ROWS)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not os.path.exists(args.methylation):
        logger.error('Methylation matrix not found: %s', args.methylation)
        return 2

    exclude_ids = set()
    if args.exclude_probes:
        if not os.path.exists(args.exclude_probes):
            logger.error('Exclusion list not found: %s', args.exclude_probes)
            return 2
        with open(args.exclude_probes) as fh:
            for line in fh:
                tok = line.strip().split(',')[0].strip().strip('"')
                if tok and tok.lower() not in ('probe_id', 'probe', 'id', ''):
                    exclude_ids.add(tok)
        logger.info('Loaded %d probe IDs to exclude', len(exclude_ids))

    logger.info('Scanning %s', args.methylation)
    scan = scan_methylation(args.methylation, args.gap_threshold,
                            args.min_group_frac, args.min_group_abs,
                            args.subsample_probes, args.chunk_rows, args.seed,
                            exclude_ids=exclude_ids, pca_scale=args.pca_scale)
    samples = scan['columns']
    if not samples:
        logger.error('No samples read from %s', args.methylation)
        return 2

    results = {}
    methods = []
    gap_meta = None
    rs_meta = None

    # --- Method A: rs control probes
    rs_mat, rs_ids = scan['rs']
    a_found = rs_mat.shape[0] if rs_mat.size else 0
    if a_found:
        dos, ncl = call_dosage(rs_mat)
        rs_meta = {'ids': rs_ids, 'n_clusters': ncl, 'dosage_mean': dos.mean(axis=1)}
        std, keep = standardize_rows(dos)
        scores, evr = pca_scores(std, args.n_pc)
        results['rs'] = {'scores': scores, 'evr': evr, 'n_found': a_found,
                         'n_used': int(keep.sum()),
                         'mean_clusters': float(ncl.mean())}
        methods.append('rs')
    else:
        results['rs'] = {'scores': None, 'evr': None, 'n_found': 0, 'n_used': 0}

    # --- Method B: gapped CpG probes
    gap_mat, gap_ids, gap_cuts = scan['gap']
    b_found = gap_mat.shape[0] if gap_mat.size else 0
    if b_found:
        dos, ncl = call_dosage(gap_mat, gap_cuts)
        std, keep = standardize_rows(dos)
        scores, evr = pca_scores(std, args.n_pc)
        # Allele frequency is only defined where three clusters were resolved
        # (the AA / AB / BB pattern of a biallelic SNP). For a two-cluster
        # probe, mean(dosage)/2 is the split between the two groups, which for
        # an X-inactivation probe is the sex ratio, not a MAF.
        tri = (ncl >= 3)
        af_tri = (np.clip(dos[tri].mean(axis=1) / 2.0, 0, 1)
                  if tri.any() else np.empty(0))
        results['gap'] = {'scores': scores, 'evr': evr, 'n_found': b_found,
                          'n_used': int(keep.sum()),
                          'mean_clusters': float(ncl.mean()),
                          'frac_trimodal': float(tri.mean()),
                          'n_trimodal': int(tri.sum()),
                          'n_bimodal': int((ncl == 2).sum()),
                          'median_af': (float(np.median(af_tri))
                                        if af_tri.size else float('nan'))}
        gap_meta = {'ids': gap_ids, 'n_clusters': ncl, 'dosage_mean': dos.mean(axis=1)}
        methods.append('gap')
    else:
        results['gap'] = {'scores': None, 'evr': None, 'n_found': 0, 'n_used': 0}

    # --- Method C: blacklist-restricted betas
    sub_mat, sub_ids = scan['subsample']
    if args.blacklist and os.path.exists(args.blacklist):
        bl = pd.read_csv(args.blacklist)
        bl_ids = set(bl.iloc[:, 0].astype(str))
        sel = np.asarray([i in bl_ids for i in sub_ids])
        c_found = int(sel.sum())
        if c_found:
            std, keep = standardize_rows(sub_mat[sel])
            scores, evr = pca_scores(std, args.n_pc)
            results['blacklist'] = {'scores': scores, 'evr': evr,
                                    'n_found': c_found, 'n_used': int(keep.sum()),
                                    'blacklist_size': len(bl_ids)}
            methods.append('blacklist')
        else:
            results['blacklist'] = {'scores': None, 'evr': None, 'n_found': 0,
                                    'n_used': 0, 'blacklist_size': len(bl_ids)}
    else:
        results['blacklist'] = {'scores': None, 'evr': None, 'n_found': 0, 'n_used': 0}

    # --- Method D: all-CpG comparator
    d_found = sub_mat.shape[0] if sub_mat.size else 0
    if d_found:
        std, keep = standardize_rows(sub_mat)
        scores, evr = pca_scores(std, args.n_pc)
        results['allcpg'] = {'scores': scores, 'evr': evr, 'n_found': d_found,
                             'n_used': int(keep.sum())}
        methods.append('allcpg')
    else:
        results['allcpg'] = {'scores': None, 'evr': None, 'n_found': 0, 'n_used': 0}

    # --- covariates and grouping
    covar_df, groups = None, None
    if args.covariates and os.path.exists(args.covariates):
        cov = pd.read_csv(args.covariates, index_col=0)
        cov.index = cov.index.astype(str)
        missing = [s for s in samples if s not in cov.index]
        if missing:
            logger.warning('%d methylation samples absent from covariates; '
                           'covariate modules skipped', len(missing))
        else:
            covar_df = cov.loc[samples]
            if args.group_column and args.group_column in covar_df.columns:
                groups = covar_df[args.group_column].to_numpy()
            elif args.group_column:
                logger.warning('Group column %s not found in covariates',
                               args.group_column)

    # --- modules
    modules = [build_inputs_module(scan, args, methods)]

    r = results['rs']
    modules.append(build_probe_module(
        'rs', r['n_found'], r['n_used'],
        [['Mean clusters per probe', fmt(r.get('mean_clusters'), 2)]],
        'PASS' if r['n_found'] >= RS_MIN_PROBES else 'INFO',
        ('The rs* probes are SNP genotyping assays carried on the array for sample '
         'identity checks. Their values are genotype calls, not methylation levels, so '
         'this axis cannot encode methylation biology at all. The 450K carries 65 and '
         'EPIC 59, which is enough to separate continental ancestry but too few to '
         'resolve within-group admixture.'),
        (f"{r['n_found']:,} rs probes were present with complete data. "
         + ('This is the expected complement for a 450K or EPIC matrix, and the axis '
            'below is usable as a coarse ancestry reference.'
            if r['n_found'] >= RS_MIN_PROBES else
            'This is below the number needed for a stable axis; the submitter likely '
            'stripped the control probes from the processed matrix, and the components '
            'here should be read as unreliable.'))))

    r = results['gap']
    frac_tri = r.get('frac_trimodal')
    cluster_note = ''
    if frac_tri is not None and np.isfinite(frac_tri) and frac_tri < 0.5:
        cluster_note = (
            ' Most selected probes resolved into TWO clusters rather than three. '
            'A biallelic SNP common enough to be detected at this sample size '
            'normally yields three clusters, so a bimodal majority points at a '
            'different mechanism — X-inactivation is the usual one, since chrX '
            'probes separate females (intermediate) from males (extreme) in two '
            'groups. Export the probe list with --probes-out and join it against '
            'annot_<dataset>/M.bed6 to check for chrX enrichment; if confirmed, '
            're-run with --exclude-probes to see which components survive.')
    modules.append(build_probe_module(
        'gap', r['n_found'], r['n_used'],
        [['Mean clusters per probe', fmt(r.get('mean_clusters'), 2)],
         ['Probes with 3+ clusters', f"{r.get('n_trimodal', 0):,}"],
         ['Probes with 2 clusters', f"{r.get('n_bimodal', 0):,}"],
         ['Fraction with 3+ clusters', fmt(frac_tri)],
         ['Median allele frequency (3-cluster probes only)',
          fmt(r.get('median_af'))]],
        'PASS' if (r['n_found'] >= GAP_MIN_PROBES
                   and frac_tri is not None and frac_tri >= 0.5) else 'WARN',
        ('CpG probes with a common SNP under the interrogated base produce a trimodal '
         'beta distribution: the value reports genotype rather than methylation. This '
         'module finds them from the data, with no reference annotation, by looking for '
         'gaps in each probe\'s sorted beta values. Each probe is then reduced to a '
         'cluster index, which discards the within-cluster methylation variation.'),
        (f"{r['n_found']:,} probes showed at least one qualifying gap, of which "
         f"{fmt(frac_tri)} had three or more clusters — the signature of a "
         'biallelic SNP. Median allele frequency near 0.2-0.4 is consistent with common '
         'variants. A count far below the low hundreds would suggest the gap threshold '
         'is too strict for this matrix, or that the normalisation compressed the '
         'clusters; a count in the tens of thousands would suggest it is too loose and '
         'ordinary variable CpGs are being admitted.' + cluster_note)))

    if 'blacklist' in methods or args.blacklist:
        r = results['blacklist']
        modules.append(build_probe_module(
            'blacklist', r['n_found'], r['n_used'],
            [['Blacklist size', f"{r.get('blacklist_size', 0):,}"]],
            'INFO',
            ('The probe blacklist already identifies SNP-affected and cross-reactive '
             'probes. Restricting PCA to those probes is a reference-based version of '
             'method B. Unlike A and B this uses beta values rather than cluster '
             'indices, so it retains methylation variation and sits between the '
             'genotype-based axes and the all-CpG PCs on the leakage question.'),
            (f"{r['n_found']:,} of the subsampled probes are on the blacklist. Note this "
             'is drawn from the probe subsample, not the full matrix, so the count '
             'scales with --subsample-probes.')))

    r = results['allcpg']
    modules.append(build_probe_module(
        'allcpg', r['n_found'], r['n_used'], [],
        'INFO',
        ('The status quo comparator: PCA over a probe subsample, which is what the '
         'Meth_PC covariates in the mapping model currently are. Included so the '
         'candidate instruments can be read against the thing they would replace.'),
        (f"PCA over {r['n_found']:,} subsampled probes. This is the axis that captures "
         'ancestry, batch, cell composition and any broad methylation module at once — '
         'the behaviour that motivates looking for a narrower instrument.')))

    fig_b64, fig_alt = scree_figure(results, methods)
    modules.append(QCModule(
        anchor='scree', title='Scree comparison', status='INFO',
        purpose=('Shows how concentrated each method\'s structure is. A genotype-based '
                 'ancestry axis in an admixed cohort typically puts substantial variance '
                 'in the first one or two components and then flattens; a slow decay '
                 'suggests the components are picking up noise rather than population '
                 'structure.'),
        interpretation=('Use this to choose a component count per method rather than '
                        'carrying a fixed number. The elbow location is the relevant '
                        'feature, not the absolute variance, which is not comparable '
                        'across methods because the inputs differ (cluster indices for A '
                        'and B, betas for C and D).'),
        figure_b64=fig_b64, figure_alt=fig_alt))

    fig_b64, fig_alt = scatter_figure(results, methods, groups,
                                      args.group_column or '')
    modules.append(QCModule(
        anchor='scatter', title='Component scatter', status='INFO',
        purpose=('PC1 against PC2 for each method. With a grouping supplied, points are '
                 'coloured by it.'),
        interpretation=('Discrete clusters aligned with the grouping indicate an axis '
                        'that separates the labelled groups. Continuous spread within a '
                        'cluster is the admixture structure that a self-reported label '
                        'cannot carry, and is the reason a derived axis might add '
                        'something over the demographic covariate alone.'),
        figure_b64=fig_b64, figure_alt=fig_alt))

    modules.append(build_overlap_module(results, methods))
    modules.append(build_leakage_module(results, methods))
    modules.append(build_group_module(results, methods, groups,
                                      args.group_column or 'group', args.n_pc))
    modules.append(build_covariate_module(results, methods, covar_df, args.n_pc))

    meta = {
        'Dataset': args.dataset,
        'Methylation matrix': args.methylation,
        'Probes read': f"{scan['n_probes']:,}",
        'Samples': f"{len(samples):,}",
        'Methods': ', '.join(methods) if methods else 'none',
    }
    html_doc = render_html(args.dataset, meta, modules,
                           report_title='Ancestry-instrument evaluation',
                           generator='tools/ancestry_probes_report.py')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        fh.write(html_doc)
    logger.info('Wrote %s', args.out)

    if args.json:
        payload = {
            'dataset': args.dataset,
            'generated': datetime.datetime.now().isoformat(timespec='seconds'),
            'methylation': args.methylation,
            'n_probes_read': scan['n_probes'],
            'n_samples': len(samples),
            'n_probes_dropped_na': scan['n_dropped_na'],
            'n_probes_excluded': scan.get('n_excluded', 0),
            'input_was_mvalues': scan['scale_is_m'],
            'parameters': {
                'pca_scale': args.pca_scale,
                'exclude_probes': args.exclude_probes,
                'gap_threshold': args.gap_threshold,
                'min_group_frac': args.min_group_frac,
                'min_group_abs': args.min_group_abs,
                'n_pc': args.n_pc,
                'subsample_probes': args.subsample_probes,
                'seed': args.seed,
            },
            'methods': {},
            'overlap': {},
        }
        for m in methods:
            r = results[m]
            payload['methods'][m] = {
                'n_found': int(r['n_found']),
                'n_used': int(r['n_used']),
                'explained_variance_ratio': (
                    [float(x) for x in r['evr']] if r['evr'] is not None else []),
                'mean_clusters': r.get('mean_clusters'),
                'frac_trimodal': r.get('frac_trimodal'),
                'n_trimodal': r.get('n_trimodal'),
                'n_bimodal': r.get('n_bimodal'),
                'median_allele_frequency': r.get('median_af'),
                'eta_squared': (eta_squared(r['scores'], groups)
                                if groups is not None and r['scores'] is not None
                                else None),
            }
        for i, a in enumerate(methods):
            for b in methods[i + 1:]:
                payload['overlap'][f'{a}|{b}'] = subspace_overlap(
                    results[a]['scores'], results[b]['scores'])
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or '.', exist_ok=True)
        with open(args.json, 'w') as fh:
            json.dump(payload, fh, indent=2)
        logger.info('Wrote %s', args.json)

    if args.probes_out:
        rows = []
        for meta, label in ((rs_meta, 'rs'), (gap_meta, 'gap')):
            if meta is None:
                continue
            for pid, ncl, dm in zip(meta['ids'], meta['n_clusters'],
                                    meta['dosage_mean']):
                rows.append({
                    'probe_id': pid,
                    'method': label,
                    'n_clusters': int(ncl),
                    # Defined only for a resolved three-cluster (biallelic) probe.
                    'allele_frequency': (round(float(min(max(dm / 2.0, 0.0), 1.0)), 6)
                                         if ncl >= 3 else ''),
                })
        if rows:
            out = pd.DataFrame(rows)
            os.makedirs(os.path.dirname(os.path.abspath(args.probes_out)) or '.',
                        exist_ok=True)
            out.to_csv(args.probes_out, sep='\t', index=False)
            logger.info('Wrote %s (%d probes)', args.probes_out, len(out))

    if args.scores_out:
        frames = {}
        for m in methods:
            s = results[m]['scores']
            if s is None or s.size == 0:
                continue
            for j in range(s.shape[1]):
                frames[f'{m}_PC{j + 1}'] = s[:, j]
        if frames:
            out = pd.DataFrame(frames, index=samples)
            out.index.name = 'sample'
            os.makedirs(os.path.dirname(os.path.abspath(args.scores_out)) or '.',
                        exist_ok=True)
            out.to_csv(args.scores_out)
            logger.info('Wrote %s', args.scores_out)

    return 0


if __name__ == '__main__':
    sys.exit(main())
