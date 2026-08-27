#!/usr/bin/env python3

## kord.kober@ucsf.edu
## github.com/kordk/torch-ecpg

"""Nested-ring figure of eCpG-transcript relationship proportions.

Follows the presentation of Kennedy et al. BMC Genomics (2018) 19:476, Fig. 3
-- one ring per dataset, innermost first -- but reports every region tecpg
assigns, rather than collapsing them.

`pipeline.sh` stage 5 (`assignRegionToEcpg_parquet.py`) labels each pair with
exactly one of seven mutually exclusive regions, measured against the GENCODE
gene model span:

    DISTAL5    > 50 kb upstream of the gene model
    CIS5       2.5-50 kb upstream of the TSS
    PROMOTER   within 2.5 kb of the TSS
    GENEBODY   inside the transcript
    CIS3       within 50 kb downstream of the gene end
    DISTAL3    > 50 kb downstream of the gene end
    TRANS      different chromosome

Wedges are drawn in that order -- 5' to 3' around the gene, then trans -- and
coloured by family so the groups read at a glance: blues for the distal and
trans regions, greens for the proximal ones, amber for the gene body. Every
region is labelled as `% (N)`, in place where the wedge is large enough and in
the value table beneath the rings in every case.

`--collapse kennedy` folds the seven into the four categories of that paper's
Fig. 4 (cis / gene body / distal / trans) when a like-for-like comparison with
the published figure is what is wanted.

Significance defaults to the pipeline's own criterion, BH-FDR < 0.05 on the
`fdr_est` column written by stage 7 (`summarizeOutput_parquet.py
--calculate-fdr`). Pass `--p-thresh` to add a p-value cut on top of it (for
example `--p-thresh 1e-11`, the Kennedy "genome-wide significant" threshold and
`PRIMARY_THRESH` in `pipelineBenchmarkKennedy.sh`), and `--max-fdr 1` to drop
the FDR criterion entirely.

Input is any region-annotated catalog carried through stages 5-7b:

    output_<dataset>/summarized.parquet
    output_<dataset>/summarized.influence.parquet
    output_<dataset>/retained/summarized.parquet   (pipelinePost.sh stage 2)

NOTE: do not point this at output_<dataset>/bootstrap_merged.parquet. That
catalog is the bootstrap candidate set built by createBootstrapList.py under
--min-per-region / --max-per-region quotas, so its regional composition is set
by those quotas rather than by the data.

Example
-------
    python3 tools/plotRegionProportions.py \\
        --input GTP=output_gtp/retained/summarized.parquet \\
        --input MESA=output_mesa/retained/summarized.parquet \\
        --out-dir figures/
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)

DEFAULT_P_COLUMN = 'precise_mt_p'
FALLBACK_P_COLUMNS = ['precise_mt_p', 'mt_p']
DEFAULT_FDR_COLUMN = 'fdr_est'
DEFAULT_REGION_COLUMN = 'region'
DEFAULT_MAX_FDR = 0.05

## tecpg v2 region label -> display name, in 5'-to-3' order around the gene,
## with trans last. This is also the wedge order.
REGION_DISPLAY = {
    'DISTAL5': "distal 5'",
    'CIS5': "cis 5'",
    'PROMOTER': 'promoter',
    'GENEBODY': 'gene body',
    'CIS3': "cis 3'",
    'DISTAL3': "distal 3'",
    'TRANS': 'trans',
}

## Kennedy et al. (2018) Fig. 4 categories, for --collapse kennedy. The
## promoter window (+/- 2.5 kb of the TSS) is inside 50 kb of the transcript,
## so it is part of cis under their definition.
REGION_KENNEDY = {
    'DISTAL5': 'distal',
    'CIS5': 'cis',
    'PROMOTER': 'cis',
    'GENEBODY': 'gene body',
    'CIS3': 'cis',
    'DISTAL3': 'distal',
    'TRANS': 'trans',
}

CATEGORY_ORDER = {
    'none': [REGION_DISPLAY[region] for region in
             ['DISTAL5', 'CIS5', 'PROMOTER', 'GENEBODY', 'CIS3', 'DISTAL3',
              'TRANS']],
    'kennedy': ['cis', 'gene body', 'distal', 'trans'],
}

## Both palettes pass the adjacent-pair and all-pairs colorblind separation
## checks against a light surface, inside the lightness band and above the
## chroma floor. Families are shades of one hue: distal/trans blues, proximal
## greens, gene body amber. Shades are assigned so that the lightest green
## never lands beside the amber, where protanopia is weakest.
CATEGORY_COLORS = {
    # seven-region default
    "distal 5'": '#1A5490',
    "cis 5'": '#5FB25F',
    'promoter': '#1D6A25',
    'gene body': '#E3A600',
    "cis 3'": '#2E8A34',
    "distal 3'": '#2E86C4',
    'trans': '#5CC0EE',
    # --collapse kennedy
    'cis': '#2E7A33',
    'distal': '#1B5FA8',
}

CATEGORY_DESCRIPTIONS = {
    "distal 5'": "> 50 kb upstream of the gene",
    "cis 5'": '2.5-50 kb upstream of the TSS',
    'promoter': 'within 2.5 kb of the TSS',
    'gene body': 'inside the transcript',
    "cis 3'": 'within 50 kb downstream of the gene',
    "distal 3'": '> 50 kb downstream of the gene',
    'trans': 'different chromosome',
    'cis': '< 50 kb from the transcript',
    'distal': '> 50 kb from the transcript, same chromosome',
}


def format_share(value, total):
    """Formats one cell as `54.1% (5,505)`."""
    if total == 0:
        return f'-- ({value:,})'
    return f'{100.0 * value / total:.1f}% ({value:,})'


#### Input parsing ####################################################
def parse_input_spec(spec):
    """Splits a LABEL=PATH argument, tolerating '=' inside the path."""
    if '=' not in spec:
        raise argparse.ArgumentTypeError(
            f"--input must be LABEL=PATH (got {spec!r})"
        )
    label, path = spec.split('=', 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError(
            f"--input must be LABEL=PATH with both parts non-empty (got {spec!r})"
        )
    return label, path


def available_columns(path):
    """Column names of a Parquet or delimited catalog, without reading it."""
    if path.endswith('.parquet'):
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(path).schema_arrow.names)
    return list(pd.read_csv(path, nrows=0).columns)


def resolve_p_column(path, requested, columns):
    """Chooses the p-value column, preferring the recalculated float64 one.

    Stage 6 (recalculate_pvalues_parquet.py) writes `precise_mt_p` in float64.
    The mapper's own `mt_p` is float32 and cancels out below ~6e-08, so falling
    back to it is reported rather than done silently.
    """
    if requested is not None:
        if requested not in columns:
            sys.exit(
                f"[ERROR] p-value column {requested!r} not present in {path}. "
                f"Available: {sorted(columns)}"
            )
        return requested
    for candidate in FALLBACK_P_COLUMNS:
        if candidate in columns:
            if candidate != DEFAULT_P_COLUMN:
                logger.warning(
                    '%s has no %s column; using float32 %s instead. Its '
                    'precision floor is ~6e-08, so a threshold below that '
                    'cannot be applied faithfully. Re-run pipeline.sh stage '
                    'precise_p to get %s.',
                    path, DEFAULT_P_COLUMN, candidate, DEFAULT_P_COLUMN,
                )
            return candidate
    return None


#### Counting #########################################################
def iter_batches(path, columns, batch_size):
    """Yields DataFrames of the requested columns from Parquet or CSV."""
    if path.endswith('.parquet'):
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size,
                                               columns=columns):
            df = batch.to_pandas()
            if df.index.names != [None]:
                df = df.reset_index()
            yield df
    else:
        for df in pd.read_csv(path, usecols=columns, chunksize=batch_size):
            yield df


def count_catalog(label, path, args, category_map, categories):
    """Counts significant pairs (or unique CpGs) per category.

    Returns (counts dict, diagnostics dict). Counting is streamed batch by
    batch so a genome-wide catalog never has to be held in memory; --unit cpg
    keeps one id set per category, which is the only unavoidable state.
    """
    catalog_columns = available_columns(path)

    if args.region_column not in catalog_columns:
        sys.exit(
            f'[ERROR] region column {args.region_column!r} not present in '
            f'{path}. Region annotation is added by pipeline.sh stage 5 '
            '(assignRegionToEcpg_parquet.py); this catalog appears to predate it.'
        )

    columns = [args.region_column]

    use_fdr = args.max_fdr < 1.0
    if use_fdr:
        if args.fdr_column not in catalog_columns:
            sys.exit(
                f'[ERROR] FDR column {args.fdr_column!r} is not present in '
                f'{path}. It is written by pipeline.sh stage 7 '
                '(summarizeOutput_parquet.py --calculate-fdr). Pass '
                '--max-fdr 1 to drop the FDR criterion, or --p-thresh to '
                'select on the p-value instead.'
            )
        columns.append(args.fdr_column)

    p_column = None
    if args.p_thresh is not None:
        p_column = resolve_p_column(path, args.p_column, catalog_columns)
        if p_column is None:
            sys.exit(
                f'[ERROR] --p-thresh given but {path} carries none of '
                f'{FALLBACK_P_COLUMNS}.'
            )
        columns.append(p_column)

    if not use_fdr and p_column is None:
        sys.exit(
            '[ERROR] no significance criterion: --max-fdr 1 disables the FDR '
            'filter and no --p-thresh was given. Every mapped pair would be '
            'counted, which is not what this figure reports.'
        )

    if args.unit == 'cpg':
        if 'mt_id' not in catalog_columns:
            sys.exit(f'[ERROR] --unit cpg needs an mt_id column; {path} has none.')
        columns.append('mt_id')

    counts = {category: 0 for category in categories}
    id_sets = {category: set() for category in categories}

    n_rows = 0
    n_sig = 0
    n_missing_criterion = 0
    n_unassigned = 0
    n_unknown_region = 0
    unknown_labels = set()

    for df in iter_batches(path, columns, args.batch_size):
        n_rows += len(df)

        if use_fdr:
            missing = df[args.fdr_column].isna()
            n_missing_criterion += int(missing.sum())
            df = df.loc[~missing & (df[args.fdr_column] < args.max_fdr)]
        if p_column is not None and not df.empty:
            missing = df[p_column].isna()
            n_missing_criterion += int(missing.sum())
            df = df.loc[~missing & (df[p_column] < args.p_thresh)]
        if df.empty:
            continue
        n_sig += len(df)

        regions = df[args.region_column]
        unassigned = regions.isna()
        n_unassigned += int(unassigned.sum())
        df = df.loc[~unassigned]
        if df.empty:
            continue

        normalized = df[args.region_column].astype(str).str.strip().str.upper()
        mapped = normalized.map(category_map)

        unknown = mapped.isna()
        if unknown.any():
            n_unknown_region += int(unknown.sum())
            unknown_labels.update(normalized[unknown].unique().tolist())
            mapped = mapped[~unknown]
            df = df.loc[mapped.index]

        if args.unit == 'cpg':
            for category, group in df.groupby(mapped):
                id_sets[category].update(group['mt_id'].astype(str).tolist())
        else:
            for category, n in mapped.value_counts().items():
                counts[category] += int(n)

    if args.unit == 'cpg':
        counts = {category: len(ids) for category, ids in id_sets.items()}

    criterion = describe_criterion(args, p_column)
    diagnostics = {
        'path': path,
        'criterion': criterion,
        'criterion_display': describe_criterion(args, p_column, display=True),
        'p_column': p_column or '',
        'fdr_column': args.fdr_column if use_fdr else '',
        'rows_read': n_rows,
        'rows_missing_criterion': n_missing_criterion,
        'rows_significant': n_sig,
        'rows_region_unassigned': n_unassigned,
        'rows_region_unrecognized': n_unknown_region,
        'unrecognized_labels': sorted(unknown_labels),
    }

    logger.info(
        '[%s] %s: %d rows read, %d significant (%s), %d had no region '
        'assignment, %d carried an unrecognized region label%s',
        label, path, n_rows, n_sig, criterion, n_unassigned, n_unknown_region,
        '' if not unknown_labels else f" ({', '.join(sorted(unknown_labels))})",
    )
    if n_missing_criterion:
        logger.warning(
            '[%s] %d rows had a missing FDR or p-value and were dropped.',
            label, n_missing_criterion,
        )
    if n_unknown_region:
        logger.warning(
            '[%s] %d rows carried region labels this tool does not map '
            '(%s). They are excluded from the figure; check that the catalog '
            'was annotated by the current assignRegionToEcpg_parquet.py.',
            label, n_unknown_region, ', '.join(sorted(unknown_labels)),
        )

    return counts, diagnostics


def describe_criterion(args, p_column, display=False):
    """One-line description of the significance filter.

    display=True gives the reader-facing form for the figure title ("FDR <
    0.05"); the default names the actual columns, which is what belongs in the
    log and in the provenance column of the counts table.
    """
    parts = []
    if args.max_fdr < 1.0:
        name = 'FDR' if display else args.fdr_column
        parts.append(f'{name} < {args.max_fdr:g}')
    if args.p_thresh is not None:
        name = 'P' if display else p_column
        parts.append(f'{name} < {args.p_thresh:g}')
    return ' and '.join(parts)


#### Figure ###########################################################
def draw_ring(ax, counts, categories, radius, width, min_label_pct):
    """Draws one dataset as a ring and direct-labels its larger wedges.

    Returns [(category, mid_angle_degrees)] for the non-empty wedges, which the
    caller uses to name the regions around the outermost ring.
    """
    values = [counts[category] for category in categories]
    colors = [CATEGORY_COLORS[category] for category in categories]

    # startangle=90/counterclock=False puts the first category at 12 o'clock
    # and runs clockwise, so every ring reads in the same direction as the
    # table. A 2px surface-coloured edge is the gap between adjacent segments.
    wedges, _ = ax.pie(
        values,
        radius=radius,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=width, edgecolor='white', linewidth=2,
                        antialiased=True),
    )

    total = sum(values)
    placements = []
    for wedge, category, value in zip(wedges, categories, values):
        if value == 0:
            continue
        mid_angle = (wedge.theta1 + wedge.theta2) / 2.0
        placements.append((category, mid_angle))

        if total == 0:
            continue
        pct = 100.0 * value / total
        if pct < min_label_pct:
            continue
        angle = np.deg2rad(mid_angle)
        r = radius - width / 2.0
        ax.text(
            r * np.cos(angle), r * np.sin(angle),
            f'{pct:.1f}%\n({value:,})',
            ha='center', va='center', linespacing=1.15,
            fontsize=8, fontweight='bold', color='white',
        )

    return placements


def spread_labels(values, min_gap, lower, upper):
    """Pushes label positions apart without reordering them.

    Given y positions sorted top to bottom, returns positions at least
    `min_gap` apart and inside [lower, upper]. Order is preserved, so the
    leader lines drawn to them never cross.
    """
    adjusted = list(values)
    for i in range(1, len(adjusted)):
        adjusted[i] = min(adjusted[i], adjusted[i - 1] - min_gap)
    # The downward pass can push the last label below the axes; lift the stack
    # back inside and re-separate upward.
    shortfall = lower - adjusted[-1] if adjusted else 0.0
    if shortfall > 0:
        adjusted = [value + shortfall for value in adjusted]
    for i in range(len(adjusted) - 2, -1, -1):
        adjusted[i] = max(adjusted[i], adjusted[i + 1] + min_gap)
    if adjusted and adjusted[0] > upper:
        overshoot = adjusted[0] - upper
        adjusted = [value - overshoot for value in adjusted]
    return adjusted


def label_regions(ax, placements, radius, x_text, y_lo, y_hi, min_gap):
    """Names each region outside the outermost ring, as in Kennedy Fig. 3.

    Labels are set in two columns and spread vertically within each column, so
    a thin wedge still gets a readable name. The leader wears the region's
    colour, which is what ties the name to its wedge; the text stays in ink.
    """
    sides = {'right': [], 'left': []}
    for category, mid_angle in placements:
        angle = np.deg2rad(mid_angle)
        side = 'right' if np.cos(angle) >= 0 else 'left'
        sides[side].append((category, angle, radius * np.sin(angle)))

    for side, entries in sides.items():
        if not entries:
            continue
        entries.sort(key=lambda entry: entry[2], reverse=True)
        ys = spread_labels([entry[2] for entry in entries], min_gap,
                           y_lo + min_gap, y_hi - min_gap)
        for (category, angle, _), y in zip(entries, ys):
            x = x_text if side == 'right' else -x_text
            ax.annotate(
                category,
                xy=((radius + 0.015) * np.cos(angle),
                    (radius + 0.015) * np.sin(angle)),
                xytext=(x, y),
                ha='left' if side == 'right' else 'right',
                va='center',
                fontsize=9.5,
                color='0.15',
                arrowprops=dict(
                    arrowstyle='-', color=CATEGORY_COLORS[category],
                    linewidth=1.3, shrinkA=3, shrinkB=1,
                    connectionstyle='arc3,rad=0.0',
                ),
                zorder=5,
            )


def ring_position(index, n_rings):
    """Names a ring by where it sits, for the title and the table header."""
    if n_rings == 1:
        return 'only ring'
    if index == 0:
        return 'inner'
    if index == n_rings - 1:
        return 'outer'
    return f'ring {index + 1}'


def draw_table(ax, results, categories):
    """Draws the value table: every category as `% (N)`, for every ring.

    This is also the figure's legend -- the swatch column carries the colour
    identity -- and it is the relief for the low mark/surface contrast of the
    lighter wedge fills, since every value is readable as text.
    """
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    n_rows = len(categories) + 2          # header + categories + total
    row_h = 1.0 / n_rows
    n_rings = len(results)

    col_w = 0.20
    value_x = [0.995 - (n_rings - 1 - j) * col_w for j in range(n_rings)]
    name_x = 0.055
    swatch_x = 0.012
    desc_x = 0.215

    def row_y(index):
        return 1.0 - (index + 0.5) * row_h

    y = row_y(0)
    ax.text(name_x, y, 'region', fontsize=9, fontweight='bold', va='center')
    ax.text(desc_x, y, 'definition', fontsize=9, fontweight='bold',
            va='center', color='0.35')
    for j, (label, _, _, _) in enumerate(results):
        # Which ring is which is stated here rather than on the rings, so the
        # space beside them belongs entirely to the region names.
        ax.text(value_x[j], y, f'{label} ({ring_position(j, n_rings)})',
                fontsize=9, fontweight='bold', va='center', ha='right')
    ax.plot([0.005, 0.995], [1.0 - row_h, 1.0 - row_h],
            color='0.75', linewidth=0.9, clip_on=False)

    for i, category in enumerate(categories):
        y = row_y(i + 1)
        ax.add_patch(Rectangle(
            (swatch_x, y - 0.30 * row_h), 0.028, 0.60 * row_h,
            facecolor=CATEGORY_COLORS[category], edgecolor='none',
        ))
        ax.text(name_x, y, category, fontsize=9, va='center')
        ax.text(desc_x, y, CATEGORY_DESCRIPTIONS[category], fontsize=8,
                va='center', color='0.40')
        for j, (_, counts, total, _) in enumerate(results):
            ax.text(value_x[j], y, format_share(counts[category], total),
                    fontsize=8.5, va='center', ha='right')

    y_line = 1.0 - (len(categories) + 1) * row_h
    ax.plot([0.005, 0.995], [y_line, y_line], color='0.75', linewidth=0.9,
            clip_on=False)
    y = row_y(len(categories) + 1)
    ax.text(name_x, y, 'total', fontsize=9, fontweight='bold', va='center')
    for j, (_, _, total, _) in enumerate(results):
        ax.text(value_x[j], y, f'{total:,}', fontsize=8.5, va='center',
                ha='right', fontweight='bold')


def build_figure(results, categories, args, criterion):
    """Renders the nested rings above their value table."""
    n_rings = len(results)
    n_rows = len(categories) + 2
    table_height = 0.30 * n_rows + 0.25

    # The rings axis keeps aspect='equal', so its panel has to be given the
    # same aspect its data limits have or matplotlib letterboxes it and leaves
    # a band of dead space above the table.
    fig_width = 9.0
    left, right, top, bottom = 0.02, 0.98, 0.93, 0.02
    # Symmetric limits: the region names sit in a column on each side of the
    # rings, so both sides need the same margin.
    x_lo, x_hi = -1.78, 1.78
    y_lo, y_hi = -1.12, 1.12
    rings_height = (fig_width * (right - left) * (y_hi - y_lo)
                    / (x_hi - x_lo))
    fig_height = (rings_height + table_height) / (top - bottom)

    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(
        2, 1,
        height_ratios=[rings_height, table_height],
        hspace=0.03, left=left, right=right, top=top, bottom=bottom,
    )
    ax = fig.add_subplot(grid[0])
    ax.set_aspect('equal')

    outer_radius = 1.0
    ring_width = 0.34 if n_rings <= 2 else min(0.34, 0.72 / n_rings)
    ring_gap = 0.04

    # results[0] is the innermost ring, so radii are assigned outward.
    placements = []
    for index, (label, counts, total, _) in enumerate(results):
        radius = (outer_radius
                  - (n_rings - 1 - index) * (ring_width + ring_gap))
        placements = draw_ring(ax, counts, categories, radius, ring_width,
                               args.min_label_pct)

    # The names hang off the outermost ring, which is the last one drawn.
    # min_gap is one label's line height converted from points into the axes'
    # own units, so the spacing holds whatever the figure is scaled to.
    inches_per_unit = rings_height / (y_hi - y_lo)
    min_gap = (9.5 / 72.0) * 1.7 / inches_per_unit
    label_regions(ax, placements, outer_radius, 1.24, y_lo, y_hi, min_gap)

    # ax.pie autoscales to the wedges alone; widen the box so the ring labels
    # placed to the left of the circle are inside the axes.
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_axis_off()

    draw_table(fig.add_subplot(grid[1]), results, categories)

    if args.title:
        title = args.title
    else:
        if n_rings == 1:
            ring_names = results[0][0]
        else:
            ring_names = '; '.join(
                f'{label} ({ring_position(i, n_rings)})'
                for i, (label, _, _, _) in enumerate(results)
            )
        unit = ('eCpG-transcript relationships' if args.unit == 'pair'
                else 'eCpGs')
        title = (f'Significant {unit} by genomic region\n'
                 f'{ring_names}  |  {criterion}')
    fig.suptitle(title, fontsize=11, y=0.995, va='top')

    return fig


#### Output ###########################################################
def write_counts(results, categories, out_dir, basename):
    """Writes the long-format counts table that backs the figure."""
    rows = []
    for label, counts, total, diagnostics in results:
        for category in categories:
            value = counts[category]
            rows.append({
                'dataset': label,
                'category': category,
                'n': value,
                'pct': (100.0 * value / total) if total else np.nan,
                'label': format_share(value, total),
                'n_total': total,
                'source': diagnostics['path'],
                'criterion': diagnostics['criterion'],
                'rows_read': diagnostics['rows_read'],
                'rows_region_unassigned': diagnostics['rows_region_unassigned'],
                'rows_region_unrecognized':
                    diagnostics['rows_region_unrecognized'],
            })
    table = pd.DataFrame(rows)
    path = os.path.join(out_dir, f'{basename}.csv')
    table.to_csv(path, index=False)
    logger.info('Wrote counts table to %s', path)
    return table


#### CLI ##############################################################
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Nested-ring figure of eCpG-transcript relationship proportions '
            'by genomic region, after Kennedy et al. BMC Genomics (2018) '
            '19:476, Fig. 3.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Example:\n'
            '  python3 tools/plotRegionProportions.py \\\n'
            '      --input GTP=output_gtp/retained/summarized.parquet \\\n'
            '      --input MESA=output_mesa/retained/summarized.parquet \\\n'
            '      --out-dir figures/\n'
        ),
    )
    parser.add_argument(
        '--input', '-i', action='append', required=True, metavar='LABEL=PATH',
        help='Region-annotated catalog to draw as one ring, e.g. '
             'GTP=output_gtp/summarized.parquet. Repeat for more rings; the '
             'first is the innermost. Parquet or CSV.',
    )
    parser.add_argument(
        '--out-dir', '-o', default='.',
        help='Directory for the figure and counts table. Default: current directory.',
    )
    parser.add_argument(
        '--basename', default='region_proportions',
        help='Base filename for the outputs. Default: region_proportions.',
    )
    parser.add_argument(
        '--max-fdr', type=float, default=DEFAULT_MAX_FDR,
        help=f'Keep pairs with an FDR estimate below this value. Default: '
             f'{DEFAULT_MAX_FDR}. Pass 1 to drop the FDR criterion, in which '
             'case --p-thresh becomes required.',
    )
    parser.add_argument(
        '--fdr-column', default=DEFAULT_FDR_COLUMN,
        help=f'FDR column. Default: {DEFAULT_FDR_COLUMN} (pipeline.sh stage 7).',
    )
    parser.add_argument(
        '--p-thresh', '-p', type=float, default=None,
        help='Optional p-value cut applied in addition to --max-fdr, e.g. '
             '1e-11 for the Kennedy 2018 "genome-wide significant" threshold. '
             'Off by default.',
    )
    parser.add_argument(
        '--p-column', default=None,
        help=f'p-value column used by --p-thresh. Default: {DEFAULT_P_COLUMN} '
             'when present, otherwise mt_p with a warning.',
    )
    parser.add_argument(
        '--region-column', default=DEFAULT_REGION_COLUMN,
        help=f'Region label column. Default: {DEFAULT_REGION_COLUMN}.',
    )
    parser.add_argument(
        '--collapse', choices=['none', 'kennedy'], default='none',
        help="How much to collapse the tecpg regions. 'none' (default) reports "
             "all seven independently; 'kennedy' folds them into the four "
             'categories of Kennedy 2018 Fig. 4 (cis / gene body / distal / '
             'trans) for a like-for-like comparison with the published figure.',
    )
    parser.add_argument(
        '--unit', choices=['pair', 'cpg'], default='pair',
        help="What each ring counts. 'pair' counts eCpG-transcript "
             "relationships (the Kennedy Fig. 3 unit, the default); 'cpg' "
             'counts distinct CpGs per category.',
    )
    parser.add_argument(
        '--min-label-pct', type=float, default=5.0,
        help='Only wedges at least this large are labelled in place; every '
             'category is labelled in the table regardless. Default: 5.',
    )
    parser.add_argument(
        '--title', default=None,
        help='Override the generated figure title.',
    )
    parser.add_argument(
        '--formats', nargs='+', default=['png', 'pdf'],
        help='Figure formats to write. Default: png pdf.',
    )
    parser.add_argument(
        '--dpi', type=int, default=300,
        help='Raster resolution. Default: 300.',
    )
    parser.add_argument(
        '--batch-size', type=int, default=500000,
        help='Rows per read batch. Default: 500000.',
    )
    parser.add_argument(
        '--debug', '-D', action='store_true', help='Verbose logging.',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='[%(levelname)s] %(message)s',
    )

    try:
        inputs = [parse_input_spec(spec) for spec in args.input]
    except argparse.ArgumentTypeError as error:
        sys.exit(f'[ERROR] {error}')

    labels = [label for label, _ in inputs]
    if len(set(labels)) != len(labels):
        sys.exit(f'[ERROR] --input labels must be unique (got {labels}).')
    for _, path in inputs:
        if not os.path.isfile(path):
            sys.exit(f'[ERROR] input not found: {path}')
        if os.path.basename(path) == 'bootstrap_merged.parquet':
            logger.warning(
                '%s is the bootstrap candidate set. createBootstrapList.py '
                'builds it under per-region quotas, so its regional '
                'composition reflects those quotas rather than the data. Use '
                'summarized.parquet (or retained/summarized.parquet) instead.',
                path,
            )

    if args.unit == 'cpg':
        logger.warning(
            'Counting distinct CpGs. One CpG can be cis to one transcript and '
            'trans to another, so the categories overlap and their counts sum '
            'to more than the number of distinct significant CpGs; each ring '
            'is normalised by that sum. --unit pair (the Kennedy Fig. 3 unit) '
            'partitions cleanly.'
        )

    if args.collapse == 'kennedy':
        category_map = dict(REGION_KENNEDY)
    else:
        category_map = dict(REGION_DISPLAY)
    categories = CATEGORY_ORDER[args.collapse]

    results = []
    criterion = ''
    criterion_display = ''
    for label, path in inputs:
        counts, diagnostics = count_catalog(label, path, args, category_map,
                                            categories)
        criterion = diagnostics['criterion']
        criterion_display = diagnostics['criterion_display']
        total = sum(counts[category] for category in categories)
        if total == 0:
            sys.exit(
                f'[ERROR] {label} ({path}) has no pairs passing {criterion} '
                'with an assigned region, so its ring cannot be drawn. Loosen '
                'the threshold or check the catalog.'
            )
        results.append((label, counts, total, diagnostics))

    os.makedirs(args.out_dir, exist_ok=True)
    table = write_counts(results, categories, args.out_dir, args.basename)
    print(table.loc[:, ['dataset', 'category', 'label', 'n_total']]
          .to_string(index=False))

    fig = build_figure(results, categories, args, criterion_display)
    for fmt in args.formats:
        path = os.path.join(args.out_dir, f'{args.basename}.{fmt}')
        fig.savefig(path, dpi=args.dpi)
        logger.info('Wrote figure to %s', path)
    plt.close(fig)


if __name__ == '__main__':
    main()
