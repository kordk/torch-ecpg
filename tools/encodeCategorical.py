#!/usr/bin/env python3
"""encodeCategorical.py — one-hot encode categorical covariate columns.

A covariate column holding integer-coded categories (MESA's racegendersite, a
dbGaP phenotype code, a batch ID) is read by the regression as a continuous
variable: level 5 is treated as five times level 1, and the "adjustment" is a
single slope along an arbitrary code ordering. This tool replaces such a column
with k-1 indicator columns so each level carries its own coefficient.

Two portable rules are applied, both configurable, neither anchored to any
cohort:

  1. Minimum cell size. A level with a single observation produces a degenerate
     indicator: that sample's leverage is exactly 1, its residual is exactly 0,
     and it spends a parameter while contributing nothing. Samples in cells
     below --min-cell-size are dropped, and the dropped levels are named in the
     log. With --min-cell-size 1 a singleton is refused outright rather than
     silently producing h = 1.

  2. Reference level. The dropped (reference) level defaults to the most
     frequent one, which is the best-conditioned choice, and the selection is
     logged. Use --reference to pin a specific level when stable column names
     across cohorts matter more.

Dropping samples from the covariate matrix is sufficient to propagate: the
downstream stages (residualize_pca.py, the regression) align M, G and C on the
sample intersection. The dropped IDs are listed in the log and the JSON report.

Usage:
  python3 tools/encodeCategorical.py \
      --input  data_mesa/C_orig.csv \
      --output data_mesa/C_orig.encoded.csv \
      --column racegendersite \
      --min-cell-size 3 \
      --report data_mesa/encodeCategorical.json
"""
import argparse
import json
import logging
import os
import re
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MIN_CELL_SIZE = 2
DEFAULT_MAX_LEVELS = 20   # backstop only; raise explicitly for wide categoricals


class EncodeError(Exception):
    """Raised when the input cannot be encoded as requested."""


def sanitize(value) -> str:
    """Make a level safe for use in a column name, without collapsing levels."""
    s = str(value).strip()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^0-9A-Za-z_.-]', '_', s)
    return s or 'NA'


def level_counts(series: pd.Series) -> pd.Series:
    """Counts per level, ordered by descending frequency then by level."""
    counts = series.value_counts(dropna=False)
    return counts.sort_values(ascending=False, kind='mergesort')


def choose_reference(counts: pd.Series, requested=None):
    """Pick the reference level: the most frequent one unless pinned."""
    if requested is None:
        return counts.index[0]
    match = [lv for lv in counts.index if str(lv) == str(requested)]
    if not match:
        raise EncodeError(
            f"--reference {requested!r} is not a level of this column; "
            f"levels present: {[str(x) for x in counts.index]}")
    return match[0]


def encode_column(df: pd.DataFrame, column: str, min_cell_size: int,
                  reference=None, max_levels: int = DEFAULT_MAX_LEVELS,
                  prefix: str = None):
    """Encode one column. Returns (new_df, record) and never mutates df."""
    if column not in df.columns:
        raise EncodeError(
            f"Column {column!r} not found. Columns present: {list(df.columns)}")

    series = df[column]
    if series.isna().any():
        n_na = int(series.isna().sum())
        raise EncodeError(
            f"Column {column!r} has {n_na} missing value(s). Missing values in a "
            "categorical covariate need an explicit decision (drop the samples, "
            "or code missingness as its own level) rather than a default here.")

    counts = level_counts(series)
    n_levels = len(counts)
    if n_levels < 2:
        raise EncodeError(
            f"Column {column!r} has {n_levels} level(s); nothing to encode.")
    if n_levels > max_levels:
        raise EncodeError(
            f"Column {column!r} has {n_levels} levels, above --max-levels "
            f"({max_levels}). This guard is a backstop against encoding a "
            "continuous column by accident, not a classification rule: it "
            "cannot tell a wide categorical from a continuous integer. If this "
            "column really is categorical, raise --max-levels to encode it.")

    # Rule 1: minimum cell size.
    if min_cell_size < 1:
        raise EncodeError("--min-cell-size below 1 is not meaningful.")
    singletons = [lv for lv in counts.index if counts[lv] == 1]
    if min_cell_size == 1 and singletons:
        raise EncodeError(
            f"Column {column!r} has singleton level(s) "
            f"{[str(x) for x in singletons]} and --min-cell-size is 1. A "
            "single-observation indicator gives that sample leverage of exactly "
            "1, which makes every leverage-based threshold meaningless. Use "
            "--min-cell-size 2 or higher to drop those samples instead.")

    small = [lv for lv in counts.index if counts[lv] < min_cell_size]
    drop_mask = series.isin(small)
    dropped_samples = df.index[drop_mask].tolist()
    kept = df.loc[~drop_mask].copy()
    if kept.empty:
        raise EncodeError(
            f"Column {column!r}: every level is below --min-cell-size "
            f"({min_cell_size}); no samples would remain.")

    counts_kept = level_counts(kept[column])
    if len(counts_kept) < 2:
        raise EncodeError(
            f"Column {column!r}: only {len(counts_kept)} level(s) remain after "
            f"applying --min-cell-size {min_cell_size}; nothing to encode.")

    # Rule 2: reference level.
    ref = choose_reference(counts_kept, reference)
    indicator_levels = [lv for lv in counts_kept.index if lv != ref]

    pfx = prefix if prefix is not None else column
    new_cols = {}
    seen = set()
    for lv in indicator_levels:
        name = f'{pfx}_{sanitize(lv)}'
        if name in seen or name in kept.columns:
            raise EncodeError(
                f"Encoded column name {name!r} collides with an existing "
                "column or another level; use --prefix to disambiguate.")
        seen.add(name)
        new_cols[name] = (kept[column] == lv).astype(float).to_numpy()

    # Insert the indicators where the original column sat, so column order is
    # stable and diffs against the previous covariate matrix stay readable.
    ordered = []
    for c in kept.columns:
        if c == column:
            ordered.extend(new_cols.keys())
        else:
            ordered.append(c)
    out = kept.drop(columns=[column])
    for name, values in new_cols.items():
        out[name] = values
    out = out[ordered]

    record = {
        'column': column,
        'n_levels_input': int(n_levels),
        'n_levels_encoded': int(len(counts_kept)),
        'n_indicators': int(len(indicator_levels)),
        'reference_level': str(ref),
        'reference_selection': 'pinned' if reference is not None else 'most_frequent',
        'reference_n': int(counts_kept[ref]),
        'min_cell_size': int(min_cell_size),
        'levels_dropped': [str(lv) for lv in small],
        'levels_dropped_n': {str(lv): int(counts[lv]) for lv in small},
        'n_samples_dropped': int(len(dropped_samples)),
        'samples_dropped': [str(s) for s in dropped_samples],
        'n_samples_in': int(len(df)),
        'n_samples_out': int(len(out)),
        'indicator_columns': list(new_cols.keys()),
        'level_counts': {str(lv): int(counts_kept[lv]) for lv in counts_kept.index},
    }

    logger.info("Column %r: %d levels, %d samples", column, n_levels, len(df))
    for lv in counts.index:
        flag = ''
        if lv in small:
            flag = f'  <- dropped (< {min_cell_size})'
        elif lv == ref:
            flag = '  <- reference'
        logger.info("    level %-12s n=%-6d%s", str(lv), int(counts[lv]), flag)
    if small:
        logger.info("  Dropped %d sample(s) in %d level(s) below --min-cell-size %d",
                    len(dropped_samples), len(small), min_cell_size)
    logger.info("  Reference level %r (%s, n=%d); emitting %d indicator column(s)",
                str(ref), record['reference_selection'], record['reference_n'],
                len(indicator_levels))

    return out, record


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True, help='Covariate CSV, samples in rows')
    p.add_argument('--output', required=True, help='Output CSV path')
    p.add_argument('--column', required=True, action='append',
                   help='Column to encode; repeat for several')
    p.add_argument('--min-cell-size', type=int, default=DEFAULT_MIN_CELL_SIZE,
                   help=f'Drop samples in levels smaller than this '
                        f'(default {DEFAULT_MIN_CELL_SIZE}; 1 refuses singletons)')
    p.add_argument('--reference', action='append', default=None,
                   help='Pin the reference level; repeat per --column, in order')
    p.add_argument('--prefix', action='append', default=None,
                   help='Column-name prefix; repeat per --column, in order')
    p.add_argument('--max-levels', type=int, default=DEFAULT_MAX_LEVELS,
                   help=f'Refuse columns with more levels than this '
                        f'(default {DEFAULT_MAX_LEVELS})')
    p.add_argument('--report', default=None, help='Optional JSON report path')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not os.path.exists(args.input):
        logger.error('Input not found: %s', args.input)
        return 2

    def per_column(opt, name):
        if opt is None:
            return [None] * len(args.column)
        if len(opt) == 1 and len(args.column) > 1:
            return opt * len(args.column)
        if len(opt) != len(args.column):
            raise EncodeError(
                f'--{name} given {len(opt)} time(s) for {len(args.column)} '
                '--column argument(s); supply one per column or one for all.')
        return opt

    # float_precision='round_trip' is required for exact preservation: pandas'
    # default C parser is not correctly rounded and shifts roughly 15% of
    # float64 values by one ULP on read. This tool rewrites every column, so
    # without it, columns it never touches would drift on each pass.
    df = pd.read_csv(args.input, index_col=0, float_precision='round_trip')
    df.index = df.index.astype(str)
    logger.info('Read %s: %d samples x %d columns',
                args.input, df.shape[0], df.shape[1])

    records = []
    try:
        refs = per_column(args.reference, 'reference')
        prefixes = per_column(args.prefix, 'prefix')
        for column, ref, pfx in zip(args.column, refs, prefixes):
            df, record = encode_column(df, column, args.min_cell_size,
                                       reference=ref, max_levels=args.max_levels,
                                       prefix=pfx)
            records.append(record)
    except EncodeError as exc:
        logger.error('%s', exc)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    # The default float formatter writes shortest round-trip text and is exact.
    # (An explicit '%.17g' is measurably worse: fixed 17 significant digits can
    # parse back to a neighbouring float.) The read side is where precision is
    # lost -- see read_csv(float_precision='round_trip') above.
    df.to_csv(args.output)
    logger.info('Wrote %s: %d samples x %d columns',
                args.output, df.shape[0], df.shape[1])

    if args.report:
        payload = {
            'input': args.input,
            'output': args.output,
            'min_cell_size': args.min_cell_size,
            'max_levels': args.max_levels,
            'columns': records,
            'n_samples_out': int(df.shape[0]),
            'n_columns_out': int(df.shape[1]),
            'columns_out': list(df.columns),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or '.',
                    exist_ok=True)
        with open(args.report, 'w') as fh:
            json.dump(payload, fh, indent=2)
        logger.info('Wrote %s', args.report)

    return 0


if __name__ == '__main__':
    sys.exit(main())
