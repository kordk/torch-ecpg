#!/usr/bin/env python3
"""Adds raw bootstrap-vs-analytic concordance scores to a bootstrapped catalog.

Every score is a deterministic function of columns already present on
bootstrap_merged.parquet. No thresholds, flags, or filters are applied and no
row is removed: this tool scores and summarizes so that a threshold decision can
be made later from measured distributions rather than assumed ones.

Scores are null wherever the bootstrap did not run, matching p_boot's coverage.
"""
import argparse
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_permute as E  # noqa: E402

SCORE_COLUMNS = [
    'boot_se_ratio',
    'boot_bias_ratio',
    'boot_ci_skew',
    'boot_p_floor_gap',
]

REQUIRED_INPUTS = [
    'mt_est', 'mt_err', 'mt_t',
    'mt_est_boot_mean', 'mt_est_boot_std',
    'ci_low', 'ci_high', 'p_boot',
]

PERCENTILES = [1, 5, 25, 50, 75, 95, 99]

LOG10 = np.log(10.0)


def _safe_divide(num, den):
    """Element-wise division yielding NaN (never inf) on a zero or non-finite denominator."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.full(num.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[ok] = num[ok] / den[ok]
    return out


def compute_scores(df):
    """Return a dict of the four raw score arrays for one chunk."""
    mt_est = df['mt_est'].to_numpy(dtype=np.float64)
    mt_err = df['mt_err'].to_numpy(dtype=np.float64)
    mt_t = df['mt_t'].to_numpy(dtype=np.float64)
    b_mean = df['mt_est_boot_mean'].to_numpy(dtype=np.float64)
    b_std = df['mt_est_boot_std'].to_numpy(dtype=np.float64)
    ci_low = df['ci_low'].to_numpy(dtype=np.float64)
    ci_high = df['ci_high'].to_numpy(dtype=np.float64)
    p_boot = df['p_boot'].to_numpy(dtype=np.float64)

    se_ratio = _safe_divide(b_std, mt_err)
    bias_ratio = _safe_divide(b_mean - mt_est, b_std)

    width = ci_high - ci_low
    skew = _safe_divide((ci_high - mt_est) - (mt_est - ci_low), width)

    # log10 of the normal-implied two-sided sign-instability probability,
    # via logsf so it stays finite for large |t| where 2*sf underflows.
    with np.errstate(invalid='ignore'):
        implied_log10 = (np.log(2.0) + stats.norm.logsf(np.abs(mt_t))) / LOG10
    observed_log10 = np.full(p_boot.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(p_boot) & (p_boot > 0)
    observed_log10[ok] = np.log10(p_boot[ok])
    floor_gap = observed_log10 - implied_log10
    floor_gap[~np.isfinite(floor_gap)] = np.nan

    return {
        'boot_se_ratio': se_ratio,
        'boot_bias_ratio': bias_ratio,
        'boot_ci_skew': skew,
        'boot_p_floor_gap': floor_gap,
    }


class Accumulator:
    """Collects finite score values so raw percentiles can be reported."""

    def __init__(self, reservoir_cap):
        self.cap = reservoir_cap
        self.values = {c: [] for c in SCORE_COLUMNS}
        self.n_finite = {c: 0 for c in SCORE_COLUMNS}
        self.n_kept = {c: 0 for c in SCORE_COLUMNS}
        self.rng = np.random.default_rng(0)

    def add(self, col, arr):
        finite = arr[np.isfinite(arr)]
        self.n_finite[col] += finite.size
        if finite.size == 0:
            return
        room = self.cap - self.n_kept[col]
        if room <= 0:
            return
        take = finite if finite.size <= room else self.rng.choice(finite, room, replace=False)
        self.values[col].append(np.asarray(take, dtype=np.float64))
        self.n_kept[col] += take.size

    def summary(self, col):
        if not self.values[col]:
            return None
        v = np.concatenate(self.values[col])
        pct = np.percentile(v, PERCENTILES)
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        return {
            'n_finite': int(self.n_finite[col]),
            'n_used_for_percentiles': int(v.size),
            'truncated': bool(self.n_finite[col] > self.n_kept[col]),
            'min': float(v.min()),
            'max': float(v.max()),
            'median': med,
            'mad': mad,
            'percentiles': {str(p): float(q) for p, q in zip(PERCENTILES, pct)},
        }


def main():
    parser = argparse.ArgumentParser(
        description="Adds raw bootstrap-vs-analytic concordance scores. No thresholds are applied."
    )
    parser.add_argument('-i', '--input', required=True, help="Bootstrapped catalog parquet")
    parser.add_argument('-o', '--output', required=True, help="Output parquet path")
    parser.add_argument('-s', '--summary-json', default=None,
                        help="Optional path for the machine-readable distribution summary")
    parser.add_argument('--chunk-size', type=int, default=100000, help="Rows per batch (default: 100000)")
    parser.add_argument('--percentile-reservoir', type=int, default=2000000,
                        help="Max finite values retained per score for percentiles (default: 2000000)")
    args = parser.parse_args()

    parquet_file = pq.ParquetFile(args.input)
    col_names = parquet_file.schema.names

    missing = [c for c in REQUIRED_INPUTS if c not in col_names]
    if missing:
        print(f"Fail-closed: required column(s) absent from input: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    collisions = [c for c in SCORE_COLUMNS if c in col_names]
    if collisions:
        print(f"Fail-closed: score column(s) already present in input: {', '.join(collisions)}. "
              "Writes must be additive.", file=sys.stderr)
        sys.exit(1)

    has_region = 'region' in col_names

    writer = None
    explicit_schema = None
    acc = Accumulator(args.percentile_reservoir)
    region_acc = {}
    total_rows = 0
    n_bootstrapped = 0

    try:
        for batch in parquet_file.iter_batches(batch_size=args.chunk_size):
            df_chunk = batch.to_pandas()
            if df_chunk.index.names != [None]:
                df_chunk = df_chunk.reset_index()

            scores = compute_scores(df_chunk)
            for c in SCORE_COLUMNS:
                df_chunk[c] = scores[c]
                acc.add(c, scores[c])

            total_rows += len(df_chunk)
            n_bootstrapped += int(np.isfinite(
                df_chunk['mt_est_boot_std'].to_numpy(dtype=np.float64)).sum())

            if has_region:
                region_series = df_chunk['region']
                for r in region_series.dropna().unique():
                    if r not in E.CANONICAL_REGIONS:
                        continue
                    m = (region_series == r).to_numpy()
                    d = region_acc.setdefault(r, {'n_rows': 0, 'n_scored': 0})
                    d['n_rows'] += int(m.sum())
                    d['n_scored'] += int(np.isfinite(scores['boot_se_ratio'][m]).sum())

            for c in ('mt_chromStart', 'gt_chromStart', 'mt_chromEnd', 'gt_chromEnd'):
                if c in df_chunk.columns:
                    df_chunk[c] = pd.to_numeric(df_chunk[c], errors='coerce').astype('Int64')

            table = pa.Table.from_pandas(df_chunk, preserve_index=False)
            if writer is None:
                explicit_schema = table.schema
                writer = pq.ParquetWriter(args.output + '.tmp', explicit_schema)
            else:
                table = table.cast(explicit_schema)
            writer.write_table(table)

    except Exception as e:
        print(f"Error processing parquet: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        if writer is not None:
            writer.close()
            writer = None
        if os.path.exists(args.output + '.tmp'):
            try:
                os.remove(args.output + '.tmp')
            except OSError:
                pass
        sys.exit(1)
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        if os.path.exists(args.output + '.tmp'):
            os.remove(args.output + '.tmp')
        print("Fail-closed: input contains no rows.", file=sys.stderr)
        sys.exit(1)

    os.replace(args.output + '.tmp', args.output)

    summary = {
        'input': os.path.abspath(args.input),
        'output': os.path.abspath(args.output),
        'total_rows': int(total_rows),
        'n_bootstrapped': int(n_bootstrapped),
        'coverage_fraction': (float(n_bootstrapped) / total_rows) if total_rows else 0.0,
        'scores': {c: acc.summary(c) for c in SCORE_COLUMNS},
        'by_region': region_acc if has_region else None,
        'note': ('Raw scores only. No thresholds, flags, or filters are applied; '
                 'no rows are removed. Percentiles describe the observed distribution '
                 'and are not cut points.'),
    }

    print("Provenance: wrote %s as raw scores; null where the bootstrap did not run. "
          "No thresholds applied, no rows removed." % ', '.join(SCORE_COLUMNS))
    print(f"Rows: {total_rows}  bootstrapped: {n_bootstrapped} "
          f"({100.0 * summary['coverage_fraction']:.4f}%)")
    print()
    hdr = f"{'score':<19}{'n_finite':>12}{'median':>12}{'MAD':>12}"
    hdr += ''.join(f"{'p' + str(p):>12}" for p in PERCENTILES)
    print(hdr)
    for c in SCORE_COLUMNS:
        s = summary['scores'][c]
        if s is None:
            print(f"{c:<19}{0:>12}{'-':>12}{'-':>12}" + ''.join(f"{'-':>12}" for _ in PERCENTILES))
            continue
        row = f"{c:<19}{s['n_finite']:>12}{s['median']:>12.5g}{s['mad']:>12.5g}"
        row += ''.join(f"{s['percentiles'][str(p)]:>12.5g}" for p in PERCENTILES)
        print(row)

    if has_region and region_acc:
        print()
        print(f"{'region':<11}{'n_rows':>12}{'n_scored':>12}")
        for r in E.CANONICAL_REGIONS:
            if r in region_acc:
                d = region_acc[r]
                print(f"{r:<11}{d['n_rows']:>12}{d['n_scored']:>12}")

    if args.summary_json:
        with open(args.summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to {args.summary_json}")


if __name__ == '__main__':
    main()
