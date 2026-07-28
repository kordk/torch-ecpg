#!/usr/bin/env python3
import argparse
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_permute as E  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Annotates p_permute from precise_mt_p based on the eval_permute report verdict."
    )
    parser.add_argument('-i', '--input', required=True, help="Input catalog parquet carrying region and p-source")
    parser.add_argument('-o', '--output', required=True, help="Output parquet path")
    parser.add_argument('-r', '--eval-report', required=True, help="Path to eval_permute_report.json")
    parser.add_argument('--p-source', default='precise_mt_p',
                        help="Column whose values are copied (default: precise_mt_p)")
    parser.add_argument('--p-column', default='p_permute', help="New column to write (default: p_permute)")
    parser.add_argument('--chunk-size', type=int, default=100000, help="Rows per batch (default: 100000)")

    args = parser.parse_args()

    # 1. Load report and validate stratify verdict
    try:
        with open(args.eval_report, 'r') as f:
            report = json.load(f)
    except Exception as e:
        print(f"Error loading {args.eval_report}: {e}", file=sys.stderr)
        sys.exit(1)

    arms = report.get('arms', {})
    if 'stratify_decision' not in arms:
        print("Fail-closed: 'stratify_decision' absent from report.", file=sys.stderr)
        sys.exit(1)

    stratify = arms['stratify_decision']
    if 'mode' not in stratify or stratify['mode'] != 'per_region':
        print(f"Fail-closed: missing or invalid stratify mode. Expected 'per_region', got {stratify.get('mode')}",
              file=sys.stderr)
        sys.exit(1)

    if 'per_region' not in stratify:
        print("Fail-closed: 'per_region' missing from stratify_decision.", file=sys.stderr)
        sys.exit(1)

    per_region = stratify['per_region']
    divergent_regions = stratify.get('divergent_regions', [])
    recommendation = stratify.get('recommendation', 'unknown')

    # 2. Compute licensed regions
    licensed = {
        R for R, v in per_region.items()
        if v.get('status') in ('ok', 'reference')
    } - set(divergent_regions)

    # 3. Read input and check guards
    parquet_file = pq.ParquetFile(args.input)
    schema = parquet_file.schema
    col_names = schema.names

    if 'region' not in col_names:
        print("Fail-closed: 'region' column absent from input.", file=sys.stderr)
        sys.exit(1)

    if args.p_source not in col_names:
        print(f"Fail-closed: '--p-source' column '{args.p_source}' absent from input.", file=sys.stderr)
        sys.exit(1)

    if args.p_column in col_names:
        print(f"Fail-closed: '--p-column' '{args.p_column}' already present in input. "
              "Writes must be additive.", file=sys.stderr)
        sys.exit(1)

    # 4. Iterate over chunks, populate column, track counts
    writer = None
    region_counts = {r: {'n_rows': 0, 'n_populated': 0} for r in E.CANONICAL_REGIONS}
    total_rows = 0
    total_populated = 0

    try:
        for i, batch in enumerate(parquet_file.iter_batches(batch_size=args.chunk_size)):
            df_chunk = batch.to_pandas()
            if df_chunk.index.names != [None]:
                df_chunk = df_chunk.reset_index()

            # Ensure all non-null regions are recognized
            unique_regions_in_chunk = df_chunk['region'].dropna().unique()
            for r in unique_regions_in_chunk:
                if r not in E.CANONICAL_REGIONS:
                    print(f"Fail-closed: Unrecognized region label '{r}'.", file=sys.stderr)
                    sys.exit(1)

            # Map the new p_column
            # Fill with np.nan initially
            p_new = np.full(len(df_chunk), np.nan, dtype=np.float64)

            # Mask for licensed regions
            licensed_mask = df_chunk['region'].isin(licensed)

            # Copy source where licensed
            if licensed_mask.any():
                # Get actual values, forcing float64 to ensure np.nan compatibility
                source_vals = df_chunk[args.p_source].astype(np.float64).values
                p_new[licensed_mask] = source_vals[licensed_mask]

            # Assign back
            df_chunk[args.p_column] = p_new

            # Update counters
            region_series = df_chunk['region']
            for r in unique_regions_in_chunk:
                r_mask = region_series == r
                r_count = r_mask.sum()
                # Not all licensed regions will have a non-null p-source, but we populated it
                # For counts, we just check where p_new is not nan
                r_pop = (~np.isnan(p_new[r_mask])).sum()

                region_counts[r]['n_rows'] += r_count
                region_counts[r]['n_populated'] += r_pop

            total_rows += len(df_chunk)
            total_populated += (~np.isnan(p_new)).sum()

            # Null regions
            null_mask = region_series.isna()
            if null_mask.any():
                if '(absent)' not in region_counts:
                    region_counts['(absent)'] = {'n_rows': 0, 'n_populated': 0}
                region_counts['(absent)']['n_rows'] += null_mask.sum()

            # Fix coordinate columns as in summarizeOutput_parquet
            if 'mt_chromStart' in df_chunk.columns:
                df_chunk['mt_chromStart'] = pd.to_numeric(df_chunk['mt_chromStart'], errors='coerce').astype('Int64')
            if 'gt_chromStart' in df_chunk.columns:
                df_chunk['gt_chromStart'] = pd.to_numeric(df_chunk['gt_chromStart'], errors='coerce').astype('Int64')
            if 'mt_chromEnd' in df_chunk.columns:
                df_chunk['mt_chromEnd'] = pd.to_numeric(df_chunk['mt_chromEnd'], errors='coerce').astype('Int64')
            if 'gt_chromEnd' in df_chunk.columns:
                df_chunk['gt_chromEnd'] = pd.to_numeric(df_chunk['gt_chromEnd'], errors='coerce').astype('Int64')

            table = pa.Table.from_pandas(df_chunk, preserve_index=False)

            if writer is None:
                explicit_schema = table.schema
                temp_output = args.output + '.tmp'
                writer = pq.ParquetWriter(temp_output, explicit_schema)
            else:
                table = table.cast(explicit_schema)

            writer.write_table(table)

    except Exception as e:
        print(f"Error processing parquet: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        if writer is not None:
            writer.close()
        # Clean up temporary file on failure
        if os.path.exists(args.output + '.tmp') and not sys.exc_info()[0] is None:
            try:
                os.remove(args.output + '.tmp')
            except OSError:
                pass

    # Success, atomic rename
    os.replace(args.output + '.tmp', args.output)

    # 5. Output summary
    print(f"{'region':<11} {'status':<17} {'licensed':<10} {'n_rows':<11} {'n_populated'}")

    # We want to print all regions that are present (n_rows > 0)
    for r in E.CANONICAL_REGIONS:
        stats = region_counts[r]
        if stats['n_rows'] > 0:
            if r in per_region:
                r_status = per_region[r].get('status', '-')
            else:
                r_status = '-'

            r_licensed = 'yes' if r in licensed else 'no'
            print(f"{r:<11} {r_status:<17} {r_licensed:<10} {stats['n_rows']:<11} {stats['n_populated']}")

    if '(absent)' in region_counts and region_counts['(absent)']['n_rows'] > 0:
        stats = region_counts['(absent)']
        print(f"{'(absent)':<11} {'-':<17} {'no':<10} {stats['n_rows']:<11} {stats['n_populated']}")

    print(f"\nTotals: {total_rows} rows, {total_populated} populated. Recommendation: {recommendation}")


if __name__ == '__main__':
    main()
