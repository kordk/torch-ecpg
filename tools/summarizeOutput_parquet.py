import argparse
import multiprocessing
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pyarrow as pa
import pyarrow.parquet as pq
import time
import seaborn as sns
from statsmodels.stats.multitest import multipletests

# Default column names. Parameterizing these lets a second FDR pass write
# beside the analytic one instead of over it; the defaults preserve the
# behaviour of every existing invocation exactly.
DEFAULT_P_COLUMN = 'precise_mt_p'
DEFAULT_FDR_COLUMN = 'fdr_est'

def process_chunk(chunk, sample_prob, df, p_column=DEFAULT_P_COLUMN):
    """
    Process a chunk of the dataframe (passed as pandas DataFrame).
    Returns:
        - set of unique gt_ids
        - set of unique mt_ids
        - count of rows
        - histogram counts (100 bins for p-values 0-1)
        - extracted p-values
        - region array (encoded as ints) or None
        - region string mapping or None
        - top hits dataframe for regions or None
        - top 50 hits dataframe for ig analysis or None
        - number of rows in the chunk before the FDR-universe NaN drop
        - number of rows dropped (missing precise_mt_p / t) before the FDR pool
    """
    # Unique counts
    unique_gt = set(chunk['gt_id'].dropna().unique()) if 'gt_id' in chunk.columns else set()
    unique_mt = set(chunk['mt_id'].dropna().unique()) if 'mt_id' in chunk.columns else set()
    row_count = len(chunk)

    # Calculate high-precision p-values and identify t_col
    t_col = None
    if 'mt_t' in chunk.columns:
        t_col = 'mt_t'
    elif 't' in chunk.columns:
        t_col = 't'

    # Drop NaNs based on the primary column being used to compute p-values
    # This prevents length mismatch errors and mapping desyncs when building region arrays
    # Track before/after counts so the dropped rows (which never enter the
    # global BH-FDR pool) are observable in the run log (M5).
    n_before_drop = len(chunk)
    if p_column in chunk.columns:
        chunk = chunk.dropna(subset=[p_column])
        p_values = chunk[p_column].astype(np.float64).values
    else:
        if not t_col:
            raise ValueError(f"Error: {p_column} and t-statistic column missing in chunk. Available columns: {list(chunk.columns)}")
        chunk = chunk.dropna(subset=[t_col])
        t_stats = chunk[t_col].astype(np.float64).values
        p_values = stats.t.sf(np.abs(t_stats), np.float64(df)) * 2.0
    n_after_drop = len(chunk)
    n_dropped = n_before_drop - n_after_drop

    # Ensure values are within [0, 1] for histogram logic
    hist_counts, _ = np.histogram(p_values, bins=100, range=(0, 1))

    # Region processing
    region_codes = None
    region_uniques = None
    top_hits_df = None

    if 'region' in chunk.columns:
        # Get region factorized arrays for minimal memory footprint of full region array
        # We replace NaNs/None with a generic 'UNKNOWN' before factorizing
        region_col = chunk['region'].fillna('UNKNOWN').astype(str)
        codes, uniques = pd.factorize(region_col)
        region_codes = codes
        region_uniques = uniques.tolist()

        # Build top 10 hits DataFrame for this chunk
        top_cols = {}
        if 'mt_id' in chunk.columns: top_cols['mt_id'] = chunk['mt_id'].values
        if 'gt_id' in chunk.columns: top_cols['gt_id'] = chunk['gt_id'].values
        top_cols['region'] = region_col.values
        if t_col and t_col in chunk.columns: top_cols['mt_t'] = chunk[t_col].values
        top_cols['p-value'] = p_values

        df_region = pd.DataFrame(top_cols)
        # We want the 10 lowest p-values per region
        top_hits_df = df_region.sort_values('p-value').groupby('region').head(10)

    # Integrated Gradients processing
    top_ig_df = None
    ig_cols = [col for col in chunk.columns if col.endswith('_ig')]
    if ig_cols:
        ig_top_cols = {}
        if 'mt_id' in chunk.columns: ig_top_cols['mt_id'] = chunk['mt_id'].values
        if 'gt_id' in chunk.columns: ig_top_cols['gt_id'] = chunk['gt_id'].values
        if t_col and t_col in chunk.columns: ig_top_cols['mt_t'] = chunk[t_col].values
        ig_top_cols['p-value'] = p_values
        for col in ig_cols:
            ig_top_cols[col] = chunk[col].values

        df_ig = pd.DataFrame(ig_top_cols)
        top_ig_df = df_ig.sort_values('p-value').head(50)

    return unique_gt, unique_mt, row_count, hist_counts, p_values, region_codes, region_uniques, top_hits_df, top_ig_df, n_before_drop, n_dropped


def main():
    description_text = """
Summarize tecpg output Parquet.

This script processes large output Parquet files from tecpg in a memory-efficient
manner by reading the data in chunks and using multiprocessing.

Outputs and Metrics Calculated:
  - Total mapping pairs (eCpGs): The total number of valid rows processed.
  - Unique genes: The total number of unique gene IDs (gt_id) found.
  - Unique CpGs: The total number of unique CpG site IDs (mt_id) found.
  - Genomic Inflation Factor (lambda): An estimate of test statistic inflation
    calculated using a reservoir sampling approach (~1 million rows).
  - P-value Histogram: A histogram image (p_value_histogram.png) plotting the
    distribution of p-values.
  - Optional FDR Output: Can output a new Parquet file with an estimated FDR
    column (`fdr_est`) or a boolean column (`is_significant`) based on BH threshold.
    Note: FDR calculations are estimates based on the `--total-tests` argument since
    the full dataset rank is not maintained.
"""

    parser = argparse.ArgumentParser(
        description=description_text,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--main-file", required=True, help="Path to the main filtered output Parquet file.")
    parser.add_argument("--reservoir-file", required=True, help="Path to the diagnostic reservoir CSV file.")
    parser.add_argument("--total-tests", type=int, required=True, help="Total number of tests performed (e.g. 13462597186) for FDR adjustment.")
    parser.add_argument("--df", type=float, required=True, help="Degrees of freedom for t-statistic to p-value conversion.")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Rows per chunk for processing.")
    parser.add_argument("--cores", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="Number of cores to use.")

    # FDR Output Options
    parser.add_argument("--output-fdr-file", help="Path to the output Parquet file to save the results with the new FDR column.")

    parser.add_argument("--p-column", default=DEFAULT_P_COLUMN, help=f"Name of the p-value column to rank and correct. Default: {DEFAULT_P_COLUMN}.")
    parser.add_argument("--fdr-column", default=DEFAULT_FDR_COLUMN, help=f"Name of the FDR column to write. Default: {DEFAULT_FDR_COLUMN}. Must not name an existing column.")
    parser.add_argument("--compare-fdr-column", default=None,
                        help="Name of an EXISTING FDR column to compare the newly written --fdr-column "
                             "against. Optional; off by default. When the two columns share a BH pool and "
                             "denominator they must agree exactly; when the new pool is a strict subset the "
                             "new values must be greater than or equal, never smaller. A smaller value is "
                             "reported as a VIOLATION and exits non-zero.")

    fdr_group = parser.add_mutually_exclusive_group()
    fdr_group.add_argument("--calculate-fdr", action="store_true", help="Calculate and append an estimated FDR column (see --fdr-column).")
    fdr_group.add_argument("--assign-fdr-passfail", action="store_true", help="Append a boolean `is_significant` column based on FDR threshold.")

    args = parser.parse_args()

    main_file = args.main_file
    reservoir_file = args.reservoir_file

    if not os.path.exists(main_file):
        print(f"Error: Main file {main_file} not found.")
        sys.exit(1)
    if not os.path.exists(reservoir_file):
        print(f"Error: Reservoir file {reservoir_file} not found.")
        sys.exit(1)

    if args.output_fdr_file and not (args.calculate_fdr or args.assign_fdr_passfail):
        print("Error: --output-fdr-file specified but no FDR approach chosen. Please specify --calculate-fdr or --assign-fdr-passfail.")
        sys.exit(1)

    # Fail closed on a missing/sentinel total_tests. This value is the BH-FDR
    # denominator (fdr_est = p * total_tests / rank); a non-positive or absent
    # value silently corrupts every estimate, so we reject it rather than
    # computing against a default.
    if args.total_tests is None or args.total_tests <= 0:
        print(
            f"Error: --total-tests must be a positive integer (got {args.total_tests}). "
            "It is the FDR denominator and has no safe default."
        )
        sys.exit(1)

    print(f"Analyzing main file {main_file} and reservoir file {reservoir_file}...")

    # For Parquet, we can quickly get row count if metadata is intact, but we'll just process batches.
    parquet_file = pq.ParquetFile(main_file)
    total_rows = parquet_file.metadata.num_rows

    # Enforce the top-N contract the FDR math depends on: the supplied parquet
    # must be the top-N most-significant of total_tests, so total_tests cannot
    # be smaller than the number of supplied rows. Otherwise fdr_est = p *
    # total_tests / rank is computed against a denominator that is internally
    # inconsistent with the input.
    if args.total_tests < total_rows:
        print(
            f"Error: --total-tests ({args.total_tests}) is smaller than the number of "
            f"rows in {main_file} ({total_rows}). The input parquet must be the top-N "
            "most-significant subset of total_tests, so total_tests >= len(input) must hold."
        )
        sys.exit(1)

    # Check schema for fallback logging
    schema_cols = parquet_file.schema.names

    # Fail closed when a caller names a p-column explicitly and it is absent.
    # The t-statistic fallback below recomputes the ANALYTIC p; falling back
    # for an explicitly requested column would write analytic values under a
    # caller-chosen name, which is a mislabelling rather than a degradation.
    if args.p_column != DEFAULT_P_COLUMN and args.p_column not in schema_cols:
        print(
            f"Error: --p-column '{args.p_column}' was requested explicitly but is not "
            f"present in {main_file}. Refusing to fall back to the t-statistic, which "
            f"would write analytic p-values under that name."
        )
        print(f"Available columns: {schema_cols}")
        sys.exit(1)

    # Refuse to overwrite an existing column. Every write is additive.
    if args.fdr_column in schema_cols:
        print(
            f"Error: --fdr-column '{args.fdr_column}' already exists in {main_file}. "
            f"Writes must be additive; choose a new column name."
        )
        sys.exit(1)

    using_fallback = args.p_column not in schema_cols
    if using_fallback:
        print(f"Warning: '{args.p_column}' column missing in {main_file}. Falling back to t-statistic calculation.")
        print(f"Available columns: {schema_cols}")

    has_region = 'region' in schema_cols
    if has_region:
        print("Log: 'region' column detected in dataset. Regional analysis will be performed.")
    else:
        print("Log: 'region' column not found in dataset. Regional analysis will be skipped.")

    target_sample = 1_000_000
    if total_rows > 0:
        sample_prob = min(1.0, (target_sample * 1.2) / total_rows)
    else:
        sample_prob = 1.0

    print(f"Total rows in Parquet: {total_rows}. Sampling probability for processing: {sample_prob:.4f}")

    pool = multiprocessing.Pool(processes=args.cores)
    results = []

    print("Processing chunks...")
    try:
        for i, batch in enumerate(parquet_file.iter_batches(batch_size=args.chunk_size)):
            df_chunk = batch.to_pandas()
            if df_chunk.index.names != [None]:
                df_chunk = df_chunk.reset_index()
            res = pool.apply_async(process_chunk, (df_chunk, sample_prob, args.df, args.p_column))
            results.append(res)

            if (i + 1) % 100 == 0:
                print(f"Submitted {i+1} chunks...", end='\r')
    except Exception as e:
        print(f"\nError processing chunks: {e}")
        pool.terminate()
        sys.exit(1)

    pool.close()
    pool.join()

    print("\nProcessing complete. Aggregating results...")

    # Aggregate
    final_gt = set()
    final_mt = set()
    total_pairs = 0
    final_hist = np.zeros(100, dtype=int)
    all_p_values = []

    all_top_hits = []
    all_top_ig = []
    # Using lists to accumulate codes before stacking
    region_code_arrays = []
    global_uniques = []
    global_uniques_map = {} # map string -> global int code

    # Track rows dropped (missing precise_mt_p / t) before the global BH-FDR
    # pool, per chunk and in total (M5).
    fdr_universe_before = 0
    fdr_universe_after = 0
    total_dropped_from_fdr = 0

    for chunk_idx, res in enumerate(results):
        try:
            gt_set, mt_set, count, hist, p_vals, codes, uniques, top_hits_df, top_ig_df, n_before_drop, n_dropped = res.get()
            final_gt.update(gt_set)
            final_mt.update(mt_set)
            total_pairs += count
            final_hist += hist
            if len(p_vals) > 0:
                all_p_values.append(p_vals)

            # Log rows dropped from the FDR universe for this chunk (M5).
            n_after_drop = n_before_drop - n_dropped
            fdr_universe_before += n_before_drop
            fdr_universe_after += n_after_drop
            total_dropped_from_fdr += n_dropped
            if n_dropped == 0:
                print(f"Chunk {chunk_idx}: no rows dropped from FDR universe ({n_before_drop} rows).")
            else:
                pct_remaining = round(n_after_drop / n_before_drop * 100, 4) if n_before_drop else 0.0
                print(
                    f"Chunk {chunk_idx}: dropped {n_dropped} rows missing {args.p_column}/t from "
                    f"FDR universe (before: {n_before_drop}, after: {n_after_drop}, "
                    f"{pct_remaining}% remaining)."
                )

            if codes is not None and uniques is not None:
                all_top_hits.append(top_hits_df)

                # Map local codes to global codes
                local_to_global = np.zeros(len(uniques), dtype=int)
                for i, region_str in enumerate(uniques):
                    if region_str not in global_uniques_map:
                        global_uniques_map[region_str] = len(global_uniques)
                        global_uniques.append(region_str)
                    local_to_global[i] = global_uniques_map[region_str]

                # Translate chunk's local codes to global using the mapping array
                # Use np.take or direct indexing (codes can contain -1 for NaNs, but we filled them)
                global_codes = local_to_global[codes]
                region_code_arrays.append(global_codes)

            if top_ig_df is not None:
                all_top_ig.append(top_ig_df)

        except Exception as e:
            sys.stderr.write(
                f"Error retrieving result from worker for chunk {chunk_idx}: {e}\n"
                "Aggregation is incomplete; no summary, FDR threshold or output "
                "file was produced.\n")
            sys.exit(1)

    # Final total of rows dropped from the FDR universe across all chunks (M5).
    if total_dropped_from_fdr == 0:
        print(f"FDR universe: no rows dropped; all {fdr_universe_before} rows entered the BH pool.")
    else:
        pct_remaining = round(fdr_universe_after / fdr_universe_before * 100, 4) if fdr_universe_before else 0.0
        print(
            f"FDR universe: dropped {total_dropped_from_fdr} rows missing {args.p_column}/t in total "
            f"(before: {fdr_universe_before}, after: {fdr_universe_after}, "
            f"{pct_remaining}% entered the BH pool)."
        )

    combined_region_codes = None
    if has_region and region_code_arrays:
        combined_region_codes = np.concatenate(region_code_arrays)

    # Summary Stats
    print("-" * 30)
    print("Summary of Results")
    print("-" * 30)
    print(f"Total mapping pairs (eCpGs): {total_pairs}")
    print(f"Unique genes: {len(final_gt)}")
    print(f"Unique CpGs: {len(final_mt)}")

    # Process Reservoir File
    print("\nProcessing reservoir file...")
    try:
        res_df = pd.read_csv(reservoir_file)
        t_col = None
        if 'mt_t' in res_df.columns:
            t_col = 'mt_t'
        elif 't' in res_df.columns:
            t_col = 't'

        if t_col:
            t_stats = res_df[t_col].dropna().astype(np.float64).values
            df_val = np.float64(args.df)
            res_p_values = stats.t.sf(np.abs(t_stats), df_val) * 2.0

            chi2_obs = stats.chi2.isf(res_p_values, 1)
            median_chi2_obs = np.median(chi2_obs)
            expected_median_chi2 = stats.chi2.ppf(0.5, 1) # ~0.4549
            lambda_gc = median_chi2_obs / expected_median_chi2

            print(f"Genomic Inflation Factor (lambda): {lambda_gc:.4f} (calculated from {len(res_p_values)} reservoir samples)")
            if lambda_gc > 1.1:
                print("WARNING: lambda > 1.1, your model may be inflated.")

            # QQ Plot Generation
            print("Generating QQ Plot...")
            sorted_p = np.sort(res_p_values)
            expected_p = (np.arange(1, len(sorted_p) + 1) - 0.5) / len(sorted_p)

            eps = np.finfo(np.float64).tiny
            obs_log_p = -np.log10(np.clip(sorted_p, eps, 1.0))
            exp_log_p = -np.log10(np.clip(expected_p, eps, 1.0))

            plt.figure(figsize=(8, 8))
            plt.scatter(exp_log_p, obs_log_p, c='black', s=5, alpha=0.5, rasterized=True)

            max_val = max(np.max(exp_log_p), np.max(obs_log_p))
            plt.plot([0, max_val], [0, max_val], color='red', linestyle='--')

            plt.title("QQ Plot (from Reservoir Samples)")
            plt.xlabel("Expected -$log_{10}(p)$")
            plt.ylabel("Observed -$log_{10}(p)$")

            qq_plot_file = "qq_plot.png"
            plt.savefig(qq_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"QQ Plot saved to {qq_plot_file}")

        else:
            print("Warning: Could not find t-statistic column in reservoir file. Skipping lambda and QQ plot.")
    except Exception as e:
        print(f"Error processing reservoir file: {e}")

    # Process FDR threshold discovery
    print("\nCalculating FDR threshold (Benjamini-Hochberg)...")
    p_max_fdr = -1.0
    p_max_fdr_01 = -1.0
    sig_count = 0
    sig_count_01 = 0
    fdr_map = None  # To hold p-value -> fdr mapping if needed

    if all_p_values:
        combined_p_values = np.concatenate(all_p_values)
        total_p_count = len(combined_p_values)
        print(f"Sorting {total_p_count} p-values from main file...")

        # Sort indices for FDR calculation mapping
        sorted_indices = np.argsort(combined_p_values)
        sorted_p_values = combined_p_values[sorted_indices]

        total_tests = args.total_tests
        # Top-N contract (the BH math at fdr_est = p * total_tests / rank
        # assumes the supplied parquet is the top-N most-significant of
        # total_tests): the number of p-values entering the pool cannot exceed
        # total_tests. total_tests >= total_rows was already enforced above;
        # re-assert against the realized pool so a desync fails loudly here.
        if total_p_count > total_tests:
            print(
                f"Error: number of p-values in the FDR pool ({total_p_count}) exceeds "
                f"--total-tests ({total_tests}). The input must be the top-N "
                "most-significant subset of total_tests for the BH estimate to be valid."
            )
            sys.exit(1)
        ranks = np.arange(1, total_p_count + 1)
        bh_limits = (ranks / total_tests) * 0.05
        bh_limits_01 = (ranks / total_tests) * 0.01

        # Calculate FDR < 0.05
        valid_mask = sorted_p_values <= bh_limits
        valid_indices = np.nonzero(valid_mask)[0]

        if len(valid_indices) > 0:
            max_idx = valid_indices[-1]
            p_max_fdr = sorted_p_values[max_idx]
            sig_count = max_idx + 1 # 0-indexed
            print(f"FDR < 0.05 Threshold found!")
            print(f"Maximum P-value satisfying FDR < 0.05 (p_max_fdr): {p_max_fdr:.3e}")
            print(f"Number of pairs remaining significant at FDR < 0.05: {sig_count:,}")
        else:
            print("No pairs remain significant at FDR < 0.05.")

        # Calculate FDR < 0.01
        valid_mask_01 = sorted_p_values <= bh_limits_01
        valid_indices_01 = np.nonzero(valid_mask_01)[0]

        if len(valid_indices_01) > 0:
            max_idx_01 = valid_indices_01[-1]
            p_max_fdr_01 = sorted_p_values[max_idx_01]
            sig_count_01 = max_idx_01 + 1 # 0-indexed
            print(f"\nFDR < 0.01 Threshold found!")
            print(f"Maximum P-value satisfying FDR < 0.01: {p_max_fdr_01:.3e}")
            print(f"Number of pairs remaining significant at FDR < 0.01: {sig_count_01:,}")
        else:
            print("\nNo pairs remain significant at FDR < 0.01.")


        if args.output_fdr_file and args.calculate_fdr:
            print("\nEstimating FDR values...")
            print(f"Note: The calculated FDR values are estimates (`{args.fdr_column}`) because they rely on the `total_tests` parameter and assume the provided dataset contains the top most significant results.")

            # fdr_est = p * total_tests / rank
            # Step down procedure: min(q(i+1), estimated_fdr(i))
            estimated_fdr = sorted_p_values * total_tests / ranks

            # Ensure monotonicity from largest to smallest
            q_values = np.zeros_like(estimated_fdr)
            q_values[-1] = min(1.0, estimated_fdr[-1])
            for i in range(len(estimated_fdr) - 2, -1, -1):
                q_values[i] = min(q_values[i+1], estimated_fdr[i])

            # Create a Series mapping the unique p-values to their max q_value
            # We use drop_duplicates(keep='first') because smaller rank index corresponds to same p-value but larger q-value?
            # Actually, for identical p-values, rank increases, so p * N / rank decreases.
            # Step-down fixes this to make them all identical. We can just map unique p-values to their q-value.
            temp_df = pd.DataFrame({'p_val': sorted_p_values, 'q_val': q_values})
            # Drop duplicates by taking the max q_value for any given p-value to be conservative
            temp_df = temp_df.groupby('p_val', as_index=False).max()
            fdr_map = pd.Series(temp_df['q_val'].values, index=temp_df['p_val']).to_dict()

        # Generate Regional Summaries
        if has_region and combined_region_codes is not None:
            print("\n" + "=" * 40)
            print("Regional Summary of Significant Hits")
            print("=" * 40)

            # Map the sorted indices to the combined region codes
            sorted_region_codes = combined_region_codes[sorted_indices]

            if sig_count > 0:
                print(f"Significant Hits per Region (FDR < 0.05):")
                # Get region codes for significant hits
                sig_codes_05 = sorted_region_codes[:sig_count]
                unique_codes_05, counts_05 = np.unique(sig_codes_05, return_counts=True)
                # Map codes back to strings and print
                for code, count in zip(unique_codes_05, counts_05):
                    region_str = global_uniques[code]
                    print(f"  {region_str:<12} {count:>10,}")
            else:
                print("No significant hits per region at FDR < 0.05.")

            print("-" * 30)

            if sig_count_01 > 0:
                print(f"Significant Hits per Region (FDR < 0.01):")
                sig_codes_01 = sorted_region_codes[:sig_count_01]
                unique_codes_01, counts_01 = np.unique(sig_codes_01, return_counts=True)
                for code, count in zip(unique_codes_01, counts_01):
                    region_str = global_uniques[code]
                    print(f"  {region_str:<12} {count:>10,}")
            else:
                print("No significant hits per region at FDR < 0.01.")

            print("=" * 40 + "\n")

        del combined_p_values
        del sorted_indices
        del sorted_p_values
        if has_region:
            del combined_region_codes
    else:
        print("No p-values found in the main file for FDR adjustment.")

    # Output Top 10 per region
    if has_region and all_top_hits:
        print("\nTop 10 Hits per Region (by Lowest P-value)")
        print("-" * 50)

        try:
            # Combine all chunk top 10s
            combined_top_df = pd.concat(all_top_hits, ignore_index=True)

            # Group by region, sort by p-value, get top 10 globally per region
            final_top_hits = combined_top_df.sort_values('p-value').groupby('region').head(10)

            # Format and print the table
            for region, group in final_top_hits.groupby('region'):
                print(f"\nRegion: {region}")
                print(group.to_string(index=False))
        except Exception as e:
            print(f"Error generating top 10 hits per region: {e}")

    # Write FDR Output
    if args.output_fdr_file:
        print(f"\nWriting output FDR Parquet file to: {args.output_fdr_file}")

        tmp_output_fdr_file = args.output_fdr_file + ".tmp"
        write_error = None

        do_compare = args.compare_fdr_column is not None
        if do_compare and args.compare_fdr_column not in schema_cols:
            print(f"Notice: --compare-fdr-column '{args.compare_fdr_column}' not found in schema. Skipping comparison.")
            do_compare = False

        cmp_n_comparable = 0
        cmp_max_abs_diff = 0.0
        cmp_min_diff = np.inf
        cmp_n_new_null_ref_present = 0
        cmp_n_new_present_ref_null = 0

        writer = None
        try:
            for i, batch in enumerate(parquet_file.iter_batches(batch_size=args.chunk_size)):
                df_chunk = batch.to_pandas()
                if df_chunk.index.names != [None]:
                    df_chunk = df_chunk.reset_index()

                # Retrieve or calculate p-values for this chunk
                if not using_fallback:
                    chunk_p_vals = df_chunk[args.p_column].values
                else:
                    t_col = None
                    if 'mt_t' in df_chunk.columns:
                        t_col = 'mt_t'
                    elif 't' in df_chunk.columns:
                        t_col = 't'
                    t_stats = df_chunk[t_col].values
                    chunk_p_vals = stats.t.sf(np.abs(t_stats), np.float64(args.df)) * 2.0

                if args.assign_fdr_passfail:
                    df_chunk['is_significant'] = chunk_p_vals <= p_max_fdr
                elif args.calculate_fdr and fdr_map is not None:
                    # Map the FDR values using exact p-values
                    # Since floats can be tricky, we map using the exact float values computed
                    # Alternatively, vectorizing the lookup with pandas map:
                    mapped = pd.Series(chunk_p_vals).map(fdr_map)
                    # A genuine map miss fills to 1.0, as before. A row whose
                    # source p-value is itself null stays null, so "not
                    # assessed" remains distinguishable from "assessed and not
                    # significant" -- the p-column may be null by design for a
                    # stratum that did not calibrate.
                    source_is_null = pd.isna(pd.Series(chunk_p_vals))
                    mapped[~source_is_null] = mapped[~source_is_null].fillna(1.0)
                    df_chunk[args.fdr_column] = mapped.values

                if do_compare:
                    new_vals = df_chunk[args.fdr_column].astype(np.float64).values
                    ref_vals = df_chunk[args.compare_fdr_column].astype(np.float64).values

                    new_is_na = np.isnan(new_vals)
                    ref_is_na = np.isnan(ref_vals)

                    cmp_n_new_null_ref_present += (new_is_na & ~ref_is_na).sum()
                    cmp_n_new_present_ref_null += (~new_is_na & ref_is_na).sum()

                    comparable_mask = ~new_is_na & ~ref_is_na
                    if comparable_mask.any():
                        c_new = new_vals[comparable_mask]
                        c_ref = ref_vals[comparable_mask]
                        diff = c_new - c_ref

                        cmp_n_comparable += len(diff)
                        chunk_max_abs = np.max(np.abs(diff))
                        chunk_min = np.min(diff)

                        if chunk_max_abs > cmp_max_abs_diff:
                            cmp_max_abs_diff = chunk_max_abs
                        if chunk_min < cmp_min_diff:
                            cmp_min_diff = chunk_min

                # Ensure coordinate columns are consistently typed as nullable Int64
                # This prevents pyarrow from deducing int64 for chunks without nulls and float64 for chunks with nulls.
                if 'mt_chromStart' in df_chunk.columns:
                    df_chunk['mt_chromStart'] = pd.to_numeric(df_chunk['mt_chromStart'], errors='coerce').astype('Int64')
                if 'gt_chromStart' in df_chunk.columns:
                    df_chunk['gt_chromStart'] = pd.to_numeric(df_chunk['gt_chromStart'], errors='coerce').astype('Int64')

                table = pa.Table.from_pandas(df_chunk, preserve_index=False)

                if writer is None:
                    # Define explicit schema once on the first chunk
                    explicit_schema = table.schema
                    writer = pq.ParquetWriter(tmp_output_fdr_file, explicit_schema)
                else:
                    # Cast subsequent chunks to the explicit schema established by the first chunk
                    table = table.cast(explicit_schema)

                writer.write_table(table)

                if (i + 1) % 100 == 0:
                    print(f"Written {i+1} chunks...", end='\r')

            print(f"\nFinished writing FDR Parquet file.")
        except Exception as e:
            write_error = e
        finally:
            if writer is not None:
                writer.close()

        if write_error is not None:
            # Cleanup must never raise: a failure here would replace the real
            # diagnosis with an unrelated traceback.
            if os.path.isfile(tmp_output_fdr_file):
                try:
                    os.remove(tmp_output_fdr_file)
                except OSError:
                    pass
            sys.stderr.write(
                f"Error writing output FDR file: {write_error}\n"
                "No output was written; the destination is unchanged.\n")
            sys.exit(1)

        os.replace(tmp_output_fdr_file, args.output_fdr_file)

        if do_compare:
            if cmp_n_comparable == 0:
                verdict = "INFO (no comparable rows)"
                cmp_min_diff_fmt = 0.0
            elif cmp_min_diff < 0.0:
                verdict = "VIOLATION (new FDR smaller than reference; check the BH denominator)"
                cmp_min_diff_fmt = cmp_min_diff
            elif cmp_max_abs_diff == 0.0:
                verdict = "EQUAL (identical pools and denominator)"
                cmp_min_diff_fmt = cmp_min_diff
            else:
                verdict = "DIRECTIONAL-OK (new pool is a subset; values moved upward only)"
                cmp_min_diff_fmt = cmp_min_diff

            if cmp_n_comparable == 0 and cmp_min_diff == np.inf:
                cmp_min_diff_fmt = 0.0

            print("-" * 60)
            print(f"FDR comparison: {args.fdr_column} vs {args.compare_fdr_column}")
            print(f"  comparable rows      : {cmp_n_comparable}")
            print(f"  max|diff|            : {cmp_max_abs_diff:.6e}")
            print(f"  min diff (signed)    : {cmp_min_diff_fmt:.6e}")
            print(f"  rows left the pool   : {cmp_n_new_null_ref_present}")
            print(f"  rows entered the pool: {cmp_n_new_present_ref_null}")
            print(f"  VERDICT: {verdict}")
            print("-" * 60)

            if verdict.startswith("VIOLATION"):
                sys.exit(1)

    # Histogram
    try:
        plt.figure(figsize=(10, 6))
        bins = np.linspace(0, 1, 101)
        plt.bar(bins[:-1], final_hist, width=0.01, align='edge', color='skyblue', edgecolor='black')
        plt.title("P-value Histogram")
        plt.xlabel("P-value")
        plt.ylabel("Count")
        output_image = "p_value_histogram.png"
        plt.savefig(output_image)
        print(f"Histogram saved to {output_image}")
    except Exception as e:
        print(f"Error plotting histogram: {e}")

    # Stacked Proportional Saliency Chart
    if all_top_ig:
        print("\nGenerating Stacked Proportional Saliency Chart...")
        try:
            combined_ig_df = pd.concat(all_top_ig, ignore_index=True)
            # Get the true global top 50 (lowest p-value at index 0)
            top_50_ig = combined_ig_df.sort_values('p-value').head(50).copy()

            # Create labels combining CpG and Gene IDs
            top_50_ig['locus_pair'] = top_50_ig['mt_id'] + " - " + top_50_ig['gt_id']

            # Extract the _ig columns
            ig_columns = [col for col in top_50_ig.columns if col.endswith('_ig')]

            # Calculate proportions
            # Sum across all _ig columns for each row
            top_50_ig['ig_sum'] = top_50_ig[ig_columns].sum(axis=1)

            # Divide each _ig column by the sum to get the proportion
            for col in ig_columns:
                top_50_ig[f"{col}_prop"] = top_50_ig[col] / top_50_ig['ig_sum']

            prop_columns = [f"{col}_prop" for col in ig_columns]

            # Prepare data for plotting
            plot_df = top_50_ig.set_index('locus_pair')[prop_columns]

            # Reorder rows so the #1 hit (lowest p-value) is at the top
            plot_df = plot_df.iloc[::-1]

            # Setup colors: distinct for mt_ig, rest use pastels
            import matplotlib as mpl
            colors = []
            cmap = mpl.colormaps['Pastel1']
            covar_idx = 0
            for col in prop_columns:
                if col == 'mt_ig_prop':
                    colors.append('darkblue') # Prominent color for methylation
                else:
                    colors.append(cmap(covar_idx % 9))
                    covar_idx += 1

            # Create the plot
            # Need to get a Figure and Axes to plot properly with pandas
            fig, ax = plt.subplots(figsize=(12, 10))
            plot_df.plot.barh(stacked=True, color=colors, width=0.8, ax=ax)

            plt.title("Stacked Proportional Saliency Profile (Top 50 Hits)")
            plt.xlabel("Proportion of Total Saliency")
            plt.ylabel("Locus Pair (CpG - Gene)")

            # Clean up legend labels
            handles, labels = ax.get_legend_handles_labels()
            cleaned_labels = [label.replace('_ig_prop', '') for label in labels]
            ax.legend(handles, cleaned_labels, title="Features", loc='center left', bbox_to_anchor=(1.0, 0.5))

            plt.tight_layout()

            output_image = "saliency_profile_top50.png"
            plt.savefig(output_image)
            plt.close()
            print(f"Saliency profile saved to {output_image}")

        except Exception as e:
            print(f"Error plotting saliency profile: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
