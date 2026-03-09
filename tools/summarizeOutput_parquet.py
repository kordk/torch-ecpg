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

# Worker function must be top-level for multiprocessing
def process_chunk(chunk, sample_prob, df):
    """
    Process a chunk of the dataframe (passed as pandas DataFrame).
    Returns:
        - set of unique gt_ids
        - set of unique mt_ids
        - count of rows
        - histogram counts (100 bins for p-values 0-1)
        - extracted p-values
    """
    # Unique counts
    unique_gt = set(chunk['gt_id'].dropna().unique()) if 'gt_id' in chunk.columns else set()
    unique_mt = set(chunk['mt_id'].dropna().unique()) if 'mt_id' in chunk.columns else set()
    row_count = len(chunk)

    # Use high-precision p-values if available
    if 'precise_mt_p' in chunk.columns:
        p_values = chunk['precise_mt_p'].dropna().astype(np.float64).values
    else:
        # Calculate high-precision p-values from t-statistics as fallback
        t_col = None
        for col in chunk.columns:
            if col.endswith('_t') or col == 't':
                t_col = col
                break

        if not t_col:
            raise ValueError(f"Error: precise_mt_p and t-statistic column missing in chunk. Available columns: {list(chunk.columns)}")

        # Log once per worker ideally, but for now we just compute it
        t_stats = chunk[t_col].dropna().astype(np.float64).values
        p_values = stats.t.sf(np.abs(t_stats), np.float64(df)) * 2.0

    # Ensure values are within [0, 1] for histogram logic
    hist_counts, _ = np.histogram(p_values, bins=100, range=(0, 1))

    return unique_gt, unique_mt, row_count, hist_counts, p_values


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

    fdr_group = parser.add_mutually_exclusive_group()
    fdr_group.add_argument("--calculate-fdr", action="store_true", help="Calculate and append an estimated FDR (`fdr_est`) column.")
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

    print(f"Analyzing main file {main_file} and reservoir file {reservoir_file}...")

    # For Parquet, we can quickly get row count if metadata is intact, but we'll just process batches.
    parquet_file = pq.ParquetFile(main_file)
    total_rows = parquet_file.metadata.num_rows

    # Check schema for fallback logging
    schema_cols = parquet_file.schema.names
    using_fallback = 'precise_mt_p' not in schema_cols
    if using_fallback:
        print(f"Warning: 'precise_mt_p' column missing in {main_file}. Falling back to t-statistic calculation.")
        print(f"Available columns: {schema_cols}")

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
            res = pool.apply_async(process_chunk, (df_chunk, sample_prob, args.df))
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

    for res in results:
        try:
            gt_set, mt_set, count, hist, p_vals = res.get()
            final_gt.update(gt_set)
            final_mt.update(mt_set)
            total_pairs += count
            final_hist += hist
            if len(p_vals) > 0:
                all_p_values.append(p_vals)
        except Exception as e:
            print(f"Error retrieving result from worker: {e}")

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
        for col in res_df.columns:
            if col.endswith('_t') or col == 't':
                t_col = col
                break

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
    fdr_map = None  # To hold p-value -> fdr mapping if needed

    if all_p_values:
        combined_p_values = np.concatenate(all_p_values)
        total_p_count = len(combined_p_values)
        print(f"Sorting {total_p_count} p-values from main file...")

        # Sort indices for FDR calculation mapping
        sorted_indices = np.argsort(combined_p_values)
        sorted_p_values = combined_p_values[sorted_indices]

        total_tests = args.total_tests
        ranks = np.arange(1, total_p_count + 1)
        bh_limits = (ranks / total_tests) * 0.05

        valid_mask = sorted_p_values <= bh_limits
        valid_indices = np.nonzero(valid_mask)[0]

        if len(valid_indices) > 0:
            max_idx = valid_indices[-1]
            p_max_fdr = sorted_p_values[max_idx]
            sig_count = max_idx + 1 # 0-indexed
            print(f"FDR < 0.05 Threshold found!")
            print(f"Maximum P-value satisfying FDR (p_max_fdr): {p_max_fdr:.3e}")
            print(f"Number of pairs remaining significant at FDR < 0.05: {sig_count:,}")
        else:
            print("No pairs remain significant at FDR < 0.05.")

        if args.output_fdr_file and args.calculate_fdr:
            print("\nEstimating FDR values...")
            print("Note: The calculated FDR values are estimates (`fdr_est`) because they rely on the `total_tests` parameter and assume the provided dataset contains the top most significant results.")

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

        del combined_p_values
        del sorted_indices
        del sorted_p_values
    else:
        print("No p-values found in the main file for FDR adjustment.")

    # Write FDR Output
    if args.output_fdr_file:
        print(f"\nWriting output FDR Parquet file to: {args.output_fdr_file}")
        writer = None
        try:
            for i, batch in enumerate(parquet_file.iter_batches(batch_size=args.chunk_size)):
                df_chunk = batch.to_pandas()

                # Retrieve or calculate p-values for this chunk
                if not using_fallback:
                    chunk_p_vals = df_chunk['precise_mt_p'].values
                else:
                    t_col = None
                    for col in df_chunk.columns:
                        if col.endswith('_t') or col == 't':
                            t_col = col
                            break
                    t_stats = df_chunk[t_col].values
                    chunk_p_vals = stats.t.sf(np.abs(t_stats), np.float64(args.df)) * 2.0

                if args.assign_fdr_passfail:
                    df_chunk['is_significant'] = chunk_p_vals <= p_max_fdr
                elif args.calculate_fdr and fdr_map is not None:
                    # Map the FDR values using exact p-values
                    # Since floats can be tricky, we map using the exact float values computed
                    # Alternatively, vectorizing the lookup with pandas map:
                    df_chunk['fdr_est'] = pd.Series(chunk_p_vals).map(fdr_map).values
                    # If any didn't map (shouldn't happen), fill with 1.0
                    df_chunk['fdr_est'] = df_chunk['fdr_est'].fillna(1.0)

                table = pa.Table.from_pandas(df_chunk, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(args.output_fdr_file, table.schema)

                writer.write_table(table)

                if (i + 1) % 100 == 0:
                    print(f"Written {i+1} chunks...", end='\r')

            print(f"\nFinished writing FDR Parquet file.")
        except Exception as e:
            print(f"Error writing output FDR file: {e}")
        finally:
            if writer is not None:
                writer.close()

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

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
