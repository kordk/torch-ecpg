import argparse
import multiprocessing
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Worker function must be top-level for multiprocessing
def process_chunk(chunk, sample_prob):
    """
    Process a chunk of the dataframe.
    Returns:
        - set of unique gt_ids
        - set of unique mt_ids
        - count of rows
        - histogram counts (100 bins for p-values 0-1)
        - sampled dataframe (based on sample_prob)
    """
    # Unique counts
    # Use dropna to avoid counting NaNs if any
    unique_gt = set(chunk['gt_id'].dropna().unique()) if 'gt_id' in chunk.columns else set()
    unique_mt = set(chunk['mt_id'].dropna().unique()) if 'mt_id' in chunk.columns else set()
    row_count = len(chunk)

    # Histogram
    # We look for a column ending in '_p' or exactly 'p'
    p_col = None
    for col in chunk.columns:
        if col.endswith('_p') or col == 'p':
            p_col = col
            break

    if p_col:
        # Drop NaNs for histogram
        p_values = chunk[p_col].dropna().values
        # Ensure values are within [0, 1] for histogram logic, though usually they are.
        # np.histogram handles range.
        hist_counts, _ = np.histogram(p_values, bins=100, range=(0, 1))
    else:
        hist_counts = np.zeros(100, dtype=int)
        p_values = np.array([])

    # Sampling
    if sample_prob >= 1.0:
        sample = chunk
    elif sample_prob > 0:
        # Vectorized sampling: generate random numbers and filter
        # Use a stable random state if reproducibility per chunk is needed,
        # but for random sampling across file, simple rand is fine.
        mask = np.random.rand(len(chunk)) < sample_prob
        sample = chunk[mask]
    else:
        sample = pd.DataFrame()

    # Extract only p-values for FDR threshold discovery to save memory
    p_values_array = np.array([], dtype=np.float64)
    if p_col:
        p_values_array = chunk[p_col].dropna().astype(np.float64).values

    return unique_gt, unique_mt, row_count, hist_counts, p_values_array

def estimate_lines(filename):
    """Estimate or count lines in the file."""
    # Fast line count using wc -l
    if os.name == 'posix':
        try:
            import subprocess
            # check_output returns bytes
            out = subprocess.check_output(['wc', '-l', filename])
            return int(out.decode('utf-8').split()[0])
        except Exception:
            pass

    # Fallback: estimate based on file size
    try:
        file_size = os.path.getsize(filename)
        if file_size == 0:
            return 0
        with open(filename, 'rb') as f:
            # Read first line (header)
            f.readline()
            # Read second line to estimate bytes per row
            line = f.readline()
            if not line:
                return 0
            bytes_per_row = len(line)

        if bytes_per_row > 0:
            return int(file_size / bytes_per_row)
    except Exception:
        pass

    return 0

def main():
    description_text = """
Summarize tecpg output CSV.

This script processes large output CSV files from tecpg in a memory-efficient
manner by reading the data in chunks and using multiprocessing.

Outputs and Metrics Calculated:
  - Total mapping pairs (eCpGs): The total number of valid rows processed.
  - Unique genes: The total number of unique gene IDs (gt_id) found.
  - Unique CpGs: The total number of unique CpG site IDs (mt_id) found.
  - Genomic Inflation Factor (lambda): An estimate of test statistic inflation
    calculated using a reservoir sampling approach (~1 million rows).
    A lambda value significantly greater than 1.0 (e.g., > 1.1) may indicate
    population stratification or other systematic biases in the data.
  - P-value Histogram: A histogram image (p_value_histogram.png) plotting the
    distribution of p-values across all processed chunks, grouped into 100 bins.
"""

    parser = argparse.ArgumentParser(
        description=description_text,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--main-file", required=True, help="Path to the main filtered output CSV file.")
    parser.add_argument("--reservoir-file", required=True, help="Path to the diagnostic reservoir CSV file.")
    parser.add_argument("--total-tests", type=int, required=True, help="Total number of tests performed (e.g. 13462597186) for FDR adjustment.")
    parser.add_argument("--df", type=float, required=True, help="Degrees of freedom for t-statistic to p-value conversion.")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Rows per chunk for processing.")
    parser.add_argument("--cores", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="Number of cores to use.")
    args = parser.parse_args()

    main_file = args.main_file
    reservoir_file = args.reservoir_file

    if not os.path.exists(main_file):
        print(f"Error: Main file {main_file} not found.")
        sys.exit(1)
    if not os.path.exists(reservoir_file):
        print(f"Error: Reservoir file {reservoir_file} not found.")
        sys.exit(1)

    print(f"Analyzing main file {main_file} and reservoir file {reservoir_file}...")

    # Estimate total rows for sampling
    total_rows_est = estimate_lines(main_file)
    # Adjust for header
    total_rows_est = max(0, total_rows_est - 1)

    target_sample = 1_000_000
    if total_rows_est > 0:
        # Add 20% buffer
        sample_prob = (target_sample * 1.2) / total_rows_est
        if sample_prob > 1.0: sample_prob = 1.0
    else:
        # If estimation fails or file is empty, try to read all (or sample 100% until we hit limit?)
        # Let's assume 1.0 if we can't estimate, effectively reading all.
        sample_prob = 1.0

    print(f"Estimated rows: {total_rows_est}. Sampling probability: {sample_prob:.4f}")

    pool = multiprocessing.Pool(processes=args.cores)
    results = []

    # Read CSV in chunks
    # We assume 'gt_id', 'mt_id' are present.
    try:
        reader = pd.read_csv(main_file, chunksize=args.chunk_size)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    print("Processing chunks...")
    try:
        for i, chunk in enumerate(reader):
            res = pool.apply_async(process_chunk, (chunk, sample_prob))
            results.append(res)
            # Periodic status update
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
        # Find t-stat column
        t_col = None
        for col in res_df.columns:
            if col.endswith('_t') or col == 't':
                t_col = col
                break

        if t_col:
            # Convert t-statistics to p-values using float64
            t_stats = res_df[t_col].dropna().astype(np.float64).values
            df_val = np.float64(args.df)

            # 2-tailed p-value from t-stat: 2 * sf(abs(t))
            res_p_values = stats.t.sf(np.abs(t_stats), df_val) * 2.0

            # Lambda calculation
            # Use isf for more accurate conversion of small p-values to chi2
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

            # Convert to -log10, add small epsilon to avoid log(0) if any p_value exactly 0
            eps = np.finfo(np.float64).tiny
            obs_log_p = -np.log10(np.clip(sorted_p, eps, 1.0))
            exp_log_p = -np.log10(np.clip(expected_p, eps, 1.0))

            plt.figure(figsize=(8, 8))
            plt.scatter(exp_log_p, obs_log_p, c='black', s=5, alpha=0.5, rasterized=True)

            # Add y=x line
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
    if all_p_values:
        combined_p_values = np.concatenate(all_p_values)
        total_p_count = len(combined_p_values)
        print(f"Sorting {total_p_count} p-values from main file...")
        combined_p_values.sort()

        # BH threshold calculation
        total_tests = args.total_tests

        # Optimize threshold discovery by finding the maximum index i where p(i) <= (i / total_tests) * 0.05
        ranks = np.arange(1, total_p_count + 1)
        bh_limits = (ranks / total_tests) * 0.05

        # valid_mask is True where p_value <= bh_limit
        valid_mask = combined_p_values <= bh_limits

        # We find the *last* index where valid_mask is True. Since p-values are sorted ascending
        # and limit is increasing, this is standard BH procedure.
        # np.nonzero(valid_mask)[0] returns indices of True values. We take the max if it exists.
        valid_indices = np.nonzero(valid_mask)[0]

        if len(valid_indices) > 0:
            max_idx = valid_indices[-1]
            p_max_fdr = combined_p_values[max_idx]
            sig_count = max_idx + 1 # 0-indexed
            print(f"FDR < 0.05 Threshold found!")
            print(f"Maximum P-value satisfying FDR (p_max_fdr): {p_max_fdr:.3e}")
            print(f"Number of pairs remaining significant at FDR < 0.05: {sig_count:,}")
        else:
            print("No pairs remain significant at FDR < 0.05.")

        del combined_p_values
    else:
        print("No p-values found in the main file for FDR adjustment.")

    # Histogram
    try:
        plt.figure(figsize=(10, 6))
        bins = np.linspace(0, 1, 101)
        # Plot bars
        # bins[:-1] are left edges. width=0.01.
        plt.bar(bins[:-1], final_hist, width=0.01, align='edge', color='skyblue', edgecolor='black')
        plt.title("P-value Histogram")
        plt.xlabel("P-value")
        plt.ylabel("Count")
        # Save
        output_image = "p_value_histogram.png"
        plt.savefig(output_image)
        print(f"Histogram saved to {output_image}")
    except Exception as e:
        print(f"Error plotting histogram: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
