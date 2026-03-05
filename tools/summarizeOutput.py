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

    # Return only necessary columns for lambda calculation to save memory during transfer
    if not sample.empty and p_col:
        # We only need p-value for lambda.
        # But maybe user wants other columns?
        # "checking if the median chi2 statistic is much higher than expected" implies only p-values needed.
        # But to be safe, let's keep all columns or at least the p-value column.
        # Keeping all columns allows for debugging if needed.
        # Given 1M rows, 10 columns -> 10M cells -> 100MB. Acceptable.
        pass

    return unique_gt, unique_mt, row_count, hist_counts, sample

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
    parser.add_argument("input_file", help="Path to the input CSV file.")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Rows per chunk for processing.")
    parser.add_argument("--cores", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="Number of cores to use.")
    args = parser.parse_args()

    input_file = args.input_file
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        sys.exit(1)

    print(f"Analyzing {input_file}...")

    # Estimate total rows for sampling
    total_rows_est = estimate_lines(input_file)
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
        reader = pd.read_csv(input_file, chunksize=args.chunk_size)
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
    sampled_dfs = []

    for res in results:
        try:
            gt_set, mt_set, count, hist, sample_df = res.get()
            final_gt.update(gt_set)
            final_mt.update(mt_set)
            total_pairs += count
            final_hist += hist
            if not sample_df.empty:
                sampled_dfs.append(sample_df)
        except Exception as e:
            print(f"Error retrieving result from worker: {e}")

    # Summary Stats
    print("-" * 30)
    print("Summary of Results")
    print("-" * 30)
    print(f"Total mapping pairs (eCpGs): {total_pairs}")
    print(f"Unique genes: {len(final_gt)}")
    print(f"Unique CpGs: {len(final_mt)}")

    # Genomic Inflation Factor
    if sampled_dfs:
        full_sample = pd.concat(sampled_dfs)
        # Downsample to exactly target_sample if larger
        if len(full_sample) > target_sample:
            full_sample = full_sample.sample(n=target_sample, random_state=42)

        # Find p-value column
        p_col = None
        for col in full_sample.columns:
            if col.endswith('_p') or col == 'p':
                p_col = col
                break

        if p_col:
            p_values = full_sample[p_col].dropna()

            # Convert to numpy array
            p_values = p_values.values

            # Calculate chi2 from p-values using Inverse Survival Function (isf)
            # This is more accurate for small p-values than ppf(1-p)
            chi2_obs = stats.chi2.isf(p_values, 1)

            # Handle potential infinities if p=0.
            # Inf is valid for median calculation (it's just a very large number).
            # But if all are inf, median is inf.

            median_chi2_obs = np.median(chi2_obs)

            # Expected median chi2 for 1 df
            expected_median_chi2 = stats.chi2.ppf(0.5, 1) # ~0.4549

            lambda_gc = median_chi2_obs / expected_median_chi2

            print(f"Genomic Inflation Factor (lambda): {lambda_gc:.4f}")
            if lambda_gc > 1.1:
                print("WARNING: lambda > 1.1, your model is inflated.")
        else:
            print("Could not find p-value column for lambda calculation.")
    else:
        print("No data sampled for lambda calculation.")

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
