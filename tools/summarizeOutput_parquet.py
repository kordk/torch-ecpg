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
import gseapy
import mygene
import pyranges as pr
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import urllib.request
import zipfile
import gzip
import shutil

def clean_and_translate_ensembl_ids(ensembl_ids):
    """
    Strips version suffixes from Ensembl IDs and translates them to HGNC gene symbols using mygene.
    Logs unmapped genes and returns a list of mapped symbols and the count of unmapped ones.
    """
    if not ensembl_ids:
        return [], 0

    # Clean IDs: strip everything after '.'
    cleaned_ids = [str(gene_id).split('.')[0] for gene_id in ensembl_ids]

    print(f"Translating {len(cleaned_ids)} cleaned Ensembl IDs to Gene Symbols...")

    # Initialize mygene info
    mg = mygene.MyGeneInfo()

    # Query mygene to translate
    try:
        # We expect cleaned_ids to be Ensembl gene IDs
        results = mg.querymany(cleaned_ids, scopes='ensembl.gene', fields='symbol', species='human', verbose=False)
    except Exception as e:
        print(f"Error querying mygene: {e}")
        return [], len(cleaned_ids)

    mapped_symbols = []
    unmapped_ids = []

    for res in results:
        if 'symbol' in res:
            mapped_symbols.append(res['symbol'])
        else:
            unmapped_ids.append(res['query'])

    # Log unmapped IDs
    if unmapped_ids:
        # unique just in case
        unmapped_ids = list(set(unmapped_ids))
        print(f"  Warning: {len(unmapped_ids)} gene IDs could not be mapped to an HGNC symbol.")
        # log up to first 20 for brevity, or all if you prefer
        print(f"  Unmapped examples: {', '.join(unmapped_ids[:20])}" + ("..." if len(unmapped_ids) > 20 else ""))

    # Return unique mapped symbols
    return list(set(mapped_symbols)), len(unmapped_ids)


# Worker function must be top-level for multiprocessing
def download_encode_files(target_dir):
    """
    Downloads standard hg19 ENCODE BED files for ChromHMM, H3K27ac, and DNase I to the target directory.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    urls = {
        "ChromHMM": "http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHmm/wgEncodeBroadHmmGm12878HMM.bed.gz",
        "H3K27ac": "http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k27acStdPk.broadPeak.gz",
        "DNase": "http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgDnaseUniform/wgEncodeAwgDnaseUwdukeGm12878UniPk.narrowPeak.gz"
    }

    files_present = True
    for key, url in urls.items():
        filename = url.split('/')[-1]
        filepath = os.path.join(target_dir, filename)
        unzipped_path = filepath[:-3] if filepath.endswith('.gz') else filepath

        if not os.path.exists(unzipped_path):
            files_present = False
            print(f"Downloading {key} track from {url} ...")
            try:
                urllib.request.urlretrieve(url, filepath)
                if filepath.endswith('.gz'):
                    print(f"Extracting {filepath} ...")
                    with gzip.open(filepath, 'rb') as f_in:
                        with open(unzipped_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(filepath)
            except Exception as e:
                print(f"Error downloading or extracting {url}: {e}")
                print(f"Please manually download it and place it in '{target_dir}'.")
                sys.exit(1)

    return files_present

def run_fisher_exact(hits_pr, background_pr, encode_pr):
    """
    Runs Fisher's exact test for overlap of hits with ENCODE track vs background.
    """
    # Number of hits overlapping the state
    A = len(hits_pr.overlap(encode_pr).df.drop_duplicates(subset=['Chromosome', 'Start', 'End'])) if not hits_pr.overlap(encode_pr).df.empty else 0
    # Number of hits NOT overlapping the state
    B = len(hits_pr) - A

    # Overlap of the entire background (which includes hits, so we subtract hits overlap later)
    bg_overlap = len(background_pr.overlap(encode_pr).df.drop_duplicates(subset=['Chromosome', 'Start', 'End'])) if not background_pr.overlap(encode_pr).df.empty else 0

    # Non-significant background INSIDE annotation = (all bg inside) - (hits inside)
    C = bg_overlap - A

    # Non-significant background OUTSIDE annotation = (total bg - total hits) - C
    total_non_hits = len(background_pr) - len(hits_pr)
    D = total_non_hits - C

    # Sanity checks
    A = max(0, A)
    B = max(0, B)
    C = max(0, C)
    D = max(0, D)

    try:
        oddsratio, pvalue = stats.fisher_exact([[A, B], [C, D]])
    except ValueError as e:
        # e.g. if one array is empty or 0 dimensions
        oddsratio, pvalue = np.nan, np.nan

    try:
        fe = (A / (A + B)) / (C / (C + D))
    except ZeroDivisionError:
        fe = np.nan

    return A, fe, pvalue


def process_chunk(chunk, sample_prob, df):
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
    if 'precise_mt_p' in chunk.columns:
        chunk = chunk.dropna(subset=['precise_mt_p'])
        p_values = chunk['precise_mt_p'].astype(np.float64).values
    else:
        if not t_col:
            raise ValueError(f"Error: precise_mt_p and t-statistic column missing in chunk. Available columns: {list(chunk.columns)}")
        chunk = chunk.dropna(subset=[t_col])
        t_stats = chunk[t_col].astype(np.float64).values
        p_values = stats.t.sf(np.abs(t_stats), np.float64(df)) * 2.0

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

    return unique_gt, unique_mt, row_count, hist_counts, p_values, region_codes, region_uniques, top_hits_df, top_ig_df


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

    # ENCODE Enrichment Options
    parser.add_argument("--encode-enrichment", action="store_true", help="Run ENCODE enrichment analysis using Fisher's Exact Test.")
    parser.add_argument("--encode-bed-dir", default="encode_beds", help="Directory containing ENCODE BED files (will auto-download if missing).")
    parser.add_argument("--background-bed", help="Path to the background universe BED file (e.g. annoEPIC.hg19.bed6). Required if --encode-enrichment is set.")

    args = parser.parse_args()

    if args.encode_enrichment and not args.background_bed:
        parser.error("--encode-enrichment requires --background-bed to be specified.")

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

    all_top_hits = []
    all_top_ig = []
    # Using lists to accumulate codes before stacking
    region_code_arrays = []
    global_uniques = []
    global_uniques_map = {} # map string -> global int code

    for res in results:
        try:
            gt_set, mt_set, count, hist, p_vals, codes, uniques, top_hits_df, top_ig_df = res.get()
            final_gt.update(gt_set)
            final_mt.update(mt_set)
            total_pairs += count
            final_hist += hist
            if len(p_vals) > 0:
                all_p_values.append(p_vals)

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
            print(f"Error retrieving result from worker: {e}")

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

    # Variables for ENCODE Enrichment Analysis
    significant_cpgs = set()
    significant_cpgs_by_region = {}
    significant_genes_by_region = {}

    # Collect Significant Genes and CpGs for Enrichment (if requested and possible)
    if p_max_fdr >= 0:
        print(f"\nCollecting significant features (FDR < 0.05) for enrichment...")
        try:
            for i, batch in enumerate(parquet_file.iter_batches(batch_size=args.chunk_size)):
                df_chunk = batch.to_pandas()

                # Use same logic to get p-values
                if not using_fallback:
                    chunk_p_vals = df_chunk['precise_mt_p'].values
                else:
                    t_col = None
                    if 'mt_t' in df_chunk.columns:
                        t_col = 'mt_t'
                    elif 't' in df_chunk.columns:
                        t_col = 't'
                    t_stats = df_chunk[t_col].values
                    chunk_p_vals = stats.t.sf(np.abs(t_stats), np.float64(args.df)) * 2.0

                # Filter for significant
                sig_mask = chunk_p_vals <= p_max_fdr
                if sig_mask.any():
                    sig_df = df_chunk[sig_mask]

                    if has_region and 'region' in sig_df.columns:
                        for region, group in sig_df.groupby('region'):
                            if 'gt_id' in sig_df.columns:
                                genes = set(group['gt_id'].dropna().unique())
                                if region not in significant_genes_by_region:
                                    significant_genes_by_region[region] = set()
                                significant_genes_by_region[region].update(genes)

                            if args.encode_enrichment:
                                if 'mt_chrom' in group.columns and 'mt_chromStart' in group.columns:
                                    cpgs = set(zip(group['mt_chrom'], group['mt_chromStart']))
                                    if region not in significant_cpgs_by_region:
                                        significant_cpgs_by_region[region] = set()
                                    significant_cpgs_by_region[region].update(cpgs)
                                    significant_cpgs.update(cpgs)
                    else:
                        # If no region column, just collect global CpGs
                        if args.encode_enrichment:
                            if 'mt_chrom' in sig_df.columns and 'mt_chromStart' in sig_df.columns:
                                cpgs = set(zip(sig_df['mt_chrom'], sig_df['mt_chromStart']))
                                significant_cpgs.update(cpgs)

                if (i + 1) % 100 == 0:
                    print(f"Processed {i+1} chunks for significant features...", end='\r')
            print(f"\nFinished collecting significant features.")
        except Exception as e:
            print(f"Error collecting significant features: {e}")

    # Run ENCODE Enrichment Analysis
    if args.encode_enrichment:
        print("\n" + "=" * 40)
        print("Running ENCODE Enrichment Analysis")
        print("=" * 40)

        # Download ENCODE files if necessary
        download_encode_files(args.encode_bed_dir)

        encode_files = {
            "ChromHMM": os.path.join(args.encode_bed_dir, "wgEncodeBroadHmmGm12878HMM.bed"),
            "H3K27ac": os.path.join(args.encode_bed_dir, "wgEncodeBroadHistoneGm12878H3k27acStdPk.broadPeak"),
            "DNase": os.path.join(args.encode_bed_dir, "wgEncodeAwgDnaseUwdukeGm12878UniPk.narrowPeak")
        }

        # Check if all files exist
        missing_files = [f for f in encode_files.values() if not os.path.exists(f)]
        if missing_files:
            print(f"Error: Missing ENCODE files: {missing_files}")
            sys.exit(1)

        print(f"Loading background universe BED: {args.background_bed}...")
        try:
            bg_df = pd.read_csv(args.background_bed, sep='\t', header=None, usecols=[0, 1, 2], names=['Chromosome', 'Start', 'End'], dtype=str)

            # Check for header in the first row
            if not bg_df.empty:
                first_start = bg_df['Start'].iloc[0]
                if not str(first_start).isdigit():
                    print(f"Detected header in background BED file, skipping first row...")
                    bg_df = bg_df.iloc[1:].copy()

            # Cast coordinates to int
            bg_df['Start'] = bg_df['Start'].astype(int)
            bg_df['End'] = bg_df['End'].astype(int)

            # Ensure 'chr' prefix
            if not bg_df['Chromosome'].astype(str).str.startswith('chr').all():
                bg_df['Chromosome'] = 'chr' + bg_df['Chromosome'].astype(str)
            bg_pr = pr.PyRanges(bg_df)
        except Exception as e:
            print(f"Error reading background BED file: {e}")
            sys.exit(1)

        print(f"Loaded {len(bg_pr)} background CpGs.")

        print(f"Processing {len(significant_cpgs)} significant global eCpGs for enrichment...")
        hits_df = pd.DataFrame(list(significant_cpgs), columns=['Chromosome', 'Start'])
        hits_df['End'] = hits_df['Start'] + 1
        # Ensure 'chr' prefix
        if not hits_df['Chromosome'].astype(str).str.startswith('chr').all():
            hits_df['Chromosome'] = 'chr' + hits_df['Chromosome'].astype(str)
        hits_pr = pr.PyRanges(hits_df)

        # Also prepare region-specific hits
        hits_pr_by_region = {}
        for region, cpgs in significant_cpgs_by_region.items():
            r_df = pd.DataFrame(list(cpgs), columns=['Chromosome', 'Start'])
            r_df['End'] = r_df['Start'] + 1
            if not r_df['Chromosome'].astype(str).str.startswith('chr').all():
                r_df['Chromosome'] = 'chr' + r_df['Chromosome'].astype(str)
            hits_pr_by_region[region] = pr.PyRanges(r_df)

        enrichment_results = []

        # ChromHMM Processing (15-state)
        print("Processing ChromHMM (15-state model)...")
        chromhmm_df = pd.read_csv(encode_files['ChromHMM'], sep='\t', header=None, usecols=[0, 1, 2, 3], names=['Chromosome', 'Start', 'End', 'State'])

        # Focus states: Active TSS, Poised Enhancer, Active Enhancer, Heterochromatin, etc.
        # States are usually formatted like "1_Active_Promoter", "2_Weak_Promoter", etc.
        states_of_interest = chromhmm_df['State'].unique()

        for state in states_of_interest:
            state_df = chromhmm_df[chromhmm_df['State'] == state]
            encode_pr = pr.PyRanges(state_df)

            # Global
            A, fe, pval = run_fisher_exact(hits_pr, bg_pr, encode_pr)
            enrichment_results.append({
                'Annotation Track': 'ChromHMM',
                'State/Region': f'Global: {state}',
                'Region_Category': 'Global',
                'State': state,
                'Overlap Count (A)': A,
                'Fold Enrichment': fe,
                'P-value': pval
            })

            # Per Region
            for region, r_pr in hits_pr_by_region.items():
                r_A, r_fe, r_pval = run_fisher_exact(r_pr, bg_pr, encode_pr)
                enrichment_results.append({
                    'Annotation Track': 'ChromHMM',
                    'State/Region': f'{region}: {state}',
                    'Region_Category': region,
                    'State': state,
                    'Overlap Count (A)': r_A,
                    'Fold Enrichment': r_fe,
                    'P-value': r_pval
                })

        # H3K27ac Processing
        print("Processing H3K27ac (Active Enhancers)...")
        h3k27ac_df = pd.read_csv(encode_files['H3K27ac'], sep='\t', header=None, usecols=[0, 1, 2], names=['Chromosome', 'Start', 'End'])
        encode_pr = pr.PyRanges(h3k27ac_df)

        A, fe, pval = run_fisher_exact(hits_pr, bg_pr, encode_pr)
        enrichment_results.append({
            'Annotation Track': 'H3K27ac',
            'State/Region': 'Global',
            'Region_Category': 'Global',
            'State': 'H3K27ac',
            'Overlap Count (A)': A,
            'Fold Enrichment': fe,
            'P-value': pval
        })
        for region, r_pr in hits_pr_by_region.items():
            r_A, r_fe, r_pval = run_fisher_exact(r_pr, bg_pr, encode_pr)
            enrichment_results.append({
                'Annotation Track': 'H3K27ac',
                'State/Region': region,
                'Region_Category': region,
                'State': 'H3K27ac',
                'Overlap Count (A)': r_A,
                'Fold Enrichment': r_fe,
                'P-value': r_pval
            })

        # DNase Processing
        print("Processing DNase I Hypersensitivity (Open Chromatin)...")
        dnase_df = pd.read_csv(encode_files['DNase'], sep='\t', header=None, usecols=[0, 1, 2], names=['Chromosome', 'Start', 'End'])
        encode_pr = pr.PyRanges(dnase_df)

        A, fe, pval = run_fisher_exact(hits_pr, bg_pr, encode_pr)
        enrichment_results.append({
            'Annotation Track': 'DNase I',
            'State/Region': 'Global',
            'Region_Category': 'Global',
            'State': 'DNase I',
            'Overlap Count (A)': A,
            'Fold Enrichment': fe,
            'P-value': pval
        })
        for region, r_pr in hits_pr_by_region.items():
            r_A, r_fe, r_pval = run_fisher_exact(r_pr, bg_pr, encode_pr)
            enrichment_results.append({
                'Annotation Track': 'DNase I',
                'State/Region': region,
                'Region_Category': region,
                'State': 'DNase I',
                'Overlap Count (A)': r_A,
                'Fold Enrichment': r_fe,
                'P-value': r_pval
            })

        # Compile Results
        res_df = pd.DataFrame(enrichment_results)

        if not res_df.empty:
            # Calculate FDR-adjusted P-values
            # Drop NaNs for FDR calculation
            valid_idx = res_df['P-value'].notna()
            if valid_idx.any():
                res_df.loc[valid_idx, 'FDR-adjusted P-value'] = multipletests(res_df.loc[valid_idx, 'P-value'], method='fdr_bh')[1]
            else:
                res_df['FDR-adjusted P-value'] = np.nan
        else:
            res_df = pd.DataFrame(columns=['Annotation Track', 'State/Region', 'Region_Category', 'State', 'Overlap Count (A)', 'Fold Enrichment', 'P-value', 'FDR-adjusted P-value'])

        # Save Summary Table
        plots_dir = "plots"
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        csv_out = os.path.join(plots_dir, "encode_enrichment_results.csv")
        res_df.to_csv(csv_out, index=False)
        print(f"Saved ENCODE Enrichment results to {csv_out}")

        # Visualization: Heatmap plotting Fold Enrichment across ChromHMM states
        print("Generating Heatmap for Fold Enrichment...")
        # Filter for ChromHMM and non-Global for faceting
        heatmap_df = res_df[(res_df['Annotation Track'] == 'ChromHMM') & (res_df['Region_Category'] != 'Global')].copy()

        if not heatmap_df.empty:
            # Pivot table
            pivot_df = heatmap_df.pivot(index="State", columns="Region_Category", values="Fold Enrichment")

            plt.figure(figsize=(10, 8))
            # Log2 transform Fold Enrichment for better visualization centered at 0 (log2(1) = 0)
            log2_fe = np.log2(pivot_df.astype(float).replace(0, np.nan))

            sns.heatmap(log2_fe, cmap='coolwarm', center=0, annot=True, fmt=".2f", cbar_kws={'label': 'Log2(Fold Enrichment)'})
            plt.title('ENCODE ChromHMM Enrichment Across Regions')
            plt.ylabel('ChromHMM State')
            plt.xlabel('eCpG Region')
            plt.tight_layout()

            png_out = os.path.join(plots_dir, "encode_enrichment_heatmap.png")
            plt.savefig(png_out, dpi=300)
            plt.close()
            print(f"Saved Fold Enrichment Heatmap to {png_out}")
        else:
            print("No regional data available for ChromHMM heatmap generation.")

        print("=" * 40 + "\n")

    # Run Enrichment Analysis on Significant Genes
    if significant_genes_by_region:
        enrichment_dir = "enrichment_results"
        if not os.path.exists(enrichment_dir):
            os.makedirs(enrichment_dir)

        print(f"\nRunning functional enrichment analysis in '{enrichment_dir}/'...")
        libraries = ['GO_Biological_Process_2021', 'KEGG_2021_Human', 'WikiPathways_2021_Human']

        for region, genes in significant_genes_by_region.items():
            print(f"\nProcessing region: {region} with {len(genes)} significant Ensembl IDs")

            # Clean and translate
            mapped_symbols, unmapped_count = clean_and_translate_ensembl_ids(list(genes))

            if not mapped_symbols:
                print(f"Skipping enrichment for {region} due to no mapped gene symbols.")
                continue

            print(f"  Successfully mapped {len(mapped_symbols)} gene symbols.")

            # Run enrichr
            for library in libraries:
                print(f"  Running enrichment against {library}...")
                try:
                    # gseapy.enrichr
                    enr = gseapy.enrichr(
                        gene_list=mapped_symbols,
                        gene_sets=library,
                        organism='human',
                        outdir=None,  # Do not auto-save to output directory immediately to allow filtering
                        no_plot=True,
                    )

                    if enr.results is not None and not enr.results.empty:
                        # Filter by Adjusted P-value < 0.05
                        sig_res = enr.results[enr.results['Adjusted P-value'] < 0.05]

                        if not sig_res.empty:
                            # Save to CSV
                            csv_filename = f"{region}_{library}_enrichment.csv".replace(" ", "_").replace("/", "_")
                            csv_path = os.path.join(enrichment_dir, csv_filename)
                            # Keep relevant columns
                            columns_to_save = ['Term', 'Overlap', 'P-value', 'Adjusted P-value', 'Genes']
                            # Filter missing columns just in case
                            columns_to_save = [col for col in columns_to_save if col in sig_res.columns]
                            sig_res[columns_to_save].to_csv(csv_path, index=False)
                            print(f"    Saved {len(sig_res)} significant terms to {csv_filename}")

                            # Visual Summary
                            top_10 = sig_res.head(10).copy()
                            if len(top_10) > 0:
                                try:
                                    # Create Dot Plot for Top 10
                                    plt.figure(figsize=(10, 8))
                                    # gseapy dotplot expects data with index as terms, 'Adjusted P-value', 'Overlap', etc.
                                    # Alternatively, we can use simple matplotlib bar plot since gseapy.plot.dotplot requires specific formats

                                    # Use a simple horizontal bar plot for Adjusted P-value
                                    top_10 = top_10.sort_values('Adjusted P-value', ascending=False)
                                    terms = top_10['Term'].apply(lambda x: (x[:47] + '...') if len(x) > 50 else x)
                                    log_p = -np.log10(top_10['Adjusted P-value'].astype(float))

                                    plt.barh(terms, log_p, color='skyblue', edgecolor='black')
                                    plt.xlabel('-log10(Adjusted P-value)')
                                    plt.title(f"Top Enriched Terms\n{region} - {library}")
                                    plt.tight_layout()

                                    plot_filename = f"{region}_{library}_top10.png".replace(" ", "_").replace("/", "_")
                                    plot_path = os.path.join(enrichment_dir, plot_filename)
                                    plt.savefig(plot_path)
                                    plt.close()
                                except Exception as plot_e:
                                    print(f"    Error plotting {region} {library}: {plot_e}")
                        else:
                            print(f"    No significant terms found (Adjusted P-value < 0.05).")
                    else:
                        print(f"    No enrichment results returned.")
                except Exception as e:
                    print(f"    Error running gseapy for {library}: {e}")

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
