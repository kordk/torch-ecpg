#!/usr/bin/env python3
import argparse
import glob
import multiprocessing
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def natural_keys(text):
    """
    Splits text into a list of integers and strings for natural sorting.
    Example: '1-2' -> ['1', 2], '1-10' -> ['1', 10]
    """
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]


def get_stats(filepath):
    """
    Reads a CSV file and calculates the number of rows (mappings),
    unique genes (gt_id), and unique CpGs (mt_id).

    Assumes the file has a header.
    """
    try:
        # Use pandas to read just the columns we need for counting unique values
        # We need to detect if 'gt_id' and 'mt_id' are in the columns
        # Reading just the header first to be safe
        header = pd.read_csv(filepath, nrows=0)
        usecols = []
        if 'gt_id' in header.columns:
            usecols.append('gt_id')
        if 'mt_id' in header.columns:
            usecols.append('mt_id')

        # If columns are missing, we can't count unique values for them
        # This might happen if the file format is different
        if not usecols:
            # Fallback: Count lines - 1 (header)
            with open(filepath, 'rb') as f:
                mappings = sum(1 for _ in f) - 1
            return mappings, set(), set()

        # Read the file in chunks to be memory efficient
        mappings = 0
        unique_genes = set()
        unique_cpgs = set()

        for chunk in pd.read_csv(filepath, usecols=usecols, chunksize=100000):
            mappings += len(chunk)
            if 'gt_id' in chunk.columns:
                unique_genes.update(chunk['gt_id'].dropna().astype(str))
            if 'mt_id' in chunk.columns:
                unique_cpgs.update(chunk['mt_id'].dropna().astype(str))

        return mappings, unique_genes, unique_cpgs
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return 0, set(), set()


def merge_file_content(input_path, output_handle, is_first_file):
    """
    Appends the content of input_path to output_handle.
    If is_first_file is False, skips the header line.
    """
    with open(input_path, 'rb') as f_in:
        if not is_first_file:
            # Skip the first line (header)
            f_in.readline()
        shutil.copyfileobj(f_in, output_handle)


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple CSV output files into a single CSV efficiently."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing the CSV chunk files."
    )
    parser.add_argument(
        "output_file",
        help="Path to the output merged CSV file."
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="File pattern to match (default: '*.csv')."
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output file format (default: 'csv')."
    )
    parser.add_argument(
        "--compression",
        choices=["snappy", "zstd"],
        default="snappy",
        help="Compression algorithm for parquet format (default: 'snappy')."
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel processes for statistics calculation."
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = args.output_file
    pattern = args.pattern
    out_format = args.format
    compression = args.compression

    if out_format == "csv" and compression != "snappy":
        print("Warning: --compression argument is ignored when --format is csv.", file=sys.stderr)

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Find all files matching the pattern
    search_path = os.path.join(input_dir, pattern)
    files = glob.glob(search_path)

    if not files:
        print(f"Error: No files found matching '{pattern}' in '{input_dir}'.", file=sys.stderr)
        sys.exit(1)

    # Sort files naturally
    files.sort(key=natural_keys)

    print(f"Found {len(files)} files to merge.")
    print(f"Output file: {output_file}")
    print(f"Using {args.processes} processes for statistics.")

    start_time = time.time()

    # Initialize stats containers
    total_mappings = 0
    all_genes = set()
    all_cpgs = set()
    empty_files_count = 0

    # We will run stats calculation in parallel
    # And file merging sequentially in the main thread

    # Create the output directory if it doesn't exist
    output_dir_path = os.path.dirname(output_file)
    if output_dir_path and not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    try:
        with ProcessPoolExecutor(max_workers=args.processes) as executor:
            # Submit stats jobs
            future_to_file = {executor.submit(get_stats, f): f for f in files}

            # Start merging files
            print("Starting merge process...")
            if out_format == "csv":
                with open(output_file, 'wb') as f_out:
                    first_file_written = False
                    for i, filepath in enumerate(files):
                        # Check if file is empty (only header)
                        is_empty = False
                        with open(filepath, 'rb') as f_check:
                            header = f_check.readline()
                            if not f_check.read(1):  # No data after header
                                empty_files_count += 1
                                is_empty = True

                        if not is_empty:
                            merge_file_content(filepath, f_out, not first_file_written)
                            first_file_written = True

                        if (i + 1) % 10 == 0 or (i + 1) == len(files):
                            print(f"Merged {i + 1}/{len(files)} files...", end='\r')
            elif out_format == "parquet":
                writer = None
                for i, filepath in enumerate(files):
                    # Read the CSV chunk
                    df = pd.read_csv(filepath)

                    if df.empty:
                        empty_files_count += 1
                    else:
                        table = pa.Table.from_pandas(df)

                        if writer is None:
                            # Initialize writer with the schema of the first non-empty file
                            writer = pq.ParquetWriter(output_file, table.schema, compression=compression)

                        writer.write_table(table)

                    if (i + 1) % 10 == 0 or (i + 1) == len(files):
                        print(f"Merged {i + 1}/{len(files)} files...", end='\r')

                if writer:
                    writer.close()
                else:
                    print("Warning: All processed files were empty. No parquet file was generated.", file=sys.stderr)

            print(f"\nMerge complete. Waiting for statistics calculation...")

            # Collect stats results as they complete
            completed_count = 0
            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    mappings, genes, cpgs = future.result()
                    total_mappings += mappings
                    all_genes.update(genes)
                    all_cpgs.update(cpgs)
                except Exception as exc:
                    print(f"File {filepath} generated an exception: {exc}")

                completed_count += 1
                if completed_count % 10 == 0 or completed_count == len(files):
                     print(f"Processed stats for {completed_count}/{len(files)} files...", end='\r')

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n\n--- Report ---")
    print(f"Total Mappings (rows): {total_mappings}")
    print(f"Unique Genes (gt_id): {len(all_genes)}")
    print(f"Unique CpGs (mt_id): {len(all_cpgs)}")
    print(f"Empty files skipped: {empty_files_count}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")

    if os.path.exists(output_file):
        file_size_bytes = os.path.getsize(output_file)
        if file_size_bytes >= 1024**3:
            file_size_str = f"{file_size_bytes / (1024**3):.2f} GB"
        elif file_size_bytes >= 1024**2:
            file_size_str = f"{file_size_bytes / (1024**2):.2f} MB"
        elif file_size_bytes >= 1024:
            file_size_str = f"{file_size_bytes / 1024:.2f} KB"
        else:
            file_size_str = f"{file_size_bytes} bytes"

        print(f"Output saved to: {output_file} ({file_size_str})")
    else:
        print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
