#!/usr/bin/env python3
import argparse
import multiprocessing
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import t


def process_chunk(chunk, df):
    """
    Process a chunk of the dataframe to recalculate p-values.

    Args:
        chunk (pd.DataFrame): The dataframe chunk.
        df (int): Degrees of freedom.

    Returns:
        pd.DataFrame: The processed chunk with updated p-values.
    """
    if 'mt_t' not in chunk.columns:
        raise ValueError("Input CSV must contain 'mt_t' column.")

    # Ensure float64 precision for t-statistics
    t_stats = chunk['mt_t'].astype(np.float64)

    # Calculate two-sided p-value using Survival Function (SF) of Student's t
    # p = 2 * sf(|t|, df)
    # sf is equivalent to 1 - cdf, but more precise for small p-values
    p_values = t.sf(np.abs(t_stats), df) * 2

    chunk['mt_p'] = p_values
    return chunk


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate p-values using Student's t-distribution with float64 precision."
    )

    parser.add_argument(
        "input_file",
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        required=True,
        help="Number of patients (samples)."
    )
    parser.add_argument(
        "--n-covariates",
        type=int,
        required=True,
        help="Number of covariates."
    )
    parser.add_argument(
        "--output-file",
        help="Path to the output CSV file. Defaults to <input_filename>_recalc.csv."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Number of rows to process per chunk. Default is 100,000."
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel processes to use. Default is number of CPUs."
    )

    args = parser.parse_args()

    input_path = args.input_file
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # Calculate degrees of freedom
    # df = n_patients - n_covariates - 2 (intercept + methylation variable)
    df = args.n_patients - args.n_covariates - 2

    if df <= 0:
        print(
            f"Error: Degrees of freedom ({df}) must be positive. Check n-patients and n-covariates.",
            file=sys.stderr
        )
        sys.exit(1)

    print(f"Calculating p-values with {df} degrees of freedom.")

    # Determine output path
    if args.output_file:
        output_path = args.output_file
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_recalc{ext}"

    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print(f"Using {args.processes} processes with chunk size {args.chunk_size}.")

    try:
        # Initialize the output file (overwrite if exists)
        write_header = True

        # Create a partial function with fixed df
        worker = partial(process_chunk, df=df)

        # Use multiprocessing pool
        # Note: chunksize in map/imap refers to how many items from the iterable
        # are sent to a worker process at a time. Since our iterable yields large
        # DataFrames (chunks), we want chunksize=1 for imap to process one DataFrame at a time.
        with multiprocessing.Pool(processes=args.processes) as pool:
            # Create an iterator for chunks
            chunks = pd.read_csv(input_path, chunksize=args.chunk_size)

            # Process chunks in parallel using imap to maintain order
            for processed_chunk in pool.imap(worker, chunks):
                mode = 'w' if write_header else 'a'
                processed_chunk.to_csv(output_path, mode=mode, header=write_header, index=False)
                write_header = False

        print("Processing complete.")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    main()
