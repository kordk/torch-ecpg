#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate p-values for Parquet files using Student's t-distribution with float64 precision."
    )

    parser.add_argument(
        "input_file",
        help="Path to the input Parquet file."
    )
    parser.add_argument(
        "--df",
        type=int,
        required=True,
        help="Number of degrees of freedom."
    )
    parser.add_argument(
        "--output-file",
        help="Path to the output Parquet file. Defaults to <input_filename>_recalc.parquet."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Number of rows to process per chunk. Default is 100,000."
    )

    args = parser.parse_args()

    input_path = args.input_file
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    df = args.df
    if df <= 0:
        print(f"Error: Degrees of freedom ({df}) must be positive.", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output_file:
        output_path = args.output_file
    else:
        base, ext = os.path.splitext(input_path)
        if ext == '':
            output_path = f"{base}_recalc.parquet"
        else:
            output_path = f"{base}_recalc{ext}"

    print(f"Calculating p-values with {df} degrees of freedom.")
    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print(f"Using chunk size {args.chunk_size}.")

    try:
        parquet_file = pq.ParquetFile(input_path)
        writer = None

        # Process the file in batches to save memory
        for batch in parquet_file.iter_batches(batch_size=args.chunk_size):
            # Convert the batch to a pandas DataFrame or dictionary
            df_chunk = batch.to_pandas()

            if 'mt_t' not in df_chunk.columns:
                raise ValueError("Input Parquet file must contain 'mt_t' column.")

            # Ensure float64 precision for t-statistics
            t_stats = df_chunk['mt_t'].astype(np.float64)

            # Calculate two-sided p-value using Survival Function (SF) of Student's t
            # p = 2 * sf(|t|, df)
            p_values = t.sf(np.abs(t_stats), df) * 2

            # Add the new column
            df_chunk['precise_mt_p'] = p_values

            # Convert back to an Arrow Table
            table = pa.Table.from_pandas(df_chunk, preserve_index=False)

            # Initialize the writer with the schema of the first chunk
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)

            writer.write_table(table)

        if writer is not None:
            writer.close()

        print("Processing complete.")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        if 'writer' in locals() and writer is not None:
            writer.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
