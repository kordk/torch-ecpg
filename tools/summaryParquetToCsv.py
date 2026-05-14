#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="summaryParquetToCsv.py - Convert a Parquet file to a CSV file in chunks")
    parser.add_argument("-i", "--input-file", required=True, help="Path to the input Parquet file")
    parser.add_argument("-o", "--output-file", required=True, help="Path to the output CSV file")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Number of rows to process per chunk. Default is 100,000.")
    parser.add_argument("-D", "--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(message)s')

    input_file = args.input_file
    output_file = args.output_file
    chunk_size = args.chunk_size

    if not os.path.exists(input_file):
        logger.error(f"[MAIN] Input file not found: {input_file}")
        sys.exit(1)

    logger.info(f"[MAIN] Input Parquet file: {input_file}")
    logger.info(f"[MAIN] Output CSV file: {output_file}")
    logger.info(f"[MAIN] Chunk size: {chunk_size}")

    try:
        parquet_file = pq.ParquetFile(input_file)

        total_rows_written = 0

        for i, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
            df = batch.to_pandas()
            if df.index.names != [None]:
                df = df.reset_index()

            # Write header only for the first chunk
            mode = 'w' if i == 0 else 'a'
            header = True if i == 0 else False

            df.to_csv(output_file, mode=mode, header=header, index=False)
            total_rows_written += len(df)

            if (i + 1) % 10 == 0:
                logger.info(f"[MAIN] Processed {i + 1} chunks ({total_rows_written} rows)...")

        logger.info(f"[MAIN] Finished converting. Total rows written: {total_rows_written}")

    except Exception as e:
        logger.error(f"[MAIN] Error processing Parquet file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
