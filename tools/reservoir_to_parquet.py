#!/usr/bin/env python3
import argparse
import sys
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Convert sample_reservoir.csv to reservoir_master.parquet")
    parser.add_argument("--in", dest="in_file", required=True, help="Input CSV file (e.g., sample_reservoir.csv)")
    parser.add_argument("--out", dest="out_file", required=True, help="Output parquet file")

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.in_file)
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.in_file}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logger.error(f"Input file is empty: {args.in_file}")
        sys.exit(1)

    required_cols = {'mt_id', 'gt_id', 'mt_t'}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Reservoir CSV missing required columns: {missing}")

    df.to_parquet(args.out_file, index=False)
    logger.info(f"Successfully converted {args.in_file} to {args.out_file}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
