#!/usr/bin/env python3
import argparse
import sys
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Subsample loci (rows) from a CSV matrix, keeping all samples (columns).")
    parser.add_argument("input_csv", help="Path to input CSV file (rows=loci, cols=samples)")
    parser.add_argument("output_csv", help="Path to output CSV file")
    parser.add_argument("n_loci", type=int, help="Target number of loci to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling (default: 42)")
    args = parser.parse_args()

    print(f"Loading {args.input_csv}...")
    try:
        df = pd.read_csv(args.input_csv, index_col=0)
    except Exception as e:
        print(f"Error loading {args.input_csv}: {e}")
        sys.exit(1)

    n_before_rows, n_before_cols = df.shape

    if args.n_loci <= 0:
        print(f"Error: Target number of loci must be positive, got {args.n_loci}")
        sys.exit(1)

    if n_before_rows > args.n_loci:
        df = df.sample(n=args.n_loci, random_state=args.seed, axis=0)
    else:
        print("... already <= n_loci; taking all N rows unchanged")

    # Assert columns unchanged
    if df.shape[1] != n_before_cols:
        print(f"Error: Output column count ({df.shape[1]}) does not match input ({n_before_cols}).")
        sys.exit(1)

    try:
        df.to_csv(args.output_csv)
    except Exception as e:
        print(f"Error saving to {args.output_csv}: {e}")
        sys.exit(1)

    print(f"subsample_loci: {args.input_csv}  {n_before_rows} -> {len(df)} rows, {df.shape[1]} cols (unchanged)")

if __name__ == "__main__":
    main()
