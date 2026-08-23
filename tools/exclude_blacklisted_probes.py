#!/usr/bin/env python3
import pandas as pd
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Exclude blacklisted probes from methylation data.")
    parser.add_argument("m_orig_file", help="Path to M_orig.csv file")
    parser.add_argument("blacklist_file",
                        help="Path to probes_blacklist.csv (first column must be probe IDs)")
    parser.add_argument("m_out_file", help="Path to output M.csv file")
    args = parser.parse_args()

    print(f"Loading {args.m_orig_file}...")
    try:
        M_orig = pd.read_csv(args.m_orig_file, index_col=0)
    except Exception as e:
        print(f"Error loading {args.m_orig_file}: {e}")
        sys.exit(1)

    print(f"Original M rows: {len(M_orig)}")

    try:
        blacklist = pd.read_csv(args.blacklist_file)
        # Assume the first column contains the probe IDs
        blacklist_probes = set(blacklist.iloc[:, 0].astype(str))
    except Exception as e:
        print(f"Error loading {args.blacklist_file}: {e}")
        sys.exit(1)

    print(f"Loaded {len(blacklist_probes)} probes to exclude.")

    # M_orig index contains probe IDs
    probes_to_keep = [p for p in M_orig.index if str(p) not in blacklist_probes]
    M_new = M_orig.loc[probes_to_keep]

    excluded_count = len(M_orig) - len(M_new)
    print(f"New M rows: {len(M_new)} (Excluded {excluded_count} probes)")

    try:
        M_new.to_csv(args.m_out_file)
        print(f"Saved filtered data to {args.m_out_file}")
    except Exception as e:
        print(f"Error saving to {args.m_out_file}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
