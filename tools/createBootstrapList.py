import argparse
import sys
import polars as pl
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Create a prioritized list of Gene-CpG pairs for bootstraping.")
    parser.add_argument("--input", type=str, default="results.precise_p.annot.fdr.parquet",
                        help="Path to the annotated Parquet file (Default: results.precise_p.annot.fdr.parquet)")
    parser.add_argument("--output", type=str, default="pairs_to_bootstrap.csv",
                        help="Path for the resulting CSV (Default: pairs_to_bootstrap.csv)")
    parser.add_argument("--rank-by", type=str, choices=["p-value", "ig_score", "magnitude"], required=True,
                        help="Metric used for ranking")
    parser.add_argument("--percent", type=float, default=0.10,
                        help="The percentage of top hits to select per region (Default: 0.10)")
    parser.add_argument("--max-per-region", type=int, default=2000,
                        help="The maximum number of pairs to select from any single region (Default: 2000)")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    # Lazily scan the parquet file to inspect columns and process efficiently
    try:
        lf = pl.scan_parquet(args.input)
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        sys.exit(1)

    columns = lf.collect_schema().names()

    # Check for essential columns
    required_cols = ['mt_id', 'gt_id', 'region']
    missing_req = [c for c in required_cols if c not in columns]
    if missing_req:
        print(f"Error: Missing essential columns: {missing_req}")
        print(f"Available columns: {columns}")
        sys.exit(1)

    # Determine sorting column and direction
    sort_col = None
    descending = False

    if args.rank_by == "p-value":
        if "precise_mt_p" in columns:
            sort_col = "precise_mt_p"
        elif "mt_p" in columns:
            sort_col = "mt_p"
        else:
            print("Error: Missing 'precise_mt_p' or 'mt_p' column for ranking by p-value.")
            print(f"Available columns: {columns}")
            sys.exit(1)
        descending = False # Smaller p-value is better
    elif args.rank_by == "ig_score":
        if "mt_ig" in columns:
            sort_col = "mt_ig"
        else:
            print("Error: Missing 'mt_ig' column for ranking by ig_score.")
            print(f"Available columns: {columns}")
            sys.exit(1)
        descending = True # Larger IG score is better
    elif args.rank_by == "magnitude":
        if "mt_est" in columns:
            sort_col = "abs_mt_est" # We will calculate this
        else:
            print("Error: Missing 'mt_est' column for ranking by magnitude.")
            print(f"Available columns: {columns}")
            sys.exit(1)
        descending = True # Larger magnitude is better

    # Load data needed, applying expressions
    select_exprs = [pl.col("mt_id"), pl.col("gt_id"), pl.col("region")]

    if args.rank_by == "magnitude":
        select_exprs.append(pl.col("mt_est").abs().alias("abs_mt_est"))
    else:
        select_exprs.append(pl.col(sort_col))

    df = lf.select(select_exprs).collect()

    # Fill null regions with UNKNOWN
    df = df.with_columns(
        pl.col("region").fill_null("UNKNOWN")
    )

    # Handle duplicates
    # Group by mt_id and gt_id, and if count > 1, we have duplicates
    dup_counts = df.group_by(["mt_id", "gt_id"]).agg(pl.len().alias("count"))
    duplicates = dup_counts.filter(pl.col("count") > 1)

    if not duplicates.is_empty():
        print(f"Note: Found {len(duplicates)} duplicate pairs. Logging to duplicate_pairs_report.txt")
        # Join back to get regions to log
        dup_full = df.join(duplicates, on=["mt_id", "gt_id"], how="inner")

        # Write report
        with open("duplicate_pairs_report.txt", "w") as f:
            f.write("mt_id\tgt_id\tregions\n")
            # Group by pair to get all regions
            grouped_dups = dup_full.group_by(["mt_id", "gt_id"]).agg(pl.col("region"))
            for row in grouped_dups.iter_rows(named=True):
                regions_str = ", ".join(row['region'])
                f.write(f"{row['mt_id']}\t{row['gt_id']}\t{regions_str}\n")

        # Deduplicate dataframe, keeping first occurrence based on our sort metric to be safe,
        # or just first seen. We will sort first then unique to keep the "best" one
        df = df.sort(sort_col, descending=descending)
        df = df.unique(subset=["mt_id", "gt_id"], keep="first", maintain_order=True)
    else:
        # Just sort
        df = df.sort(sort_col, descending=descending)

    # Get regional counts
    region_counts = df.group_by("region").agg(pl.len().alias("total_hits")).sort("region")

    regions = ["PROMOTER", "GENEBODY", "CIS", "DISTAL", "TRANS", "UNKNOWN"]

    # Find all regions present
    present_regions = df["region"].unique().to_list()
    # Add any unexpected regions to our list for reporting
    for r in present_regions:
        if r not in regions:
            regions.append(r)

    # Perform selection and build summary
    final_dfs = []
    summary_data = []

    total_original_hits = 0
    total_selected_hits = 0

    for region in regions:
        region_df = df.filter(pl.col("region") == region)
        total_hits = len(region_df)

        if total_hits == 0:
            continue

        total_original_hits += total_hits

        target_count = int(total_hits * args.percent)
        final_count = min(target_count, args.max_per_region)
        total_selected_hits += final_count

        is_capped = target_count > args.max_per_region
        capped_str = " (CAPPED)" if is_capped else ""

        summary_data.append({
            "Region": region,
            "Total Hits": f"{total_hits:,}",
            "Target (%)": f"{target_count:,}",
            "Final Selected": f"{final_count:,}{capped_str}"
        })

        # We already sorted df globally, so we can just take the top N
        final_dfs.append(region_df.head(final_count))

    # Combine all selected pairs
    if final_dfs:
        final_df = pl.concat(final_dfs)
        # Select only required columns and write
        final_df.select(["mt_id", "gt_id"]).write_csv(args.output)
    else:
        print("Warning: No pairs selected.")
        # Create empty CSV with headers
        with open(args.output, 'w') as f:
            f.write("mt_id,gt_id\n")

    # Print summary
    print("=" * 50)
    print("Bootstrap Pair List Generation Summary")
    print("=" * 50)
    print(f"Ranking Metric: {args.rank_by}")
    print(f"Selection Criteria: Top {args.percent*100:.0f}% | Cap: {args.max_per_region} per region")
    print()
    print(f"{'Region':<12} {'Total Hits':<13} {'Target (%)':<14} {'Final Selected'}")
    print("-" * 50)

    for row in summary_data:
        print(f"{row['Region']:<12} {row['Total Hits']:<13} {row['Target (%)']:<14} {row['Final Selected']}")

    print("-" * 50)
    print(f"{'TOTAL':<12} {f'{total_original_hits:,}':<13} {'':<14} {f'{total_selected_hits:,}'}")
    print("=" * 50)
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
