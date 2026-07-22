#!/usr/bin/env python3
import argparse
import logging
import sys
import pandas as pd

logger = logging.getLogger(__name__)


def _reset_if_indexed(df: pd.DataFrame) -> pd.DataFrame:
    """Promote a named index to columns.

    Map outputs store (mt_id, gt_id) in a named MultiIndex, and mergeOutputs'
    parquet->parquet path is a raw Arrow passthrough that preserves it. CSV
    inputs carry them as columns, which is why this only bites on parquet.
    Same guard used by assignRegionToEcpg_parquet.py / mergeOutputs.py /
    permute.py.
    """
    if df.index.names != [None]:
        return df.reset_index()
    return df


def assemble_master(
        cis_map_df: pd.DataFrame,
        reservoir_df: pd.DataFrame,
        mt_t_atol: float) -> pd.DataFrame:
    cis_map_df = _reset_if_indexed(cis_map_df)
    reservoir_df = _reset_if_indexed(reservoir_df)
    required_cols = {'mt_id', 'gt_id', 'mt_t'}

    # Check for required columns
    for source_name, df in [('cis_map', cis_map_df),
                            ('reservoir', reservoir_df)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Source '{source_name}' is missing required "
                f"columns: {missing}"
            )

    # Determine core columns to keep
    keep_cols = list(required_cols)
    if 'mt_p' in cis_map_df.columns and 'mt_p' in reservoir_df.columns:
        keep_cols.append('mt_p')

    # Subset and tag
    cis_subset = cis_map_df[keep_cols].copy()
    cis_subset['_source'] = 'cis_map'

    res_subset = reservoir_df[keep_cols].copy()
    res_subset['_source'] = 'reservoir'

    # Concat
    assembled = pd.concat([cis_subset, res_subset], ignore_index=True)

    # Dedupe with agreement check
    # Find overlaps
    # A faster way: group by mt_id, gt_id and check if max(mt_t) - min(mt_t)
    # <= mt_t_atol
    grouped = assembled.groupby(['mt_id', 'gt_id'])['mt_t']
    mt_t_diffs = grouped.max() - grouped.min()

    disagreements = mt_t_diffs[mt_t_diffs > mt_t_atol]
    if not disagreements.empty:
        # Get some examples
        examples = disagreements.head(3).index.tolist()

        # Pull original values for the first example
        ex_mt_id, ex_gt_id = examples[0]
        ex_rows = assembled[(assembled['mt_id'] == ex_mt_id)
                            & (assembled['gt_id'] == ex_gt_id)]
        ex_details = []
        for _, row in ex_rows.iterrows():
            ex_details.append(f"{row['_source']}: mt_t={row['mt_t']}")

        raise ValueError(
            f"Overlap disagreement: {len(disagreements)} pairs present "
            f"in both sources differ in mt_t by more than {mt_t_atol}. "
            f"Example (mt_id={ex_mt_id}, gt_id={ex_gt_id}): "
            f"{', '.join(ex_details)}"
        )

    # Dedupe by keeping the first occurrence (since they agree)
    assembled = assembled.drop_duplicates(
        subset=['mt_id', 'gt_id'], keep='first')

    # Empty guard
    if assembled.empty:
        raise ValueError("Assembled master dataframe is empty.")

    # Drop tag
    assembled = assembled.drop(columns=['_source'])

    return assembled


def main():
    parser = argparse.ArgumentParser(
        description="Assemble a gene-anchored master file by "
                    "combining a cis-map and a reservoir.")
    parser.add_argument(
        '--cis-map',
        required=True,
        help="Path to cis-map output parquet.")
    parser.add_argument(
        '--reservoir',
        required=True,
        help="Path to uniform reservoir (.parquet or .csv).")
    parser.add_argument(
        '--out',
        required=True,
        help="Path to output assembled master parquet.")
    parser.add_argument(
        '--mt-t-atol',
        type=float,
        default=1e-3,
        help="Max |Δ mt_t| tolerated for a pair present in both sources.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s')

    try:
        logger.info("Loading cis-map from {0}".format(args.cis_map))
        cis_map_df = pd.read_parquet(args.cis_map)

        logger.info("Loading reservoir from {0}".format(args.reservoir))
        if args.reservoir.endswith('.csv'):
            reservoir_df = pd.read_csv(args.reservoir)
        else:
            reservoir_df = pd.read_parquet(args.reservoir)
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        logger.error("Error loading inputs: {0}".format(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Error loading inputs: {0}".format(e))
        sys.exit(1)

    try:
        assembled = assemble_master(cis_map_df, reservoir_df, args.mt_t_atol)
    except ValueError as e:
        logger.error("Validation error: {0}".format(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Error assembling master: {0}".format(e))
        sys.exit(1)

    overlap_count = len(cis_map_df) + len(reservoir_df) - len(assembled)
    logger.info(
        "Assembled master size: {0} (from {1} cis-map, "
        "{2} reservoir, {3} overlapping pairs)".format(
            len(assembled),
            len(cis_map_df),
            len(reservoir_df),
            overlap_count))

    try:
        logger.info("Writing output to {0}".format(args.out))
        assembled.to_parquet(args.out, index=False)
    except Exception as e:
        logger.error("Error writing output: {0}".format(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
