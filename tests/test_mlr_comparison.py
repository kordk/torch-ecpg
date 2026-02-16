import os
import sys
import shutil
import glob
import pandas as pd
import numpy as np

# Ensure we can import tecpg
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from tecpg.test_data import generate_data
from tecpg.regression_full import regression_full
from tecpg.processing import tecpg_mlr_lstsq
from tecpg.logger import Logger

def summarize_comparison(df1, df2):
    """
    Prints a summary comparison between two dataframes.
    """
    print("\nComparison Summary:")
    print("-" * 118)
    print(f"{'Column':<20} | {'Max Abs Diff':<15} | {'Mean Abs Diff':<15} | {'Max Rel Diff':<15} | {'Mean Rel Diff':<15} | {'Correlation':<15}")
    print("-" * 118)

    if df1.empty:
        print("DataFrames are empty.")
        print("-" * 118)
        return

    diff = (df1 - df2).abs()
    rel_diff = diff / (df1.abs() + 1e-9)

    for col in df1.columns:
        if not pd.api.types.is_numeric_dtype(df1[col]):
            continue

        max_abs = diff[col].max()
        mean_abs = diff[col].mean()
        max_rel = rel_diff[col].max()
        mean_rel = rel_diff[col].mean()

        try:
            corr = df1[col].corr(df2[col])
        except Exception:
            corr = float('nan')

        print(f"{col:<20} | {max_abs:<15.4e} | {mean_abs:<15.4e} | {max_rel:<15.4e} | {mean_rel:<15.4e} | {corr:<15.6f}")

    print("-" * 118)

def read_chunk_results(output_dir):
    files = glob.glob(os.path.join(output_dir, "*.csv"))
    dfs = []
    for f in files:
        # Load, index_col=[0, 1] for MultiIndex
        try:
            df = pd.read_csv(f, index_col=[0, 1])
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            return None

    if not dfs:
        return None

    full_df = pd.concat(dfs)
    full_df.sort_index(inplace=True)
    return full_df

def run_comparison_test(test_name, M, G, C, M_annot=None, G_annot=None, **kwargs):
    print(f"\n--- Running Test: {test_name} ---")
    logger = Logger()

    args = {
        'M': M, 'G': G, 'C': C,
        'M_annot': M_annot, 'G_annot': G_annot,
        'logger': logger,
        'p_only': False,
        'methylation_only': False
    }
    args.update(kwargs)

    # Check if chunking is used
    chunking = 'output_dir' in args or 'meth_loci_per_chunk' in args or 'gene_loci_per_chunk' in args

    output_dir_manual = None
    output_dir_lstsq = None

    if chunking:
        # Define output dirs if not present or handle cleanup
        base_output_dir = args.get('output_dir', 'test_output_comparison')
        output_dir_manual = base_output_dir + "_manual"
        output_dir_lstsq = base_output_dir + "_lstsq"

        # Prepare manual run
        if os.path.exists(output_dir_manual):
            shutil.rmtree(output_dir_manual)
        args['output_dir'] = output_dir_manual

    print("Running regression_full (manual)...")
    logger.start_timer('info', 'Starting regression_full...')
    res_manual = regression_full(**args)

    if output_dir_manual:
        print(f"Reading manual results from {output_dir_manual}...")
        res_manual = read_chunk_results(output_dir_manual)

    # Prepare lstsq run
    if chunking:
        if os.path.exists(output_dir_lstsq):
            shutil.rmtree(output_dir_lstsq)
        args['output_dir'] = output_dir_lstsq

    print("Running tecpg_mlr_lstsq (lstsq)...")
    logger.start_timer('info', 'Starting tecpg_mlr_lstsq...')
    res_lstsq = tecpg_mlr_lstsq(**args)

    if output_dir_lstsq:
        print(f"Reading lstsq results from {output_dir_lstsq}...")
        res_lstsq = read_chunk_results(output_dir_lstsq)

    # Cleanup output dirs
    if output_dir_manual and os.path.exists(output_dir_manual):
        shutil.rmtree(output_dir_manual)
    if output_dir_lstsq and os.path.exists(output_dir_lstsq):
        shutil.rmtree(output_dir_lstsq)

    if res_manual is None or res_lstsq is None:
        raise AssertionError("One of the results is None. Comparison failed.")

    print("Comparing results...")

    # Ensure sorted index
    res_manual.sort_index(inplace=True)
    res_lstsq.sort_index(inplace=True)

    # Check shape
    if res_manual.shape != res_lstsq.shape:
        raise AssertionError(f"Shape mismatch: manual {res_manual.shape}, lstsq {res_lstsq.shape}")

    # Check index
    if not res_manual.index.equals(res_lstsq.index):
        print("Manual index head:", res_manual.index[:5])
        print("Lstsq index head:", res_lstsq.index[:5])
        raise AssertionError("Index mismatch")

    # Check columns
    if not res_manual.columns.equals(res_lstsq.columns):
        print(f"Manual columns: {res_manual.columns}")
        print(f"Lstsq columns: {res_lstsq.columns}")
        raise AssertionError("Column mismatch")

    # Compare values with tolerance
    summarize_comparison(res_manual, res_lstsq)

    try:
        pd.testing.assert_frame_equal(res_manual, res_lstsq, rtol=1e-3, atol=1e-3)
        print("Results are equal within tolerance (rtol=1e-3, atol=1e-3)!")
    except AssertionError as e:
        diff = (res_manual - res_lstsq).abs()
        max_diff = diff.max().max()
        rel_diff = diff / (res_manual.abs() + 1e-9)
        max_rel_diff = rel_diff.max().max()
        print(f"Max absolute difference: {max_diff}")
        print(f"Max relative difference: {max_rel_diff}")
        raise e

def main():
    try:
        print("Generating data without annotation...")
        sample_size = 50
        m_rows = 50
        g_rows = 50
        M, G, C = generate_data(sample_size, m_rows, g_rows, annotation=False)

        run_comparison_test("All Region", M, G, C, region='all')

        run_comparison_test(
            "All Region Chunked",
            M, G, C,
            region='all',
            meth_loci_per_chunk=20,
            output_dir="test_output_chunked"
        )

        print("\nGenerating data with annotation...")
        M, G, C, M_annot, G_annot = generate_data(sample_size, m_rows, g_rows, annotation=True)
        M_annot.set_index('name', inplace=True)
        G_annot.set_index('name', inplace=True)

        run_comparison_test(
            "Cis Region",
            M, G, C, M_annot, G_annot,
            region='cis',
            window_base=0, upstream=50000, downstream=50000
        )

        run_comparison_test(
            "Trans Region",
            M, G, C, M_annot, G_annot,
            region='trans'
        )

        print("\nAll tests passed successfully!")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
