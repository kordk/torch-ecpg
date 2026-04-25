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

try:
    from tests.validation_utils import save_scatter_plot
except ImportError:
    from validation_utils import save_scatter_plot

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

def run_comparison_test(test_name, M, G, C, M_annot=None, G_annot=None, logit_transform=False, **kwargs):
    print(f"\n--- Running Test: {test_name} ---")
    logger = Logger()

    args = {
        'M': M, 'G': G, 'C': C,
        'M_annot': M_annot, 'G_annot': G_annot,
        'logger': logger,
        'p_only': False,
        'methylation_only': False,
        'logit_transform': logit_transform,
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

    sanitized_name = test_name.replace(" ", "_").lower()
    save_scatter_plot(
        res_manual['mt_est'], res_lstsq['mt_est'],
        'Manual Estimate', 'Lstsq Estimate',
        f'Comparison ({test_name}) - Estimate',
        f'comparison_{sanitized_name}_est.png'
    )
    save_scatter_plot(
        res_manual['mt_err'], res_lstsq['mt_err'],
        'Manual Std Error', 'Lstsq Std Error',
        f'Comparison ({test_name}) - Std Error',
        f'comparison_{sanitized_name}_err.png'
    )
    save_scatter_plot(
        res_manual['mt_t'], res_lstsq['mt_t'],
        'Manual T-stat', 'Lstsq T-stat',
        f'Comparison ({test_name}) - T-statistic',
        f'comparison_{sanitized_name}_t.png'
    )
    save_scatter_plot(
        res_manual['mt_p'], res_lstsq['mt_p'],
        'Manual P-value', 'Lstsq P-value',
        f'Comparison ({test_name}) - P-value',
        f'comparison_{sanitized_name}_p.png'
    )

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

def test_lstsq_memory_opt_various():
    import scipy.stats

    print("\n--- Running Test: Memory Opt Num. Equivalence ---")
    np.random.seed(42)
    S = 30
    M = 20
    G = 15
    K = 4

    M_data = pd.DataFrame(np.random.rand(M, S), index=[f'mt_{i}' for i in range(M)])
    G_data = pd.DataFrame(np.random.rand(G, S), index=[f'gt_{i}' for i in range(G)])
    C_data = pd.DataFrame(np.random.rand(S, K - 2), columns=[str(i) for i in range(K-2)])

    for meth_only, chunk in [(True, False), (False, True), (False, False)]:
        print(f"Testing meth_only={meth_only}, chunking={chunk}")

        kwargs = {
            'region': 'all',
            'methylation_only': meth_only,
            'p_only': False,
            'logit_transform': False,
            'logger': Logger()
        }

        if chunk:
            kwargs['gene_loci_per_chunk'] = 7
            kwargs['meth_loci_per_chunk'] = 7
            kwargs['output_dir'] = 'test_mem_opt_out'
            if os.path.exists('test_mem_opt_out'):
                shutil.rmtree('test_mem_opt_out')
            os.makedirs('test_mem_opt_out')

        df_tecpg = tecpg_mlr_lstsq(M_data, G_data, C_data, **kwargs)

        if chunk:
            df_tecpg = read_chunk_results('test_mem_opt_out')
            shutil.rmtree('test_mem_opt_out')

        results = []

        for gt_idx in range(G):
            for mt_idx in range(M):
                y = G_data.iloc[gt_idx].values

                x_m = M_data.iloc[mt_idx].values
                X = np.column_stack((np.ones(S), x_m, C_data.values))

                b, res, rank, s = np.linalg.lstsq(X, y, rcond=None)

                df = S - K
                if len(res) > 0:
                    rss = res[0]
                else:
                    rss = np.sum((y - X @ b)**2)

                sigma2 = rss / df
                var_b = sigma2 * np.linalg.inv(X.T @ X).diagonal()
                se = np.sqrt(var_b)

                t = b / se
                p = 2 * (1 - scipy.stats.norm.cdf(np.abs(t)))

                row = {
                    'gt_id': G_data.index[gt_idx],
                    'mt_id': M_data.index[mt_idx]
                }

                col_names = ['const', 'mt', '0', '1']
                for k in range(K):
                    if meth_only and k != 1:
                        continue
                    row[f'{col_names[k]}_est'] = b[k]
                    row[f'{col_names[k]}_err'] = se[k]
                    row[f'{col_names[k]}_t'] = t[k]
                    row[f'{col_names[k]}_p'] = p[k]

                results.append(row)

        df_np = pd.DataFrame(results).set_index(['gt_id', 'mt_id'])

        # match columns and type
        df_np = df_np[df_tecpg.columns].astype(df_tecpg.dtypes.iloc[0])

        # ensure indices are sorted similarly
        df_tecpg = df_tecpg.sort_index()
        df_np = df_np.sort_index()

        pd.testing.assert_frame_equal(df_tecpg, df_np, rtol=1e-4, atol=1e-4)
        print("Memory Opt Num. Equivalence passed!")

def main():
    try:
        print("Generating data without annotation...")
        sample_size = 50
        m_rows = 50
        g_rows = 50
        M, G, C = generate_data(sample_size, m_rows, g_rows, annotation=False)

        for logit_transform in [False, True]:
            transform_suffix = " (M-values)" if logit_transform else " (Beta)"

            run_comparison_test(
                f"All Region{transform_suffix}",
                M, G, C,
                region='all',
                logit_transform=logit_transform
            )

            run_comparison_test(
                f"All Region Chunked{transform_suffix}",
                M, G, C,
                region='all',
                meth_loci_per_chunk=20,
                output_dir=f"test_output_chunked{'_transformed' if logit_transform else ''}",
                logit_transform=logit_transform
            )

        print("\nGenerating data with annotation...")
        M, G, C, M_annot, G_annot = generate_data(sample_size, m_rows, g_rows, annotation=True)
        M_annot.set_index('name', inplace=True)
        G_annot.set_index('name', inplace=True)

        for logit_transform in [False, True]:
            transform_suffix = " (M-values)" if logit_transform else " (Beta)"

            run_comparison_test(
                f"Cis Region{transform_suffix}",
                M, G, C, M_annot, G_annot,
                region='cis',
                window_base=0, upstream=50000, downstream=50000,
                logit_transform=logit_transform
            )

            run_comparison_test(
                f"Trans Region{transform_suffix}",
                M, G, C, M_annot, G_annot,
                region='trans',
                logit_transform=logit_transform
            )

        print("\nAll tests passed successfully!")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_lstsq_memory_opt_various()

    test_lstsq_memory_opt_various()

    main()

def test_lstsq_memory_opt_various():
    import scipy.stats

    print("\n--- Running Test: Memory Opt Num. Equivalence ---")
    np.random.seed(42)
    S = 30
    M = 20
    G = 15
    K = 4

    M_data = pd.DataFrame(np.random.rand(M, S), index=[f'mt_{i}' for i in range(M)])
    G_data = pd.DataFrame(np.random.rand(G, S), index=[f'gt_{i}' for i in range(G)])
    C_data = pd.DataFrame(np.random.rand(S, K - 2), columns=[str(i) for i in range(K-2)])

    for meth_only, chunk in [(True, False), (False, True), (False, False)]:
        print(f"Testing meth_only={meth_only}, chunking={chunk}")

        kwargs = {
            'region': 'all',
            'methylation_only': meth_only,
            'p_only': False,
            'logit_transform': False,
            'logger': Logger()
        }

        if chunk:
            kwargs['gene_loci_per_chunk'] = 7
            kwargs['meth_loci_per_chunk'] = 7
            kwargs['output_dir'] = 'test_mem_opt_out'
            if os.path.exists('test_mem_opt_out'):
                shutil.rmtree('test_mem_opt_out')
            os.makedirs('test_mem_opt_out')

        df_tecpg = tecpg_mlr_lstsq(M_data, G_data, C_data, **kwargs)

        if chunk:
            df_tecpg = read_chunk_results('test_mem_opt_out')
            shutil.rmtree('test_mem_opt_out')

        results = []

        for gt_idx in range(G):
            for mt_idx in range(M):
                y = G_data.iloc[gt_idx].values

                x_m = M_data.iloc[mt_idx].values
                X = np.column_stack((np.ones(S), x_m, C_data.values))

                b, res, rank, s = np.linalg.lstsq(X, y, rcond=None)

                df = S - K
                if len(res) > 0:
                    rss = res[0]
                else:
                    rss = np.sum((y - X @ b)**2)

                sigma2 = rss / df
                var_b = sigma2 * np.linalg.inv(X.T @ X).diagonal()
                se = np.sqrt(var_b)

                t = b / se
                p = 2 * (1 - scipy.stats.norm.cdf(np.abs(t)))

                row = {
                    'gt_id': G_data.index[gt_idx],
                    'mt_id': M_data.index[mt_idx]
                }

                col_names = ['const', 'mt', '0', '1']
                for k in range(K):
                    if meth_only and k != 1:
                        continue
                    row[f'{col_names[k]}_est'] = b[k]
                    row[f'{col_names[k]}_err'] = se[k]
                    row[f'{col_names[k]}_t'] = t[k]
                    row[f'{col_names[k]}_p'] = p[k]

                results.append(row)

        df_np = pd.DataFrame(results).set_index(['gt_id', 'mt_id'])

        # match columns and type
        df_np = df_np[df_tecpg.columns].astype(df_tecpg.dtypes.iloc[0])

        # ensure indices are sorted similarly
        df_tecpg = df_tecpg.sort_index()
        df_np = df_np.sort_index()

        pd.testing.assert_frame_equal(df_tecpg, df_np, rtol=1e-4, atol=1e-4)
        print("Memory Opt Num. Equivalence passed!")
