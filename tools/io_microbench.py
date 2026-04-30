import argparse
import os
import shutil
import tempfile
import time
import pandas as pd
import numpy as np
from datetime import datetime

# Import the exact save_dataframe_part used by tecpg
from tecpg.import_data import save_dataframe_part
from tecpg.logger import Logger


def generate_synthetic_df(rows=500_000):
    """
    Generates a synthetic DataFrame matching the shape tecpg actually produces.
    (rows x ~5 numeric columns)
    """
    return pd.DataFrame({
        'mt_id': [f'cg{i}' for i in range(rows)],
        'gt_id': [f'ENSG00000{i}' for i in range(rows)],
        'p_value': np.random.rand(rows),
        'mt_ig': np.random.rand(rows),
        'magnitude': np.random.rand(rows),
    }).set_index(['mt_id', 'gt_id'])


def run_bench(df, target_dir, n_iterations, logger):
    """
    Writes the dataframe N times using save_dataframe_part to simulate chunk appending
    """
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, 'microbench_results.csv')

    # ensure file doesn't exist before we start simulating
    if os.path.exists(file_path):
        os.remove(file_path)

    times_ms = []

    for i in range(n_iterations):
        start_t = time.perf_counter()

        save_dataframe_part(
            dataframe=df,
            file_path=file_path,
            chunk_number=i,
            logger=logger
        )

        end_t = time.perf_counter()
        times_ms.append((end_t - start_t) * 1000.0)

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    total_time_s = sum(times_ms) / 1000.0
    mb_per_s = file_size_mb / total_time_s if total_time_s > 0 else 0

    p50 = np.percentile(times_ms, 50)
    p99 = np.percentile(times_ms, 99)

    return p50, p99, mb_per_s, file_path


def main():
    parser = argparse.ArgumentParser(description="Microbenchmark for tecpg writer bottleneck")
    parser.add_argument('--output-dir', required=True, help="Target slow directory (e.g. network/slow FS)")
    parser.add_argument('--local-fast-dir', default=os.path.join(tempfile.gettempdir(), 'tecpg-writebench'), help="Local fast directory (e.g. NVMe path)")
    parser.add_argument('-n', '--iterations', type=int, default=20, help="Number of write iterations")
    parser.add_argument('--rows', type=int, default=500_000, help="Number of rows per chunk")
    parser.add_argument('--keep', action='store_true', help="Keep the written test files after benchmarking")
    args = parser.parse_args()

    # We need TECPG_PROFILE=1 equivalent logger so debug logs actually happen
    logger = Logger()
    logger.is_debug = True
    logger.carry_data['profile'] = True

    print(f"Generating synthetic dataframe with {args.rows} rows...")
    df = generate_synthetic_df(args.rows)
    print(f"Dataframe memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")

    # Run benchmark for slow dir
    print(f"\nRunning slow dir benchmark ({args.output_dir})...")
    # We create a dedicated sub-directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slow_bench_dir = os.path.join(args.output_dir, f"microbench_{timestamp}")
    slow_p50, slow_p99, slow_mb_s, slow_file = run_bench(df, slow_bench_dir, args.iterations, logger)

    # Run benchmark for fast dir
    print(f"\nRunning fast dir benchmark ({args.local_fast_dir})...")
    fast_bench_dir = os.path.join(args.local_fast_dir, f"microbench_{timestamp}")
    fast_p50, fast_p99, fast_mb_s, fast_file = run_bench(df, fast_bench_dir, args.iterations, logger)

    # Print results out cleanly so the bash script can parse/redirect
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Target Slow ({args.output_dir}):")
    print(f"  p50:  {slow_p50:.2f} ms")
    print(f"  p99:  {slow_p99:.2f} ms")
    print(f"  Speed: {slow_mb_s:.2f} MB/s")
    print()
    print(f"Target Fast ({args.local_fast_dir}):")
    print(f"  p50:  {fast_p50:.2f} ms")
    print(f"  p99:  {fast_p99:.2f} ms")
    print(f"  Speed: {fast_mb_s:.2f} MB/s")

    if not args.keep:
        print("\nCleaning up test files...")
        shutil.rmtree(slow_bench_dir, ignore_errors=True)
        shutil.rmtree(fast_bench_dir, ignore_errors=True)
    else:
        print("\nKeeping test files:")
        print(f"  {slow_file}")
        print(f"  {fast_file}")

if __name__ == '__main__':
    main()
