
import os
import sys
import random
import pandas as pd
import numpy as np
import scipy.stats
from typing import List

# Ensure we can import tecpg
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import tecpg
from tecpg.test_data import generate_data
from tecpg.regression_full import regression_full
from tecpg.logger import Logger

try:
    from tests.validation_utils import run_statsmodels_ols, compare_results
except ImportError:
    from validation_utils import run_statsmodels_ols, compare_results

def main():
    print("Starting tecpg accuracy validation test...")

    # 1. Generate Data
    sample_size = 100
    m_loci = 500
    g_loci = 100
    print(f"Generating synthetic data: {sample_size} samples, {m_loci} M loci, {g_loci} G loci.")

    # Using tecpg.test_data.generate_data
    # Returns M, G, C, M_annot, G_annot
    M, G, C, M_annot, G_annot = generate_data(sample_size, m_loci, g_loci, annotation=True)

    # 2. Run tecpg Analysis
    print("Running tecpg regression (all pairs)...")
    logger = Logger(carry_data={'use_cpu': True}) # Force CPU for deterministic behavior if possible
    logger.start_timer()

    # We use 'all' mode as requested for validation
    tecpg_res_df = regression_full(
        M, G, C,
        M_annot, G_annot,
        region='all',
        p_thresh=None, # Get all results to validate
        logger=logger
    )

    print(f"tecpg analysis complete. Result shape: {tecpg_res_df.shape}")

    # 3. Select Subset for Validation
    n_validate = 100
    print(f"Selecting {n_validate} random pairs for validation against statsmodels...")

    # The index of tecpg_res_df is MultiIndex (gt_id, mt_id)
    all_pairs = tecpg_res_df.index.tolist()
    if len(all_pairs) > n_validate:
        validation_pairs = random.sample(all_pairs, n_validate)
    else:
        validation_pairs = all_pairs

    results = []

    # 4. Run Independent Validation
    for gt_id, mt_id in validation_pairs:
        # Get data for this pair
        m_series = M.loc[mt_id]
        g_series = G.loc[gt_id]

        # Get tecpg result row
        # tecpg index is (gt_id, mt_id)
        tecpg_row = tecpg_res_df.loc[(gt_id, mt_id)]

        tecpg_vals = {
            'mt_est': tecpg_row['mt_est'],
            'mt_err': tecpg_row['mt_err'],
            'mt_t': tecpg_row['mt_t'],
            'mt_p': tecpg_row['mt_p']
        }

        # Run statsmodels
        sm_vals = run_statsmodels_ols(m_series, g_series, C)

        # Compare
        diffs = compare_results(tecpg_vals, sm_vals)
        diffs['gt_id'] = gt_id
        diffs['mt_id'] = mt_id
        diffs['sm_p'] = sm_vals['mt_p']
        diffs['tecpg_p'] = tecpg_vals['mt_p']

        results.append(diffs)

    results_df = pd.DataFrame(results)

    # 5. Analyze and Report
    print("\nValidation Results Summary:")
    print("-" * 30)

    metrics = ['diff_est', 'diff_err', 'diff_t', 'diff_p']
    for metric in metrics:
        mean_diff = results_df[metric].mean()
        max_diff = results_df[metric].max()
        print(f"{metric:10} | Mean: {mean_diff:.6e} | Max: {max_diff:.6e}")

    print("-" * 30)

    # Check Accuracy Criteria
    # Coefficients and T-stats should be very close (floating point error)
    # P-values will differ because tecpg uses Normal approx, statsmodels uses Student-t

    # Allowable error for float32 vs float64 calculations
    TOLERANCE_EST = 2e-4
    TOLERANCE_T = 1e-3

    failures = results_df[
        (results_df['diff_est'] > TOLERANCE_EST) |
        (results_df['diff_t'] > TOLERANCE_T)
    ]

    if len(failures) > 0:
        print(f"WARNING: {len(failures)} pairs exceeded tolerance for Estimate or T-stat.")
        print(failures.head())
        # Raise error to fail the test if tolerances are exceeded
        raise AssertionError(f"Validation failed: {len(failures)} pairs exceeded tolerance.")
    else:
        print("SUCCESS: Estimates and T-statistics match within tolerance.")

    # P-value analysis
    # We expect p-values to differ. Let's quantify it.
    print("\nP-value Comparison (Normal approx vs Student-t):")
    print(f"Average absolute p-value difference: {results_df['diff_p'].mean():.6f}")

    # Check if the difference is consistent with theoretical expectation
    # (Optional: compute theoretical max difference for given DF)
    df_val = sample_size - (1 + 1 + len(C.columns)) # N - (intercept + M + covariates)
    print(f"Degrees of Freedom: {df_val}")

    # 6. Save Report
    report_path = "validation_report.csv"
    results_df.to_csv(report_path, index=False)
    print(f"Detailed validation results saved to {report_path}")

if __name__ == "__main__":
    main()
