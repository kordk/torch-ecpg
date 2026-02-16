
import os
import sys
import pandas as pd
import numpy as np
import torch
import unittest

# Ensure we can import tecpg
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from tecpg.test_data import generate_data
from tecpg.regression_full import regression_full
from tecpg.processing import tecpg_mlr_lstsq
from tecpg.logger import Logger

class TestMlrLstsq(unittest.TestCase):
    def test_mlr_lstsq_consistency(self):
        """
        Tests if tecpg_mlr_lstsq produces results consistent with regression_full.
        """
        print("Generating synthetic data...")
        # Generate small dataset for testing
        sample_size = 50
        m_loci = 100
        g_loci = 50
        M, G, C, M_annot, G_annot = generate_data(sample_size, m_loci, g_loci, annotation=True)

        # Set index for annotations as expected by regression functions
        M_annot.set_index('name', inplace=True)
        G_annot.set_index('name', inplace=True)

        # Ensure indices are aligned and sorted for deterministic comparison
        M = M.sort_index()
        G = G.sort_index()
        C = C.sort_index()
        M_annot = M_annot.sort_index()
        G_annot = G_annot.sort_index()

        logger = Logger(carry_data={'use_cpu': True})
        logger.start_timer()

        print("Running regression_full (original)...")
        res_full = regression_full(
            M, G, C,
            M_annot, G_annot,
            region='all',
            p_thresh=None,
            logger=logger
        )

        print("Running tecpg_mlr_lstsq (new)...")
        res_lstsq = tecpg_mlr_lstsq(
            M, G, C,
            M_annot, G_annot,
            region='all',
            p_thresh=None,
            logger=logger
        )

        # Align DataFrames
        # Sort index to ensure matching rows
        res_full = res_full.sort_index()
        res_lstsq = res_lstsq.sort_index()

        # Check if indices match
        pd.testing.assert_index_equal(res_full.index, res_lstsq.index)

        # Check if columns match
        pd.testing.assert_index_equal(res_full.columns, res_lstsq.columns)

        # Compare values with tolerance
        # Tolerance: 1e-4 for float32 precision usually.

        print("Comparing results...")

        # Calculate max difference
        diff = (res_full - res_lstsq).abs()
        max_diff = diff.max().max()
        print(f"Max absolute difference: {max_diff}")

        # Check estimates
        est_cols = [c for c in res_full.columns if c.endswith('_est')]
        pd.testing.assert_frame_equal(res_full[est_cols], res_lstsq[est_cols], atol=5e-4, rtol=5e-4)

        # Check standard errors
        err_cols = [c for c in res_full.columns if c.endswith('_err')]
        pd.testing.assert_frame_equal(res_full[err_cols], res_lstsq[err_cols], atol=5e-4, rtol=5e-4)

        # Check t-values
        t_cols = [c for c in res_full.columns if c.endswith('_t')]
        pd.testing.assert_frame_equal(res_full[t_cols], res_lstsq[t_cols], atol=5e-3, rtol=5e-3)

        # Check p-values
        p_cols = [c for c in res_full.columns if c.endswith('_p')]
        pd.testing.assert_frame_equal(res_full[p_cols], res_lstsq[p_cols], atol=5e-4, rtol=5e-4)

        print("Test passed!")

if __name__ == '__main__':
    unittest.main()
