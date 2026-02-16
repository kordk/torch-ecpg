
import unittest
import os
import sys
import torch
import numpy as np
import pandas as pd

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tecpg.test_data import generate_data
from tecpg.regression_full import regression_full
from tecpg.processing import tecpg_mlr_lstsq
from tecpg.logger import Logger
from tecpg.config import DTYPE

class TestMLRLstsq(unittest.TestCase):
    def setUp(self):
        self.sample_size = 50
        self.m_rows = 100
        self.g_rows = 50
        self.M, self.G, self.C, self.M_annot, self.G_annot = generate_data(
            self.sample_size, self.m_rows, self.g_rows, annotation=True
        )
        self.M_annot.set_index('name', inplace=True)
        self.G_annot.set_index('name', inplace=True)

        # Ensure column order for regression_full assumption
        self.M_annot = self.M_annot[['chrom', 'chromStart', 'chromEnd', 'score', 'strand']]
        self.G_annot = self.G_annot[['chrom', 'chromStart', 'chromEnd', 'score', 'strand']]

        # Force some overlaps for CIS testing
        # Set first 5 M loci to be on same chrom and close to first 5 G loci
        for i in range(5):
            g_name = self.G.index[i]
            m_name = self.M.index[i]

            # Get G position
            chrom = self.G_annot.loc[g_name, 'chrom']
            start = self.G_annot.loc[g_name, 'chromStart']

            # Set M position close to it
            self.M_annot.loc[m_name, 'chrom'] = chrom
            self.M_annot.loc[m_name, 'chromStart'] = start + 100 # within window
            self.M_annot.loc[m_name, 'chromEnd'] = start + 101

            # Ensure G is on + strand to avoid potential issues with negative strand logic
            self.G_annot.loc[g_name, 'strand'] = '+'

        self.logger = Logger(carry_data={'use_cpu': True})
        self.logger.start_timer()

    def test_lstsq_vs_manual_all(self):
        """
        Test that tecpg_mlr_lstsq produces similar results to regression_full
        for region='all'.
        """
        print("\nRunning regression_full...")
        res_full = regression_full(
            self.M, self.G, self.C, self.M_annot, self.G_annot,
            region='all',
            p_thresh=None,
            methylation_only=True,
            p_only=False,
            logger=self.logger
        )

        print("\nRunning tecpg_mlr_lstsq...")
        res_lstsq = tecpg_mlr_lstsq(
            self.M, self.G, self.C, self.M_annot, self.G_annot,
            region='all',
            p_thresh=None,
            methylation_only=True,
            p_only=False,
            logger=self.logger
        )

        self.compare_dataframes(res_full, res_lstsq)

    def test_lstsq_vs_manual_cis(self):
        """
        Test that tecpg_mlr_lstsq produces similar results to regression_full
        for region='cis'.
        """
        print("\nRunning regression_full (cis)...")
        # Ensure we have window params
        window_base = 0
        upstream = 50000
        downstream = 3000

        res_full = regression_full(
            self.M, self.G, self.C, self.M_annot, self.G_annot,
            region='cis',
            window_base=window_base,
            upstream=upstream,
            downstream=downstream,
            p_thresh=None,
            methylation_only=True,
            p_only=False,
            logger=self.logger
        )

        print("\nRunning tecpg_mlr_lstsq (cis)...")
        res_lstsq = tecpg_mlr_lstsq(
            self.M, self.G, self.C, self.M_annot, self.G_annot,
            region='cis',
            window_base=window_base,
            upstream=upstream,
            downstream=downstream,
            p_thresh=None,
            methylation_only=True,
            p_only=False,
            logger=self.logger
        )

        self.compare_dataframes(res_full, res_lstsq)

    def compare_dataframes(self, df1, df2, tolerance=None):
        # Align indices and columns
        df1 = df1.sort_index().sort_index(axis=1)
        df2 = df2.sort_index().sort_index(axis=1)

        pd.testing.assert_index_equal(df1.index, df2.index)
        pd.testing.assert_index_equal(df1.columns, df2.columns)

        if df1.empty and df2.empty:
            print("Both dataframes are empty.")
            return
        elif df1.empty or df2.empty:
            self.fail(f"One dataframe is empty while the other is not. df1 empty: {df1.empty}, df2 empty: {df2.empty}")

        # Check max difference
        diff = np.abs(df1.values - df2.values)
        max_diff = np.nanmax(diff)
        mean_diff = np.nanmean(diff)

        print(f"Max difference: {max_diff}")
        print(f"Mean difference: {mean_diff}")

        # Assert that max difference is within tolerance
        if tolerance is None:
            if DTYPE == torch.float32:
                tolerance = 2e-3 # Relax tolerance for float32
            else:
                tolerance = 1e-10

        self.assertTrue(max_diff < tolerance, f"Max difference {max_diff} exceeds tolerance {tolerance}")

if __name__ == '__main__':
    unittest.main()
