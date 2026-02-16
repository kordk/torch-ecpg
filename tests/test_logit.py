import os
import sys
import unittest
import pandas as pd
import numpy as np
import torch

# Ensure we can import tecpg
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from tecpg.test_data import generate_data
from tecpg.regression_full import regression_full
from tecpg.regression_single import regression_single
from tecpg.processing import tecpg_mlr_lstsq
from tecpg.helper import logit_transform_pandas
from tecpg.logger import Logger

class TestLogitTransform(unittest.TestCase):
    def setUp(self):
        # Set seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        # Generate small dummy data
        self.sample_size = 50
        self.m_loci = 20
        self.g_loci = 10
        self.M, self.G, self.C, self.M_annot, self.G_annot = generate_data(
            self.sample_size, self.m_loci, self.g_loci, annotation=True
        )
        self.logger = Logger(carry_data={'use_cpu': True})
        self.logger.start_timer()

    def test_logit_transform_full(self):
        # Manual transform mimicking internal float32 precision
        # regression_full converts M to float32 tensor BEFORE transform when using flag.
        # So we must cast to float32 first to match precision.
        M_f32 = self.M.astype('float32')
        M_trans = logit_transform_pandas(M_f32)

        # Run regression_full with manual transform
        res_manual = regression_full(
            M_trans, self.G, self.C,
            self.M_annot, self.G_annot,
            region='all', p_thresh=None,
            logger=self.logger,
            logit_transform=False
        )

        # Run regression_full with flag
        res_flag = regression_full(
            self.M, self.G, self.C,
            self.M_annot, self.G_annot,
            region='all', p_thresh=None,
            logger=self.logger,
            logit_transform=True
        )

        # Compare results
        # We expect them to be identical or very close
        pd.testing.assert_frame_equal(res_manual, res_flag, check_exact=False, rtol=1e-3)

        # Ensure they are DIFFERENT from non-transformed results
        res_raw = regression_full(
            self.M, self.G, self.C,
            self.M_annot, self.G_annot,
            region='all', p_thresh=None,
            logger=self.logger,
            logit_transform=False
        )

        # Check estimates are different
        self.assertFalse(np.allclose(res_raw['mt_est'], res_flag['mt_est'], rtol=1e-3))

    def test_logit_transform_lstsq(self):
        # Manual transform mimicking internal float32 precision
        M_f32 = self.M.astype('float32')
        M_trans = logit_transform_pandas(M_f32)

        # Run lstsq with manual transform
        res_manual = tecpg_mlr_lstsq(
            M_trans, self.G, self.C,
            self.M_annot, self.G_annot,
            region='all', p_thresh=None,
            logger=self.logger,
            logit_transform=False
        )

        # Run lstsq with flag
        res_flag = tecpg_mlr_lstsq(
            self.M, self.G, self.C,
            self.M_annot, self.G_annot,
            region='all', p_thresh=None,
            logger=self.logger,
            logit_transform=True
        )

        # Compare results
        pd.testing.assert_frame_equal(res_manual, res_flag, check_exact=False, rtol=1e-3)

        # Ensure difference from raw
        res_raw = tecpg_mlr_lstsq(
            self.M, self.G, self.C,
            self.M_annot, self.G_annot,
            region='all', p_thresh=None,
            logger=self.logger,
            logit_transform=False
        )
        self.assertFalse(np.allclose(res_raw['mt_est'], res_flag['mt_est'], rtol=1e-3))

    def test_logit_transform_single(self):
        # Manual transform
        # regression_single uses pandas for transform internally (float64), so manual transform on float64 M matches perfectly.
        M_trans = logit_transform_pandas(self.M)

        # Run single with manual transform
        # regression_single returns concatenated DF.
        res_manual = regression_single(
            M_trans, self.G, self.C,
            region='all', p_thresh=None,
            logger=self.logger,
            logit_transform=False
        )

        # Run single with flag
        res_flag = regression_single(
            self.M, self.G, self.C,
            region='all', p_thresh=None,
            logger=self.logger,
            logit_transform=True
        )

        # Compare results
        pd.testing.assert_frame_equal(res_manual, res_flag, check_exact=False, rtol=1e-3)

        # Ensure difference from raw
        res_raw = regression_single(
            self.M, self.G, self.C,
            region='all', p_thresh=None,
            logger=self.logger,
            logit_transform=False
        )
        self.assertFalse(np.allclose(
            res_raw['mt_est'].astype(float),
            res_flag['mt_est'].astype(float),
            rtol=1e-3
        ))

if __name__ == '__main__':
    unittest.main()
