import unittest
import torch
import pandas as pd
import numpy as np
import sys
import os

# Ensure we import the local tecpg module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tecpg.helper import logit_transform_torch, logit_transform_pandas
from tecpg.logger import Logger

class TestLogitTransform(unittest.TestCase):

    def test_mathematical_invariants(self):
        """
        Verify the mathematical invariants for the logit transform.
        Values derived from user requirements.
        """
        beta_values = [0.5, 0.75, 0.25, 0.9, 0.1]
        expected_m_values = [0.0, 1.584963, -1.584963, 3.169925, -3.169925]

        # Test Torch implementation
        beta_tensor = torch.tensor(beta_values, dtype=torch.float64)
        m_tensor = logit_transform_torch(beta_tensor)

        for i, val in enumerate(m_tensor):
            self.assertAlmostEqual(val.item(), expected_m_values[i], places=6,
                                   msg=f"Torch: Mismatch for beta={beta_values[i]}")

        # Test Pandas implementation
        beta_df = pd.DataFrame(beta_values)
        m_df = logit_transform_pandas(beta_df)

        for i, val in enumerate(m_df[0]):
            self.assertAlmostEqual(val, expected_m_values[i], places=6,
                                   msg=f"Pandas: Mismatch for beta={beta_values[i]}")

    def test_symmetry_check(self):
        """
        Verify that M(beta) + M(1 - beta) is approximately 0.
        """
        betas = [0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.05, 0.95]

        # Torch
        beta_tensor = torch.tensor(betas, dtype=torch.float64)
        m_tensor = logit_transform_torch(beta_tensor)

        n = len(betas)
        for i in range(0, n, 2):
            val1 = m_tensor[i]
            val2 = m_tensor[i+1]
            sum_val = val1 + val2
            self.assertAlmostEqual(sum_val.item(), 0.0, places=6,
                                   msg=f"Torch: Symmetry failure for beta={betas[i]}")

        # Pandas
        beta_df = pd.DataFrame(betas)
        m_df = logit_transform_pandas(beta_df)

        for i in range(0, n, 2):
            val1 = m_df.iloc[i, 0]
            val2 = m_df.iloc[i+1, 0]
            sum_val = val1 + val2
            self.assertAlmostEqual(sum_val, 0.0, places=6,
                                   msg=f"Pandas: Symmetry failure for beta={betas[i]}")

    def test_clamping_boundary_validation(self):
        """
        Verify clamping prevents NaN/Inf and keeps values within reasonable range.
        Epsilon is 1e-6, so max M-value should be approx 19.93.
        """
        epsilon = 1e-6
        # Test extreme values including 0 and 1
        betas = [0.0, 1.0, 1e-7, 1 - 1e-7, 0.5]

        # Torch
        beta_tensor = torch.tensor(betas, dtype=torch.float64)
        m_tensor = logit_transform_torch(beta_tensor, epsilon=epsilon)

        self.assertTrue(torch.isfinite(m_tensor).all(), "Torch: Inf/NaN detected in M-values")
        self.assertTrue((m_tensor < 20.0).all(), "Torch: Clamping failed: M-value too high")
        self.assertTrue((m_tensor > -20.0).all(), "Torch: Clamping failed: M-value too low")

        # Check specifically that 0.0 maps to approx -19.93 and 1.0 to approx 19.93
        val_0 = m_tensor[0].item()
        val_1 = m_tensor[1].item()

        # log2(epsilon / (1-epsilon)) approx log2(1e-6) = -19.9315686
        expected_extreme = np.log2(epsilon / (1 - epsilon))

        self.assertAlmostEqual(val_0, expected_extreme, places=4, msg="Torch: 0.0 clamping check failed")
        self.assertAlmostEqual(val_1, -expected_extreme, places=4, msg="Torch: 1.0 clamping check failed")

        # Pandas
        beta_df = pd.DataFrame(betas)
        m_df = logit_transform_pandas(beta_df, epsilon=epsilon)

        m_vals = m_df[0].values
        self.assertTrue(np.isfinite(m_vals).all(), "Pandas: Inf/NaN detected in M-values")
        self.assertTrue((m_vals < 20.0).all(), "Pandas: Clamping failed: M-value too high")
        self.assertTrue((m_vals > -20.0).all(), "Pandas: Clamping failed: M-value too low")

        val_0_pd = m_vals[0]
        val_1_pd = m_vals[1]

        self.assertAlmostEqual(val_0_pd, expected_extreme, places=4, msg="Pandas: 0.0 clamping check failed")
        self.assertAlmostEqual(val_1_pd, -expected_extreme, places=4, msg="Pandas: 1.0 clamping check failed")

if __name__ == '__main__':
    unittest.main()
