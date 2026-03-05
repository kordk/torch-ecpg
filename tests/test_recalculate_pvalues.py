import os
import subprocess
import sys
import unittest
import pandas as pd
import numpy as np
from scipy.stats import t

class TestRecalculatePValues(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'tests/temp_test_recalc'
        os.makedirs(self.test_dir, exist_ok=True)
        self.input_file = os.path.join(self.test_dir, 'test_input.csv')
        self.output_file = os.path.join(self.test_dir, 'test_output.csv')

        # Create a dummy CSV
        # mt_est, mt_err, mt_t, mt_p (dummy)
        data = {
            'gt_id': ['G1', 'G2', 'G3', 'G4'],
            'mt_id': ['M1', 'M2', 'M3', 'M4'],
            'mt_est': [0.5, -0.5, 0.0, 10.0],
            'mt_err': [0.1, 0.1, 0.1, 0.1],
            'mt_t': [5.0, -5.0, 0.0, 100.0],
            'mt_p': [0.0, 0.0, 1.0, 0.0] # Dummy values to be overwritten
        }
        df = pd.DataFrame(data)
        df.to_csv(self.input_file, index=False)

    def tearDown(self):
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_recalculation(self):
        n_patients = 100
        n_covariates = 5
        # Expected degrees of freedom:
        # df = n_patients - n_covariates - 2
        # df = 100 - 5 - 2 = 93
        expected_df = 93

        # Run the script
        # Using sys.executable ensures we use the same python interpreter
        cmd = [
            sys.executable,
            'tools/recalculate_pvalues.py',
            self.input_file,
            '--n-patients', str(n_patients),
            '--n-covariates', str(n_covariates),
            '--output-file', self.output_file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Print output if it failed
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")

        self.assertEqual(result.returncode, 0, f"Script failed with return code {result.returncode}")

        # Read output
        self.assertTrue(os.path.exists(self.output_file), "Output file was not created")
        output_df = pd.read_csv(self.output_file)

        # Check that other columns are preserved
        self.assertTrue('gt_id' in output_df.columns)
        self.assertEqual(output_df['gt_id'][0], 'G1')

        # Calculate expected p-values manually
        t_stats = np.array([5.0, -5.0, 0.0, 100.0])
        expected_p = t.sf(np.abs(t_stats), expected_df) * 2

        # Compare
        # Using a very tight tolerance because we expect float64 precision match
        pd.testing.assert_series_equal(
            output_df['mt_p'],
            pd.Series(expected_p, name='mt_p'),
            rtol=1e-12,
            atol=1e-12
        )

    def test_recalculation_default_output(self):
        n_patients = 50
        n_covariates = 2

        # expected output filename
        # test_input.csv -> test_input_recalc.csv
        base, ext = os.path.splitext(self.input_file)
        expected_output_file = f"{base}_recalc{ext}"

        # Run the script without --output-file
        cmd = [
            sys.executable,
            'tools/recalculate_pvalues.py',
            self.input_file,
            '--n-patients', str(n_patients),
            '--n-covariates', str(n_covariates)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)

        self.assertTrue(os.path.exists(expected_output_file))

        # Cleanup specific to this test
        if os.path.exists(expected_output_file):
            os.remove(expected_output_file)

if __name__ == '__main__':
    unittest.main()
