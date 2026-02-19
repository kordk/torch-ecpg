import os
import shutil
import subprocess
import sys
import unittest
import pandas as pd
import numpy as np

# Adjust path to import the tool if needed, but we will run it as a subprocess
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL_PATH = os.path.join(REPO_ROOT, 'tecpg', 'tools', 'mergeOutputs.py')

class TestMergeTool(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_merge_output'
        self.input_dir = os.path.join(self.test_dir, 'input')
        self.output_file = os.path.join(self.test_dir, 'merged_output.csv')

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.input_dir)

        # Create dummy data
        self.create_dummy_data()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_dummy_data(self):
        self.chunks = []
        self.expected_mappings = 0
        self.expected_genes = set()
        self.expected_cpgs = set()

        # Create 3 chunks with overlap
        # Chunk 1
        df1 = pd.DataFrame({
            'gt_id': ['g1', 'g1', 'g2'],
            'mt_id': ['c1', 'c2', 'c1'],
            'val': [0.1, 0.2, 0.3]
        })
        self.save_chunk(df1, '1-1.csv')

        # Chunk 2 (ensure natural sort order checks: 1-2 vs 1-10)
        df2 = pd.DataFrame({
            'gt_id': ['g2', 'g3'],
            'mt_id': ['c2', 'c3'],
            'val': [0.4, 0.5]
        })
        self.save_chunk(df2, '1-2.csv')

        # Chunk 3 (late chunk)
        df3 = pd.DataFrame({
            'gt_id': ['g1'],
            'mt_id': ['c3'],
            'val': [0.6]
        })
        self.save_chunk(df3, '1-10.csv')

    def save_chunk(self, df, filename):
        filepath = os.path.join(self.input_dir, filename)
        df.to_csv(filepath, index=False)
        self.chunks.append(filepath)

        self.expected_mappings += len(df)
        self.expected_genes.update(df['gt_id'])
        self.expected_cpgs.update(df['mt_id'])

    def test_merge_tool(self):
        # Run the tool
        cmd = [sys.executable, TOOL_PATH, self.input_dir, self.output_file]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check return code
        self.assertEqual(result.returncode, 0, f"Tool failed with stderr: {result.stderr}")

        # Check output file existence
        self.assertTrue(os.path.exists(self.output_file))

        # Check merged content
        merged_df = pd.read_csv(self.output_file)

        # Verify total rows
        self.assertEqual(len(merged_df), self.expected_mappings)

        # Verify columns
        expected_cols = ['gt_id', 'mt_id', 'val']
        self.assertEqual(list(merged_df.columns), expected_cols)

        # Verify order (1-1, then 1-2, then 1-10)
        # 1-1: 3 rows
        # 1-2: 2 rows
        # 1-10: 1 row
        # Total 6 rows
        # We can check specific values to verify order
        # First 3 rows should be from 1-1
        self.assertEqual(merged_df.iloc[0]['gt_id'], 'g1')
        self.assertEqual(merged_df.iloc[2]['gt_id'], 'g2')
        # Next 2 rows from 1-2
        self.assertEqual(merged_df.iloc[3]['gt_id'], 'g2')
        # Last row from 1-10
        self.assertEqual(merged_df.iloc[5]['gt_id'], 'g1')
        self.assertEqual(merged_df.iloc[5]['mt_id'], 'c3')

        # Verify stats in stdout
        output = result.stdout
        self.assertIn(f"Total Mappings (rows): {self.expected_mappings}", output)
        self.assertIn(f"Unique Genes (gt_id): {len(self.expected_genes)}", output)
        self.assertIn(f"Unique CpGs (mt_id): {len(self.expected_cpgs)}", output)

if __name__ == '__main__':
    unittest.main()
