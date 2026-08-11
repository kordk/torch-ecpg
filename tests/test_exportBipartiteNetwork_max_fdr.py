import math
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

# Add the repository root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools import exportBipartiteNetwork


def base_columns(n):
    return {
        'mt_id': [f'cpg{i + 1}' for i in range(n)],
        'mt_chrom': ['chr1'] * n,
        'mt_chromStart': [100 * (i + 1) for i in range(n)],
        'mt_strand': ['+'] * n,
        'gt_id': [f'gene{chr(65 + i)}' for i in range(n)],
        'gt_chrom': ['chr1'] * n,
        'gt_chromStart': [500 + 100 * i for i in range(n)],
        'gt_strand': ['+'] * n,
        'mt_est': [0.5] * n,
        'mt_p': [0.01] * n,
    }


class TestMaxFdrFilter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_file = os.path.join(self.temp_dir.name, 'input.parquet')
        self.out_prefix = os.path.join(self.temp_dir.name, 'cytoscape')
        self.out_edges = f'{self.out_prefix}_edges.csv'

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_parquet(self, data):
        pd.DataFrame(data).to_parquet(self.input_file)

    def _run(self, extra_args):
        argv = ['exportBipartiteNetwork.py', '-i', self.input_file,
                '-o', self.out_prefix] + extra_args
        with patch('sys.argv', argv):
            exportBipartiteNetwork.main()
        return pd.read_csv(self.out_edges)

    def test_max_fdr_filters_rows(self):
        data = base_columns(4)
        data['mt_ig'] = [1.0, 2.0, 3.0, 4.0]
        data['fdr_est'] = [0.01, 0.2, 0.04, float('nan')]
        self._write_parquet(data)
        edges = self._run(['--max-fdr', '0.05'])
        sources = edges['Source'].tolist()
        self.assertEqual(len(edges), 2)
        self.assertIn('cpg1', sources)
        self.assertIn('cpg3', sources)
        self.assertNotIn('cpg2', sources)
        self.assertNotIn('cpg4', sources)

    def test_max_fdr_nan_rows_excluded(self):
        data = base_columns(2)
        data['mt_ig'] = [1.0, 2.0]
        data['fdr_est'] = [0.01, float('nan')]
        self._write_parquet(data)
        edges = self._run(['--max-fdr', '0.05'])
        self.assertEqual(edges['Source'].tolist(), ['cpg1'])

    def test_max_fdr_missing_column_warns_and_skips(self):
        data = base_columns(3)
        data['mt_ig'] = [1.0, 2.0, 3.0]
        # No fdr_est column
        self._write_parquet(data)
        with self.assertLogs(level='WARNING') as cm:
            edges = self._run(['--max-fdr', '0.05'])
        self.assertEqual(len(edges), 3)
        self.assertTrue(any('fdr_est' in msg for msg in cm.output))

    def test_no_max_fdr_flag_leaves_universe_unchanged(self):
        data = base_columns(4)
        data['mt_ig'] = [1.0, 2.0, 3.0, 4.0]
        data['fdr_est'] = [0.01, 0.2, 0.04, float('nan')]
        self._write_parquet(data)
        edges = self._run([])
        self.assertEqual(len(edges), 4)

    def test_max_fdr_applied_before_top_k(self):
        # Highest-IG row fails FDR; with --max-fdr active, --top-k 1 must select
        # the top-IG row among FDR-surviving rows, not the global top-IG row.
        data = base_columns(3)
        data['mt_ig'] = [1.0, 9.0, 2.0]
        data['fdr_est'] = [0.01, 0.9, 0.02]
        self._write_parquet(data)
        edges = self._run(['--max-fdr', '0.05', '--top-k', '1'])
        self.assertEqual(edges['Source'].tolist(), ['cpg3'])


if __name__ == '__main__':
    unittest.main()
