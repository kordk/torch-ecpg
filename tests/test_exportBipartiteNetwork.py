import os
import sys
import tempfile
import unittest
import pandas as pd
from unittest.mock import patch
import io

# Add the tools directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools import exportBipartiteNetwork

class TestExportCytoscape(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_file = os.path.join(self.temp_dir.name, "input.parquet")
        self.out_prefix = os.path.join(self.temp_dir.name, "cytoscape")
        self.out_edges = f"{self.out_prefix}_edges.csv"
        self.out_nodes = f"{self.out_prefix}_nodes.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_dummy_parquet(self, data, filename):
        df = pd.DataFrame(data)
        df.to_parquet(filename)

    def test_basic_success_with_mt_ig(self):
        data = {
            'mt_id': ['cpg1', 'cpg2'],
            'mt_chrom': ['chr1', 'chr2'],
            'mt_chromStart': [100, 200],
            'mt_strand': ['+', '-'],
            'gt_id': ['geneA', 'geneB'],
            'gt_chrom': ['chr1', 'chr2'],
            'gt_chromStart': [500, 600],
            'gt_strand': ['+', '-'],
            'mt_est': [0.5, -0.3],
            'mt_p': [0.01, 0.05],
            'region': ['PROMOTER', 'DISTAL5'],
            'mt_ig': [1.5, 2.5],  # cpg2 should sort first
            'age_ig': [0.1, 0.2]
        }
        self.create_dummy_parquet(data, self.input_file)

        test_args = ['exportBipartiteNetwork.py', '-i', self.input_file, '-o', self.out_prefix, '--top-k', '10']
        with patch('sys.argv', test_args):
            exportBipartiteNetwork.main()

        self.assertTrue(os.path.exists(self.out_edges))
        self.assertTrue(os.path.exists(self.out_nodes))

        edges_df = pd.read_csv(self.out_edges)
        nodes_df = pd.read_csv(self.out_nodes)

        # Check edge df columns
        expected_edge_cols = ['Source', 'Target', 'Interaction', 'mt_est', 'mt_p', 'mt_ig', 'age_ig']
        for col in expected_edge_cols:
            self.assertIn(col, edges_df.columns)

        # cpg2 should be first due to mt_ig sorting
        self.assertEqual(edges_df.iloc[0]['Source'], 'cpg2')

        # Check nodes
        self.assertEqual(len(nodes_df), 4) # 2 cpgs + 2 genes
        expected_node_cols = ['Node_ID', 'Chrom', 'Start', 'Strand', 'Node_Type']
        for col in expected_node_cols:
            self.assertIn(col, nodes_df.columns)

    def test_fallback_mt_t_generates_abs_t(self):
        data = {
            'mt_id': ['cpg1', 'cpg2'],
            'mt_chrom': ['chr1', 'chr2'],
            'mt_chromStart': [100, 200],
            'mt_strand': ['+', '-'],
            'gt_id': ['geneA', 'geneB'],
            'gt_chrom': ['chr1', 'chr2'],
            'gt_chromStart': [500, 600],
            'gt_strand': ['+', '-'],
            'mt_est': [0.5, -0.3],
            'mt_p': [0.01, 0.05],
            'region': ['PROMOTER', 'DISTAL5'],
            'mt_t': [-5.0, 2.0] # cpg1 abs_t is 5.0, should be first
        }
        self.create_dummy_parquet(data, self.input_file)

        test_args = ['exportBipartiteNetwork.py', '-i', self.input_file, '-o', self.out_prefix]
        with patch('sys.argv', test_args):
            exportBipartiteNetwork.main()

        edges_df = pd.read_csv(self.out_edges)

        self.assertIn('mt_t', edges_df.columns)
        self.assertIn('abs_t', edges_df.columns)

        # cpg1 should be first
        self.assertEqual(edges_df.iloc[0]['Source'], 'cpg1')
        self.assertEqual(edges_df.iloc[0]['abs_t'], 5.0)

    def test_missing_node_cols(self):
        # Missing gt_chrom
        data = {
            'mt_id': ['cpg1'], 'mt_chrom': ['chr1'], 'mt_chromStart': [100], 'mt_strand': ['+'],
            'gt_id': ['geneA'], 'gt_chromStart': [500], 'gt_strand': ['+'],
            'mt_est': [0.5], 'mt_p': [0.01], 'mt_ig': [1.5]
        }
        self.create_dummy_parquet(data, self.input_file)

        test_args = ['exportBipartiteNetwork.py', '-i', self.input_file, '-o', self.out_prefix]
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                exportBipartiteNetwork.main()
            self.assertEqual(cm.exception.code, 1)

    def test_missing_edge_cols(self):
        # Missing mt_p
        data = {
            'mt_id': ['cpg1'], 'mt_chrom': ['chr1'], 'mt_chromStart': [100], 'mt_strand': ['+'],
            'gt_id': ['geneA'], 'gt_chrom': ['chr1'], 'gt_chromStart': [500], 'gt_strand': ['+'],
            'mt_est': [0.5], 'mt_ig': [1.5]
        }
        self.create_dummy_parquet(data, self.input_file)

        test_args = ['exportBipartiteNetwork.py', '-i', self.input_file, '-o', self.out_prefix]
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                exportBipartiteNetwork.main()
            self.assertEqual(cm.exception.code, 1)

    def test_missing_mt_ig_and_mt_t(self):
        # No mt_ig, no mt_t
        data = {
            'mt_id': ['cpg1'], 'mt_chrom': ['chr1'], 'mt_chromStart': [100], 'mt_strand': ['+'],
            'gt_id': ['geneA'], 'gt_chrom': ['chr1'], 'gt_chromStart': [500], 'gt_strand': ['+'],
            'mt_est': [0.5], 'mt_p': [0.01]
        }
        self.create_dummy_parquet(data, self.input_file)

        test_args = ['exportBipartiteNetwork.py', '-i', self.input_file, '-o', self.out_prefix]
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                exportBipartiteNetwork.main()
            self.assertEqual(cm.exception.code, 1)

    def test_default_region_undefined(self):
        data = {
            'mt_id': ['cpg1'], 'mt_chrom': ['chr1'], 'mt_chromStart': [100], 'mt_strand': ['+'],
            'gt_id': ['geneA'], 'gt_chrom': ['chr1'], 'gt_chromStart': [500], 'gt_strand': ['+'],
            'mt_est': [0.5], 'mt_p': [0.01], 'mt_ig': [1.5]
            # No region column
        }
        self.create_dummy_parquet(data, self.input_file)

        test_args = ['exportBipartiteNetwork.py', '-i', self.input_file, '-o', self.out_prefix]
        with patch('sys.argv', test_args):
            exportBipartiteNetwork.main()

        edges_df = pd.read_csv(self.out_edges)
        self.assertIn('Interaction', edges_df.columns)
        self.assertEqual(edges_df.iloc[0]['Interaction'], 'Undefined')

    def test_min_effect_filter(self):
        data = {
            'mt_id': ['cpg1', 'cpg2', 'cpg3'],
            'mt_chrom': ['chr1', 'chr1', 'chr1'],
            'mt_chromStart': [100, 200, 300],
            'mt_strand': ['+', '+', '+'],
            'gt_id': ['geneA', 'geneB', 'geneC'],
            'gt_chrom': ['chr1', 'chr1', 'chr1'],
            'gt_chromStart': [500, 600, 700],
            'gt_strand': ['+', '+', '+'],
            'mt_est': [0.5, -0.3, 0.1],  # cpg3 should be filtered out by min_effect=0.2
            'mt_p': [0.01, 0.05, 0.01],
            'mt_ig': [1.5, 2.5, 3.5]
        }
        self.create_dummy_parquet(data, self.input_file)

        test_args = ['exportBipartiteNetwork.py', '-i', self.input_file, '-o', self.out_prefix, '--min-effect', '0.2']
        with patch('sys.argv', test_args):
            exportBipartiteNetwork.main()

        edges_df = pd.read_csv(self.out_edges)
        nodes_df = pd.read_csv(self.out_nodes)

        self.assertEqual(len(edges_df), 2)
        # Should contain cpg1 and cpg2, but not cpg3
        sources = edges_df['Source'].tolist()
        self.assertIn('cpg1', sources)
        self.assertIn('cpg2', sources)
        self.assertNotIn('cpg3', sources)

        # Verify isolated node is excluded
        node_ids = nodes_df['Node_ID'].tolist()
        self.assertNotIn('cpg3', node_ids)
        self.assertNotIn('geneC', node_ids)

    def test_max_boot_p_filter(self):
        data = {
            'mt_id': ['cpg1', 'cpg2', 'cpg3'],
            'mt_chrom': ['chr1', 'chr1', 'chr1'],
            'mt_chromStart': [100, 200, 300],
            'mt_strand': ['+', '+', '+'],
            'gt_id': ['geneA', 'geneB', 'geneC'],
            'gt_chrom': ['chr1', 'chr1', 'chr1'],
            'gt_chromStart': [500, 600, 700],
            'gt_strand': ['+', '+', '+'],
            'mt_est': [0.5, -0.3, 0.1],
            'mt_p': [0.01, 0.05, 0.01],
            'p_boot': [0.01, 0.1, 0.05], # cpg2 should be filtered out by max_boot_p=0.05
            'mt_ig': [1.5, 2.5, 3.5]
        }
        self.create_dummy_parquet(data, self.input_file)

        test_args = ['exportBipartiteNetwork.py', '-i', self.input_file, '-o', self.out_prefix, '--max-boot-p', '0.05']
        with patch('sys.argv', test_args):
            exportBipartiteNetwork.main()

        edges_df = pd.read_csv(self.out_edges)
        nodes_df = pd.read_csv(self.out_nodes)

        self.assertEqual(len(edges_df), 2)
        sources = edges_df['Source'].tolist()
        self.assertIn('cpg1', sources)
        self.assertIn('cpg3', sources)
        self.assertNotIn('cpg2', sources)

        # Verify isolated node is excluded
        node_ids = nodes_df['Node_ID'].tolist()
        self.assertNotIn('cpg2', node_ids)
        self.assertNotIn('geneB', node_ids)

    def test_max_boot_p_filter_missing_column(self):
        data = {
            'mt_id': ['cpg1', 'cpg2', 'cpg3'],
            'mt_chrom': ['chr1', 'chr1', 'chr1'],
            'mt_chromStart': [100, 200, 300],
            'mt_strand': ['+', '+', '+'],
            'gt_id': ['geneA', 'geneB', 'geneC'],
            'gt_chrom': ['chr1', 'chr1', 'chr1'],
            'gt_chromStart': [500, 600, 700],
            'gt_strand': ['+', '+', '+'],
            'mt_est': [0.5, -0.3, 0.1],
            'mt_p': [0.01, 0.05, 0.01],
            'mt_ig': [1.5, 2.5, 3.5]
            # No p_boot column
        }
        self.create_dummy_parquet(data, self.input_file)

        test_args = ['exportBipartiteNetwork.py', '-i', self.input_file, '-o', self.out_prefix, '--max-boot-p', '0.05']
        with patch('sys.argv', test_args):
            exportBipartiteNetwork.main()

        edges_df = pd.read_csv(self.out_edges)

        # All 3 edges should remain because missing p_boot causes the filter to be skipped
        self.assertEqual(len(edges_df), 3)

if __name__ == '__main__':
    unittest.main()
