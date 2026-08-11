import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

# Add the repository root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# tools/visualizeBipartiteNetwork.py imports fa2 and umap at module level; neither
# is used by the biclustered heatmap under test. Stub them so the module imports
# in minimal environments (repo convention: see tests/test_imports_mocked.py).
sys.modules.setdefault('fa2', MagicMock())
sys.modules.setdefault('umap', MagicMock())

from tools import visualizeBipartiteNetwork as viz


def make_signed_edges(include_mt_est=True):
    data = {
        'Source': ['cpg1', 'cpg1', 'cpg2', 'cpg3'],
        'Target': ['geneA', 'geneB', 'geneA', 'geneB'],
        'mt_ig': [0.9, 0.8, 0.7, 0.6],
        'mt_est': [-0.5, 0.4, 0.3, -0.2],
    }
    if not include_mt_est:
        del data['mt_est']
    return pd.DataFrame(data)


class TestSignedHeatmapFunction(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_signed_heatmap_file_written_with_custom_out_name(self):
        viz.plot_bi_adjacency_heatmap(
            make_signed_edges(), cpg_col='Source', gene_col='Target',
            weight_col='mt_est', out_dir=self.temp_dir.name,
            out_name='SignedBiclusteredBiAdjacencyHeatmap.png',
            title='Signed Biclustered Bi-Adjacency Heatmap (mt_est)',
        )
        self.assertTrue(os.path.exists(os.path.join(
            self.temp_dir.name, 'SignedBiclusteredBiAdjacencyHeatmap.png')))
        # Additive invariant: the signed call must not touch the original filename.
        self.assertFalse(os.path.exists(os.path.join(
            self.temp_dir.name, 'BiclusteredBiAdjacencyHeatmap.png')))

    def test_default_call_preserves_original_filename(self):
        viz.plot_bi_adjacency_heatmap(
            make_signed_edges(), cpg_col='Source', gene_col='Target',
            weight_col='mt_ig', out_dir=self.temp_dir.name,
        )
        self.assertTrue(os.path.exists(os.path.join(
            self.temp_dir.name, 'BiclusteredBiAdjacencyHeatmap.png')))

    def test_diverging_cmap_selected_for_signed_weights(self):
        with patch('seaborn.clustermap') as mock_cm, \
                patch('matplotlib.pyplot.close'):
            mock_cm.return_value = MagicMock()
            viz.plot_bi_adjacency_heatmap(
                make_signed_edges(), cpg_col='Source', gene_col='Target',
                weight_col='mt_est', out_dir=self.temp_dir.name,
                out_name='SignedBiclusteredBiAdjacencyHeatmap.png',
                title='Signed Biclustered Bi-Adjacency Heatmap (mt_est)',
            )
            self.assertTrue(mock_cm.called)
            kwargs = mock_cm.call_args.kwargs
            self.assertEqual(kwargs.get('cmap'), 'RdBu_r')
            self.assertEqual(kwargs.get('center'), 0)
            self.assertIsNone(kwargs.get('mask'))


class TestMainWiring(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_inputs(self, include_mt_est=True):
        edges_path = os.path.join(self.temp_dir.name, 'edges.csv')
        make_signed_edges(include_mt_est).to_csv(edges_path, index=False)
        nodes = pd.DataFrame({
            'Node_ID': ['cpg1', 'cpg2', 'cpg3', 'geneA', 'geneB'],
            'Node_Type': ['CpG', 'CpG', 'CpG', 'Gene', 'Gene'],
            'Region': ['PROMOTER', 'TRANS', 'CIS5', 'Undefined', 'Undefined'],
        })
        nodes_path = os.path.join(self.temp_dir.name, 'nodes.csv')
        nodes.to_csv(nodes_path, index=False)
        return edges_path, nodes_path

    def _run_main(self, edges_path, nodes_path):
        argv = ['visualizeBipartiteNetwork.py',
                '--edges', edges_path, '--nodes', nodes_path,
                '--out-dir', self.temp_dir.name, '--threshold', '0.5']
        with patch.object(viz, 'plot_network'), \
                patch.object(viz, 'plot_umap'), \
                patch.object(viz, 'plot_degree_distribution'), \
                patch.object(viz, 'plot_arc_diagram'), \
                patch.object(viz, 'project_bipartite_to_unipartite',
                             return_value=(MagicMock(), pd.DataFrame(
                                 {'Node1': [], 'Node2': [], 'weight': []}))), \
                patch.object(viz, 'plot_bi_adjacency_heatmap') as mock_heat, \
                patch.object(sys, 'argv', argv):
            viz.main()
        return mock_heat

    def test_main_invokes_signed_heatmap_when_mt_est_present(self):
        mock_heat = self._run_main(*self._write_inputs(include_mt_est=True))
        self.assertEqual(mock_heat.call_count, 2)
        signed_kwargs = mock_heat.call_args_list[1].kwargs
        self.assertEqual(signed_kwargs.get('weight_col'), 'mt_est')
        self.assertEqual(signed_kwargs.get('out_name'),
                         'SignedBiclusteredBiAdjacencyHeatmap.png')

    def test_main_skips_signed_heatmap_when_mt_est_absent(self):
        with self.assertLogs(level='WARNING') as cm:
            mock_heat = self._run_main(*self._write_inputs(include_mt_est=False))
        self.assertEqual(mock_heat.call_count, 1)
        self.assertTrue(any('mt_est' in msg for msg in cm.output))


if __name__ == '__main__':
    unittest.main()