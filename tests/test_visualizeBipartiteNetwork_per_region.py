import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

# Add the repository root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# tools/visualizeBipartiteNetwork.py imports fa2 and umap at module level; neither
# is used by the code paths under test. Stub them so the module imports in minimal
# environments (repo convention: see tests/test_imports_mocked.py).
sys.modules.setdefault('fa2', MagicMock())
sys.modules.setdefault('umap', MagicMock())

from tools import visualizeBipartiteNetwork as viz


def make_edges(regions=('PROMOTER', 'PROMOTER', 'TRANS'), include_interaction=True,
               below_threshold_region='TRANS'):
    """Three above-threshold edges plus one below-threshold edge.

    The below-threshold edge (mt_ig=0.1 < default threshold 0.5) exists so that
    tests can distinguish 'filtered_edges' from the raw edge table.
    """
    data = {
        'Source': ['cpg1', 'cpg1', 'cpg2', 'cpg3'],
        'Target': ['geneA', 'geneB', 'geneA', 'geneB'],
        'mt_ig': [0.9, 0.8, 0.7, 0.1],
        'mt_est': [-0.5, 0.4, 0.3, -0.2],
    }
    if include_interaction:
        data['Interaction'] = list(regions) + [below_threshold_region]
    return pd.DataFrame(data)


def make_nodes():
    return pd.DataFrame({
        'Node_ID': ['cpg1', 'cpg2', 'cpg3', 'geneA', 'geneB'],
        'Node_Type': ['CpG', 'CpG', 'CpG', 'Gene', 'Gene'],
        'Region': ['PROMOTER', 'TRANS', 'TRANS', 'Undefined', 'Undefined'],
    })


class TestGenerateFiguresPassThrough(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generate_figures_threads_out_dir_to_every_figure(self):
        out = self.temp_dir.name
        edges = make_edges().iloc[:3]
        nodes = make_nodes()
        with patch.object(viz, 'plot_network') as m_net, \
                patch.object(viz, 'plot_umap') as m_umap, \
                patch.object(viz, 'plot_degree_distribution') as m_deg, \
                patch.object(viz, 'plot_bi_adjacency_heatmap') as m_heat, \
                patch.object(viz, 'plot_arc_diagram') as m_arc, \
                patch.object(viz, 'project_bipartite_to_unipartite',
                             return_value=(MagicMock(), pd.DataFrame(
                                 {'Node1': [], 'Node2': [], 'weight': []}))):
            viz.generate_figures(MagicMock(), edges, nodes, 'mt_ig', out)
        self.assertEqual(m_net.call_args.args[1], out)
        self.assertEqual(m_umap.call_args.args[3], out)
        self.assertEqual(m_deg.call_args.args[2], out)
        for call in m_heat.call_args_list:
            self.assertEqual(call.kwargs.get('out_dir'), out)
        self.assertEqual(m_arc.call_args.kwargs.get('out_dir'), out)
        self.assertTrue(os.path.exists(
            os.path.join(out, 'UnipartiteProjection_Edges.csv')))


class TestPerRegionWiring(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_inputs(self, edges):
        edges_path = os.path.join(self.temp_dir.name, 'edges.csv')
        edges.to_csv(edges_path, index=False)
        nodes_path = os.path.join(self.temp_dir.name, 'nodes.csv')
        make_nodes().to_csv(nodes_path, index=False)
        return edges_path, nodes_path

    def _run_main(self, edges_path, nodes_path, per_region):
        argv = ['visualizeBipartiteNetwork.py',
                '--edges', edges_path, '--nodes', nodes_path,
                '--out-dir', self.temp_dir.name, '--threshold', '0.5']
        if per_region:
            argv.append('--per-region')
        with patch.object(viz, 'generate_figures') as mock_gen, \
                patch.object(sys, 'argv', argv):
            viz.main()
        return mock_gen

    def test_default_off_single_pooled_call_no_region_dirs(self):
        mock_gen = self._run_main(*self._write_inputs(make_edges()),
                                  per_region=False)
        self.assertEqual(mock_gen.call_count, 1)
        self.assertEqual(mock_gen.call_args.args[4], self.temp_dir.name)
        subdirs = [d for d in os.listdir(self.temp_dir.name)
                   if d.startswith('region_')]
        self.assertEqual(subdirs, [])

    def test_per_region_calls_per_stratum_with_correct_subsets(self):
        mock_gen = self._run_main(*self._write_inputs(make_edges()),
                                  per_region=True)
        # 1 pooled + PROMOTER + TRANS
        self.assertEqual(mock_gen.call_count, 3)
        pooled = mock_gen.call_args_list[0]
        self.assertEqual(len(pooled.args[1]), 3)
        self.assertEqual(pooled.args[4], self.temp_dir.name)
        by_dir = {os.path.basename(c.args[4]): c.args[1]
                  for c in mock_gen.call_args_list[1:]}
        self.assertEqual(set(by_dir), {'region_PROMOTER', 'region_TRANS'})
        self.assertEqual(len(by_dir['region_PROMOTER']), 2)
        self.assertEqual(len(by_dir['region_TRANS']), 1)
        for name, df in by_dir.items():
            self.assertEqual(set(df['Interaction'].astype(str)),
                             {name.replace('region_', '')})
            self.assertTrue(os.path.isdir(
                os.path.join(self.temp_dir.name, name)))

    def test_per_region_subsets_are_disjoint_and_cover_filtered(self):
        mock_gen = self._run_main(*self._write_inputs(make_edges()),
                                  per_region=True)
        pooled_len = len(mock_gen.call_args_list[0].args[1])
        region_lens = [len(c.args[1]) for c in mock_gen.call_args_list[1:]]
        self.assertEqual(sum(region_lens), pooled_len)

    def test_per_region_skips_with_warning_when_interaction_absent(self):
        edges = make_edges(include_interaction=False)
        with self.assertLogs(level='WARNING') as cm:
            mock_gen = self._run_main(*self._write_inputs(edges),
                                      per_region=True)
        self.assertEqual(mock_gen.call_count, 1)
        self.assertTrue(any('Interaction' in msg for msg in cm.output))

    def test_region_directory_name_is_sanitized(self):
        edges = make_edges(regions=('CIS/5', 'CIS/5', 'CIS/5'),
                           below_threshold_region='CIS/5')
        self._run_main(*self._write_inputs(edges), per_region=True)
        self.assertTrue(os.path.isdir(
            os.path.join(self.temp_dir.name, 'region_CIS_5')))
        self.assertFalse(os.path.isdir(
            os.path.join(self.temp_dir.name, 'region_CIS')))


if __name__ == '__main__':
    unittest.main()
