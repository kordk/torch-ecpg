import math
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import networkx as nx
import pandas as pd

# Add the repository root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# tools/visualizeBipartiteNetwork.py imports fa2 and umap at module level; neither
# is used by the code paths under test. Stub them so the module imports in minimal
# environments (repo convention: see tests/test_imports_mocked.py).
sys.modules.setdefault('fa2', MagicMock())
sys.modules.setdefault('umap', MagicMock())

from tools import visualizeBipartiteNetwork as viz


def make_edges():
    return pd.DataFrame({
        'Source': ['cpg1', 'cpg1', 'cpg2'],
        'Target': ['geneA', 'geneB', 'geneA'],
        'mt_ig': [0.9, 0.8, 0.7],
        'mt_est': [-0.5, 0.4, 0.3],
        'Interaction': ['PROMOTER', 'PROMOTER', 'TRANS'],
    })


def make_nodes():
    return pd.DataFrame({
        'Node_ID': ['cpg1', 'cpg2', 'geneA', 'geneB'],
        'Node_Type': ['CpG', 'CpG', 'Gene', 'Gene'],
        'Region': ['PROMOTER', 'TRANS', 'Undefined', 'Undefined'],
    })


class TestGenerateFiguresHypergeomWiring(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _harness_patches(self):
        return [
            patch.object(viz, 'plot_network'),
            patch.object(viz, 'plot_umap'),
            patch.object(viz, 'plot_degree_distribution'),
            patch.object(viz, 'plot_bi_adjacency_heatmap'),
            patch.object(viz, 'plot_arc_diagram'),
        ]

    def test_projection_called_with_count_then_hypergeometric(self):
        patches = self._harness_patches()
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with patch.object(viz, 'plot_unipartite_projection'), \
                patch.object(viz, 'project_bipartite_to_unipartite',
                             return_value=(MagicMock(), pd.DataFrame(
                                 {'Node1': [], 'Node2': [], 'weight': []}))) as m_proj:
            viz.generate_figures(MagicMock(), make_edges(), make_nodes(),
                                 'mt_ig', self.temp_dir.name)
        methods = [c.kwargs.get('weight_method')
                   for c in m_proj.call_args_list]
        self.assertEqual(methods, ['count', 'hypergeometric'])

    def test_plot_called_for_both_projections_with_distinct_filenames(self):
        patches = self._harness_patches()
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with patch.object(viz, 'plot_unipartite_projection') as m_plot, \
                patch.object(viz, 'project_bipartite_to_unipartite',
                             return_value=(MagicMock(), pd.DataFrame(
                                 {'Node1': [], 'Node2': [], 'weight': []}))):
            viz.generate_figures(MagicMock(), make_edges(), make_nodes(),
                                 'mt_ig', self.temp_dir.name)
        self.assertEqual(m_plot.call_count, 2)
        first, second = m_plot.call_args_list
        self.assertIsNone(first.kwargs.get('out_name'))
        self.assertEqual(second.kwargs.get('out_name'),
                         'UnipartiteProjectionHypergeom.png')

    def test_hypergeom_edge_list_written_additively(self):
        patches = self._harness_patches()
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with patch.object(viz, 'plot_unipartite_projection'):
            viz.generate_figures(MagicMock(), make_edges(), make_nodes(),
                                 'mt_ig', self.temp_dir.name)
        self.assertTrue(os.path.exists(os.path.join(
            self.temp_dir.name, 'UnipartiteProjection_Edges.csv')))
        self.assertTrue(os.path.exists(os.path.join(
            self.temp_dir.name, 'UnipartiteProjection_Hypergeom_Edges.csv')))


class TestPlotUnipartiteProjection(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.graph = nx.Graph()
        self.graph.add_edge('geneA', 'geneB', weight=1.0)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_filename_preserved(self):
        viz.plot_unipartite_projection(self.graph, self.temp_dir.name)
        self.assertTrue(os.path.exists(os.path.join(
            self.temp_dir.name, 'UnipartiteProjection.png')))

    def test_custom_filename_does_not_touch_default(self):
        viz.plot_unipartite_projection(
            self.graph, self.temp_dir.name,
            out_name='UnipartiteProjectionHypergeom.png',
            title='Unipartite Projection (Genes, Hypergeometric)')
        self.assertTrue(os.path.exists(os.path.join(
            self.temp_dir.name, 'UnipartiteProjectionHypergeom.png')))
        self.assertFalse(os.path.exists(os.path.join(
            self.temp_dir.name, 'UnipartiteProjection.png')))


class TestHypergeomWeightSemantics(unittest.TestCase):
    def test_weight_matches_independent_computation(self):
        # geneA regulators {c1,c2,c3}; geneB regulators {c1,c2}; geneC {c4}.
        # Universe M=4 CpGs. Pair (A,B): k=2 shared, n=3, N=2.
        # P(X >= 2) with population 4, successes 3, draws 2:
        #   C(3,2)*C(1,0)/C(4,2) = 3/6 = 0.5  ->  weight = -log10(0.5)
        df = pd.DataFrame({
            'Source': ['c1', 'c2', 'c3', 'c1', 'c2', 'c4'],
            'Target': ['geneA', 'geneA', 'geneA', 'geneB', 'geneB', 'geneC'],
            'weight': [1.0] * 6,
        })
        _, proj_df = viz.project_bipartite_to_unipartite(
            df, cpg_col='Source', gene_col='Target', weight_col='weight',
            target='gene', weight_method='hypergeometric')
        self.assertEqual(len(proj_df), 1)
        row = proj_df.iloc[0]
        self.assertEqual({row['Node1'], row['Node2']}, {'geneA', 'geneB'})
        self.assertAlmostEqual(row['weight'], -math.log10(0.5), places=10)


if __name__ == '__main__':
    unittest.main()
