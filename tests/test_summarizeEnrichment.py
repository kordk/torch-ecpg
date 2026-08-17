import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools import summarizeEnrichment as se


def _terms(n, prefix='T', start_p=1e-6):
    # ascending adjusted p; deliberately written out of order to test sorting
    rows = [{'Term': f'{prefix}{i}', 'Overlap': f'{i + 1}/100', 'P-value': start_p * (i + 1),
             'Adjusted P-value': start_p * 2 * (i + 1),
             'Genes': ';'.join(f'G{j}' for j in range(15))} for i in range(n)]
    df = pd.DataFrame(rows)
    return df.iloc[::-1].reset_index(drop=True)  # reversed order on disk


class TestFilenameParsing(unittest.TestCase):
    def test_library_with_underscores(self):
        self.assertEqual(se.parse_enrichment_filename('PROMOTER_fdr_GO_Biological_Process_2021_enrichment.csv'),
                         ('PROMOTER', 'fdr', 'GO_Biological_Process_2021'))

    def test_ig_method(self):
        self.assertEqual(se.parse_enrichment_filename('TRANS_ig_KEGG_2021_Human_enrichment.csv'),
                         ('TRANS', 'ig', 'KEGG_2021_Human'))

    def test_non_matching_returns_none(self):
        self.assertIsNone(se.parse_enrichment_filename('PROMOTER_KEGG_comparison.txt'))
        self.assertIsNone(se.parse_enrichment_filename('encode_enrichment_results.csv'))


class TestReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.enr_dir = os.path.join(self.tmp.name, 'enrichment')
        self.res_dir = os.path.join(self.enr_dir, 'enrichment_results')
        os.makedirs(self.res_dir)
        self.out = os.path.join(self.tmp.name, 'summary.html')

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, df):
        df.to_csv(os.path.join(self.res_dir, name), index=False)

    def _run(self, extra=None):
        argv = ['summarizeEnrichment.py', '--enrichment-dir', self.enr_dir, '--out', self.out] + (extra or [])
        with patch.object(sys, 'argv', argv):
            se.main()
        with open(self.out, encoding='utf-8') as fh:
            return fh.read()

    def test_top_n_cap_and_sort_order(self):
        self._write('PROMOTER_fdr_GO_Biological_Process_2021_enrichment.csv', _terms(40))
        html = self._run(['--top-n', '25'])
        # 40 significant terms reported; table shows 25
        self.assertIn('40 significant term(s); showing top 25', html)
        # T0 has the smallest adjusted p and must appear; T39 must not (rank 40)
        self.assertIn('>T0<', html)
        self.assertNotIn('>T39<', html)
        # Order: T0 row precedes T24 row in the table
        self.assertLess(html.index('>T0<'), html.index('>T24<'))

    def test_default_top_n_is_25(self):
        self._write('PROMOTER_fdr_GO_Biological_Process_2021_enrichment.csv', _terms(40))
        html = self._run()
        self.assertIn('showing top 25', html)

    def test_one_section_and_one_figure_per_analysis_plus_overview(self):
        self._write('PROMOTER_fdr_KEGG_2021_Human_enrichment.csv', _terms(5, 'A'))
        self._write('TRANS_ig_KEGG_2021_Human_enrichment.csv', _terms(3, 'B'))
        html = self._run()
        self.assertEqual(html.count("<div class='analysis'"), 2)
        # 2 per-analysis figures + 1 overview figure, all embedded
        self.assertEqual(html.count('data:image/png;base64,'), 3)
        self.assertIn('PROMOTER | fdr | KEGG_2021_Human', html)
        self.assertIn('TRANS | ig | KEGG_2021_Human', html)
        self.assertNotIn(se.NO_RESULTS_BANNER, html)

    def test_genes_column_truncated(self):
        self._write('CIS5_fdr_KEGG_2021_Human_enrichment.csv', _terms(2))
        html = self._run()
        self.assertIn('... (+5)', html)

    def test_empty_directory_writes_report_with_banner_and_exits_zero(self):
        html = self._run()
        self.assertIn(se.NO_RESULTS_BANNER, html)
        self.assertEqual(html.count("<div class='analysis'"), 0)

    def test_encode_results_included_when_present(self):
        enc = pd.DataFrame({'Annotation Track': ['ChromHMM'] * 3,
                            'State/Region': ['Global: 1', 'Global: 2', 'PROMOTER: 1'],
                            'Overlap Count (A)': [10, 20, 5],
                            'Fold Enrichment': [1.5, 2.0, 0.5],
                            'P-value': [0.5, 0.001, 0.2],
                            'Adj P-value': [0.5, 0.003, 0.3]})
        enc.to_csv(os.path.join(self.enr_dir, 'encode_enrichment_results.csv'), index=False)
        html = self._run()
        self.assertIn('ENCODE ChromHMM enrichment', html)
        self.assertNotIn(se.NO_RESULTS_BANNER, html)
        # sorted by Adj P-value ascending: 'Global: 2' first
        self.assertLess(html.index('Global: 2'), html.index('Global: 1'))

    def test_missing_directory_exits_nonzero(self):
        argv = ['summarizeEnrichment.py', '--enrichment-dir', os.path.join(self.tmp.name, 'nope'), '--out', self.out]
        with patch.object(sys, 'argv', argv):
            with self.assertRaises(SystemExit) as cm:
                se.main()
        self.assertEqual(cm.exception.code, 1)


if __name__ == '__main__':
    unittest.main()
