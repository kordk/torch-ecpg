import os
import shutil
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# The tool we are testing is executed directly or imported
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))
import runEnrichment

class TestRunEnrichment(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_runEnrichment_dir'
        self.fdr_input = os.path.join(self.test_dir, 'summarized.parquet')
        self.ig_input = os.path.join(self.test_dir, 'bootstrap_merged.parquet')
        self.out_dir = os.path.join(self.test_dir, 'output')

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.out_dir)

        # Create dummy FDR Parquet
        fdr_df = pd.DataFrame({
            'mt_id': ['cg1', 'cg2'],
            'gt_id': ['ENSG1', 'ENSG2'],
            'region': ['CIS', 'CIS'],
            'precise_mt_p': [0.01, 0.06],
            'fdr_est': [0.01, 0.06],
            'mt_chrom': ['chr1', 'chr1'],
            'mt_chromStart': [100, 200]
        })
        pq.write_table(pa.Table.from_pandas(fdr_df), self.fdr_input)

        # Create dummy IG Parquet (with mt_ig)
        ig_df = pd.DataFrame({
            'mt_id': ['cg3', 'cg4'],
            'gt_id': ['ENSG3', 'ENSG4'],
            'region': ['CIS', 'CIS'],
            'mt_ig': [10.0, 1.0],
            'cov1_ig': [1.0, 10.0]
        })
        pq.write_table(pa.Table.from_pandas(ig_df), self.ig_input)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('runEnrichment.gseapy.get_library_name')
    @patch('runEnrichment.gseapy.enrichr')
    @patch('runEnrichment.clean_and_translate_gene_ids')
    @patch('runEnrichment.time.sleep', return_value=None)
    def test_enrichment_logic(self, mock_sleep, mock_clean, mock_enrichr, mock_get_lib):
        mock_get_lib.return_value = ['ValidLibrary']
        mock_clean.return_value = (['GENE1', 'GENE2'], 0)

        # Mock enrichr to fail once, then succeed
        def side_effect(*args, **kwargs):
            if side_effect.calls == 0:
                side_effect.calls += 1
                raise Exception("Simulated 504 Timeout")
            mock_enr = MagicMock()
            mock_enr.results = pd.DataFrame({
                'Term': ['Pathway1'],
                'Overlap': ['1/10'],
                'P-value': [0.01],
                'Adjusted P-value': [0.01],
                'Genes': ['GENE1']
            })
            return mock_enr
        side_effect.calls = 0
        mock_enrichr.side_effect = side_effect

        args = runEnrichment.argparse.Namespace(
            fdr_input=self.fdr_input,
            ig_input=self.ig_input,
            out_dir=self.out_dir,
            rank_by=["fdr", "ig"],
            fdr_threshold=0.05,
            ig_inflection_method="auto",
            encode_enrichment=False,
            encode_bed_dir="",
            background_bed="",
            enrichment_max_genes=3000,
            enrichment_libraries=["ValidLibrary", "InvalidLibrary"],
            dry_run_enrichment=False,
            chunk_size=100000,
            df=100
        )

        # Call main function logic
        with patch('sys.argv', ['runEnrichment.py']):
            # It's easier to just call the methods directly or run main with patched args
            pass

        # Since main() uses argparse, let's patch sys.argv and call main
        test_args = [
            'runEnrichment.py',
            '--fdr-input', self.fdr_input,
            '--ig-input', self.ig_input,
            '--out-dir', self.out_dir,
            '--rank-by', 'fdr', 'ig',
            '--enrichment-libraries', 'ValidLibrary', 'InvalidLibrary'
        ]
        with patch('sys.argv', test_args):
            runEnrichment.main()

        # Assertions
        enrichment_dir = os.path.join(self.out_dir, "enrichment_results")
        self.assertTrue(os.path.exists(enrichment_dir))

        # Check if fdr and ig csvs were created (for ValidLibrary, because InvalidLibrary should be skipped)
        fdr_csv = os.path.join(enrichment_dir, 'CIS_fdr_ValidLibrary_enrichment.csv')
        ig_csv = os.path.join(enrichment_dir, 'CIS_ig_ValidLibrary_enrichment.csv')

        self.assertTrue(os.path.exists(fdr_csv))
        self.assertTrue(os.path.exists(ig_csv))

        # Check if comparison file was created
        comp_file = os.path.join(enrichment_dir, 'CIS_ValidLibrary_comparison.txt')
        self.assertTrue(os.path.exists(comp_file))

        with open(comp_file, 'r') as f:
            content = f.read()
            self.assertIn("Shared Terms", content)

    def test_missing_ig_column_warning(self):
        # Create a Parquet file without mt_ig
        bad_ig_input = os.path.join(self.test_dir, 'bad_ig.parquet')
        bad_df = pd.DataFrame({
            'mt_id': ['cg1'],
            'gt_id': ['ENSG1']
        })
        pq.write_table(pa.Table.from_pandas(bad_df), bad_ig_input)

        test_args = [
            'runEnrichment.py',
            '--ig-input', bad_ig_input,
            '--out-dir', self.out_dir,
            '--rank-by', 'ig'
        ]

        with patch('sys.argv', test_args):
            with self.assertLogs('runEnrichment', level='ERROR') as cm:
                runEnrichment.main()
                self.assertTrue(any("must contain 'mt_ig' column" in log for log in cm.output))

if __name__ == '__main__':
    unittest.main()
