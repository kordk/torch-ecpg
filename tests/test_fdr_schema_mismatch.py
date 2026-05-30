import os
import shutil
import subprocess
import sys
import unittest
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Adjust path to import the tool if needed, but we will run it as a subprocess
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL_PATH = os.path.join(REPO_ROOT, 'tools', 'summarizeOutput_parquet.py')

class TestFdrSchemaMismatch(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_fdr_schema_mismatch'
        self.input_file = os.path.join(self.test_dir, 'input.parquet')
        self.reservoir_file = os.path.join(self.test_dir, 'reservoir.csv')
        self.output_file = os.path.join(self.test_dir, 'output.parquet')

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)

        # Create dummy data with schema mismatch scenario
        self.create_dummy_data()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_dummy_data(self):
        # We need an input parquet file that contains multiple chunks.
        # Chunk 1: all-present coordinates. Since we process it with Pandas in the summarize script,
        # let's write it via Pandas.
        # Wait, the issue happens because `assignRegionToEcpg_parquet.py` creates columns with pd.Int64Dtype()
        # nulls in it. Or we can just simulate the input that has a unified Parquet schema
        # (int64 for coordinates, with nulls) but when Pandas reads it chunk by chunk, Pandas infers
        # chunk 1 (no nulls) as int64 and chunk 2 (with nulls) as float64 if not using arrow types.
        # Actually, if pyarrow reads it, df_chunk = batch.to_pandas() will convert an int64 Arrow column with nulls
        # into a float64 Pandas column, while an int64 Arrow column without nulls will be converted to int64 Pandas column!

        schema = pa.schema([
            ('id', pa.int64()),
            ('mt_chromStart', pa.int64()), # Allow nulls in parquet
            ('gt_chromStart', pa.int64()),
            ('precise_mt_p', pa.float64())
        ])

        # Chunk 1: no nulls
        df1 = pd.DataFrame({
            'id': [1, 2],
            'mt_chromStart': [100, 200],
            'gt_chromStart': [100, 200],
            'precise_mt_p': [0.01, 0.05]
        })
        table1 = pa.Table.from_pandas(df1, schema=schema)

        # Chunk 2: with nulls
        df2 = pd.DataFrame({
            'id': [3, 4],
            'mt_chromStart': [300, None],
            'gt_chromStart': [None, 400],
            'precise_mt_p': [0.1, 0.2]
        })
        # Arrow schema allows nulls for int64. Let's make sure it writes.
        table2 = pa.Table.from_pandas(df2, schema=schema)

        with pq.ParquetWriter(self.input_file, schema) as writer:
            writer.write_table(table1)
            writer.write_table(table2)

        # create a dummy reservoir file
        df_res = pd.DataFrame({
            'id': [1,2,3,4],
            'precise_mt_p': [0.01, 0.05, 0.1, 0.2]
        })
        df_res.to_csv(self.reservoir_file, index=False)

    def test_schema_mismatch(self):
        cmd = [sys.executable, TOOL_PATH,
               '--main-file', self.input_file,
               '--reservoir-file', self.reservoir_file,
               '--total-tests', '4',
               '--df', '100',
               '--calculate-fdr',
               '--output-fdr-file', self.output_file,
               '--chunk-size', '2'] # Force chunks to match batches we wrote
        result = subprocess.run(cmd, capture_output=True, text=True)

        # The tool should not fail
        self.assertEqual(result.returncode, 0, f"Tool failed with stdout: {result.stdout}\nstderr: {result.stderr}")

        # The output file should exist
        self.assertTrue(os.path.exists(self.output_file))

        # Output should be valid parquet and contain 4 rows
        table = pq.read_table(self.output_file)
        self.assertEqual(table.num_rows, 4)

        # Output schema for mt_chromStart should be int64 or float64 consistently
        # It shouldn't crash during writing.

if __name__ == '__main__':
    unittest.main()
