"""Guards for the edge p-value column in tools/exportBipartiteNetwork.py.

The edge table hardcoded mt_p, which is float32 and is computed by a
subtraction in which values below about 5.96e-08 (2**-24) are lost to
cancellation. It therefore reads as exactly zero across the range the
top-ranked edges occupy, since ranking is by IG or |t| descending. The exported
p-values for the most significant edges were therefore zero while precise_mt_p,
float64, sat unused in the same file. These tests pin the replacement:
precise_mt_p is preferred, mt_p remains a fallback, and using the fallback says
so.
"""
import logging
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import exportBipartiteNetwork  # noqa: E402

# The value precise_mt_p carries for a top edge, and the zero mt_p carries for
# the same edge once cancellation has taken it.
TINY_P = 1e-12
FLUSHED_P = 0.0


def _rows(include_precise=True, include_mt_p=True):
    data = {
        "mt_id": ["cpg1", "cpg2"],
        "mt_chrom": ["chr1", "chr2"],
        "mt_chromStart": [100, 200],
        "mt_strand": ["+", "-"],
        "gt_id": ["geneA", "geneB"],
        "gt_chrom": ["chr1", "chr2"],
        "gt_chromStart": [500, 600],
        "gt_strand": ["+", "-"],
        "mt_est": [0.5, -0.4],
        "mt_ig": [1.5, 1.2],
    }
    if include_mt_p:
        data["mt_p"] = np.array([FLUSHED_P, 0.05], dtype=np.float32)
    if include_precise:
        data["precise_mt_p"] = np.array([TINY_P, 0.05], dtype=np.float64)
    return data


class TestEdgePColumn(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_file = os.path.join(self.temp_dir.name, "input.parquet")
        self.out_prefix = os.path.join(self.temp_dir.name, "cyto")
        self.out_edges = f"{self.out_prefix}_edges.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write(self, data):
        pd.DataFrame(data).to_parquet(self.input_file)

    def _run(self):
        args = ["exportBipartiteNetwork.py", "-i", self.input_file,
                "-o", self.out_prefix]
        with patch("sys.argv", args):
            exportBipartiteNetwork.main()
        return pd.read_csv(self.out_edges)

    def test_fixture_encodes_the_loss_being_avoided(self):
        """The premise: for the top edge, mt_p is zero and precise_mt_p is not."""
        data = _rows()
        self.assertEqual(float(data["mt_p"][0]), 0.0)
        self.assertGreater(float(data["precise_mt_p"][0]), 0.0)

    def test_precise_column_is_used_when_present(self):
        self._write(_rows())
        edges = self._run()
        self.assertIn("precise_mt_p", edges.columns)
        self.assertNotIn("mt_p", edges.columns)
        self.assertGreater(edges["precise_mt_p"].min(), 0.0)

    def test_mt_p_is_used_when_the_precise_column_is_absent(self):
        self._write(_rows(include_precise=False))
        edges = self._run()
        self.assertIn("mt_p", edges.columns)
        self.assertNotIn("precise_mt_p", edges.columns)

    def test_falling_back_to_mt_p_is_announced(self):
        self._write(_rows(include_precise=False))
        with self.assertLogs(level=logging.WARNING) as cm:
            self._run()
        joined = " ".join(cm.output)
        self.assertIn("mt_p", joined)
        self.assertIn("cancellation", joined)

    def test_using_the_precise_column_does_not_warn_about_precision_loss(self):
        self._write(_rows())
        with self.assertLogs(level=logging.INFO) as cm:
            self._run()
        joined = " ".join(cm.output)
        self.assertIn("precise_mt_p", joined)
        self.assertNotIn("cancellation", joined)

    def test_neither_p_column_present_exits_one(self):
        self._write(_rows(include_precise=False, include_mt_p=False))
        args = ["exportBipartiteNetwork.py", "-i", self.input_file,
                "-o", self.out_prefix]
        with patch("sys.argv", args):
            with self.assertRaises(SystemExit) as cm:
                exportBipartiteNetwork.main()
        self.assertEqual(cm.exception.code, 1)
        self.assertFalse(os.path.exists(self.out_edges))

    def test_boot_suffixed_ig_columns_are_not_exported_as_ig(self):
        """mt_ig_boot ends with _boot, so the _ig sweep must not collect it."""
        data = _rows()
        data["mt_ig_boot"] = [9.9, 8.8]
        self._write(data)
        edges = self._run()
        self.assertIn("mt_ig", edges.columns)
        self.assertNotIn("mt_ig_boot", edges.columns)


if __name__ == "__main__":
    unittest.main()
