"""Guards for significance selection in tools/runEnrichment.py.

The FDR path compared raw p-values against --fdr-threshold, whose name, help
text and default of 0.05 all describe an FDR. On a catalog already cut at
p <= 1e-3 that comparison is satisfied by every row, so the enrichment
foreground was the whole catalog while presenting as FDR-controlled. These
tests pin the replacement contract: selection is made on an FDR column, the
column is required, and how many rows it selected is reported.
"""
import logging
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import runEnrichment  # noqa: E402


def _write_fdr_input(path, fdr_values, p_values=None, include_fdr=True):
    n = len(fdr_values)
    if p_values is None:
        p_values = [1e-9] * n
    df = pd.DataFrame({
        "mt_id": [f"cg{i}" for i in range(n)],
        "gt_id": [f"ENSG{i}" for i in range(n)],
        "region": ["CIS"] * n,
        "precise_mt_p": p_values,
        "mt_chrom": ["chr1"] * n,
        "mt_chromStart": [100 * (i + 1) for i in range(n)],
    })
    if include_fdr:
        df["fdr_est"] = fdr_values
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    return path


def _argv(fdr_input, out_dir, extra=None):
    argv = ["runEnrichment.py", "--fdr-input", str(fdr_input),
            "--out-dir", str(out_dir), "--rank-by", "fdr",
            "--dry-run-enrichment", "--enrichment-libraries", "L1"]
    if extra:
        argv.extend(extra)
    return argv


def _run(monkeypatch, caplog, fdr_input, out_dir, extra=None):
    monkeypatch.setattr(sys, "argv", _argv(fdr_input, out_dir, extra))
    monkeypatch.setattr(runEnrichment.gseapy, "get_library_name", lambda: ["L1"])
    monkeypatch.setattr(runEnrichment, "clean_and_translate_gene_ids",
                        lambda genes, *a, **k: (list(genes), 0))
    monkeypatch.setattr(runEnrichment.time, "sleep", lambda *a, **k: None)
    with caplog.at_level(logging.INFO, logger="runEnrichment"):
        runEnrichment.main()
    return caplog.text


def test_selection_is_made_on_the_fdr_column_not_the_p_value(tmp_path, monkeypatch, caplog):
    """Every row here has p = 1e-9; only one has an FDR at or under the cut.

    Under the previous comparison against raw p-values, both rows were selected.
    """
    inp = _write_fdr_input(tmp_path / "summarized.parquet", [0.01, 0.90])
    text = _run(monkeypatch, caplog, inp, tmp_path / "out")
    assert "selected 1 of 2 rows" in text


def test_null_fdr_rows_are_not_selected(tmp_path, monkeypatch, caplog):
    """A row with no FDR estimate has not been assessed and is not significant."""
    inp = _write_fdr_input(tmp_path / "summarized.parquet", [0.01, float("nan"), 0.02])
    text = _run(monkeypatch, caplog, inp, tmp_path / "out")
    assert "selected 2 of 3 rows" in text


def test_threshold_is_applied_to_the_named_column(tmp_path, monkeypatch, caplog):
    """--fdr-column routes selection to an alternative FDR estimate."""
    inp = tmp_path / "summarized.parquet"
    df = pd.DataFrame({
        "mt_id": ["cg0", "cg1"], "gt_id": ["ENSG0", "ENSG1"],
        "region": ["CIS", "CIS"], "precise_mt_p": [1e-9, 1e-9],
        "mt_chrom": ["chr1", "chr1"], "mt_chromStart": [100, 200],
        "fdr_est": [0.90, 0.90], "fdr_permute": [0.01, 0.90],
    })
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), inp)
    text = _run(monkeypatch, caplog, inp, tmp_path / "out",
                extra=["--fdr-column", "fdr_permute"])
    assert "selected 1 of 2 rows" in text
    assert "'fdr_permute'" in text


def test_missing_fdr_column_fails_closed(tmp_path, monkeypatch, caplog):
    """Falling back to raw p-values is what produced an uncontrolled foreground."""
    inp = _write_fdr_input(tmp_path / "summarized.parquet", [0.01, 0.90],
                           include_fdr=False)
    monkeypatch.setattr(sys, "argv", _argv(inp, tmp_path / "out"))
    with caplog.at_level(logging.ERROR, logger="runEnrichment"):
        with pytest.raises(SystemExit) as exc:
            runEnrichment.main()
    assert exc.value.code == 1
    assert "fdr_est" in caplog.text
    assert "refusing to fall back to raw p-values" in caplog.text
