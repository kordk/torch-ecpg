"""D8c: tools/runEnrichment.py must fail closed.

Before this change the FDR and IG selection paths caught every Exception,
logged it, and continued; and an Enrichr call that exhausted its retries was
logged and skipped. In both cases the process exited 0 and the pipeline
(set -e) reported success over an empty or partial enrichment output. These
tests pin the replacement contract: a selection-path exception exits 1; an
exhausted Enrichr retry exits 1 after all other libraries have been attempted
and any successful CSVs have been written; and a transient failure that is
recovered by a retry still exits 0.
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


def _write_fdr_input(path):
    df = pd.DataFrame({
        "mt_id": ["cg1", "cg2"],
        "gt_id": ["ENSG1", "ENSG2"],
        "region": ["CIS", "CIS"],
        "precise_mt_p": [1e-9, 1e-9],
        "fdr_est": [0.01, 0.02],
        "mt_chrom": ["chr1", "chr1"],
        "mt_chromStart": [100, 200],
    })
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    return path


def _write_ig_input(path):
    df = pd.DataFrame({
        "mt_id": ["cg3", "cg4"],
        "gt_id": ["ENSG3", "ENSG4"],
        "region": ["CIS", "CIS"],
        "mt_ig": [10.0, 1.0],
        "cov1_ig": [1.0, 10.0],
    })
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    return path


def _base_patches(monkeypatch, libraries):
    monkeypatch.setattr(runEnrichment.gseapy, "get_library_name", lambda: list(libraries))
    monkeypatch.setattr(runEnrichment, "clean_and_translate_gene_ids",
                        lambda genes, *a, **k: (list(genes), 0))
    monkeypatch.setattr(runEnrichment.time, "sleep", lambda *a, **k: None)


class _Enr:
    def __init__(self):
        self.results = pd.DataFrame({
            "Term": ["Pathway1"], "Overlap": ["1/10"], "P-value": [0.01],
            "Adjusted P-value": [0.01], "Genes": ["ENSG1"],
        })


def test_fdr_path_exception_exits_nonzero(tmp_path, monkeypatch, caplog):
    fdr_input = _write_fdr_input(tmp_path / "fdr.parquet")
    _base_patches(monkeypatch, ["L1"])
    monkeypatch.setattr(sys, "argv", ["runEnrichment.py", "--fdr-input", str(fdr_input),
                                      "--out-dir", str(tmp_path), "--rank-by", "fdr",
                                      "--enrichment-libraries", "L1"])

    def boom(*a, **k):
        raise RuntimeError("simulated parquet read failure")
    monkeypatch.setattr(runEnrichment.pq, "ParquetFile", boom)

    with caplog.at_level(logging.ERROR, logger="runEnrichment"):
        with pytest.raises(SystemExit) as exc:
            runEnrichment.main()
    assert exc.value.code == 1
    assert "Error processing FDR path" in caplog.text


def test_ig_path_exception_exits_nonzero(tmp_path, monkeypatch, caplog):
    ig_input = _write_ig_input(tmp_path / "ig.parquet")
    _base_patches(monkeypatch, ["L1"])
    monkeypatch.setattr(sys, "argv", ["runEnrichment.py", "--ig-input", str(ig_input),
                                      "--out-dir", str(tmp_path), "--rank-by", "ig",
                                      "--enrichment-libraries", "L1"])

    def boom(*a, **k):
        raise RuntimeError("simulated inflection failure")
    monkeypatch.setattr(runEnrichment, "detect_inflection", boom)

    with caplog.at_level(logging.ERROR, logger="runEnrichment"):
        with pytest.raises(SystemExit) as exc:
            runEnrichment.main()
    assert exc.value.code == 1
    assert "Error processing IG path" in caplog.text


def test_exhausted_enrichr_retries_exit_nonzero_after_writing_successes(tmp_path, monkeypatch, caplog):
    fdr_input = _write_fdr_input(tmp_path / "fdr.parquet")
    _base_patches(monkeypatch, ["Lgood", "Lbad"])
    monkeypatch.setattr(sys, "argv", ["runEnrichment.py", "--fdr-input", str(fdr_input),
                                      "--out-dir", str(tmp_path), "--rank-by", "fdr",
                                      "--enrichment-libraries", "Lgood", "Lbad"])

    def enrichr(gene_list=None, gene_sets=None, **k):
        if gene_sets == "Lbad":
            raise RuntimeError("simulated persistent 504")
        return _Enr()
    monkeypatch.setattr(runEnrichment.gseapy, "enrichr", enrichr)

    with caplog.at_level(logging.ERROR, logger="runEnrichment"):
        with pytest.raises(SystemExit) as exc:
            runEnrichment.main()
    assert exc.value.code == 1
    # The successful library's CSV was still written (additive; nothing lost).
    good_csv = tmp_path / "enrichment_results" / "CIS_fdr_Lgood_enrichment.csv"
    assert good_csv.exists()
    bad_csv = tmp_path / "enrichment_results" / "CIS_fdr_Lbad_enrichment.csv"
    assert not bad_csv.exists()
    assert "1 Enrichr call(s) failed after retries" in caplog.text


def test_transient_enrichr_failure_recovered_by_retry_exits_zero(tmp_path, monkeypatch):
    fdr_input = _write_fdr_input(tmp_path / "fdr.parquet")
    _base_patches(monkeypatch, ["L1"])
    monkeypatch.setattr(sys, "argv", ["runEnrichment.py", "--fdr-input", str(fdr_input),
                                      "--out-dir", str(tmp_path), "--rank-by", "fdr",
                                      "--enrichment-libraries", "L1"])
    calls = {"n": 0}

    def enrichr(gene_list=None, gene_sets=None, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated transient 504")
        return _Enr()
    monkeypatch.setattr(runEnrichment.gseapy, "enrichr", enrichr)

    runEnrichment.main()  # must return normally (exit 0)
    assert (tmp_path / "enrichment_results" / "CIS_fdr_L1_enrichment.csv").exists()


def test_run_enrichr_returns_failure_count(tmp_path, monkeypatch):
    _base_patches(monkeypatch, ["Lbad"])

    def enrichr(gene_list=None, gene_sets=None, **k):
        raise RuntimeError("simulated persistent 504")
    monkeypatch.setattr(runEnrichment.gseapy, "enrichr", enrichr)

    args = runEnrichment.argparse.Namespace(
        out_dir=str(tmp_path), enrichment_libraries=["Lbad"],
        enrichment_max_genes=3000, dry_run_enrichment=False)
    n_failed = runEnrichment.run_enrichr("fdr", args, {"CIS": {"ENSG1": 1e-9}})
    assert n_failed == 1
