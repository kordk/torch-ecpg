import pytest
import subprocess
import os

# Local helper, isolated import
def _write_mock_catalog(path, data_dict):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table(data_dict)
    pq.write_table(table, path)

@pytest.fixture
def catalog_path(tmp_path):
    path = str(tmp_path / "mock_catalog.parquet")
    _write_mock_catalog(path, {
        "gt_id": ["g1", "g2", "g1", "g3"],  # 3 distinct
        "mt_id": ["m1", "m2", "m3", "m4"]   # 4 distinct
    })
    return path

@pytest.fixture
def custom_catalog_path(tmp_path):
    path = str(tmp_path / "custom_catalog.parquet")
    _write_mock_catalog(path, {
        "custom_g": ["g1", "g2", "g3", "g4", "g5"],  # 5 distinct
        "custom_m": ["m1", "m1", "m2", "m2", "m3"]   # 3 distinct
    })
    return path

def _run_tool(*args):
    cmd = ["python3", "tools/check_catalog_grid.py"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def test_within_bounds_exits_zero(catalog_path):
    res = _run_tool("--catalog", catalog_path, "--max-genes", "10", "--max-loci", "10")
    assert res.returncode == 0
    assert "catalog grid: 4 rows, 3 distinct gt_id, 4 distinct mt_id" in res.stdout

def test_genes_exceed_bound_fails_closed(catalog_path):
    res = _run_tool("--catalog", catalog_path, "--max-genes", "2", "--max-loci", "10")
    assert res.returncode == 1
    assert "gt_id" in res.stderr
    assert "(3)" in res.stderr
    assert "(2)" in res.stderr

def test_loci_exceed_bound_fails_closed(catalog_path):
    res = _run_tool("--catalog", catalog_path, "--max-genes", "10", "--max-loci", "2")
    assert res.returncode == 1
    assert "mt_id" in res.stderr
    assert "(4)" in res.stderr
    assert "(2)" in res.stderr

def test_counts_equal_to_bound_pass(catalog_path):
    res = _run_tool("--catalog", catalog_path, "--max-genes", "3", "--max-loci", "4")
    assert res.returncode == 0

def test_omitted_bounds_skip_validation(catalog_path):
    res = _run_tool("--catalog", catalog_path)
    assert res.returncode == 0

def test_only_gene_bound_given_checks_only_genes(catalog_path):
    res = _run_tool("--catalog", catalog_path, "--max-genes", "5")
    assert res.returncode == 0

def test_missing_column_fails_closed(catalog_path):
    res = _run_tool("--catalog", catalog_path, "--gene-column", "missing_gene")
    assert res.returncode == 1
    assert "missing_gene" in res.stderr
    assert "gt_id" in res.stderr
    assert "mt_id" in res.stderr

def test_custom_column_names_are_honoured(custom_catalog_path):
    res = _run_tool("--catalog", custom_catalog_path, "--gene-column", "custom_g", "--locus-column", "custom_m", "--max-genes", "10", "--max-loci", "10")
    assert res.returncode == 0
    assert "5 distinct custom_g" in res.stdout
    assert "3 distinct custom_m" in res.stdout

def test_summary_line_reports_actual_counts(catalog_path):
    res = _run_tool("--catalog", catalog_path)
    assert res.returncode == 0
    assert "catalog grid: 4 rows, 3 distinct gt_id, 4 distinct mt_id" in res.stdout
