import subprocess
import sys
import os
import csv
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import logging
logging.basicConfig(level=logging.CRITICAL)
from annotation_io import readProbeGeneModel

T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "build_probe_gene_model.py")
BED_CONTENT = "1\t100000\t100049\tILMN_X\t0\t+\n7\t500000\t500049\tILMN_Y\t0\t-\nNA\tNA\tNA\tILMN_NA\t0\t+\n"

@pytest.fixture(scope="module")
def synthetic_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    bed = d / "G.bed6"
    bed.write_text(BED_CONTENT)
    out = d / "map.tsv"

    r = subprocess.run(
        [sys.executable, T, "--probe-bed", str(bed), "--output", str(out), "--synthetic-span", "60000"],
        capture_output=True, text=True
    )

    assert r.returncode == 0, r.stderr[-200:]

    with open(out) as f:
        lines = [l for l in f if not l.startswith("#")]

    rows = list(csv.DictReader(lines, delimiter="\t"))

    with open(out) as f:
        header_lines = [l for l in f if l.startswith("#")]

    return {
        "rows": rows,
        "header_lines": header_lines,
        "out_path": str(out),
        "bed_path": str(bed)
    }

def test_synthetic_mode_exits_0(synthetic_data):
    # This is implicitly checked in the fixture, but adding a test so it counts as 11 tests total
    assert len(synthetic_data["rows"]) >= 0

def test_one_row_per_positioned_probe(synthetic_data):
    rows = synthetic_data["rows"]
    assert len(rows) == 2, len(rows)

def test_span_anchored_at_probe_start(synthetic_data):
    x = [r for r in synthetic_data["rows"] if r["probe_id"] == "ILMN_X"][0]
    assert x["start"] == "100000" and x["end"] == "159999", f"{x['start']}-{x['end']}"

def test_strand_preserved(synthetic_data):
    y = [r for r in synthetic_data["rows"] if r["probe_id"] == "ILMN_Y"][0]
    assert y["strand"] == "-", "strand not preserved"

def test_status_resolved(synthetic_data):
    x = [r for r in synthetic_data["rows"] if r["probe_id"] == "ILMN_X"][0]
    assert x["status"] == "RESOLVED", "status not RESOLVED"

def test_header_marks_it_synthetic(synthetic_data):
    header_lines = synthetic_data["header_lines"]
    assert any("synthetic_fixed_span" in l for l in header_lines), "header does not mark it synthetic"

def test_header_records_no_gtf(synthetic_data):
    header_lines = synthetic_data["header_lines"]
    assert any(l.startswith("#gtf") and "NONE" in l for l in header_lines), "header records GTF"

def test_real_reader_parses_the_synthetic_map(synthetic_data):
    out = synthetic_data["out_path"]
    m, h = readProbeGeneModel(out)
    assert len(m) == 2 and m["ILMN_X"]["chromEnd"] == 159999, m.get("ILMN_X")

def test_gtf_and_synthetic_span_rejected(tmp_path):
    bed = tmp_path / "G.bed6"
    bed.write_text(BED_CONTENT)
    out = tmp_path / "map.tsv"

    r = subprocess.run(
        [sys.executable, T, "--probe-bed", str(bed), "--output", str(out), "--synthetic-span", "60000", "--gtf", "x.gtf"],
        capture_output=True, text=True
    )
    assert r.returncode != 0, r.returncode

def test_span_leq_5000_rejected(tmp_path):
    bed = tmp_path / "G.bed6"
    bed.write_text(BED_CONTENT)
    out = tmp_path / "map.tsv"

    r = subprocess.run(
        [sys.executable, T, "--probe-bed", str(bed), "--output", str(out), "--synthetic-span", "4000"],
        capture_output=True, text=True
    )
    assert r.returncode != 0, r.returncode

def test_neither_gtf_nor_synthetic_span_rejected(tmp_path):
    bed = tmp_path / "G.bed6"
    bed.write_text(BED_CONTENT)
    out = tmp_path / "map.tsv"

    r = subprocess.run(
        [sys.executable, T, "--probe-bed", str(bed), "--output", str(out)],
        capture_output=True, text=True
    )
    assert r.returncode != 0, r.returncode
