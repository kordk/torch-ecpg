import hashlib
import os
import subprocess
import sys

import pytest

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools",
                    "build_probe_gene_model.py")

GTF = (
    'chr1\tX\tgene\t1000\t5000\t.\t+\t.\tgene_id "G1.1"; gene_name "AAA"; '
    'gene_type "protein_coding";\n'
    'chr1\tX\texon\t1000\t1100\t.\t+\t.\tgene_id "G1.1"; gene_name "AAA";\n'
)
BED = "1\t1010\t1060\tILMN_A\t0\t+\n"


def _header(path):
    out = {}
    for line in open(path):
        if not line.startswith("#"):
            break
        parts = line[1:].rstrip("\n").split("\t")
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("prov")
    (d / "t.gtf").write_text(GTF)
    (d / "G.bed6").write_text(BED)
    return d


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_gtf_mode_header_carries_probe_bed_sha256(workdir):
    out = workdir / "gtf_map.tsv"
    r = subprocess.run([sys.executable, TOOL, "--gtf", str(workdir / "t.gtf"),
                        "--probe-bed", str(workdir / "G.bed6"),
                        "--output", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    hdr = _header(out)
    assert hdr.get("probe_bed_sha256") == _sha(workdir / "G.bed6"), hdr


def test_synthetic_mode_header_carries_probe_bed_sha256(workdir):
    out = workdir / "syn_map.tsv"
    r = subprocess.run([sys.executable, TOOL, "--synthetic-span", "60000",
                        "--probe-bed", str(workdir / "G.bed6"),
                        "--output", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    hdr = _header(out)
    assert hdr.get("probe_bed_sha256") == _sha(workdir / "G.bed6"), hdr


def test_sha_changes_when_probe_bed_changes(workdir):
    out = workdir / "map2.tsv"
    bed2 = workdir / "G2.bed6"
    bed2.write_text(BED + "1\t2000\t2049\tILMN_B\t0\t-\n")
    r = subprocess.run([sys.executable, TOOL, "--synthetic-span", "60000",
                        "--probe-bed", str(bed2), "--output", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    hdr = _header(out)
    assert hdr.get("probe_bed_sha256") == _sha(bed2)
    assert hdr.get("probe_bed_sha256") != _sha(workdir / "G.bed6")
