import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import build_gene_model_bed as B  # noqa: E402


def _read_bed(path):
    rows = {}
    with open(path) as fh:
        header = fh.readline().strip().split("\t")
        assert header == ["chrom", "chromStart", "chromEnd", "name", "score", "strand"]
        for line in fh:
            c, s, e, n, sc, st = line.rstrip("\n").split("\t")
            rows[n] = (c, int(s), int(e), st)
    return rows


def test_end_to_end_derives_1based_genemodel_bed(tmp_path):
    gtf = tmp_path / "mini.gtf"
    gtf.write_text(
        'chr1\tHAVANA\tgene\t101\t5000\t.\t+\t.\tgene_id "ENSG1.3"; gene_name "AAA";\n'
        'chr2\tHAVANA\tgene\t10\t20\t.\t-\t.\tgene_id "ENSG2.1"; gene_name "BBB";\n'
        'chrY\tHAVANA\tgene\t99\t200\t.\t+\t.\tgene_id "ENSG3.2"; gene_name "BBB";\n'
    )
    reann = tmp_path / "reann.txt"
    reann.write_text(
        "X.PROBE_ID\tGene_symbol\n"
        "ILMN_1\tAAA\n"       # matched -> kept
        "ILMN_2\tBBB\n"       # symbol maps to 2 distinct models -> dropped
        "ILMN_3\t\n"          # no symbol -> dropped
        "ILMN_4\tZZZ\n"       # unmatched -> dropped
    )
    out = tmp_path / "gm.bed6"

    n = B.main(["--gtf", str(gtf), "--reannotator", str(reann), "--output", str(out)])
    assert n == 1

    rows = _read_bed(str(out))
    assert set(rows) == {"ILMN_1"}
    # GFF 1-based start 101 -> parser 0-based 100 -> +1 back to 1-based 101; end verbatim.
    assert rows["ILMN_1"] == ("chr1", 101, 5000, "+")


def test_reads_probe_symbols_first_occurrence(tmp_path):
    reann = tmp_path / "r.txt"
    reann.write_text(
        "X.PROBE_ID\tGene_symbol\n"
        "ILMN_9\tAAA\n"
        "ILMN_9\tBBB\n"   # duplicate probe id -> first wins
    )
    pairs = B.read_probe_symbols(str(reann))
    assert pairs == [("ILMN_9", "AAA")]
