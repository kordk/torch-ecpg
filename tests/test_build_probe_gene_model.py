import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import build_probe_gene_model as B  # noqa: E402

GTF = """\
#!genome-build GRCh37
chr1\tHAVANA\tgene\t1000\t5000\t.\t+\t.\tgene_id "ENSG1.1"; gene_name "AAA"; gene_type "protein_coding";
chr1\tHAVANA\texon\t1000\t1100\t.\t+\t.\tgene_id "ENSG1.1"; gene_name "AAA";
chr1\tHAVANA\texon\t1050\t1200\t.\t+\t.\tgene_id "ENSG1.1"; gene_name "AAA";
chr1\tHAVANA\tgene\t900\t6000\t.\t-\t.\tgene_id "ENSG2.1"; gene_name "LOC123"; gene_type "lncRNA";
chr1\tHAVANA\texon\t1060\t1210\t.\t-\t.\tgene_id "ENSG2.1"; gene_name "LOC123";
chr1\tHAVANA\tgene\t8000\t9000\t.\t+\t.\tgene_id "ENSG3.1"; gene_name "BBB"; gene_type "protein_coding";
chr1\tHAVANA\texon\t8000\t8100\t.\t+\t.\tgene_id "ENSG3.1"; gene_name "BBB";
chr1\tHAVANA\tgene\t8000\t9500\t.\t+\t.\tgene_id "ENSG4.1"; gene_name "CCC"; gene_type "protein_coding";
chr1\tHAVANA\texon\t8000\t8100\t.\t+\t.\tgene_id "ENSG4.1"; gene_name "CCC";
chr1\tHAVANA\tgene\t20000\t21000\t.\t+\t.\tgene_id "ENSG5.1"; gene_name "DDD"; gene_type "protein_coding"; tag "readthrough_gene";
chr1\tHAVANA\texon\t20000\t20100\t.\t+\t.\tgene_id "ENSG5.1"; gene_name "DDD";
chr1\tHAVANA\tgene\t20000\t20500\t.\t+\t.\tgene_id "ENSG6.1"; gene_name "EEE"; gene_type "protein_coding";
chr1\tHAVANA\texon\t20000\t20100\t.\t+\t.\tgene_id "ENSG6.1"; gene_name "EEE";
chr1\tHAVANA\tgene\t30000\t31000\t.\t+\t.\tgene_id "ENSG7.1"; gene_name "LOC999"; gene_type "lncRNA";
chr1\tHAVANA\texon\t30000\t30100\t.\t+\t.\tgene_id "ENSG7.1"; gene_name "LOC999";
chr1\tHAVANA\tgene\t40000\t41000\t.\t+\t.\tgene_id "ENSG8.1"; gene_name "LOC888"; gene_type "lncRNA";
chr1\tHAVANA\texon\t40000\t40100\t.\t+\t.\tgene_id "ENSG8.1"; gene_name "LOC888";
chr1\tHAVANA\tgene\t40000\t42000\t.\t+\t.\tgene_id "ENSG9.1"; gene_name "C1orf77"; gene_type "protein_coding";
chr1\tHAVANA\texon\t40000\t40100\t.\t+\t.\tgene_id "ENSG9.1"; gene_name "C1orf77";
chr1\tHAVANA\tgene\t60000\t61000\t.\t+\t.\tgene_id "ENSG10.1"; gene_name "FFF"; gene_type "protein_coding";
chr1\tHAVANA\texon\t60000\t60100\t.\t+\t.\tgene_id "ENSG10.1"; gene_name "FFF";
chr1\tHAVANA\texon\t62000\t62200\t.\t+\t.\tgene_id "ENSG10.1"; gene_name "FFF"; tag "readthrough_transcript";
chr1\tHAVANA\tgene\t70000\t71000\t.\t+\t.\tgene_id "ENSG11.1"; gene_name "GGG"; gene_type "protein_coding";
chr1\tHAVANA\texon\t70000\t70100\t.\t+\t.\tgene_id "ENSG11.1"; gene_name "GGG";
chr1\tHAVANA\tgene\t70000\t75000\t.\t+\t.\tgene_id "ENSG12.1"; gene_name "HHH"; gene_type "protein_coding";
chr1\tHAVANA\texon\t70000\t70100\t.\t+\t.\tgene_id "ENSG12.1"; gene_name "HHH"; tag "readthrough_transcript";
chr1\tHAVANA\tgene\t80000\t81000\t.\t+\t.\tgene_id "ENSG13.1"; gene_name "III"; gene_type "protein_coding";
chr1\tHAVANA\ttranscript\t80000\t81000\t.\t+\t.\tgene_id "ENSG13.1"; transcript_id "ENST13.1"; gene_name "III"; tag "readthrough_transcript";
chr1\tHAVANA\texon\t80000\t80100\t.\t+\t.\tgene_id "ENSG13.1"; gene_name "III";
chr1\tHAVANA\tgene\t80000\t82000\t.\t+\t.\tgene_id "ENSG14.1"; gene_name "JJJ"; gene_type "protein_coding";
chr1\tHAVANA\ttranscript\t80000\t82000\t.\t+\t.\tgene_id "ENSG14.1"; transcript_id "ENST14.1"; gene_name "JJJ";
chr1\tHAVANA\texon\t80000\t80100\t.\t+\t.\tgene_id "ENSG14.1"; gene_name "JJJ";
"""

BED = """\
1\t1060\t1109\tILMN_MERGED\t0\t+
1\t1150\t1170\tILMN_SMALLOV\t0\t+
1\t8020\t8069\tILMN_TWOGENE\t0\t+
1\t20020\t20069\tILMN_READTHRU\t0\t+
1\t30020\t30069\tILMN_LOCONLY\t0\t+
1\t40020\t40069\tILMN_ALLBAD\t0\t+
1\t62050\t62099\tILMN_RTEXON\t0\t+
1\t60020\t60069\tILMN_RTGENEKEPT\t0\t+
1\t70020\t70069\tILMN_RTEXONDUP\t0\t+
1\t80020\t80069\tILMN_RTTRANSCRIPT\t0\t+
1\t50000\t50049\tILMN_NOOVERLAP\t0\t+
NA\tNA\tNA\tILMN_NOCOORD\t0\t+
"""

@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    d = tmp_path_factory.mktemp("pgm")
    gtf = d / "t.gtf"
    gtf.write_text(GTF)
    bed = d / "G.bed6"
    bed.write_text(BED)
    out = d / "probe_gene_model.tsv"

    B.derive(str(gtf), str(bed), str(out))

    rows_dict = {}
    with open(out) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    r = csv.DictReader(lines, delimiter="\t")
    for row in r:
        rows_dict[row["probe_id"]] = row

    return rows_dict


def test_merged_exons_and_chr_prefix_normalization(rows):
    r = rows["ILMN_MERGED"]
    assert r["status"] == "RESOLVED" and r["gtf_gene_symbols"] == "AAA" \
        and r["start"] == "1000" and r["end"] == "5000" and r["strand"] == "+", \
        f'-> {r["gtf_gene_symbols"]} {r["start"]}-{r["end"]}{r["strand"]}'
    assert r["chrom"] == "1", f'-> {r["chrom"]}'


def test_overlap_of_21bp_rejected(rows):
    r = rows["ILMN_SMALLOV"]
    assert r["status"] == "NO_OVERLAP", f'-> {r["status"]}'


def test_two_clean_genes_retained_span_union(rows):
    r = rows["ILMN_TWOGENE"]
    assert r["status"] == "RESOLVED" and r["n_genes"] == "2" \
        and sorted(r["gtf_gene_symbols"].split(",")) == ["BBB", "CCC"] \
        and r["start"] == "8000" and r["end"] == "9500", \
        f'-> n={r["n_genes"]} {r["start"]}-{r["end"]}'


def test_readthrough_gene_dropped_from_duplicates(rows):
    r = rows["ILMN_READTHRU"]
    assert r["gtf_gene_symbols"] == "EEE" and r["end"] == "20500", \
        f'-> {r["gtf_gene_symbols"]} end={r["end"]}'


def test_sole_poorly_characterised_gene_is_kept(rows):
    r = rows["ILMN_LOCONLY"]
    assert r["status"] == "RESOLVED" and r["gtf_gene_symbols"] == "LOC999", \
        f'-> {r["status"]} {r["gtf_gene_symbols"]}'


def test_all_candidates_poorly_characterised_fallback_keeps_all(rows):
    r = rows["ILMN_ALLBAD"]
    assert r["status"] == "RESOLVED" and r["n_genes"] == "2" \
        and r["start"] == "40000" and r["end"] == "42000", \
        f'-> n={r["n_genes"]} {r["start"]}-{r["end"]}'


def test_read_through_transcript_exon_excluded_from_overlap(rows):
    r = rows["ILMN_RTEXON"]
    assert r["status"] == "NO_OVERLAP", f'-> {r["status"]}'


def test_gene_with_read_through_isoform_is_still_kept(rows):
    r = rows["ILMN_RTGENEKEPT"]
    assert r["status"] == "RESOLVED" and r["gtf_gene_symbols"] == "FFF", \
        f'-> {r["status"]} {r["gtf_gene_symbols"]}'


def test_read_through_exon_does_not_add_duplicate_gene(rows):
    r = rows["ILMN_RTEXONDUP"]
    assert r["gtf_gene_symbols"] == "GGG" and r["n_genes"] == "1" \
        and r["end"] == "71000", \
        f'-> {r["gtf_gene_symbols"]} n={r["n_genes"]} end={r["end"]}'


def test_readthrough_transcript_does_not_disqualify_its_gene(rows):
    r = rows["ILMN_RTTRANSCRIPT"]
    assert r["status"] == "RESOLVED" and r["n_genes"] == "2" \
        and sorted(r["gtf_gene_symbols"].split(",")) == ["III", "JJJ"] \
        and r["start"] == "80000" and r["end"] == "82000", \
        f'-> {r["gtf_gene_symbols"]} n={r["n_genes"]} {r["start"]}-{r["end"]}'


def test_no_overlap_yields_no_overlap_status(rows):
    r = rows["ILMN_NOOVERLAP"]
    assert r["status"] == "NO_OVERLAP", f'-> {r["status"]}'


def test_na_coordinate_probe_omitted(rows):
    assert "ILMN_NOCOORD" not in rows
