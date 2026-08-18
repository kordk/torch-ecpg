from tools.annotation_io import readAnnotationFileToDict


def test_gff_captures_gene_model_and_gene_name(tmp_path):
    gtf = tmp_path / "mini.gtf"
    gtf.write_text(
        "#comment\n"
        'chr1\tHAVANA\tgene\t101\t500\t.\t+\t.\tgene_id "ENSG00000000001.3"; gene_name "AAA";\n'
        'chr1\tHAVANA\ttranscript\t101\t200\t.\t+\t.\tgene_id "ENSG00000000001.3"; gene_name "AAA";\n'
        'chr2\tHAVANA\tgene\t1001\t2000\t.\t-\t.\tgene_id "ENSG00000000002.1"; gene_name "BBB";\n'
    )
    loci = readAnnotationFileToDict(str(gtf))

    # Only 'gene' features are kept, keyed by versionless Ensembl id.
    assert set(loci) == {"ENSG00000000001", "ENSG00000000002"}

    a = loci["ENSG00000000001"]
    # GFF 1-based inclusive -> 0-based half-open: start-1, end as-is.
    assert (a["chrom"], a["chromStart"], a["chromEnd"], a["strand"]) == ("chr1", 100, 500, "+")
    # gene_name (symbol) captured additively.
    assert a["gene_name"] == "AAA"
    assert loci["ENSG00000000002"]["gene_name"] == "BBB"


def test_bed6_coordinates_verbatim(tmp_path):
    bed = tmp_path / "mini.bed6"
    bed.write_text(
        "chrom\tchromStart\tchromEnd\tname\tscore\tstrand\n"
        "chr3\t128604584\t128604633\tILMN_1792672\t0\t-\n"
        "NA\t.\t.\tILMN_dropme\t0\t+\n"
    )
    loci = readAnnotationFileToDict(str(bed))

    # BED6 stores columns 2/3 verbatim; the NA-coordinate row is skipped (no entry).
    assert set(loci) == {"ILMN_1792672"}
    p = loci["ILMN_1792672"]
    assert (p["chrom"], p["chromStart"], p["chromEnd"], p["strand"]) == ("chr3", 128604584, 128604633, "-")
