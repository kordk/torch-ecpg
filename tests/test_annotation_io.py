from tools.annotation_io import readAnnotationFileToDict, build_symbol_to_model, build_probe_to_model


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

def test_symbol_resolver_drops_ambiguous_keeps_clean():
    loci = {
        # AAA: single model -> kept
        "ENSG1": {"chrom": "chr1", "chromStart": 100, "chromEnd": 500, "strand": "+", "gene_name": "AAA"},
        # BBB: two DISTINCT models -> dropped (ambiguous)
        "ENSG2": {"chrom": "chr2", "chromStart": 10, "chromEnd": 20, "strand": "+", "gene_name": "BBB"},
        "ENSG3": {"chrom": "chrY", "chromStart": 10, "chromEnd": 20, "strand": "+", "gene_name": "BBB"},
        # CCC: two entries, IDENTICAL coordinates -> collapse to one model, kept
        "ENSG4": {"chrom": "chr4", "chromStart": 7, "chromEnd": 9, "strand": "-", "gene_name": "CCC"},
        "ENSG5": {"chrom": "chr4", "chromStart": 7, "chromEnd": 9, "strand": "-", "gene_name": "CCC"},
        # entry with no gene_name (e.g. BED6-parsed) -> skipped
        "ILMN_x": {"chrom": "chr9", "chromStart": 1, "chromEnd": 2, "strand": "+"},
    }
    resolved = build_symbol_to_model(loci)

    assert set(resolved) == {"AAA", "CCC"}          # BBB dropped, no-symbol skipped
    assert resolved["AAA"] == {"chrom": "chr1", "chromStart": 100, "chromEnd": 500, "strand": "+"}
    assert resolved["CCC"] == {"chrom": "chr4", "chromStart": 7, "chromEnd": 9, "strand": "-"}


def test_symbol_resolver_end_to_end_from_gtf(tmp_path):
    gtf = tmp_path / "mini.gtf"
    gtf.write_text(
        'chr1\tHAVANA\tgene\t101\t500\t.\t+\t.\tgene_id "ENSG00000000001.3"; gene_name "AAA";\n'
        'chr2\tHAVANA\tgene\t1001\t2000\t.\t-\t.\tgene_id "ENSG00000000002.1"; gene_name "BBB";\n'
        'chrY\tHAVANA\tgene\t3001\t4000\t.\t+\t.\tgene_id "ENSG00000000003.2"; gene_name "BBB";\n'
    )
    resolved = build_symbol_to_model(readAnnotationFileToDict(str(gtf)))
    # AAA unique -> kept; BBB maps to two distinct gene features -> dropped.
    assert set(resolved) == {"AAA"}
    assert resolved["AAA"]["chromStart"] == 100  # GFF 1-based -> 0-based

def test_probe_resolver_drops_and_keeps():
    symbol_to_model = {
        "AAA": {"chrom": "chr1", "chromStart": 100, "chromEnd": 500, "strand": "+"},
        "CCC": {"chrom": "chr4", "chromStart": 7, "chromEnd": 9, "strand": "-"},
    }
    pairs = [
        ("ILMN_1", "AAA"),      # single, matched -> kept
        ("ILMN_2", "BBB"),      # single, unmatched (dropped by K2) -> dropped
        ("ILMN_3", "AAA,BBB"),  # multi-symbol -> dropped (must NOT take-first)
        ("ILMN_4", ""),         # no symbol -> dropped
        ("ILMN_5", "ZZZ"),      # single, not in GTF -> dropped
        ("ILMN_6", "CCC"),      # single, matched -> kept
        ("ILMN_7", "NA"),       # NA token -> dropped
    ]
    resolved = build_probe_to_model(pairs, symbol_to_model)

    assert set(resolved) == {"ILMN_1", "ILMN_6"}
    # A matched probe carries the kb-scale gene model, not a probe footprint.
    assert resolved["ILMN_1"] == {"chrom": "chr1", "chromStart": 100, "chromEnd": 500, "strand": "+"}
    assert resolved["ILMN_6"]["chromEnd"] == 9


def test_parser_reads_gzip_transparently(tmp_path):
    import gzip as _gz
    content = (
        'chr1\tHAVANA\tgene\t101\t500\t.\t+\t.\tgene_id "ENSG00000000001.3"; gene_name "AAA";\n'
        'chr2\tHAVANA\tgene\t1001\t2000\t.\t-\t.\tgene_id "ENSG00000000002.1"; gene_name "BBB";\n'
    )
    plain = tmp_path / "mini.gtf"
    plain.write_text(content)
    gz = tmp_path / "mini.gtf.gz"
    with _gz.open(str(gz), "wt") as fh:
        fh.write(content)

    from_plain = readAnnotationFileToDict(str(plain))
    from_gz = readAnnotationFileToDict(str(gz))

    # Gzip path yields the identical parse as the plain path.
    assert from_gz == from_plain
    assert set(from_gz) == {"ENSG00000000001", "ENSG00000000002"}
    assert from_gz["ENSG00000000001"]["gene_name"] == "AAA"
