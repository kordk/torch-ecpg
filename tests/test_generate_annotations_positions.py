import ast
import os
import logging
import pandas as pd
import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "generate_annotations.py")

def load_names(*names):
    """Extract top-level defs/assigns by name from the (non-import-safe) script."""
    tree = ast.parse(open(SRC).read())
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef,)) and node.name in names:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id in names:
                keep.append(node)
    got = {n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id for n in keep}
    missing = set(names) - got
    assert not missing, f"not found in source: {missing}"
    mod = ast.Module(body=keep, type_ignores=[])
    import os as _os
    ns = {"pd": pd, "logging": logging, "os": _os}
    exec(compile(mod, SRC, "exec"), ns)  # nosec B102
    return ns

# Extract needed namespace members once at module level so tests can use them.
ns = load_names("parse_hg19_coords", "GEO_MAX_RANGE_GAP", "load_ucsc_wg6",
                "clean_geo_chromosome", "_CANONICAL_CHROMS", "UCSC_COLNAMES")
parse = ns["parse_hg19_coords"]
GEO_MAX_RANGE_GAP = ns["GEO_MAX_RANGE_GAP"]


def test_single_range():
    # single range (ILMN_1651209)
    assert parse("1667141-1667190") == (1667141, 1667190), "single range unchanged"


def test_junction_gap_96_merges():
    # real junction, gap 96 (ILMN_1343291) -> merged interval
    assert parse("74284362-74284378:74284474-74284506") == (74284362, 74284506), "junction (gap 96) merges"


def test_overlapping_unordered_ranges_merge():
    # overlapping ranges, unordered in the string (ILMN_1776515) -> merged after sort
    assert parse("11886993-11887024:11886983-11887000") == (11886983, 11887024), "overlapping unordered ranges merge"


def test_disjoint_gap_167mb_declined():
    # the 167 Mb monster (ILMN_1726152) -> declined
    detail = str(parse("784230-784255:167928066-167928089"))
    assert parse("784230-784255:167928066-167928089") == (None, None), f"disjoint (gap 167 Mb) declined   {detail}"


def test_disjoint_gap_90mb_declined():
    # the 90 Mb monster (ILMN_3236102) -> declined
    assert parse("44969405-44969427:135297017-135297043") == (None, None), "disjoint (gap 90 Mb) declined"


def test_boundary_gap_exactly_max_merges():
    # boundary: gap exactly GEO_MAX_RANGE_GAP merges
    assert parse(f"100-149:{149+GEO_MAX_RANGE_GAP}-{198+GEO_MAX_RANGE_GAP}") == (100, 198+GEO_MAX_RANGE_GAP), f"gap == {GEO_MAX_RANGE_GAP} merges"


def test_boundary_gap_one_more_than_max_declined():
    # boundary: one more declines
    assert parse(f"100-149:{150+GEO_MAX_RANGE_GAP}-{199+GEO_MAX_RANGE_GAP}") == (None, None), f"gap == {GEO_MAX_RANGE_GAP}+1 declined"


def test_blank_returns_none():
    # blanks unchanged
    assert parse("") == (None, None), "blank -> (None, None)"


def test_junk_returns_none():
    # junk unchanged
    assert parse("not-coords") == (None, None), "junk  -> (None, None)"


def test_ucsc_filtering(tmp_path):
    path = tmp_path / "ucsc.txt"
    rows = [
        # bin chrom chromStart chromEnd name score strand + 5 filler BED12 cols
        "585\tchr1\t1000\t1050\tILMN_SINGLE\t1000\t+\t1000\t1050\t1\t1\t50,\t0,",
        "585\tchr2\t2000\t2048\tILMN_DUPROW\t1000\t-\t2000\t2048\t1\t1\t48,\t0,",
        "585\tchr2\t2000\t2048\tILMN_DUPROW\t1000\t-\t2000\t2048\t1\t1\t48,\t0,",
        "585\tchr3\t3000\t3050\tILMN_MULTI\t1000\t+\t3000\t3050\t1\t1\t50,\t0,",
        "585\tchr4\t4000\t4050\tILMN_MULTI\t1000\t+\t4000\t4050\t1\t1\t50,\t0,",
    ]
    path.write_text("\n".join(rows) + "\n")
    lookup = ns["load_ucsc_wg6"](str(path))

    assert "ILMN_SINGLE" in lookup, "single alignment kept"

    detail_duprow = str(sorted(lookup))
    assert "ILMN_DUPROW" in lookup, f"exact duplicate rows collapse to one alignment -> kept   {detail_duprow}"

    assert "ILMN_MULTI" not in lookup, "two DISTINCT alignments -> dropped"

    detail_kept_record = str(lookup["ILMN_DUPROW"])
    assert lookup["ILMN_DUPROW"] == ("2", 2001, 2048, "-"), f"kept record is the (chrom, start1, end, strand) tuple with 1-based start   {detail_kept_record}"
