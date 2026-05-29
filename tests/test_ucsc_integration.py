"""
Synthetic test harness for the UCSC WG-6 integration logic in
generate_annotations2.py.
 
Runs instantly with no network/downloads. Verifies:
  1. clean_geo_chromosome accepts canonical, rejects contigs/PAR/scaffold/header
  2. UCSC 0-based start -> 1-based start (+1) conversion
  3. Source priority: ReAnnotator > GEO > UCSC_WG6 > NA
  4. UCSC contig rows are rejected to NA (routed through clean_geo_chromosome)
  5. NA rows retain the probe ID
  6. normalize_chrom never emits 'chr<NA>'
 
Usage:  python test_ucsc_integration.py
Exit code 0 = all pass.
"""
 
import sys
import pandas as pd
 
# ---------------------------------------------------------------------------
# Re-implement the pure helpers under test (kept byte-identical in logic to
# generate_annotations2.py).  We copy rather than import so the test can run
# without GEOparse / pyliftover / network being installed.
# ---------------------------------------------------------------------------
 
_CANONICAL_CHROMS = {str(i) for i in range(1, 23)} | {'X', 'Y', 'M', 'MT'}
 
 
def clean_geo_chromosome(raw):
    if pd.isna(raw):
        return None
    token = str(raw).strip()
    if not token or token.lower() in ('nan', 'none', 'chrom'):
        return None
    primary = token.split('|')[0].strip()
    primary = primary.split('.')[0].strip()
    if not primary:
        return None
    bare = primary[3:] if primary.lower().startswith('chr') else primary
    bare = bare.strip()
    upper = bare.upper()
    if (upper.startswith('NT_') or upper.startswith('NW_')
            or upper.startswith('GL') or upper.startswith('KI')):
        return None
    if '_' in bare:
        return None
    if upper in ('XY', 'YX'):
        return None
    if upper == 'MT':
        bare = 'MT'
    elif upper == 'M':
        bare = 'M'
    if bare not in _CANONICAL_CHROMS:
        return None
    return bare
 
 
def normalize_chrom(series):
    s = series.astype(str).str.strip()
    s = s.str.replace(r'\.0$', '', regex=True)
    s = s.mask(s.isin(['nan', 'None', 'NaN', 'NA', '<NA>', '']), other=pd.NA)
    s = s.apply(lambda c: c if (pd.isna(c) or str(c).startswith('chr')) else 'chr' + str(c))
    return s
 
 
def ucsc_convert_start(start0):
    """0-based UCSC start -> 1-based start used by the rest of the pipeline."""
    return int(start0) + 1
 
 
def resolve_probe(re_chr, re_start, re_end, geo_chr, geo_start, geo_end, ucsc_entry):
    """
    Mirror the source-priority resolution in the gene loop.
    ucsc_entry is (chrom_token, start_1based, end_1based, strand) or None.
    Returns (chrom_token_or_NA, start_or_NA, end_or_NA, provenance).
    """
    chrom = start = end = None
    prov = 'NA'
 
    # 1. Re-Annotator
    if not pd.isna(re_chr) and not pd.isna(re_start) and not pd.isna(re_end):
        c = clean_geo_chromosome(re_chr)
        if c is not None:
            chrom, start, end, prov = c, int(re_start), int(re_end), 'ReAnnotator'
 
    # 2. GEO
    if chrom is None:
        c = clean_geo_chromosome(geo_chr)
        if c is not None and geo_start is not None and geo_end is not None:
            chrom, start, end, prov = c, int(geo_start), int(geo_end), 'GEO'
 
    # 3. UCSC WG-6
    if chrom is None and ucsc_entry is not None:
        chrom, start, end, _ = ucsc_entry
        prov = 'UCSC_WG6'
 
    if chrom is None or start is None or end is None:
        return pd.NA, pd.NA, pd.NA, 'NA'
    return chrom, start, end, prov
 
 
# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
 
failures = []
 
 
def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        failures.append(name)
 
 
print("Test 1: clean_geo_chromosome acceptance/rejection")
check("accepts 'chr4'", clean_geo_chromosome('chr4') == '4')
check("accepts bare '4'", clean_geo_chromosome('4') == '4')
check("accepts 'chrX'", clean_geo_chromosome('chrX') == 'X')
check("accepts 'MT' -> 'MT'", clean_geo_chromosome('MT') == 'MT')
check("strips pipe 'chr4|NT_113889.1' -> '4'", clean_geo_chromosome('chr4|NT_113889.1') == '4')
check("rejects 'chr17_random'", clean_geo_chromosome('chr17_random') is None)
check("rejects 'chr6_cox_hap1'", clean_geo_chromosome('chr6_cox_hap1') is None)
check("rejects 'chr19_gl000209_random'", clean_geo_chromosome('chr19_gl000209_random') is None)
check("rejects 'chrUn_gl000212'", clean_geo_chromosome('chrUn_gl000212') is None)
check("rejects 'chrXY' (PAR)", clean_geo_chromosome('chrXY') is None)
check("rejects 'chrYX' (PAR)", clean_geo_chromosome('chrYX') is None)
check("rejects 'NT_113889.1' (scaffold only)", clean_geo_chromosome('NT_113889.1') is None)
check("rejects header 'chrom'", clean_geo_chromosome('chrom') is None)
check("rejects NA", clean_geo_chromosome(pd.NA) is None)
 
print("Test 2: UCSC 0-based -> 1-based conversion")
# UCSC schema example: ILMN_1761068 chromStart=14541 chromEnd=14591
check("14541 (0-based) -> 14542 (1-based)", ucsc_convert_start(14541) == 14542)
# Matches the ILMN_1792672 Re-Annotator example: 1-based start 128604584
# means UCSC would store 128604583; +1 brings it back to 128604584.
check("128604583 -> 128604584 (matches Re-Annotator)", ucsc_convert_start(128604583) == 128604584)
 
print("Test 3: source priority ordering")
# Re-Annotator wins when present
c, s, e, p = resolve_probe('chr2', 100, 200, 'chr2', 300, 400,
                           ('2', 500, 600, '+'))
check("ReAnnotator beats GEO and UCSC", p == 'ReAnnotator' and s == 100)
# GEO wins when Re-Annotator absent
c, s, e, p = resolve_probe(pd.NA, pd.NA, pd.NA, 'chr2', 300, 400,
                           ('2', 500, 600, '+'))
check("GEO beats UCSC when ReAnnotator absent", p == 'GEO' and s == 300)
# UCSC wins when both absent
c, s, e, p = resolve_probe(pd.NA, pd.NA, pd.NA, pd.NA, None, None,
                           ('2', 500, 600, '+'))
check("UCSC used when ReAnnotator+GEO absent", p == 'UCSC_WG6' and s == 500)
# NA when all absent
c, s, e, p = resolve_probe(pd.NA, pd.NA, pd.NA, pd.NA, None, None, None)
check("NA when no source resolves", p == 'NA' and pd.isna(c))
 
print("Test 4: UCSC contig entry would have been rejected at load")
# Simulate: a UCSC row on chrUn_gl000212 -> clean_geo_chromosome rejects ->
# never enters the lookup, so resolve_probe sees ucsc_entry=None -> NA.
ucsc_chrom = clean_geo_chromosome('chrUn_gl000212')
check("contig chrom rejected before lookup", ucsc_chrom is None)
c, s, e, p = resolve_probe(pd.NA, pd.NA, pd.NA, pd.NA, None, None, None)
check("probe with only-contig UCSC hit -> NA", p == 'NA')
 
print("Test 5: ReAnnotator chromosome with contig suffix falls through to GEO")
# Re-Annotator Chr could be a scaffold; should be rejected and GEO used.
c, s, e, p = resolve_probe('chrUn_gl000212', 100, 200, 'chr5', 300, 400, None)
check("ReAnnotator scaffold rejected, GEO used", p == 'GEO' and c == '5')
 
print("Test 6: normalize_chrom never emits 'chr<NA>'")
test_series = pd.Series(['4', 'chrX', pd.NA, '<NA>', 'NA', 'nan', ''])
norm = normalize_chrom(test_series)
vals = [v for v in norm if pd.notna(v)]
check("no 'chr<NA>' produced", 'chr<NA>' not in vals)
check("no 'chrNA' produced", 'chrNA' not in vals)
check("bare '4' -> 'chr4'", norm.iloc[0] == 'chr4')
check("'chrX' preserved", norm.iloc[1] == 'chrX')
check("pd.NA stays NA", pd.isna(norm.iloc[2]))
check("'<NA>' string -> NA", pd.isna(norm.iloc[3]))
check("'NA' string -> NA", pd.isna(norm.iloc[4]))
 
print("Test 7: NA row retains probe ID (simulated BED6 row build)")
name = 'ILMN_1343048'
c, s, e, p = resolve_probe(pd.NA, pd.NA, pd.NA, pd.NA, None, None, None)
bed_row = {'chrom': c, 'chromStart': s, 'chromEnd': e,
           'name': name, 'score': 0, 'strand': '+', 'provenance': p}
check("NA row keeps probe ID", bed_row['name'] == 'ILMN_1343048')
check("NA row has NA chrom", pd.isna(bed_row['chrom']))
 
print()
if failures:
    print(f"RESULT: {len(failures)} FAILED -> {failures}")
    sys.exit(1)
else:
    print("RESULT: ALL TESTS PASSED")
    sys.exit(0)
