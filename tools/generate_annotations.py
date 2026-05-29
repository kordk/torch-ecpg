import os
import GEOparse
import pandas as pd
from pyliftover import LiftOver
import warnings
import logging
warnings.filterwarnings('ignore')
 
 
def write_bed6(df, filepath):
    def normalize_chrom(series):
        s = series.astype(str).str.strip()
        s = s.str.replace(r'\.0$', '', regex=True)
        # Treat all missing-value spellings (including pandas' '<NA>') as NA
        # BEFORE any 'chr' prefixing, so we never produce 'chr<NA>'.
        s = s.mask(s.isin(['nan', 'None', 'NaN', 'NA', '<NA>', '']), other=pd.NA)
        s = s.apply(lambda c: c if (pd.isna(c) or str(c).startswith('chr')) else 'chr' + str(c))
        return s
 
    df = df.copy()
    if 'chrom' in df.columns:
        df['chrom'] = normalize_chrom(df['chrom'])
 
        valid_chroms = df['chrom'].dropna().astype(str)
        invalid_float = valid_chroms[valid_chroms.str.contains(r'\.\d+$')]
        invalid_prefix = valid_chroms[~valid_chroms.str.startswith('chr')]
 
        if not invalid_float.empty or not invalid_prefix.empty:
            offending = list(set(invalid_float) | set(invalid_prefix))
            logging.critical(f"Invalid chrom values in {filepath}: {offending}")
            raise ValueError(f"Corrupt chrom values detected: {offending}")
 
    total = len(df)
    valid_coords = df['chromStart'].notna().sum()
    na_coords = total - valid_coords
 
    prov_summary = ""
    if 'provenance' in df.columns:
        prov_counts = df['provenance'].value_counts(dropna=False).to_dict()
        prov_summary = " | Provenance: " + ", ".join([f"{k}: {v}" for k, v in prov_counts.items()])
        df = df.drop(columns=['provenance'])
 
    print(f"[{filepath}] Total: {total} | Valid: {valid_coords} | NA: {na_coords}{prov_summary}")
 
    df.to_csv(filepath, sep='\t', index=False)
 
 
# ---------------------------------------------------------------------------
# Chromosome sanitization helpers
# ---------------------------------------------------------------------------
 
# Canonical chromosomes we accept for regional analysis
_CANONICAL_CHROMS = {str(i) for i in range(1, 23)} | {'X', 'Y', 'M', 'MT'}
 
 
def clean_geo_chromosome(raw):
    """
    Sanitize a raw GEO / Re-Annotator 'Chromosome' value down to a canonical
    chromosome token, or return None if it cannot be resolved to one.
 
    Handles:
      - pipe-delimited primary + alternate contig:  'chr4|NT_113889.1' -> '4'
      - version-suffixed accessions:                'NT_113889.1'      -> None
      - chr-prefixed canonical labels:              'chr4', 'chrX'     -> '4','X'
      - bare canonical labels:                      '4', 'X', 'MT'     -> '4','X','MT'
      - unplaced / alt / hap contigs:               'chr17_random',
                                                    'chr6_cox_hap1'    -> None
      - pseudoautosomal ambiguous:                  'chrXY','chrYX'    -> None
      - stray header text:                          'chrom'            -> None
 
    Returns a bare canonical token (no 'chr' prefix); format_chrom adds the
    prefix later.
    """
    if pd.isna(raw):
        return None
 
    token = str(raw).strip()
    if not token or token.lower() in ('nan', 'none', 'chrom'):
        return None
 
    # Take the primary assignment before any pipe, then drop any version suffix
    primary = token.split('|')[0].strip()
    primary = primary.split('.')[0].strip()
 
    if not primary:
        return None
 
    # Strip a leading 'chr' so we can validate against canonical names
    bare = primary[3:] if primary.lower().startswith('chr') else primary
    bare = bare.strip()
 
    upper = bare.upper()
 
    # Reject unplaced / alternate scaffold accessions
    if (upper.startswith('NT_') or upper.startswith('NW_')
            or upper.startswith('GL') or upper.startswith('KI')):
        return None
 
    # Reject UCSC-style alternate / unplaced contigs (chrN_random, chrN_hapM,
    # chr6_cox_hap1, chr6_qbl_hap2, etc.) -- anything with an underscore.
    if '_' in bare:
        return None
 
    # Reject pseudoautosomal ambiguous labels (XY / YX) -- do not silently
    # assign PAR probes to a single sex chromosome.
    if upper in ('XY', 'YX'):
        return None
 
    # Normalize mitochondrial spelling to a canonical token
    if upper == 'MT':
        bare = 'MT'
    elif upper == 'M':
        bare = 'M'
 
    # Only accept canonical chromosomes
    if bare not in _CANONICAL_CHROMS:
        return None
 
    return bare
 
 
def parse_hg19_coords(val):
    if pd.isna(val) or not str(val).strip():
        return None, None
    val = str(val)
    parts = val.split(':')
    starts = []
    ends = []
    for p in parts:
        s_e = p.split('-')
        if len(s_e) == 2:
            try:
                starts.append(int(s_e[0]))
                ends.append(int(s_e[1]))
            except ValueError:
                pass
    if starts and ends:
        return min(starts), max(ends)
    return None, None
 
 
def format_chrom(chr_val):
    chr_val = str(chr_val).strip()
    if chr_val == 'MT':
        chr_val = 'M'
    return f"chr{chr_val}" if not chr_val.startswith("chr") else chr_val
 
 
# ---------------------------------------------------------------------------
# Liftover (hg19 -> hg38) with a small cache to avoid repeated lookups
# ---------------------------------------------------------------------------
 
lo = LiftOver('hg19', 'hg38')
_lift_cache = {}
 
 
def _lift_one(chrom_fmt, pos):
    key = (chrom_fmt, pos)
    if key in _lift_cache:
        return _lift_cache[key]
    res = lo.convert_coordinate(chrom_fmt, pos)
    out = res[0][1] if res and len(res) > 0 else None
    out_chrom = res[0][0] if res and len(res) > 0 else None
    _lift_cache[key] = (out_chrom, out)
    return out_chrom, out
 
 
def liftover_coords(chrom, start, end):
    c = format_chrom(chrom)
    out_chrom_s, new_start = _lift_one(c, int(start))
    if end is not None:
        _, new_end = _lift_one(c, int(end))
    else:
        new_end = new_start
 
    if out_chrom_s is not None and new_start is not None and new_end is not None:
        return str(out_chrom_s).replace('chr', ''), new_start, new_end
    return None, None, None
 
 
# ---------------------------------------------------------------------------
# Fetch platform tables
# ---------------------------------------------------------------------------
 
dest_dir = "temp_geo"
os.makedirs(dest_dir, exist_ok=True)
 
print("Fetching GPL21145 (EPIC)...")
gpl_epic = GEOparse.get_GEO("GPL21145", destdir=dest_dir)
print("Fetching GPL13534 (450k)...")
gpl_450k = GEOparse.get_GEO("GPL13534", destdir=dest_dir)
print("Fetching GPL10558 (HT-12 V4)...")
gpl_ht12_v4 = GEOparse.get_GEO("GPL10558", destdir=dest_dir)
print("Fetching GPL6947 (HT-12 V3)...")
gpl_ht12_v3 = GEOparse.get_GEO("GPL6947", destdir=dest_dir)
 
 
# ---------------------------------------------------------------------------
# Methylation (EPIC + 450k)
# ---------------------------------------------------------------------------
 
print("Processing Methylation Data...")
epic_df = gpl_epic.table[['ID', 'CHR', 'MAPINFO']].dropna(subset=['ID'])
epic_df['CHR'] = epic_df['CHR'].astype(str)
epic_df['MAPINFO'] = pd.to_numeric(epic_df['MAPINFO'], errors='coerce')
 
k450_df = gpl_450k.table[['ID', 'CHR', 'MAPINFO']].dropna(subset=['ID'])
k450_df['CHR'] = k450_df['CHR'].astype(str)
k450_df['MAPINFO'] = pd.to_numeric(k450_df['MAPINFO'], errors='coerce')
 
meth_df = pd.concat([epic_df, k450_df]).drop_duplicates(subset=['ID'])
# NOTE: unmapped probes are intentionally retained with NA coordinates.
 
meth_hg19 = []
meth_hg38 = []
 
for idx, row in meth_df.iterrows():
    name = row['ID']
    raw_chrom = row['CHR']
    start = row['MAPINFO']
 
    clean = clean_geo_chromosome(raw_chrom)
 
    if pd.isna(start) or clean is None:
        chrom = pd.NA
        start = pd.NA
        prov = 'NA'
    else:
        chrom = clean
        start = int(start)
        prov = 'GEO'
 
    meth_hg19.append({
        'chrom': chrom, 'chromStart': start, 'chromEnd': start,
        'name': name, 'score': 0, 'strand': '+', 'provenance': prov
    })
 
    c38, s38, e38 = None, None, None
    if not pd.isna(chrom) and not pd.isna(start):
        c38, s38, e38 = liftover_coords(chrom, start, start)
 
    if c38 is not None:
        meth_hg38.append({
            'chrom': c38, 'chromStart': s38, 'chromEnd': e38,
            'name': name, 'score': 0, 'strand': '+', 'provenance': prov
        })
    else:
        meth_hg38.append({
            'chrom': pd.NA, 'chromStart': pd.NA, 'chromEnd': pd.NA,
            'name': name, 'score': 0, 'strand': '+', 'provenance': 'NA'
        })
 
df_meth_hg19 = pd.DataFrame(meth_hg19)
df_meth_hg38 = pd.DataFrame(meth_hg38)
 
write_bed6(df_meth_hg19, "demo/annoEPIC_comprehensive.hg19.bed6")
write_bed6(df_meth_hg38, "demo/annoEPIC_comprehensive.hg38.bed6")
 
 
# ---------------------------------------------------------------------------
# Gene expression (HT-12 V4 + V3), Re-Annotator first, GEO fallback
# ---------------------------------------------------------------------------
 
print("Processing Gene Expression Data...")
ht12_v4_df = gpl_ht12_v4.table[['ID', 'Chromosome', 'Probe_Coordinates', 'Probe_Chr_Orientation']].dropna(subset=['ID'])
ht12_v3_df = gpl_ht12_v3.table[['ID', 'Chromosome', 'Probe_Coordinates', 'Probe_Chr_Orientation']].dropna(subset=['ID'])
 
ge_df = pd.concat([ht12_v4_df, ht12_v3_df]).drop_duplicates(subset=['ID'])
 
# NOTE: the Re-Annotator file is tab-delimited.
reannotator = pd.read_csv("demo/reannotator_humanHt12v4.txt", sep="\t")
reannotator = reannotator[['X.PROBE_ID', 'Chr', 'P_start', 'P_end', 'Strand']].drop_duplicates(subset=['X.PROBE_ID'])
reannotator.columns = ['ID', 'Re_Chr', 'Re_Start', 'Re_End', 'Re_Strand']
 
merged = ge_df.merge(reannotator, on='ID', how='left')
 
# ---------------------------------------------------------------------------
# UCSC Illumina WG-6 track (3rd-priority source for probes unmapped by
# Re-Annotator + GEO).  WG-6 and HT-12 share the same probe set / ILMN_ IDs.
#
# IMPORTANT coordinate convention:
#   UCSC chromStart is 0-BASED (per the track schema), whereas Re-Annotator
#   and GEO coordinates in this pipeline are 1-BASED.  To keep ONE convention
#   across the whole BED6 file we convert UCSC start -> 1-based with +1.
#   (chromEnd is the 1-based end in both conventions.)
#
# The UCSC table is the standard BED-style dump with columns:
#   bin, chrom, chromStart, chromEnd, name, score, strand, thickStart, ...
# We only need chrom, chromStart, chromEnd, name, strand.
# ---------------------------------------------------------------------------
UCSC_WG6_PATH = "demo/ucsc_illuminaProbes.hg19.txt"
 
UCSC_COLNAMES = [
    'bin', 'chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand',
    'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'chromStarts'
]
 
 
def load_ucsc_wg6(path):
    """
    Load the UCSC illuminaProbes hg19 table into a dict keyed by ILMN_ ID:
        { ID: (chrom_token, start_1based, end_1based, strand) }
    Returns an empty dict (with a warning) if the file is absent, so the
    pipeline degrades gracefully to the Re-Annotator + GEO result.
    """
    if not os.path.exists(path):
        logging.warning(
            f"UCSC WG-6 file not found at {path}; skipping UCSC recovery layer. "
            f"Download via the UCSC Table Browser (hg19, group=expression, "
            f"track=illuminaProbes, output=all fields) to enable it."
        )
        return {}
 
    # The UCSC dump may or may not carry a header line. Detect it.
    with open(path) as fh:
        first = fh.readline().strip().split('\t')
    has_header = (first and first[0] in ('#bin', 'bin')) or ('chrom' in first)
 
    if has_header:
        udf = pd.read_csv(path, sep='\t')
        udf.columns = [c.lstrip('#') for c in udf.columns]
    else:
        udf = pd.read_csv(path, sep='\t', header=None, names=UCSC_COLNAMES)
 
    needed = ['chrom', 'chromStart', 'chromEnd', 'name', 'strand']
    missing = [c for c in needed if c not in udf.columns]
    if missing:
        logging.warning(
            f"UCSC WG-6 file missing expected columns {missing}; "
            f"skipping UCSC recovery layer."
        )
        return {}
 
    udf = udf[needed].dropna(subset=['name'])
    # One row per probe ID (single best alignment); drop multi-mapping dupes.
    udf = udf.drop_duplicates(subset=['name'], keep='first')
 
    lookup = {}
    for _, r in udf.iterrows():
        chrom = clean_geo_chromosome(r['chrom'])   # rejects _random / chrUn / gl000
        if chrom is None:
            continue
        try:
            start0 = int(r['chromStart'])          # 0-based
            end1 = int(r['chromEnd'])              # 1-based end
        except (ValueError, TypeError):
            continue
        start1 = start0 + 1                         # convert to 1-based start
        strand = str(r['strand']).strip()
        if strand not in ('+', '-'):
            strand = '+'
        lookup[str(r['name']).strip()] = (chrom, start1, end1, strand)
 
    print(f"Loaded {len(lookup)} UCSC WG-6 probe mappings (after cleaning).")
    return lookup
 
 
ucsc_wg6 = load_ucsc_wg6(UCSC_WG6_PATH)
 
ge_hg19 = []
ge_hg38 = []
 
for idx, row in merged.iterrows():
    name = row['ID']
 
    chrom = None
    start = None
    end = None
    strand = '+'
    prov = 'NA'
 
    # Try Re-Annotator first
    if not pd.isna(row['Re_Chr']) and not pd.isna(row['Re_Start']) and not pd.isna(row['Re_End']):
        chrom = clean_geo_chromosome(row['Re_Chr'])
        if chrom is not None:
            start = int(row['Re_Start'])
            end = int(row['Re_End'])
            prov = 'ReAnnotator'
            if not pd.isna(row['Re_Strand']):
                re_str = str(row['Re_Strand']).strip()
                if re_str in ('+', '-'):
                    strand = re_str
 
    # Fallback to GEO if Re-Annotator did not resolve a clean chromosome
    if chrom is None:
        geo_chrom = clean_geo_chromosome(row['Chromosome'])
        if geo_chrom:
            coords = row['Probe_Coordinates']
            geo_start, geo_end = parse_hg19_coords(coords)
            if geo_start is not None and geo_end is not None:
                chrom = geo_chrom
                start = geo_start
                end = geo_end
                prov = 'GEO'
 
                if not pd.isna(row['Probe_Chr_Orientation']):
                    geo_strand = str(row['Probe_Chr_Orientation']).strip()
                    if geo_strand in ('+', '-'):
                        strand = geo_strand
 
    # Fallback to UCSC WG-6 if still unmapped. Coordinates are already
    # cleaned + converted to 1-based in load_ucsc_wg6().
    if chrom is None and name in ucsc_wg6:
        u_chrom, u_start, u_end, u_strand = ucsc_wg6[name]
        chrom = u_chrom
        start = u_start
        end = u_end
        strand = u_strand
        prov = 'UCSC_WG6'
 
    if not chrom or start is None or end is None:
        chrom = pd.NA
        start = pd.NA
        end = pd.NA
        prov = 'NA'
 
    ge_hg19.append({
        'chrom': format_chrom(chrom) if pd.notna(chrom) else pd.NA,
        'chromStart': start, 'chromEnd': end,
        'name': name, 'score': 0, 'strand': strand, 'provenance': prov
    })
 
    c38, s38, e38 = None, None, None
    if pd.notna(chrom) and pd.notna(start) and pd.notna(end):
        c38, s38, e38 = liftover_coords(chrom, start, end)
 
    if c38 is not None:
        ge_hg38.append({
            'chrom': format_chrom(c38), 'chromStart': s38, 'chromEnd': e38,
            'name': name, 'score': 0, 'strand': strand, 'provenance': prov
        })
    else:
        ge_hg38.append({
            'chrom': pd.NA, 'chromStart': pd.NA, 'chromEnd': pd.NA,
            'name': name, 'score': 0, 'strand': strand, 'provenance': 'NA'
        })
 
df_ge_hg19 = pd.DataFrame(ge_hg19)
df_ge_hg38 = pd.DataFrame(ge_hg38)
 
write_bed6(df_ge_hg19, "demo/annoHT12_comprehensive.hg19.bed6")
write_bed6(df_ge_hg38, "demo/annoHT12_comprehensive.hg38.bed6")
 
print("Done generating annotations.")

