import os
import GEOparse
import pandas as pd
from pyliftover import LiftOver
import warnings
warnings.filterwarnings('ignore')

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

lo = LiftOver('hg19', 'hg38')

def liftover_coords(chrom, start, end):
    c = format_chrom(chrom)
    new_start = lo.convert_coordinate(c, start)
    new_end = lo.convert_coordinate(c, end) if end is not None else lo.convert_coordinate(c, start)

    if new_start and len(new_start) > 0 and new_end and len(new_end) > 0:
        return str(new_start[0][0]).replace('chr', ''), new_start[0][1], new_end[0][1]
    return None, None, None

print("Processing Methylation Data...")
epic_df = gpl_epic.table[['ID', 'CHR', 'MAPINFO']].dropna()
epic_df['CHR'] = epic_df['CHR'].astype(str)
epic_df['MAPINFO'] = pd.to_numeric(epic_df['MAPINFO'], errors='coerce')

k450_df = gpl_450k.table[['ID', 'CHR', 'MAPINFO']].dropna()
k450_df['CHR'] = k450_df['CHR'].astype(str)
k450_df['MAPINFO'] = pd.to_numeric(k450_df['MAPINFO'], errors='coerce')

meth_df = pd.concat([epic_df, k450_df]).drop_duplicates(subset=['ID'])
meth_df = meth_df.dropna(subset=['MAPINFO'])
meth_df['MAPINFO'] = meth_df['MAPINFO'].astype(int)

meth_hg19 = []
meth_hg38 = []

for idx, row in meth_df.iterrows():
    chrom = str(row['CHR']).strip()
    if not chrom: continue
    start = row['MAPINFO']
    name = row['ID']
    meth_hg19.append({
        'chrom': chrom, 'chromStart': start, 'chromEnd': start,
        'name': name, 'score': 0, 'strand': '+'
    })
    c38, s38, e38 = liftover_coords(chrom, start, start)
    if c38 is not None:
        meth_hg38.append({
            'chrom': c38, 'chromStart': s38, 'chromEnd': e38,
            'name': name, 'score': 0, 'strand': '+'
        })

df_meth_hg19 = pd.DataFrame(meth_hg19)
df_meth_hg38 = pd.DataFrame(meth_hg38)

df_meth_hg19.to_csv("demo/annoEPIC_comprehensive.hg19.bed6", sep="\t", index=False)
df_meth_hg38.to_csv("demo/annoEPIC_comprehensive.hg38.bed6", sep="\t", index=False)

print("Processing Gene Expression Data...")
ht12_v4_df = gpl_ht12_v4.table[['ID', 'Chromosome', 'Probe_Coordinates']].dropna()
ht12_v3_df = gpl_ht12_v3.table[['ID', 'Chromosome', 'Probe_Coordinates']].dropna()

ge_df = pd.concat([ht12_v4_df, ht12_v3_df]).drop_duplicates(subset=['ID'])

ge_hg19 = []
ge_hg38 = []

for idx, row in ge_df.iterrows():
    chrom = str(row['Chromosome']).strip()
    if not chrom: continue
    coords = row['Probe_Coordinates']
    start, end = parse_hg19_coords(coords)
    name = row['ID']

    if start is not None and end is not None:
        ge_hg19.append({
            'chrom': chrom, 'chromStart': start, 'chromEnd': end,
            'name': name, 'score': 0, 'strand': '+'
        })

        c38, s38, e38 = liftover_coords(chrom, start, end)
        if c38 is not None:
            ge_hg38.append({
                'chrom': c38, 'chromStart': s38, 'chromEnd': e38,
                'name': name, 'score': 0, 'strand': '+'
            })

df_ge_hg19 = pd.DataFrame(ge_hg19)
df_ge_hg38 = pd.DataFrame(ge_hg38)

df_ge_hg19.to_csv("demo/annoHT12_comprehensive.hg19.bed6", sep="\t", index=False)
df_ge_hg38.to_csv("demo/annoHT12_comprehensive.hg38.bed6", sep="\t", index=False)

print("Done generating annotations.")
