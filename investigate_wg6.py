import json
import urllib.request
import os
import GEOparse
import pandas as pd

def fetch_ucsc_illuminaProbes():
    print("Fetching UCSC illuminaProbes track for hg19...")
    url = "https://api.genome.ucsc.edu/getData/track?genome=hg19;track=illuminaProbes"
    try:
        response = urllib.request.urlopen(url)  # nosec B310
        data = json.loads(response.read())
        return data
    except Exception as e:
        print(f"Error fetching UCSC data: {e}")
        return None

def parse_ucsc_data(data):
    probes = []
    if 'illuminaProbes' not in data:
        return pd.DataFrame()

    for chrom, items in data['illuminaProbes'].items():
        for item in items:
            probes.append({
                'ID': item['name'],
                'UCSC_Chrom': item['chrom'],
                'UCSC_Start': item['chromStart'],
                'UCSC_End': item['chromEnd'],
                'UCSC_Strand': item['strand']
            })
    return pd.DataFrame(probes)

def fetch_geo_data():
    dest_dir = "temp_geo"
    os.makedirs(dest_dir, exist_ok=True)

    print("Fetching GPL10558 (HT-12 V4)...")
    gpl_ht12_v4 = GEOparse.get_GEO("GPL10558", destdir=dest_dir)
    print("Fetching GPL6947 (HT-12 V3)...")
    gpl_ht12_v3 = GEOparse.get_GEO("GPL6947", destdir=dest_dir)

    ht12_v4_df = gpl_ht12_v4.table[['ID', 'Chromosome', 'Probe_Coordinates', 'Probe_Chr_Orientation']].dropna(how='all')
    ht12_v3_df = gpl_ht12_v3.table[['ID', 'Chromosome', 'Probe_Coordinates', 'Probe_Chr_Orientation']].dropna(how='all')

    return ht12_v4_df, ht12_v3_df

def format_chrom(chr_val):
    chr_val = str(chr_val).strip()
    if chr_val == 'MT':
        chr_val = 'M'
    return f"chr{chr_val}" if not chr_val.startswith("chr") else chr_val

def main():
    log_messages = []
    def log(msg):
        print(msg)
        log_messages.append(msg)

    log("=== Starting WG-6 Investigation ===")

    # 1. Fetch UCSC Data
    if os.path.exists("ucsc_illuminaProbes.json"):
        log("Loading UCSC data from local cache...")
        with open("ucsc_illuminaProbes.json", "r") as f:
            ucsc_raw = json.load(f)
    else:
        ucsc_raw = fetch_ucsc_illuminaProbes()
        with open("ucsc_illuminaProbes.json", "w") as f:
            json.dump(ucsc_raw, f)

    ucsc_df = parse_ucsc_data(ucsc_raw)
    log(f"Parsed {len(ucsc_df)} probes from UCSC illuminaProbes track.")

    if len(ucsc_df) > 0:
        first_probe = ucsc_df.iloc[0]
        log(f"UCSC track coordinate convention: 0-based start, 1-based end (BED format).")
        log(f"UCSC chromosome convention: '{first_probe['UCSC_Chrom']}' (chr-prefixed).")

    # 2. Fetch GEO Base Data to identify origin (V4/V3/Shared)
    ht12_v4_df, ht12_v3_df = fetch_geo_data()

    # Tag origin
    ht12_v4_df['Origin_V4'] = True
    ht12_v3_df['Origin_V3'] = True

    # Combine and deduplicate
    combined_ge = pd.merge(ht12_v4_df, ht12_v3_df, on=['ID', 'Chromosome', 'Probe_Coordinates', 'Probe_Chr_Orientation'], how='outer')
    combined_ge['Origin_V4'] = combined_ge['Origin_V4'].fillna(False)
    combined_ge['Origin_V3'] = combined_ge['Origin_V3'].fillna(False)

    combined_ge = combined_ge.drop_duplicates(subset=['ID'])
    log(f"Total unique probes in combined V4+V3 universe: {len(combined_ge)}")

    def get_origin_label(row):
        if row['Origin_V4'] and row['Origin_V3']: return 'Shared'
        if row['Origin_V4']: return 'V4-only'
        if row['Origin_V3']: return 'V3-only'
        return 'Unknown'

    combined_ge['Origin'] = combined_ge.apply(get_origin_label, axis=1)

    # 3. Load base pipeline output for unmapped probes
    base_df = pd.read_csv("demo/annoHT12.hg19.bed6", sep="\t")

    base_mapping = base_df.copy()
    base_mapping['ID'] = base_mapping['name']

    def is_mapped(row):
        if pd.isna(row['chromStart']) or str(row['chromStart']).upper() == 'NA': return False
        return True

    base_mapping['Mapped'] = base_mapping.apply(is_mapped, axis=1)

    def clean_coord(val):
        if pd.isna(val) or str(val).upper() == 'NA': return None
        try:
            return int(val)
        except ValueError:
            return None

    base_mapping['Base_Start'] = base_mapping['chromStart'].apply(clean_coord)
    base_mapping['Base_Chrom'] = base_mapping['chrom'].apply(lambda x: format_chrom(x) if not pd.isna(x) and str(x).upper() != 'NA' else None)

    base_mapping = pd.merge(base_mapping, combined_ge[['ID', 'Origin']], on='ID', how='left')

    unmapped_base = base_mapping[~base_mapping['Mapped']]
    log(f"Total unmapped probes in base pipeline: {len(unmapped_base)}")

    # 5. Incremental Recovery
    recovery_df = pd.merge(unmapped_base, ucsc_df, on='ID', how='inner')
    recovered_count = len(recovery_df)
    log(f"Number of residual unmapped probes recovered by UCSC track: {recovered_count}")

    true_residual = len(unmapped_base) - recovered_count
    log(f"Number of probes remaining unmapped even after UCSC: {true_residual}")

    log("\n--- Recovery Breakdown by Platform Origin ---")
    origin_counts = unmapped_base['Origin'].value_counts()
    recovered_origin_counts = recovery_df['Origin'].value_counts()

    for origin in origin_counts.index:
        unmapped_n = origin_counts.get(origin, 0)
        recovered_n = recovered_origin_counts.get(origin, 0)
        log(f"  {origin}: {recovered_n} recovered out of {unmapped_n} unmapped")

    log("\n--- Chromosome Distribution of UCSC-recovered Probes ---")
    if recovered_count > 0:
        chrom_counts = recovery_df['UCSC_Chrom'].value_counts()
        for chrom, count in chrom_counts.items():
            log(f"  {chrom}: {count}")
    else:
        log("  None")

    # 6. Concordance Cross-Check
    log("\n--- Concordance Cross-Check (Re-Annotator vs UCSC) ---")
    mapped_base = base_mapping[base_mapping['Mapped']].copy()

    concordance_df = pd.merge(mapped_base, ucsc_df, on='ID', how='inner')
    log(f"Probes present in both Base Pipeline and UCSC WG-6: {len(concordance_df)}")

    dist_exact = 0
    dist_1_25 = 0
    dist_26_1000 = 0
    dist_gt_1000 = 0
    dist_diff_chr = 0

    disagreements = []

    for idx, row in concordance_df.iterrows():
        base_chr = row['Base_Chrom']
        ucsc_chr = row['UCSC_Chrom']

        if base_chr != ucsc_chr:
            dist_diff_chr += 1
            if len(disagreements) < 5:
                disagreements.append(row)
            continue

        # UCSC is 0-based, Base Pipeline (Re-Annotator/GEO) is 1-based start.
        # Add 1 to UCSC start to compare directly.
        ucsc_start_1based = row['UCSC_Start'] + 1

        diff = abs(row['Base_Start'] - ucsc_start_1based)

        if diff == 0:
            dist_exact += 1
        elif diff <= 25:
            dist_1_25 += 1
        elif diff <= 1000:
            dist_26_1000 += 1
            if len(disagreements) < 5:
                disagreements.append(row)
        else:
            dist_gt_1000 += 1
            if len(disagreements) < 5:
                disagreements.append(row)

    log("\nOffset Distribution (comparing start coordinates):")
    log(f"  Exact match (0 bp): {dist_exact}")
    log(f"  Within 1-25 bp: {dist_1_25}")
    log(f"  Within 26-1000 bp: {dist_26_1000}")
    log(f"  > 1000 bp: {dist_gt_1000}")
    log(f"  Different chromosome: {dist_diff_chr}")

    if disagreements:
        log("\nExample Disagreements (diff chr or >25bp diff):")
        for d in disagreements:
            log(f"  ID: {d['ID']} | Base Pipeline: {d['Base_Chrom']}:{d['Base_Start']} | UCSC: {d['UCSC_Chrom']}:{d['UCSC_Start']}")

    with open("wg6_investigation.txt", "w") as f:
        f.write("\n".join(log_messages))

    log("\nInvestigation complete. Results saved to wg6_investigation.txt.")

if __name__ == "__main__":
    main()
