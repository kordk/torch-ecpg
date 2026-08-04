import argparse
import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import gseapy
import mygene
import pyranges as pr
import time
import urllib.request
import scipy.stats as stats
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def download_gencode_gtf(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/GRCh37_mapping/gencode.v49lift37.annotation.gtf.gz"
    filename = url.split('/')[-1]
    filepath = os.path.join(target_dir, filename)
    if not os.path.exists(filepath):
        logger.info(f"Downloading GENCODE GTF track from {url} ...")
        try:
            if not url.lower().startswith(('http://', 'https://')):
                raise ValueError(f"Refusing to download non-http(s) URL: {url}")
            urllib.request.urlretrieve(url, filepath)  # nosec B310 - scheme validated to http(s) above
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None
    return filepath

def get_illumina_coordinates(unmapped_ids):
    ucsc_file = 'demo/ucsc_illuminaProbes.hg19.txt'
    bed_files = ['demo/annoHT12.hg19.bed6', 'demo/annoHT12_comprehensive.hg19.bed6']
    coords = []
    if os.path.exists(ucsc_file):
        try:
            with open(ucsc_file) as fh:
                first = fh.readline().strip().split('\t')
            has_header = (first and first[0] in ('#bin', 'bin')) or ('chrom' in first)
            if has_header:
                udf = pd.read_csv(ucsc_file, sep='\t')
                udf.columns = [c.lstrip('#') for c in udf.columns]
            else:
                UCSC_COLNAMES = ['bin', 'chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'chromStarts']
                udf = pd.read_csv(ucsc_file, sep='\t', header=None, names=UCSC_COLNAMES)
            needed = ['chrom', 'chromStart', 'chromEnd', 'name']
            if all(c in udf.columns for c in needed):
                udf = udf[needed]
                _udf_before = len(udf)
                udf = udf.dropna(subset=['name'])
                logger.info(
                    f"Drop site runEnrichment.ucsc_recovery[name]: dropped UCSC "
                    f"rows with missing 'name': {_udf_before} -> {len(udf)} "
                    f"({_udf_before - len(udf)} dropped)"
                )
                udf = udf[udf['name'].isin(unmapped_ids)]
                for _, row in udf.iterrows():
                    coords.append({'Chromosome': row['chrom'], 'Start': int(row['chromStart']), 'End': int(row['chromEnd']), 'name': row['name']})
                found_ids = set([c['name'] for c in coords])
                unmapped_ids = list(set(unmapped_ids) - found_ids)
        except Exception as e:
            logger.error(f"Error reading UCSC file: {e}")

    for bed_file in bed_files:
        if not unmapped_ids:
            break
        if os.path.exists(bed_file):
            try:
                bdf = pd.read_csv(bed_file, sep='\t')
                if 'chrom' not in bdf.columns or 'name' not in bdf.columns:
                    bdf = pd.read_csv(bed_file, sep='\t', header=None)
                    if len(bdf.columns) >= 4:
                        bdf.columns = ['chrom', 'chromStart', 'chromEnd', 'name'] + list(bdf.columns[4:])
                bdf = bdf[bdf['name'].isin(unmapped_ids)]
                for _, row in bdf.iterrows():
                    if pd.notna(row['chrom']) and pd.notna(row['chromStart']) and pd.notna(row['chromEnd']):
                        coords.append({'Chromosome': str(row['chrom']), 'Start': int(float(row['chromStart'])), 'End': int(float(row['chromEnd'])), 'name': row['name']})
                found_ids = set([c['name'] for c in coords])
                unmapped_ids = list(set(unmapped_ids) - found_ids)
            except Exception as e:
                logger.error(f"Error reading BED file {bed_file}: {e}")

    return pd.DataFrame(coords) if coords else pd.DataFrame(columns=['Chromosome', 'Start', 'End', 'name'])

class GencodeMapper:
    def __init__(self, encode_dir="encode_beds"):
        self.encode_dir = encode_dir
        self.gencode_pr = None
        self.genes_pr = None

    def load(self):
        if self.genes_pr is not None:
            return True
        gtf_path = download_gencode_gtf(self.encode_dir)
        if gtf_path:
            try:
                logger.info("Loading GENCODE GTF globally...")
                self.gencode_pr = pr.read_gtf(gtf_path)
                self.genes_pr = self.gencode_pr[self.gencode_pr.Feature == 'gene']
                return True
            except Exception as e:
                logger.error(f"Error loading GENCODE GTF: {e}")
        return False

gencode_mapper = GencodeMapper()

def clean_and_translate_gene_ids(gene_ids):
    if not gene_ids:
        return [], 0

    cleaned_ids = [str(g).split('.')[0] for g in gene_ids]
    is_illumina = any(g.startswith('ILMN_') for g in cleaned_ids)

    mapped_symbols = set()
    unmapped_ids = []

    if is_illumina:
        reann_file = 'demo/reannotator_humanHt12v4.txt'
        still_unmapped = cleaned_ids.copy()

        if os.path.exists(reann_file):
            try:
                reann_df = pd.read_csv(reann_file, sep='\t')
                if 'X.PROBE_ID' in reann_df.columns and 'Gene_symbol' in reann_df.columns:
                    _reann_before = len(reann_df)
                    reann_df = reann_df.dropna(subset=['Gene_symbol'])
                    logger.info(
                        f"Drop site runEnrichment.reannotator[Gene_symbol]: "
                        f"dropped probes with missing 'Gene_symbol': "
                        f"{_reann_before} -> {len(reann_df)} "
                        f"({_reann_before - len(reann_df)} dropped)"
                    )
                    mapping = dict(zip(reann_df['X.PROBE_ID'], reann_df['Gene_symbol']))
                    new_unmapped = []
                    for g in still_unmapped:
                        if g in mapping:
                            syms = str(mapping[g]).split(',')
                            for s in syms:
                                mapped_symbols.add(s.strip())
                        else:
                            new_unmapped.append(g)
                    still_unmapped = new_unmapped
            except Exception as e:
                logger.error(f"Error reading Re-Annotator file: {e}")

        if still_unmapped:
            coords_df = get_illumina_coordinates(still_unmapped)
            if not coords_df.empty and gencode_mapper.load():
                try:
                    if not coords_df['Chromosome'].astype(str).str.startswith('chr').all():
                        coords_df['Chromosome'] = 'chr' + coords_df['Chromosome'].astype(str)
                    probes_pr = pr.PyRanges(coords_df)
                    joined = probes_pr.join(gencode_mapper.genes_pr, apply_strand_suffix=False)

                    if not joined.df.empty and 'gene_name' in joined.df.columns:
                        for _, row in joined.df.iterrows():
                            if pd.notna(row['gene_name']):
                                mapped_symbols.add(str(row['gene_name']).strip())
                                if row['name'] in still_unmapped:
                                    still_unmapped.remove(row['name'])
                except Exception as e:
                    logger.error(f"Error during GENCODE intersection: {e}")
        unmapped_ids = still_unmapped
    else:
        mg = mygene.MyGeneInfo()
        try:
            results = mg.querymany(cleaned_ids, scopes='ensembl.gene', fields='symbol', species='human', verbose=False)
            for res in results:
                if 'symbol' in res:
                    mapped_symbols.add(res['symbol'])
                else:
                    unmapped_ids.append(res['query'])
        except Exception as e:
            logger.error(f"Error querying mygene: {e}")
            unmapped_ids = cleaned_ids

    return list(mapped_symbols), len(unmapped_ids)

def detect_inflection(y, method='auto'):
    n = len(y)
    if n < 3:
        return None, "Not enough points"

    x = np.arange(n)
    use_kneed = False
    if method in ['auto', 'kneed']:
        try:
            from kneed import KneeLocator
            use_kneed = True
        except ImportError:
            if method == 'kneed':
                logger.error("kneed package not found but required by --inflection-method kneed.")
            use_kneed = False

    if use_kneed:
        kl = KneeLocator(x, y, curve='convex', direction='decreasing')
        knee = kl.knee
        if knee is not None:
            return knee, "kneed"

    x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
    y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])

    norm_points = np.column_stack((x_norm, y_norm))
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return 0, "chord"
    line_unitvec = line_vec / line_len
    p_vec = norm_points - p1
    t = np.dot(p_vec, line_unitvec)
    nearest = p1 + np.outer(t, line_unitvec)
    dist = np.linalg.norm(norm_points - nearest, axis=1)
    knee = np.argmax(dist)
    return knee, "chord"

def download_encode_files(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    urls = {
        'ChromHMM': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHmm/wgEncodeBroadHmmGm12878HMM.bed.gz',
    }

    filepaths = {}
    for key, url in urls.items():
        filename = url.split('/')[-1]
        filepath = os.path.join(target_dir, filename)
        if not os.path.exists(filepath):
            try:
                if not url.lower().startswith(('http://', 'https://')):
                    raise ValueError(f"Refusing to download non-http(s) URL: {url}")
                urllib.request.urlretrieve(url, filepath)  # nosec B310 - scheme validated to http(s) above
            except Exception as e:
                logger.error(f"Error downloading {url}: {e}")
                return None
        filepaths[key] = filepath
    return filepaths

def run_fisher_exact(hits_pr, background_pr, encode_pr):
    A = len(hits_pr.overlap(encode_pr).df.drop_duplicates(subset=['Chromosome', 'Start', 'End'])) if not hits_pr.overlap(encode_pr).df.empty else 0
    B = len(hits_pr) - A
    bg_overlap = len(background_pr.overlap(encode_pr).df.drop_duplicates(subset=['Chromosome', 'Start', 'End'])) if not background_pr.overlap(encode_pr).df.empty else 0
    C = max(0, bg_overlap - A)
    D = max(0, (len(background_pr) - len(hits_pr)) - C)

    table = [[A, B], [C, D]]
    _, p_val = stats.fisher_exact(table)

    fe = (A / (A + B)) / (C / (C + D)) if (C > 0 and (A + B) > 0 and (C + D) > 0) else 0.0
    return A, fe, p_val

def run_encode_enrichment(args, fdr_genes_by_region, significant_cpgs, significant_cpgs_by_region):
    if not args.background_bed:
        logger.error("--encode-enrichment requires --background-bed to be specified.")
        return
    encode_files = download_encode_files(args.encode_bed_dir)
    if not encode_files:
        logger.error("Failed to prepare ENCODE bed files. Skipping ENCODE enrichment.")
        return
    try:
        if args.background_bed.endswith('.gz'):
            bg_df = pd.read_csv(args.background_bed, sep='\t', header=None, compression='gzip')
        else:
            bg_df = pd.read_csv(args.background_bed, sep='\t', header=None)

        if len(bg_df.columns) >= 3:
            bg_df.columns = ['Chromosome', 'Start', 'End'] + list(bg_df.columns[3:])
        else:
            logger.error("Background BED file must have at least 3 columns.")
            return

        if not bg_df['Chromosome'].astype(str).str.startswith('chr').all():
            bg_df['Chromosome'] = 'chr' + bg_df['Chromosome'].astype(str)
        bg_pr = pr.PyRanges(bg_df)
    except Exception as e:
        logger.error(f"Error reading background BED file: {e}")
        return

    hits_df = pd.DataFrame(list(significant_cpgs), columns=['Chromosome', 'Start'])
    hits_df['End'] = hits_df['Start'] + 1
    if not hits_df['Chromosome'].astype(str).str.startswith('chr').all():
        hits_df['Chromosome'] = 'chr' + hits_df['Chromosome'].astype(str)
    hits_pr = pr.PyRanges(hits_df)

    hits_pr_by_region = {}
    for region, cpgs in significant_cpgs_by_region.items():
        r_df = pd.DataFrame(list(cpgs), columns=['Chromosome', 'Start'])
        r_df['End'] = r_df['Start'] + 1
        if not r_df['Chromosome'].astype(str).str.startswith('chr').all():
            r_df['Chromosome'] = 'chr' + r_df['Chromosome'].astype(str)
        hits_pr_by_region[region] = pr.PyRanges(r_df)

    enrichment_results = []
    chromhmm_df = pd.read_csv(encode_files['ChromHMM'], sep='\t', header=None, usecols=[0, 1, 2, 3], names=['Chromosome', 'Start', 'End', 'State'])
    states_of_interest = chromhmm_df['State'].unique()

    for state in states_of_interest:
        state_df = chromhmm_df[chromhmm_df['State'] == state]
        encode_pr = pr.PyRanges(state_df)

        A, fe, pval = run_fisher_exact(hits_pr, bg_pr, encode_pr)
        enrichment_results.append({
            'Annotation Track': 'ChromHMM',
            'State/Region': f'Global: {state}',
            'Region_Category': 'Global',
            'State': state,
            'Overlap Count (A)': A,
            'Fold Enrichment': fe,
            'P-value': pval
        })

        for region, r_pr in hits_pr_by_region.items():
            r_A, r_fe, r_pval = run_fisher_exact(r_pr, bg_pr, encode_pr)
            enrichment_results.append({
                'Annotation Track': 'ChromHMM',
                'State/Region': f'{region}: {state}',
                'Region_Category': region,
                'State': state,
                'Overlap Count (A)': r_A,
                'Fold Enrichment': r_fe,
                'P-value': r_pval
            })

    res_df = pd.DataFrame(enrichment_results)
    if 'P-value' in res_df.columns:
        from statsmodels.stats.multitest import multipletests
        pvals = res_df['P-value'].values
        _, p_adj, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        res_df['Adj P-value'] = p_adj

    csv_out = os.path.join(args.out_dir, "encode_enrichment_results.csv")
    res_df.to_csv(csv_out, index=False)
    logger.info(f"Saved ENCODE enrichment results to {csv_out}")

def run_enrichr(method, args, genes_by_region):
    enrichment_dir = os.path.join(args.out_dir, "enrichment_results")
    if not os.path.exists(enrichment_dir):
        os.makedirs(enrichment_dir)

    libraries = args.enrichment_libraries
    valid_libraries = set(gseapy.get_library_name())
    validated_libraries = []
    for lib in libraries:
        if lib in valid_libraries:
            validated_libraries.append(lib)
        else:
            logger.warning(f"Enrichment library '{lib}' is not available in gseapy. Skipping.")

    if not validated_libraries:
        logger.error("No valid enrichment libraries available. Skipping enrichment.")
        return

    for region, gene_dict in genes_by_region.items():
        sorted_genes = sorted(gene_dict.keys(), key=lambda g: gene_dict[g])
        if len(sorted_genes) > args.enrichment_max_genes:
            logger.info(f"Processing {method} region: {region} with {len(sorted_genes)} significant Ensembl IDs (capping to top {args.enrichment_max_genes})")
            genes_to_submit = sorted_genes[:args.enrichment_max_genes]
        else:
            logger.info(f"Processing {method} region: {region} with {len(sorted_genes)} significant Ensembl IDs")
            genes_to_submit = sorted_genes

        mapped_symbols, unmapped_count = clean_and_translate_gene_ids(genes_to_submit)
        if not mapped_symbols:
            logger.warning(f"Skipping enrichment for {region} ({method}) due to no mapped gene symbols.")
            continue

        for library in validated_libraries:
            max_retries = 3
            base_delay = 5
            enr = None
            for attempt in range(max_retries + 1):
                try:
                    if args.dry_run_enrichment:
                        if attempt == 0:
                            raise Exception("Simulated HTTP 504 error during dry run.")
                        else:
                            class MockEnr:
                                results = pd.DataFrame(columns=['Term', 'Overlap', 'P-value', 'Adjusted P-value', 'Genes'])
                            enr = MockEnr()
                            break
                    enr = gseapy.enrichr(
                        gene_list=mapped_symbols,
                        gene_sets=library,
                        organism='human',
                        outdir=None,
                        no_plot=True,
                    )
                    break
                except Exception as e:
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Error running gseapy for {library} ({method}): {e}. Retrying in {delay} seconds (Attempt {attempt + 1}/{max_retries})...")
                        time.sleep(delay)
                    else:
                        logger.error(f"Failed to run gseapy for {library} ({method}) after {max_retries} retries: {e}")

            if enr is not None and enr.results is not None and not enr.results.empty:
                sig_res = enr.results[enr.results['Adjusted P-value'] < 0.05]
                if not sig_res.empty:
                    csv_filename = f"{region}_{method}_{library}_enrichment.csv".replace(" ", "_").replace("/", "_")
                    csv_path = os.path.join(enrichment_dir, csv_filename)
                    columns_to_save = ['Term', 'Overlap', 'P-value', 'Adjusted P-value', 'Genes']
                    columns_to_save = [col for col in columns_to_save if col in sig_res.columns]
                    sig_res[columns_to_save].to_csv(csv_path, index=False)
                    logger.info(f"Saved {len(sig_res)} significant terms to {csv_filename}")
                else:
                    logger.info(f"No significant terms found (Adjusted P-value < 0.05) for {region} ({method}) in {library}.")
            elif enr is not None:
                logger.info(f"No enrichment results returned for {region} ({method}) in {library}.")

def main():
    parser = argparse.ArgumentParser(description="Standalone Functional and ENCODE Enrichment Analysis.")
    parser.add_argument("--fdr-input", default="output/summarized.parquet", help="Path to FDR input Parquet file.")
    parser.add_argument("--ig-input", default="output/bootstrap_merged.parquet", help="Path to IG input Parquet file.")
    parser.add_argument("--out-dir", default=".", help="Output directory for results.")
    parser.add_argument("--rank-by", nargs="+", choices=["fdr", "ig"], default=["fdr"], help="Methods to run enrichment on.")
    parser.add_argument("--fdr-threshold", type=float, default=0.05, help="FDR threshold for significance.")
    parser.add_argument("--fdr-column", default="fdr_est", help="Column carrying the FDR estimate used for significance selection.")
    parser.add_argument("--ig-inflection-method", default="auto", choices=["auto", "kneed", "chord"], help="Method for IG inflection detection.")
    parser.add_argument("--encode-enrichment", action="store_true", help="Run ENCODE enrichment analysis.")
    parser.add_argument("--encode-bed-dir", default="encode_beds", help="Directory for ENCODE BED files.")
    parser.add_argument("--background-bed", help="Path to background universe BED file.")
    parser.add_argument("--enrichment-max-genes", type=int, default=3000, help="Maximum number of unique genes to submit.")
    parser.add_argument("--enrichment-libraries", nargs="+", default=["GO_Biological_Process_2021", "KEGG_2021_Human", "WikiPathway_2023_Human"], help="List of Enrichr libraries.")
    parser.add_argument("--dry-run-enrichment", action="store_true", help="Simulate functional enrichment API calls.")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Rows per chunk for reading Parquet.")
    parser.add_argument("--df", type=float, default=100, help="Degrees of freedom for fallback p-value calculation.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    fdr_genes_by_region = defaultdict(dict)
    ig_genes_by_region = defaultdict(dict)
    fdr_significant_cpgs = set()
    fdr_cpgs_by_region = defaultdict(set)

    # 1. FDR Path
    if "fdr" in args.rank_by:
        logger.info(f"Running FDR path using {args.fdr_input}...")
        if not os.path.exists(args.fdr_input):
            logger.error(f"FDR input file {args.fdr_input} does not exist.")
        else:
            try:
                parquet_file = pq.ParquetFile(args.fdr_input)
                if args.fdr_column not in parquet_file.schema.names:
                    logger.error(
                        f"FDR column '{args.fdr_column}' not found in {args.fdr_input}. "
                        f"Observed columns: {list(parquet_file.schema.names)}. "
                        "Significance selection must be made on an FDR estimate; "
                        "refusing to fall back to raw p-values."
                    )
                    sys.exit(1)
                n_rows_read = 0
                n_rows_selected = 0

                for batch in parquet_file.iter_batches(batch_size=args.chunk_size):
                    df_chunk = batch.to_pandas()
                    if df_chunk.index.names != [None]:
                        df_chunk = df_chunk.reset_index()

                    fdr_vals = df_chunk[args.fdr_column].astype(np.float64).values
                    sig_mask = fdr_vals <= args.fdr_threshold
                    n_rows_read += len(df_chunk)
                    n_rows_selected += int(sig_mask.sum())
                    if sig_mask.any():
                        sig_df = df_chunk[sig_mask]
                        if 'region' in sig_df.columns:
                            for region, group in sig_df.groupby('region'):
                                if 'gt_id' in sig_df.columns:
                                    _grp_gene = group.dropna(subset=['gt_id'])
                                    logger.info(
                                        f"Drop site runEnrichment.fdr_genes[gt_id] "
                                        f"region={region}: dropped significant rows "
                                        f"with missing 'gt_id': {len(group)} -> "
                                        f"{len(_grp_gene)} "
                                        f"({len(group) - len(_grp_gene)} dropped)"
                                    )
                                    for _, row in _grp_gene.iterrows():
                                        gene = row['gt_id']
                                        pval = row.get('precise_mt_p', row.get('mt_p', 1.0))
                                        if gene not in fdr_genes_by_region[region] or pval < fdr_genes_by_region[region][gene]:
                                            fdr_genes_by_region[region][gene] = pval
                                if args.encode_enrichment and 'mt_chrom' in sig_df.columns and 'mt_chromStart' in sig_df.columns:
                                    _grp_cpg = group.dropna(subset=['mt_chrom', 'mt_chromStart'])
                                    logger.info(
                                        f"Drop site runEnrichment.fdr_cpgs[mt_chrom,"
                                        f"mt_chromStart] region={region}: dropped "
                                        f"significant rows with missing CpG coords: "
                                        f"{len(group)} -> {len(_grp_cpg)} "
                                        f"({len(group) - len(_grp_cpg)} dropped)"
                                    )
                                    for _, row in _grp_cpg.iterrows():
                                        chrom = row['mt_chrom']
                                        start = int(row['mt_chromStart'])
                                        fdr_significant_cpgs.add((chrom, start))
                                        fdr_cpgs_by_region[region].add((chrom, start))
                logger.info(
                    f"FDR selection on '{args.fdr_column}' <= {args.fdr_threshold}: "
                    f"selected {n_rows_selected} of {n_rows_read} rows."
                )
                logger.info(f"FDR processing complete. Collected regions: {list(fdr_genes_by_region.keys())}")
            except Exception as e:
                logger.error(f"Error processing FDR path: {e}")

    # 2. IG Path
    if "ig" in args.rank_by:
        logger.info(f"Running IG path using {args.ig_input}...")
        if not os.path.exists(args.ig_input):
            logger.error(f"IG input file {args.ig_input} does not exist.")
        else:
            try:
                parquet_file = pq.ParquetFile(args.ig_input)
                columns = parquet_file.schema.names
                if 'mt_ig' not in columns:
                    logger.error("IG input file must contain 'mt_ig' column for IG selection path.")
                else:
                    ig_cols = [c for c in columns if c.endswith('_ig')]
                    cols_to_load = ['mt_id', 'gt_id', 'region'] + ig_cols
                    df = pq.read_table(args.ig_input, columns=cols_to_load).to_pandas()

                    df['total_ig'] = df[ig_cols].abs().sum(axis=1)
                    df['mt_ig_frac'] = (df['mt_ig'].abs() / df['total_ig']).fillna(0)
                    df = df.sort_values('mt_ig_frac', ascending=False).reset_index(drop=True)

                    knee, method_used = detect_inflection(df['mt_ig_frac'].values, args.ig_inflection_method)
                    top_n = knee + 1 if knee is not None else len(df)
                    logger.info(f"IG inflection point detected at rank {top_n} (method: {method_used}). Selecting top {top_n} pairs.")

                    top_df = df.head(top_n)
                    if 'region' in top_df.columns:
                        for region, group in top_df.groupby('region'):
                            _grp_ig = group.dropna(subset=['gt_id'])
                            logger.info(
                                f"Drop site runEnrichment.ig_genes[gt_id] "
                                f"region={region}: dropped top-ranked rows with "
                                f"missing 'gt_id': {len(group)} -> {len(_grp_ig)} "
                                f"({len(group) - len(_grp_ig)} dropped)"
                            )
                            for _, row in _grp_ig.iterrows():
                                gene = row['gt_id']
                                frac = row['mt_ig_frac']
                                if gene not in ig_genes_by_region[region] or frac > ig_genes_by_region[region][gene]:
                                    ig_genes_by_region[region][gene] = frac # Higher is better here
                    logger.info(f"IG processing complete. Collected regions: {list(ig_genes_by_region.keys())}")
            except Exception as e:
                logger.error(f"Error processing IG path: {e}")

    # Run Enrichment
    if fdr_genes_by_region:
        logger.info("Running Enrichr for FDR gene sets...")
        run_enrichr("fdr", args, fdr_genes_by_region)
    if ig_genes_by_region:
        # Note: we need to sort IG descending, so we negate the fractions when passing to run_enrichr, which sorts ascending by default
        ig_genes_by_region_neg = defaultdict(dict)
        for r, d in ig_genes_by_region.items():
            for g, v in d.items():
                ig_genes_by_region_neg[r][g] = -v
        logger.info("Running Enrichr for IG gene sets...")
        run_enrichr("ig", args, ig_genes_by_region_neg)

    # Comparison output
    if "fdr" in args.rank_by and "ig" in args.rank_by:
        enrichment_dir = os.path.join(args.out_dir, "enrichment_results")
        if os.path.exists(enrichment_dir):
            for region in set(fdr_genes_by_region.keys()).union(ig_genes_by_region.keys()):
                for lib in args.enrichment_libraries:
                    fdr_file = os.path.join(enrichment_dir, f"{region}_fdr_{lib}_enrichment.csv".replace(" ", "_").replace("/", "_"))
                    ig_file = os.path.join(enrichment_dir, f"{region}_ig_{lib}_enrichment.csv".replace(" ", "_").replace("/", "_"))

                    fdr_terms = set()
                    ig_terms = set()

                    if os.path.exists(fdr_file):
                        fdr_terms = set(pd.read_csv(fdr_file)['Term'])
                    if os.path.exists(ig_file):
                        ig_terms = set(pd.read_csv(ig_file)['Term'])

                    if fdr_terms or ig_terms:
                        shared = fdr_terms.intersection(ig_terms)
                        fdr_unique = fdr_terms - ig_terms
                        ig_unique = ig_terms - fdr_terms

                        comp_file = os.path.join(enrichment_dir, f"{region}_{lib}_comparison.txt".replace(" ", "_").replace("/", "_"))
                        with open(comp_file, 'w') as f:
                            f.write(f"Comparison of Enriched Terms for {region} - {lib}\n")
                            f.write("="*60 + "\n\n")
                            f.write(f"Shared Terms ({len(shared)}):\n")
                            for t in sorted(shared): f.write(f"  - {t}\n")
                            f.write(f"\nUnique to FDR ({len(fdr_unique)}):\n")
                            for t in sorted(fdr_unique): f.write(f"  - {t}\n")
                            f.write(f"\nUnique to IG ({len(ig_unique)}):\n")
                            for t in sorted(ig_unique): f.write(f"  - {t}\n")

    if args.encode_enrichment:
        logger.info("Running ENCODE enrichment...")
        run_encode_enrichment(args, fdr_genes_by_region, fdr_significant_cpgs, fdr_cpgs_by_region)

if __name__ == "__main__":
    main()
