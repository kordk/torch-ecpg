"""Kennedy et al. 2018 (BMC Genomics 19:476) Figure 6: chromatin-feature
enrichment of eCpGs. Chunk E2: loaders, universe contract, and the two panels
as integer 2x2 tables. Statistics (E3) and the heatmap (E4) build on this.

  Panel A  universe = tested CpGs (the M-matrix row set that entered MLR),
           each CpG once. Rows = ALL + one per region label; a CpG is in row R
           iff it has >= 1 significant pair in region R.
           2x2 = (CpG in row set) x (CpG location overlaps feature).
  Panel B  universe = significant pairs, each pair once. Rows = region labels.
           2x2 = (pair.region == R) x (pair's CpG overlaps feature).

Coordinates: CpG as the 1-bp half-open interval [mt_chromStart, +1); tracks as
BED half-open. Chromosome names are normalised by chromatin_features.
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import chromatin_features as cf

logger = logging.getLogger(__name__)

REGION_ORDER = ['ALL', 'PROMOTER', 'GENEBODY', 'CIS5', 'CIS3', 'DISTAL5', 'DISTAL3', 'TRANS']
CIS_ROWS = {'PROMOTER', 'GENEBODY', 'CIS5', 'CIS3'}


class UniverseContractError(RuntimeError):
    pass


# ----------------------------------------------------------------- loaders
def load_universe_ids(m_matrix_path):
    """Row index of the M matrix (first column only; the matrix itself is not read)."""
    ids = pd.read_csv(m_matrix_path, usecols=[0], dtype=str).iloc[:, 0]
    ids = ids.astype(str).str.strip()
    if ids.duplicated().any():
        raise UniverseContractError(f'{m_matrix_path}: duplicate row ids in M matrix index')
    return ids.tolist()


def load_cpg_bed(bed_path):
    df = pd.read_csv(bed_path, sep='\t', header=None, comment='#', usecols=[0, 1, 3],
                     names=['chrom', 'start', 'mt_id'], dtype={0: str, 3: str})
    df['chrom'] = df['chrom'].map(cf.normalize_chrom)
    bad = df['chrom'].isna()
    if bad.any():
        logger.info(f'Drop site chromatin_enrichment.cpg_bed[chrom]: excluded {int(bad.sum())} of {len(df)} '
                    f'CpG coordinates on unrecognised chromosomes')
        df = df[~bad]
    df = df.drop_duplicates('mt_id')
    return df.set_index('mt_id')


def build_universe(universe_ids, cpg_bed):
    """Return (universe DataFrame indexed by mt_id with chrom/start, n_no_coords)."""
    u = pd.DataFrame(index=pd.Index(universe_ids, name='mt_id'))
    u = u.join(cpg_bed[['chrom', 'start']], how='left')
    no_coords = u['chrom'].isna()
    n_no = int(no_coords.sum())
    logger.info(f'Drop site chromatin_enrichment.universe[coords]: {n_no} of {len(u)} tested CpGs lack '
                f'coordinates in the CpG BED and are excluded from the universe')
    u = u[~no_coords].copy()
    u['start'] = u['start'].astype(np.int64)
    return u, n_no


def load_significant_pairs(catalog_path, sig_column, sig_threshold, chunk_size=200_000):
    pf = pq.ParquetFile(catalog_path)
    names = set(pf.schema.names)
    if sig_column not in names:
        raise UniverseContractError(
            f"significance column '{sig_column}' not found in {catalog_path}; observed columns: "
            f"{sorted(names)}. Refusing to fall back to a raw p-value.")
    need = ['mt_id', 'gt_id', 'region', sig_column]
    missing = [c for c in need if c not in names]
    if missing:
        raise UniverseContractError(f'catalog missing required columns {missing}')
    parts, n_read = [], 0
    for batch in pf.iter_batches(batch_size=chunk_size, columns=need):
        df = batch.to_pandas()
        n_read += len(df)
        vals = pd.to_numeric(df[sig_column], errors='coerce')
        sel = df[vals <= sig_threshold]
        if len(sel):
            parts.append(sel[['mt_id', 'gt_id', 'region']])
    sig = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['mt_id', 'gt_id', 'region'])
    sig['mt_id'] = sig['mt_id'].astype(str)
    sig['gt_id'] = sig['gt_id'].astype(str)
    sig['region'] = sig['region'].astype(str)
    n_before = len(sig)
    sig = sig.drop_duplicates(['mt_id', 'gt_id', 'region'])
    if len(sig) != n_before:
        logger.info(f'Drop site chromatin_enrichment.pairs[dup]: {n_before - len(sig)} duplicate pair rows collapsed')
    logger.info(f"Significance selection '{sig_column}' <= {sig_threshold}: {len(sig)} of {n_read} rows")
    return sig


def check_universe_contract(sig, universe):
    missing = sorted(set(sig['mt_id']) - set(universe.index))
    if missing:
        raise UniverseContractError(
            f'{len(missing)} significant CpG id(s) are not in the tested-CpG universe '
            f'(first 10: {missing[:10]}). The universe is the M-matrix row set that entered MLR; '
            f'a significant CpG outside it means the catalog and the M matrix are from different runs.')


# ------------------------------------------------------------------ tables
def region_rows(sig):
    present = [r for r in REGION_ORDER if r != 'ALL' and r in set(sig['region'])]
    extra = sorted(set(sig['region']) - set(REGION_ORDER))
    return present + extra


def annotate_overlaps(universe, tracks):
    """bool DataFrame (universe index x feature name)."""
    return pd.DataFrame({name: idx.overlap(universe['chrom'].to_numpy(), universe['start'].to_numpy())
                         for name, idx in tracks.items()}, index=universe.index)


def two_by_two(in_row, in_feat):
    a = int(np.sum(in_row & in_feat)); b = int(np.sum(in_row & ~in_feat))
    c = int(np.sum(~in_row & in_feat)); d = int(np.sum(~in_row & ~in_feat))
    return a, b, c, d


def build_panel_a(sig, universe, overlaps):
    rows = ['ALL'] + region_rows(sig)
    cpg_by_row = {'ALL': set(sig['mt_id'])}
    for r in rows[1:]:
        cpg_by_row[r] = set(sig.loc[sig['region'] == r, 'mt_id'])
    out = []
    uidx = universe.index
    for r in rows:
        in_row = uidx.isin(cpg_by_row[r]).astype(bool)
        n_row = int(in_row.sum())
        for feat in overlaps.columns:
            a, b, c, d = two_by_two(in_row, overlaps[feat].to_numpy())
            rec = dict(panel='A', row=r, cis=int(r in CIS_ROWS), feature=feat, n_row=n_row, a=a, b=b, c=c, d=d)
            out.append(rec)
    return pd.DataFrame(out)


def build_panel_b(sig, universe, overlaps):
    rows = region_rows(sig)
    pairs = sig.join(overlaps, on='mt_id', how='left')
    out = []
    for r in rows:
        in_row = (pairs['region'] == r).to_numpy()
        n_row = int(in_row.sum())
        for feat in overlaps.columns:
            a, b, c, d = two_by_two(in_row, pairs[feat].to_numpy().astype(bool))
            rec = dict(panel='B', row=r, cis=int(r in CIS_ROWS), feature=feat, n_row=n_row, a=a, b=b, c=c, d=d)
            out.append(rec)
    return pd.DataFrame(out)


# ----------------------------------------------------------------- outputs
def metrics_json(panel_a, panel_b, extra):
    def cells(df):
        return [dict(row=r.row, feature=r.feature, n_row=int(r.n_row), a=int(r.a), b=int(r.b), c=int(r.c), d=int(r.d))
                for r in df.itertuples()]
    out = dict(extra)
    out['panel_a'] = cells(panel_a)
    out['panel_b'] = cells(panel_b)
    return out


# --------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description='Kennedy Fig. 6 chromatin-feature enrichment of eCpGs.')
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--cpg-universe', required=True, help='M matrix whose row index is the tested-CpG set')
    ap.add_argument('--cpg-bed', required=True, help='M.bed6 (CpG coordinates)')
    ap.add_argument('--tracks', required=True, help='feature-track manifest TSV')
    ap.add_argument('--genome-build', required=True)
    ap.add_argument('--sig-column', default='fdr_est')
    ap.add_argument('--sig-threshold', type=float, default=0.05)
    ap.add_argument('--expect-n-universe', type=int, default=None, help='cross-check against the mlr log row count')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
    os.makedirs(args.out_dir, exist_ok=True)

    tracks, prov = cf.load_tracks(args.tracks, args.genome_build)
    ids = load_universe_ids(args.cpg_universe)
    if args.expect_n_universe is not None and len(ids) != args.expect_n_universe:
        raise UniverseContractError(f'universe has {len(ids)} rows; mlr log reported {args.expect_n_universe}')
    universe, n_no_coords = build_universe(ids, load_cpg_bed(args.cpg_bed))
    sig = load_significant_pairs(args.catalog, args.sig_column, args.sig_threshold)
    check_universe_contract(sig, universe)
    overlaps = annotate_overlaps(universe, tracks)
    pa = build_panel_a(sig, universe, overlaps)
    pb = build_panel_b(sig, universe, overlaps)
    pa.to_csv(os.path.join(args.out_dir, 'chromatin_enrichment_panelA.tsv'), sep='\t', index=False)
    pb.to_csv(os.path.join(args.out_dir, 'chromatin_enrichment_panelB.tsv'), sep='\t', index=False)
    meta = dict(genome_build=args.genome_build, sig_column=args.sig_column, sig_threshold=args.sig_threshold,
                n_universe_ids=len(ids), n_universe_no_coords=n_no_coords, n_universe=len(universe),
                n_sig_pairs=len(sig), n_sig_cpgs=int(sig['mt_id'].nunique()),
                catalog=os.path.abspath(args.catalog), catalog_sha256=cf.sha256_file(args.catalog),
                tracks_manifest_sha256=cf.sha256_file(args.tracks), track_provenance=prov)
    with open(os.path.join(args.out_dir, 'chromatin_enrichment_metrics.json'), 'w') as fh:
        json.dump(metrics_json(pa, pb, meta), fh, indent=1)
    logger.info(f'panel A: {len(pa)} cells; panel B: {len(pb)} cells; outputs in {args.out_dir}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (cf.BuildMismatch, cf.ManifestError, UniverseContractError) as e:
        logger.error(str(e))
        sys.exit(2)
