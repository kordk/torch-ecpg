#!/usr/bin/env python3
"""Derive an ILMN-keyed probe->gene map from a GENCODE GTF and the probe BED,
following Kennedy et al. 2018 supplemental methods:

  * a probe is annotated to a gene when the probe interval overlaps that
    gene's merged EXON intervals by MORE THAN 25 bp
  * where several genes qualify, poorly-characterised names (KIAA*, FLJ*,
    LOC*, *orf*) and read-through GENES (GENCODE tag readthrough_gene) are
    removed from the duplicates; exons of read-through TRANSCRIPTS are
    excluded from the overlap test but do not disqualify their gene
  * the surviving genes are all retained, and the region span is the UNION
    of their gene-feature spans
  * a gene's span is min(TSS)..max(TES), which the GENCODE `gene` feature
    already provides

Stdlib only. 1-based inclusive coordinates throughout, matching M.bed6/G.bed6.
"""

import argparse
import csv
import gzip
import logging
import os
import re
import sys

logger = logging.getLogger(__name__)

MIN_EXON_OVERLAP = 25          # Kennedy: "more than 25 bp"
POORLY_CHARACTERISED = re.compile(r"^(KIAA|FLJ|LOC)|orf", re.IGNORECASE)


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def _attr(attributes, key):
    for a in attributes.split(";"):
        a = a.strip()
        if a.startswith(key + " "):
            return a[len(key) + 1:].strip('"')
    return None


def _tags(attributes):
    out = []
    for a in attributes.split(";"):
        a = a.strip()
        if a.startswith("tag "):
            out.append(a[4:].strip('"'))
    return out


def parse_gtf(path):
    """Return (genes, exons).

    genes: gene_id -> dict(chrom,start,end,strand,name,gene_type,readthrough)
    exons: gene_id -> list of (start, end)   [1-based inclusive, unmerged]
    """
    genes, exons, readthrough = {}, {}, set()
    with _open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            feat = f[2]
            if feat not in ("gene", "exon"):
                continue
            gid = _attr(f[8], "gene_id")
            if not gid:
                continue
            if feat == "gene":
                genes[gid] = {
                    "chrom": f[0],
                    "start": int(f[3]),          # GTF is already 1-based
                    "end": int(f[4]),
                    "strand": f[6],
                    "name": _attr(f[8], "gene_name") or gid,
                    "gene_type": _attr(f[8], "gene_type") or "",
                }
                # GENCODE marks read-through LOCI with readthrough_gene. Only that
                # tag disqualifies a gene. readthrough_transcript is an ISOFORM
                # property -- ~38k transcripts carry it across ordinary genes, so
                # promoting it to the gene would discard thousands of real genes.
                if "readthrough_gene" in _tags(f[8]):
                    readthrough.add(gid)
            elif feat == "exon":
                # A read-through isoform extends a gene's apparent exon footprint
                # into its neighbour. Since gene assignment is by exon overlap,
                # those exons would attach probes to the wrong gene. Exclude the
                # exon, keep the gene.
                if "readthrough_transcript" in _tags(f[8]):
                    continue
                exons.setdefault(gid, []).append((int(f[3]), int(f[4])))
    for gid in readthrough:
        if gid in genes:
            genes[gid]["readthrough"] = True
    for g in genes.values():
        g.setdefault("readthrough", False)
    return genes, exons


def merge_intervals(ivs):
    """Kennedy: overlapping exons of one gene collapse to maximal intervals."""
    if not ivs:
        return []
    ivs = sorted(ivs)
    out = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def overlap_bp(a_start, a_end, b_start, b_end):
    """1-based inclusive overlap length."""
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def read_probe_bed(path):
    """probe_id -> (chrom, start, end, strand); NA rows skipped."""
    probes = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 6 or f[0] in ("chrom", "NA"):
                continue
            chrom, start, end, name = f[0].strip(), f[1].strip(), f[2].strip(), f[3]
            if not chrom or chrom.upper() in ("NA", "<NA>", "NAN") or not start or not end:
                continue
            probes[name] = (chrom, int(start), int(end), f[5])
    return probes


def norm_chrom(c):
    c = str(c)
    return c[3:] if c.lower().startswith("chr") else c


def build_exon_index(genes, exons):
    """chrom -> list of (start, end, gene_id) over MERGED exons."""
    idx = {}
    for gid, ivs in exons.items():
        g = genes.get(gid)
        if not g:
            continue
        c = norm_chrom(g["chrom"])
        for s, e in merge_intervals(ivs):
            idx.setdefault(c, []).append((s, e, gid))
    for c in idx:
        idx[c].sort()
    return idx


def candidates_for(probe, idx, genes):
    """gene_ids whose merged exons overlap the probe by > MIN_EXON_OVERLAP."""
    chrom, pstart, pend, _ = probe
    c = norm_chrom(chrom)
    best = {}
    for s, e, gid in idx.get(c, []):
        if s > pend:
            break
        ov = overlap_bp(pstart, pend, s, e)
        if ov > MIN_EXON_OVERLAP:
            best[gid] = max(best.get(gid, 0), ov)
    return best


def filter_duplicates(cand, genes):
    """Kennedy: only applied to resolve duplicates; never empties the set."""
    if len(cand) <= 1:
        return cand, "none"
    kept = {gid: ov for gid, ov in cand.items()
            if not genes[gid]["readthrough"]
            and not POORLY_CHARACTERISED.search(genes[gid]["name"])}
    if not kept:
        return cand, "filter_emptied_kept_all"
    return kept, ("filtered" if len(kept) < len(cand) else "none")


def union_span(gids, genes):
    chroms = {norm_chrom(genes[g]["chrom"]) for g in gids}
    if len(chroms) != 1:
        return None
    start = min(genes[g]["start"] for g in gids)
    end = max(genes[g]["end"] for g in gids)
    strands = {genes[g]["strand"] for g in gids}
    if len(strands) == 1:
        strand = strands.pop()
    else:
        strand = genes[max(gids, key=lambda g: genes[g]["end"] - genes[g]["start"])]["strand"]
    return chroms.pop(), start, end, strand


HEADER_COLS = ["probe_id", "status", "chrom", "start", "end", "strand",
               "n_genes", "gtf_gene_ids", "gtf_gene_symbols", "gtf_gene_types"]


def derive(gtf_path, probe_bed_path, out_path):
    genes, exons = parse_gtf(gtf_path)
    idx = build_exon_index(genes, exons)
    probes = read_probe_bed(probe_bed_path)

    rows, counts = [], {"RESOLVED": 0, "NO_OVERLAP": 0, "MULTI_CHROM": 0}
    for pid in sorted(probes):
        cand = candidates_for(probes[pid], idx, genes)
        if not cand:
            rows.append([pid, "NO_OVERLAP", "", "", "", "", 0, "", "", ""])
            counts["NO_OVERLAP"] += 1
            continue
        kept, _ = filter_duplicates(cand, genes)
        gids = sorted(kept, key=lambda g: (-kept[g], genes[g]["name"]))
        span = union_span(gids, genes)
        if span is None:
            rows.append([pid, "MULTI_CHROM", "", "", "", "", len(gids),
                         ",".join(gids), ",".join(genes[g]["name"] for g in gids),
                         ",".join(genes[g]["gene_type"] for g in gids)])
            counts["MULTI_CHROM"] += 1
            continue
        chrom, start, end, strand = span
        rows.append([pid, "RESOLVED", chrom, start, end, strand, len(gids),
                     ",".join(gids), ",".join(genes[g]["name"] for g in gids),
                     ",".join(genes[g]["gene_type"] for g in gids)])
        counts["RESOLVED"] += 1

    with open(out_path, "w", newline="") as fh:
        fh.write("#tecpg_probe_gene_model\tv1\n")
        fh.write("#method\texon_overlap_gt_%dbp_union_span\n" % MIN_EXON_OVERLAP)
        fh.write("#gtf\t%s\n" % os.path.abspath(gtf_path))
        fh.write("#probe_bed\t%s\n" % os.path.abspath(probe_bed_path))
        fh.write("#coords\t1-based inclusive\n")
        fh.write("#counts\t%s\n" % " ".join(f"{k}={v}" for k, v in counts.items()))
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(HEADER_COLS)
        w.writerows(rows)
    logger.info("[build_probe_gene_model] wrote %d probes to %s (%s)",
                len(rows), out_path,
                " ".join(f"{k}={v}" for k, v in counts.items()))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--probe-bed", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    derive(a.gtf, a.probe_bed, a.output)


if __name__ == "__main__":
    main()
