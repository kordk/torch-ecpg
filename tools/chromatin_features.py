"""Reference implementation (E0) of tools/chromatin_features.py.

Interval index for chromatin-feature tracks and 1-bp CpG overlap lookup.
No pyranges: per-chromosome sorted, merged half-open intervals with
numpy.searchsorted. All coordinates are 0-based half-open [start, end).
"""
import gzip
import hashlib
import logging
import os
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Chromosome-name aliases. Anything not resolvable to chr1..chr22/chrX/chrY/chrM
# is reported by normalize_chrom as None (excluded, logged by the caller), never
# silently prefixed.
_CANON = {f'chr{i}' for i in range(1, 23)} | {'chrX', 'chrY', 'chrM'}
_ALIAS = {'MT': 'chrM', 'chrMT': 'chrM', '23': 'chrX', '24': 'chrY', 'chr23': 'chrX', 'chr24': 'chrY'}


class BuildMismatch(RuntimeError):
    pass


class ManifestError(RuntimeError):
    pass


def normalize_chrom(name):
    s = str(name).strip()
    if s in _ALIAS:
        s = _ALIAS[s]
    if not s.startswith('chr'):
        s = 'chr' + s
    return s if s in _CANON else None


class IntervalIndex:
    """Sorted, merged, non-overlapping intervals per chromosome."""

    def __init__(self, df):
        # df: columns Chromosome, Start, End (already normalized chromosomes)
        self.by_chrom = {}
        self.n_intervals = 0
        self.total_bp = 0
        if len(df) == 0:
            return
        for chrom, g in df.groupby('Chromosome', sort=True):
            starts = g['Start'].to_numpy(dtype=np.int64)
            ends = g['End'].to_numpy(dtype=np.int64)
            order = np.argsort(starts, kind='stable')
            starts, ends = starts[order], ends[order]
            ms, me = [], []
            cs, ce = starts[0], ends[0]
            for s, e in zip(starts[1:], ends[1:]):
                if s <= ce:          # touching or overlapping -> merge
                    ce = max(ce, e)
                else:
                    ms.append(cs); me.append(ce)
                    cs, ce = s, e
            ms.append(cs); me.append(ce)
            ms = np.asarray(ms, dtype=np.int64); me = np.asarray(me, dtype=np.int64)
            self.by_chrom[chrom] = (ms, me)
            self.n_intervals += len(ms)
            self.total_bp += int((me - ms).sum())

    def overlap(self, chroms, positions):
        """bool array: does [pos, pos+1) overlap any interval."""
        chroms = np.asarray(chroms, dtype=object)
        positions = np.asarray(positions, dtype=np.int64)
        out = np.zeros(len(positions), dtype=bool)
        for chrom in pd.unique(chroms):
            if chrom not in self.by_chrom:
                continue
            ms, me = self.by_chrom[chrom]
            idx = np.flatnonzero(chroms == chrom)
            pos = positions[idx]
            k = np.searchsorted(ms, pos, side='right') - 1
            ok = k >= 0
            hit = np.zeros(len(pos), dtype=bool)
            hit[ok] = pos[ok] < me[k[ok]]
            out[idx] = hit
        return out

    def to_df(self):
        rows = []
        for chrom, (ms, me) in self.by_chrom.items():
            rows.append(pd.DataFrame({'Chromosome': chrom, 'Start': ms, 'End': me}))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['Chromosome', 'Start', 'End'])


def _open(path):
    return gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path, 'r')


def _norm_df(df, label):
    df = df.copy()
    df['Chromosome'] = df['Chromosome'].map(normalize_chrom)
    bad = df['Chromosome'].isna()
    if bad.any():
        logger.info(f'Drop site chromatin_features.{label}[chrom]: excluded {int(bad.sum())} of {len(df)} '
                    f'intervals on unrecognised chromosomes')
        df = df[~bad]
    df['Start'] = df['Start'].astype(np.int64)
    df['End'] = df['End'].astype(np.int64)
    if (df['End'] <= df['Start']).any():
        raise ManifestError(f'{label}: non-positive interval length')
    return df[['Chromosome', 'Start', 'End']]


def load_bed(path, name_filter=None, label='bed'):
    df = pd.read_csv(path, sep='\t', header=None, comment='#', usecols=[0, 1, 2] + ([3] if name_filter else []),
                     names=['Chromosome', 'Start', 'End'] + (['Name'] if name_filter else []), dtype={0: str})
    if name_filter:
        keep = set(name_filter)
        df = df[df['Name'].astype(str).isin(keep)]
        if len(df) == 0:
            raise ManifestError(f'{label}: name filter {sorted(keep)} matched no rows in {path}')
    return _norm_df(df, label)


def load_gtf_genes(path, gene_types, label='gtf'):
    """Gene records whose gene_type is in gene_types. Minimal parser, no pyranges."""
    keep = set(gene_types)
    rows = []
    with _open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 9 or f[2] != 'gene':
                continue
            attrs = f[8]
            gt = None
            for tok in attrs.split(';'):
                tok = tok.strip()
                if tok.startswith('gene_type '):
                    gt = tok.split(' ', 1)[1].strip('"')
                    break
            if gt in keep:
                rows.append((f[0], int(f[3]) - 1, int(f[4])))   # GTF 1-based closed -> half-open
    if not rows:
        raise ManifestError(f'{label}: gene_type filter {sorted(keep)} matched no gene records in {path}')
    return _norm_df(pd.DataFrame(rows, columns=['Chromosome', 'Start', 'End']), label)


def flank(index, width):
    """Regions extending `width` bp outward on each side of every interval in
    `index`, excluding the source intervals themselves (shore from island,
    shelf from shore). Clamped at 0."""
    src = index.to_df()
    left = src.assign(End=src['Start'], Start=(src['Start'] - width).clip(lower=0))
    right = src.assign(Start=src['End'], End=src['End'] + width)
    cand = IntervalIndex(pd.concat([left, right], ignore_index=True))
    return subtract(cand, index)


def subtract(a, b):
    """Intervals of a not covered by b."""
    rows = []
    for chrom, (as_, ae) in a.by_chrom.items():
        if chrom not in b.by_chrom:
            for s, e in zip(as_, ae):
                rows.append((chrom, s, e))
            continue
        bs, be = b.by_chrom[chrom]
        for s, e in zip(as_, ae):
            cur = s
            j = np.searchsorted(bs, s, side='right') - 1
            j = max(j, 0)
            while j < len(bs) and bs[j] < e:
                if be[j] > cur and bs[j] < e:
                    if bs[j] > cur:
                        rows.append((chrom, cur, min(bs[j], e)))
                    cur = max(cur, be[j])
                j += 1
            if cur < e:
                rows.append((chrom, cur, e))
    df = pd.DataFrame(rows, columns=['Chromosome', 'Start', 'End'])
    df = df[df['End'] > df['Start']]
    return IntervalIndex(df)


def union(indexes):
    return IntervalIndex(pd.concat([i.to_df() for i in indexes], ignore_index=True))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_tracks(manifest_path, genome_build, env=None):
    """Parse the feature-track manifest and return an ordered dict
    name -> IntervalIndex. Rows are resolved in file order; `derived` and
    `union` rows reference earlier names. Raises BuildMismatch on the first
    row whose build != genome_build (checked for ALL rows before any load)."""
    env = env if env is not None else os.environ
    man = pd.read_csv(manifest_path, sep='\t', comment='#', dtype=str).fillna('.')
    required = ['name', 'kind', 'path', 'build', 'select']
    missing = [c for c in required if c not in man.columns]
    if missing:
        raise ManifestError(f'manifest missing columns {missing}')
    for _, r in man.iterrows():
        if r['build'] != genome_build:
            raise BuildMismatch(f"track '{r['name']}' is build {r['build']}; catalog is {genome_build}")
    tracks = {}
    provenance = {}
    for _, r in man.iterrows():
        name, kind, path, sel = r['name'], r['kind'], os.path.expandvars(r['path']), r['select']
        if name in tracks:
            raise ManifestError(f'duplicate track name {name}')
        if kind == 'bed':
            tracks[name] = IntervalIndex(load_bed(path, label=name))
            provenance[name] = sha256_file(path)
        elif kind == 'bed4':
            if not sel.startswith('name='):
                raise ManifestError(f'{name}: bed4 requires select=name=<a,b,...>')
            tracks[name] = IntervalIndex(load_bed(path, name_filter=sel[5:].split(','), label=name))
            provenance[name] = sha256_file(path)
        elif kind == 'gtf':
            if not sel.startswith('gene_type='):
                raise ManifestError(f'{name}: gtf requires select=gene_type=<a,b,...>')
            tracks[name] = IntervalIndex(load_gtf_genes(path, sel[10:].split(','), label=name))
            provenance[name] = sha256_file(path)
        elif kind == 'derived':
            # select = "flank=<bp>" or "flank=<bp>;exclude=<t1,t2,...>". The
            # source track is always excluded; `exclude` names further tracks
            # to subtract (shelf = flank of shore, excluding shore AND island).
            if path not in tracks:
                raise ManifestError(f'{name}: derived from unknown track {path}')
            opts = dict(tok.split('=', 1) for tok in sel.split(';') if '=' in tok)
            if 'flank' not in opts:
                raise ManifestError(f'{name}: derived requires select=flank=<bp>[;exclude=<a,b>]')
            excl = [t for t in opts.get('exclude', '').split(',') if t]
            unknown = [t for t in excl if t not in tracks]
            if unknown:
                raise ManifestError(f'{name}: exclude names unknown tracks {unknown}')
            idx = flank(tracks[path], int(opts['flank']))
            for t in excl:
                idx = subtract(idx, tracks[t])
            tracks[name] = idx
            provenance[name] = f'derived:{path}:{sel}'
        elif kind == 'union':
            parts = path.split(',')
            unknown = [p for p in parts if p not in tracks]
            if unknown:
                raise ManifestError(f'{name}: union of unknown tracks {unknown}')
            tracks[name] = union([tracks[p] for p in parts])
            provenance[name] = f'union:{path}'
        else:
            raise ManifestError(f'{name}: unknown kind {kind}')
        logger.info(f'track {name}: {tracks[name].n_intervals} merged intervals, {tracks[name].total_bp} bp')
    return tracks, provenance
