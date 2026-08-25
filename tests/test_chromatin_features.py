"""Tests for tools/chromatin_features.py (Kennedy Fig. 6 support, chunk E1).

Oracle tests: every overlap and every derived interval set is checked against
a brute-force pandas computation on the raw, unmerged intervals. Coordinates
are 0-based half-open throughout.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
import chromatin_features as cf  # noqa: E402


# ------------------------------------------------------------------ helpers
def make_bed(path, rows):
    pd.DataFrame(rows).to_csv(path, sep='\t', header=False, index=False)


def brute_overlap(chrom, pos, iv):
    """iv: DataFrame Chromosome/Start/End (raw, unmerged)."""
    out = np.zeros(len(pos), dtype=bool)
    for i in range(len(pos)):
        sub = iv[iv['Chromosome'] == chrom[i]]
        out[i] = bool(((sub['Start'] <= pos[i]) & (pos[i] < sub['End'])).any())
    return out


@pytest.fixture
def tracks_fixture(tmp_path):
    """One CGI BED, one 15-state-like bed4 track, and a manifest deriving
    shore/shelf and selecting four states plus a union. Seeded locally so the
    fixture is order-independent."""
    rng = np.random.default_rng(20180619)
    isl = tmp_path / 'cgi.bed'
    make_bed(isl, [dict(c='chr1', s=10_000, e=12_000), dict(c='chr1', s=14_500, e=15_000),
                   dict(c='chr2', s=50_000, e=60_000)])
    hmm = tmp_path / 'hmm.bed'
    hmm_rows = []
    for c in ('chr1', 'chr2'):
        edges = np.sort(rng.choice(np.arange(0, 100_001, 500), 40, replace=False))
        for s, e in zip(edges[:-1], edges[1:]):
            hmm_rows.append(dict(c=c, s=int(s), e=int(e), n=f'{rng.integers(1, 16)}_State'))
    make_bed(hmm, hmm_rows)
    names = sorted({r['n'] for r in hmm_rows}, key=lambda x: int(x.split('_')[0]))
    man = tmp_path / 'tracks.tsv'
    lines = ['name\tkind\tpath\tbuild\tselect',
             f'CGI\tbed\t{isl}\thg19\t.',
             'CGI_shore\tderived\tCGI\thg19\tflank=1500',
             'CGI_shelf\tderived\tCGI_shore\thg19\tflank=1500;exclude=CGI']
    for nm in names[:4]:
        lines.append(f'HMM_{nm.split("_")[0]}\tbed4\t{hmm}\thg19\tname={nm}')
    lines.append(f'HMM_union\tunion\tHMM_{names[0].split("_")[0]},HMM_{names[1].split("_")[0]}\thg19\t.')
    man.write_text('\n'.join(lines) + '\n')
    chrom = np.where(np.arange(200) < 120, 'chr1', 'chr2').astype(object)
    pos = rng.integers(0, 100_000, 200)
    return dict(tmp=tmp_path, man=man, isl=isl, hmm=hmm, hmm_rows=hmm_rows, names=names, chrom=chrom, pos=pos)


# -------------------------------------------------------------------- T1
def test_T1_shore_shelf_geometry():
    isl = cf.IntervalIndex(pd.DataFrame({'Chromosome': ['chr1', 'chr1'], 'Start': [1000, 9000], 'End': [2000, 9500]}))
    shore = cf.flank(isl, 1500)
    shelf = cf.subtract(cf.flank(shore, 1500), isl)
    s = shore.to_df().sort_values('Start').to_records(index=False).tolist()
    assert s == [('chr1', 0, 1000), ('chr1', 2000, 3500), ('chr1', 7500, 9000), ('chr1', 9500, 11000)]
    sh = shelf.to_df().sort_values('Start').to_records(index=False).tolist()
    assert sh == [('chr1', 3500, 5000), ('chr1', 6000, 7500), ('chr1', 11000, 12500)]
    # mutual exclusivity of island / shore / shelf at every bp
    pos = np.arange(0, 13000)
    ch = np.array(['chr1'] * len(pos), dtype=object)
    cover = isl.overlap(ch, pos).astype(int) + shore.overlap(ch, pos).astype(int) + shelf.overlap(ch, pos).astype(int)
    assert cover.max() == 1
    # two islands 2 kb apart: overlapping shores merge without double counting
    isl2 = cf.IntervalIndex(pd.DataFrame({'Chromosome': ['chr1', 'chr1'], 'Start': [1000, 4000], 'End': [2000, 5000]}))
    shore2 = cf.flank(isl2, 1500)
    assert shore2.to_df().sort_values('Start').to_records(index=False).tolist() == [('chr1', 0, 1000), ('chr1', 2000, 4000), ('chr1', 5000, 6500)]


def test_T1b_manifest_shelf_excludes_island(tracks_fixture):
    tracks, _ = cf.load_tracks(tracks_fixture['man'], 'hg19')
    pos = np.arange(0, 100_000)
    ch = np.array(['chr1'] * len(pos), dtype=object)
    cover = sum(tracks[k].overlap(ch, pos).astype(int) for k in ('CGI', 'CGI_shore', 'CGI_shelf'))
    assert cover.max() == 1
    # a position inside the first island is not shelf
    assert not tracks['CGI_shelf'].overlap(np.array(['chr1'], dtype=object), np.array([11_000]))[0]


# -------------------------------------------------------------------- T2
def test_T2_overlap_matches_brute_force():
    rng = np.random.default_rng(42)
    iv = pd.DataFrame({'Chromosome': rng.choice(['chr1', 'chr2'], 50),
                       'Start': rng.integers(0, 10_000, 50)})
    iv['End'] = iv['Start'] + rng.integers(1, 800, 50)
    idx = cf.IntervalIndex(iv)
    chrom = rng.choice(['chr1', 'chr2', 'chr3'], 500).astype(object)
    pos = rng.integers(0, 11_000, 500)
    assert np.array_equal(idx.overlap(chrom, pos), brute_overlap(chrom, pos, iv))
    # boundary semantics: start inclusive, end exclusive
    one = cf.IntervalIndex(pd.DataFrame({'Chromosome': ['chr1'], 'Start': [100], 'End': [200]}))
    assert one.overlap(np.array(['chr1'] * 4, dtype=object), np.array([99, 100, 199, 200])).tolist() == [False, True, True, False]


# -------------------------------------------------------------------- T3
def test_T3_chromosome_normalisation():
    assert cf.normalize_chrom('1') == 'chr1' == cf.normalize_chrom('chr1')
    assert cf.normalize_chrom('MT') == 'chrM' and cf.normalize_chrom('23') == 'chrX'
    assert cf.normalize_chrom('chrUn_gl000220') is None
    assert cf.normalize_chrom('chr1_random') is None


# -------------------------------------------------------------------- T4
def test_T4_build_guard(tracks_fixture, tmp_path):
    man = tracks_fixture['man'].read_text().replace('\thg19\t', '\thg38\t', 1)  # first data row (CGI) becomes hg38
    bad = tmp_path / 'bad.tsv'
    bad.write_text(man)
    with pytest.raises(cf.BuildMismatch) as ei:
        cf.load_tracks(bad, 'hg19')
    assert "'CGI'" in str(ei.value) and 'hg38' in str(ei.value)


# -------------------------------------------------------------------- T5
def test_T5_union_and_bed4_select(tracks_fixture):
    tracks, _ = cf.load_tracks(tracks_fixture['man'], 'hg19')
    hr = pd.DataFrame(tracks_fixture['hmm_rows']).rename(columns={'c': 'Chromosome', 's': 'Start', 'e': 'End', 'n': 'Name'})
    n0, n1 = tracks_fixture['names'][0], tracks_fixture['names'][1]
    k0 = f'HMM_{n0.split("_")[0]}'
    raw_union = hr[hr['Name'].isin([n0, n1])]
    chrom = tracks_fixture['chrom']; pos = tracks_fixture['pos']
    assert np.array_equal(tracks['HMM_union'].overlap(chrom, pos), brute_overlap(chrom, pos, raw_union))
    assert np.array_equal(tracks[k0].overlap(chrom, pos), brute_overlap(chrom, pos, hr[hr['Name'] == n0]))
    # bed4 filter that matches nothing is a manifest error
    man2 = tracks_fixture['tmp'] / 'm2.tsv'
    man2.write_text('name\tkind\tpath\tbuild\tselect\nX\tbed4\t' + str(tracks_fixture['hmm']) + '\thg19\tname=99_Nothing\n')
    with pytest.raises(cf.ManifestError):
        cf.load_tracks(man2, 'hg19')


def test_T5b_gtf_gene_type_select(tmp_path):
    gtf = tmp_path / 'genes.gtf'
    gtf.write_text('\n'.join([
        '##description: hand-written fixture',
        'chr1\tHAVANA\tgene\t1001\t2000\t.\t+\t.\tgene_id "ENSG1.1"; gene_type "lincRNA"; gene_name "L1";',
        'chr1\tHAVANA\texon\t1001\t1200\t.\t+\t.\tgene_id "ENSG1.1"; gene_type "lincRNA"; gene_name "L1";',
        'chr1\tHAVANA\tgene\t5001\t5100\t.\t-\t.\tgene_id "ENSG2.1"; gene_type "miRNA"; gene_name "M1";',
        'chr2\tHAVANA\tgene\t7001\t8000\t.\t+\t.\tgene_id "ENSG3.1"; gene_type "protein_coding"; gene_name "P1";',
        'chr2\tHAVANA\tgene\t9001\t9050\t.\t+\t.\tgene_id "ENSG4.1"; gene_type "snoRNA"; gene_name "S1";',
    ]) + '\n')
    df = cf.load_gtf_genes(gtf, ['snoRNA', 'miRNA'])
    # GTF 1-based closed -> 0-based half-open
    assert df.sort_values(['Chromosome', 'Start']).to_records(index=False).tolist() == [('chr1', 5000, 5100), ('chr2', 9000, 9050)]
    idx = cf.IntervalIndex(df)
    ch = np.array(['chr1', 'chr1', 'chr1', 'chr2'], dtype=object)
    assert idx.overlap(ch, np.array([4999, 5000, 5099, 5100])).tolist() == [False, True, True, False]
    with pytest.raises(cf.ManifestError):
        cf.load_gtf_genes(gtf, ['tRNA'])
    # manifest kind=gtf with select=gene_type=
    man = tmp_path / 'm.tsv'
    man.write_text(f'name\tkind\tpath\tbuild\tselect\nsno_miRNA\tgtf\t{gtf}\thg19\tgene_type=snoRNA,miRNA\n')
    tracks, prov = cf.load_tracks(man, 'hg19')
    assert tracks['sno_miRNA'].n_intervals == 2 and len(prov['sno_miRNA']) == 64
