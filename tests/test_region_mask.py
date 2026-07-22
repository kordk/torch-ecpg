import torch
from tecpg.helper import compute_region_mask


def _fixture():
    m_chrom = torch.tensor([1, 1, 1, 2], dtype=torch.int8)
    g_chrom = torch.tensor([1, 1, 1, 1], dtype=torch.int8)
    m_pos = torch.tensor([900_000, 900_000, 1_000_000, 900_000], dtype=torch.int32)
    g_pos = torch.tensor([1_000_000, 1_000_000, 3_000_000, 1_000_000], dtype=torch.int32)
    g_strand = torch.tensor([1, -1, 1, 1], dtype=torch.int8)
    return m_chrom, m_pos, g_chrom, g_pos, g_strand


def test_cis_window_no_int8_overflow():
    m_chrom, m_pos, g_chrom, g_pos, g_strand = _fixture()
    mask = compute_region_mask(
        'cis', m_chrom, m_pos, g_chrom, g_pos, g_strand,
        window_base=0, upstream=1_000_000, downstream=1_000_000,
    )
    # A +strand CpG 100 kb from its gene is inside a +/-1 Mb window.
    # Under the int8 overflow the effective window was ~+/-64 bp, so this was False.
    assert bool(mask[0]) is True
    assert bool(mask[2]) is False   # 2 Mb away -> outside the window
    assert bool(mask[3]) is False   # different chromosome -> never cis


def test_trans_mask_unchanged():
    m_chrom, m_pos, g_chrom, g_pos, g_strand = _fixture()
    mask = compute_region_mask(
        'trans', m_chrom, m_pos, g_chrom, g_pos, g_strand,
        window_base=0, upstream=1_000_000, downstream=1_000_000,
    )
    assert [bool(x) for x in mask] == [False, False, False, True]
