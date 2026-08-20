"""Survivor-indexed (gt_id, mt_id) construction must reproduce the
unchunked path's index pair-for-pair across the filter matrix.

The chunked save path derives index labels from the survivors' flat grid
indices (gene = f // mt_count, meth = f % mt_count) instead of building
full (chunk_len * mt_count) string arrays and masking them down. The
unchunked path (gene_loci_per_chunk=None) keeps the independent
repeat/tile construction and is the oracle here.
"""
import os
import sys

import numpy as np
import pandas as pd

from tecpg.logger import Logger
from tecpg.processing import tecpg_mlr_qr
from tecpg.test_data import generate_data

_S, _M, _G = 40, 300, 24
_MD, _GD, _CD = generate_data(_S, _M, _G, seed=11)

# Annotations in the BED-like layout tecpg_mlr_qr expects (it drops
# chromEnd/score, and strand from M). Chromosomes are drawn from a small
# set so 'trans' and 'cis' each keep a nontrivial, nonuniform subset of
# the grid.
_RNG = np.random.default_rng(11)
def _bed(index, rng, strand_choices):
    start = rng.integers(1, 2_000_000, len(index))
    return pd.DataFrame(
        {
            'chrom': rng.integers(1, 4, len(index)),
            'chromStart': start,
            'chromEnd': start + 50,
            'score': np.zeros(len(index), dtype=int),
            'strand': rng.choice(strand_choices, len(index)),
        },
        index=index,
    )


_MA = _bed(_MD.index, _RNG, [1])
_GA = _bed(_GD.index, _RNG, [-1, 1])


def _run(out_dir=None, *, region='all', p_thresh=None, chunked=True):
    kwargs = dict(
        region=region,
        window_base=1_000_000,
        upstream=1_000_000,
        downstream=1_000_000,
        methylation_only=True,
        p_thresh=p_thresh,
        p_only=False,
        logit_transform=False,
        seed=42,
        logger=Logger(),
    )
    annots = (_MA, _GA) if region != 'all' else (None, None)
    if chunked:
        kwargs.update(
            output_dir=str(out_dir),
            gene_loci_per_chunk=7,     # 24 genes -> 4 uneven gene chunks
            meth_loci_per_chunk=100,   # 300 loci -> 3 meth chunks
        )
    return tecpg_mlr_qr(_MD, _GD, _CD, annots[0], annots[1], **kwargs)


def _read_parts(out_dir):
    parts = sorted(f for f in os.listdir(out_dir) if f.endswith('.csv'))
    frames = [
        pd.read_csv(os.path.join(out_dir, f), index_col=[0, 1])
        for f in parts
    ]
    frames = [f for f in frames if len(f)]
    if not frames:
        return None
    return pd.concat(frames).sort_index()


def _assert_matches_oracle(tmp_path, *, region, p_thresh, expect_empty=False):
    out = tmp_path / 'chunked'
    out.mkdir()
    _run(out, region=region, p_thresh=p_thresh, chunked=True)
    chunked = _read_parts(out)
    oracle = _run(region=region, p_thresh=p_thresh, chunked=False)
    oracle = oracle.sort_index() if oracle is not None and len(oracle) else None

    if expect_empty:
        assert chunked is None and (oracle is None or len(oracle) == 0), (
            'expected zero survivors on both paths'
        )
        return
    assert chunked is not None and oracle is not None, (
        'one path produced no rows'
    )
    assert list(chunked.index) == list(oracle.index)
    assert list(chunked.columns) == list(oracle.columns)
    np.testing.assert_allclose(
        chunked.to_numpy(), oracle.to_numpy(), rtol=1e-5, atol=1e-6
    )


def test_all_region_with_p_thresh(tmp_path):
    _assert_matches_oracle(tmp_path, region='all', p_thresh=0.4)


def test_trans_region_without_p_thresh(tmp_path):
    _assert_matches_oracle(tmp_path, region='trans', p_thresh=None)


def test_trans_region_with_p_thresh(tmp_path):
    _assert_matches_oracle(tmp_path, region='trans', p_thresh=0.4)


def test_cis_region_with_p_thresh(tmp_path):
    _assert_matches_oracle(tmp_path, region='cis', p_thresh=0.6)


def test_zero_survivors_do_not_crash(tmp_path):
    _assert_matches_oracle(
        tmp_path, region='all', p_thresh=1e-300, expect_empty=True
    )


def test_unfiltered_full_grid_path_unchanged(tmp_path):
    # region='all' and p_thresh=None: every pair is emitted; the chunked
    # path keeps the repeat/tile construction and must still match.
    _assert_matches_oracle(tmp_path, region='all', p_thresh=None)
