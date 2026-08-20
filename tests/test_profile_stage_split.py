"""Per-chunk profile stages must each be measured, and the D2H hoist that
makes them measurable must not change the saved results.

Before this guard, prof_t3/prof_t4/prof_t5 were initialised at the top of the
gene-chunk loop and never re-stamped, so compute_ms, d2h_ms and post_ms were
identically 0.0 and write_enqueue_ms spanned the whole chunk.
"""
import io
import os
import re
import sys

import numpy as np
import pandas as pd

from tecpg.logger import Logger
from tecpg.processing import tecpg_mlr_qr

_RNG = np.random.default_rng(7)
_S, _G, _M, _K = 30, 15, 400, 3
_M_DATA = pd.DataFrame(
    _RNG.random((_M, _S)),
    index=[f'cg{i:06d}' for i in range(_M)],
    columns=[f's{i}' for i in range(_S)],
)
_G_DATA = pd.DataFrame(
    _RNG.random((_G, _S)),
    index=[f'ILMN_{i:06d}' for i in range(_G)],
    columns=[f's{i}' for i in range(_S)],
)
_C_DATA = pd.DataFrame(
    _RNG.random((_S, _K - 2)),
    index=[f's{i}' for i in range(_S)],
    columns=[str(i) for i in range(_K - 2)],
)

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
_SUMMARY_RE = re.compile(r'(?P<metric>[a-z_0-9]+): sum=(?P<sum>[-0-9.]+)ms')


def _run_chunked(out_dir, p_thresh):
    """Runs a chunked CPU map and returns the captured stdout."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        tecpg_mlr_qr(
            _M_DATA, _G_DATA, _C_DATA,
            region='all',
            methylation_only=True,
            p_thresh=p_thresh,
            p_only=False,
            logit_transform=False,
            output_dir=str(out_dir),
            gene_loci_per_chunk=5,
            meth_loci_per_chunk=25,
            seed=42,
            logger=Logger(),
        )
        sys.stdout.seek(0)
        return sys.stdout.read()
    finally:
        sys.stdout = old_stdout


def _summary_sums(output):
    sums = {}
    for line in output.splitlines():
        match = _SUMMARY_RE.search(_ANSI_RE.sub('', line))
        if match:
            sums[match.group('metric')] = float(match.group('sum'))
    return sums


def test_profile_stages_are_measured_in_chunked_run(tmp_path):
    out = tmp_path / 'chunked'
    out.mkdir()
    sums = _summary_sums(_run_chunked(out, p_thresh=None))

    for metric in ('prep_ms', 'h2d_ms', 'compute_ms', 'd2h_ms',
                   'post_ms', 'write_enqueue_ms'):
        assert metric in sums, f'{metric} missing from END OF RUN SUMMARY'

    # The three stages that were never stamped before this guard. Each has
    # real work behind it (solve + threshold gather, .cpu() copies, index
    # construction), so a strictly positive total across the 48 chunks is the
    # minimum evidence that the stamp exists and is placed after that work.
    # The summary prints sums at 0.1 ms resolution; 48 chunks keep each sum
    # comfortably above that floor even on the CPU backend.
    assert sums['compute_ms'] > 0.0
    assert sums['d2h_ms'] > 0.0
    assert sums['post_ms'] > 0.0


def test_chunked_output_matches_unchunked_after_d2h_hoist(tmp_path):
    out = tmp_path / 'chunked_p'
    out.mkdir()
    _run_chunked(out, p_thresh=0.5)

    parts = sorted(f for f in os.listdir(out) if f.endswith('.csv'))
    assert parts, 'chunked run wrote no part files'
    chunked = pd.concat(
        [pd.read_csv(os.path.join(out, f), index_col=[0, 1]) for f in parts]
    ).sort_index()

    unchunked = tecpg_mlr_qr(
        _M_DATA, _G_DATA, _C_DATA,
        region='all',
        methylation_only=True,
        p_thresh=0.5,
        p_only=False,
        logit_transform=False,
        seed=42,
        logger=Logger(),
    ).sort_index()

    # Index (gt_id, mt_id) and values must agree pair for pair: the hoisted
    # results_host / p_mask_np must still be aligned with the repeat/tile
    # index construction they were moved ahead of.
    assert list(chunked.index) == list(unchunked.index)
    assert list(chunked.columns) == list(unchunked.columns)
    np.testing.assert_allclose(
        chunked.to_numpy(), unchunked.to_numpy(), rtol=1e-5, atol=1e-6
    )
