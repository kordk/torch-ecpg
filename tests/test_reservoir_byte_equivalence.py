"""Reservoir sampling output must be exactly identical to the frozen
golden fixtures in tests/data/reservoir_golden/.

The goldens were generated from the pre-refactor sequential implementation
(the per-item placement loop), seed-for-seed, with the device forced to
CPU so the reservoir RNG stream (torch CPU generator) is identical on
every machine regardless of CUDA availability. The sample-then-index
refactor must reproduce them to the byte: the reservoir feeds qr_permute
null calibration, so sampling semantics are load-bearing and the
acceptance bar is identity, not tolerance.

Cases cover the algorithm's regimes: pure fill (buffer never overflows),
the fill-to-replace boundary crossing inside one chunk, steady-state
replacement over many chunks, region-masked input, and a degenerate tiny
buffer where duplicate replacement targets are guaranteed (exercising
last-write-wins).
"""
import os

import numpy as np
import pandas as pd
import pytest

from tecpg.logger import Logger
from tecpg.processing import tecpg_mlr_qr

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), 'data',
                          'reservoir_golden')

CASES = {
    'fill_only': dict(M=60, G=8, mchunk=30, gchunk=4, count=10000,
                      region='all'),
    'boundary': dict(M=60, G=8, mchunk=30, gchunk=4, count=150,
                     region='all'),
    'steady': dict(M=200, G=12, mchunk=50, gchunk=4, count=90,
                   region='all'),
    'region_tr': dict(M=200, G=12, mchunk=50, gchunk=4, count=90,
                      region='trans'),
    'tiny_count': dict(M=200, G=12, mchunk=50, gchunk=4, count=7,
                       region='all'),
}


def _annotations(m_index, g_index):
    rng = np.random.default_rng(3)

    def bed(index, strands):
        start = rng.integers(1, 2_000_000, len(index))
        return pd.DataFrame(
            {
                'chrom': rng.integers(1, 4, len(index)),
                'chromStart': start,
                'chromEnd': start + 50,
                'score': np.zeros(len(index), dtype=int),
                'strand': rng.choice(strands, len(index)),
            },
            index=index,
        )

    return bed(m_index, [1]), bed(g_index, [-1, 1])


def run_reservoir_case(case, out_dir):
    """Runs one chunked map with the reservoir on; returns the sample path.

    The logger forces use_cpu so the reservoir RNG (device generator)
    draws the identical stream on CPU-only and CUDA machines.
    """
    cfg = CASES[case]
    rng = np.random.default_rng(7)
    S = 24
    people = [f's{i}' for i in range(S)]
    Md = pd.DataFrame(rng.random((cfg['M'], S)),
                      index=[f'cg{i}' for i in range(cfg['M'])],
                      columns=people)
    Gd = pd.DataFrame(rng.random((cfg['G'], S)),
                      index=[f'ILMN_{i}' for i in range(cfg['G'])],
                      columns=people)
    Cd = pd.DataFrame({'age': rng.integers(20, 60, S)}, index=people)
    kwargs = {}
    if cfg['region'] != 'all':
        MA, GA = _annotations(Md.index, Gd.index)
        kwargs = dict(window_base=1_000_000, upstream=1_000_000,
                      downstream=1_000_000)
    else:
        MA = GA = None
    os.makedirs(out_dir, exist_ok=True)
    tecpg_mlr_qr(
        Md, Gd, Cd, MA, GA,
        region=cfg['region'],
        methylation_only=True,
        p_thresh=0.5,
        p_only=False,
        logit_transform=False,
        output_dir=str(out_dir),
        output_format='parquet',
        gene_loci_per_chunk=cfg['gchunk'],
        meth_loci_per_chunk=cfg['mchunk'],
        seed=42,
        reservoir_count=cfg['count'],
        logger=Logger(carry_data={'use_cpu': True}),
        **kwargs,
    )
    return os.path.join(str(out_dir), 'sample_reservoir.csv')


@pytest.mark.parametrize('case', sorted(CASES))
def test_reservoir_output_matches_golden(case, tmp_path):
    sample = run_reservoir_case(case, tmp_path / case)
    assert os.path.exists(sample), f'{case}: no reservoir sample written'
    golden = os.path.join(GOLDEN_DIR, f'{case}.csv')
    assert os.path.exists(golden), (
        f'{golden} missing — generate goldens from the PRE-refactor code '
        f'(see the P1b prompt pre-flight) before running this test'
    )
    # Parsed-exact comparison rather than raw bytes: the identical RNG
    # stream and identical computation make the underlying values
    # bit-identical, but CSV float *formatting* can differ across pandas
    # versions; parsing both sides through the same reader compares the
    # values themselves, exactly (check_exact — no tolerance).
    new = pd.read_csv(sample, index_col=[0, 1])
    old = pd.read_csv(golden, index_col=[0, 1])
    pd.testing.assert_frame_equal(new, old, check_exact=True), (
        f'{case}: sampling semantics changed — do not regenerate the '
        f'golden to make this pass'
    )
    assert list(new.index) == list(old.index), f'{case}: row order differs'
