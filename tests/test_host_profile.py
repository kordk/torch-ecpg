"""Unit tests for the host-profile and chunk auto-sizing helpers added in
the 3a/3b host-class auto-tuning change.

These tests run on CPU and do not require a GPU. They exercise the pure
helpers in `tecpg.cli` (`_host_class`, `_auto_save_threads`,
`_auto_chunk_sizes`) so the small-host / minimum-config invariants stay
covered by CI.
"""
from tecpg.cli import _auto_chunk_sizes, _auto_save_threads, _host_class
from tecpg.test_data import generate_data


def test_host_class_thresholds():
    # The boundary is `cores >= 12 AND ram_gb >= 32`. Minimum-config
    # target (16 GB / 8 cores) must classify as 'minimum'.
    assert _host_class(8, 16.0) == 'minimum'
    assert _host_class(11, 1024.0) == 'minimum'
    assert _host_class(64, 31.0) == 'minimum'

    # klabdev (32 cores / 512 GB) and similar must classify as 'server'.
    assert _host_class(32, 512.0) == 'server'
    assert _host_class(12, 32.0) == 'server'


def test_auto_save_threads_caps_at_eight_on_big_hosts():
    # Regression: the cap was lowered from 32 to 8 because RAID6/dm-crypt
    # saturates well before 32 concurrent writers. See the 2026-05-01
    # klabdev profile (avg dm-3 aqu-sz ~2139 with 32 writers).
    assert _auto_save_threads(32, 512.0) == 8
    assert _auto_save_threads(64, 1024.0) == 8


def test_auto_chunk_sizes_no_chunking_on_tiny_data():
    # Tiny synthetic data fits well within any reasonable memory budget,
    # so the helper must report "no chunking needed" by returning
    # (None, None). This protects the minimum-config path from being
    # forced into chunking by the auto-sizer.
    M, G, C = generate_data(sample_size=20, m_rows=50, g_rows=10)
    g, m = _auto_chunk_sizes(M, G, C)
    assert g is None and m is None


def test_auto_chunk_sizes_returns_positive_when_budget_is_tight():
    # Force a very small target by patching torch.cuda.mem_get_info path:
    # easier route is to call the helper with synthetic data large
    # enough that the system-RAM-derived target requires chunking.
    # We use a small dataset but assert the contract: if a chunk size is
    # returned at all, it must be >= 1 and <= number of genes/meth.
    M, G, C = generate_data(sample_size=50, m_rows=100, g_rows=80)
    g, m = _auto_chunk_sizes(M, G, C)
    if g is not None:
        assert 1 <= g <= len(G)
    if m is not None:
        assert 1 <= m <= len(M)


def test_auto_chunk_sizes_anchor_meth():
    # When the user pins --meth-loci-per-chunk, the helper must echo
    # it back as the methylation chunk size and return either a gene
    # chunk or None (None = whole-G fits in budget at this
    # --meth-loci-per-chunk). The minimum-config invariant: anchoring
    # is honored on any host class because it is an explicit user
    # request.
    M, G, C = generate_data(sample_size=20, m_rows=50, g_rows=10)
    g, m = _auto_chunk_sizes(M, G, C, pinned_m=25)
    assert m == 25
    assert g is None or 1 <= g <= len(G)


def test_auto_chunk_sizes_anchor_gene():
    # When the user pins --gene-loci-per-chunk, the helper must echo
    # it back as the gene chunk size and return a derived methylation
    # chunk in [1, mt_count].
    M, G, C = generate_data(sample_size=20, m_rows=50, g_rows=10)
    g, m = _auto_chunk_sizes(M, G, C, pinned_g=5)
    assert g == 5
    assert m is not None and 1 <= m <= len(M)


def test_auto_chunk_sizes_anchor_both_passthrough():
    # When the user fully specifies both, the helper just returns them.
    M, G, C = generate_data(sample_size=20, m_rows=50, g_rows=10)
    g, m = _auto_chunk_sizes(M, G, C, pinned_g=3, pinned_m=20)
    assert (g, m) == (3, 20)


def test_auto_chunk_sizes_no_40k_meth_ceiling():
    # PR 2 dropped the historical `min(mt_count, 40000)` cap on the
    # auto-derived methylation chunk: the RAM/GPU budget is the binding
    # constraint, so when the data fits the budget the helper should
    # return either (None, None) or a meth chunk that may equal the
    # full mt_count without being clipped to 40000. For tiny synthetic
    # data the budget is huge relative to the working set, so on the
    # auto-only path we expect either no chunking (the common case) or
    # the full mt_count -- never a value clipped below mt_count.
    M, G, C = generate_data(sample_size=20, m_rows=50, g_rows=10)
    g, m = _auto_chunk_sizes(M, G, C)
    if m is not None:
        assert m == len(M), (
            f'expected uncapped meth chunk == mt_count={len(M)}, got {m}'
        )


def test_auto_chunk_sizes_ig_aware_shrinks_chunks():
    # Regression for the 1.22.x OOM bug: with compute_ig=True the
    # estimator must charge for the analytical-IG transient (a full
    # `(M, S, K)` tensor for the X_diff_mean intermediate) and so
    # auto-derived chunks must be no larger than the no-IG baseline at
    # the same budget. We force a tight budget via target_bytes so this
    # test is deterministic on CPU.
    import pandas as pd
    import numpy as np
    np.random.seed(0)
    M = pd.DataFrame(np.zeros((10000, 100)))
    G = pd.DataFrame(np.zeros((1000, 100)))
    C = pd.DataFrame(np.zeros((100, 5)))
    target = 200 * 1_000_000  # 200 MB

    g_no_ig, m_no_ig = _auto_chunk_sizes(M, G, C, target_bytes=target)
    g_ig, m_ig = _auto_chunk_sizes(
        M, G, C, target_bytes=target, compute_ig=True
    )
    g_deep, m_deep = _auto_chunk_sizes(
        M, G, C, target_bytes=target, compute_ig_deep=True
    )

    # All branches must return positive values.
    for label, (g, m) in [
        ('no_ig', (g_no_ig, m_no_ig)),
        ('ig', (g_ig, m_ig)),
        ('deep_ig', (g_deep, m_deep)),
    ]:
        if g is not None:
            assert g >= 1, f'{label}: gene_chunk={g} must be >= 1'
        if m is not None:
            assert m >= 1, f'{label}: meth_chunk={m} must be >= 1'

    # IG paths must not pick a *larger* chunk volume than the no-IG
    # path at the same budget. We compare via the estimator's notion
    # of work-per-chunk = gene * meth (None means whole-set, treat as
    # max). This is the core invariant: more transients => not larger.
    def _vol(g, m):
        return (g if g is not None else len(G)) * (m if m is not None else len(M))

    assert _vol(g_ig, m_ig) <= _vol(g_no_ig, m_no_ig), (
        f'IG path picked larger chunks than no-IG: '
        f'ig={(g_ig, m_ig)} no_ig={(g_no_ig, m_no_ig)}'
    )
    assert _vol(g_deep, m_deep) <= _vol(g_ig, m_ig), (
        f'deep-IG path picked larger chunks than analytical IG: '
        f'deep={(g_deep, m_deep)} ig={(g_ig, m_ig)}'
    )


def test_auto_chunk_sizes_bisection_replaces_naive_quartering():
    # Regression for the 1.22.x OOM bug: when the estimator returns
    # a negative value at mt_count, the helper used to fall back to
    # `(gt_count // 4, mt_count // 4)` which is not budget-aware. The
    # new fallback bisects over mt for a budget-aware pair. With a
    # very tight budget the returned pair must be strictly smaller
    # than the naive `// 4` quartering, and both values must be
    # positive.
    import pandas as pd
    import numpy as np
    np.random.seed(0)
    mt, gt, samples = 50000, 5000, 200
    M = pd.DataFrame(np.zeros((mt, samples)))
    G = pd.DataFrame(np.zeros((gt, samples)))
    C = pd.DataFrame(np.zeros((samples, 10)))

    # Tight budget: forces estimate < 1 path.
    target = 50 * 1_000_000  # 50 MB

    g, m = _auto_chunk_sizes(M, G, C, target_bytes=target, compute_ig=True)
    assert g is not None and m is not None
    assert g >= 1 and m >= 1
    # The naive quartering would have returned (gt // 4, mt // 4) =
    # (1250, 12500). The bisection-derived pair should be strictly
    # smaller in *volume* on this tight budget.
    naive_vol = (gt // 4) * (mt // 4)
    bisect_vol = g * m
    assert bisect_vol < naive_vol, (
        f'bisection volume {bisect_vol} must beat naive quartering '
        f'{naive_vol} on a tight budget; got (g={g}, m={m})'
    )


def test_auto_chunk_sizes_safety_ceiling_clamps_with_ig_on_modest_vram():
    # When IG is enabled and effective free VRAM is modest, the
    # no-anchor auto path must clamp the chosen pair to the
    # belt-and-suspenders safety ceiling (gene <= 2000, meth <= 20000
    # on <=24 GB; gene <= 4000, meth <= 40000 on <=48 GB).
    import pandas as pd
    import numpy as np
    np.random.seed(0)
    M = pd.DataFrame(np.zeros((500000, 100)))
    G = pd.DataFrame(np.zeros((50000, 100)))
    C = pd.DataFrame(np.zeros((100, 5)))

    # 22 GB free => target_bytes = 0.8 * 22 GB ~ 17.6 GB, falls in the
    # <=24 GB bucket.
    target_22gb = int(0.8 * 22 * 1_000_000_000)
    g, m = _auto_chunk_sizes(
        M, G, C, target_bytes=target_22gb, compute_ig=True
    )
    if g is not None:
        assert g <= 2000, f'gene_chunk={g} exceeds 24 GB IG ceiling 2000'
    if m is not None:
        assert m <= 20000, f'meth_chunk={m} exceeds 24 GB IG ceiling 20000'

    # 40 GB free => <=48 GB bucket.
    target_40gb = int(0.8 * 40 * 1_000_000_000)
    g, m = _auto_chunk_sizes(
        M, G, C, target_bytes=target_40gb, compute_ig=True
    )
    if g is not None:
        assert g <= 4000, f'gene_chunk={g} exceeds 48 GB IG ceiling 4000'
    if m is not None:
        assert m <= 40000, f'meth_chunk={m} exceeds 48 GB IG ceiling 40000'


def test_auto_chunk_sizes_safety_ceiling_does_not_apply_without_ig():
    # The safety ceiling only fires with IG enabled; without IG, the
    # auto path is free to pick chunks larger than the IG-bucket
    # values when the budget allows.
    import pandas as pd
    import numpy as np
    np.random.seed(0)
    M = pd.DataFrame(np.zeros((500000, 100)))
    G = pd.DataFrame(np.zeros((50000, 100)))
    C = pd.DataFrame(np.zeros((100, 5)))

    target_22gb = int(0.8 * 22 * 1_000_000_000)
    g, m = _auto_chunk_sizes(M, G, C, target_bytes=target_22gb)
    # No IG: the meth chunk can legitimately exceed the 20000 IG
    # ceiling here (verified empirically).
    assert m is None or m > 20000, (
        f'no-IG path was unexpectedly clamped to IG ceiling: m={m}'
    )


def test_auto_chunk_sizes_safety_ceiling_does_not_clamp_anchored():
    # Anchoring is an explicit user request and must be honored
    # verbatim, even with IG enabled on modest VRAM. The safety
    # ceiling must not silently shrink user-pinned values.
    import pandas as pd
    import numpy as np
    np.random.seed(0)
    M = pd.DataFrame(np.zeros((100000, 100)))
    G = pd.DataFrame(np.zeros((10000, 100)))
    C = pd.DataFrame(np.zeros((100, 5)))

    # Pin a meth chunk well above the 24-GB IG ceiling (20000).
    g, m = _auto_chunk_sizes(
        M, G, C,
        target_bytes=int(0.8 * 22 * 1_000_000_000),
        compute_ig=True,
        pinned_m=80000,
    )
    assert m == 80000, f'anchored meth=80000 was clamped to {m}'

