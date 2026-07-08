import itertools
import math
import os
from typing import Any, List, Optional

import click
import psutil

from .config import DTYPE
from .tool import (
    estimate_loci_per_chunk_e_peak,
    estimate_loci_per_chunk_results_peak,
)


def _host_class(physical_cores: int, ram_gb: float) -> str:
    """Classify the host as 'minimum' (laptop-class, ~16 GB / 8 cores) or
    'server' (workstation/research server). Used as the single source of
    truth for default selection across save-pool sizing, output format,
    prefetch depth, and chunk-size auto-derivation.
    """
    if physical_cores < 12 or ram_gb < 32:
        return 'minimum'
    return 'server'


def _auto_save_threads(physical_cores: int, ram_gb: float) -> int:
    """Pick a default writer-pool size based on host resources.

    Small hosts (laptop-class: <12 physical cores or <32 GB RAM) keep the
    original conservative formula so the 16 GB minimum-config path is
    unchanged. Larger hosts (workstations/servers) scale modestly above
    that floor; profiling on RAID6/dm-crypt LUNs (klabdev) showed the
    underlying device saturates well before 32 concurrent writers, after
    which extra workers only add kernel-writeback CPU cost and pickle
    traffic across `spawn`. Cap at 8.
    """
    if _host_class(physical_cores, ram_gb) == 'minimum':
        # Conservative path for laptop-class hosts.
        return max(2, min(16, min(physical_cores // 4, int(ram_gb // 8))))
    # Larger host: scale with cores and RAM, leave a couple of cores for
    # the GPU producer / OS, and cap at 8.
    return max(2, min(8, min(physical_cores - 2, int(ram_gb // 4))))


def _ig_safety_ceiling(
    target_bytes: int,
    compute_ig: bool,
    compute_ig_deep: bool,
):
    """Return ``(gene_max, meth_max)`` belt-and-suspenders ceiling for the
    no-anchor auto path when IG is enabled, or ``None`` when no clamp
    applies.

    The estimator in ``tecpg.tool`` is conservative but cannot model
    every torch allocator/fragmentation effect. On modest GPUs with IG
    enabled we therefore clamp the auto-sized pair to a known-safe
    ceiling that mirrors the previously-validated static settings
    (``meth=15000, gene=1000``) used by ``pipeline.sh`` before
    auto-sizing. Larger GPUs scale the ceiling up.

    The clamp does *not* apply when the user has explicitly anchored
    one or both chunk dimensions: anchoring is an explicit user
    request and must be honored verbatim.
    """
    if not (compute_ig or compute_ig_deep):
        return None

    # `target_bytes` is already 80% of free VRAM (or system RAM in CPU
    # mode). Translate back to a rough "free GB" view for the buckets.
    free_gb = (target_bytes / 0.8) / 1_000_000_000

    # Buckets are intentionally coarse: the goal is to never silently
    # exceed values that have been observed to OOM on the same class
    # of host.
    if free_gb <= 24:
        return (2000, 20000)
    if free_gb <= 48:
        return (4000, 40000)
    return None


def _auto_chunk_sizes(
    M,
    G,
    C,
    p_only: bool = False,
    full_output: bool = False,
    region: str = 'all',
    logger=None,
    pinned_g: Optional[int] = None,
    pinned_m: Optional[int] = None,
    compute_ig: bool = False,
    compute_ig_deep: bool = False,
    target_bytes: Optional[int] = None,
):
    """Derive (gene_loci_per_chunk, meth_loci_per_chunk) from data shape and
    available device memory.

    Returns a tuple of ints, or (None, None) when no chunking is required
    (i.e. the data fits in the target memory budget). The caller decides
    what to do with `(None, None)` -- on server-class hosts we treat that
    as "let the inner kernel run un-chunked", matching historical
    behavior when the user supplies neither --gene-loci-per-chunk nor
    --meth-loci-per-chunk.

    The target memory budget is 80% of free GPU memory when CUDA is
    available, otherwise 80% of total system RAM. This matches the
    convention used by the existing `tecpg chunks` subcommand. Tests
    may override the budget directly via `target_bytes`.

    The estimators in `tecpg.tool` (`estimate_loci_per_chunk_e_peak` /
    `estimate_loci_per_chunk_results_peak`) reflect the post-PR-1 inner
    kernel: when `methylation_only=True` and no IG is requested, the
    per-CpG `(B, S, T, P)` tensors are realized at the active
    methylation column only (K=1), and the late `torch.cat([...])` that
    used to roughly double peak in the non-`p_only` path has been
    replaced by an in-place buffer assembly. The formulas do not need
    new terms for those changes -- the `full_output=False` branch
    already implicitly modeled K=1, and the `2 *` factor in the
    results-peak formula is preserved as a conservative upper bound on
    transient overlap during buffer assembly (4 * K * N pre-allocated
    buffer plus B/S/T/P still live until each is incrementally freed).

    IG awareness:
      - When `compute_ig` or `compute_ig_deep` is true, the estimators
        receive an extra `(M, S, K)`-equivalent constants term plus a
        per-locus IG factor. Without this term, on tight budgets (e.g.
        an L4 with ~22 GB free VRAM and a 336k x 39k x 340 dataset)
        the estimator returned a negative chunk size and the no-anchor
        fallback below silently quartered the inputs, OOM-ing the GPU.

    Anchor mode:
      - If `pinned_m` is supplied, it is treated as the methylation
        chunk size and the gene-loci-per-chunk is auto-derived from the
        budget given that anchor.
      - If `pinned_g` is supplied (and not `pinned_m`), it is treated
        as the gene-loci-per-chunk and `meth_loci_per_chunk` is sized
        down (via bisection over `mt_count`) until the per-chunk peak
        fits the budget at the requested `pinned_g`.
      - If both are supplied, the helper returns them unchanged (the
        user has fully specified chunking).
    """
    # Local imports keep the click-parse-time cost of cli.py low and
    # avoid importing torch at module import time (it's already imported
    # below for the cli body, but this helper may be called from tests).
    import torch as _torch

    samples = len(C)
    mt_count = len(M)
    gt_count = len(G)
    covar_count = len(C.columns)
    datum_bytes = _torch.ones(1, dtype=DTYPE).element_size()

    if target_bytes is not None:
        target_label = 'caller-supplied target'
    elif _torch.cuda.is_available():
        target_bytes = int(_torch.cuda.mem_get_info()[0] * 0.8)
        target_label = 'CUDA free memory'
    else:
        target_bytes = int(psutil.virtual_memory().available * 0.8)
        target_label = 'system RAM available'

    def _estimate_for(mt: int) -> float:
        # The qr path used by the server profile produces
        # "p-only-like" output regardless of whether p_only/full_output
        # flags are set, because the inner kernel only realizes the
        # active K columns; we use the more conservative of the E-peak
        # and results-peak estimates. Region filtration shrinks the
        # result; we don't model that here (estimates stay conservative
        # when --cis/--distal/--trans is set).
        est_e = estimate_loci_per_chunk_e_peak(
            target_bytes,
            samples,
            mt,
            gt_count,
            covar_count,
            datum_bytes,
            filtration=1.0,
            full_output=full_output,
            p_only=p_only,
            p_filtration=False,
            region_filtration=False,
            compute_ig=compute_ig,
            compute_ig_deep=compute_ig_deep,
        )
        est_r = estimate_loci_per_chunk_results_peak(
            target_bytes,
            samples,
            mt,
            gt_count,
            covar_count,
            datum_bytes,
            filtration=1.0,
            full_output=full_output,
            p_only=p_only,
            region_filtration=False,
            compute_ig=compute_ig,
            compute_ig_deep=compute_ig_deep,
        )
        return min(est_e, est_r)

    def _bisect_largest_mt_for_g(target_g: int) -> int:
        """Return the largest mt in [1, mt_count] for which the
        estimator admits at least `target_g` genes. Returns 0 if even
        mt=1 cannot accommodate `target_g` (caller decides fallback).

        Closure over `mt_count` and `_estimate_for` from the enclosing
        `_auto_chunk_sizes` scope.
        """
        lo, hi = 1, mt_count
        best_mt = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if _estimate_for(mid) >= target_g:
                best_mt = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best_mt

    # Both anchored: nothing to derive.
    if pinned_g is not None and pinned_m is not None:
        return (pinned_g, pinned_m)

    # Anchor methylation, derive gene chunk.
    if pinned_m is not None:
        anchored_mt = max(1, min(int(pinned_m), mt_count))
        estimate = _estimate_for(anchored_mt)
        if logger is not None:
            logger.info(
                'Chunk-size estimator (anchored --meth-loci-per-chunk={0}): '
                'target={1:.1f} MB ({2}), estimate={3:.0f}',
                anchored_mt,
                target_bytes / 1_000_000,
                target_label,
                estimate,
            )
        if estimate >= gt_count:
            # Whole-G fits in budget at the user's --meth-loci-per-chunk:
            # no --gene-loci-per-chunk chunking required.
            return (None, anchored_mt)
        if estimate < 1:
            # The user's anchored --meth-loci-per-chunk does not leave
            # room for even a single gene at the current budget. We
            # cannot bisect over mt (it is anchored), so degrade
            # gracefully to gene_chunk=1 with a loud warning rather
            # than silently picking `gt_count // 4`, which has no
            # relationship to the budget and was the source of the
            # 1.22.x OOM regression on tight GPUs.
            if logger is not None:
                logger.warning(
                    'Chunk-size estimator: anchored '
                    '--meth-loci-per-chunk={0} exceeds budget '
                    '(target={1:.1f} MB ({2}), samples={3}, mt={4}, '
                    'gt={5}, covars={6}, compute_ig={7}, '
                    'compute_ig_deep={8}); falling back to '
                    '--gene-loci-per-chunk=1. Consider lowering '
                    '--meth-loci-per-chunk or disabling IG.',
                    anchored_mt, target_bytes / 1_000_000, target_label,
                    samples, mt_count, gt_count, covar_count,
                    compute_ig, compute_ig_deep,
                )
            return (1, anchored_mt)
        return (max(1, int(estimate)), anchored_mt)

    # Anchor gene, derive methylation chunk via bisection over mt.
    if pinned_g is not None:
        anchored_g = max(1, min(int(pinned_g), gt_count))
        # Find the largest mt in [1, mt_count] such that the
        # estimator's gene-loci-per-chunk is >= anchored_g.
        best_mt = _bisect_largest_mt_for_g(anchored_g)
        if best_mt == 0:
            # Even mt=1 cannot accommodate the requested
            # --gene-loci-per-chunk; degrade gracefully.
            best_mt = 1
            if logger is not None:
                logger.warning(
                    'Chunk-size estimator: anchored '
                    '--gene-loci-per-chunk={0} exceeds budget even at '
                    '--meth-loci-per-chunk=1 (target={1:.1f} MB ({2}), '
                    'samples={3}, mt={4}, gt={5}, covars={6}, '
                    'compute_ig={7}, compute_ig_deep={8}); falling '
                    'back to --meth-loci-per-chunk=1.',
                    anchored_g, target_bytes / 1_000_000, target_label,
                    samples, mt_count, gt_count, covar_count,
                    compute_ig, compute_ig_deep,
                )
        if logger is not None:
            logger.info(
                'Chunk-size estimator (anchored --gene-loci-per-chunk={0}): '
                'target={1:.1f} MB ({2}), derived '
                '--meth-loci-per-chunk={3}',
                anchored_g, target_bytes / 1_000_000, target_label, best_mt,
            )
        return (anchored_g, best_mt)

    # Neither anchored: original logic, but no 40000 meth ceiling.
    estimate = _estimate_for(mt_count)
    if logger is not None:
        logger.info(
            'Chunk-size estimator: target={0:.1f} MB ({1}), '
            'estimate={2:.0f}',
            target_bytes / 1_000_000,
            target_label,
            estimate,
        )

    if estimate >= gt_count:
        # Whole-G fits in budget: no chunking needed.
        gene_chunk: Optional[int] = None
        meth_chunk: Optional[int] = None
    elif estimate < 1:
        # Cannot fit even one gene at the current methylation count.
        # Bisect over mt to find a balanced (mt, g) pair that fits the
        # budget: the largest mt for which the estimator admits a
        # non-trivial gene chunk. This replaces the previous naive
        # `(gt_count // 4, mt_count // 4)` fallback which was not
        # budget-aware and caused the 1.22.x OOM regression on tight
        # GPUs (see CHANGELOG 1.22.2-dev).
        #
        # `g_floor` keeps the gene chunk above a small minimum so we do
        # not pick `mt_count`-style oversized meth chunks in exchange
        # for `gene_chunk=1` (essentially serial). 64 is a small power
        # of two that mirrors typical GPU warp/wave scheduling and is
        # negligible relative to realistic gene counts.
        if logger is not None:
            logger.warning(
                'Chunk-size estimator: budget too tight at '
                'mt_count={0} (target={1:.1f} MB ({2}), samples={3}, '
                'gt={4}, covars={5}, compute_ig={6}, '
                'compute_ig_deep={7}); bisecting over mt for a '
                'safe chunk pair.',
                mt_count, target_bytes / 1_000_000, target_label,
                samples, gt_count, covar_count,
                compute_ig, compute_ig_deep,
            )
        g_floor = min(64, gt_count)
        best_mt = _bisect_largest_mt_for_g(g_floor)
        if best_mt == 0:
            # Even at g_floor we cannot fit; relax to g=1.
            best_mt = _bisect_largest_mt_for_g(1)
        if best_mt == 0:
            # Pathological: even (mt=1, gene=1) does not fit. This
            # would mean constants alone exceed budget; we cannot
            # recover. Pick the smallest possible pair and let the
            # runtime surface the OOM at a known-tiny size rather
            # than at the previous oversized fallback.
            if logger is not None:
                logger.warning(
                    'Chunk-size estimator: even (mt=1, gene=1) '
                    'exceeds the budget; returning minimal '
                    '(gene_chunk=1, meth_chunk=1). The run may OOM; '
                    'consider running on a host with more memory.',
                )
            gene_chunk = 1
            meth_chunk = 1
        else:
            gene_chunk = max(1, min(int(_estimate_for(best_mt)), gt_count))
            meth_chunk = best_mt
    else:
        gene_chunk = max(1, int(estimate))
        # Use the full methylation set per chunk: the RAM/GPU budget is
        # the binding constraint (the estimator above is keyed on
        # `mt_count`), so capping at 40000 was redundant pessimism that
        # could only force extra outer iterations on hosts where a
        # larger --meth-loci-per-chunk fit fine. The post-PR-1
        # footprint makes the un-capped value safer still.
        meth_chunk = mt_count

    # Belt-and-suspenders: when IG is enabled on a modest GPU, clamp
    # the no-anchor auto pair to a known-safe ceiling. The clamp does
    # nothing when both values are already within the ceiling or when
    # IG is disabled. It is intentionally not applied to anchored
    # modes -- anchoring is an explicit user request.
    if gene_chunk is not None and meth_chunk is not None:
        ceiling = _ig_safety_ceiling(
            target_bytes, compute_ig, compute_ig_deep,
        )
        if ceiling is not None:
            gene_max, meth_max = ceiling
            clamped_g = min(gene_chunk, gene_max)
            clamped_m = min(meth_chunk, meth_max)
            if (
                logger is not None
                and (clamped_g != gene_chunk or clamped_m != meth_chunk)
            ):
                logger.warning(
                    'Chunk-size estimator: IG enabled with tight VRAM '
                    '(target={0:.1f} MB ({1})); clamping auto-sized '
                    '(gene={2}, meth={3}) to safety ceiling '
                    '(gene<={4}, meth<={5}).',
                    target_bytes / 1_000_000, target_label,
                    gene_chunk, meth_chunk, gene_max, meth_max,
                )
            gene_chunk, meth_chunk = clamped_g, clamped_m

    return (gene_chunk, meth_chunk)

import pandas
import psutil
import torch

from . import __version__
from .config import (
    DEFAULT_CIS_DOWNSTREAM,
    DEFAULT_CIS_UPSTREAM,
    DEFAULT_CIS_WINDOW_BASE,
    DEFAULT_DISTAL_DOWNSTREAM,
    DEFAULT_DISTAL_UPSTREAM,
    DEFAULT_DISTAL_WINDOW_BASE,
    DEFAULT_FLOAT_FORMAT,
    DTYPE,
    data,
    using_gpu,
)
from .gtp import save_gtp_data
from .mesa import save_mesa_data
from .helper import default_region_parameter, initialize_dir, verify_and_trim_samples
from .import_data import read_dataframes, save_dataframes
from .logger import Logger
from .pearson_full import (
    pearson_chunk_save_tensor,
    pearson_chunk_tensor,
    pearson_full_tensor,
)
from .processing import tecpg_mlr_qr
from .regression_full import regression_full
from .regression_single import regression_single
from .test_data import generate_data
from .tool import (
    estimate_constants_bytes,
    estimate_loci_per_chunk_e_peak,
    estimate_loci_per_chunk_results_peak,
)


@click.group()
@click.version_option(version=__version__, message='tecpg version %(version)s')
@click.option(
    '-r',
    '--root-path',
    show_default=True,
    default=data['root_path'],
    type=click.Path(file_okay=False, readable=True, resolve_path=True),
)
@click.option(
    '-i',
    '--input-dir',
    show_default=True,
    default=data['input_dir'],
    type=click.Path(file_okay=False),
)
@click.option(
    '-a',
    '--annot-dir',
    show_default=True,
    default=data['annot_dir'],
    type=click.Path(file_okay=False),
)
@click.option(
    '-o',
    '--output-dir',
    show_default=True,
    default=data['output_dir'],
    type=click.Path(file_okay=False),
)
@click.option(
    '-m',
    '--meth-file',
    show_default=True,
    default=data['meth_file'],
    type=click.Path(dir_okay=False),
)
@click.option(
    '-g',
    '--gene-file',
    show_default=True,
    default=data['gene_file'],
    type=click.Path(dir_okay=False),
)
@click.option(
    '-c',
    '--covar-file',
    show_default=True,
    default=data['covar_file'],
    type=click.Path(dir_okay=False),
)
@click.option(
    '-M',
    '--meth-annot',
    show_default=True,
    default=data['meth_annot'],
    type=click.Path(dir_okay=False),
)
@click.option(
    '-G',
    '--gene-annot',
    show_default=True,
    default=data['gene_annot'],
    type=click.Path(dir_okay=False),
)
@click.option(
    '-f',
    '--output-file',
    show_default=True,
    default=data['output'],
    type=click.Path(dir_okay=False),
)
@click.option(
    '-t',
    '--cpu-threads',
    show_default=True,
    default=0,
    type=int,
    help='If 0, runs on the GPU if available',
)
@click.option(
    '-v', '--verbosity', show_default=True, default=1, type=int, count=True
)
@click.option(
    '-D', '--debug', is_flag=True, show_default=True, default=False, type=bool
)
@click.option(
    '-l',
    '--log-dir',
    show_default=True,
    default=data['log_dir'],
    type=click.Path(file_okay=False),
)
@click.option(
    '-n',
    '--no-log-file',
    is_flag=True,
    show_default=True,
    default=False,
    type=bool,
)
@click.option(
    '-F',
    '--float-format',
    show_default=True,
    default=DEFAULT_FLOAT_FORMAT,
    type=str,
)
@click.option(
    '--save-threads',
    show_default=True,
    default=None,
    envvar='TECPG_SAVE_THREADS',
    type=int,
    help='Number of writer processes for saving output. If unset, auto-scales based on physical CPU count and available RAM. Set explicitly (or via TECPG_SAVE_THREADS) to override.',
)
@click.option(
    '--save-queue-depth',
    show_default=True,
    default=0,
    envvar='TECPG_SAVE_QUEUE',
    type=int,
    help=(
        'Maximum number of save tasks allowed in flight at once. 0 (default) auto-scales to '
        '~4x save_threads, bounded by available RAM. Increasing this hides slow writers (e.g. '
        'CSV over dm-crypt/RAID6) at the cost of more buffered DataFrames in CPU memory.'
    ),
)
@click.option(
    '--blas-threads',
    show_default=True,
    default=0,
    envvar='TECPG_BLAS_THREADS',
    type=int,
    help='Number of threads for BLAS/OpenMP operations. Env var TECPG_BLAS_THREADS is preferred. This CLI flag works via a pre-import shim.',
)
@click.option(
    '--host-profile',
    show_default=True,
    default='auto',
    envvar='TECPG_HOST_PROFILE',
    type=click.Choice(['auto', 'minimum', 'server'], case_sensitive=False),
    help=(
        "Host-class preset that drives defaults for save-pool size, output "
        "format, and chunk auto-sizing. 'auto' (default) detects from "
        "physical CPU count and total RAM: <12 cores or <32 GB RAM is "
        "treated as 'minimum' (laptop-class), otherwise 'server'. "
        "'minimum' preserves the conservative 16 GB / 8-core defaults. "
        "'server' enables Parquet output, a thread-pool writer, and "
        "auto-derived chunk sizes. Explicit per-flag overrides "
        "(e.g. --save-threads, --output-format, --gene-loci-per-chunk, "
        "--meth-loci-per-chunk) always win."
    ),
)
@click.pass_context
def cli(
    ctx: Optional[click.Context] = None,
    root_path: Optional[str] = None,
    input_dir: Optional[str] = None,
    annot_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    meth_file: Optional[str] = None,
    gene_file: Optional[str] = None,
    covar_file: Optional[str] = None,
    meth_annot: Optional[str] = None,
    gene_annot: Optional[str] = None,
    output_file: Optional[str] = None,
    cpu_threads: Optional[int] = None,
    verbosity: Optional[int] = None,
    debug: Optional[bool] = None,
    log_dir: Optional[str] = None,
    no_log_file: Optional[bool] = None,
    float_format: Optional[str] = None,
    save_threads: Optional[int] = None,
    save_queue_depth: Optional[int] = None,
    blas_threads: Optional[int] = None,
    host_profile: Optional[str] = None,
    obj: Optional[dict] = None,
) -> None:
    """The root cli group"""
    assert obj is None
    ctx.ensure_object(dict)

    data['root_path'] = click.format_filename(root_path)
    data['input_dir'] = click.format_filename(input_dir)
    data['annot_dir'] = click.format_filename(annot_dir)
    data['output_dir'] = click.format_filename(output_dir)
    data['meth_file'] = click.format_filename(meth_file)
    data['gene_file'] = click.format_filename(gene_file)
    data['covar_file'] = click.format_filename(covar_file)
    data['meth_annot'] = click.format_filename(meth_annot)
    data['gene_annot'] = click.format_filename(gene_annot)
    data['output'] = click.format_filename(output_file)
    data['log_dir'] = click.format_filename(log_dir)

    log_path = None if no_log_file else os.path.join(root_path, log_dir)
    logger = Logger(verbosity, debug, log_path)
    logger.info('tecpg version {0}', __version__)
    using_gpu(**logger)
    if cpu_threads:
        torch.set_num_threads(cpu_threads)
        logger.carry_data['use_cpu'] = True
    logger.carry_data['float_format'] = (
        DEFAULT_FLOAT_FORMAT if float_format is None else float_format
    )

    # Resolve host-class profile. 'auto' detects from physical CPU count
    # and total RAM; explicit 'minimum'/'server' wins. The resolved value
    # is the single source of truth consumed by other defaults below
    # (save_threads, save_queue, output_format, chunk auto-sizing).
    physical = psutil.cpu_count(logical=False) or os.cpu_count() or 2
    ram_gb = psutil.virtual_memory().total / (1024**3)
    requested_profile = (host_profile or 'auto').lower()
    if requested_profile == 'auto':
        resolved_profile = _host_class(physical, ram_gb)
        logger.info(
            'Auto-detected host_profile={0} (physical_cores={1}, ram_gb={2:.1f})',
            resolved_profile, physical, ram_gb,
        )
    else:
        resolved_profile = requested_profile
        logger.info(
            'User-supplied host_profile={0} (physical_cores={1}, ram_gb={2:.1f})',
            resolved_profile, physical, ram_gb,
        )
    logger.carry_data['host_profile'] = resolved_profile

    if save_threads is not None:
        logger.info('User-supplied save_threads: {0}', save_threads)
        logger.carry_data['save_threads'] = save_threads
    else:
        if resolved_profile == 'minimum':
            # Force the conservative formula even on big hardware when the
            # user explicitly asked for the minimum profile.
            auto_threads = max(
                2, min(16, min(physical // 4, int(ram_gb // 8)))
            )
        else:
            auto_threads = _auto_save_threads(physical, ram_gb)
        logger.info('Auto-scaled save_threads to {0}', auto_threads)
        logger.carry_data['save_threads'] = auto_threads

    # Resolve save-queue-depth: 0 means auto. Auto scales with save_threads
    # but is bounded by RAM (each in-flight save buffers a result DataFrame
    # in CPU memory until written) so 16 GB hosts aren't pushed into swap.
    _resolved_save_threads = logger.carry_data['save_threads']
    if save_queue_depth and save_queue_depth > 0:
        logger.info('User-supplied save_queue_depth: {0}', save_queue_depth)
        logger.carry_data['save_queue_depth'] = save_queue_depth
    else:
        _ram_gb = psutil.virtual_memory().total / (1024**3)
        # Floor: keep the historical minimum so small hosts behave as before.
        _floor = _resolved_save_threads + 1
        # Ceiling: 4x writer pool, but never deeper than ~1 chunk per 8 GB RAM.
        auto_queue = max(_floor, min(4 * _resolved_save_threads, int(_ram_gb // 8)))
        logger.info('Auto-scaled save_queue_depth to {0}', auto_queue)
        logger.carry_data['save_queue_depth'] = auto_queue

    if blas_threads and blas_threads > 0:
        env_omp = os.environ.get('OMP_NUM_THREADS')
        if env_omp != str(blas_threads):
            logger.warning(
                'Warning: --blas-threads {} was passed, but OMP_NUM_THREADS is {}. '
                'This may mean the env-var shim missed the flag. '
                'Consider using TECPG_BLAS_THREADS directly.',
                blas_threads, env_omp
            )
        logger.carry_data['blas_threads'] = blas_threads
    else:
        logger.carry_data['blas_threads'] = 0

    ctx.obj['logger'] = logger


@cli.group()
def run() -> None:
    """Base group for running algorithms."""


@run.command()
@click.option('-c', '--chunks', show_default=True, default=0, type=int)
@click.option('-s', '--save-chunks', show_default=True, default=0, type=int)
@click.option(
    '-f',
    '--flatten',
    is_flag=True,
    show_default=True,
    default=True,
    type=bool,
)
@click.pass_context
def corr(
    ctx: click.Context, chunks: int, save_chunks: int, flatten: bool
) -> None:
    """
    Calculate the pearson correlation coefficient.

    Calculate the pearson correlation coefficient with methylation and
    gene expression matrices. Optional compute and save chunking to
    avoid GPU and CPU memory limits.
    """
    logger: Logger = ctx.obj['logger']

    data_path = os.path.join(data['root_path'], data['input_dir'])
    file_names = [data['meth_file'], data['gene_file']]
    dataframes = read_dataframes(data_path, file_names=file_names, **logger)
    M = dataframes[data['meth_file']]
    G = dataframes[data['gene_file']]

    M, G = verify_and_trim_samples(M, G, logger=logger)

    output_path = os.path.join(data['root_path'], data['output_dir'])
    output = None
    if chunks == 0:
        output = pearson_full_tensor(M, G, flatten=flatten, **logger)
    elif save_chunks == 0:
        output = pearson_chunk_tensor(M, G, chunks, flatten=flatten, **logger)
    else:
        pearson_chunk_save_tensor(
            M, G, chunks, save_chunks, output_path, flatten=flatten, **logger
        )
    if output is not None:
        save_dataframes([output], output_path, [data['output']], **logger)

    logger.save()


@run.command()
@click.option('--gene-loci-per-chunk', show_default=True, type=int)
@click.option('--meth-loci-per-chunk', show_default=True, type=int)
@click.option('-p', '--p-thresh', show_default=True, default=0.001, type=float)
@click.option(
    '--all', 'region', show_default=True, flag_value='all', default=True
)
@click.option('--cis', 'region', show_default=True, flag_value='cis')
@click.option('--distal', 'region', show_default=True, flag_value='distal')
@click.option('--trans', 'region', show_default=True, flag_value='trans')
@click.option('-w', '--window-base', show_default=True, type=int)
@click.option('-d', '--downstream', show_default=True, type=int)
@click.option('-u', '--upstream', show_default=True, type=int)
@click.option(
    '--full-output',
    '-f',
    is_flag=True,
    show_default=True,
    default=False,
    type=bool,
)
@click.option(
    '--p-only', '-P', is_flag=True, show_default=True, default=False, type=bool
)
@click.option(
    '--mlr-method',
    type=click.Choice(['legacy_normal_eq', 'qr', 'qr_bootstrap']),
    default='legacy_normal_eq',
    show_default=True,
    help=(
        "The MLR computation method to use. 'legacy_normal_eq' uses the original"
        " optimized inversion; 'qr' uses QR decomposition (torch.linalg.qr) + torch.linalg.solve_triangular;"
        " 'qr_bootstrap' runs empirical bootstrap on specific pairs."
    ),
)
@click.option(
    '--pairs-file',
    type=click.Path(exists=True, dir_okay=False),
    help='Path to a CSV file containing mt_id and gt_id columns. Required for qr_bootstrap.',
)
@click.option(
    '--master-parquet',
    type=click.Path(exists=True, dir_okay=False),
    help='Path to the master annotated Parquet file to merge bootstrap results onto. Required for qr_bootstrap.',
)
@click.option(
    '--bootstrap-iterations',
    show_default=True,
    default=1000,
    type=int,
    help='Number of resamples for qr_bootstrap.',
)
@click.option(
    '--bootstrap-batch-size',
    show_default=True,
    default=10,
    type=int,
    help='Number of pairs to process simultaneously in the bootstrap loop. Note: --gene-loci-per-chunk and --meth-loci-per-chunk chunks are ignored for bootstraps.',
)
@click.option(
    '--logit-transform',
    is_flag=True,
    show_default=True,
    default=False,
    type=bool,
    help='Whether to logit-transform M-values (log2(beta/(1-beta)))',
)
@click.option(
    '--thermal-threshold',
    show_default=True,
    default=80,
    type=int,
    help='GPU temperature threshold for throttling (Celsius)',
)
@click.option(
    '--thermal-wait',
    show_default=True,
    default=30,
    type=int,
    help='Seconds to wait when throttling',
)
@click.option(
    '--reservoir-count',
    show_default=True,
    type=int,
    help='Number of tests to retain in the reservoir buffer (only for qr method)',
)
@click.option(
    '--subsample-mt-count',
    show_default=True,
    type=int,
    help='Number of methylation loci (CpGs) to randomly select',
)
@click.option(
    '--subsample-g-count',
    show_default=True,
    type=int,
    help='Number of gene expression loci to randomly select',
)
@click.option(
    '--seed',
    show_default=True,
    default=42,
    type=int,
    help='Seed for random subsampling and the qr_bootstrap resample draw '
    '(recorded with bootstrap outputs)',
)
@click.option(
    '--permute-label-test',
    is_flag=True,
    show_default=True,
    default=False,
    type=bool,
    help='Whether to perform a permutation (Negative Control) test by shuffling subject IDs in G (only for qr method)',
)
@click.option(
    '--compute-ig',
    is_flag=True,
    help='Compute fast analytical IG scores.',
)
@click.option(
    '--compute-ig-deep',
    is_flag=True,
    help='Compute deep Captum-based IG scores (Slow).',
)
@click.option(
    '--ig-baseline',
    type=click.Choice(['mean', 'zero'], case_sensitive=False),
    default='mean',
    help='Baseline for IG attribution (default: mean).',
)
@click.option(
    '--ig-covariates',
    is_flag=True,
    show_default=True,
    default=False,
    help='Output IG scores for all covariates.',
)
@click.option(
    '--ig-covariates-list',
    type=click.Path(exists=True, dir_okay=False),
    help='Path to a text file containing covariates to output IG scores for (one per line).',
)
@click.option(
    '--prefetch-chunks',
    show_default=True,
    default=-1,
    envvar='TECPG_PREFETCH',
    type=int,
    help=(
        'Number of chunks to prefetch to the GPU to overlap H2D with compute. '
        '-1 (default) auto-scales: ~min(4, free_ram_gb // 8), so a 16 GB host '
        'gets 0-1 and a server with hundreds of GB gets ~4. Set 0 to disable.'
    ),
)
@click.option(
    '--output-format',
    type=click.Choice(['auto', 'csv', 'parquet'], case_sensitive=False),
    default='auto',
    show_default=True,
    help=(
        "Output format for chunked regression results. 'parquet' uses pyarrow + "
        "snappy compression and is typically 5-10x smaller and 5-10x faster to "
        "write than CSV; recommended on slow filesystems (e.g. dm-crypt/RAID6). "
        "'csv' preserves the previous behavior for downstream tooling. "
        "'auto' (default) selects 'parquet' on server-class hosts and 'csv' on "
        "minimum-class hosts (controlled by --host-profile)."
    ),
)
@click.option(
    '--aggressive-gc',
    is_flag=True,
    show_default=True,
    default=False,
    help='Call torch.cuda.empty_cache() after every gene chunk. Default is to only empty the cache under memory pressure (> 75% allocated). Useful on memory-constrained GPUs.',
)
@click.pass_context
def mlr(
    ctx: click.Context,
    gene_loci_per_chunk: Optional[int],
    meth_loci_per_chunk: Optional[int],
    p_thresh: Optional[float],
    region: str,
    window_base: Optional[int],
    downstream: Optional[int],
    upstream: Optional[int],
    full_output: bool,
    p_only: bool,
    mlr_method: str,
    logit_transform: bool,
    thermal_threshold: int,
    thermal_wait: int,
    reservoir_count: Optional[int],
    subsample_mt_count: Optional[int],
    subsample_g_count: Optional[int],
    seed: int,
    permute_label_test: bool,
    compute_ig: bool,
    compute_ig_deep: bool,
    ig_baseline: str,
    ig_covariates: bool,
    ig_covariates_list: Optional[str],
    pairs_file: Optional[str],
    master_parquet: Optional[str],
    bootstrap_iterations: int,
    bootstrap_batch_size: int,
    prefetch_chunks: int,
    output_format: str,
    aggressive_gc: bool,
) -> None:
    logger: Logger = ctx.obj['logger']

    # Auto-resolve prefetch_chunks (-1 sentinel) based on free RAM and the
    # host profile. Prefetch only makes sense when there's a real CUDA
    # device to overlap with -- on CPU-only minimum-config hosts, the
    # prefetch path tries to call .pin_memory() and fails. Done here
    # rather than as a click default so users can still env-set
    # TECPG_PREFETCH=0 explicitly to disable, and we have psutil/logger
    # available.
    if prefetch_chunks < 0:
        if not torch.cuda.is_available():
            prefetch_chunks = 0
            logger.info(
                'Auto-scaled prefetch_chunks to 0 (no CUDA device)',
            )
        elif logger.carry_data.get('host_profile') == 'minimum':
            # The minimum profile asks for the conservative path even
            # when a GPU is present.
            prefetch_chunks = 0
            logger.info(
                'Auto-scaled prefetch_chunks to 0 (host_profile=minimum)',
            )
        else:
            free_ram_gb = psutil.virtual_memory().available / (1024**3)
            prefetch_chunks = min(4, int(free_ram_gb // 8))
            logger.info(
                'Auto-scaled prefetch_chunks to {0} (free_ram={1:.1f} GB)',
                prefetch_chunks,
                free_ram_gb,
            )

    output_format = output_format.lower()
    if output_format == 'auto':
        host_profile = logger.carry_data.get('host_profile', 'minimum')
        # Server-class hosts default to parquet (faster + far smaller on
        # RAID6/dm-crypt). Minimum-class hosts keep the historical CSV
        # default so downstream tooling and small-fixture tests are
        # unchanged. Explicit --output-format always wins.
        output_format = 'parquet' if host_profile == 'server' else 'csv'
        logger.info(
            'Auto-resolved output_format={0} (host_profile={1})',
            output_format, host_profile,
        )
    logger.carry_data['output_format'] = output_format

    chunking = (
        gene_loci_per_chunk is not None or meth_loci_per_chunk is not None
    )
    data_path = os.path.join(data['root_path'], data['input_dir'])
    output_path = os.path.join(data['root_path'], data['output_dir'])

    file_names = [data['meth_file'], data['gene_file'], data['covar_file']]
    dataframes = read_dataframes(data_path, file_names=file_names, **logger)
    M = dataframes[data['meth_file']]
    G = dataframes[data['gene_file']]
    C = dataframes[data['covar_file']]

    M, G, C = verify_and_trim_samples(M, G, C, logger=logger)

    # Auto-derive chunk sizes when (a) on a server-class host and the
    # user supplied neither --gene-loci-per-chunk nor
    # --meth-loci-per-chunk, or (b) the user anchored exactly one of
    # them and wants the other auto-derived from the budget. We use the
    # same memory heuristics as the `tecpg chunks` subcommand. On
    # minimum-class hosts the no-anchor branch is intentionally
    # disabled -- the user's explicit choice (or no chunking) is
    # preserved -- but the anchor branch is honored on any host because
    # it is an explicit user request.
    host_profile = logger.carry_data.get('host_profile', 'minimum')
    user_anchored_one = (
        (gene_loci_per_chunk is None) ^ (meth_loci_per_chunk is None)
    )
    user_anchored_none = (
        gene_loci_per_chunk is None and meth_loci_per_chunk is None
    )
    should_auto = user_anchored_one or (
        user_anchored_none and host_profile == 'server'
    )
    if should_auto:
        # Capture which side was anchored before we mutate the locals.
        if user_anchored_one:
            anchor_label = 'm' if gene_loci_per_chunk is None else 'g'
        else:
            anchor_label = 'none'
        try:
            auto_g, auto_m = _auto_chunk_sizes(
                M, G, C,
                p_only=p_only,
                full_output=full_output,
                region=region,
                logger=logger,
                pinned_g=gene_loci_per_chunk,
                pinned_m=meth_loci_per_chunk,
                compute_ig=compute_ig,
                compute_ig_deep=compute_ig_deep,
            )
        except Exception as exc:
            # Heuristic estimation is best-effort. If it fails for any
            # reason (tiny data, unexpected dtype, etc.) fall back to the
            # historical "no chunking" default rather than aborting.
            logger.warning(
                'chunk auto-sizing failed ({0}); falling back to no chunking',
                exc,
            )
            auto_g = auto_m = None
        if auto_g is not None and auto_m is not None:
            gene_loci_per_chunk = auto_g
            meth_loci_per_chunk = auto_m
            chunking = True
            logger.info(
                'Auto-scaled chunk sizes: gene_loci_per_chunk={0}, '
                'meth_loci_per_chunk={1} (host_profile={2}, anchored={3})',
                gene_loci_per_chunk, meth_loci_per_chunk, host_profile,
                anchor_label,
            )
        elif auto_g is None and auto_m is not None:
            # No --gene-loci-per-chunk chunking needed at the user's
            # anchored --meth-loci-per-chunk.
            meth_loci_per_chunk = auto_m
            chunking = True
            logger.info(
                'Auto-scaled chunk sizes: meth_loci_per_chunk={0} '
                '(no --gene-loci-per-chunk chunking required at this '
                '--meth-loci-per-chunk, host_profile={1})',
                meth_loci_per_chunk, host_profile,
            )

    if region != 'all':
        annot_path = os.path.join(data['root_path'], data['annot_dir'])
        M_annot = pandas.read_csv(
            os.path.join(annot_path, data['meth_annot']), sep=None, engine='python'
        ).set_index('name')
        G_annot = pandas.read_csv(
            os.path.join(annot_path, data['gene_annot']), sep=None, engine='python'
        ).set_index('name')

    window_base = default_region_parameter(
        'window_base',
        window_base,
        region,
        {'cis': DEFAULT_CIS_WINDOW_BASE, 'distal': DEFAULT_DISTAL_WINDOW_BASE},
    )
    downstream = default_region_parameter(
        'downstream',
        downstream,
        region,
        {'cis': DEFAULT_CIS_DOWNSTREAM, 'distal': DEFAULT_DISTAL_DOWNSTREAM},
    )
    upstream = default_region_parameter(
        'upstream',
        upstream,
        region,
        {'cis': DEFAULT_CIS_UPSTREAM, 'distal': DEFAULT_DISTAL_UPSTREAM},
    )

    if compute_ig and compute_ig_deep:
        error = 'Cannot use both --compute-ig and --compute-ig-deep simultaneously.'
        logger.error(error)
        raise click.UsageError(error)

    if compute_ig_deep and p_thresh is None:
        error = '--compute-ig-deep requires a --p-thresh to be set to avoid computational collapse.'
        logger.error(error)
        raise click.UsageError(error)

    ig_covariates_filter = None
    if ig_covariates_list:
        with open(ig_covariates_list, 'r') as f:
            target_covariates = [line.strip() for line in f if line.strip()]

        missing_covariates = [covar for covar in target_covariates if covar not in C.columns]
        if missing_covariates:
            error = f"The following covariates from --ig-covariates-list are missing in the design matrix: {', '.join(missing_covariates)}"
            logger.error(error)
            raise click.UsageError(error)
        ig_covariates_filter = target_covariates
    elif ig_covariates:
        ig_covariates_filter = 'all'

    methylation_only = not full_output

    kwargs = {
        'M': M,
        'G': G,
        'C': C,
        'M_annot': M_annot if region != 'all' else None,
        'G_annot': G_annot if region != 'all' else None,
        'region': region,
        'window_base': window_base,
        'downstream': downstream,
        'upstream': upstream,
        'gene_loci_per_chunk': gene_loci_per_chunk,
        'meth_loci_per_chunk': meth_loci_per_chunk,
        'p_thresh': p_thresh,
        'output_dir': output_path if chunking else None,
        'methylation_only': methylation_only,
        'p_only': p_only,
        'logit_transform': logit_transform,
        'thermal_threshold': thermal_threshold,
        'thermal_wait': thermal_wait,
        'subsample_mt_count': subsample_mt_count,
        'subsample_g_count': subsample_g_count,
        'seed': seed,
        'permute_label_test': permute_label_test,
        'compute_ig': compute_ig,
        'compute_ig_deep': compute_ig_deep,
        'ig_baseline': ig_baseline,
        'ig_covariates_filter': ig_covariates_filter,
        'prefetch_chunks': prefetch_chunks,
        'aggressive_gc': aggressive_gc,
        'output_format': output_format,
    }

    logger.info(
        'Running mlr with options: {0}',
        {
            k: v
            for k, v in kwargs.items()
            if k not in ['M', 'G', 'C', 'M_annot', 'G_annot']
        },
    )

    if p_thresh is not None:
        logger.info('Using p-value threshold: {0}', p_thresh)

    if mlr_method == 'qr_bootstrap':
        if not pairs_file or not master_parquet:
            error = '--pairs-file and --master-parquet are required for qr_bootstrap.'
            logger.error(error)
            raise click.UsageError(error)

        from .bootstrap import tecpg_mlr_qr_bootstrap

        # The merged bootstrap output is named `bootstrap_merged.<ext>` by
        # default (matching the chunked path, README, pipelinePost.sh and the
        # docs) so the whole post-processing pipeline keeps working. We only
        # honor an explicit --output-file; the default 'out.csv' is treated as
        # "unset" so we don't write a Parquet file misleadingly named .csv.
        output_file_path = data['output']
        if output_file_path == 'out.csv':
            ext = 'parquet' if output_format == 'parquet' else 'csv'
            output_file_path = os.path.join(
                output_path, f'bootstrap_merged.{ext}'
            )
        elif output_path and not output_file_path.startswith('/'):
            output_file_path = os.path.join(output_path, output_file_path)

        tecpg_mlr_qr_bootstrap(
            M=M,
            G=G,
            C=C,
            pairs_file=pairs_file,
            master_parquet=master_parquet,
            output_file=output_file_path,
            iterations=bootstrap_iterations,
            batch_size=bootstrap_batch_size,
            thermal_threshold=thermal_threshold,
            thermal_wait=thermal_wait,
            compute_ig=compute_ig,
            compute_ig_deep=compute_ig_deep,
            ig_baseline=ig_baseline,
            ig_covariates_filter=ig_covariates_filter,
            seed=seed,
            logger=logger
        )
        return

    if mlr_method == 'qr':
        if reservoir_count is None:
            total_comparisons = len(M) * len(G)
            reservoir_count = min(1_000_000, max(1, int(0.01 * total_comparisons)))
        kwargs['reservoir_count'] = reservoir_count
        # Inject output_dir into logger carry_data for reservoir saver
        logger.carry_data['output_dir'] = output_path
        output = tecpg_mlr_qr(**kwargs, **logger)
    else:
        if reservoir_count is not None:
            logger.warning('--reservoir-count is only supported for mlr-method qr')
        if permute_label_test:
            logger.warning('--permute-label-test is only supported for mlr-method qr')
        kwargs.pop('permute_label_test', None)
        if compute_ig or compute_ig_deep:
            logger.warning('Integrated Gradients (--compute-ig/--compute-ig-deep) are only supported for mlr-method qr. They will be ignored.')
        kwargs.pop('compute_ig', None)
        kwargs.pop('compute_ig_deep', None)
        kwargs.pop('ig_baseline', None)
        kwargs.pop('ig_covariates_filter', None)
        output = regression_full(**kwargs, **logger)
    if not chunking:
        save_dataframes([output], output_path, [data['output']], clear_dir=False, **logger)


@run.command()
@click.option(
    '-r', '--regressions-per-chunk', show_default=True, default=0, type=int
)
@click.option('-p', '--p-thresh', show_default=True, default=0.00001, type=float)
@click.option(
    '--all', 'region', show_default=True, flag_value='all', default=True
)
@click.option('--cis', 'region', show_default=True, flag_value='cis')
@click.option('--distal', 'region', show_default=True, flag_value='distal')
@click.option('--trans', 'region', show_default=True, flag_value='trans')
@click.option('-w', '--window', show_default=True, type=int)
@click.option(
    '--full-output',
    '-f',
    is_flag=True,
    show_default=True,
    default=False,
    type=bool,
)
@click.option(
    '--no-est', is_flag=True, show_default=True, default=False, type=bool
)
@click.option(
    '--no-err', is_flag=True, show_default=True, default=False, type=bool
)
@click.option(
    '--no-t', is_flag=True, show_default=True, default=False, type=bool
)
@click.option(
    '--no-p', is_flag=True, show_default=True, default=False, type=bool
)
@click.option(
    '--logit-transform',
    is_flag=True,
    show_default=True,
    default=False,
    type=bool,
    help='Whether to logit-transform M-values (log2(beta/(1-beta)))',
)
@click.option(
    '--thermal-threshold',
    show_default=True,
    default=80,
    type=int,
    help='GPU temperature threshold for throttling (Celsius)',
)
@click.option(
    '--thermal-wait',
    show_default=True,
    default=30,
    type=int,
    help='Seconds to wait when throttling',
)
@click.option(
    '--compute-ig',
    is_flag=True,
    help='Compute fast analytical IG scores.',
)
@click.option(
    '--compute-ig-deep',
    is_flag=True,
    help='Compute deep Captum-based IG scores (Slow).',
)
@click.option(
    '--ig-baseline',
    type=click.Choice(['mean', 'zero'], case_sensitive=False),
    default='mean',
    help='Baseline for IG attribution (default: mean).',
)
@click.pass_context
def mlr_single(
    ctx: click.Context,
    regressions_per_chunk: int,
    p_thresh: Optional[float],
    region: str,
    window: Optional[int],
    full_output: bool,
    no_est: bool,
    no_err: bool,
    no_t: bool,
    no_p: bool,
    logit_transform: bool,
    thermal_threshold: int,
    thermal_wait: int,
    compute_ig: bool,
    compute_ig_deep: bool,
    ig_baseline: str,
) -> None:
    """
    Calculates the multiple linear regression.

    Calculate the multiple linear regression with methylation, gene
    expression, and covariate matrices. Optional chunking to avoid
    memory limits.
    """
    logger: Logger = ctx.obj['logger']

    data_path = os.path.join(data['root_path'], data['input_dir'])
    output_path = os.path.join(data['root_path'], data['output_dir'])

    file_names = [data['meth_file'], data['gene_file'], data['covar_file']]
    dataframes = read_dataframes(data_path, file_names=file_names, **logger)
    M = dataframes[data['meth_file']]
    G = dataframes[data['gene_file']]
    C = dataframes[data['covar_file']]

    M, G, C = verify_and_trim_samples(M, G, C, logger=logger)

    include = (not no_est, not no_err, not no_t, not no_p)

    if region != 'all':
        annot_path = os.path.join(data['root_path'], data['annot_dir'])
        M_annot = (
            pandas.read_csv(
                os.path.join(annot_path, data['meth_annot']), sep=None
            )
            .set_index('name')
            .drop(['chromEnd', 'score', 'strand'])
        )
        G_annot = (
            pandas.read_csv(
                os.path.join(annot_path, data['gene_annot']), sep=None
            )
            .set_index('name')
            .drop(['chromEnd', 'score', 'strand'])
        )

    methylation_only = not full_output
    if region in ['cis', 'distal'] and window is None:
        logger.info('No region window provided. Resorting to default.')
        if region == 'cis':
            logger.info(
                'Using default window for cis of {0} bases',
                DEFAULT_CIS_UPSTREAM,
            )
            window = DEFAULT_CIS_UPSTREAM
        if region == 'distal':
            logger.info(
                'Using default window for distal of {0} bases',
                DEFAULT_DISTAL_UPSTREAM,
            )
            window = DEFAULT_DISTAL_UPSTREAM

    if compute_ig or compute_ig_deep:
        logger.warning('Integrated Gradients (--compute-ig/--compute-ig-deep) are only supported for mlr-method qr via mlr command. They will be ignored in mlr_single.')

    kwargs = {
        'M': M,
        'G': G,
        'C': C,
        'include': include,
        'regressions_per_chunk': regressions_per_chunk,
        'p_thresh': p_thresh,
        'region': region,
        'window': window,
        'M_annot': M_annot if region != 'all' else None,
        'G_annot': G_annot if region != 'all' else None,
        'methylation_only': methylation_only,
        'update_period': 1,
        'output_dir': output_path if regressions_per_chunk else None,
        'logit_transform': logit_transform,
        'thermal_threshold': thermal_threshold,
        'thermal_wait': thermal_wait,
    }

    logger.info(
        'Running mlr_single with options: {0}',
        {
            k: v
            for k, v in kwargs.items()
            if k not in ['M', 'G', 'C', 'M_annot', 'G_annot']
        },
    )

    if p_thresh is not None:
        logger.info('Using p-value threshold: {0}', p_thresh)

    output = regression_single(**kwargs, **logger)
    if not regressions_per_chunk:
        save_dataframes([output], output_path, [data['output']], **logger)


@cli.group(name='data')
def _data() -> None:
    """Base group for data management."""


@_data.command()
@click.option('-s', '--samples', type=int, prompt=True)
@click.option('-m', '--meth-rows', type=int, prompt=True)
@click.option('-g', '--gene-rows', type=int, prompt=True)
@click.option(
    '-n',
    '--no-annotation',
    is_flag=True,
    show_default=True,
    default=False,
    type=bool,
)
@click.option(
    '--seed',
    type=int,
    default=None,
    show_default=True,
    help=(
        'Seed for reproducible dummy data generation. When omitted the'
        ' global RNG is used and output is non-deterministic.'
    ),
)
@click.pass_context
def dummy(
    ctx: click.Context,
    samples: int,
    meth_rows: int,
    gene_rows: int,
    no_annotation: bool,
    seed: Optional[int],
) -> None:
    """
    Generates dummy data.

    Generates dummy data in the output directory with a given size with
    file names M.csv, G.csv, and C.csv.
    """
    logger: Logger = ctx.obj['logger']
    annotation = not no_annotation

    dataframes = generate_data(
        samples, meth_rows, gene_rows, annotation=annotation, seed=seed
    )
    file_names = [data['meth_file'], data['gene_file'], data['covar_file']]
    data_path = os.path.join(data['root_path'], data['input_dir'])
    save_dataframes(dataframes[:3], data_path, file_names, **logger)
    if annotation:
        file_names = [data['meth_annot'], data['gene_annot']]
        data_path = os.path.join(data['root_path'], data['annot_dir'])
        save_dataframes(
            dataframes[3:],
            data_path,
            file_names,
            sep='\t',
            index=False,
            **logger,
        )

    logger.save()


def abort_if_false(ctx: click.Context, _, value):
    if not value:
        ctx.abort()


@_data.command()
@click.option(
    '-g',
    '--gtp-dir',
    show_default=True,
    default='GTP',
    type=click.Path(file_okay=False),
)
@click.option(
    '-y',
    '--yes',
    is_flag=True,
    callback=abort_if_false,
    expose_value=False,
    prompt='Are you sure you want to overwrite the data directory?',
)
@click.option(
    '--full-covar',
    is_flag=True,
    show_default=True,
    default=False,
    type=bool,
)
@click.pass_context
def gtp(ctx: click.Context, gtp_dir: Any, full_covar: bool) -> None:
    """
    Downloads and extracts GTP data.

    Downloads the methylation, gene expression, and covariate data from
    Grady Trauma Project study. Stores the raw data in gtp-dir. The raw
    data is extracted and processes before being saved in the data
    directory.
    """
    logger: Logger = ctx.obj['logger']

    gtp_path = os.path.join(data['root_path'], gtp_dir)
    data_path = os.path.join(data['root_path'], data['input_dir'])
    file_names = [data['meth_file'], data['gene_file'], data['covar_file']]
    simplify_covar = not full_covar
    save_gtp_data(
        gtp_path,
        data_path,
        file_names,
        simplify_covar=simplify_covar,
        **logger,
    )

    logger.save()



@_data.command()
@click.option(
    '-m',
    '--mesa-dir',
    show_default=True,
    default='MESA',
    type=click.Path(file_okay=False),
)
@click.option(
    '-y',
    '--yes',
    is_flag=True,
    callback=abort_if_false,
    expose_value=False,
    prompt='Are you sure you want to overwrite the data directory?',
)
@click.option(
    '--full-covar',
    is_flag=True,
    show_default=True,
    default=False,
    type=bool,
)
@click.pass_context
def mesa(ctx: click.Context, mesa_dir: Any, full_covar: bool) -> None:
    """
    Downloads and extracts MESA data.

    Downloads the methylation, gene expression, and covariate data from
    MESA study. Stores the raw data in mesa-dir. The raw data is
    extracted and processes before being saved in the data directory.
    """
    logger: Logger = ctx.obj['logger']

    mesa_path = os.path.join(data['root_path'], mesa_dir)
    data_path = os.path.join(data['root_path'], data['input_dir'])
    file_names = [data['meth_file'], data['gene_file'], data['covar_file']]
    simplify_covar = not full_covar
    save_mesa_data(
        mesa_path,
        data_path,
        file_names,
        simplify_covar=simplify_covar,
        **logger,
    )

    logger.save()

@cli.command()
@click.argument(
    'root-dirs',
    nargs=-1,
    type=click.Path(file_okay=False),
)
@click.option(
    '-y',
    '--yes',
    is_flag=True,
    callback=abort_if_false,
    expose_value=False,
    prompt=(
        'Are you sure you want to reset and initialize'
        f' {data["root_path"]}/[root_dir]?'
    ),
)
@click.pass_context
def init(ctx: click.Context, root_dirs: List[str]) -> None:
    """
    Creates and initializes directory.

    Creates root_dir in the root_path. Creates input_dir and output_dir
    in this new directory. Changes directory too this new directory.
    """
    logger: Logger = ctx.obj['logger']

    if not root_dirs:
        root_dir = 'tecpg_testing'
    else:
        root_dir = root_dirs[0]

    path = os.path.join(data['root_path'], root_dir)
    initialize_dir(path, **logger)
    os.mkdir(os.path.join(path, data['input_dir']))
    os.mkdir(os.path.join(path, data['output_dir']))
    os.mkdir(os.path.join(path, data['log_dir']))
    logger.info('Enter the {0} directory to start working.', path)

    log_dir = os.path.join(path, data['log_dir'])
    logger.save(log_dir=log_dir)


@cli.command()
@click.option('-t', '--target-bytes', type=int)
@click.option('-s', '--samples', type=int)
@click.option('-m', '--mt-count', type=int)
@click.option('-g', '--gt-count', type=int)
@click.option('-c', '--covar-count', type=int)
@click.option('-M', '--meth-loci-per-chunk', type=int)
@click.option('-F', '--filtration', type=float)
@click.option('-f', '--full-output', type=bool)
@click.option('-P', '--p-only', type=bool)
@click.option('-p', '--p-filtration', type=bool)
@click.option('-r', '--region-filtration', type=bool)
@click.option('-C', '--cpu', show_default=True, default=False, type=bool)
@click.pass_context
def chunks(
    ctx: click.Context,
    target_bytes: Optional[int],
    samples: Optional[int],
    mt_count: Optional[int],
    gt_count: Optional[int],
    covar_count: Optional[int],
    meth_loci_per_chunk: Optional[int],
    filtration: Optional[float],
    full_output: Optional[bool],
    p_only: Optional[bool],
    p_filtration: Optional[bool],
    region_filtration: Optional[bool],
    cpu: bool,
) -> None:
    '''
    Estimates --gene-loci-per-chunk.

    Estimate optimal --gene-loci-per-chunk to maximize parallelization within
    memory limits given certain variables about the input and the
    system.
    '''
    logger: Logger = ctx.obj['logger']

    if filtration is None and (
        None in (p_filtration, region_filtration)
        or True in (p_filtration, region_filtration)
    ):
        error = (
            'Define --filtration, a float from 0 to 1 for the proportion of'
            ' rows left after region or p-value filtration, if'
            ' region_filtration or p_filtration is included'
        )
        logger.error(error)
        raise ValueError(error)
    if filtration is None:
        filtration = 1
    datum_bytes = torch.ones(1, dtype=DTYPE).element_size()

    if target_bytes is None:
        if cpu or not torch.cuda.is_available():
            target_bytes = psutil.virtual_memory().total * 0.8
            logger.info(
                (
                    'Target memory not supplied. Inferred target of {0} MB of'
                    ' CPU memory (80% of detected)'
                ),
                target_bytes / 1_000_000,
            )
        else:
            target_bytes = torch.cuda.mem_get_info()[0] * 0.8
            logger.info(
                (
                    'Target memory not supplied. Inferred target of {0} MB of'
                    ' CUDA memory (80% of detected)'
                ),
                target_bytes / 1_000_000,
            )
    if None in (samples, mt_count, gt_count, covar_count):
        logger.info(
            'Data size not complete. Inferring from data in working directory.'
        )
        data_path = os.path.join(data['root_path'], data['input_dir'])
        file_names = [
            data['meth_file'],
            data['gene_file'],
            data['covar_file'],
        ]
        dataframes = read_dataframes(data_path, file_names=file_names, **logger)
        M = dataframes[data['meth_file']]
        G = dataframes[data['gene_file']]
        C = dataframes[data['covar_file']]
        if samples is None:
            samples = len(C)
            logger.info('Samples not provided. Inferred {0}.', samples)
        if mt_count is None:
            mt_count = len(M)
            logger.info(
                'Methylation loci count not provided. Inferred {0}.', mt_count
            )
        if gt_count is None:
            gt_count = len(G)
            logger.info(
                'Gene expression loci count not provided. Inferred {0}.',
                gt_count,
            )
        if covar_count is None:
            covar_count = len(C.columns)
            logger.info(
                'Covariate count not provided. Inferred {0}.', covar_count
            )

    if meth_loci_per_chunk:
        mt_count = meth_loci_per_chunk

    logger.info(
        'Estimated loci per chunk for target peak memory usage of {0} bytes:',
        target_bytes,
    )
    if region_filtration is not True:
        constants_bytes = estimate_constants_bytes(
            samples, mt_count, gt_count, covar_count, datum_bytes, False
        )
        logger.info(
            '{0} bytes for constants (without region filtration)',
            constants_bytes,
        )
    if region_filtration is not False:
        constants_bytes = estimate_constants_bytes(
            samples, mt_count, gt_count, covar_count, datum_bytes, True
        )
        logger.info(
            '{0} bytes for constants (with region filtration)', constants_bytes
        )
    logger.info('Full output, p only, p filtration, region filtration')
    for (
        full_output_,
        p_only_,
        p_filtration_,
        region_filtration_,
    ) in itertools.product((False, True), repeat=4):
        if (
            (full_output is not None and full_output != full_output_)
            or (p_only is not None and p_only != p_only_)
            or (p_filtration is not None and p_filtration != p_filtration_)
            or (
                region_filtration is not None
                and region_filtration != region_filtration_
            )
        ):
            continue
        estimate_e = estimate_loci_per_chunk_e_peak(
            target_bytes,
            samples,
            mt_count,
            gt_count,
            covar_count,
            datum_bytes,
            filtration,
            full_output_,
            p_only_,
            p_filtration_,
            region_filtration_,
        )
        estimate_results = estimate_loci_per_chunk_results_peak(
            target_bytes,
            samples,
            mt_count,
            gt_count,
            covar_count,
            datum_bytes,
            filtration,
            full_output_,
            p_only_,
            region_filtration_,
        )

        if estimate_e < estimate_results:
            estimate = estimate_e
            peak = 'Peak memory after scalars and E'
        else:
            estimate = estimate_results
            peak = 'Peak memory after results concatenation'

        if estimate >= gt_count:
            estimate = 'No chunking needed'
        elif estimate < 1:
            estimate = 'Not possible'
        else:
            estimate = f'{math.floor(estimate)} loci per chunk'

        logger.info(
            '{0}, {1}, {2}, {3}: {4}, {5}',
            full_output_,
            p_only_,
            p_filtration_,
            region_filtration_,
            estimate,
            peak,
        )


def start() -> None:
    cli(obj={})
