import itertools
import math
import os
from typing import Any, List, Optional

import click

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


def _auto_chunk_sizes(
    M,
    G,
    C,
    p_only: bool = False,
    full_output: bool = False,
    region: str = 'all',
    logger=None,
):
    """Derive (gene_loci_per_chunk, meth_loci_per_chunk) from data shape and
    available device memory.

    Returns a tuple of ints, or (None, None) when no chunking is required
    (i.e. the data fits in the target memory budget). The caller decides
    what to do with `(None, None)` -- on server-class hosts we treat that
    as "let the inner kernel run un-chunked", matching historical
    behavior when the user supplies neither -g nor -m.

    The target memory budget is 80% of free GPU memory when CUDA is
    available, otherwise 80% of total system RAM. This matches the
    convention used by the existing `tecpg chunks` subcommand.
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

    if _torch.cuda.is_available():
        target_bytes = int(_torch.cuda.mem_get_info()[0] * 0.8)
        target_label = 'CUDA free memory'
    else:
        target_bytes = int(psutil.virtual_memory().available * 0.8)
        target_label = 'system RAM available'

    # The lstsq path used by the server profile produces "p-only-like"
    # output regardless of whether p_only/full_output flags are set,
    # because the inner kernel only realizes the active K columns; we
    # use the more conservative of the E-peak and results-peak estimates.
    # Region filtration shrinks the result; we don't model that here
    # (estimates stay conservative when --cis/--distal/--trans is set).
    estimate_e = estimate_loci_per_chunk_e_peak(
        target_bytes,
        samples,
        mt_count,
        gt_count,
        covar_count,
        datum_bytes,
        filtration=1.0,
        full_output=full_output,
        p_only=p_only,
        p_filtration=False,
        region_filtration=False,
    )
    estimate_results = estimate_loci_per_chunk_results_peak(
        target_bytes,
        samples,
        mt_count,
        gt_count,
        covar_count,
        datum_bytes,
        filtration=1.0,
        full_output=full_output,
        p_only=p_only,
        region_filtration=False,
    )

    estimate = min(estimate_e, estimate_results)

    if logger is not None:
        logger.info(
            'Chunk-size estimator: target={0:.1f} MB ({1}), '
            'estimate_e={2:.0f}, estimate_results={3:.0f}',
            target_bytes / 1_000_000,
            target_label,
            estimate_e,
            estimate_results,
        )

    if estimate >= gt_count:
        # Whole-G fits in budget: no chunking needed.
        return (None, None)
    if estimate < 1:
        # Cannot fit even one gene with the current methylation count;
        # split methylation as well. This is rare on server-class hosts
        # but we fall back to a safe (small) chunk.
        meth_chunk = max(1, mt_count // 4)
        gene_chunk = max(1, gt_count // 4)
        return (gene_chunk, meth_chunk)

    gene_chunk = max(1, int(estimate))
    # Use the full methylation set per chunk by default; the inner loop
    # iterates methylation chunks as the outer loop, so a larger -m means
    # fewer outer iterations and better amortization. Profiling on
    # klabdev showed -m 40000 was beneficial; cap at 40000 to stay
    # within reasonable index-build cost on the producer thread.
    meth_chunk = min(mt_count, 40000)
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
from .helper import default_region_parameter, initialize_dir
from .import_data import read_dataframes, save_dataframes
from .logger import Logger
from .pearson_full import (
    pearson_chunk_save_tensor,
    pearson_chunk_tensor,
    pearson_full_tensor,
)
from .processing import tecpg_mlr_lstsq
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
    '-d', '--debug', is_flag=True, show_default=True, default=False, type=bool
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
        "(e.g. --save-threads, --output-format, -g, -m) always win."
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
    data['output_file'] = click.format_filename(output_file)
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
        save_dataframes([output], output_path, [data['output_file']], **logger)

    logger.save()


@run.command()
@click.option('-g', '--gene-loci-per-chunk', show_default=True, type=int)
@click.option('-m', '--meth-loci-per-chunk', show_default=True, type=int)
@click.option('-p', '--p-thresh', show_default=True, type=float)
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
    type=click.Choice(['manual', 'lstsq', 'lstsq_bootstrap']),
    default='manual',
    show_default=True,
    help=(
        "The MLR computation method to use. 'manual' uses the original"
        " optimized inversion; 'lstsq' uses torch.linalg.lstsq;"
        " 'lstsq_bootstrap' runs empirical bootstrap on specific pairs."
    ),
)
@click.option(
    '--pairs-file',
    type=click.Path(exists=True, dir_okay=False),
    help='Path to a CSV file containing mt_id and gt_id columns. Required for lstsq_bootstrap.',
)
@click.option(
    '--master-parquet',
    type=click.Path(exists=True, dir_okay=False),
    help='Path to the master annotated Parquet file to merge bootstrap results onto. Required for lstsq_bootstrap.',
)
@click.option(
    '--bootstrap-iterations',
    show_default=True,
    default=1000,
    type=int,
    help='Number of resamples for lstsq_bootstrap.',
)
@click.option(
    '--bootstrap-batch-size',
    show_default=True,
    default=10,
    type=int,
    help='Number of pairs to process simultaneously in the bootstrap loop. Note: -g and -m chunks are ignored for bootstraps.',
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
    help='Number of tests to retain in the reservoir buffer (only for lstsq method)',
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
    help='Seed for random subsampling',
)
@click.option(
    '--permute-label-test',
    is_flag=True,
    show_default=True,
    default=False,
    type=bool,
    help='Whether to perform a permutation (Negative Control) test by shuffling subject IDs in G (only for lstsq method)',
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
    help='Call torch.cuda.empty_cache() after every gene chunk. Default is to only empty the cache under memory pressure (> 85% allocated). Useful on memory-constrained GPUs.',
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

    # Auto-derive chunk sizes on server-class hosts when the user did not
    # supply -g / -m. We use the same memory heuristics as the `tecpg
    # chunks` subcommand. On minimum-class hosts we never auto-set chunk
    # sizes -- the user's explicit choice (or no chunking) is preserved.
    host_profile = logger.carry_data.get('host_profile', 'minimum')
    if (
        host_profile == 'server'
        and gene_loci_per_chunk is None
        and meth_loci_per_chunk is None
    ):
        try:
            auto_g, auto_m = _auto_chunk_sizes(
                M, G, C,
                p_only=p_only,
                full_output=full_output,
                region=region,
                logger=logger,
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
                'meth_loci_per_chunk={1} (host_profile=server)',
                gene_loci_per_chunk, meth_loci_per_chunk,
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

    if mlr_method == 'lstsq_bootstrap':
        if not pairs_file or not master_parquet:
            error = '--pairs-file and --master-parquet are required for lstsq_bootstrap.'
            logger.error(error)
            raise click.UsageError(error)

        from .bootstrap import tecpg_mlr_lstsq_bootstrap

        output_file_path = data['output_file']
        if chunking:
            output_file_path = os.path.join(output_path, 'bootstrap_merged.parquet')
        elif output_path and not output_file_path.startswith('/'):
            output_file_path = os.path.join(output_path, output_file_path)

        tecpg_mlr_lstsq_bootstrap(
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
            logger=logger
        )
        return

    if mlr_method == 'lstsq':
        if reservoir_count is None:
            total_comparisons = len(M) * len(G)
            reservoir_count = min(1_000_000, max(1, int(0.01 * total_comparisons)))
        kwargs['reservoir_count'] = reservoir_count
        # Inject output_dir into logger carry_data for reservoir saver
        logger.carry_data['output_dir'] = output_path
        output = tecpg_mlr_lstsq(**kwargs, **logger)
    else:
        if reservoir_count is not None:
            logger.warning('--reservoir-count is only supported for mlr-method lstsq')
        if permute_label_test:
            logger.warning('--permute-label-test is only supported for mlr-method lstsq')
        kwargs.pop('permute_label_test', None)
        if compute_ig or compute_ig_deep:
            logger.warning('Integrated Gradients (--compute-ig/--compute-ig-deep) are only supported for mlr-method lstsq. They will be ignored.')
        kwargs.pop('compute_ig', None)
        kwargs.pop('compute_ig_deep', None)
        kwargs.pop('ig_baseline', None)
        kwargs.pop('ig_covariates_filter', None)
        output = regression_full(**kwargs, **logger)
    if not chunking:
        save_dataframes([output], output_path, [data['output_file']], **logger)


@run.command()
@click.option(
    '-r', '--regressions-per-chunk', show_default=True, default=0, type=int
)
@click.option('-p', '--p-thresh', show_default=True, type=float)
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
        logger.warning('Integrated Gradients (--compute-ig/--compute-ig-deep) are only supported for mlr-method lstsq via mlr command. They will be ignored in mlr_single.')

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

    output = regression_single(**kwargs, **logger)
    if not regressions_per_chunk:
        save_dataframes([output], output_path, [data['output_file']], **logger)


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
@click.pass_context
def dummy(
    ctx: click.Context,
    samples: int,
    meth_rows: int,
    gene_rows: int,
    no_annotation: bool,
) -> None:
    """
    Generates dummy data.

    Generates dummy data in the output directory with a given size with
    file names M.csv, G.csv, and C.csv.
    """
    logger: Logger = ctx.obj['logger']
    annotation = not no_annotation

    dataframes = generate_data(
        samples, meth_rows, gene_rows, annotation=annotation
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
