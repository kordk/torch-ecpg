import math
import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import deque
from typing import Literal, Optional

import numpy
import pandas

# GPU caching-allocator high-water mark. When >75% of total VRAM is held
# by PyTorch's caching allocator, fall back to torch.cuda.empty_cache()
# between gene chunks. Lowered from 0.85 to 0.75 in the CUDA-only
# memory-pressure tuning pass: empty_cache() is a global GPU-side sync
# (expensive), but at >85% allocated PyTorch is already so close to OOM
# that fragmentation routinely tips chunks over before the next chunk is
# scheduled. Triggering one chunk earlier costs a single sync but
# materially reduces OOM risk on memory-constrained GPUs (e.g. the
# 24 GB klabdev L4 path). This branch is only ever hit when
# torch.cuda.memory_allocated() is non-zero, so CPU runs are unaffected.
HIGH_WATER = 0.75
import torch
from colorama import Fore as colors

from .config import DTYPE, get_device
from .gpu_monitor import gpu_guardian, report_thermal_status, throttle_if_needed
from .helper import compute_region_mask, logit_transform_torch, trim_dataframes
from .import_data import initialize_dir, save_dataframe_part
from .logger import Logger, analyze_bottleneck


def qr_peak_bytes(
    mt_count: int,
    nrows: int,
    ncols: int,
    datum_bytes: int,
) -> int:
    """Device bytes provably required by one methylation chunk's QR.

    The dominant per-chunk allocations are the design matrix
    ``X`` of shape ``(mt_count, nrows, K)``, the reduced-mode ``Q``
    returned by ``torch.linalg.qr`` (identical shape, since
    ``nrows >= K``), and ``R`` of shape ``(mt_count, K, K)``, where
    ``K = ncols + 1``.

    This is a deliberate *lower* bound on the chunk's true peak: it
    counts only tensors that are unconditionally allocated, and
    excludes the gene-loop working set, cuSOLVER workspace, and
    allocator slack. A lower bound cannot abort a run that would
    otherwise have succeeded on account of over-estimation.
    """
    K = ncols + 1
    return datum_bytes * mt_count * (2 * nrows * K + K * K)


def _cuda_free_bytes(device: 'torch.device') -> Optional[int]:
    """Free device bytes, or ``None`` when the device is not CUDA."""
    if device is None or getattr(device, 'type', None) != 'cuda':
        return None
    return int(torch.cuda.mem_get_info(device)[0])


def check_chunk_headroom(
    mt_count: int,
    nrows: int,
    ncols: int,
    datum_bytes: int,
    device: 'torch.device',
    meth_chunk_index: int,
    meth_chunk_count: int,
    logger,
    free_bytes_fn=_cuda_free_bytes,
) -> None:
    """Fail fast when the next methylation chunk cannot fit.

    No-op on non-CUDA devices. Logs the measured headroom at every
    boundary so the per-chunk deficit trend is visible even on runs
    that succeed. Set ``TECPG_HEADROOM_WARN_ONLY=1`` to downgrade the
    failure to a warning and continue -- which is how the deficit is
    collected across more than one chunk boundary.
    """
    free_bytes = free_bytes_fn(device)
    if free_bytes is None:
        return

    required = qr_peak_bytes(mt_count, nrows, ncols, datum_bytes)
    free_mb = free_bytes / (1024 * 1024)
    required_mb = required / (1024 * 1024)
    headroom_mb = free_mb - required_mb

    logger.info(
        f"{meth_chunk_index + 1}/{meth_chunk_count} mt_count={mt_count} "
        f"required={required_mb:.2f}MB free={free_mb:.2f}MB "
        f"headroom={headroom_mb:.2f}MB"
    )

    if free_bytes >= required:
        return

    deficit = required - free_bytes
    message = (
        f"tecpg: insufficient GPU memory for methylation chunk "
        f"{meth_chunk_index + 1}/{meth_chunk_count}. "
        f"mt_count={mt_count}, required={required}, free={free_bytes}, "
        f"deficit={deficit}."
    )

    if os.environ.get('TECPG_HEADROOM_WARN_ONLY') == '1':
        logger.warning(f"{message} (Guard bypassed via TECPG_HEADROOM_WARN_ONLY=1)")
        return

    raise RuntimeError(message)


def create_normal_p(device: torch.device, dtype: torch.dtype):
    scalar = (
        torch.tensor(2, device=device, dtype=dtype).sqrt().reciprocal().neg()
    )

    def prob(value: torch.Tensor) -> torch.Tensor:
        return torch.erf(scalar * value.abs()) + 1

    return prob


def tecpg_mlr_qr(
    M: pandas.DataFrame,
    G: pandas.DataFrame,
    C: pandas.DataFrame,
    M_annot: Optional[pandas.DataFrame] = None,
    G_annot: Optional[pandas.DataFrame] = None,
    region: Literal['all', 'cis', 'distal', 'trans'] = 'all',
    window_base: Optional[int] = None,
    downstream: Optional[int] = None,
    upstream: Optional[int] = None,
    gene_loci_per_chunk: Optional[int] = None,
    meth_loci_per_chunk: Optional[int] = None,
    p_thresh: Optional[float] = 0.00001,
    output_dir: Optional[str] = None,
    methylation_only: bool = True,
    p_only: bool = False,
    logit_transform: bool = False,
    thermal_threshold: int = 80,
    thermal_wait: int = 30,
    file_format: str = '{meth_chunk}-{gene_chunk}.csv',
    reservoir_count: Optional[int] = None,
    subsample_mt_count: Optional[int] = None,
    subsample_g_count: Optional[int] = None,
    seed: int = 42,
    permute_label_test: bool = False,
    compute_ig: bool = False,
    compute_ig_deep: bool = False,
    compute_influence: bool = False,
    ig_baseline: str = 'mean',
    ig_covariates_filter: Optional[list] | str = None,
    prefetch_chunks: int = 0,
    aggressive_gc: bool = False,
    output_format: str = 'csv',
    *,
    logger: Logger = Logger(),
) -> Optional[pandas.DataFrame]:
    '''
    Calculates the multiple linear regression of the input dataframes M,
    G, and C using torch.linalg.qr.
    '''
    chunking = (
        gene_loci_per_chunk is not None or meth_loci_per_chunk is not None
    )

    # Pool selection: parquet writes via pyarrow release the GIL and
    # benefit from a thread pool (no per-chunk DataFrame pickling across
    # `spawn`, no spawn warm-up cost). CSV writes via pandas.to_csv hold
    # the GIL, so we keep the historical process pool there.
    max_workers = logger.carry_data.get('save_threads', 2)
    if (output_format or 'csv').lower() == 'parquet':
        pool_cm = ThreadPoolExecutor(max_workers=max_workers)
    else:
        ctx = multiprocessing.get_context('spawn')
        pool_cm = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
    with pool_cm as pool:
        if isinstance(pool, ProcessPoolExecutor):
            # Warm up worker processes so per-chunk pool.submit doesn't
            # pay spawn latency on the first save.
            _ = list(pool.map(int, range(max_workers)))
        return _tecpg_mlr_qr_inner(
            M, G, C, M_annot, G_annot, region, window_base, downstream, upstream,
            gene_loci_per_chunk, meth_loci_per_chunk, p_thresh, output_dir,
            methylation_only, p_only, logit_transform, thermal_threshold, thermal_wait,
            file_format, reservoir_count, subsample_mt_count, subsample_g_count, seed,
            permute_label_test, compute_ig, compute_ig_deep, compute_influence, ig_baseline,
            ig_covariates_filter, prefetch_chunks, aggressive_gc, pool, max_workers,
            output_format=output_format, logger=logger, chunking=chunking
        )

def _tecpg_mlr_qr_inner(
    M: pandas.DataFrame,
    G: pandas.DataFrame,
    C: pandas.DataFrame,
    M_annot: Optional[pandas.DataFrame] = None,
    G_annot: Optional[pandas.DataFrame] = None,
    region: Literal['all', 'cis', 'distal', 'trans'] = 'all',
    window_base: Optional[int] = None,
    downstream: Optional[int] = None,
    upstream: Optional[int] = None,
    gene_loci_per_chunk: Optional[int] = None,
    meth_loci_per_chunk: Optional[int] = None,
    p_thresh: Optional[float] = 0.00001,
    output_dir: Optional[str] = None,
    methylation_only: bool = True,
    p_only: bool = False,
    logit_transform: bool = False,
    thermal_threshold: int = 80,
    thermal_wait: int = 30,
    file_format: str = '{meth_chunk}-{gene_chunk}.csv',
    reservoir_count: Optional[int] = None,
    subsample_mt_count: Optional[int] = None,
    subsample_g_count: Optional[int] = None,
    seed: int = 42,
    permute_label_test: bool = False,
    compute_ig: bool = False,
    compute_ig_deep: bool = False,
    compute_influence: bool = False,
    ig_baseline: str = 'mean',
    ig_covariates_filter: Optional[list] | str = None,
    prefetch_chunks: int = 0,
    aggressive_gc: bool = False,
    pool=None,
    max_workers: int = 2,
    *,
    output_format: str = 'csv',
    logger: Logger = Logger(),
    chunking: bool = False,
) -> Optional[pandas.DataFrame]:

    import psutil
    logger.print_startup_banner(
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda if torch.cuda.is_available() else 'N/A',
        device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        vram_gb=torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
        compute_cap=torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A',
        mt_count=len(M),
        gene_loci_per_chunk=gene_loci_per_chunk,
        meth_loci_per_chunk=meth_loci_per_chunk,
        dtype=str(DTYPE),
        workers=max_workers,
        save_threads_effective=max_workers if max_workers > 0 else os.cpu_count(),
        prefetch_chunks_effective=prefetch_chunks,
        blas_threads_effective=os.environ.get('OMP_NUM_THREADS', 'N/A'),
        torch_num_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        cpu_count_logical=os.cpu_count(),
        cpu_count_physical=psutil.cpu_count(logical=False),
        **logger.resource_check()
    )

    # Detect errors in the input values
    if (output_dir is None) != (not chunking):
        error = 'Output dir and chunk size must be defined together.'
        logger.error(error)
        raise ValueError(error)
    if region not in ['all', 'cis', 'distal', 'trans']:
        error = f'Region {region} not valid. Use all, cis, distal, or trans.'
        logger.error(error)
        raise ValueError(error)
    if region != 'all' and (G_annot is None or M_annot is None):
        error = (
            f'Missing M or G annotation files using region filtration {region}'
        )
        logger.error(error)
        raise ValueError(error)
    if region in ['cis', 'distal'] and (
        window_base is None or downstream is None or upstream is None
    ):
        error = (
            f'Region filtration {region} requires window_base, downstream, and'
            ' upstream not to be None'
        )
        logger.error(error)
        raise ValueError(error)

    # Prepare annotation tensors if region filtration is used
    if region != 'all':
        from .chrom import canonicalize_chrom

        logger.info('Initializing region filtration')
        G_loci_before = len(G.index)
        G_annot = G_annot.drop(columns=['chromEnd', 'score']).reindex(G.index)
        import pandas as pd
        G_annot['chrom'] = canonicalize_chrom(G_annot['chrom'])
        G_annot['strand'] = pd.to_numeric(G_annot['strand'].replace({'+': 1, '-': -1}), errors='coerce')
        G_annot = G_annot.dropna()
        logger.info(
            'Drop site processing.region_filtration[G_annot]: dropped gene '
            'expression loci with missing/unmappable annotation '
            '(reindex + dropna): {0} -> {1} ({2} dropped)',
            G_loci_before, len(G_annot), G_loci_before - len(G_annot),
        )
        M_loci_before = len(M.index)
        M_annot = M_annot.drop(columns=['chromEnd', 'score', 'strand']).reindex(M.index)
        M_annot['chrom'] = canonicalize_chrom(M_annot['chrom'])
        M_annot = M_annot.dropna()
        logger.info(
            'Drop site processing.region_filtration[M_annot]: dropped '
            'methylation loci with missing/unmappable annotation '
            '(reindex + dropna): {0} -> {1} ({2} dropped)',
            M_loci_before, len(M_annot), M_loci_before - len(M_annot),
        )

        trim_dataframes([G_annot, G], **logger)
        trim_dataframes([M_annot, M], **logger)

        G_chrom, G_pos, G_strand = G_annot.to_numpy().T.astype(int)
        M_chrom, M_pos = M_annot.to_numpy().T.astype(int)

        G_chrom_t = torch.tensor(G_chrom, device=get_device(**logger), dtype=torch.int8)
        G_pos_t = torch.tensor(G_pos, device=get_device(**logger), dtype=torch.int32)
        G_strand_t = torch.tensor(G_strand, device=get_device(**logger), dtype=torch.int8)

    # Subsampling logic
    if subsample_mt_count is not None or subsample_g_count is not None:
        logger.info('Initializing random subsampling with seed {0}', seed)
        rng = numpy.random.default_rng(seed)

        if subsample_mt_count is not None:
            if subsample_mt_count > len(M):
                logger.warning('Requested mt_count {0} > available {1}. Defaulting to full dataset.', subsample_mt_count, len(M))
            else:
                indices = rng.choice(len(M), size=subsample_mt_count, replace=False)
                indices.sort()
                M = M.iloc[indices]
                if M_annot is not None:
                    M_annot = M_annot.iloc[indices]
                    M_chrom = M_annot.to_numpy().T[0].astype(int)
                    M_pos = M_annot.to_numpy().T[1].astype(int)

        if subsample_g_count is not None:
            if subsample_g_count > len(G):
                logger.warning('Requested g_count {0} > available {1}. Defaulting to full dataset.', subsample_g_count, len(G))
            else:
                indices = rng.choice(len(G), size=subsample_g_count, replace=False)
                indices.sort()
                G = G.iloc[indices]
                if G_annot is not None:
                    G_annot = G_annot.iloc[indices]
                    G_chrom = G_annot.to_numpy().T[0].astype(int)
                    G_pos = G_annot.to_numpy().T[1].astype(int)
                    G_strand = G_annot.to_numpy().T[2].astype(int)
                    G_chrom_t = torch.tensor(G_chrom, device=get_device(**logger), dtype=torch.int8)
                    G_pos_t = torch.tensor(G_pos, device=get_device(**logger), dtype=torch.int32)
                    G_strand_t = torch.tensor(G_strand, device=get_device(**logger), dtype=torch.int8)

        logger.info("Subsampling active: Testing {0} CpGs x {1} Genes = {2} total tests", len(M), len(G), len(M) * len(G))
    elif region != 'all':
        logger.info("After region filtering: Testing {0} CpGs x {1} Genes = {2} total tests", len(M), len(G), len(M) * len(G))

    if compute_influence:
        logger.info(
            'Influence diagnostic enabled: emitting mt_h_max (per-CpG max sample '
            'leverage, max_i ||Q_i||^2). Deletion point = max-leverage subject. '
            'No residual tensor is materialized.'
        )

    # Initializes some constants
    logger.info(
        'Running tecpg_mlr_qr with options: {0}',
        {
            k: v
            for k, v in locals().items()
            if k not in ['M', 'G', 'C', 'M_annot', 'G_annot', 'logger']
        },
    )
    logger.info('Final count prior to analysis: {0} genes and {1} methylation loci.', len(G), len(M))
    logger.info('Initializing regression variables (qr)')
    device = get_device(**logger)
    dtype = DTYPE
    if meth_loci_per_chunk is not None:
        meth_chunk_count = math.ceil(len(M) / meth_loci_per_chunk)
    else:
        meth_chunk_count = 1

    nrows, ncols = C.shape[0], C.shape[1] + 1
    G_np = G.to_numpy()

    if permute_label_test:
        logger.info('Permuting label test active: Shuffling subject IDs in G to create negative control')
        rng_permute = numpy.random.default_rng(seed)
        permutation = rng_permute.permutation(len(G.columns))
        G_np = G_np[:, permutation]

    gt_count = len(G)
    gt_site_names = numpy.array(G.index.values)
    if gt_site_names.dtype == object:
        logger.info('Gene (G) index array was inferred as object. Enforcing string type to prevent serialization errors.')
        gt_site_names = gt_site_names.astype(str)

    df = nrows - ncols - 1
    logger.info(
        'Statistical Power Audit: df = {0} (calculated as {1} subjects - {2} covariates - 1 methylation - 1 intercept)',
        df,
        nrows,
        C.shape[1],
    )
    normal_p = create_normal_p(device, dtype)

    if gene_loci_per_chunk is not None:
        gene_chunk_count = math.ceil(len(G) / gene_loci_per_chunk)
    else:
        gene_chunk_count = 1

    if chunking:
        logger.info('Initializing output directory')
        initialize_dir(output_dir, **logger)

    if ig_baseline not in ['mean', 'zero']:
        error = f"Unsupported ig_baseline '{ig_baseline}'. Must be 'mean' or 'zero'."
        logger.error(error)
        raise ValueError(error)

    # Determines the column names for the output dataframe
    index_names = ['gt_id', 'mt_id']
    categories = (
        ['mt']
        if methylation_only
        else (['const', 'mt'] + C.columns.to_list())
    )
    if p_only:
        columns = [column + '_p' for column in categories]
    else:
        suffixes = ['_est', '_err', '_t', '_p']
        columns = [
            column + suffix for suffix in suffixes for column in categories
        ]

    # Add _ig columns if IG is computed
    ig_col_indices = []
    if compute_ig or compute_ig_deep:
        # Intercept gets excluded from IG outputs (index 0). Meth is index 1. Covariates start at index 2.
        ig_categories = ['mt']
        ig_col_indices = [0] # Relative index in the K-1 slice (without intercept)

        if ig_covariates_filter == 'all':
            ig_categories.extend(C.columns.to_list())
            ig_col_indices.extend(range(1, len(C.columns) + 1))
        elif isinstance(ig_covariates_filter, list):
            for covar in ig_covariates_filter:
                ig_categories.append(covar)
                ig_col_indices.append(C.columns.get_loc(covar) + 1)

        ig_columns = [col + '_ig' for col in ig_categories]
        columns.extend(ig_columns)

    if compute_influence:
        columns.append('mt_h_max')

    # Create covariate tensor
    Ct_base: torch.Tensor = torch.tensor(
        C.to_numpy(), device=device, dtype=dtype
    ).unsqueeze(0)

    # Initialize variables for use in the regression calculation loop
    end_index = 0
    results = []
    filtration = True
    output_sizes = []
    if region != 'all':
        region_indices_list = []
    if p_thresh is None:
        p_indices_list = None
        if region == 'all':
            filtration = False
    else:
        p_indices_list = []

    # Variables for Reservoir Sampling
    do_reservoir = reservoir_count is not None and reservoir_count > 0
    if do_reservoir:
        expected_items = (
            len(M) * len(G)
            if region == 'all'
            else "unknown (region filtration applied)"
        )
        logger.info(
            f"Initializing reservoir sampling. Will retain up to {reservoir_count} "
            f"results out of an expected {expected_items} total items. "
            f"Reservoir RNG seeded with {seed}."
        )
    reservoir_buffer = [] # Store tuple of (results_tensor, gt_sites, mt_sites)
    reservoir_processed = 0
    # Dedicated generator for reservoir draws. Seeded from the run's seed so
    # the sample is reproducible, and kept separate from the global torch RNG
    # so that seeding it here neither perturbs nor is perturbed by any other
    # consumer of torch randomness in the same process.
    reservoir_rng = torch.Generator(device=device)
    reservoir_rng.manual_seed(seed)

    # Create methylation chunk (mc_) and chunk saving (inner_) logger
    mc_logger = logger.alias()
    mc_logger.info_color = colors.GREEN
    inner_logger = mc_logger.alias()

    methylation_loop_start_time = time.time()
    methylation_last_chunk_end_time = methylation_loop_start_time
    methylation_chunk_times = []

    # Use the process pool
    futures = deque()
    # Save-queue depth: how many save tasks can be in flight before the GPU
    # producer thread is blocked. carry_data is set in cli; default to the
    # historical (max_workers + 1) for direct programmatic callers.
    save_queue_depth = max(
        max_workers + 1,
        int(logger.carry_data.get('save_queue_depth', max_workers + 1)),
    )

    # Run summary diagnostics
    total_chunks_saved = 0
    total_bytes_written = 0
    total_tests_evaluated = 0
    total_tests_passed_filter = 0
    run_metrics = {
        'prep_ms': [], 'h2d_ms': [], 'compute_ms': [], 'd2h_ms': [], 'post_ms': [],
        'write_enqueue_ms': [], 'gpu_idle_between_chunks_ms': []
    }
    last_chunk_end_time = None

    with gpu_guardian(logger, thermal_threshold) as gpu_monitor:
        # Loop for methylation chunks or ran once with index 0 if no
        # methylation chunking
        for meth_chunk_index in range(meth_chunk_count):
            throttle_if_needed(gpu_monitor, thermal_threshold, thermal_wait, logger)
            report_thermal_status(gpu_monitor, thermal_threshold, logger)

            logger.memory_check('tecpg_mlr_qr')
            # Log methylation chunk index
            logger.info(
                'STARTING METHYLATION CHUNK {0}/{1}',
                meth_chunk_index + 1,
                meth_chunk_count,
            )
            mc_logger.info_template = '[INFO] [tecpg_mlr_qr] Chunk ' + str(meth_chunk_index + 1) + ': {message}'
            inner_logger.info_template = '[INFO] [tecpg_mlr_qr] Chunk ' + str(meth_chunk_index + 1) + ': {message}'
            mc_logger.debug_template = '[DEBUG] [tecpg_mlr_qr] Chunk ' + str(meth_chunk_index + 1) + ': {message}'
            inner_logger.debug_template = '[DEBUG] [tecpg_mlr_qr] Chunk ' + str(meth_chunk_index + 1) + ': {message}'
            mc_logger.current_count = 0
            inner_logger.current_count = 0
            mc_logger.start_timer('info', 'Running tecpg_mlr_qr...')

            # Slice M into M_chunk or copy for no methylation chunking
            if meth_loci_per_chunk is not None:
                start_index = end_index
                end_index = (meth_chunk_index + 1) * meth_loci_per_chunk
                M_chunk = M[start_index:end_index]
            else:
                M_chunk = M
                start_index = 0
                end_index = len(M)

            mt_count = len(M_chunk)
            Ct = Ct_base.expand(mt_count, -1, -1)
            mt_site_names = numpy.array(M_chunk.index.values)
            if mt_site_names.dtype == object:
                logger.info('Methylation (M) chunk index array was inferred as object. Enforcing string type to prevent serialization errors.')
                mt_site_names = mt_site_names.astype(str)

            if region == 'all' and p_thresh is None:
                # If no filtration, output size is constant per gene chunk
                pass # Calculated later


            # Create methylation loci chromosome and position tensors
            # for the current chunk
            if region != 'all':
                if meth_loci_per_chunk is None:
                    M_chrom_t = torch.tensor(
                        M_chrom, device=device, dtype=torch.int8
                    )
                    M_pos_t = torch.tensor(
                        M_pos, device=device, dtype=torch.int32
                    )
                else:
                    M_chrom_t = torch.tensor(
                        M_chrom[start_index:end_index],
                        device=device,
                        dtype=torch.int8,
                    )
                    M_pos_t = torch.tensor(
                        M_pos[start_index:end_index],
                        device=device,
                        dtype=torch.int32,
                    )

            check_chunk_headroom(
                mt_count,
                nrows,
                ncols,
                torch.ones(1, dtype=dtype).element_size(),
                device,
                meth_chunk_index,
                meth_chunk_count,
                mc_logger,
            )

            # Calculate design matrix X for the current methylation chunk
            Mt: torch.Tensor = torch.tensor(
                M_chunk.to_numpy(), device=device, dtype=dtype
            ).unsqueeze(2)

            if logit_transform:
                Mt = logit_transform_torch(Mt, logger=mc_logger)

            ones = torch.ones((mt_count, nrows, 1), device=device, dtype=dtype)
            X: torch.Tensor = torch.cat((ones, Mt, Ct), 2) # (M, S, K)
            del Mt, ones

            mc_logger.memory_check('tecpg_mlr_qr - peak')

            if compute_ig or compute_ig_deep:
                if ig_baseline == 'mean':
                    X_baseline = X.mean(dim=1, keepdim=True)
                else:
                    X_baseline = torch.zeros_like(X[:, 0:1, :])

            if compute_ig:
                X_diff_mean = (X - X_baseline).abs().mean(dim=1)  # Shape: (M, K)
                if methylation_only:
                    X_diff_mean = X_diff_mean[:, 1:]
                else:
                    X_diff_mean = X_diff_mean[:, 1:]

            # Pre-calculate diagonal of (X^T X)^-1 for Standard Error using QR decomposition
            # X = QR => X^T X = R^T R. (X^T X)^-1 = (R^T R)^-1 = R^-1 (R^-1)^T.
            # We need the diagonal elements.
            Q, R = torch.linalg.qr(X, mode='reduced')

            if compute_influence:
                # Per-sample leverage h_i = ||Q_i||^2 (row norms of Q, reduce over K),
                # then max over samples S -> per-CpG (M_chunk,). Gene-independent.
                h_max = (Q * Q).sum(dim=2).amax(dim=1)

            # K is ncols + 1 (because X is cat(ones, Mt, Ct)). Mt adds 1 column. Ct adds ncols - 1 (since ncols is C.shape[1] + 1)
            # Actually, C.shape[1] is number of covariates.
            # X = [ones(1), Mt(1), Ct(C.shape[1])]
            # So K = 1 + 1 + C.shape[1] = C.shape[1] + 2
            # Notice above: `ncols = C.shape[1] + 1`, which means `ncols` is missing the `Mt` column!
            # So `K = X.shape[2]`
            K = X.shape[2]

            # PR1/A6: free the (M, S, K) design tensor as soon as QR is done.
            # X is only re-read inside the gene loop by the Deep IG path
            # (per-hit slices `X[m_idx]` / `X_baseline[m_idx]`); analytical IG
            # has already consumed it via X_diff_mean above. Keeping X alive
            # across the whole gene loop otherwise wastes (M * S * K) * dtype
            # bytes on every chunk.
            if not compute_ig_deep:
                del X

            # Calculate R_inv. R is upper triangular.
            R_inv = torch.linalg.solve_triangular(
                R,
                torch.eye(K, device=device, dtype=dtype).expand(mt_count, -1, -1),
                upper=True
            )
            # (R^-1)(R^-1)^T diagonal is sum of squares of rows of R^-1.
            # R_inv is (M, K, K).
            # We want diag((R_inv) @ (R_inv).mT)
            # Element [i, j, j] = sum_k R_inv[i, j, k] * R_inv[i, j, k]
            XtXi_diag_sqrt = (R_inv.pow(2).sum(dim=2)).sqrt()
            del R

            # Display amount of total memory occupied by the constants
            if allocated_memory := torch.cuda.memory_allocated():
                device_properties: torch.cuda._CudaDeviceProperties = (
                    torch.cuda.get_device_properties(0)
                )
                total_memory: int = device_properties.total_memory
                # empty_cache() is a global GPU-side sync. Only invoke it
                # under memory pressure (or when the user explicitly asked
                # for aggressive GC), matching the gated paths elsewhere.
                if aggressive_gc or (total_memory and torch.cuda.memory_allocated() / total_memory > HIGH_WATER):
                    torch.cuda.empty_cache()
                mc_logger.info(
                    (
                        'CUDA device memory: {0} MB allocated by constants out'
                        ' of {1} MB total'
                    ),
                    allocated_memory / 1_000_000,
                    total_memory / 1_000_000,
                )

            # Loop over gene chunks
            inner_logger.start_timer('info', 'Calculating regression (qr)...')

            gene_end_index = 0

            # Setup prefetch executor
            prefetch_executor = None
            if prefetch_chunks > 0:
                prefetch_executor = ThreadPoolExecutor(max_workers=prefetch_chunks)

            prefetch_queue = deque()

            def prep_chunk(g_start, g_end):
                G_chunk_np = G_np[g_start:g_end]
                # Pinned memory for fast H2D transfer
                Y_host = torch.tensor(G_chunk_np.T, dtype=dtype).pin_memory()
                # Non-blocking transfer to device
                Y_dev = Y_host.to(device, non_blocking=True)
                return Y_dev

            for gene_chunk_index in range(gene_chunk_count):
                prof_t0 = prof_t1 = prof_t2 = prof_t3 = prof_t4 = prof_t5 = time.perf_counter()
                prof_prep_time = prof_h2d_time = prof_gpu_time = prof_d2h_time = prof_post_time = prof_write_time = 0.0

                # Prune ready futures for accurate save queue fill metric.
                # NOTE: do not reuse the name `save_queue_depth` here -- that
                # variable is the configured back-pressure cap consumed by the
                # `while len(futures) >= save_queue_depth` loops below. Use a
                # separate `save_queue_fill` for the per-chunk profile log.
                while futures and futures[0].done():
                    futures.popleft().result()
                save_queue_fill = len(futures)

                # Calculate prefetch fill
                prefetch_fill = len(prefetch_queue) if prefetch_executor else 0

                prof_t0 = time.perf_counter()
                gpu_idle_between_chunks_ms = (prof_t0 - last_chunk_end_time) * 1000 if last_chunk_end_time else 0.0

                gene_start_index = gene_end_index
                if gene_loci_per_chunk is not None:
                    gene_end_index = min((gene_chunk_index + 1) * gene_loci_per_chunk, len(G))
                else:
                    gene_end_index = len(G)

                chunk_len = gene_end_index - gene_start_index

                # Enqueue prefetch tasks to fill the pipeline up to prefetch_chunks
                if prefetch_executor:
                    # Current chunk enqueued if not already there
                    if len(prefetch_queue) == 0:
                        prefetch_queue.append(prefetch_executor.submit(prep_chunk, gene_start_index, gene_end_index))

                    # Future chunks
                    for offset in range(1, prefetch_chunks + 1):
                        lookahead_index = gene_chunk_index + offset
                        if lookahead_index < gene_chunk_count and len(prefetch_queue) <= offset:
                            if gene_loci_per_chunk is not None:
                                l_start = lookahead_index * gene_loci_per_chunk
                                l_end = min((lookahead_index + 1) * gene_loci_per_chunk, len(G))
                            else:
                                l_start = 0
                                l_end = len(G)
                            prefetch_queue.append(prefetch_executor.submit(prep_chunk, l_start, l_end))

                if inner_logger.carry_data.get('profile') and torch.cuda.is_available():
                    torch.cuda.synchronize()
                prof_t1 = time.perf_counter()
                prof_prep_time += (prof_t1 - prof_t0)

                # Transpose gene expression matrix to serve as target matrix Y
                # G_chunk_np is (G_chunk, S). Transpose to (S, G_chunk).
                if prefetch_executor:
                    Y = prefetch_queue.popleft().result()
                else:
                    G_chunk_np = G_np[gene_start_index:gene_end_index]
                    Y = torch.tensor(G_chunk_np.T, device=device, dtype=dtype) # (S, G_chunk)

                if inner_logger.carry_data.get('profile') and torch.cuda.is_available():
                    torch.cuda.synchronize()
                prof_t2 = time.perf_counter()
                prof_h2d_time += (prof_t2 - prof_t1)

                # Solve reusing Q and R_inv from QR decomposition
                # Q is (M_chunk, S, K). Y is (S, G_chunk).
                # To avoid materializing Y_expanded, do QtY = torch.einsum('msk,sg->mkg', Q, Y)
                QtY = torch.einsum('msk,sg->mkg', Q, Y)
                inner_logger.memory_check('tecpg_mlr_qr - QtY computed')

                # Coefficients B
                B = R_inv.matmul(QtY) # (M_chunk, K, G_chunk)
                inner_logger.memory_check('tecpg_mlr_qr - solve (QR reuse)')

                # Calculate Residuals (RSS) algebraically without materializing E
                # ||Y - XB||^2 = ||Y||^2 - ||Q^T Y||^2
                Y_norm_sq = (Y * Y).sum(dim=0) # (G_chunk,)
                QtY_norm_sq = (QtY * QtY).sum(dim=1) # (M_chunk, G_chunk)

                # clamp_min(0) guards against float32 cancellation producing small negatives
                RSS = (Y_norm_sq.unsqueeze(0) - QtY_norm_sq).clamp_min(0) # (M_chunk, G_chunk)
                inner_logger.memory_check('tecpg_mlr_qr - RSS (no E)')

                # Sigma = sqrt(RSS / df)
                # Standard Errors S = XtXi_diag_sqrt * Sigma
                # XtXi_diag_sqrt is (M, K). RSS is (M, G).
                # We need S of shape (M, K, G).
                # Expand XtXi_diag_sqrt to (M, K, 1).
                # Expand RSS to (M, 1, G).

                Sigma = (RSS / df).sqrt().unsqueeze(1) # (M, 1, G)
                S = XtXi_diag_sqrt.unsqueeze(2) * Sigma # (M, K, G)

                del QtY, Y_norm_sq, QtY_norm_sq, RSS, Sigma, Y

                # Calculate T and P
                # B is (M, K, G). S is (M, K, G).

                # Reshape to (M * G, K) for easier filtering if we were to filter now?
                # regression_full output format requires (M, K) per regression.
                # Here we have K coeffs per regression.
                # We need to align with regression_full output.
                # regression_full output columns: [mt_est, mt_err, mt_t, mt_p, ...]
                # It flattens the result.
                # Let's verify output format.
                # regression_full produces results list of tensors of shape (N, 4*K) or something?
                # No, results.append(torch.cat((B, S, T, P), dim=1)).
                # B, S, T, P in regression_full are (N_subset, K).
                # So concatenated is (N_subset, 4*K).
                # Here we have B (M, K, G).
                # We need to permute to (M, G, K).
                B = B.permute(0, 2, 1) # (M, G, K)
                S = S.permute(0, 2, 1)

                # PR1/A1-A2: when only the methylation coefficient is kept and
                # neither analytical nor deep IG needs the other K-1 columns,
                # slice B and S to the meth column (index 1) before forming
                # T and P. This is bit-equivalent — division and normal_p are
                # element-wise on K — but builds the per-CpG S/T/P tensors at
                # 1/K of the memory and skips work on columns the late
                # `[:, 1:2]` slice would discard anyway.
                early_meth_slice = (
                    methylation_only
                    and not compute_ig
                    and not compute_ig_deep
                )
                if early_meth_slice:
                    B = B[:, :, 1:2]
                    S = S[:, :, 1:2]

                T = B / S
                P = normal_p(T)
                inner_logger.memory_check('tecpg_mlr_qr - pvals')

                # Now we have tensors of shape (M, G, K).

                if compute_ig:
                    # B shape: (M, G, K). X_diff_mean shape: (M, K-1)
                    # We compute Analytical IG = X_diff_mean * |W|
                    # Skip the intercept index 0 of B
                    IG_analytical = X_diff_mean.unsqueeze(1) * B[:, :, 1:].abs()  # Shape: (M, G, K-1)

                    # Keep only the requested covariates
                    IG_analytical = IG_analytical[:, :, ig_col_indices]

                # We need to flatten to (M*G, K) to apply filters efficiently?
                # Or apply masks on the (M, G) grid.

                if region != 'all':
                    # Create mask
                    # G_chrom_t etc are full length. Slice for current chunk.
                    G_chrom_chunk = G_chrom_t[gene_start_index:gene_end_index]
                    G_pos_chunk = G_pos_t[gene_start_index:gene_end_index]
                    G_strand_chunk = G_strand_t[gene_start_index:gene_end_index]

                    # Compute mask (M, G)
                    # Broadcast: (M, 1) vs (1, G) -> (M, G)
                    region_mask = compute_region_mask(
                        region,
                        M_chrom_t.unsqueeze(1),
                        M_pos_t.unsqueeze(1),
                        G_chrom_chunk.unsqueeze(0),
                        G_pos_chunk.unsqueeze(0),
                        G_strand_chunk.unsqueeze(0),
                        window_base=window_base,
                        upstream=upstream,
                        downstream=downstream,
                    )

                # Flatten the tensors to (M*G, K)
                # We want the order to match regression_full:
                # regression_full loops M (outer), then G (inner).
                # results.append() inside G loop.
                # So effective order is M1_G1, M1_G2... NO.
                # regression_full:
                # Loop Meth Chunks (M_chunk)
                #   Loop Gene (g)
                #     results.append(mask(M_chunk) against g) -> This gives M_sub * 1.
                #     So for one gene g, we have multiple meth sites.
                #     Order in results list is: [M_sub_g1, M_sub_g2, ...]
                #     Where M_sub_g1 are meth sites for g1.
                #     So the primary index variation is Meth, then Gene?
                #     No, "results.append(torch.cat((B, S, T, P), dim=1))" happens inside G loop.
                #     So for G1, we add a block of M results.
                #     So the dataframe index is (Gene, Meth).
                #     `index_names = ['gt_id', 'mt_id']`.
                #     Yes, index is (Gene, Meth).

                # So we need to flatten (M, G, K) to (M*G, K) such that G varies slowest?
                # No, if index is (G, M), we want G to be the outer block.
                # B is (M, G, K).
                # Permute to (G, M, K).
                # Reshape to (G*M, K).
                num_coeffs = B.shape[-1]
                B = B.permute(1, 0, 2).reshape(-1, num_coeffs)
                S = S.permute(1, 0, 2).reshape(-1, num_coeffs)
                T = T.permute(1, 0, 2).reshape(-1, num_coeffs)
                P = P.permute(1, 0, 2).reshape(-1, num_coeffs)

                if compute_ig:
                    IG_analytical = IG_analytical.permute(1, 0, 2).reshape(-1, len(ig_col_indices))

                if region != 'all':
                    # mask is (M, G).
                    # Permute to (G, M).
                    # Reshape to (G*M).
                    region_mask = region_mask.permute(1, 0).reshape(-1)

                    total_tests_evaluated += region_mask.sum().item()

                    B = B[region_mask]
                    S = S[region_mask]
                    T = T[region_mask]
                    P = P[region_mask]

                    if compute_ig:
                        IG_analytical = IG_analytical[region_mask]

                    region_indices_list.append(region_mask) # Need to save for index generation
                else:
                    total_tests_evaluated += B.shape[0]

                # Save the full B for Deep IG if needed
                B_full = B if compute_ig_deep else None

                if methylation_only and not early_meth_slice:
                    # B is (N, K). Columns: Const, Meth, Covars...
                    # Meth is index 1.
                    # Skipped when early_meth_slice already trimmed the K
                    # axis above; B/S/T/P are already (N, 1) in that case.
                    B = B[:, 1:2]
                    S = S[:, 1:2]
                    T = T[:, 1:2]
                    P = P[:, 1:2]

                if p_only:
                    current_results = P
                else:
                    # Assemble (N, 4*K) output by writing B, S, T, P into a
                    # pre-allocated buffer and freeing each slice as it is
                    # copied. Equivalent to torch.cat((B, S, T, P), dim=1),
                    # but avoids holding all four tensors plus the
                    # concatenated result alive simultaneously (which roughly
                    # doubled peak memory at this point). P is kept because
                    # downstream code still indexes it for p-value filtering.
                    n_rows, k_cols = B.shape
                    current_results = torch.empty(
                        (n_rows, 4 * k_cols), device=B.device, dtype=B.dtype
                    )
                    current_results[:, 0:k_cols] = B
                    del B
                    current_results[:, k_cols:2 * k_cols] = S
                    del S
                    current_results[:, 2 * k_cols:3 * k_cols] = T
                    del T
                    current_results[:, 3 * k_cols:4 * k_cols] = P

                if compute_ig:
                    current_results = torch.cat((current_results, IG_analytical), dim=1)

                if compute_influence:
                    h_max_rows = h_max.unsqueeze(1).expand(-1, chunk_len).permute(1, 0).reshape(-1)
                    if region != 'all':
                        h_max_rows = h_max_rows[region_mask]
                    current_results = torch.cat((current_results, h_max_rows.unsqueeze(1)), dim=1)

                if do_reservoir:
                    # Collect batch results for reservoir sampling BEFORE p-value filtration
                    batch_size = len(current_results)
                    if batch_size > 0:
                        # Construct indices for reservoir
                        gt_chunk_names_res = gt_site_names[gene_start_index:gene_end_index]
                        gt_sites_res = numpy.repeat(gt_chunk_names_res, mt_count)
                        mt_sites_res = numpy.tile(mt_site_names, chunk_len)

                        if region != 'all':
                            # region_mask was applied, we need the numpy mask
                            mask_np_res = region_indices_list[-1].cpu().numpy()
                            gt_sites_res = gt_sites_res[mask_np_res]
                            mt_sites_res = mt_sites_res[mask_np_res]

                        # Generate random rolls for reservoir
                        rolls = torch.rand(
                            batch_size, device=device, generator=reservoir_rng
                        )
                        # Probability P for each item j in batch is reservoir_count / (reservoir_processed + j + 1)
                        # To do this correctly:
                        # P_j = reservoir_count / (reservoir_processed + j + 1)
                        # If roll < P_j, we keep it. Where does it go? Random index in [0, reservoir_count - 1]

                        if reservoir_processed + batch_size <= reservoir_count:
                            # If buffer is not full, just append everything
                            reservoir_buffer.append((current_results.clone(), gt_sites_res, mt_sites_res))
                            reservoir_processed += batch_size
                        else:
                            # Vectorized Reservoir Sampling
                            # Generate indices j for the batch (1-based relative to processed)
                            j_indices = torch.arange(1, batch_size + 1, device=device)
                            total_processed_j = reservoir_processed + j_indices

                            # Items with index <= reservoir_count ALWAYS get kept (if any)
                            # However, we handle the 'buffer not full initially' gracefully

                            keep_probs = reservoir_count / total_processed_j
                            keep_mask = rolls < keep_probs

                            # For the items before reservoir_count, we MUST keep them
                            if reservoir_processed < reservoir_count:
                                initial_keep_count = reservoir_count - reservoir_processed
                                keep_mask[:initial_keep_count] = True

                            kept_indices = torch.nonzero(keep_mask).squeeze(1)

                            if len(kept_indices) > 0:
                                kept_results = current_results[kept_indices]
                                kept_gt = gt_sites_res[kept_indices.cpu().numpy()]
                                kept_mt = mt_sites_res[kept_indices.cpu().numpy()]

                                # Where to place them in the buffer?
                                # For each kept item, if its total index <= reservoir_count, we append it
                                # If its total index > reservoir_count, it replaces a random element [0, reservoir_count-1]
                                replace_indices = torch.randint(
                                    0,
                                    reservoir_count,
                                    (len(kept_indices),),
                                    device=device,
                                    generator=reservoir_rng,
                                )

                                # Process the kept items
                                for idx, kept_idx in enumerate(kept_indices):
                                    total_idx = reservoir_processed + kept_idx.item() + 1
                                    if total_idx <= reservoir_count:
                                        # Buffer is not full, we just append
                                        reservoir_buffer.append((kept_results[idx:idx+1].clone(), kept_gt[idx:idx+1], kept_mt[idx:idx+1]))
                                    else:
                                        # Buffer is full, replace an existing element
                                        target_idx = replace_indices[idx].item()

                                        # Since reservoir_buffer is a list of tensors of varying sizes,
                                        # replacing by index requires finding which tensor/element it belongs to.
                                        # This can be slow. To optimize, if the buffer gets complex, we flatten it.
                                        # Instead of full flattening, we'll store individual replacements in a list
                                        # and flatten later, or we can flatten the reservoir buffer once it reaches capacity.

                                        # Flatten buffer if not already flattened
                                        if isinstance(reservoir_buffer, list):
                                            if sum(len(x[1]) for x in reservoir_buffer) > 0:
                                                res_res_cat = torch.cat([x[0] for x in reservoir_buffer])
                                                res_gt_cat = numpy.concatenate([x[1] for x in reservoir_buffer])
                                                res_mt_cat = numpy.concatenate([x[2] for x in reservoir_buffer])
                                                reservoir_buffer = (res_res_cat, res_gt_cat, res_mt_cat)
                                            else:
                                                # Handle empty buffer weirdness
                                                pass

                                        if isinstance(reservoir_buffer, tuple):
                                            res_res_cat, res_gt_cat, res_mt_cat = reservoir_buffer
                                            res_res_cat[target_idx] = kept_results[idx]
                                            res_gt_cat[target_idx] = kept_gt[idx]
                                            res_mt_cat[target_idx] = kept_mt[idx]

                            reservoir_processed += batch_size

                # P-value filtration
                if p_thresh is not None:
                    # P shape (N, K_out). Meth p-value is at index 0 (if meth_only) or 1 (if not).
                    # Wait, if meth_only, P is (N, 1).
                    # If not meth_only, P is (N, K). Const, Meth, ...
                    # regression_full: p_indices = P[:, 0 if methylation_only else 1] <= p_thresh
                    p_col = 0 if methylation_only else 1
                    p_indices = P[:, p_col] <= p_thresh
                    p_indices_list.append(p_indices)

                    P = P[p_indices]
                    # Note: B, S, T are no longer kept around as separate
                    # tensors at this point -- they have been written into
                    # current_results above and freed. Filtering of the
                    # combined block happens via current_results[p_indices]
                    # below.

                if filtration:
                    output_sizes.append(len(P))

                chunk_results = current_results[p_indices] if p_thresh is not None else current_results

                if compute_ig_deep and p_thresh is not None:
                    from captum.attr import IntegratedGradients

                    class LinearForwardWrapper(torch.nn.Module):
                        def __init__(self, w):
                            super().__init__()
                            self.w = w

                        def forward(self, x):
                            # x shape: (S, K), w shape: (K,)
                            # outputs: (S, 1)
                            return x.matmul(self.w).unsqueeze(1)

                    deep_ig_scores = []
                    # p_indices is a boolean mask of shape (N_after_region,)
                    # we need the indices where it is true
                    valid_flat_indices = torch.nonzero(p_indices).squeeze(1)

                    if len(valid_flat_indices) > 0:
                        # Full W for significant hits
                        B_full_filtered = B_full[p_indices] # shape: (N_sig, K)

                        # If region is not 'all', p_indices applies to the remaining hits
                        # We need the original (M, G) flat index to get m_idx
                        if region != 'all':
                            # region_mask is boolean of shape (M*G,)
                            # Get the original indices before region mask
                            region_flat_indices = torch.nonzero(region_mask).squeeze(1)
                            original_flat_indices = region_flat_indices[valid_flat_indices]
                        else:
                            original_flat_indices = valid_flat_indices

                        for i, orig_flat_idx in enumerate(original_flat_indices):
                            w = B_full_filtered[i]  # Shape: (K,)

                            # Unravel the flat index to get the Methylation index (m_idx)
                            # B was shaped (G*M, K) because we did permute(1, 0, 2) on (M, G, K)
                            # Wait, we did:
                            # B = B.permute(1, 0, 2).reshape(-1, num_coeffs)  -> (G, M, K)
                            # So row-major means outer loop is G, inner is M.
                            # So index = g_idx * M_count + m_idx
                            # Therefore:
                            # m_idx = orig_flat_idx % mt_count
                            m_idx = orig_flat_idx % mt_count

                            x_hit = X[m_idx]                 # Shape: (S, K)
                            x_baseline_hit = X_baseline[m_idx] # Shape: (1, K)

                            wrapper = LinearForwardWrapper(w)
                            ig = IntegratedGradients(wrapper)

                            attributions = ig.attribute(inputs=x_hit, baselines=x_baseline_hit, target=0, n_steps=50)

                            # hit_saliency vector: (K,)
                            hit_saliency = attributions.abs().mean(dim=0)

                            # Remove the intercept (index 0)
                            hit_saliency = hit_saliency[1:]

                            # Keep only the requested covariates
                            hit_saliency = hit_saliency[ig_col_indices]

                            deep_ig_scores.append(hit_saliency)

                        deep_ig_tensor = torch.stack(deep_ig_scores)
                        chunk_results = torch.cat((chunk_results, deep_ig_tensor), dim=1)
                    else:
                        # No significant hits, but we need to match the columns
                        num_ig_cols = len(ig_col_indices)
                        empty_ig = torch.empty((0, num_ig_cols), device=device, dtype=dtype)
                        chunk_results = torch.cat((chunk_results, empty_ig), dim=1)

                if p_only:
                    results.append(P)
                else:
                    results.append(chunk_results)

                # End of the device compute stage (solve, T/P, masks, threshold
                # gather). Synchronised under TECPG_PROFILE=1 so that
                # prof_gpu_time measures queued device work rather than kernel
                # launch latency, mirroring the prep/h2d stamps above.
                if inner_logger.carry_data.get('profile') and torch.cuda.is_available():
                    torch.cuda.synchronize()
                prof_t3 = time.perf_counter()
                prof_gpu_time += (prof_t3 - prof_t2)

                # Save output if gene chunking is used
                # In regression_full, saving happens inside the loop if gene_loci_per_chunk is set.
                if gene_loci_per_chunk:
                    # Construct indices
                    # We need indices for the current gene chunk.
                    # G_chunk names:
                    gt_chunk_names = gt_site_names[gene_start_index:gene_end_index]

                    # D2H stage: every device-to-host copy this chunk needs is
                    # issued here, together, so that prof_d2h_time measures the
                    # transfers (and the implicit stream sync) and prof_post_time
                    # below measures host index construction only.
                    #
                    # P1: when any filter is active, compose the survivors'
                    # flat indices into the (G, M) grid on device and copy only
                    # that int64 vector, instead of copying full-grid boolean
                    # masks and building two (chunk_len * mt_count) host string
                    # arrays only to discard all but the survivors. Row order
                    # is unchanged: chunk_results rows are the True positions
                    # of the composed mask in ascending order, which is exactly
                    # torch.nonzero order.
                    if region != 'all':
                        flat_dev = torch.nonzero(region_indices_list[-1]).squeeze(1)
                        if p_thresh is not None:
                            flat_dev = flat_dev[p_indices_list[-1]]
                    elif p_thresh is not None:
                        flat_dev = torch.nonzero(p_indices_list[-1]).squeeze(1)
                    else:
                        flat_dev = None  # unfiltered: full-grid index path below
                    flat_np = flat_dev.cpu().numpy() if flat_dev is not None else None
                    del flat_dev
                    results_host = torch.cat(results).cpu().numpy()

                    prof_t4 = time.perf_counter()
                    prof_d2h_time += (prof_t4 - prof_t3)

                    # Index construction. Flatten order is
                    # B.permute(1, 0, 2).reshape(-1, K): G outer, M inner, so a
                    # survivor's flat grid index f decomposes as
                    # gene = f // mt_count, meth = f % mt_count.
                    if flat_np is not None:
                        # O(survivors): index the name arrays directly.
                        gt_sites = gt_chunk_names[flat_np // mt_count]
                        mt_sites = mt_site_names[flat_np % mt_count]
                        if region != 'all':
                            del region_indices_list[:]
                        if p_thresh is not None:
                            del p_indices_list[:]
                    else:
                        # Unfiltered output keeps the full-grid construction:
                        # every pair is emitted, so there is nothing to skip.
                        gt_sites = numpy.repeat(gt_chunk_names, mt_count) # [g1, g1, ..., g2, g2, ...]
                        mt_sites = numpy.tile(mt_site_names, chunk_len) # [m1, m2, ..., m1, m2, ...]

                    index_chunk = [gt_sites, mt_sites]

                    # Create path and save
                    gene_index_str = str(gene_chunk_index + 1) # This differs slightly from regression_full count which counts genes?
                    # regression_full: gene_index_str = str(mc_logger.current_count + 1)
                    # mc_logger.current_count increments when saving.
                    # Here we can just use loop index + 1.
                    meth_index_str = str(meth_chunk_index + 1)
                    file_name = file_format.format(
                        meth_chunk=meth_index_str, gene_chunk=gene_index_str
                    )
                    file_path = os.path.join(output_dir, file_name)

                    # End of the host post-processing stage (index construction
                    # and output path). What follows, up to prof_t6, is the
                    # save-payload construction and is reported as `write`
                    # (write_enqueue_ms); pool.submit and the save-queue
                    # back-pressure wait remain in the next chunk's idle gap.
                    prof_t5 = time.perf_counter()
                    prof_post_time += (prof_t5 - prof_t4)

                    out = pandas.DataFrame(
                        results_host,
                        index=index_chunk,
                        columns=columns,
                    )
                    out.index.set_names(index_names, inplace=True)

                    prof_t6 = time.perf_counter()
                    prof_write_time += (prof_t6 - prof_t5)

                    run_metrics['prep_ms'].append(prof_prep_time * 1000)
                    run_metrics['h2d_ms'].append(prof_h2d_time * 1000)
                    run_metrics['compute_ms'].append(prof_gpu_time * 1000)
                    run_metrics['d2h_ms'].append(prof_d2h_time * 1000)
                    run_metrics['post_ms'].append(prof_post_time * 1000)
                    run_metrics['write_enqueue_ms'].append(prof_write_time * 1000)
                    run_metrics['gpu_idle_between_chunks_ms'].append(gpu_idle_between_chunks_ms)
                    total_chunks_saved += 1
                    total_bytes_written += out.memory_usage(deep=True).sum()
                    total_tests_passed_filter += len(out)
                    last_chunk_end_time = time.perf_counter()

                    prof_total = prof_t6 - prof_t0
                    M_c = len(M_chunk)
                    G_c = chunk_len
                    K_val = K
                    S_val = nrows
                    res = inner_logger.resource_check()
                    util_sm = gpu_monitor.avg_util_sm if 'gpu_monitor' in locals() and gpu_monitor and hasattr(gpu_monitor, 'avg_util_sm') else 0

                    if inner_logger.carry_data.get('profile') and torch.cuda.is_available():
                        gflops = (2 * M_c * (K_val**2) * S_val + 2 * M_c * K_val * G_c * S_val) / 1e9 / max(prof_gpu_time, 1e-9)
                        reg_sec = (M_c * G_c) / max(prof_total, 1e-9)

                        inner_logger.debug(
                            f"PROFILE chunk m={meth_chunk_index+1}/{meth_chunk_count} g={gene_chunk_index+1}/{gene_chunk_count} | "
                            f"prep={prof_prep_time*1000:.1f}ms h2d={prof_h2d_time*1000:.1f}ms "
                            f"gpu={prof_gpu_time*1000:.1f}ms d2h={prof_d2h_time*1000:.1f}ms "
                            f"post={prof_post_time*1000:.1f}ms write={prof_write_time*1000:.1f}ms "
                            f"idle={gpu_idle_between_chunks_ms:.1f}ms save_q={save_queue_fill} pref_f={prefetch_fill} "
                            f"total={prof_total*1000:.1f}ms reg/s={reg_sec:.2e} gflops={gflops:.1f} util_sm={util_sm:.1f}% ram_avail={res['ram_avail_gb']:.1f}GB"
                        )

                    bottleneck = analyze_bottleneck(
                        prof_gpu_time, prof_total, prof_h2d_time, prof_d2h_time, prof_write_time,
                        util_sm, res['ram_avail_gb'], res['cpu_percent'],
                        gene_chunk_size=chunk_len, meth_chunk_size=M_c
                    )
                    if bottleneck:
                        inner_logger.info(bottleneck)

                    # Save
                    mc_logger.count(
                        'Saving part {i}/{0}',
                        gene_chunk_count,
                    )
                    # Backpressure: bounded by save_queue_depth (auto-scales)
                    while len(futures) >= save_queue_depth:
                        futures.popleft().result()
                    futures.append(pool.submit(
                        save_dataframe_part,
                        out, file_path, gene_chunk_index + 1,
                        first=None,
                        output_format=output_format,
                        **dict(mc_logger),
                    ))
                    del out, gt_sites, mt_sites, index_chunk, results_host
                    del results[:]

                    # Force GC
                    if allocated_memory:
                        total_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
                        if aggressive_gc or (total_memory and torch.cuda.memory_allocated() / total_memory > HIGH_WATER):
                            torch.cuda.empty_cache()

            del Q, R_inv, XtXi_diag_sqrt
            import gc; gc.collect()

            mc_logger.time('Looped over methylation loci in {l} seconds')
            mc_logger.time('Calculated tecpg_mlr_qr in {t} seconds')

            # If no gene chunking (gene_loci_per_chunk is None), save/return results
            if gene_loci_per_chunk is None:
                # We have one result in `results`.
                # Generate indices.
                gt_sites = numpy.repeat(gt_site_names, mt_count)
                mt_sites = numpy.tile(mt_site_names, gt_count)

                if region != 'all':
                    mask_np = region_indices_list[-1].cpu().numpy()
                    del region_indices_list[:]
                    gt_sites = gt_sites[mask_np]
                    mt_sites = mt_sites[mask_np]

                if p_thresh is not None:
                    p_mask_np = p_indices_list[-1].cpu().numpy()
                    del p_indices_list[:]
                    gt_sites = gt_sites[p_mask_np]
                    mt_sites = mt_sites[p_mask_np]

                index_chunk = [gt_sites, mt_sites]

                out = pandas.DataFrame(
                    torch.cat(results).cpu().numpy(),
                    index=index_chunk,
                    columns=columns,
                )
                out.index.set_names(index_names, inplace=True)

                total_tests_passed_filter += len(out)

                if meth_loci_per_chunk is not None:
                    gene_index_str = '1'
                    meth_index_str = str(meth_chunk_index + 1)
                    file_name = file_format.format(
                        meth_chunk=meth_index_str, gene_chunk=gene_index_str
                    )
                    file_path = os.path.join(output_dir, file_name)

                    mc_logger.count(
                        'Saving methylation chunk {0}/{1}',
                        meth_chunk_index + 1,
                        meth_chunk_count,
                    )
                    # Backpressure: bounded by save_queue_depth (auto-scales)
                    while len(futures) >= save_queue_depth:
                        futures.popleft().result()
                    futures.append(pool.submit(
                        save_dataframe_part,
                        out, file_path, meth_chunk_index + 1,
                        first=None,
                        output_format=output_format,
                        **dict(mc_logger),
                    ))
                    del out, gt_sites, mt_sites, index_chunk
                    del results[:]

            completed_chunks = meth_chunk_index + 1
            remaining_chunks = meth_chunk_count - completed_chunks

            chunk_end_time = time.time()
            chunk_duration = chunk_end_time - methylation_last_chunk_end_time
            methylation_last_chunk_end_time = chunk_end_time

            methylation_chunk_times.append(chunk_duration)

            if completed_chunks > 1:
                # Exclude the first chunk from average to avoid startup overhead skewing the estimate
                average_time = sum(methylation_chunk_times[1:]) / (completed_chunks - 1)
            else:
                average_time = chunk_duration

            estimated_remaining_seconds = average_time * remaining_chunks
            estimated_remaining_hours = estimated_remaining_seconds / 3600

            logger.time(
                'FINISHED METHYLATION CHUNK {0} IN {l} SECONDS. ESTIMATED TIME REMAINING: {1:.2f} SECONDS ({2:.2f} HOURS)',
                completed_chunks,
                estimated_remaining_seconds,
                estimated_remaining_hours,
            )

        # Wait for chunks
        if chunking:
            logger.time('Waiting for chunks to save...')
            wait_start = time.time()
            while futures:
                futures.popleft().result()
            pool.shutdown(wait=True)
            logger.time('Finished waiting for chunks to save in {l} seconds')

        if prefetch_executor:
            prefetch_executor.shutdown(wait=True)

        # Print end-of-run summary
        import numpy as np
        summary_str = ["--- END OF RUN SUMMARY ---"]
        summary_str.append(f"Genes evaluated: {len(G)}")
        summary_str.append(f"Methylation loci evaluated: {len(M)}")
        summary_str.append(f"Total tests evaluated: {total_tests_evaluated} (TOTAL_TESTS={total_tests_evaluated})")
        summary_str.append(f"Tests passed p-value filter and saved: {total_tests_passed_filter}")
        summary_str.append(f"Chunks saved: {total_chunks_saved}")
        summary_str.append(f"Total bytes written: {total_bytes_written} ({total_bytes_written/1024/1024:.2f} MB)")
        for metric, vals in run_metrics.items():
            if len(vals) > 0:
                summary_str.append(f"{metric}: sum={sum(vals):.1f}ms, mean={np.mean(vals):.1f}ms, p95={np.percentile(vals, 95):.1f}ms")
        for line in summary_str:
            logger.info(line)

        if do_reservoir and reservoir_processed > 0:
            logger.info('Saving reservoir sample ({0} rows)', min(reservoir_processed, reservoir_count))
            if isinstance(reservoir_buffer, list):
                res_res_cat = torch.cat([x[0] for x in reservoir_buffer])
                res_gt_cat = numpy.concatenate([x[1] for x in reservoir_buffer])
                res_mt_cat = numpy.concatenate([x[2] for x in reservoir_buffer])
            else:
                res_res_cat, res_gt_cat, res_mt_cat = reservoir_buffer

            index_chunk = [res_gt_cat, res_mt_cat]
            out_reservoir = pandas.DataFrame(
                res_res_cat.cpu().numpy(),
                index=index_chunk,
                columns=columns,
            )
            out_reservoir.index.set_names(index_names, inplace=True)

            # Additional logging for reservoir processing status
            logger.info(f"Reservoir sampling completed. Processed {reservoir_processed} total items. "
                        f"Target reservoir count was {reservoir_count}.")

            # Determine memory usage of the resulting DataFrame
            mem_usage_bytes = out_reservoir.memory_usage(deep=True).sum()
            mem_usage_mb = mem_usage_bytes / (1024 ** 2)

            logger.info(f"Reservoir DataFrame shape: {out_reservoir.shape} (rows, columns). "
                        f"Memory footprint: {mem_usage_mb:.2f} MB.")

            # Use output_dir from logger carry_data if not explicitly passed
            carry_output = logger.carry_data.get('output_dir', None)
            res_output_dir = output_dir if output_dir is not None else (carry_output if carry_output else os.getcwd())

            # Ensure output directory exists before writing
            if not os.path.exists(res_output_dir):
                try:
                    os.makedirs(res_output_dir, exist_ok=True)
                except OSError:
                    pass

            res_file_path = os.path.join(res_output_dir, 'sample_reservoir.csv')
            out_reservoir.to_csv(res_file_path)
            logger.time('Finished saving reservoir sample to {0}', res_file_path)

        logger.time(
            'Finished calculating the multiple linear regression (qr) in {t} total'
            ' seconds'
        )

        if not chunking:
            return out
