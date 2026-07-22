import os
import shutil
from typing import Dict, List, Tuple, TypeVar, Optional

import numpy as np
import pandas
import requests
import torch

from .logger import Logger

T = TypeVar('T')


def random_list(length: int, minimum: float, maximum: float) -> List[float]:
    """
    Returns a list of length, with random float values ranging from
    minimum to maximum. Returns a list of floats.
    """
    return list(np.random.rand(length) * (maximum - minimum) + minimum)


def download_files(
    output_dir: str,
    files: List[Tuple[str, str]],
    *,
    logger: Logger = Logger(),
) -> None:
    """
    Downloads files from files, a list of tuples with file names and
    their corresponding urls. Saves files in output_dir. The function is
    very fast for large files. If verbose is true, it will print out the
    currently downloading file as the function runs.
    """
    n = len(files)
    logger.start_timer('info', 'Downloading {0} files...', n)

    for file_name, url in files:
        with requests.get(url, stream=True) as stream:
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, 'wb') as file:
                logger.time('Downloading {i}/{0}: {1}...', n, file_name)
                shutil.copyfileobj(stream.raw, file)
                logger.time_check('Downloaded in {l} seconds', n)

    logger.time_check(
        'Finished downloading {0} files in {t} seconds.',
        n,
    )


def initialize_dir(directory: str, *, logger: Logger = Logger()) -> None:
    """Clears and creates provided directory"""
    if os.path.isdir(directory):
        logger.info('Removing directory {0}...', directory)
        shutil.rmtree(directory)
    logger.info('Creating directory {0}...', directory)
    os.mkdir(directory)


def read_csv(
    file_name: str, sep: str = ',', *, logger: Logger = Logger()
) -> pandas.DataFrame:
    """
    Reads file_name as a csv with separator sep and returns
    pandas.DataFrame. Reads pandas-style csv, where indices and columns
    are automatically generated.
    """
    logger.info(
        'Reading csv file {0} with separator {1}',
        file_name,
        '[tab]' if sep == '\t' else sep,
    )
    df = pandas.read_csv(file_name, sep=sep, index_col=[0], dtype={0: str})
    if df.index.dtype == 'object':
        logger.info(
            'The index for file {0} was loaded as an object type. '
            'This may indicate mixed types (e.g. integers and strings).',
            file_name,
        )
    return df


def verify_and_trim_samples(
    M: pandas.DataFrame,
    G: pandas.DataFrame,
    C: Optional[pandas.DataFrame] = None,
    *,
    logger: Optional[Logger] = None,
) -> Tuple[pandas.DataFrame, ...]:
    """
    Verifies that sample labels match across M, G, and optionally C matrices.
    Sample labels are the columns of M and G, and the index of C.
    If they do not match perfectly, logs the discrepancies and trims them
    to the shared intersection of samples.
    """
    if logger is None:
        logger = Logger()

    M_samples = set(M.columns.astype(str))
    G_samples = set(G.columns.astype(str))

    if C is not None:
        C_samples = set(C.index.astype(str))
        shared = M_samples.intersection(G_samples, C_samples)
    else:
        shared = M_samples.intersection(G_samples)

    if len(shared) == len(M_samples) == len(G_samples) and (C is None or len(shared) == len(C_samples)):
        logger.info('Sample verification: All {0} samples match across matrices.', len(shared))
        M_out, G_out = M, G
        if C is not None:
            C_out = C
        else:
            C_out = None
    else:
        # Report discrepancies
        m_diff = len(M_samples) - len(shared)
        g_diff = len(G_samples) - len(shared)

        logger.info(
            'Sample mismatch detected: M has {0} non-overlapping samples, G has {1} non-overlapping samples.',
            m_diff, g_diff
        )
        if C is not None:
            c_diff = len(C_samples) - len(shared)
            logger.info('Sample mismatch detected: C has {0} non-overlapping samples.', c_diff)

        logger.info('Trimming matrices to {0} shared samples.', len(shared))

        # Use the order from M to ensure consistent column ordering
        shared_list = [s for s in M.columns if str(s) in shared]

        M_out = M.loc[:, shared_list]

        # Match G columns efficiently without deep copying entire matrix
        # Identify which columns correspond to the shared list by converting to string
        shared_str_set = {str(s) for s in shared_list}
        g_shared_cols = [c for c in G.columns if str(c) in shared_str_set]

        # Sort g_shared_cols according to the order in shared_list
        g_col_map = {str(c): c for c in g_shared_cols}
        g_ordered_cols = [g_col_map[str(s)] for s in shared_list if str(s) in g_col_map]

        G_out = G.loc[:, g_ordered_cols]

        # Rename the columns of G_out to match the original types from M
        G_out.columns = shared_list

        if C is not None:
            # Avoid pandas.errors.InvalidIndexError by casting index temporarily if needed,
            # but C.index is probably fine.
            original_index_name = C.index.name
            C_str_index = C.copy()
            C_str_index.index = C_str_index.index.astype(str)
            C_out = C_str_index.loc[[str(s) for s in shared_list]]
            C_out.index = shared_list
            C_out.index.name = original_index_name
        else:
            C_out = None

    if C_out is not None:
        # Calculate column variances for C
        variances = C_out.var()
        zero_var_mask = variances < 1e-8
        zero_var_cols = C_out.columns[zero_var_mask].tolist()

        if zero_var_cols:
            logger.info(
                'Found {0} covariate columns with zero variance (< 1e-8): {1}. Dropping them.',
                len(zero_var_cols), zero_var_cols
            )
            C_out = C_out.drop(columns=zero_var_cols)
            if C_out.empty:
                raise ValueError("All columns in the covariate matrix C were dropped due to zero variance. The resulting matrix is empty.")

        return M_out, G_out, C_out

    return M_out, G_out


def trim_dataframes(
    dataframes: List[pandas.DataFrame],
    *,
    logger: Logger = Logger(),
    **drop_kwargs,
) -> None:
    if len(dataframes) < 2:
        logger.warning('Skipped trimming dataframes: less than two inputs')
        return

    indices = [set(df.index) for df in dataframes]
    shared = indices[0].intersection(*indices[1:])

    for position, (df, index) in enumerate(zip(dataframes, indices)):
        before = len(df.index)
        dropped_index = index - shared
        df.drop(dropped_index, inplace=True, **drop_kwargs)
        after = len(df.index)
        if before != after:
            logger.info(
                'Drop site helper.trim_dataframes[df {0}]: dropped loci/rows '
                'not shared across all aligned dataframes: {1} -> {2} '
                '({3} dropped)',
                position, before, after, before - after,
            )


def default_region_parameter(
    region_parameter_name: str,
    region_parameter: T | None,
    region: str,
    defaults: Dict[str, T],
    *,
    logger: Logger = Logger(),
) -> T | None:
    if region in defaults and region_parameter == None:
        updated_region_parameter = defaults[region]
        logger.info(
            'Using default value {0} for region parameter {1} and region {2}',
            updated_region_parameter,
            region_parameter_name,
            region,
        )
        return updated_region_parameter
    if region not in defaults and region_parameter != None:
        logger.info(
            'Region parameter {0} provided but ignored for region {1}',
            region_parameter_name,
            region,
        )
        return None
    return region_parameter


def logit_transform_torch(
    tensor: torch.Tensor, epsilon: float = 1e-6, *, logger: Logger = Logger()
) -> torch.Tensor:
    """
    Clips the tensor values to [epsilon, 1 - epsilon] and applies a
    logit transformation: log2(x / (1 - x)).
    """
    logger.info('[Transformation] Applying Logit-Transform (Beta -> M-values)')

    min_val = tensor.min().item()
    max_val = tensor.max().item()
    logger.info(
        '[Transformation] Input Beta Range: [{0:.4f}, {1:.4f}]',
        min_val,
        max_val,
    )

    count_0 = (tensor == 0.0).sum().item()
    count_1 = (tensor == 1.0).sum().item()
    logger.info(
        '[Transformation] Clamping Applied (epsilon={0}): {1:,} values at 0.0'
        ' | {2:,} values at 1.0',
        epsilon,
        count_0,
        count_1,
    )

    tensor = tensor.clamp(epsilon, 1 - epsilon)
    result = torch.log2(tensor / (1 - tensor))

    out_min = result.min().item()
    out_max = result.max().item()
    out_mean = result.mean().item()
    out_std = result.std().item()

    logger.info(
        '[Transformation] Output M-value Range: [{0:.4f}, {1:.4f}]',
        out_min,
        out_max,
    )
    logger.info(
        '[Transformation] Output Distribution: Mean = {0:.4f} | SD = {1:.4f}',
        out_mean,
        out_std,
    )
    logger.info('[Transformation] Conversion successful. Proceeding to MLR.')

    return result


def logit_transform_pandas(
    df: pandas.DataFrame, epsilon: float = 1e-6, *, logger: Logger = Logger()
) -> pandas.DataFrame:
    """
    Clips the dataframe values to [epsilon, 1 - epsilon] and applies a
    logit transformation: log2(x / (1 - x)).
    """
    logger.info('[Transformation] Applying Logit-Transform (Beta -> M-values)')

    min_val = df.min().min()
    max_val = df.max().max()
    logger.info(
        '[Transformation] Input Beta Range: [{0:.4f}, {1:.4f}]',
        min_val,
        max_val,
    )

    count_0 = (df == 0.0).sum().sum()
    count_1 = (df == 1.0).sum().sum()
    logger.info(
        '[Transformation] Clamping Applied (epsilon={0}): {1:,} values at 0.0'
        ' | {2:,} values at 1.0',
        epsilon,
        count_0,
        count_1,
    )

    df = df.clip(epsilon, 1 - epsilon)
    result = np.log2(df / (1 - df))

    out_min = result.min().min()
    out_max = result.max().max()
    out_mean = result.mean().mean()
    if isinstance(result, pandas.Series):
        out_std = result.std()
    else:
        out_std = result.stack().std()

    logger.info(
        '[Transformation] Output M-value Range: [{0:.4f}, {1:.4f}]',
        out_min,
        out_max,
    )
    logger.info(
        '[Transformation] Output Distribution: Mean = {0:.4f} | SD = {1:.4f}',
        out_mean,
        out_std,
    )
    logger.info('[Transformation] Conversion successful. Proceeding to MLR.')

    return result

def compute_region_mask(region, m_chrom, m_pos, g_chrom, g_pos, g_strand, *,
                        window_base=None, upstream=None, downstream=None):
    """In-region membership for methylation-gene pairs.

    Inputs are broadcastable tensors. The caller controls shape:
    the qr grid path passes (M,1) vs (1,G) -> (M,G); the permutation
    path passes aligned (P,) vs (P,) -> (P,).
    Reproduces the existing qr-path predicate exactly.
    """
    if region in ('cis', 'distal'):
        delta = g_pos - m_pos
        # g_strand is int8 at the call sites; window magnitudes (~1e6)
        # overflow int8, so widen before multiplying.
        gs = g_strand.to(torch.int64)
        # Multiplying by strand flips the window orientation but also reverses
        # the bounds, which makes a symmetric negative-strand window empty.
        # Order them so the interval is always valid.
        lower = torch.minimum(
            gs * (window_base - upstream), gs * (window_base + downstream)
        )
        upper = torch.maximum(
            gs * (window_base - upstream), gs * (window_base + downstream)
        )
        return (
            (m_chrom == g_chrom)
            & (lower < delta)
            & (delta < upper)
        )
    if region == 'trans':
        return m_chrom != g_chrom
    raise ValueError(f"compute_region_mask: unsupported region {region!r}")
