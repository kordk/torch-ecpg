import math
from multiprocessing import Pool
from typing import Optional
import pandas
import torch
from config import device
from logger import Logger
from import_data import save_dataframe_part


def pearson_full_tensor(
    M: pandas.DataFrame,
    G: pandas.DataFrame,
    n: Optional[int] = None,
    *,
    logger: Logger = Logger(),
) -> pandas.DataFrame:
    '''
    Calculates the pearson correlation coefficient matrix for two full
    dataframes. Returns a dataframe of the output with M.index indices
    and G.index columns. If n, the sample size, is not provided, it will
    infer it from len(M.columns).

    Calculating the pearson correlation coefficient matrix given the
    entire two matrices is monumentally faster than single iteration
    pearson.

    The algorithm works by removing much of the unneeded recomputation
    present in pearson_corr_tensor. The algorithm is roughly based on
    ((x . y) - n * mean(x) * mean(y)) / ((n - 1) * std(x) * std(y)).

    n = M rows
    k = G rows
    With the iterative formula above, the means are calculated (n*k)
    times for x and y. With the full solution, the means are calculated
    (n+m) times. The same is true for stds.

    # The code below is the expanded version of the function. While the
    # code is more readable in this form, it runs into CUDA memory
    # issues if the input is too large. It may differ slightly from the
    # current version.

    M_t = torch.tensor(M.to_numpy()).to(device)
    G_t = torch.tensor(G.to_numpy()).to(device)
    M_means = M_t.mean(axis=1)
    G_means = G_t.mean(axis=1)
    dots = M_t @ G_t.T
    sub = M_means.outer(G_means)
    sub *= n
    M_std = M_t.std(axis=1)
    G_std = G_t.std(axis=1)
    denom = M_std.outer(G_std)
    denom *= n - 1
    corr: torch.Tensor = (dots - sub) / denom
    corr_pd = pandas.DataFrame(
        corr.tolist(),
        index=M.index,
        columns=G.index,
    )
    '''
    logger.start_timer('info', 'Calculating pearson_full_tensor...')
    if n is None:
        n = len(M.columns)

    M_t = torch.tensor(M.to_numpy()).to(device)
    logger.time('Converted M to tensor in {l} seconds')
    G_t = torch.tensor(G.to_numpy()).to(device)
    logger.time('Converted G to tensor in {l} seconds')

    corr_pd = pandas.DataFrame(
        (
            (M_t @ G_t.T - M_t.mean(axis=1).outer(G_t.mean(axis=1)) * n)
            / (M_t.std(axis=1).outer(G_t.std(axis=1)) * (n - 1))
        ).tolist(),
        index=M.index,
        columns=G.index,
    )
    logger.time('Calculated corr_pd dataframe in {l} seconds.')

    logger.time(
        'Finished calculating pearson_full_tensor in {t} seconds.', ignore=True
    )
    return corr_pd


def pearson_chunk_tensor(
    M: pandas.DataFrame,
    G: pandas.DataFrame,
    chunks: int,
    n: Optional[int] = None,
    *,
    logger: Logger = Logger(),
) -> pandas.DataFrame:
    '''
    See pearson_full_tensor.

    Utilizes chunks of values to avoid running into GPU memory limits.
    The chunk version is often faster than the full version by a small
    amount. Adjust chunks such that torch is not using too much GPU
    memory.

    Chunks is the number of chunks into which to divide the M dataframe.
    Verbose prints messages to gauge time.

    Returns pandas.DataFrame.
    '''
    if n is None:
        n = len(M.columns)
    chunk_rows = len(M) // chunks
    logger.start_timer(
        'info',
        'Running pearson chunk tensor with'
        f' {chunk_rows * len(G) * n} values per chunk ({chunk_rows} rows'
        ' per chunk).',
    )

    M_t = torch.tensor(M.to_numpy()).to(device)
    logger.time('Converted M to tensor in {l} seconds')
    G_t = torch.tensor(G.to_numpy()).to(device)
    logger.time('Converted G to tensor in {l} seconds')
    G_means = G_t.mean(axis=1)
    logger.time('Calculated G means in {l} seconds')
    G_std = G_t.std(axis=1)
    logger.time('Calculated G standard deviations in {l} seconds')

    inner_logger = logger.alias()
    inner_logger.start_timer('info', 'Calculating chunks...')
    corr_pd = pandas.DataFrame()
    for i in range(0, len(M_t), chunk_rows):
        j = i + chunk_rows
        M_chunk = M_t[i:j]
        corr_pd = corr_pd.append(
            pandas.DataFrame(
                (
                    (M_chunk @ G_t.T - M_chunk.mean(axis=1).outer(G_means) * n)
                    / (M_chunk.std(axis=1).outer(G_std) * (n - 1))
                ).tolist(),
                index=M.index[i:j],
                columns=G.index,
            )
        )
        del M_chunk
        inner_logger.time(
            'Completed chunk {i}/{0} in {l} seconds. Average chunk time: {a}'
            ' seconds',
            chunks,
        )
        inner_logger.info(
            'Estimated time remaining: {0} seconds',
            inner_logger.remaining_time(chunks),
        )

    logger.time('Calculated pearson_chunk_tensor in {t} seconds')
    return corr_pd


def pearson_chunk_save_tensor(
    M: pandas.DataFrame,
    G: pandas.DataFrame,
    chunks: int,
    save_chunks: int,
    output_dir: str,
    n: Optional[int] = None,
    *,
    logger: Logger = Logger(),
) -> None:
    '''
    See pearson_full_tensor.

    The pearson_chunk_save_tensor function is the fastest pearson
    correlation coefficient function available

    Saves chunks of data to output_dir every save_chunks chunks to avoid
    CPU memory limits. The saving is done asynchronously using
    multiprocessing as to not interfere with the GPU work.

    Utilizes chunks of values to avoid running into GPU memory limits.
    The chunk version is often faster than the full version by a small
    amount. Adjust chunks such that torch is not using too much GPU
    memory.

    Chunks is the number of chunks into which to divide the M dataframe.
    Verbose prints messages to gauge time.

    Returns pandas.DataFrame.
    '''
    if n is None:
        n = len(M.columns)
    chunk_rows = len(M) // chunks
    save_chunk_count = math.ceil(chunks / save_chunks)

    logger.start_timer(
        'info',
        'Running pearson chunk tensor with'
        f' {chunk_rows * len(G) * n} values per chunk ({chunk_rows} rows'
        ' per chunk).',
    )

    M_t = torch.tensor(M.to_numpy()).to(device)
    logger.time('Converted M to tensor in {l} seconds')
    G_t = torch.tensor(G.to_numpy()).to(device)
    logger.time('Converted G to tensor in {l} seconds')
    G_means = G_t.mean(axis=1)
    logger.time('Calculated G means in {l} seconds')
    G_std = G_t.std(axis=1)
    logger.time('Calculated G standard deviations in {l} seconds')

    inner_logger = logger.alias()
    inner_logger.start_timer('info', 'Calculating chunks...')
    logger.start_counter('info', '')
    chunks_elapsed = 0
    corr_pd = pandas.DataFrame()
    with Pool(processes=8) as pool:
        for i in range(0, len(M_t), chunk_rows):
            chunks_elapsed += 1
            if chunks_elapsed > save_chunks:
                chunks_elapsed = 1
                file_name = output_dir + str(logger.current_count + 1) + '.csv'
                logger.count('Saving part {i}/{0}: ', save_chunk_count)
                pool.apply_async(
                    save_dataframe_part, (corr_pd, file_name), dict(logger)
                )
                del corr_pd
                corr_pd = pandas.DataFrame()

            j = i + chunk_rows
            M_chunk = M_t[i:j]
            corr_pd = corr_pd.append(
                pandas.DataFrame(
                    (
                        (
                            M_chunk @ G_t.T
                            - M_chunk.mean(axis=1).outer(G_means) * n
                        )
                        / (M_chunk.std(axis=1).outer(G_std) * (n - 1))
                    ).tolist(),
                    index=M.index[i:j],
                    columns=G.index,
                )
            )
            del M_chunk
            inner_logger.time(
                'Completed chunk {i}/{0} in {l} seconds. Average chunk time:'
                ' {a} seconds',
                chunks,
            )
            inner_logger.info(
                'Estimated time remaining: {0} seconds',
                inner_logger.remaining_time(chunks),
            )

        file_name = output_dir + str(logger.current_count + 1) + '.csv'
        logger.count('Saving part {i}/{0}: ', save_chunk_count)
        pool.apply_async(
            save_dataframe_part, (corr_pd, file_name), dict(logger)
        )

        pool.close()
        pool.join()

    del corr_pd
    logger.time('Calculated pearson_chunk_tensor in {t} seconds')
