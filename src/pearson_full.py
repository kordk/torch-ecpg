from typing import Optional
import pandas
import torch
from config import device


def pearson_full_tensor(
    M: pandas.DataFrame, G: pandas.DataFrame, n: Optional[int] = None
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
    # issues if the input is too large.

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
    if n is None:
        n = len(M.columns)

    M_t = torch.tensor(M.to_numpy()).to(device)
    G_t = torch.tensor(G.to_numpy()).to(device)

    corr_pd = pandas.DataFrame(
        (
            (M_t @ G_t.T - M_t.mean(axis=1).outer(G_t.mean(axis=1)) * n)
            / (M_t.std(axis=1).outer(G_t.std(axis=1)) * (n - 1))
        ).tolist(),
        index=M.index,
        columns=G.index,
    )

    return corr_pd


def pearson_chunk_tensor(
    M: pandas.DataFrame,
    G: pandas.DataFrame,
    chunks: int,
    n: Optional[int] = None,
    verbose: bool = False,
) -> pandas.DataFrame:
    '''
    Utilizes chunks of values to avoid running into GPU memory limits.
    The chunk version is often faster than the full version by a small
    amount. Adjust chunks such that torch is not using too much GPU
    memory.

    Chunks is the number of chunks into which to divide the M dataframe.
    Verbose prints messages to gauge time.

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
    # issues if the input is too large.

    M_t = torch.tensor(M.to_numpy()).to(device)
    G_t = torch.tensor(G.to_numpy()).to(device)
    G_means = G_t.mean(axis=1)
    G_std = G_t.std(axis=1)
    corr_pd = pandas.DataFrame()
    for index, i in enumerate(range(0, len(M_t), chunk_rows), 1):
        if verbose:
            print(f'Chunk {index}/{len(M.index) // chunk_rows}')
        M_chunk = M_t[i : i + chunk_rows]
        M_chunk_means = M_chunk.mean(axis=1)
        M_chunk_std = M_chunk.mean(axis=1)
        dots = M_chunk @ G_t.T
        sub = M_chunk_means.outer(G_means)
        sub *= n
        denom = (M_chunk_std.outer(G_std)
        denom *= n - 1
        corr_t_chunk = (dots - sub) / denom
        corr_pd_chunk = pandas.DataFrame(
            corr_t_chunk.tolist(),
            index=M.index[i : i + chunk_rows],
            columns=G.index,
        )
        corr_pd = corr_pd.append(corr_pd_chunk)
        del M_chunk
        del corr_t_chunk
    '''
    if n is None:
        n = len(M.columns)
    chunk_rows = len(M) // chunks
    if verbose:
        print(
            'Running pearson chunk tensor with'
            f' {chunk_rows * len(G) * n} values per chunk ({chunk_rows} rows'
            ' per chunk).'
        )

    M_t = torch.tensor(M.to_numpy()).to(device)
    G_t = torch.tensor(G.to_numpy()).to(device)
    G_means = G_t.mean(axis=1)
    G_std = G_t.std(axis=1)

    corr_pd = pandas.DataFrame()
    for index, i in enumerate(range(0, len(M_t), chunk_rows), 1):
        if verbose:
            print(f'Chunk {index}/{len(M.index) // chunk_rows}')
        M_chunk = M_t[i : i + chunk_rows]
        corr_t_chunk = (
            M_chunk @ G_t.T - M_chunk.mean(axis=1).outer(G_means) * n
        ) / (M_chunk.std(axis=1).outer(G_std) * (n - 1))
        corr_pd_chunk = pandas.DataFrame(
            corr_t_chunk.tolist(),
            index=M.index[i : i + chunk_rows],
            columns=G.index,
        )
        corr_pd = corr_pd.append(corr_pd_chunk)
        del M_chunk
        del corr_t_chunk

    return corr_pd
