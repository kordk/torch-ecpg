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
    '''
    if n is None:
        n = len(M.columns)

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
    corr_pd = pandas.DataFrame(corr.tolist(), index=M.index, columns=G.index)
    return corr_pd
