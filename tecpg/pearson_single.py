from typing import List, Optional
import numpy as np
from scipy.stats import pearsonr
import torch
from .config import device


def scipy_pearsonr_corr(x: List[float], y: List[float]) -> float:
    '''
    Alias to scipy.stats.pearsonr. Only returns correlation coefficient,
    not p-value.

    Takes in two lists of floats (x and y), and returns their Pearson
    correlation coefficient as a float.
    '''
    return pearsonr(x, y)[0]


def pearson_corr_basic(
    x: List[float], y: List[float], n: Optional[int] = None
) -> float:
    '''
    Calculates the pearson correlation coefficient on the cpu based on
    the formula: https://wikimedia.org/api/rest_v1/media/math/render/svg
    /2b9c2079a3ffc1aacd36201ea0a3fb2460dc226f.
    '''
    if n is None:
        n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numer = denom_x = denom_y = 0
    for x_val, y_val in zip(x, y):
        x_diff, y_diff = x_val - x_mean, y_val - y_mean
        numer += x_diff * y_diff
        denom_x += x_diff ** 2
        denom_y += y_diff ** 2

    corr = numer / np.sqrt(denom_x * denom_y)
    return corr


def pearson_corr_tensor(
    x: List[float], y: List[float], n: Optional[int] = None
) -> float:
    '''
    Calculates the pearson correlation coefficient using a tensor to
    utilize gpu acceleration, if possible. Based on the formula: https:
    //wikimedia.org/api/rest_v1/media/math/render/svg/1ea4ff80b5f62cbad4
    2cd98edef63a4e5dcfe930.
    '''
    if n is None:
        n = len(x)
    x_t, y_t = torch.tensor(x).to(device), torch.tensor(y).to(device)
    dot = x_t.dot(y_t)
    sub = n * x_t.mean() * y_t.mean()
    denom = (n - 1) * x_t.std() * y_t.std()
    corr = (dot - sub) / denom
    return float(corr)
