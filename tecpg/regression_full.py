from typing import Tuple
import numpy as np
import pandas as pd
import torch
from .test_data import generate_data
from .config import get_device
from .logger import Logger


def regression_full(
    M: pd.DataFrame,
    G: pd.DataFrame,
    C: pd.DataFrame,
    include: Tuple[bool, bool, bool, bool] = (True, True, True, True),
    *,
    logger: Logger = Logger()
) -> pd.DataFrame:
    '''
    Calculates the multiple linear regression of the input dataframes M,
    G, and C, being methylation beta values, gene expression values, and
    covariates using torch. This is done for every pair of methylation
    site and gene site.

    Returns a flattened multiindex dataframe mapping methylation site
    and gene site to regression results: beta (est), std_err (err),
    t_stats (t), and p-value (p).

    The parameter include specifies which regression results to include.
    The parameter is a tuple of three booleans for the estimate, the
    error, the t-statistic, and the p-value.

    Multiple Linear Regression modeled after https://python-bloggers.com
    /2022/03/multiple-linear-regression-using-tensorflow/ adapted for
    torch and optimized for pairwise iteration.

    Note: Could be *heavily* optimized due to the pairwise iteration overlap.
    '''
    device = get_device(**logger)
    regressions = len(M.index) * len(G.index)
    logger.start_timer('info', 'Running full regression.')

    nrows, ncols = C.shape[0], C.shape[1] + 1
    Ct = torch.tensor(C.to_numpy()).to(device)
    logger.time('Converted C to tensor in {l} seconds')
    one = torch.ones((nrows, 1), dtype=torch.float64).to(device)
    oneX: torch.Tensor = torch.concat((one, one, Ct), 1).to(device)
    logger.time('Created root oneX tensor in {l} seconds')

    index = pd.MultiIndex(
        levels=[[], []],
        codes=[[], []],
        names=['meth_site', 'gene_site'],
    )
    categories = ['const', 'gt'] + C.columns.to_list()
    columns = []
    for category in categories:
        names_group = (
            category + '_est',
            category + '_err',
            category + '_t',
            category + '_p',
        )
        columns.extend(
            names for included, names in zip(include, names_group) if included
        )
    out_df = pd.DataFrame(
        index=index,
        columns=columns,
    )
    logger.time('Set up output dataframe')

    df = nrows - ncols
    dist = torch.distributions.studentT.StudentT(df).log_prob

    inner_logger = logger.alias()
    inner_logger.start_timer('info', 'Calculating chunks...')
    for meth_site, M_row in M.iterrows():
        y = torch.tensor(M_row).to(device)

        for gene_site, G_row in G.iterrows():
            results = []
            oneX[:, 1] = torch.tensor(G_row.to_numpy()).to(device)
            XtX = oneX.mT.matmul(oneX)
            Xty = oneX.mT.matmul(y)
            beta = XtX.inverse().matmul(Xty)
            if include[0]:
                results.append(beta.cpu().numpy())
            if include[1] or include[2] or include[3]:
                err = y - oneX.matmul(beta)
                s2 = err.T.matmul(err) / (nrows - ncols - 1)
                cov_beta = s2 * XtX.inverse()
                std_err = torch.diagonal(cov_beta).sqrt()
            if include[1]:
                results.append(std_err.cpu().numpy())
            if include[2] or include[3]:
                t_stats = beta / std_err
            if include[2]:
                results.append(t_stats.cpu().numpy())
            if include[3]:
                p_value = dist(t_stats)
                results.append(p_value.cpu().numpy())

            row = pd.DataFrame(
                np.array(list(zip(results))).reshape(1, -1),
                index=[(meth_site, gene_site)],
                columns=columns,
            )
            out_df = pd.concat((out_df, row))

            if inner_logger.timer_count % 500 == 0:
                inner_logger.time(
                    'Completed regression {i}/{0} in {l} seconds. Average'
                    ' regression time: {a} seconds',
                    regressions,
                )
                inner_logger.info(
                    'Estimated time remaining: {0} seconds',
                    inner_logger.remaining_time(regressions),
                )
            else:
                inner_logger.time()

    return out_df


def test() -> None:
    M, G, C = generate_data(100, 100, 100)
    C['sex'] = C['sex'].astype(int)
    print(regression_full(M, G, C))


if __name__ == '__main__':
    test()
