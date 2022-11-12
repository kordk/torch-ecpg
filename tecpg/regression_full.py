from contextlib import nullcontext
from multiprocessing import Pool
import os
from time import time
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from tecpg.import_data import save_dataframe_part
import torch
from .test_data import generate_data
from .config import get_device
from .logger import Logger


def regression_full(
    M: pd.DataFrame,
    G: pd.DataFrame,
    C: pd.DataFrame,
    include: Tuple[bool, bool, bool, bool] = (True, True, True, True),
    update_period: Optional[float] = 1,
    chunk_size: int = 0,
    p_thresh: Optional[float] = None,
    output_dir: Optional[str] = None,
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

    Logs an update on the completion of the function every update_period
    seconds. If update_period is None, these logs will be omitted.

    If chunk_size is greater than 0, the output will be chunked with
    chunk_size regressions per chunk. The chunks will be saved in
    output_dir.

    The p_thresh argument omits regression results with a gene
    expression p-value below p_thresh. This will force p-value
    calculation and filter the results even if p-values are not included
    in the output. If p_thresh is None (default), regression results
    will not be filtered.

    The parameter include specifies which regression results to include.
    The parameter is a tuple of three booleans for the estimate, the
    error, the t-statistic, and the p-value.

    Multiple Linear Regression modeled after https://python-bloggers.com
    /2022/03/multiple-linear-regression-using-tensorflow/ adapted for
    torch and optimized for pairwise iteration.

    Note: Could be *heavily* optimized due to the pairwise iteration
    overlap and redundance of multiple matrix multiplications with
    similar inputs.
    '''
    device = get_device(**logger)
    regressions = len(M.index) * len(G.index)
    filter_p = p_thresh is not None
    logger.start_timer('info', 'Running full regression.')

    if output_dir is None and chunk_size:
        message = 'Output directory of None is not valid for chunk_size > 0'
        logger.error(message)
        raise ValueError(message)

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

    df = nrows - ncols - 1
    dist = torch.distributions.studentT.StudentT(df).log_prob

    inner_logger = logger.alias()
    inner_logger.start_timer('info', 'Calculating chunks...')
    i = 0
    last_time = time()
    with Pool() if chunk_size else nullcontext() as pool:
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
                if include[1] or include[2] or include[3] or filter_p:
                    err = y - oneX.matmul(beta)
                    s2 = err.T.matmul(err) / (nrows - ncols - 1)
                    cov_beta = s2 * XtX.inverse()
                    std_err = torch.diagonal(cov_beta).sqrt()
                if include[1]:
                    results.append(std_err.cpu().numpy())
                if include[2] or include[3] or filter_p:
                    t_stats = beta / std_err
                if include[2]:
                    results.append(t_stats.cpu().numpy())
                if include[3] or filter_p:
                    p_value = dist(t_stats)
                    p_value_np = p_value.cpu().numpy()
                    results.append(p_value_np)

                if not filter_p or p_value_np[1] >= p_thresh:
                    i += 1

                    row = pd.DataFrame(
                        np.array(list(zip(*results))).reshape(1, -1),
                        index=[(meth_site, gene_site)],
                        columns=columns,
                    )
                    out_df = pd.concat((out_df, row))

                    if chunk_size and i % chunk_size == 0:
                        file_name = str(logger.current_count + 1) + '.csv'
                        file_path = os.path.join(output_dir, file_name)
                        logger.count(
                            'Saving part {i}'
                            + ('' if filter_p else '/{0}')
                            + ': ',
                            regressions // chunk_size,
                        )
                        pool.apply_async(
                            save_dataframe_part,
                            (out_df, file_path),
                            dict(logger),
                        )

                        del out_df
                        out_df = pd.DataFrame(
                            index=index,
                            columns=columns,
                        )

                if update_period is not None:
                    if time() - last_time > update_period:
                        last_time = time()
                        inner_logger.time(
                            'Completed regression {i}/{0} in {l} seconds.'
                            ' Average regression time: {a} seconds',
                            regressions,
                        )
                        inner_logger.info(
                            'Estimated time remaining: {0} seconds',
                            inner_logger.remaining_time(regressions),
                        )
                    else:
                        inner_logger.time()

        if chunk_size and len(out_df):
            file_name = str(logger.current_count + 1) + '.csv'
            file_path = os.path.join(output_dir, file_name)
            logger.count(
                'Saving part {i}' + ('' if filter_p else '/{0}') + ': ',
                regressions // chunk_size,
            )
            pool.apply_async(
                save_dataframe_part,
                (out_df, file_path),
                dict(logger),
            )

        pool.close()
        pool.join()

    logger.time('Calculated regression_full in {t} seconds')
    if chunk_size == 0:
        return out_df


def test() -> None:
    M, G, C = generate_data(100, 100, 100)
    C['sex'] = C['sex'].astype(int)
    print(regression_full(M, G, C))


if __name__ == '__main__':
    test()
