import math
import os
from multiprocessing import Pool
from typing import Optional

import numpy as np
import pandas as pd
import torch

from tecpg.config import get_device
from tecpg.import_data import initialize_dir, save_dataframe_part
from tecpg.logger import Logger
from tecpg.test_data import generate_data


def regression_full(
    M: pd.DataFrame,
    G: pd.DataFrame,
    C: pd.DataFrame,
    meth_loci_per_chunk: Optional[int] = None,
    output_dir: Optional[str] = None,
    *,
    logger: Logger = Logger(),
) -> Optional[pd.DataFrame]:
    if (output_dir is None) != (meth_loci_per_chunk is None):
        error = 'Output dir and chunk size must be defined together.'
        logger.error(error)
        raise ValueError(error)

    logger.info('Initializing regression variables')
    device = get_device(**logger)
    nrows, ncols = C.shape[0], C.shape[1] + 1
    gt_count = len(G)
    gt_site_names = list(G.index.values)
    df = nrows - ncols - 1
    dft = torch.tensor(df).to(device)
    log_prob = torch.distributions.studentT.StudentT(df).log_prob
    M_np = M.to_numpy()
    results = []
    if meth_loci_per_chunk:
        columns = ['const_p', 'gt_p'] + [val + '_p' for val in C.columns]
        chunk_count = math.ceil(len(M) / meth_loci_per_chunk)
        mt_site_names = list(M.index.values)
        logger.info('Initializing output directory')
        initialize_dir(output_dir, **logger)

    logger.start_timer('info', 'Running regression_full...')
    C: torch.Tensor = (
        torch.tensor(C.to_numpy()).to(device).repeat(gt_count, 1, 1)
    )
    G: torch.Tensor = torch.tensor(G.to_numpy()).to(device).unsqueeze(2)
    ones = torch.ones((gt_count, nrows, 1), dtype=torch.float64).to(device)
    X: torch.Tensor = torch.cat((ones, G, C), 2)
    Xt = X.mT
    XtXi = Xt.bmm(X).inverse()
    XtXi_Xt = XtXi.bmm(Xt)
    logger.time('Calculated X constants in {l} seconds')
    with Pool() as pool:
        for index, M_row in enumerate(M_np, 1):
            Y = torch.tensor(M_row).to(device)
            B = XtXi_Xt.matmul(Y)
            E = (Y.unsqueeze(1) - X.bmm(B.unsqueeze(2))).squeeze(2)
            scalars = (torch.sum(E * E, 1) / dft).view((-1, 1))
            S = (torch.diagonal(XtXi, dim1=1, dim2=2) * scalars).sqrt()
            T = B / S
            P = torch.exp(log_prob(T))
            results.append(P.numpy())
            if meth_loci_per_chunk and index % meth_loci_per_chunk == 0:
                mt_site_name_chunk = mt_site_names[
                    index - meth_loci_per_chunk : index
                ]
                index_chunk = pd.MultiIndex.from_product(
                    [mt_site_name_chunk, gt_site_names],
                    names=['mt_site', 'gt_site'],
                )
                file_name = str(logger.current_count + 1) + '.csv'
                file_path = os.path.join(output_dir, file_name)
                out = pd.DataFrame(
                    np.concatenate(results),
                    index=index_chunk,
                    columns=columns,
                )
                logger.count(
                    'Saving part {i}/{0}:',
                    chunk_count,
                )
                pool.apply_async(
                    save_dataframe_part,
                    (out, file_path, logger.current_count),
                    dict(logger),
                )
                results.clear()

        logger.time('Looped over methylation loci in {l} seconds')
        logger.time('Calculated regression_full in {t} seconds')

        if meth_loci_per_chunk:
            logger.time('Waiting for chunks to save...')
            pool.close()
            pool.join()
            logger.time('Finished waiting for chunks to save in {l} seconds')
        else:
            return np.concatenate(results)


def test() -> None:
    M, G, C = generate_data(100, 100, 100)
    print(regression_full(M, G, C, 10))


if __name__ == '__main__':
    test()
