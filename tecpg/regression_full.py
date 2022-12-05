import math
import os
from multiprocessing import Pool
from typing import Optional

import numpy
import pandas
import torch

from .config import get_device
from .import_data import initialize_dir, save_dataframe_part
from .logger import Logger
from .test_data import generate_data


def regression_full(
    M: pandas.DataFrame,
    G: pandas.DataFrame,
    C: pandas.DataFrame,
    loci_per_chunk: Optional[int] = None,
    p_thresh: Optional[float] = None,
    output_dir: Optional[str] = None,
    *,
    logger: Logger = Logger(),
) -> Optional[pandas.DataFrame]:
    if (output_dir is None) != (loci_per_chunk is None):
        error = 'Output dir and chunk size must be defined together.'
        logger.error(error)
        raise ValueError(error)

    logger.info('Initializing regression variables')
    device = get_device(**logger)
    dtype = torch.float32
    nrows, ncols = C.shape[0], C.shape[1] + 1
    mt_count, gt_count = len(M), len(G)
    gt_site_names, mt_site_names = list(G.index.values), list(M.index.values)
    df = nrows - ncols - 1
    logger.info('Running with {0} degrees of freedom', df)
    dft = torch.tensor(df, device=device, dtype=dtype)
    log_prob = torch.distributions.studentT.StudentT(df).log_prob
    G_np = G.to_numpy()
    results = []
    columns = ['const_p', 'mt_p'] + [val + '_p' for val in C.columns]
    last_index = 0
    if loci_per_chunk:
        chunk_count = math.ceil(len(G) / loci_per_chunk)
        logger.info('Initializing output directory')
        initialize_dir(output_dir, **logger)

    logger.start_timer('info', 'Running regression_full...')
    C: torch.Tensor = torch.tensor(
        C.to_numpy(), device=device, dtype=dtype
    ).repeat(mt_count, 1, 1)
    logger.time('Converted C to tensor in {l} seconds')
    M: torch.Tensor = torch.tensor(
        M.to_numpy(), device=device, dtype=dtype
    ).unsqueeze(2)
    logger.time('Converted M to tensor in {l} seconds')
    ones = torch.ones((mt_count, nrows, 1), device=device, dtype=dtype)
    logger.time('Created ones in {l} seconds')
    X: torch.Tensor = torch.cat((ones, M, C), 2)
    logger.time('Created X in {l} seconds')
    Xt = X.mT
    logger.time('Transposed X in {l} seconds')
    XtXi = Xt.bmm(X).inverse()
    logger.time('Calculated XtXi in {l} seconds')
    XtXi_Xt = XtXi.bmm(Xt)
    logger.time('Calculated XtXi_Xt in {l} seconds')
    logger.time('Calculated X constants in {t} seconds')
    with Pool() as pool:
        for index, G_row in enumerate(G_np, 1):
            Y = torch.tensor(G_row, device=device, dtype=dtype)
            B = XtXi_Xt.matmul(Y)
            E = (Y.unsqueeze(1) - X.bmm(B.unsqueeze(2))).squeeze(2)
            scalars = (torch.sum(E * E, 1) / dft).view((-1, 1))
            S = (torch.diagonal(XtXi, dim1=1, dim2=2) * scalars).sqrt()
            T = B / S
            P = torch.exp(log_prob(T))
            results.append(P)
            if loci_per_chunk and (
                index % loci_per_chunk == 0 or index == gt_count
            ):
                gt_site_name_chunk = gt_site_names[last_index:index]
                last_index = index
                index_chunk = pandas.MultiIndex.from_product(
                    [gt_site_name_chunk, mt_site_names],
                    names=['gt_site', 'mt_site'],
                )
                file_name = str(logger.current_count + 1) + '.csv'
                file_path = os.path.join(output_dir, file_name)
                out = pandas.DataFrame(
                    torch.cat(results),
                    index=index_chunk,
                    columns=columns,
                ).astype(float)
                if p_thresh is not None:
                    out = out[out.mt_p > p_thresh]
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

        if loci_per_chunk:
            logger.time('Waiting for chunks to save...')
            pool.close()
            pool.join()
            logger.time('Finished waiting for chunks to save in {l} seconds')
        else:
            logger.start_timer('info', 'Generating dataframe from results...')
            index_chunk = pandas.MultiIndex.from_product(
                [gt_site_names, mt_site_names],
                names=['gt_site', 'mt_site'],
            )
            logger.time('Finished creating indices in {l} seconds')
            out = pandas.DataFrame(
                torch.cat(results),
                index=index_chunk,
                columns=columns,
            ).astype(float)
            logger.time(
                'Finished creating preliminary dataframe in {l} seconds'
            )
            if p_thresh is not None:
                out = out[out.mt_p > p_thresh]
            logger.time('Finished filtering p-values in {l} seconds')
            logger.time('Created output dataframe in {t} total seconds')
            return out


def test() -> None:
    M, G, C = generate_data(100, 100, 100)
    print(regression_full(M, G, C, 10))


if __name__ == '__main__':
    test()
