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
    gt_count, mt_count = len(G), len(M)
    mt_site_names = numpy.array(M.index.values)
    gt_site_names = numpy.array(G.index.values)
    df = nrows - ncols - 1
    logger.info('Running with {0} degrees of freedom', df)
    dft_sqrt = torch.tensor(df, device=device, dtype=dtype).sqrt()
    log_prob = torch.distributions.studentT.StudentT(df).log_prob
    M_np = M.to_numpy()
    index_names = ['gt_site', 'mt_site']
    columns = ['const_p', 'mt_p'] + [val + '_p' for val in C.columns]
    last_index = 0
    results = []
    if loci_per_chunk:
        chunk_count = math.ceil(len(M) / loci_per_chunk)
        logger.info('Initializing output directory')
        initialize_dir(output_dir, **logger)
    if p_thresh is not None:
        output_sizes = []
        indices_list = []

    logger.start_timer('info', 'Running regression_full...')
    Ct: torch.Tensor = torch.tensor(
        C.to_numpy(), device=device, dtype=dtype
    ).repeat(gt_count, 1, 1)
    logger.time('Converted C to tensor in {l} seconds')
    Gt: torch.Tensor = torch.tensor(
        G.to_numpy(), device=device, dtype=dtype
    ).unsqueeze(2)
    logger.time('Converted G to tensor in {l} seconds')
    ones = torch.ones((gt_count, nrows, 1), device=device, dtype=dtype)
    logger.time('Created ones in {l} seconds')
    X: torch.Tensor = torch.cat((ones, Gt, Ct), 2)
    del Ct, Gt, ones
    logger.time('Created X in {l} seconds')
    Xt = X.mT
    logger.time('Transposed X in {l} seconds')
    XtXi = Xt.bmm(X).inverse()
    logger.time('Calculated XtXi in {l} seconds')
    XtXi_diag_sqrt = torch.diagonal(XtXi, dim1=1, dim2=2).sqrt()
    logger.time('Calculated XtXi_diag in {l} seconds')
    XtXi_Xt = XtXi.bmm(Xt)
    del Xt, XtXi
    logger.time('Calculated XtXi_Xt in {l} seconds')
    logger.time('Calculated X constants in {t} seconds')
    if allocated_memory := torch.cuda.memory_allocated():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.empty_cache()
        logger.info(
            'CUDA device memory: {0} MB allocated by constants out of {1}'
            ' MB total',
            allocated_memory / 1_000_000,
            total_memory / 1_000_000,
        )
    inner_logger = logger.alias()
    inner_logger.start_timer('info', 'Calculating chunks...')
    with Pool() as pool:
        for index, M_row in enumerate(M_np, 1):
            Y = torch.tensor(M_row, device=device, dtype=dtype)
            B = XtXi_Xt.matmul(Y)
            E = (Y.unsqueeze(1) - X.bmm(B.unsqueeze(2))).squeeze(2)
            scalars = (torch.sum(E * E, 1)).view((-1, 1)).sqrt() / dft_sqrt
            S = XtXi_diag_sqrt * scalars
            T = B / S
            P = torch.exp(log_prob(T))
            if p_thresh is None:
                results.append(P)
            else:
                indices = P[:, 1] <= p_thresh
                output_sizes.append(indices.count_nonzero().item())
                indices_list.append(indices)
                results.append(P[indices])
            if loci_per_chunk and (
                index % loci_per_chunk == 0 or index == mt_count
            ):
                mt_site_name_chunk = mt_site_names[last_index:index]
                last_index = index
                if p_thresh is None:
                    mt_sites = mt_site_name_chunk.repeat(gt_count)
                    gt_sites = numpy.tile(gt_site_names, len(results))
                else:
                    mt_sites = mt_site_name_chunk.repeat(output_sizes)
                    mask = torch.cat(indices_list).cpu().numpy()
                    gt_sites = numpy.tile(gt_site_names, len(results))[mask]
                index_chunk = [gt_sites, mt_sites]

                file_name = str(logger.current_count + 1) + '.csv'
                file_path = os.path.join(output_dir, file_name)
                out = pandas.DataFrame(
                    torch.cat(results).cpu().numpy(),
                    index=index_chunk,
                    columns=columns,
                )
                out.index.set_names(index_names, inplace=True)
                logger.count(
                    'Saving part {i}/{0}:',
                    chunk_count,
                )
                pool.apply_async(
                    save_dataframe_part,
                    (out, file_path, logger.current_count),
                    dict(logger),
                )

                inner_logger.time(
                    'Completed chunk {i}/{0} in {l} seconds.'
                    ' Average chunk time: {a} seconds',
                    mt_count,
                )

                results.clear()
                if p_thresh is not None:
                    output_sizes.clear()
                    indices_list.clear()

        logger.time('Looped over methylation loci in {l} seconds')
        logger.time('Calculated regression_full in {t} seconds')

        if loci_per_chunk:
            logger.time('Waiting for chunks to save...')
            pool.close()
            pool.join()
            logger.time('Finished waiting for chunks to save in {l} seconds')
            return

        logger.start_timer('info', 'Generating dataframe from results...')
        if p_thresh is None:
            mt_sites = mt_site_names.repeat(gt_count)
            gt_sites = numpy.tile(gt_site_names, len(results))
        else:
            mt_sites = mt_site_names.repeat(output_sizes)
            mask = torch.cat(indices_list).cpu().numpy()
            gt_sites = numpy.tile(gt_site_names, len(results))[mask]
        index_chunk = [gt_sites, mt_sites]
        logger.time('Finished creating indices in {l} seconds')
        out = pandas.DataFrame(
            torch.cat(results).cpu().numpy(),
            index=index_chunk,
            columns=columns,
        )
        out.index.set_names(index_names, inplace=True)
        logger.time('Finished creating preliminary dataframe in {l} seconds')
        logger.time('Created output dataframe in {t} total seconds')
        return out


def test() -> None:
    M, G, C = generate_data(100, 1000, 1000)
    logger = Logger(carry_data={'use_cpu': True})
    print(regression_full(M, G, C, p_thresh=0.3, **logger))


if __name__ == '__main__':
    test()
