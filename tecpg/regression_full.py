import math
import os
import time
from multiprocessing import Pool
from typing import Literal, Optional

import numpy
import pandas
import torch

from .config import get_device
from .import_data import initialize_dir, save_dataframe_part
from .logger import Logger
from .test_data import generate_data


def create_prob(df: int, device: torch.device, dtype: torch.dtype):
    offset = torch.tensor(
        -0.5 * math.log(df)
        - 0.5 * math.log(math.pi)
        - math.lgamma(0.5 * df)
        + math.lgamma(0.5 * (df + 1.0)),
        device=device,
        dtype=dtype,
    )
    scalar = torch.tensor(0.5 * (df + 1.0), device=device, dtype=dtype)

    def prob(value: torch.Tensor):
        return (offset - torch.log1p(value ** 2.0 / df) * scalar).exp()

    return prob


def regression_full(
    M: pandas.DataFrame,
    G: pandas.DataFrame,
    C: pandas.DataFrame,
    M_annot: Optional[pandas.DataFrame] = None,
    G_annot: Optional[pandas.DataFrame] = None,
    region: Literal['all', 'cis', 'distal', 'trans'] = 'all',
    window: Optional[int] = None,
    loci_per_chunk: Optional[int] = None,
    p_thresh: Optional[float] = None,
    output_dir: Optional[str] = None,
    methylation_only: bool = True,
    *,
    logger: Logger = Logger(),
) -> Optional[pandas.DataFrame]:
    if (output_dir is None) != (loci_per_chunk is None):
        error = 'Output dir and chunk size must be defined together.'
        logger.error(error)
        raise ValueError(error)
    if region not in ['all', 'cis', 'distal', 'trans']:
        error = f'Region {region} not valid. Use all, cis, distal, or trans.'
        logger.error(error)
        raise ValueError(error)
    if region != 'all' and (G_annot is None or M_annot is None):
        error = (
            f'Missing M or G annotation files using region filtration {region}'
        )
        logger.error(error)
        raise ValueError(error)
    if region in ['cis', 'distal'] and window is None:
        error = f'Window is None for region filtration {region}'
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
    prob = create_prob(df, device, dtype)
    M_np = M.to_numpy()
    index_names = ['gt_site', 'mt_site']
    if methylation_only:
        columns = ['mt_p']
    else:
        columns = ['const_p', 'mt_p'] + [val + '_p' for val in C.columns]

    last_index = 0
    results = []
    filtration = True
    output_sizes = []
    if region != 'all':
        region_indices_list = []
    if p_thresh is None:
        p_indices_list = None
        if region == 'all':
            filtration = False
            output_sizes = gt_count
    else:
        p_indices_list = []

    if output_dir is not None:
        chunk_count = math.ceil(len(M) / loci_per_chunk)
        logger.info('Initializing output directory')
        initialize_dir(output_dir, **logger)

    logger.start_timer('info', 'Running regression_full...')
    if region != 'all':
        M_annot = (
            M_annot.drop(columns=['chromEnd', 'score', 'strand'])
            .reindex(M.index)
            .replace({'X': -1, 'Y': -2})
        )
        G_annot = (
            G_annot.drop(columns=['chromEnd', 'score', 'strand'])
            .reindex(G.index)
            .replace({'X': -1, 'Y': -2})
        )
        M_chrom, M_pos = M_annot.to_numpy().T
        G_chrom, G_pos = G_annot.to_numpy().T
        M_chrom_t = torch.tensor(M_chrom, device=device, dtype=torch.int)
        M_pos_t = torch.tensor(M_pos, device=device, dtype=torch.int)
        G_chrom_t = torch.tensor(G_chrom, device=device, dtype=torch.int)
        G_pos_t = torch.tensor(G_pos, device=device, dtype=torch.int)

    Ct: torch.Tensor = torch.tensor(
        C.to_numpy(), device=device, dtype=dtype
    ).repeat(gt_count, 1, 1)
    Gt: torch.Tensor = torch.tensor(
        G.to_numpy(), device=device, dtype=dtype
    ).unsqueeze(2)
    ones = torch.ones((gt_count, nrows, 1), device=device, dtype=dtype)
    X: torch.Tensor = torch.cat((ones, Gt, Ct), 2)
    del Gt, Ct, ones
    Xt = X.mT
    XtXi = Xt.bmm(X).inverse()
    XtXi_diag_sqrt = torch.diagonal(XtXi, dim1=1, dim2=2).sqrt()
    XtXi_Xt = XtXi.bmm(Xt)
    del Xt, XtXi
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
    inner_logger.start_timer('info', 'Calculating regression...')
    with Pool() as pool:
        for index, M_row in enumerate(M_np, 1):
            Y = torch.tensor(M_row, device=device, dtype=dtype)

            if region == 'all':
                B = XtXi_Xt.matmul(Y)
                E = (Y.unsqueeze(1) - X.bmm(B.unsqueeze(2))).squeeze(2)
                scalars = (torch.sum(E * E, 1)).view((-1, 1)).sqrt() / dft_sqrt
                S = XtXi_diag_sqrt * scalars
                T = B / S
                P = prob(T)
            else:
                if region == 'cis':
                    region_indices = (
                        (M_chrom_t[index - 1, None] == G_chrom_t)
                        .logical_and(
                            M_pos_t[index - 1, None] < G_pos_t + window
                        )
                        .logical_and(
                            M_pos_t[index - 1, None] > G_pos_t - window
                        )
                    )
                elif region == 'distal':
                    region_indices = (
                        M_chrom_t[index - 1, None] == G_chrom_t
                    ).logical_and(
                        (
                            M_pos_t[index - 1, None] < G_pos_t - window
                        ).logical_or(
                            M_pos_t[index - 1, None] > G_pos_t + window
                        )
                    )
                elif region == 'trans':
                    region_indices = M_chrom_t[index - 1, None] != G_chrom_t

                B = XtXi_Xt[region_indices].matmul(Y)
                E = (
                    Y.unsqueeze(1) - X[region_indices].bmm(B.unsqueeze(2))
                ).squeeze(2)
                scalars = (torch.sum(E * E, 1)).view((-1, 1)).sqrt() / dft_sqrt
                S = XtXi_diag_sqrt[region_indices] * scalars

                region_indices_list.append(region_indices)

            T = B / S
            if methylation_only:
                T = T[:, 1:2]
            P = prob(T)

            if p_thresh is not None:
                p_indices = P[:, 0 if methylation_only else 1] <= p_thresh
                p_indices_list.append(p_indices)
                P = P[p_indices]
            if filtration:
                output_sizes.append(len(P))
            results.append(P)

            if loci_per_chunk and (
                index % loci_per_chunk == 0 or index == mt_count
            ):
                mt_sites = mt_site_names[last_index:index].repeat(output_sizes)
                last_index = index
                if region != 'all':
                    region_mask = torch.cat(region_indices_list).cpu().numpy()
                if p_thresh is None:
                    if region == 'all':
                        gt_sites = numpy.tile(gt_site_names, len(results))
                    else:
                        gt_sites = numpy.tile(gt_site_names, len(results))[
                            region_mask
                        ]
                else:
                    mask = torch.cat(p_indices_list).cpu().numpy()
                    if region == 'all':
                        gt_sites = numpy.tile(gt_site_names, len(results))[
                            mask
                        ]
                    else:
                        gt_sites = numpy.tile(gt_site_names, len(results))[
                            region_mask
                        ][mask]
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
                if filtration:
                    output_sizes.clear()
                if p_thresh is not None:
                    p_indices_list.clear()
                if region != 'all':
                    region_indices_list.clear()

        logger.time('Looped over methylation loci in {l} seconds')
        logger.time('Calculated regression_full in {t} seconds')

        if loci_per_chunk:
            logger.time('Waiting for chunks to save...')
            pool.close()
            pool.join()
            logger.time('Finished waiting for chunks to save in {l} seconds')
            return

        logger.start_timer('info', 'Generating dataframe from results...')
        if region != 'all':
            region_mask = torch.cat(region_indices_list).cpu().numpy()
        mt_sites = mt_site_names.repeat(output_sizes)
        if p_indices_list is None:
            if region == 'all':
                gt_sites = numpy.tile(gt_site_names, len(results))
            else:
                gt_sites = numpy.tile(gt_site_names, len(results))[region_mask]
        else:
            mask = torch.cat(p_indices_list).cpu().numpy()
            if region == 'all':
                gt_sites = numpy.tile(gt_site_names, len(results))[mask]
            else:
                gt_sites = numpy.tile(gt_site_names, len(results))[
                    region_mask
                ][mask]
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
    M, G, C, M_annot, G_annot = generate_data(300, 5000, 5000, True)
    M_annot.set_index('name', inplace=True)
    G_annot.set_index('name', inplace=True)
    logger = Logger(carry_data={'use_cpu': False})
    print(
        regression_full(
            M, G, C, M_annot, G_annot, 'all', p_thresh=0.3, **logger
        )
    )


def test_prob() -> None:
    torch.cuda.empty_cache()
    df = 200
    device = torch.device('cuda')
    dtype = torch.float32
    prob_one = lambda t: torch.distributions.StudentT(df).log_prob(t).exp()
    prob_two = create_prob(df, device, dtype)
    test = torch.rand((100_000_000, 4), device=device, dtype=dtype)
    total_time_one = 0
    total_time_two = 0
    runs = 10
    for _ in range(runs):
        start_time = time.perf_counter()
        prob_one(test)
        total_time_one += time.perf_counter() - start_time
    for _ in range(runs):
        start_time = time.perf_counter()
        prob_two(test)
        total_time_two += time.perf_counter() - start_time
    print(total_time_one / runs, total_time_two / runs)


if __name__ == '__main__':
    test()
