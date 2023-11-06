import math
import os
import time
from multiprocessing import Pool
from typing import Literal, Optional

import numpy
import pandas
import torch
from colorama import Fore as colors

from .config import DTYPE, get_device
from .helper import trim_dataframes
from .import_data import initialize_dir, save_dataframe_part
from .logger import Logger
from .test_data import generate_data


def create_studentt_p(df: int, device: torch.device, dtype: torch.dtype):
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
        return (offset - torch.log1p(value**2.0 / df) * scalar).exp()

    return prob


def create_normal_p(device: torch.device, dtype: torch.dtype):
    scalar = (
        torch.tensor(2, device=device, dtype=dtype).sqrt().reciprocal().neg()
    )

    def prob(value: torch.Tensor) -> torch.Tensor:
        return torch.erf(scalar * value.abs()) + 1

    return prob


def regression_full(
    M: pandas.DataFrame,
    G: pandas.DataFrame,
    C: pandas.DataFrame,
    M_annot: Optional[pandas.DataFrame] = None,
    G_annot: Optional[pandas.DataFrame] = None,
    region: Literal['all', 'cis', 'distal', 'trans'] = 'all',
    window_base: Optional[int] = None,
    downstream: Optional[int] = None,
    upstream: Optional[int] = None,
    gene_loci_per_chunk: Optional[int] = None,
    meth_loci_per_chunk: Optional[int] = None,
    p_thresh: Optional[float] = None,
    output_dir: Optional[str] = None,
    methylation_only: bool = True,
    p_only: bool = False,
    file_format: str = '{meth_chunk}-{gene_chunk}.csv',
    *,
    logger: Logger = Logger(),
) -> Optional[pandas.DataFrame]:
    '''
    Calculates the multiple linear regression of the input dataframes M,
    G, and C, being methylation beta values, gene expression values, and
    covariates using torch. This is done for every pair of methylation
    id and gene id. The regression formula is G ~ M + C1 + C2 + ...

    The p-values are the Student's T CDF function evaluated on the t
    statistic for each regression. Torch does not currently support the
    Student's T CDF function or any function that would help to
    implement it in python. Instead, the normal distribution CDF is used
    as an approximation of the Student's T CDF. For more degrees of
    freedom (ie. more samples, fewer covariates), this approximation
    is more accurate.

    M_annot and G_annot are annotation files that provide the positions
    of each methylation and gene expression id. They are optional and
    only required for region filtration.

    Cis and distal filtration only allow regressions where the
    methylation locus position is within the region with the origin of
    window_base bases away from the gene transcript start site starting
    upstream bases to one side and downstream to another, with the
    orientation of the region dictated by the strand of the gene
    expression locus. Trans analyses only allow regressions with
    methylation and gene expression ids on the same chromosome. The last
    region filtration mode, all, does not filter and instead keeps all
    regressions.

    P-value filtration filters the output of the regressions by only
    including regressions with  p-value below p_thresh.

    For larger inputs, one may encounter memory limits. If this is the
    case, there are two ways of chunking the input data to avoid these
    limits: methylation chunking and gene expression chunking. Specify
    the number of meth_loci_per_chunk and gene_loci_per_chunk. Gene
    expression chunking is less detrimental to performance than
    methylation chunking, but both shuold be avoided as they sacrifice
    parallelization and speed. Both chunking methods are optional, and
    they can be combined together as well. Use the chunks command to
    estimate how many gene_loci_per_chunk to use for given settings. If
    no chunking is used, the output dataframe is returned. If chunking
    is used, chunks are saved to output files in output_dir.

    The methylation_only boolean option which defaults to true
    determines whether only methylation results should be saved in the
    output. If false, the intercept, methylation, and covariate results
    are saved. Regardless of this value, the intercept and covariates
    are used in the regression calculation. This parameter only affects
    the inclusion of output data.

    The p_only boolean option which defaults to false determines whether
    to include the estimate, standard error, Student's T statistic, and
    p-value (if false) or just the p-value (if true) for  faster saving
    time and lower output size.

    The file_format parameter is  string that determines the file name
    of each chunk saved in output_dir based on  formatting string with
    the parameters meth_chunk for the methylation chunk number and
    gene_chunk for the gene expression chunk number. The string is
    formatted using:
        file_format.format(
            meth_chunk=meth_index_str,
            gene_chunk=gene_index_str,
        )
    '''
    chunking = (
        gene_loci_per_chunk is not None or meth_loci_per_chunk is not None
    )

    # Detect errors in the input values
    if (output_dir is None) != (not chunking):
        error = 'Output dir and chunk size must be defined together.'
        logger.error(error)
        raise ValueError(error)
    if region not in ['all', 'cis', 'distal', 'trans']:
        error = f'Region {region} not valid. Use all, cis, distal, or trans.'
        logger.error(error)
        raise ValueError(error)
    if region != 'all' and (G_annot is None or G_annot is None):
        error = (
            f'Missing M or G annotation files using region filtration {region}'
        )
        logger.error(error)
        raise ValueError(error)
    if region in ['cis', 'distal'] and (
        window_base is None or downstream is None or upstream is None
    ):
        error = (
            f'Region filtration {region} requires window_base, downstream, and'
            ' upstream not to be None'
        )
        logger.error(error)
        raise ValueError(error)

    # Prepare annotation tensors if region filtration is used
    if region != 'all':
        logger.info('Initializing region filtration')
        G_annot = (
            G_annot.drop(columns=['chromEnd', 'score'])
            .reindex(G.index)
            .replace({'X': -1, 'Y': -2, '+': 1, '-': -1})
            .dropna()
        )
        M_annot = (
            M_annot.drop(columns=['chromEnd', 'score', 'strand'])
            .reindex(M.index)
            .replace({'X': -1, 'Y': -2})
            .dropna()
        )

        trim_dataframes([G_annot, G], **logger)
        trim_dataframes([M_annot, M], **logger)

        G_chrom, G_pos, G_strand = G_annot.to_numpy().T.astype(int)
        M_chrom, M_pos = M_annot.to_numpy().T.astype(int)

        G_chrom_t = torch.tensor(G_chrom, device=device, dtype=torch.int8)
        G_pos_t = torch.tensor(G_pos, device=device, dtype=torch.int32)
        G_strand_t = torch.tensor(G_strand, device=device, dtype=torch.int8)

    # Initializes some constants
    logger.info('Initializing regression variables')
    device = get_device(**logger)
    dtype = DTYPE
    if meth_loci_per_chunk is not None:
        meth_chunk_count = math.ceil(len(M) / meth_loci_per_chunk)
    nrows, ncols = C.shape[0], C.shape[1] + 1
    G_np = G.to_numpy()
    gt_count = len(G)
    gt_site_names = numpy.array(G.index.values)
    df = nrows - ncols - 1
    logger.info('Running with {0} degrees of freedom', df)
    dft_sqrt = torch.tensor(df, device=device, dtype=dtype).sqrt()
    # prob = create_studentt_p(df, device, dtype)
    normal_p = create_normal_p(device, dtype)
    if gene_loci_per_chunk is not None:
        chunk_count = math.ceil(len(G) / gene_loci_per_chunk)

    if chunking:
        logger.info('Initializing output directory')
        initialize_dir(output_dir, **logger)

    # Determines the column names for the output dataframe
    index_names = ['gt_id', 'mt_id']
    if p_only:
        if methylation_only:
            columns = ['mt_p']
        else:
            columns = ['const_p', 'mt_p'] + [val + '_p' for val in C.columns]
    else:
        categories = (
            ['mt']
            if methylation_only
            else (['const', 'mt'] + C.columns.to_list())
        )
        suffixes = ['_est', '_err', '_t', '_p']
        columns = [
            column + suffix for suffix in suffixes for column in categories
        ]

    # Create covariate tensor
    if meth_loci_per_chunk is None:
        Ct: torch.Tensor = torch.tensor(
            C.to_numpy(), device=device, dtype=dtype
        ).repeat(len(M), 1, 1)
    else:
        Ct: torch.Tensor = torch.tensor(
            C.to_numpy(), device=device, dtype=dtype
        ).repeat(meth_loci_per_chunk, 1, 1)

    # Initialize variables for use in the regression calculation loop
    end_index = 0
    results = []
    filtration = True
    output_sizes = []
    if region != 'all':
        region_indices_list = []
    if p_thresh is None:
        p_indices_list = None
        if region == 'all':
            filtration = False
    else:
        p_indices_list = []

    # Create methylation chunk (mc_) and chunk saving (inner_) logger
    mc_logger = logger.alias()
    mc_logger.info_color = colors.GREEN
    inner_logger = mc_logger.alias()

    # Use the multiprocessing pool
    with Pool() as pool:
        # Loop for methylation chunks or ran once with index 0 if no
        # methylation chunking
        for meth_chunk_index in (
            (0,) if meth_loci_per_chunk is None else range(meth_chunk_count)
        ):
            # Log methylation chunk index
            logger.info('STARTING METHYLATION CHUNK {0}', meth_chunk_index + 1)
            mc_logger.info_template = (
                '[CHUNK' + str(meth_chunk_index + 1) + '{modifier}] {message}'
            )
            mc_logger.current_count = 0

            # Slice M into M_chunk or copy for no methylation chunking
            if meth_loci_per_chunk is not None:
                start_index = end_index
                end_index = (meth_chunk_index + 1) * meth_loci_per_chunk
                M_chunk = M[start_index:end_index]
                if len(M_chunk) < meth_loci_per_chunk:
                    Ct = Ct[: len(M_chunk)]
            else:
                M_chunk = M

            mt_count = len(M_chunk)
            mt_site_names = numpy.array(M_chunk.index.values)
            if region == 'all' and p_thresh is None:
                output_sizes = mt_count

            mc_logger.start_timer('info', 'Running regression_full...')

            # Create methylation loci chromosome and position tensors
            # for the current chunk
            if region != 'all':
                if meth_loci_per_chunk is None:
                    M_chrom_t = torch.tensor(
                        M_chrom, device=device, dtype=torch.int8
                    )
                    M_pos_t = torch.tensor(
                        M_pos, device=device, dtype=torch.int32
                    )
                else:
                    M_chrom_t = torch.tensor(
                        M_chrom[start_index:end_index],
                        device=device,
                        dtype=torch.int8,
                    )
                    M_pos_t = torch.tensor(
                        M_pos[start_index:end_index],
                        device=device,
                        dtype=torch.int32,
                    )

            # Calculate constants for the current methylation chunk to
            # massively increase performance
            Mt: torch.Tensor = torch.tensor(
                M_chunk.to_numpy(), device=device, dtype=dtype
            ).unsqueeze(2)
            ones = torch.ones((mt_count, nrows, 1), device=device, dtype=dtype)
            X: torch.Tensor = torch.cat((ones, Mt, Ct), 2)
            del Mt, ones
            Xt = X.mT
            XtXi = Xt.bmm(X).inverse()
            XtXi_diag_sqrt = torch.diagonal(XtXi, dim1=1, dim2=2).sqrt()
            XtXi_Xt = XtXi.bmm(Xt)
            del Xt, XtXi

            # Display amount of total memory occupied by the constants
            # for the current methylation chunk (if the device is CUDA
            # enabled)
            if allocated_memory := torch.cuda.memory_allocated():
                device_properties: torch.cuda._CudaDeviceProperties = (
                    torch.cuda.get_device_properties(0)
                )
                total_memory: int = device_properties.total_memory
                torch.cuda.empty_cache()
                mc_logger.info(
                    (
                        'CUDA device memory: {0} MB allocated by constants out'
                        ' of {1} MB total'
                    ),
                    allocated_memory / 1_000_000,
                    total_memory / 1_000_000,
                )

            # Inner loop over each gene expression locus
            last_index = 0
            inner_logger.start_timer('info', 'Calculating regression...')
            for index, G_row in enumerate(G_np, 1):
                Y = torch.tensor(G_row, device=device, dtype=dtype)

                if region == 'all':
                    # No region filtration
                    B = XtXi_Xt.matmul(Y)
                    E = (Y.unsqueeze(1) - X.bmm(B.unsqueeze(2))).squeeze(2)
                    del Y
                    scalars = (torch.sum(E * E, 1)).view(
                        (-1, 1)
                    ).sqrt() / dft_sqrt
                    del E
                    S = XtXi_diag_sqrt * scalars
                    del scalars
                else:
                    # Creates a boolean mask for the region filtration
                    if region in ('cis', 'distal'):
                        # The None in the index of the G tensors makes
                        # the shape of the tensor suitable to broadcast
                        # to the shape of the M tensors
                        region_indices_mask = (
                            (G_chrom_t[index - 1, None] == M_chrom_t)
                            .logical_and(
                                G_strand_t[index - 1, None]
                                * (window_base - upstream)
                                < G_pos_t[index - 1, None] - M_pos_t
                            )
                            .logical_and(
                                G_pos_t[index - 1, None] - M_pos_t
                                < (
                                    G_strand_t[index - 1, None]
                                    * (window_base + downstream)
                                )
                            )
                        )
                    elif region == 'trans':
                        region_indices_mask = (
                            G_chrom_t[index - 1, None] != M_chrom_t
                        )

                    B = XtXi_Xt[region_indices_mask].matmul(Y)
                    E = (
                        Y.unsqueeze(1)
                        - X[region_indices_mask].bmm(B.unsqueeze(2))
                    ).squeeze(2)
                    del Y
                    scalars = (torch.sum(E * E, 1)).view(
                        (-1, 1)
                    ).sqrt() / dft_sqrt
                    del E
                    S = XtXi_diag_sqrt[region_indices_mask] * scalars
                    del scalars

                    region_indices_list.append(region_indices_mask)

                # Remove unnecessary values if only methylation is
                # desired
                if methylation_only:
                    B = B[:, 1:2]
                    S = S[:, 1:2]

                T = B / S
                P = normal_p(T)

                # Keep results below p_thresh if supplied
                if p_thresh is not None:
                    p_indices = P[:, 0 if methylation_only else 1] <= p_thresh
                    p_indices_list.append(p_indices)
                    P = P[p_indices]
                    if not p_only:
                        B = B[p_indices]
                        S = S[p_indices]
                        T = T[p_indices]

                if filtration:
                    output_sizes.append(len(P))

                # Append current regression results
                if p_only:
                    results.append(P)
                else:
                    results.append(torch.cat((B, S, T, P), dim=1))

                # Save output when gene chunking is used
                if gene_loci_per_chunk and (
                    index % gene_loci_per_chunk == 0 or index == gt_count
                ):
                    # Filter results
                    gt_sites = gt_site_names[last_index:index].repeat(
                        output_sizes
                    )
                    if filtration:
                        del output_sizes[:]
                    last_index = index
                    if region != 'all':
                        region_mask = (
                            torch.cat(region_indices_list).cpu().numpy()
                        )
                        del region_indices_list[:]
                    if p_thresh is None:
                        if region == 'all':
                            mt_sites = numpy.tile(mt_site_names, len(results))
                        else:
                            mt_sites = numpy.tile(mt_site_names, len(results))[
                                region_mask
                            ]
                    else:
                        mask = torch.cat(p_indices_list).cpu().numpy()
                        del p_indices_list[:]
                        if region == 'all':
                            mt_sites = numpy.tile(mt_site_names, len(results))[
                                mask
                            ]
                        else:
                            mt_sites = numpy.tile(mt_site_names, len(results))[
                                region_mask
                            ][mask]
                    index_chunk = [gt_sites, mt_sites]

                    # Create path to save and file name
                    gene_index_str = str(mc_logger.current_count + 1)
                    meth_index_str = str(meth_chunk_index + 1)
                    file_name = file_format.format(
                        meth_chunk=meth_index_str, gene_chunk=gene_index_str
                    )
                    file_path = os.path.join(output_dir, file_name)

                    # Create output dataframe
                    out = pandas.DataFrame(
                        torch.cat(results).cpu().numpy(),
                        index=index_chunk,
                        columns=columns,
                    )
                    out.index.set_names(index_names, inplace=True)

                    # CUDA memory notice
                    if index == gene_loci_per_chunk and allocated_memory:
                        torch.cuda.empty_cache()
                        allocated_memory = torch.cuda.max_memory_allocated()
                        mc_logger.info(
                            (
                                'CUDA device memory, chunk 1: {0} MB allocated'
                                ' out of {1} MB total. If needed, increase'
                                ' --loci-per-chunk accordingly'
                            ),
                            allocated_memory / 1_000_000,
                            total_memory / 1_000_000,
                        )

                    # Save output with multiprocessing pool
                    mc_logger.count(
                        'Saving part {i}/{0}:',
                        chunk_count,
                    )
                    pool.apply_async(
                        save_dataframe_part,
                        (out, file_path, mc_logger.current_count),
                        dict(mc_logger),
                    )

                    # Report gene chunk time
                    inner_logger.time(
                        (
                            'Completed chunk {i}/{0} in {l} seconds.'
                            ' Average chunk time: {a} seconds'
                        ),
                        chunk_count,
                    )

                    del results[:]

            mc_logger.time('Looped over methylation loci in {l} seconds')
            mc_logger.time('Calculated regression_full in {t} seconds')

            # Filter gene chunking is not used and save results if
            # methylation chunking is used
            if gene_loci_per_chunk is None:
                # Filter results
                mc_logger.start_timer(
                    'info', 'Generating dataframe from results...'
                )
                if region != 'all':
                    region_mask = torch.cat(region_indices_list).cpu().numpy()
                    del region_indices_list[:]
                gt_sites = gt_site_names.repeat(output_sizes)
                if filtration:
                    del output_sizes[:]
                if p_indices_list is None:
                    if region == 'all':
                        mt_sites = numpy.tile(mt_site_names, len(results))
                    else:
                        mt_sites = numpy.tile(mt_site_names, len(results))[
                            region_mask
                        ]
                else:
                    mask = torch.cat(p_indices_list).cpu().numpy()
                    del p_indices_list[:]
                    if region == 'all':
                        mt_sites = numpy.tile(mt_site_names, len(results))[
                            mask
                        ]
                    else:
                        mt_sites = numpy.tile(mt_site_names, len(results))[
                            region_mask
                        ][mask]
                index_chunk = [gt_sites, mt_sites]

                # Create output dataframe
                mc_logger.time('Finished creating indices in {l} seconds')
                out = pandas.DataFrame(
                    torch.cat(results).cpu().numpy(),
                    index=index_chunk,
                    columns=columns,
                )
                out.index.set_names(index_names, inplace=True)
                mc_logger.time(
                    'Finished creating preliminary dataframe in {l} seconds'
                )
                mc_logger.time('Created output dataframe in {t} total seconds')

                # Save results if methylation chunking is used
                if meth_loci_per_chunk is not None:
                    # Create path to save and file name
                    gene_index_str = '1'
                    meth_index_str = str(meth_chunk_index + 1)
                    file_name = file_format.format(
                        meth_chunk=meth_index_str, gene_chunk=gene_index_str
                    )
                    file_path = os.path.join(output_dir, file_name)

                    # Save methylation chunk
                    mc_logger.count(
                        'Saving methylation chunk {0}/{1}:',
                        meth_chunk_index + 1,
                        meth_chunk_count,
                    )
                    pool.apply_async(
                        save_dataframe_part,
                        (out, file_path, meth_chunk_index + 1),
                        dict(mc_logger),
                    )

                    inner_logger.time(
                        (
                            'Completed methylation chunk {0}/{1} in {l}'
                            ' seconds. Average chunk time: {a} seconds'
                        ),
                        meth_chunk_index + 1,
                        meth_chunk_count,
                    )

                    del results[:]

            logger.time(
                'FINISHED METHYLATION CHUNK {0} IN {l} SECONDS',
                meth_chunk_index + 1,
            )

        # Wait for chunks to save
        if chunking:
            logger.time('Waiting for chunks to save...')
            pool.close()
            pool.join()
            logger.time('Finished waiting for chunks to save in {l} seconds')

        logger.time(
            'Finished calculating the multiple linear regression in {t} total'
            ' seconds'
        )

        # Return output as pandas.DataFrame if neither gene nor
        # methylation chunking are used
        if not chunking:
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
    dtype = DTYPE
    prob_one = lambda t: torch.distributions.StudentT(df).log_prob(t).exp()
    prob_two = create_studentt_p(df, device, dtype)
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
