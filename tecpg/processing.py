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
from .gpu_monitor import gpu_guardian, report_thermal_status, throttle_if_needed
from .helper import logit_transform_torch, trim_dataframes
from .import_data import initialize_dir, save_dataframe_part
from .logger import Logger


def create_normal_p(device: torch.device, dtype: torch.dtype):
    scalar = (
        torch.tensor(2, device=device, dtype=dtype).sqrt().reciprocal().neg()
    )

    def prob(value: torch.Tensor) -> torch.Tensor:
        return torch.erf(scalar * value.abs()) + 1

    return prob


def tecpg_mlr_lstsq(
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
    logit_transform: bool = False,
    thermal_threshold: int = 80,
    thermal_wait: int = 30,
    file_format: str = '{meth_chunk}-{gene_chunk}.csv',
    reservoir_count: Optional[int] = None,
    *,
    logger: Logger = Logger(),
) -> Optional[pandas.DataFrame]:
    '''
    Calculates the multiple linear regression of the input dataframes M,
    G, and C using torch.linalg.lstsq.
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
    if region != 'all' and (G_annot is None or M_annot is None):
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

        G_chrom_t = torch.tensor(G_chrom, device=get_device(**logger), dtype=torch.int8)
        G_pos_t = torch.tensor(G_pos, device=get_device(**logger), dtype=torch.int32)
        G_strand_t = torch.tensor(G_strand, device=get_device(**logger), dtype=torch.int8)

    # Initializes some constants
    logger.info(
        'Running tecpg_mlr_lstsq with options: {0}',
        {
            k: v
            for k, v in locals().items()
            if k not in ['M', 'G', 'C', 'M_annot', 'G_annot', 'logger']
        },
    )
    logger.info('Initializing regression variables (lstsq)')
    device = get_device(**logger)
    dtype = DTYPE
    if meth_loci_per_chunk is not None:
        meth_chunk_count = math.ceil(len(M) / meth_loci_per_chunk)
    else:
        meth_chunk_count = 1

    nrows, ncols = C.shape[0], C.shape[1] + 1
    G_np = G.to_numpy()
    gt_count = len(G)
    gt_site_names = numpy.array(G.index.values)
    df = nrows - ncols - 1
    logger.info('Running with {0} degrees of freedom', df)
    normal_p = create_normal_p(device, dtype)

    if gene_loci_per_chunk is not None:
        gene_chunk_count = math.ceil(len(G) / gene_loci_per_chunk)
    else:
        gene_chunk_count = 1

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
        Ct_base: torch.Tensor = torch.tensor(
            C.to_numpy(), device=device, dtype=dtype
        ).repeat(len(M), 1, 1)
    else:
        Ct_base: torch.Tensor = torch.tensor(
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

    # Variables for Reservoir Sampling
    do_reservoir = reservoir_count is not None and reservoir_count > 0
    if do_reservoir:
        expected_items = (
            int(meth_count * gene_count)
            if region == 'all'
            else "unknown (region filtration applied)"
        )
        logger.info(
            f"Initializing reservoir sampling. Will retain up to {reservoir_count} "
            f"results out of an expected {expected_items} total items."
        )
    reservoir_buffer = [] # Store tuple of (results_tensor, gt_sites, mt_sites)
    reservoir_processed = 0

    # Create methylation chunk (mc_) and chunk saving (inner_) logger
    mc_logger = logger.alias()
    mc_logger.info_color = colors.GREEN
    inner_logger = mc_logger.alias()

    # Use the multiprocessing pool
    with gpu_guardian(logger) as gpu_handle, Pool() as pool:
        # Loop for methylation chunks or ran once with index 0 if no
        # methylation chunking
        for meth_chunk_index in range(meth_chunk_count):
            throttle_if_needed(gpu_handle, thermal_threshold, thermal_wait, logger)
            report_thermal_status(gpu_handle, thermal_threshold, logger)

            logger.memory_check('tecpg_mlr_lstsq')
            # Log methylation chunk index
            logger.info(
                'STARTING METHYLATION CHUNK {0}/{1}',
                meth_chunk_index + 1,
                meth_chunk_count,
            )
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
                    Ct = Ct_base[: len(M_chunk)]
                else:
                    Ct = Ct_base
            else:
                M_chunk = M
                Ct = Ct_base
                start_index = 0
                end_index = len(M)

            mt_count = len(M_chunk)
            mt_site_names = numpy.array(M_chunk.index.values)
            if region == 'all' and p_thresh is None:
                # If no filtration, output size is constant per gene chunk
                pass # Calculated later

            mc_logger.start_timer('info', 'Running tecpg_mlr_lstsq...')

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

            # Calculate design matrix X for the current methylation chunk
            Mt: torch.Tensor = torch.tensor(
                M_chunk.to_numpy(), device=device, dtype=dtype
            ).unsqueeze(2)

            if logit_transform:
                Mt = logit_transform_torch(Mt, logger=mc_logger)

            ones = torch.ones((mt_count, nrows, 1), device=device, dtype=dtype)
            X: torch.Tensor = torch.cat((ones, Mt, Ct), 2) # (M, S, K)
            del Mt, ones

            mc_logger.memory_check('tecpg_mlr_lstsq - peak')

            # Pre-calculate diagonal of (X^T X)^-1 for Standard Error using QR decomposition
            # X = QR => X^T X = R^T R. (X^T X)^-1 = (R^T R)^-1 = R^-1 (R^-1)^T.
            # We need the diagonal elements.
            Q, R = torch.linalg.qr(X, mode='reduced')
            # Calculate R_inv. R is upper triangular.
            # torch.linalg.inv works, or solve_triangular
            # For batch, inv is fine.
            R_inv = torch.linalg.inv(R)
            # XtXi_diag = sum(R_inv^2, dim=2) ? No.
            # (R^-1)(R^-1)^T diagonal is sum of squares of rows of R^-1.
            # R_inv is (M, K, K).
            # We want diag((R_inv) @ (R_inv).mT)
            # Element [i, j, j] = sum_k R_inv[i, j, k] * R_inv[i, j, k]
            XtXi_diag_sqrt = (R_inv.pow(2).sum(dim=2)).sqrt()
            del Q, R, R_inv

            # Display amount of total memory occupied by the constants
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

            # Loop over gene chunks
            inner_logger.start_timer('info', 'Calculating regression (lstsq)...')

            gene_end_index = 0

            for gene_chunk_index in range(gene_chunk_count):
                gene_start_index = gene_end_index
                if gene_loci_per_chunk is not None:
                    gene_end_index = min((gene_chunk_index + 1) * gene_loci_per_chunk, len(G))
                else:
                    gene_end_index = len(G)

                G_chunk_np = G_np[gene_start_index:gene_end_index]
                chunk_len = gene_end_index - gene_start_index

                # Transpose gene expression matrix to serve as target matrix Y
                # G_chunk_np is (G_chunk, S). Transpose to (S, G_chunk).
                Y = torch.tensor(G_chunk_np.T, device=device, dtype=dtype) # (S, G_chunk)

                # Solve using lstsq(X, Y)
                # X is (M_chunk, S, K). Y is (S, G_chunk).
                # We need to solve for B of shape (M_chunk, K, G_chunk).
                # Broadcast Y to (1, S, G_chunk)?
                # torch.linalg.lstsq(A, B):
                # A: (*, m, n). B: (*, m, k).
                # If we want output (*, n, k).
                # Here A=X is (M_chunk, S, K).
                # We want B to match M_chunk dim.
                # So expand Y to (M_chunk, S, G_chunk).
                Y_expanded = Y.unsqueeze(0).expand(mt_count, -1, -1)
                inner_logger.memory_check('tecpg_mlr_lstsq - target expanded')

                # Coefficients B
                lstsq_result = torch.linalg.lstsq(X, Y_expanded)
                inner_logger.memory_check('tecpg_mlr_lstsq - lstsq result')
                B = lstsq_result.solution # (M_chunk, K, G_chunk)

                # Calculate Residuals E = Y - X B
                # X: (M, S, K). B: (M, K, G).
                # X @ B -> (M, S, G).
                E = Y_expanded - X.matmul(B)
                inner_logger.memory_check('tecpg_mlr_lstsq - residuals')

                # RSS = sum(E^2, dim=1) -> (M, G)
                RSS = E.pow(2).sum(dim=1)

                # Sigma = sqrt(RSS / df)
                # Standard Errors S = XtXi_diag_sqrt * Sigma
                # XtXi_diag_sqrt is (M, K). RSS is (M, G).
                # We need S of shape (M, K, G).
                # Expand XtXi_diag_sqrt to (M, K, 1).
                # Expand RSS to (M, 1, G).

                Sigma = (RSS / df).sqrt().unsqueeze(1) # (M, 1, G)
                S = XtXi_diag_sqrt.unsqueeze(2) * Sigma # (M, K, G)

                del E, RSS, Sigma, Y_expanded, Y

                # Calculate T and P
                # B is (M, K, G). S is (M, K, G).

                # Reshape to (M * G, K) for easier filtering if we were to filter now?
                # regression_full output format requires (M, K) per regression.
                # Here we have K coeffs per regression.
                # We need to align with regression_full output.
                # regression_full output columns: [mt_est, mt_err, mt_t, mt_p, ...]
                # It flattens the result.
                # Let's verify output format.
                # regression_full produces results list of tensors of shape (N, 4*K) or something?
                # No, results.append(torch.cat((B, S, T, P), dim=1)).
                # B, S, T, P in regression_full are (N_subset, K).
                # So concatenated is (N_subset, 4*K).
                # Here we have B (M, K, G).
                # We need to permute to (M, G, K).
                B = B.permute(0, 2, 1) # (M, G, K)
                S = S.permute(0, 2, 1)
                T = B / S
                P = normal_p(T)
                inner_logger.memory_check('tecpg_mlr_lstsq - pvals')

                # Now we have tensors of shape (M, G, K).
                # We need to flatten to (M*G, K) to apply filters efficiently?
                # Or apply masks on the (M, G) grid.

                if region != 'all':
                    # Create mask
                    # G_chrom_t etc are full length. Slice for current chunk.
                    G_chrom_chunk = G_chrom_t[gene_start_index:gene_end_index]
                    G_pos_chunk = G_pos_t[gene_start_index:gene_end_index]
                    G_strand_chunk = G_strand_t[gene_start_index:gene_end_index]

                    # Compute mask (M, G)
                    if region in ('cis', 'distal'):
                        # M_chrom_t: (M,)
                        # G_chrom_chunk: (G,)
                        # Broadcast: (M, 1) vs (1, G) -> (M, G)
                        # Note: regression_full loop was over genes, so it did (M, 1) vs scalar.
                        # Here:
                        region_mask = (
                            (M_chrom_t.unsqueeze(1) == G_chrom_chunk.unsqueeze(0))
                            .logical_and(
                                G_strand_chunk.unsqueeze(0) * (window_base - upstream) < (G_pos_chunk.unsqueeze(0) - M_pos_t.unsqueeze(1))
                            )
                            .logical_and(
                                (G_pos_chunk.unsqueeze(0) - M_pos_t.unsqueeze(1)) < (G_strand_chunk.unsqueeze(0) * (window_base + downstream))
                            )
                        )
                    elif region == 'trans':
                        region_mask = (M_chrom_t.unsqueeze(1) != G_chrom_chunk.unsqueeze(0))

                # Flatten the tensors to (M*G, K)
                # We want the order to match regression_full:
                # regression_full loops M (outer), then G (inner).
                # results.append() inside G loop.
                # So effective order is M1_G1, M1_G2... NO.
                # regression_full:
                # Loop Meth Chunks (M_chunk)
                #   Loop Gene (g)
                #     results.append(mask(M_chunk) against g) -> This gives M_sub * 1.
                #     So for one gene g, we have multiple meth sites.
                #     Order in results list is: [M_sub_g1, M_sub_g2, ...]
                #     Where M_sub_g1 are meth sites for g1.
                #     So the primary index variation is Meth, then Gene?
                #     No, "results.append(torch.cat((B, S, T, P), dim=1))" happens inside G loop.
                #     So for G1, we add a block of M results.
                #     So the dataframe index is (Gene, Meth).
                #     `index_names = ['gt_id', 'mt_id']`.
                #     Yes, index is (Gene, Meth).

                # So we need to flatten (M, G, K) to (M*G, K) such that G varies slowest?
                # No, if index is (G, M), we want G to be the outer block.
                # B is (M, G, K).
                # Permute to (G, M, K).
                # Reshape to (G*M, K).
                num_coeffs = B.shape[-1]
                B = B.permute(1, 0, 2).reshape(-1, num_coeffs)
                S = S.permute(1, 0, 2).reshape(-1, num_coeffs)
                T = T.permute(1, 0, 2).reshape(-1, num_coeffs)
                P = P.permute(1, 0, 2).reshape(-1, num_coeffs)

                if region != 'all':
                    # mask is (M, G).
                    # Permute to (G, M).
                    # Reshape to (G*M).
                    region_mask = region_mask.permute(1, 0).reshape(-1)

                    B = B[region_mask]
                    S = S[region_mask]
                    T = T[region_mask]
                    P = P[region_mask]

                    region_indices_list.append(region_mask) # Need to save for index generation

                if methylation_only:
                    # B is (N, K). Columns: Const, Meth, Covars...
                    # Meth is index 1.
                    B = B[:, 1:2]
                    S = S[:, 1:2]
                    T = T[:, 1:2]
                    P = P[:, 1:2]

                if p_only:
                    current_results = P
                else:
                    current_results = torch.cat((B, S, T, P), dim=1)

                if do_reservoir:
                    # Collect batch results for reservoir sampling BEFORE p-value filtration
                    batch_size = len(current_results)
                    if batch_size > 0:
                        # Construct indices for reservoir
                        gt_chunk_names_res = gt_site_names[gene_start_index:gene_end_index]
                        gt_sites_res = numpy.repeat(gt_chunk_names_res, mt_count)
                        mt_sites_res = numpy.tile(mt_site_names, chunk_len)

                        if region != 'all':
                            # region_mask was applied, we need the numpy mask
                            mask_np_res = region_indices_list[-1].cpu().numpy()
                            gt_sites_res = gt_sites_res[mask_np_res]
                            mt_sites_res = mt_sites_res[mask_np_res]

                        # Generate random rolls for reservoir
                        rolls = torch.rand(batch_size, device=device)
                        # Probability P for each item j in batch is reservoir_count / (reservoir_processed + j + 1)
                        # To do this correctly:
                        # P_j = reservoir_count / (reservoir_processed + j + 1)
                        # If roll < P_j, we keep it. Where does it go? Random index in [0, reservoir_count - 1]

                        if reservoir_processed + batch_size <= reservoir_count:
                            # If buffer is not full, just append everything
                            reservoir_buffer.append((current_results.clone(), gt_sites_res, mt_sites_res))
                            reservoir_processed += batch_size
                        else:
                            # Vectorized Reservoir Sampling
                            # Generate indices j for the batch (1-based relative to processed)
                            j_indices = torch.arange(1, batch_size + 1, device=device)
                            total_processed_j = reservoir_processed + j_indices

                            # Items with index <= reservoir_count ALWAYS get kept (if any)
                            # However, we handle the 'buffer not full initially' gracefully

                            keep_probs = reservoir_count / total_processed_j
                            keep_mask = rolls < keep_probs

                            # For the items before reservoir_count, we MUST keep them
                            if reservoir_processed < reservoir_count:
                                initial_keep_count = reservoir_count - reservoir_processed
                                keep_mask[:initial_keep_count] = True

                            kept_indices = torch.nonzero(keep_mask).squeeze(1)

                            if len(kept_indices) > 0:
                                kept_results = current_results[kept_indices]
                                kept_gt = gt_sites_res[kept_indices.cpu().numpy()]
                                kept_mt = mt_sites_res[kept_indices.cpu().numpy()]

                                # Where to place them in the buffer?
                                # For each kept item, if its total index <= reservoir_count, we append it
                                # If its total index > reservoir_count, it replaces a random element [0, reservoir_count-1]
                                replace_indices = torch.randint(0, reservoir_count, (len(kept_indices),), device=device)

                                # Process the kept items
                                for idx, kept_idx in enumerate(kept_indices):
                                    total_idx = reservoir_processed + kept_idx.item() + 1
                                    if total_idx <= reservoir_count:
                                        # Buffer is not full, we just append
                                        reservoir_buffer.append((kept_results[idx:idx+1].clone(), kept_gt[idx:idx+1], kept_mt[idx:idx+1]))
                                    else:
                                        # Buffer is full, replace an existing element
                                        target_idx = replace_indices[idx].item()

                                        # Since reservoir_buffer is a list of tensors of varying sizes,
                                        # replacing by index requires finding which tensor/element it belongs to.
                                        # This can be slow. To optimize, if the buffer gets complex, we flatten it.
                                        # Instead of full flattening, we'll store individual replacements in a list
                                        # and flatten later, or we can flatten the reservoir buffer once it reaches capacity.

                                        # Flatten buffer if not already flattened
                                        if isinstance(reservoir_buffer, list):
                                            if sum(len(x[1]) for x in reservoir_buffer) > 0:
                                                res_res_cat = torch.cat([x[0] for x in reservoir_buffer])
                                                res_gt_cat = numpy.concatenate([x[1] for x in reservoir_buffer])
                                                res_mt_cat = numpy.concatenate([x[2] for x in reservoir_buffer])
                                                reservoir_buffer = (res_res_cat, res_gt_cat, res_mt_cat)
                                            else:
                                                # Handle empty buffer weirdness
                                                pass

                                        if isinstance(reservoir_buffer, tuple):
                                            res_res_cat, res_gt_cat, res_mt_cat = reservoir_buffer
                                            res_res_cat[target_idx] = kept_results[idx]
                                            res_gt_cat[target_idx] = kept_gt[idx]
                                            res_mt_cat[target_idx] = kept_mt[idx]

                            reservoir_processed += batch_size

                # P-value filtration
                if p_thresh is not None:
                    # P shape (N, K_out). Meth p-value is at index 0 (if meth_only) or 1 (if not).
                    # Wait, if meth_only, P is (N, 1).
                    # If not meth_only, P is (N, K). Const, Meth, ...
                    # regression_full: p_indices = P[:, 0 if methylation_only else 1] <= p_thresh
                    p_col = 0 if methylation_only else 1
                    p_indices = P[:, p_col] <= p_thresh
                    p_indices_list.append(p_indices)

                    P = P[p_indices]
                    if not p_only:
                        B = B[p_indices]
                        S = S[p_indices]
                        T = T[p_indices]

                if filtration:
                    output_sizes.append(len(P))

                if p_only:
                    results.append(P)
                else:
                    results.append(current_results[p_indices] if p_thresh is not None else current_results)

                # Save output if gene chunking is used
                # In regression_full, saving happens inside the loop if gene_loci_per_chunk is set.
                if gene_loci_per_chunk:
                    # Construct indices
                    # We need indices for the current gene chunk.
                    # G_chunk names:
                    gt_chunk_names = gt_site_names[gene_start_index:gene_end_index]

                    # If region == 'all' and no p_thresh
                    # We have (G_chunk * M_chunk) results.
                    # M sites are same for all G in this chunk.

                    # Generate full indices for the block. At this point, any filtration
                    # has already been applied to the data; indexing is identical for
                    # filtered and non-filtered cases.
                    # Generate full indices for the block
                    # gt_chunk_names (G_chunk)
                    # mt_site_names (M_chunk)

                    # Repeat G for each M
                    gt_sites = numpy.repeat(gt_chunk_names, mt_count) # [g1, g1, ..., g2, g2, ...]
                    mt_sites = numpy.tile(mt_site_names, chunk_len) # [m1, m2, ..., m1, m2, ...]

                    # Apply region mask
                    if region != 'all':
                        # region_mask was (G, M) then flattened.
                        # It corresponds to the order of gt_sites/mt_sites above?
                        # B = B.permute(1, 0, 2).reshape(-1, ncols) -> (G, M, K) -> (G*M, K).
                        # Yes.
                        # region_mask is on CPU/GPU?
                        # region_mask was tensor.
                        # Need numpy mask.
                        mask_np = region_indices_list[-1].cpu().numpy()
                        del region_indices_list[:]

                        gt_sites = gt_sites[mask_np]
                        mt_sites = mt_sites[mask_np]
                    else:
                        mask_np = None # implied all True

                    # Apply p-value mask
                    if p_thresh is not None:
                        p_mask_np = p_indices_list[-1].cpu().numpy()
                        del p_indices_list[:]

                        gt_sites = gt_sites[p_mask_np]
                        mt_sites = mt_sites[p_mask_np]

                    index_chunk = [gt_sites, mt_sites]

                    # Create path and save
                    gene_index_str = str(gene_chunk_index + 1) # This differs slightly from regression_full count which counts genes?
                    # regression_full: gene_index_str = str(mc_logger.current_count + 1)
                    # mc_logger.current_count increments when saving.
                    # Here we can just use loop index + 1.
                    meth_index_str = str(meth_chunk_index + 1)
                    file_name = file_format.format(
                        meth_chunk=meth_index_str, gene_chunk=gene_index_str
                    )
                    file_path = os.path.join(output_dir, file_name)

                    out = pandas.DataFrame(
                        torch.cat(results).cpu().numpy(),
                        index=index_chunk,
                        columns=columns,
                    )
                    out.index.set_names(index_names, inplace=True)

                    # Save
                    mc_logger.count(
                        'Saving part {i}/{0}',
                        gene_chunk_count,
                    )
                    pool.apply_async(
                        save_dataframe_part,
                        (out, file_path, gene_chunk_index + 1),
                        dict(mc_logger),
                    )
                    del results[:]

                    # Force GC
                    if allocated_memory:
                         torch.cuda.empty_cache()

            mc_logger.time('Looped over methylation loci in {l} seconds')
            mc_logger.time('Calculated tecpg_mlr_lstsq in {t} seconds')

            # If no gene chunking (gene_loci_per_chunk is None), save/return results
            if gene_loci_per_chunk is None:
                # We have one result in `results`.
                # Generate indices.
                gt_sites = numpy.repeat(gt_site_names, mt_count)
                mt_sites = numpy.tile(mt_site_names, gt_count)

                if region != 'all':
                    mask_np = region_indices_list[-1].cpu().numpy()
                    del region_indices_list[:]
                    gt_sites = gt_sites[mask_np]
                    mt_sites = mt_sites[mask_np]

                if p_thresh is not None:
                    p_mask_np = p_indices_list[-1].cpu().numpy()
                    del p_indices_list[:]
                    gt_sites = gt_sites[p_mask_np]
                    mt_sites = mt_sites[p_mask_np]

                index_chunk = [gt_sites, mt_sites]

                out = pandas.DataFrame(
                    torch.cat(results).cpu().numpy(),
                    index=index_chunk,
                    columns=columns,
                )
                out.index.set_names(index_names, inplace=True)

                if meth_loci_per_chunk is not None:
                    gene_index_str = '1'
                    meth_index_str = str(meth_chunk_index + 1)
                    file_name = file_format.format(
                        meth_chunk=meth_index_str, gene_chunk=gene_index_str
                    )
                    file_path = os.path.join(output_dir, file_name)

                    mc_logger.count(
                        'Saving methylation chunk {0}/{1}',
                        meth_chunk_index + 1,
                        meth_chunk_count,
                    )
                    pool.apply_async(
                        save_dataframe_part,
                        (out, file_path, meth_chunk_index + 1),
                        dict(mc_logger),
                    )
                    del results[:]

            logger.time(
                'FINISHED METHYLATION CHUNK {0} IN {l} SECONDS',
                meth_chunk_index + 1,
            )

        # Wait for chunks
        if chunking:
            logger.time('Waiting for chunks to save...')
            pool.close()
            pool.join()
            logger.time('Finished waiting for chunks to save in {l} seconds')

        if do_reservoir and reservoir_processed > 0:
            logger.info('Saving reservoir sample ({0} rows)', min(reservoir_processed, reservoir_count))
            if isinstance(reservoir_buffer, list):
                res_res_cat = torch.cat([x[0] for x in reservoir_buffer])
                res_gt_cat = numpy.concatenate([x[1] for x in reservoir_buffer])
                res_mt_cat = numpy.concatenate([x[2] for x in reservoir_buffer])
            else:
                res_res_cat, res_gt_cat, res_mt_cat = reservoir_buffer

            index_chunk = [res_gt_cat, res_mt_cat]
            out_reservoir = pandas.DataFrame(
                res_res_cat.cpu().numpy(),
                index=index_chunk,
                columns=columns,
            )
            out_reservoir.index.set_names(index_names, inplace=True)

            # Additional logging for reservoir processing status
            logger.info(f"Reservoir sampling completed. Processed {reservoir_processed} total items. "
                        f"Target reservoir count was {reservoir_count}.")

            # Determine memory usage of the resulting DataFrame
            mem_usage_bytes = out_reservoir.memory_usage(deep=True).sum()
            mem_usage_mb = mem_usage_bytes / (1024 ** 2)

            logger.info(f"Reservoir DataFrame shape: {out_reservoir.shape} (rows, columns). "
                        f"Memory footprint: {mem_usage_mb:.2f} MB.")

            # Use output_dir from logger carry_data if not explicitly passed
            carry_output = logger.carry_data.get('output_dir', None)
            res_output_dir = output_dir if output_dir is not None else (carry_output if carry_output else os.getcwd())

            # Ensure output directory exists before writing
            if not os.path.exists(res_output_dir):
                try:
                    os.makedirs(res_output_dir, exist_ok=True)
                except OSError:
                    pass

            res_file_path = os.path.join(res_output_dir, 'sample_reservoir.csv')
            out_reservoir.to_csv(res_file_path)
            logger.time('Finished saving reservoir sample to {0}', res_file_path)

        logger.time(
            'Finished calculating the multiple linear regression (lstsq) in {t} total'
            ' seconds'
        )

        if not chunking:
            return out
