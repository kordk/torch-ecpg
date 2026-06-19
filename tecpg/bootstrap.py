import os
import math
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from .config import get_device, DTYPE
from .logger import Logger

def tecpg_mlr_lstsq_bootstrap(
    M: pd.DataFrame,
    G: pd.DataFrame,
    C: pd.DataFrame,
    pairs_file: str,
    master_parquet: str,
    output_file: str,
    iterations: int = 1000,
    batch_size: int = 10,
    thermal_threshold: int = 80,
    thermal_wait: int = 30,
    *,
    logger: Logger = Logger(),
) -> None:
    # Set up
    device = get_device(**logger)
    dtype = DTYPE

    logger.info(f"Loading bootstrap pairs from {pairs_file}...")
    try:
        pairs_df = pd.read_csv(pairs_file)
        if 'mt_id' not in pairs_df.columns or 'gt_id' not in pairs_df.columns:
            logger.error("Pairs file must contain 'mt_id' and 'gt_id' columns.")
            return

        pairs_df['gt_id_clean'] = pairs_df['gt_id']

        # We need to test if G.index contains stripped versions or not.
        # If G.index has no versions, but pairs does, we should strip pairs.
        # But we do that dynamically.

    except Exception as e:
        logger.error(f"Failed to read pairs file: {e}")
        return

    logger.info(f"Loaded {len(pairs_df)} pairs for bootstrapping. Generating bootstrap index matrix...")

    # Pre-generate matrix of bootstrap indices
    N_total = len(C)
    # Shape: (iterations, N_total)
    boot_indices = np.random.choice(N_total, size=(iterations, N_total), replace=True)

    # To maximize speed, we want to run the lstsq on chunks of pairs
    # Prepare covariate data
    C_np = C.to_numpy()
    # (iterations, N_total, K)
    C_boot = torch.tensor(C_np[boot_indices], device=device, dtype=dtype)
    ones_boot = torch.ones((iterations, N_total, 1), device=device, dtype=dtype)

    # We will build X_boot: [ones, M, C]
    # To save memory, we can build X_boot per pair or per batch of pairs.

    batch_size = min(batch_size, len(pairs_df)) # Process pairs in batches

    results = []

    # Determine ID formatting:
    # Memory instructions: "Ensure the Ensembl ID stripping logic (.split('.')) is applied during the matrix lookup phase to avoid losing versioned gene hits."
    # We strip the versions from both pairs and G index, or rather we create a mapping from stripped to index
    m_names = {name: i for i, name in enumerate(M.index)}

    # Create G name map, stripping version if needed
    g_names = {}
    for i, name in enumerate(G.index):
        clean_name = str(name).split('.')[0]
        g_names[clean_name] = i

    pairs_df['gt_id_clean'] = pairs_df['gt_id'].astype(str).str.split('.').str[0]

    m_matrix = torch.tensor(M.to_numpy(), device=device, dtype=dtype)
    g_matrix = torch.tensor(G.to_numpy(), device=device, dtype=dtype)

    # Keep track of pairs that couldn't be matched
    valid_pairs = []

    # Pre-allocate for all valid pairs
    mt_est_means = []
    mt_est_stds = []
    ci_lows = []
    ci_highs = []
    p_boots = []
    degenerate_counts = []

    logger.info("Running bootstrap loops...")
    start_time = time.time()

    for i in range(0, len(pairs_df), batch_size):
        batch = pairs_df.iloc[i:i+batch_size]

        batch_m_idx = []
        batch_g_idx = []
        batch_valid = []

        for idx, row in batch.iterrows():
            m_id = row['mt_id']
            g_id = row['gt_id_clean']

            if m_id in m_names and g_id in g_names:
                batch_m_idx.append(m_names[m_id])
                batch_g_idx.append(g_names[g_id])
                batch_valid.append(row)

        if not batch_m_idx:
            continue

        valid_pairs.extend(batch_valid)

        # M_batch: (batch_size, N_total)
        M_batch = m_matrix[batch_m_idx]
        # G_batch: (batch_size, N_total)
        G_batch = g_matrix[batch_g_idx]

        # Expand across iterations
        # M_boot: (batch_size, iterations, N_total)
        # For each pair (i), we want M_batch[i][boot_indices]
        # M_batch is (B, N). boot_indices is (I, N).
        # M_batch[:, boot_indices] -> (B, I, N)
        M_boot = M_batch[:, boot_indices]
        G_boot = G_batch[:, boot_indices]

        # Build X for this batch
        # X: [ones, M, C]
        # C_boot is (I, N, K_c). We need to broadcast it to (B, I, N, K_c).
        # ones_boot is (I, N, 1). -> (B, I, N, 1)
        B = len(batch_m_idx)
        I = iterations
        N = N_total
        K_c = C_np.shape[1]

        # M_boot is (B, I, N). We need it to be (B, I, N, 1).
        M_boot_expanded = M_boot.unsqueeze(-1)

        C_boot_expanded = C_boot.unsqueeze(0).expand(B, -1, -1, -1)
        ones_boot_expanded = ones_boot.unsqueeze(0).expand(B, -1, -1, -1)

        # X_boot: (B, I, N, K) where K = 1 (const) + 1 (meth) + K_c (covars)
        X_boot = torch.cat((ones_boot_expanded, M_boot_expanded, C_boot_expanded), dim=-1)

        # Y_boot: (B, I, N, 1)
        Y_boot = G_boot.unsqueeze(-1)

        # We need to solve for B: X_boot * B = Y_boot
        # X_boot is (B, I, N, K), Y_boot is (B, I, N, 1)
        # Flatten B and I to use lstsq natively
        # lstsq takes (..., N, K) and (..., N, 1)
        X_flat = X_boot.view(B * I, N, 2 + K_c)
        Y_flat = Y_boot.view(B * I, N, 1)

        logger.info(
            f"Batch tensor shapes - M_boot: {M_boot.shape}, G_boot: {G_boot.shape}, "
            f"X_flat: {X_flat.shape}"
        )
        # Calculate size in GB for X_flat
        x_flat_gb = X_flat.nelement() * X_flat.element_size() / (1024 ** 3)
        logger.info(f"Estimated size of X_flat: {x_flat_gb:.2f} GB")

        # Memory snapshot
        if torch.cuda.is_available():
            alloc_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            max_alloc_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logger.info(f"GPU memory before lstsq - Allocated: {alloc_mem:.2f} GB, Max Allocated: {max_alloc_mem:.2f} GB")

        # Solve
        Q, R = torch.linalg.qr(X_flat, mode='reduced')
        QtY = Q.mT @ Y_flat
        beta_flat = torch.linalg.solve_triangular(R, QtY, upper=True) # (B * I, 2 + K_c, 1)

        # We want the methylation coefficient, which is at index 1
        mt_est_flat = beta_flat[:, 1, 0] # (B * I,)

        mt_est = mt_est_flat.view(B, I) # (B, I)

        # Degenerate-resample guard
        # QR emits nan/inf on rank-deficient resamples where lstsq did not;
        # on the production CUDA path lstsq/gels was undefined anyway.
        # We filter to only finite values to ensure degenerate draws are excluded.
        valid = torch.isfinite(mt_est)

        batch_mt_est_mean = []
        batch_mt_est_std = []
        batch_ci_low = []
        batch_ci_high = []
        batch_p_boot = []
        batch_degen = []

        for row in range(B):
            valid_mask = valid[row]
            filtered_est = mt_est[row][valid_mask]

            degen_count = I - valid_mask.sum().item()
            batch_degen.append(degen_count)

            if filtered_est.numel() == 0:
                batch_mt_est_mean.append(float('nan'))
                batch_mt_est_std.append(float('nan'))
                batch_ci_low.append(float('nan'))
                batch_ci_high.append(float('nan'))
                batch_p_boot.append(float('nan'))
            else:
                batch_mt_est_mean.append(filtered_est.mean().item())
                if filtered_est.numel() > 1:
                    batch_mt_est_std.append(filtered_est.std(unbiased=True).item())
                else:
                    batch_mt_est_std.append(float('nan'))

                batch_ci_low.append(torch.quantile(filtered_est, 0.025).item())
                batch_ci_high.append(torch.quantile(filtered_est, 0.975).item())

                prop_less_eq_zero = (filtered_est <= 0).float().mean()
                prop_greater_eq_zero = (filtered_est >= 0).float().mean()
                p_boot = torch.min(prop_less_eq_zero, prop_greater_eq_zero) * 2.0
                p_boot = torch.clamp(p_boot, max=1.0)
                batch_p_boot.append(p_boot.item())

        mt_est_means.extend(batch_mt_est_mean)
        mt_est_stds.extend(batch_mt_est_std)
        ci_lows.extend(batch_ci_low)
        ci_highs.extend(batch_ci_high)
        p_boots.extend(batch_p_boot)
        degenerate_counts.extend(batch_degen)

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"Processed {i + batch_size} / {len(pairs_df)} pairs...")

    total_degen = sum(degenerate_counts)
    total_resamples = len(valid_pairs) * iterations
    logger.info(f"Finished bootstrap in {time.time() - start_time:.2f} seconds. Valid pairs: {len(valid_pairs)}")
    logger.info(f"Total degenerate resamples: {total_degen} / {total_resamples}")

    # Create a DataFrame with the results
    if not valid_pairs:
        logger.warning("No valid pairs found to bootstrap.")
        res_df = pd.DataFrame(columns=["mt_id", "gt_id", "mt_est_boot_mean", "mt_est_boot_std", "ci_low", "ci_high", "p_boot", "degenerate_resamples"])
    else:
        res_df = pd.DataFrame(valid_pairs)
    res_df['mt_est_boot_mean'] = mt_est_means
    res_df['mt_est_boot_std'] = mt_est_stds
    res_df['ci_low'] = ci_lows
    res_df['ci_high'] = ci_highs
    res_df['p_boot'] = p_boots
    res_df['degenerate_resamples'] = degenerate_counts

    # We only need to join these new columns, plus mt_id and gt_id
    res_df = res_df[['mt_id', 'gt_id', 'mt_est_boot_mean', 'mt_est_boot_std', 'ci_low', 'ci_high', 'p_boot', 'degenerate_resamples']]

    logger.info(f"Merging results with master Parquet file: {master_parquet} ...")

    # Read master Parquet file
    try:
        master_df = pd.read_parquet(master_parquet)
    except Exception as e:
        logger.error(f"Failed to read master parquet file: {e}")
        return

    # Perform Left Join on [mt_id, gt_id]
    # Rows not in res_df will automatically get NaN
    # But before join, check if those columns already exist to avoid suffixing
    cols_to_drop = [c for c in ['mt_est_boot_mean', 'mt_est_boot_std', 'ci_low', 'ci_high', 'p_boot', 'degenerate_resamples'] if c in master_df.columns]
    if cols_to_drop:
        master_df = master_df.drop(columns=cols_to_drop)

    # We might need to ensure types match for joining
    master_df['mt_id'] = master_df['mt_id'].astype(str)
    master_df['gt_id'] = master_df['gt_id'].astype(str)
    res_df['mt_id'] = res_df['mt_id'].astype(str)
    res_df['gt_id'] = res_df['gt_id'].astype(str)

    merged_df = master_df.merge(res_df, on=['mt_id', 'gt_id'], how='left')

    logger.info(f"Saving merged results to {output_file} ...")

    # Ensure output dir exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    merged_df.to_parquet(output_file, engine='pyarrow', compression='snappy')

    logger.info("Done.")
