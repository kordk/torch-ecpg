import os
import math
import time
from multiprocessing import Pool
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

    batch_size = min(100, len(pairs_df)) # Process pairs in batches

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

        # Solve
        lstsq_res = torch.linalg.lstsq(X_flat, Y_flat)
        beta_flat = lstsq_res.solution # (B * I, 2 + K_c, 1)

        # We want the methylation coefficient, which is at index 1
        mt_est_flat = beta_flat[:, 1, 0] # (B * I,)

        mt_est = mt_est_flat.view(B, I) # (B, I)

        # Calculate statistics
        mt_est_mean = mt_est.mean(dim=1)
        mt_est_std = mt_est.std(dim=1, unbiased=True) # match pandas/numpy default delta DOF

        # Percentiles
        # torch.quantile is good
        ci_low = torch.quantile(mt_est, 0.025, dim=1)
        ci_high = torch.quantile(mt_est, 0.975, dim=1)

        # p_boot calculation: proportion of iterations where effect size equals zero or reverses sign
        # Reverses sign relative to the mean? Or just <= 0 if mean > 0, >= 0 if mean < 0?
        # Standard: empirical p-value for two-tailed test is usually
        # min( P(x <= 0), P(x >= 0) ) * 2
        # Or proportion of iterations where it crosses zero.
        # Requirement: "proportion of iterations where the effect size equals zero or reverses sign"
        # We'll do:
        # if mean > 0: p = mean(est <= 0) * 2? No, just "proportion".
        # Let's use:
        prop_less_eq_zero = (mt_est <= 0).float().mean(dim=1)
        prop_greater_eq_zero = (mt_est >= 0).float().mean(dim=1)

        # A simple two-tailed equivalent is 2 * min(p(<=0), p(>=0))
        # Wait, if "proportion of iterations where the effect size equals zero or reverses sign"
        # implies: if actual effect is positive, it's (est <= 0). If actual effect is negative, it's (est >= 0).
        # We can just use the min.
        p_boot = torch.min(prop_less_eq_zero, prop_greater_eq_zero) * 2.0
        # cap at 1.0
        p_boot = torch.clamp(p_boot, max=1.0)

        mt_est_means.extend(mt_est_mean.cpu().numpy())
        mt_est_stds.extend(mt_est_std.cpu().numpy())
        ci_lows.extend(ci_low.cpu().numpy())
        ci_highs.extend(ci_high.cpu().numpy())
        p_boots.extend(p_boot.cpu().numpy())

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"Processed {i + batch_size} / {len(pairs_df)} pairs...")

    logger.info(f"Finished bootstrap in {time.time() - start_time:.2f} seconds. Valid pairs: {len(valid_pairs)}")

    # Create a DataFrame with the results
    if not valid_pairs:
        logger.warning("No valid pairs found to bootstrap.")
        res_df = pd.DataFrame(columns=["mt_id", "gt_id", "mt_est_boot_mean", "mt_est_boot_std", "ci_low", "ci_high", "p_boot"])
    else:
        res_df = pd.DataFrame(valid_pairs)
    res_df['mt_est_boot_mean'] = mt_est_means
    res_df['mt_est_boot_std'] = mt_est_stds
    res_df['ci_low'] = ci_lows
    res_df['ci_high'] = ci_highs
    res_df['p_boot'] = p_boots

    # We only need to join these new columns, plus mt_id and gt_id
    res_df = res_df[['mt_id', 'gt_id', 'mt_est_boot_mean', 'mt_est_boot_std', 'ci_low', 'ci_high', 'p_boot']]

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
    cols_to_drop = [c for c in ['mt_est_boot_mean', 'mt_est_boot_std', 'ci_low', 'ci_high', 'p_boot'] if c in master_df.columns]
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
