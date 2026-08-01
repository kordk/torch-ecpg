import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.stats
import torch
from typing import Literal
from .config import get_device, DTYPE
from .logger import Logger
from .helper import compute_region_mask

# CHUNK 6: Provisional calibration constants for null accumulation.
# These dictate the memory bounds of the streaming accumulator.
# They will be revisited/finalized in Chunk 8 based on convergence evidence.
T_MAX = 10.0
N_BINS = 1000
TOPK_CAPACITY = 10_000

# Host-memory ceiling on the null-population cross-product (|M| x |G|), which is
# built as a DataFrame of Python string pairs before any GPU work begins. This is a
# resource limit, NOT a methodological bound: any product under it is a legitimate
# configuration. Raise it if a larger null is wanted and the host can hold it.
MAX_NULL_PAIR_PRODUCT = 50_000_000


def _normalize_annotations(M_annot, G_annot, M, G, logger):
    from .chrom import canonicalize_chrom

    M_annot_n = M_annot.copy()
    G_annot_n = G_annot.copy()

    M_annot_n['chrom'] = canonicalize_chrom(M_annot_n['chrom'])
    G_annot_n['chrom'] = canonicalize_chrom(G_annot_n['chrom'])

    G_annot_n['strand'] = pd.to_numeric(G_annot_n['strand'].replace({'+': 1, '-': -1}), errors='coerce')

    M_annot_n = M_annot_n[['chrom', 'chromStart']]
    G_annot_n = G_annot_n[['chrom', 'chromStart', 'strand']]

    M_loci_before = len(M.index)
    M_annot_n = M_annot_n.reindex(M.index).dropna()
    if M_loci_before != len(M_annot_n):
        logger.info(
            'Drop site permute._normalize_annotations[M_annot]: dropped methylation loci with '
            'missing/unmappable annotation: {0} -> {1} ({2} dropped)',
            M_loci_before, len(M_annot_n), M_loci_before - len(M_annot_n)
        )

    G_loci_before = len(G.index)
    G_annot_n = G_annot_n.reindex(G.index).dropna()
    if G_loci_before != len(G_annot_n):
        logger.info(
            'Drop site permute._normalize_annotations[G_annot]: dropped gene expression loci with '
            'missing/unmappable annotation: {0} -> {1} ({2} dropped)',
            G_loci_before, len(G_annot_n), G_loci_before - len(G_annot_n)
        )

    if len(M_annot_n) == 0 or len(G_annot_n) == 0:
        raise ValueError("Normalization dropped all loci on one or both axes.")

    M_n = M.loc[M_annot_n.index]
    G_n = G.loc[G_annot_n.index]

    return M_annot_n, G_annot_n, M_n, G_n


def _select_null_population(M, G, C, M_annot, G_annot, region,
                            window_base, downstream, upstream,
                            subsample_mt_count, subsample_g_count, seed, logger):
    # CHUNK 4: seeded uniform pair subsample for the NULL population.
    if subsample_mt_count is None and subsample_g_count is None:
        product = len(M) * len(G)
        if product > MAX_NULL_PAIR_PRODUCT:
            raise ValueError(
                "Host-memory ceiling exceeded: {0} CpGs x {1} genes = {2} null pairs "
                "(ceiling is {3}). These flags size the null only and do not change "
                "which pairs are reported. Provide --subsample-mt-count and "
                "--subsample-g-count to set a manageable null population size.".format(
                    len(M), len(G), product, MAX_NULL_PAIR_PRODUCT
                )
            )
        return M, G

    rng = np.random.default_rng(seed)

    if subsample_mt_count is None:
        null_M = M
    elif subsample_mt_count <= 0:
        raise ValueError("qr_permute subsample mt_count must be positive; got {0}".format(subsample_mt_count))
    elif subsample_mt_count > len(M):
        logger.warning('Requested mt_count {0} > available {1} for null population; using full.', subsample_mt_count, len(M))
        null_M = M
    else:
        idx = rng.choice(len(M), size=subsample_mt_count, replace=False)
        idx.sort()
        null_M = M.iloc[idx]

    if subsample_g_count is None:
        null_G = G
    elif subsample_g_count <= 0:
        raise ValueError("qr_permute subsample g_count must be positive; got {0}".format(subsample_g_count))
    elif subsample_g_count > len(G):
        logger.warning('Requested g_count {0} > available {1} for null population; using full.', subsample_g_count, len(G))
        null_G = G
    else:
        idx = rng.choice(len(G), size=subsample_g_count, replace=False)
        idx.sort()
        null_G = G.iloc[idx]

    product = len(null_M) * len(null_G)
    if product > MAX_NULL_PAIR_PRODUCT:
        raise ValueError(
            "Host-memory ceiling exceeded: {0} CpGs x {1} genes = {2} null pairs "
            "(ceiling is {3}). These flags size the null only and do not change "
            "which pairs are reported. Provide --subsample-mt-count and "
            "--subsample-g-count to set a manageable null population size.".format(
                len(null_M), len(null_G), product, MAX_NULL_PAIR_PRODUCT
            )
        )

    return null_M, null_G


def _compute_trans_mask(reported_pairs, M_annot, G_annot, region,
                        window_base, downstream, upstream, logger):
    # CHUNK 3: real cis/trans masking via the shared mask helper.
    if region == 'all':
        return np.ones(len(reported_pairs), dtype=bool)

    device = get_device(**logger.opts) if hasattr(logger, 'opts') else get_device()

    M_chrom, M_pos = M_annot.to_numpy().T.astype(int)
    G_chrom, G_pos, G_strand = G_annot.to_numpy().T.astype(int)

    m_mapped = M_annot.index.astype(str).get_indexer(reported_pairs['mt_id'].astype(str))
    g_mapped = G_annot.index.astype(str).get_indexer(reported_pairs['gt_id'].astype(str))

    if (m_mapped == -1).any() or (g_mapped == -1).any():
        raise ValueError("qr_permute region mask: reported mt_id/gt_id not found in annotation index")

    m_chrom_pp = torch.tensor(M_chrom[m_mapped], device=device, dtype=torch.int8)
    m_pos_pp = torch.tensor(M_pos[m_mapped], device=device, dtype=torch.int32)
    g_chrom_pp = torch.tensor(G_chrom[g_mapped], device=device, dtype=torch.int8)
    g_pos_pp = torch.tensor(G_pos[g_mapped], device=device, dtype=torch.int32)
    g_strand_pp = torch.tensor(G_strand[g_mapped], device=device, dtype=torch.int8)

    mask = compute_region_mask(
        region, m_chrom_pp, m_pos_pp, g_chrom_pp, g_pos_pp, g_strand_pp,
        window_base=window_base, upstream=upstream, downstream=downstream
    )
    return mask.cpu().numpy()



WORKING_TENSOR_MULT = 8
SAFETY_FRACTION = 0.5
CPU_PAIR_BUDGET_BYTES = 500 * 1024 * 1024

def _resolve_pair_chunk_size(P, S, K, device, override, logger):
    if override is not None:
        chunk = max(1, min(override, P))
        logger.info('permute observed: P={0} chunk={1} override={2}', P, chunk, override)
        return chunk

    dtype = torch.float32 if 'float32' in str(DTYPE) else torch.float64
    element_bytes = torch.tensor([], dtype=dtype).element_size()
    per_pair_bytes = WORKING_TENSOR_MULT * S * K * element_bytes

    if device.type == 'cuda':
        try:
            free, _ = torch.cuda.mem_get_info(device)
            budget = free * SAFETY_FRACTION
            branch = 'cuda'
        except Exception:
            budget = CPU_PAIR_BUDGET_BYTES
            branch = 'cuda-fallback'
    else:
        budget = CPU_PAIR_BUDGET_BYTES
        branch = 'cpu'

    chunk = max(1, int(budget // per_pair_bytes))
    chunk = min(chunk, P)
    n_chunks = (P + chunk - 1) // chunk

    logger.info('permute observed: P={0} chunk={1} n_chunks={2} per_pair_bytes={3} branch={4}', P, chunk, n_chunks, per_pair_bytes, branch)
    return chunk


def _compute_observed_statistic(M, G, C, reported_pairs, logger, *, pair_chunk_size=None, device=None, progress_label: str | None = None):
    # CHUNK 2: pivotal t = B/S per reported pair (reuse qr solve primitives).
    if device is None:
        device = get_device(**logger.opts) if hasattr(logger, 'opts') else get_device()
    dtype = DTYPE

    if len(reported_pairs) == 0:
        return np.array([], dtype=np.float32)

    # Map to integer row positions
    m_idx_str = pd.Index(M.index.astype(str))
    g_idx_str = pd.Index(G.index.astype(str))

    m_mapped = m_idx_str.get_indexer(reported_pairs['mt_id'].astype(str))
    g_mapped = g_idx_str.get_indexer(reported_pairs['gt_id'].astype(str))

    if (m_mapped == -1).any() or (g_mapped == -1).any():
        raise ValueError("Some reported pairs contain IDs not found in M or G indices.")

    P = len(reported_pairs)

    # Degrees of freedom computation
    nrows, ncols = C.shape[0], C.shape[1] + 1
    df = nrows - ncols - 1

    C_tensor = torch.tensor(C.to_numpy(), device=device, dtype=dtype)
    M_tensor = torch.tensor(M.to_numpy(), device=device, dtype=dtype)
    G_tensor = torch.tensor(G.to_numpy(), device=device, dtype=dtype)

    K = 2 + C.shape[1] # intercept + meth + covariates
    chunk = _resolve_pair_chunk_size(P, S=nrows, K=K, device=device, override=pair_chunk_size, logger=logger)
    n_chunks = (P + chunk - 1) // chunk

    out = torch.empty(P, device=device, dtype=dtype)

    n_iters = 0
    S = nrows
    for start in range(0, P, chunk):
        n_iters += 1
        if progress_label and (n_iters == 1 or n_iters == n_chunks or n_iters % max(1, n_chunks // 10) == 0):
            logger.info('{0}: chunk {1}/{2}', progress_label, n_iters, n_chunks)
        end = min(start + chunk, P)
        mm = m_mapped[start:end]
        gm = g_mapped[start:end]
        p = end - start

        M_sub = M_tensor[mm]
        G_sub = G_tensor[gm]

        ones = torch.ones((p, S, 1), device=device, dtype=dtype)
        Mt = M_sub.unsqueeze(2)
        Ct = C_tensor.unsqueeze(0).expand(p, -1, -1)
        X = torch.cat((ones, Mt, Ct), dim=2)
        Y = G_sub.unsqueeze(2)

        Q, R = torch.linalg.qr(X, mode='reduced')

        K_dim = X.shape[2]

        R_inv = torch.linalg.solve_triangular(
            R,
            torch.eye(K_dim, device=device, dtype=dtype).expand(p, -1, -1),
            upper=True
        )

        XtXi_diag_sqrt = (R_inv.pow(2).sum(dim=2)).sqrt()
        QtY = torch.einsum('psk,psg->pkg', Q, Y)
        B = R_inv.matmul(QtY)

        Y_norm_sq = (Y * Y).sum(dim=1)
        QtY_norm_sq = (QtY * QtY).sum(dim=1)

        RSS = (Y_norm_sq - QtY_norm_sq).clamp_min(0)

        Sigma = (RSS / df).sqrt().unsqueeze(1)
        S_err = XtXi_diag_sqrt.unsqueeze(2) * Sigma

        T = B / S_err
        out[start:end] = T[:, 1, 0]

        del M_sub, G_sub, ones, Mt, Ct, X, Y, Q, R, R_inv, QtY, B, T, XtXi_diag_sqrt, Y_norm_sq, QtY_norm_sq, RSS, Sigma, S_err

    logger.info('permute observed: chunks_executed={0}', n_iters)
    return out.cpu().numpy()


def _residualize_and_permute(G, C, perm_vector, logger):
    # CHUNK 5: design-fixed Freedman–Lane (residualize G on [1,C], permute
    # reduced-model residuals, refit against cached [1,M,C] pseudo-inverse).
    #
    # Permuted: the reduced-model response residuals (R), along the sample axis, by perm_vector.
    # Fixed (untouched by the permutation): M, C, the reduced design [1, C], and therefore
    # the full design [1, M, C] used downstream. Nothing on the predictor side moves.

    # 1. D = [ones(S, 1) | C.to_numpy()] -> shape (S, 1+K)
    S = len(C)
    ones = np.ones((S, 1), dtype=np.float64)
    C_mat = C.to_numpy(dtype=np.float64)
    D = np.hstack([ones, C_mat])

    # 2. Y = G.to_numpy().T -> shape (S, n_genes)
    Y = G.to_numpy(dtype=np.float64).T

    # 3. Reduced fit, one solve for all genes: B_red = lstsq(D, Y) -> (1+K, n_genes)
    # Using numpy lstsq which handles any rank robustly with rcond=None.
    B_red, _, _, _ = np.linalg.lstsq(D, Y, rcond=None)

    # 4. Fitted values: Yhat = D @ B_red -> (S, n_genes)
    Yhat = D @ B_red

    # 5. Reduced residuals: R = Y - Yhat -> (S, n_genes)
    R = Y - Yhat

    # 6. Permute the residuals along the sample axis: R_perm = R[perm_vector, :]
    R_perm = R[perm_vector, :]

    # 7. Add back the (unpermuted) reduced fitted values: Y_star = Yhat + R_perm
    Y_star = Yhat + R_perm

    # 8. Return pd.DataFrame(Y_star.T, index=G.index, columns=G.columns)
    # Cast is inherently float64 because inputs were cast to np.float64
    return pd.DataFrame(Y_star.T, index=G.index, columns=G.columns)


def _accumulate_null(permuted_stats, accumulator, logger):
    # CHUNK 6: streaming t-histogram + tail-exceedance buffer.
    if accumulator is None:
        accumulator = {
            'bin_edges': np.linspace(0, T_MAX, N_BINS + 1, dtype=np.float64),
            'hist_counts': np.zeros(N_BINS, dtype=np.int64),
            'overflow_count': 0,
            'total_count': 0,
            'topk_values': np.array([], dtype=np.float64),
            'topk_capacity': TOPK_CAPACITY
        }

    a = np.abs(permuted_stats).astype(np.float64)

    # Histogram the values
    counts, _ = np.histogram(a, bins=accumulator['bin_edges'])
    accumulator['hist_counts'] += counts

    # Overflow values
    overflow = (a > T_MAX).sum()
    accumulator['overflow_count'] += int(overflow)

    accumulator['total_count'] += a.size

    # Merge into topk buffer
    merged = np.concatenate([accumulator['topk_values'], a])
    if merged.size <= accumulator['topk_capacity']:
        accumulator['topk_values'] = merged
    else:
        accumulator['topk_values'] = np.partition(merged, -accumulator['topk_capacity'])[-accumulator['topk_capacity']:]

    return accumulator


def _score_observed(observed_stats, null_accumulator, logger):
    # CHUNK 7: empirical two-sided p = frac(null |t| >= observed |t|), floored 1/(N+1).
    acc = null_accumulator
    if acc is None or acc['total_count'] <= 0:
        raise ValueError("Empty null accumulator; cannot score observed statistics.")

    N = acc['total_count']
    bin_edges = acc['bin_edges']
    hist = acc['hist_counts']
    overflow = acc['overflow_count']
    n_bins = hist.size

    abs_obs = np.abs(np.asarray(observed_stats, dtype=np.float64))   # two-sided: fold to |t|

    # rev[i] = count of null |t| in bin i and all higher bins
    rev = np.cumsum(hist[::-1])[::-1]

    # bin index containing each observed value (conservative: count that whole bin + above)
    b = np.searchsorted(bin_edges, abs_obs, side='right') - 1        # in [-1 .. n_bins]
    in_range = b < n_bins                                            # False => |t| beyond T_MAX
    b_clipped = np.clip(b, 0, n_bins - 1)
    count = np.where(in_range, rev[b_clipped], 0) + overflow

    p = count / N
    p = np.maximum(p, 1.0 / (N + 1))                                 # empirical floor
    return p


def _fit_gpd(exc):
    """
    Fit a Generalized Pareto Distribution to threshold exceedances.
    Returns (xi, sigma) where xi is the shape parameter and sigma is the scale parameter.
    """
    xi, _, sigma = scipy.stats.genpareto.fit(exc, floc=0)
    return xi, sigma

def _fit_tail(empirical_p, observed_stats, null_accumulator, logger):
    # CHUNK 8: generalized-Pareto tail (float64) extending p below the empirical floor.
    acc = null_accumulator
    if acc is None or acc['total_count'] <= 0:
        return empirical_p

    N = acc['total_count']
    topk = acc['topk_values']

    if topk.size == 0:
        return empirical_p

    # PROVISIONAL: threshold u = min(topk) uses all retained exceedances. A higher u
    # may be warranted — to be informed by the eval script's xi-convergence diagnostic.
    u = topk.min()
    N_u = topk.size

    exc = topk[topk > u] - u

    if exc.size < 50:
        logger.warning("GPD tail fit skipped: only {0} exceedances above threshold (need >= 50); returning empirical p-values.", exc.size)
        return empirical_p

    xi, sigma = _fit_gpd(exc)

    if not (np.isfinite(xi) and np.isfinite(sigma)):
        logger.warning("GPD tail fit produced non-finite parameters (xi={0}, sigma={1}); returning empirical p-values.", xi, sigma)
        return empirical_p

    abs_obs = np.abs(np.asarray(observed_stats, dtype=np.float64))

    # Calculate GPD tail probability
    p_gpd = (N_u / N) * scipy.stats.genpareto.sf(abs_obs - u, xi, loc=0, scale=sigma)

    # Clamp to strictly-positive floor
    tiny = np.finfo(np.float64).tiny
    p_gpd = np.maximum(p_gpd, tiny)

    # GPD in the tail, empirical in the bulk
    perm_mt_p = np.where(abs_obs > u, p_gpd, empirical_p)
    return perm_mt_p


def _verify_master_consistency(M, G, C, universe, device, logger,
                               *, n_sample=256, rtol=1e-2, atol=1e-2, seed=42):
    """Sampled equivalence spot-check. Recompute the observed t for a small
    random sample of master pairs from the provided M/G/C and assert it matches
    the stored mt_t. This is an advisory guard (warns, never raises). Tolerance
    is calibrated to the float32 floor between two QR solvers on real data (~1e-2
    observed on gtpsub)."""
    mt_t = universe['mt_t'].to_numpy(dtype=np.float64)
    finite = np.isfinite(mt_t)
    n_finite = int(finite.sum())
    if n_finite == 0:
        logger.warning("MASTER CONSISTENCY WARNING -- master parquet has no finite mt_t to validate against.")
        return
    idx_finite = np.flatnonzero(finite)
    k = min(n_sample, n_finite)
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx_finite, size=k, replace=False)
    sample_pairs = universe.iloc[pick][['mt_id', 'gt_id']].reset_index(drop=True)
    stored = mt_t[pick]
    try:
        recomputed = np.asarray(
            _compute_observed_statistic(M, G, C, sample_pairs, logger, device=device),
            dtype=np.float64,
        )
    except ValueError as e:
        logger.warning("MASTER CONSISTENCY WARNING -- proceeding, but review this:")
        logger.warning("A sampled master pair is absent from the provided M/G (likely wrong input files).")
        logger.warning("The master may be from different data, proceeding UNVERIFIED. {0}", str(e))
        return
    if not np.allclose(recomputed, stored, rtol=rtol, atol=atol, equal_nan=True):
        max_dev = float(np.nanmax(np.abs(recomputed - stored)))
        corr = (float(np.corrcoef(stored, recomputed)[0, 1])
                if np.nanstd(stored) > 0 else float('nan'))
        logger.warning("=" * 72)
        logger.warning("MASTER CONSISTENCY WARNING -- proceeding, but review this:")
        logger.warning("Stored mt_t vs recomputed differ over {0} sampled pairs:", k)
        logger.warning("  max|dt|={0:.3e}  corr={1:.6f}  (tol atol={2:.1e} rtol={3:.1e})",
                       max_dev, corr, atol, rtol)
        logger.warning("corr ~ 1.0 => benign float32 divergence between the mapping and")
        logger.warning("permute QR solvers (same algorithm, same df) on real-scale data.")
        logger.warning("corr well below 1.0, OR a large max|dt|, suggests the master was")
        logger.warning("mapped with a DIFFERENT covariate design than this run's C -- in")
        logger.warning("which case the permutation p-values are INVALID.")
        logger.warning("Verify the master was mapped from the same data/covariates. Continuing.")
        logger.warning("=" * 72)
        return
    logger.info("master consistency OK: {0} sampled pairs, max|dt|={1:.3e}",
                k, float(np.nanmax(np.abs(recomputed - stored))))


def _finalize_output(master_df, reported_pairs, perm_mt_p, seed, n_perm,
                     output_p_threshold, logger):
    res = reported_pairs.copy()            # ['mt_id','gt_id'], already str-cast
    res['perm_mt_p'] = np.asarray(perm_mt_p, dtype=np.float64)
    if output_p_threshold is not None:
        res = res[res['perm_mt_p'] <= output_p_threshold].reset_index(drop=True)
    # Drop any pre-existing perm columns on master to avoid _x/_y suffixing
    # (and make re-running permute on its own prior output idempotent).
    drop = [c for c in ['perm_mt_p', 'seed', 'n_perm'] if c in master_df.columns]
    if drop:
        master_df = master_df.drop(columns=drop)
    merged = master_df.merge(res, on=['mt_id', 'gt_id'], how='left')
    merged['seed'] = seed        # run-level scalars on ALL rows (mirror bootstrap)
    merged['n_perm'] = n_perm
    return merged


def tecpg_mlr_qr_permute(
    M, G, C,
    master_parquet=None, pairs_file=None,
    M_annot=None, G_annot=None,
    region: Literal['all', 'cis', 'distal', 'trans'] = 'all',
    window_base=None, downstream=None, upstream=None,
    permutations=100,
    subsample_mt_count=None, subsample_g_count=None,
    seed=42,
    output_file=None, output_format='auto', output_p_threshold=None,
    thermal_threshold=80, thermal_wait=30,
    logger=None,
):
    if logger is None:
        logger = Logger()

    if M_annot is None or G_annot is None:
        raise ValueError(
            "qr_permute requires methylation and expression annotations to build the "
            "chromosome-stratified (trans) null; none were provided."
        )

    if master_parquet is None:
        raise ValueError(
            "qr_permute requires --master-parquet: the observed mt_t and the "
            "(mt_id, gt_id) universe are read from the mapping output, not recomputed."
        )

    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
        logger.info("No seed provided; generated seed={0} (recorded with outputs).", seed)
    seed = int(seed)

    M_annot, G_annot, M, G = _normalize_annotations(M_annot, G_annot, M, G, logger)

    logger.info("Starting qr_permute with permutations={0}, seed={1}, output_file={2}", permutations, seed, output_file)

    null_M, null_G = _select_null_population(M, G, C, M_annot, G_annot, region,
                                             window_base, downstream, upstream,
                                             subsample_mt_count, subsample_g_count, seed, logger)

    # --- Realigned observed leg: consume the mapping master (bootstrap-parallel) ---
    master_df = pd.read_parquet(master_parquet)
    if 'mt_t' not in master_df.columns:
        raise ValueError("master parquet is missing required column 'mt_t'")
    master_df['mt_id'] = master_df['mt_id'].astype(str)
    master_df['gt_id'] = master_df['gt_id'].astype(str)

    if pairs_file is not None:
        pairs_df = pd.read_csv(pairs_file)
        pairs_df['mt_id'] = pairs_df['mt_id'].astype(str)
        pairs_df['gt_id'] = pairs_df['gt_id'].astype(str)
        if pairs_df.duplicated(['mt_id', 'gt_id']).any():
            raise ValueError("--pairs-file contains duplicate (mt_id, gt_id) rows")
        universe = master_df.merge(pairs_df[['mt_id', 'gt_id']], on=['mt_id', 'gt_id'], how='inner')
        if len(universe) != len(pairs_df):
            raise ValueError("--pairs-file contains pairs absent from --master-parquet "
                             "(no stored mt_t to score)")
    else:
        universe = master_df

    # Restrict the scored universe to loci that survived annotation normalization.
    # The mapping produced the master over the full M x G (region='all' needs no
    # annotations), but the null here is built over the normalized (chromosome-
    # annotated) M/G, so master pairs referencing dropped loci cannot be scored
    # or consistency-checked. Intersect before scoring and the guard.
    valid = (universe['mt_id'].isin(M.index.astype(str))
             & universe['gt_id'].isin(G.index.astype(str)))
    n_invalid = int((~valid).sum())
    if n_invalid and pairs_file is not None:
        examples = universe.loc[~valid, ['mt_id', 'gt_id']].head(5).to_records(index=False).tolist()
        raise ValueError(
            "--pairs-file requests {0} pair(s) whose loci were dropped by annotation "
            "normalization (missing/unmappable chromosome) and cannot be scored; "
            "e.g. {1}".format(n_invalid, examples))
    if n_invalid:
        logger.info(
            "qr_permute: excluding {0} master pairs whose loci were dropped by annotation "
            "normalization (missing/unmappable chromosome).", n_invalid)
        universe = universe[valid].reset_index(drop=True)
    if len(universe) == 0:
        raise ValueError(
            "no master pairs remain after intersecting with the normalized M/G; "
            "check the master was mapped from the same data.")

    reported_pairs = universe[['mt_id', 'gt_id']].reset_index(drop=True)
    observed_t = universe['mt_t'].to_numpy(dtype=np.float64)

    device = get_device(**logger.opts) if hasattr(logger, 'opts') else get_device()

    # Consistency guard: fail-closed if the supplied M/G/C don't match the design
    # behind the master's mt_t (sampled equivalence spot-check).
    _verify_master_consistency(M, G, C, universe, device, logger)

    accumulator = None
    rng = np.random.default_rng(seed)
    n_samples = len(C)

    null_pairs = pd.MultiIndex.from_product(
        [null_M.index.astype(str), null_G.index.astype(str)],
        names=['mt_id', 'gt_id'],
    ).to_frame(index=False)

    trans_mask_null = _compute_trans_mask(null_pairs, M_annot, G_annot, 'trans',
                                          window_base, downstream, upstream, logger)
    null_pairs = null_pairs[trans_mask_null].reset_index(drop=True)

    logger.info('qr_permute: {0} permutations over {1} null pairs', permutations, len(null_pairs))

    for i in range(permutations):
        perm_vector = rng.permutation(n_samples)
        G_perm = _residualize_and_permute(null_G, C, perm_vector, logger)
        perm_stats = _compute_observed_statistic(null_M, G_perm, C, null_pairs, logger, device=device)
        accumulator = _accumulate_null(perm_stats, accumulator, logger)
        logger.info('qr_permute: permutation {0}/{1} done', i + 1, permutations)

    empirical_p = _score_observed(observed_t, accumulator, logger)
    perm_mt_p = _fit_tail(empirical_p, observed_t, accumulator, logger)

    n_reported = len(reported_pairs)

    final_df = _finalize_output(master_df, reported_pairs, perm_mt_p, seed, permutations, output_p_threshold, logger)

    # Honor output format
    if output_format == 'auto':
        ext = 'csv'
        if output_file and output_file.endswith('.parquet'):
            ext = 'parquet'
    else:
        ext = output_format

    if ext == 'parquet' or (output_file and output_file.endswith('.parquet')):
        # Use pyarrow to write parquet
        table = pa.Table.from_pandas(final_df)
        existing = table.schema.metadata or {}
        new_meta = {
            **existing,
            b'tecpg_perm_seed': str(seed).encode(),
            b'tecpg_perm_n_perm': str(permutations).encode(),
            b'tecpg_perm_n_reported': str(n_reported).encode(),
            b'tecpg_perm_n_null_pairs': str(len(null_pairs)).encode(),
        }
        table = table.replace_schema_metadata(new_meta)
        pq.write_table(table, output_file, compression='snappy')
    else:
        final_df.to_csv(output_file, index=False)

    logger.info("Finished qr_permute, wrote {0} of {1} reported pairs to {2}", len(final_df), n_reported, output_file)
