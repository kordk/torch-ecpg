def _ig_overhead(
    samples: int,
    mt_count: int,
    covar_count: int,
    datum_bytes: int,
    compute_ig: bool,
    compute_ig_deep: bool,
) -> 'tuple[int, float]':
    """Conservative IG memory overhead model used by both estimators.

    Returns ``(extra_constants_bytes, locus_factor)``:

    * ``extra_constants_bytes`` budgets the IG-only transients that are
      sized in ``mt_count`` rather than per gene-chunk-locus. Specifically:

      - Analytical IG (``compute_ig=True``) realizes
        ``X_diff_mean = (X - X_baseline).abs().mean(dim=1)`` whose
        intermediate before the reduction has the same ``(M, S, K)``
        shape as ``X``, plus ``X_baseline`` of shape ``(M, 1, K)`` that
        stays alive through the gene loop. We charge one full ``X``
        equivalent to be conservative.
      - Deep IG (``compute_ig_deep=True``) additionally retains ``X``
        alive through the gene loop and accumulates per-step
        interpolated activations during the autograd backward pass.
        We charge another two ``X`` equivalents on top of the
        analytical IG term as a conservative upper bound.

    * ``locus_factor`` (>= 1.0) scales the per-locus chunk term to
      account for the larger transient working set when IG is enabled.

    These factors are conservative on purpose: under-estimating peak
    causes OOM, over-estimating only forces extra outer iterations.
    """
    if not (compute_ig or compute_ig_deep):
        return 0, 1.0

    K = covar_count + 2  # ones, Mt, then C covariates
    x_bytes = mt_count * samples * K * datum_bytes
    extra_constants = x_bytes  # one extra (M, S, K) live tensor for IG
    locus_factor = 1.5

    if compute_ig_deep:
        extra_constants += 2 * x_bytes
        locus_factor = 4.0

    return extra_constants, locus_factor


def estimate_loci_per_chunk_e_peak(
    target_bytes: int,
    samples: int,
    mt_count: int,
    gt_count: int,
    covar_count: int = 2,
    datum_bytes: int = 4,
    filtration: float = 1,
    full_output: bool = False,
    p_only: bool = True,
    p_filtration: bool = False,
    region_filtration: bool = False,
    compute_ig: bool = False,
    compute_ig_deep: bool = False,
) -> float:
    constants_bytes = estimate_constants_bytes(
        samples,
        mt_count,
        gt_count,
        covar_count,
        datum_bytes,
        region_filtration,
    )

    ig_extra_constants, ig_locus_factor = _ig_overhead(
        samples, mt_count, covar_count, datum_bytes,
        compute_ig, compute_ig_deep,
    )
    constants_bytes += ig_extra_constants

    chunk_constants = (
        filtration * mt_count * samples * datum_bytes
        + 2 * filtration * mt_count * datum_bytes
    )
    if region_filtration:
        chunk_constants += mt_count

    locus_bytes = filtration * mt_count * datum_bytes
    if not p_only:
        locus_bytes *= 4
    if full_output:
        locus_bytes *= 2 + covar_count
    if region_filtration:
        locus_bytes += mt_count
    if p_filtration:
        locus_bytes += filtration * mt_count
    locus_bytes *= ig_locus_factor

    e_loci_per_chunk = (
        target_bytes - constants_bytes - chunk_constants
    ) / locus_bytes + 1

    return e_loci_per_chunk


def estimate_loci_per_chunk_results_peak(
    target_bytes: int,
    samples: int,
    mt_count: int,
    gt_count: int,
    covar_count: int = 2,
    datum_bytes: int = 4,
    filtration: float = 1,
    full_output: bool = False,
    p_only: bool = True,
    region_filtration: bool = False,
    compute_ig: bool = False,
    compute_ig_deep: bool = False,
) -> float:
    constants_bytes = estimate_constants_bytes(
        samples,
        mt_count,
        gt_count,
        covar_count,
        datum_bytes,
        region_filtration,
    )

    ig_extra_constants, ig_locus_factor = _ig_overhead(
        samples, mt_count, covar_count, datum_bytes,
        compute_ig, compute_ig_deep,
    )
    constants_bytes += ig_extra_constants

    locus_bytes = 2 * filtration * mt_count * datum_bytes
    if not p_only:
        locus_bytes *= 4
    if full_output:
        locus_bytes *= 2 + covar_count
    locus_bytes *= ig_locus_factor

    results_loci_per_chunk = (target_bytes - constants_bytes) / locus_bytes

    return results_loci_per_chunk


def estimate_constants_bytes(
    samples: int,
    mt_count: int,
    gt_count: int,
    covar_count: int = 2,
    datum_bytes: int = 4,
    region_filtration: bool = False,
) -> int:
    constants_bytes = (
        datum_bytes
        + (2 + covar_count) * mt_count * datum_bytes
        + 2 * mt_count * samples * datum_bytes * (covar_count + 2)
    )
    if region_filtration:
        # 4 bytes for int32 pos and 1 byte for int8 chrom
        constants_bytes += 5 * mt_count * gt_count

    return constants_bytes
