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
) -> float:
    constants_bytes = estimate_constants_bytes(
        samples,
        mt_count,
        gt_count,
        covar_count,
        datum_bytes,
        region_filtration,
    )

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
) -> float:
    constants_bytes = estimate_constants_bytes(
        samples,
        mt_count,
        gt_count,
        covar_count,
        datum_bytes,
        region_filtration,
    )

    locus_bytes = 2 * filtration * mt_count * datum_bytes
    if not p_only:
        locus_bytes *= 4
    if full_output:
        locus_bytes *= 2 + covar_count

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
