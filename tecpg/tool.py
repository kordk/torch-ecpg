from typing import NamedTuple


def estimate_loci_per_chunk(
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
):
    constants_bytes = (
        datum_bytes
        + (2 + covar_count) * mt_count * datum_bytes
        + 2 * mt_count * samples * datum_bytes * (covar_count + 2)
    )
    if region_filtration:
        constants_bytes += 2 * datum_bytes * mt_count * gt_count

    locus_bytes = filtration * mt_count * datum_bytes + 2 * mt_count * (
        p_filtration + region_filtration
    )
    if not p_only:
        locus_bytes *= 4
    if full_output:
        locus_bytes *= 2 + covar_count

    loci_per_chunk = (target_bytes - constants_bytes) / locus_bytes
    return loci_per_chunk
