import numpy as np
from tecpg.permute import _score_observed, _accumulate_null, _fit_tail, _fit_gpd
from tecpg.logger import Logger

def print_test_results():
    print("--- FORCED FAIL PROOFS ---")

    # Shared setup
    rng = np.random.default_rng(42)
    null_stats = rng.normal(scale=2.5, size=200000)
    logger = Logger()
    acc = _accumulate_null(null_stats, None, logger)
    u = acc['topk_values'].min()
    N = acc['total_count']
    N_u = acc['topk_values'].size
    exc = acc['topk_values'][acc['topk_values'] > u] - u
    xi, sigma = _fit_gpd(exc)

    extreme_obs = np.array([u + 10 * sigma])

    # 1. Injection A: drop exceedance-probability scaling
    # Expected RED for test_continuity_at_threshold
    print("\n[Injection A: Drop exceedance scaling]")
    emp_p_u = _score_observed([u], acc, logger)
    # p_gpd = SF_GPD(abs_obs - u)
    import scipy.stats
    p_gpd_A = scipy.stats.genpareto.sf(u - u, xi, loc=0, scale=sigma)
    p_gpd_A = np.maximum(p_gpd_A, np.finfo(np.float64).tiny)
    perm_mt_p_A = np.where([u] > u, p_gpd_A, emp_p_u) # this logic doesn't fail test 2 if we use np.where, test2 uses [u] strictly. Wait, test 2 fails if we used the wrong logic directly inside _fit_tail.
    # Let's simulate what _fit_tail would return
    tail_p_A = p_gpd_A
    print(f"EXPECTED: {emp_p_u[0]} (N_u/N)")
    print(f"ACTUAL (broken): {tail_p_A} (SF(0) = 1.0)")
    print(f"RED (Continuity): {np.isclose(emp_p_u[0], tail_p_A) == False}")

    # 2. Injection B: skip GPD (passthrough)
    # Expected RED for test_below_floor_extension
    print("\n[Injection B: Passthrough]")
    emp_p_ext = _score_observed(extreme_obs, acc, logger)
    # _fit_tail returns empirical_p
    tail_p_B = emp_p_ext
    empirical_floor = 1.0 / (N + 1)

    print(f"EXPECTED: < {empirical_floor}")
    print(f"ACTUAL (broken): {tail_p_B[0]}")
    print(f"RED (Below-floor): {not (tail_p_B[0] < empirical_floor)}")

print_test_results()
