import numpy as np
import pandas as pd
from tecpg.permute import _residualize_and_permute
from tecpg.logger import Logger

def _generate_strong_covariate_fixture(seed, S, n_genes, K):
    """
    Generate G and C such that G = C @ Beta + noise.
    This creates a strong covariate dependence for G, making Yhat vary significantly
    across samples, and making the forced-fail Manly/Skip injections diverge decisively.
    """
    rng = np.random.default_rng(seed)

    # Generate C (e.g. covariates like age/sex)
    C_data = rng.normal(size=(S, K))
    C = pd.DataFrame(C_data, columns=[f"cov_{i}" for i in range(K)])

    # Gene names
    G_idx = [f"gene_{i}" for i in range(n_genes)]

    # Beta weights for covariates -> genes. shape: (K, n_genes)
    # Make them large to have strong signal
    Beta = rng.normal(scale=10.0, size=(K, n_genes))

    # Signal component: C @ Beta -> (S, n_genes)
    Signal = C_data @ Beta

    # Add non-trivial noise (e.g. std=2.0)
    Noise = rng.normal(scale=2.0, size=(S, n_genes))

    G_data = Signal + Noise

    # G shape is (n_genes, S)
    G = pd.DataFrame(G_data.T, index=G_idx, columns=C.index)

    return G, C


def test_fl_oracle():
    """
    test 1: Oracle - FL vs numpy reference.
    """
    S, n_genes, K = 80, 20, 3
    seed = 42
    G, C = _generate_strong_covariate_fixture(seed, S, n_genes, K)

    rng = np.random.default_rng(seed)
    # Fixed non-identity permutation
    perm_vector = rng.permutation(S)

    # Pure numpy reference for steps 1-8
    ones = np.ones((S, 1), dtype=np.float64)
    C_mat = C.to_numpy(dtype=np.float64)
    D = np.hstack([ones, C_mat])
    Y = G.to_numpy(dtype=np.float64).T

    B_red, _, _, _ = np.linalg.lstsq(D, Y, rcond=None)
    Yhat = D @ B_red
    R = Y - Yhat
    R_perm = R[perm_vector, :]
    Y_star = Yhat + R_perm
    reference = Y_star.T

    logger = Logger()
    G_star = _residualize_and_permute(G, C, perm_vector, logger)

    np.testing.assert_allclose(G_star.to_numpy(), reference, rtol=1e-3, atol=1e-4)
    assert G_star.index.equals(G.index)
    assert G_star.columns.equals(G.columns)
    assert G_star.shape == (n_genes, S)


def test_identity_invariant():
    """
    test 2: Identity-permutation invariant.
    """
    S, n_genes, K = 80, 20, 3
    seed = 123
    G, C = _generate_strong_covariate_fixture(seed, S, n_genes, K)

    # Identity permutation
    perm_vector = np.arange(S)

    logger = Logger()
    G_star = _residualize_and_permute(G, C, perm_vector, logger)

    # Should be exactly G, up to float tolerance
    np.testing.assert_allclose(G_star.to_numpy(), G.to_numpy(), rtol=1e-12, atol=1e-12)


def test_determinism():
    """
    test 3: Determinism. Two calls with the same (G, C, perm_vector) return identical output.
    """
    S, n_genes, K = 80, 20, 3
    seed = 999
    G, C = _generate_strong_covariate_fixture(seed, S, n_genes, K)

    rng = np.random.default_rng(seed)
    perm_vector = rng.permutation(S)

    logger = Logger()
    G_star1 = _residualize_and_permute(G, C, perm_vector, logger)
    G_star2 = _residualize_and_permute(G, C, perm_vector, logger)

    np.testing.assert_array_equal(G_star1.to_numpy(), G_star2.to_numpy())
