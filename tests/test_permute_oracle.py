import numpy as np
import scipy.stats
from tecpg.test_data import generate_data
from tecpg.permute import _compute_observed_statistic
from tecpg.logger import Logger
import pandas as pd

def test_oracle_qr_regression_vs_plain_ols_permute():
    M, G, C = generate_data(
        80, 20, 15, annotation=False,
        seed=1234,
    )

    n_samples = C.shape[0]
    k = C.shape[1] + 2
    df = n_samples - k

    rng = np.random.default_rng(1234)
    mt_ids = list(M.index)
    gt_ids = list(G.index)
    sampled = [
        (rng.choice(gt_ids), rng.choice(mt_ids)) for _ in range(8)
    ]

    sampled_df = pd.DataFrame(sampled, columns=['gt_id', 'mt_id'])

    logger = Logger()
    observed_t = _compute_observed_statistic(M, G, C, sampled_df, logger)

    for i, (gt_id, mt_id) in enumerate(sampled):
        y = G.loc[gt_id].to_numpy(dtype=float)
        x_m = M.loc[mt_id].to_numpy(dtype=float)
        X = np.column_stack(
            (np.ones(n_samples), x_m, C.to_numpy(dtype=float))
        )
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residual = y - X @ beta
        sigma2 = np.sum(residual ** 2) / df
        var_beta = sigma2 * np.linalg.inv(X.T @ X).diagonal()
        se = np.sqrt(var_beta)
        t = beta / se

        np.testing.assert_allclose(
            observed_t[i], t[1], rtol=1e-3, atol=1e-4,
        )

if __name__ == '__main__':
    test_oracle_qr_regression_vs_plain_ols_permute()
    print("Test passed!")
