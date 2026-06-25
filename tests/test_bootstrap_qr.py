import torch
import pytest

def test_bootstrap_well_conditioned():
    torch.manual_seed(42)
    device = 'cpu'

    B, I, N, K_c = 2, 10, 50, 1

    # Well conditioned M, G, C
    ones_boot = torch.ones((B, I, N, 1), device=device)
    M_boot = torch.randn((B, I, N, 1), device=device)
    C_boot = torch.randn((B, I, N, K_c), device=device)

    X_boot = torch.cat((ones_boot, M_boot, C_boot), dim=-1)
    Y_boot = torch.randn((B, I, N, 1), device=device)

    X_flat = X_boot.view(B * I, N, 2 + K_c)
    Y_flat = Y_boot.view(B * I, N, 1)

    # 1. lstsq
    lstsq_res = torch.linalg.lstsq(X_flat, Y_flat)
    beta_flat_lstsq = lstsq_res.solution
    mt_est_flat_lstsq = beta_flat_lstsq[:, 1, 0]

    # 2. qr + solve_triangular
    Q, R = torch.linalg.qr(X_flat, mode='reduced')
    QtY = Q.mT @ Y_flat
    beta_flat_qr = torch.linalg.solve_triangular(R, QtY, upper=True)
    mt_est_flat_qr = beta_flat_qr[:, 1, 0]

    assert torch.allclose(mt_est_flat_lstsq, mt_est_flat_qr, atol=1e-4, rtol=1e-3)


def test_bootstrap_degenerate():
    torch.manual_seed(42)
    device = 'cpu'

    B, I, N, K_c = 1, 4, 5, 1

    # Rank deficient: M_boot is exactly zero for 1 iteration, normal for the rest
    ones_boot = torch.ones((B, I, N, 1), device=device)
    M_boot = torch.randn((B, I, N, 1), device=device)
    M_boot[0, 0, :, 0] = 0.0 # Make the first iteration degenerate

    C_boot = torch.randn((B, I, N, K_c), device=device)

    X_boot = torch.cat((ones_boot, M_boot, C_boot), dim=-1)
    Y_boot = torch.randn((B, I, N, 1), device=device)

    X_flat = X_boot.view(B * I, N, 2 + K_c)
    Y_flat = Y_boot.view(B * I, N, 1)

    # QR
    Q, R = torch.linalg.qr(X_flat, mode='reduced')
    QtY = Q.mT @ Y_flat
    beta_flat_qr = torch.linalg.solve_triangular(R, QtY, upper=True)
    mt_est_flat_qr = beta_flat_qr[:, 1, 0]

    mt_est = mt_est_flat_qr.view(B, I)

    # Mock production guard logic
    valid = torch.isfinite(mt_est)

    for row in range(B):
        valid_mask = valid[row]
        filtered_est = mt_est[row][valid_mask]

        degen_count = I - valid_mask.sum().item()
        assert degen_count == 1 # 1 degenerate iteration

        # Test stats finite behavior
        mt_est_mean = filtered_est.mean().item()
        mt_est_std = filtered_est.std(unbiased=True).item()
        ci_low = torch.quantile(filtered_est, 0.025).item()
        ci_high = torch.quantile(filtered_est, 0.975).item()

        import math
        assert math.isfinite(mt_est_mean)
        assert math.isfinite(mt_est_std)
        assert math.isfinite(ci_low)
        assert math.isfinite(ci_high)

    # Test ALL degenerate case
    M_boot_all_degen = torch.zeros((B, I, N, 1), device=device)
    X_boot_all = torch.cat((ones_boot, M_boot_all_degen, C_boot), dim=-1)
    X_flat_all = X_boot_all.view(B * I, N, 2 + K_c)

    Q_all, R_all = torch.linalg.qr(X_flat_all, mode='reduced')
    QtY_all = Q_all.mT @ Y_flat
    beta_flat_qr_all = torch.linalg.solve_triangular(R_all, QtY_all, upper=True)
    mt_est_all = beta_flat_qr_all[:, 1, 0].view(B, I)

    valid_all = torch.isfinite(mt_est_all)
    for row in range(B):
        valid_mask_all = valid_all[row]
        filtered_est_all = mt_est_all[row][valid_mask_all]

        degen_count_all = I - valid_mask_all.sum().item()
        assert degen_count_all == I
        assert filtered_est_all.numel() == 0

import os
import pandas as pd
from unittest.mock import patch
from tecpg.bootstrap import tecpg_mlr_qr_bootstrap

def test_bootstrap_end_to_end(tmp_path):
    torch.manual_seed(42)
    device = 'cpu'

    # Create small synthetic M, G, C dataframes
    n_samples = 20
    M = pd.DataFrame(torch.randn((3, n_samples)).numpy(), index=['cg001', 'cg002', 'cg003'])

    # Set up cg002 to be degenerate ONLY when sample 0 is NOT drawn.
    M.loc['cg002'] = 0.0
    M.loc['cg002', 0] = 1.0

    G = pd.DataFrame(torch.randn((3, n_samples)).numpy(), index=['ENSG001', 'ENSG002', 'ENSG003'])
    C = pd.DataFrame(torch.randn((n_samples, 2)).numpy(), index=[f'sample_{i}' for i in range(n_samples)])
    M.columns = C.index
    G.columns = C.index

    # Create pairs file
    pairs_file = tmp_path / "pairs.csv"
    pairs_df = pd.DataFrame({'mt_id': ['cg001', 'cg002', 'cg003'], 'gt_id': ['ENSG001', 'ENSG002', 'ENSG003']})
    pairs_df.to_csv(pairs_file, index=False)

    # Create master parquet
    master_parquet = tmp_path / "master.parquet"
    master_df = pd.DataFrame({
        'mt_id': ['cg001', 'cg002', 'cg003'],
        'gt_id': ['ENSG001', 'ENSG002', 'ENSG003'],
        'other_col': [1, 2, 3]
    })
    master_df.to_parquet(master_parquet)

    output_file = tmp_path / "output.parquet"

    import numpy as np

    iterations = 5
    fixed_indices = np.random.choice(n_samples, size=(iterations, n_samples), replace=True)

    # Force iteration 0 to NOT sample 0, making cg002 degenerate (0 variance) for iteration 0.
    # We must sample at least 4 distinct indices to ensure X (with 4 columns) maintains full rank
    # for cg001 and cg003, preventing global rank deficiency.
    fixed_indices[0, :5] = 1
    fixed_indices[0, 5:10] = 2
    fixed_indices[0, 10:15] = 3
    fixed_indices[0, 15:] = 4

    # Ensure iterations 1-4 sample 0, so cg002 is normal in other iterations.
    fixed_indices[1:, 0] = 0

    # Run end-to-end (patch get_device to return cpu, and np.random.choice to return our fixed indices)
    with patch('tecpg.bootstrap.get_device', return_value=torch.device('cpu')), \
         patch('tecpg.bootstrap.np.random.choice', return_value=fixed_indices):
        tecpg_mlr_qr_bootstrap(
            M=M,
            G=G,
            C=C,
            pairs_file=str(pairs_file),
            master_parquet=str(master_parquet),
            output_file=str(output_file),
            iterations=iterations,
            batch_size=3,
            compute_ig=True,
            ig_covariates_filter='all'
        )

    assert os.path.exists(output_file)
    result_df = pd.read_parquet(output_file)

    # Check outputs and rows
    assert len(result_df) == 3
    assert 'degenerate_resamples' in result_df.columns
    assert 'mt_est_boot_mean' in result_df.columns
    assert 'mt_est_boot_std' in result_df.columns
    assert 'ci_low' in result_df.columns
    assert 'ci_high' in result_df.columns
    assert 'p_boot' in result_df.columns

    # Check IG columns
    assert 'mt_ig' in result_df.columns
    assert '0_ig' in result_df.columns
    assert '1_ig' in result_df.columns

    import math

    # Row 0 (cg001): Normal case - 0 degenerate iterations
    row0 = result_df[result_df['mt_id'] == 'cg001'].iloc[0]
    assert row0['degenerate_resamples'] == 0
    assert math.isfinite(row0['mt_est_boot_mean'])
    assert math.isfinite(row0['mt_est_boot_std'])
    assert math.isfinite(row0['ci_low'])
    assert math.isfinite(row0['ci_high'])
    assert 0.0 <= row0['p_boot'] <= 1.0

    # Row 1 (cg002): Mixed degenerate case - 1 degenerate iteration, 4 valid
    row1 = result_df[result_df['mt_id'] == 'cg002'].iloc[0]
    assert row1['degenerate_resamples'] == 1
    assert math.isfinite(row1['mt_est_boot_mean'])
    assert math.isfinite(row1['mt_est_boot_std'])
    assert math.isfinite(row1['ci_low'])
    assert math.isfinite(row1['ci_high'])
    assert 0.0 <= row1['p_boot'] <= 1.0

    # Row 2 (cg003): Normal case - 0 degenerate iterations
    row2 = result_df[result_df['mt_id'] == 'cg003'].iloc[0]
    assert row2['degenerate_resamples'] == 0
    assert math.isfinite(row2['mt_est_boot_mean'])
    assert math.isfinite(row2['mt_est_boot_std'])


# ---------------------------------------------------------------------------
# CLI-level regression test for IG kwarg forwarding.
#
# The bug: the `qr_bootstrap` branch of `tecpg run mlr` (tecpg/cli.py) called
# `tecpg_mlr_qr_bootstrap(...)` with an explicit argument list that omitted
# `compute_ig`, `compute_ig_deep`, `ig_baseline`, and `ig_covariates_filter`.
# Because those parameters default to "off" in the bootstrap function, the
# per-covariate (and scalar `mt_ig`) Integrated Gradients columns silently
# vanished from the merged bootstrap parquet even when `--compute-ig
# --ig-covariates` were passed on the command line.
#
# `test_bootstrap_end_to_end` above calls `tecpg_mlr_qr_bootstrap` directly,
# so it cannot catch a CLI-layer forwarding regression. This test drives the
# actual `tecpg run mlr ... --mlr-method qr_bootstrap` entry point and asserts
# on the OUTPUT SCHEMA, since this is a silent failure (no error, just missing
# columns).
# ---------------------------------------------------------------------------
def test_bootstrap_cli_forwards_ig_kwargs(tmp_path):
    import numpy as np
    from click.testing import CliRunner
    from tecpg.cli import cli

    torch.manual_seed(0)
    np.random.seed(0)

    n_samples = 20
    samples = [f'sample_{i}' for i in range(n_samples)]
    mt_ids = ['cg001', 'cg002', 'cg003']
    gt_ids = ['ENSG001', 'ENSG002', 'ENSG003']

    input_dir = tmp_path / 'data'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()
    output_dir.mkdir()

    # M / G: loci x samples. C: samples x covariates (named so per-covariate
    # IG columns are recognizable: age_ig, sex_ig).
    M = pd.DataFrame(np.random.randn(3, n_samples), index=mt_ids, columns=samples)
    G = pd.DataFrame(np.random.randn(3, n_samples), index=gt_ids, columns=samples)
    C = pd.DataFrame(
        np.random.randn(n_samples, 2), index=samples, columns=['age', 'sex']
    )

    M.to_csv(input_dir / 'M.csv')
    G.to_csv(input_dir / 'G.csv')
    C.to_csv(input_dir / 'C.csv')

    pairs_file = tmp_path / 'pairs.csv'
    pd.DataFrame({'mt_id': mt_ids, 'gt_id': gt_ids}).to_csv(pairs_file, index=False)

    master_parquet = tmp_path / 'master.parquet'
    pd.DataFrame(
        {'mt_id': mt_ids, 'gt_id': gt_ids, 'other_col': [1, 2, 3]}
    ).to_parquet(master_parquet)

    output_file = 'bootstrap_merged.parquet'

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            '--root-path', str(tmp_path),
            '--input-dir', 'data',
            '--output-dir', 'output',
            '--output-file', output_file,
            '--cpu-threads', '1',
            '--host-profile', 'minimum',
            '--no-log-file',
            'run', 'mlr',
            '--mlr-method', 'qr_bootstrap',
            '--pairs-file', str(pairs_file),
            '--master-parquet', str(master_parquet),
            '--bootstrap-iterations', '5',
            '--bootstrap-batch-size', '3',
            '--compute-ig',
            '--ig-covariates',
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output

    out_path = output_dir / output_file
    assert out_path.exists(), f'Expected bootstrap output at {out_path}'

    result_df = pd.read_parquet(out_path)

    # Core bootstrap columns must be present.
    for col in [
        'mt_est_boot_mean', 'mt_est_boot_std',
        'ci_low', 'ci_high', 'p_boot', 'degenerate_resamples',
    ]:
        assert col in result_df.columns, f'missing core column {col}'

    # Regression assertion: the scalar mt_ig column AND the per-covariate IG
    # columns must survive the CLI -> bootstrap call. Before the fix these
    # were silently dropped because the CLI did not forward the IG kwargs.
    assert 'mt_ig' in result_df.columns, (
        'scalar mt_ig column missing -- CLI did not forward IG kwargs to '
        'tecpg_mlr_qr_bootstrap'
    )
    for covar in C.columns:
        col = f'{covar}_ig'
        assert col in result_df.columns, (
            f'per-covariate IG column {col!r} missing -- CLI did not forward '
            'ig_covariates_filter to tecpg_mlr_qr_bootstrap'
        )

    # The merged-in IG values should be populated for the matched pairs.
    matched = result_df[result_df['mt_id'].isin(mt_ids)]
    assert matched['mt_ig'].notna().all()
    for covar in C.columns:
        assert matched[f'{covar}_ig'].notna().all()
