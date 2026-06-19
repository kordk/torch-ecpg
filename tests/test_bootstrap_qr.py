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

    B, I, N, K_c = 1, 1, 5, 1

    # Rank deficient: M_boot is exactly zero
    ones_boot = torch.ones((B, I, N, 1), device=device)
    M_boot = torch.zeros((B, I, N, 1), device=device)
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

    valid = torch.isfinite(mt_est)

    # Check that it drops the resample
    assert valid.sum().item() == 0
    degen_count = I - valid.sum().item()
    assert degen_count == I
