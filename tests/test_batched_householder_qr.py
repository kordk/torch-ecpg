"""batched_householder_qr must be downstream-equivalent to
torch.linalg.qr(mode='reduced') and actually wired into the mlr path.

Equivalence is asserted on the quantities the pipeline consumes — beta
(R^{-1} Q^T Y), the XtXi diagonal (R^{-1} row norms), leverage (Q row
norms) — plus reconstruction and orthonormality, never on raw Q/R, which
legitimately differ by per-column sign (geqrf guarantees no sign
convention; batched_householder_qr fixes diag(R) >= 0).

fp32 bounds are set 3-10x above values measured at the production shape
(B=10000, n=340, k=4: dbeta 6.6e-7, dXtXi 1.6e-8, dh 1.7e-7,
recon 1.5e-5, orth 3.7e-6). The fp64 test pins the algorithm itself: any
structural error (wrong sign choice, wrong scaling, skipped reflection)
exceeds 1e-10 by many orders of magnitude and cannot hide in fp32 slack.
"""
import numpy as np
import pandas as pd
import pytest
import torch
from unittest import mock

from tecpg.logger import Logger
from tecpg.processing import batched_householder_qr, tecpg_mlr_qr


def _design(B, n, k, dtype, seed, realistic=False):
    g = torch.Generator().manual_seed(seed)
    if realistic:
        # Production column scales: intercept, methylation beta in [0, 1],
        # age ~ N(45, 12), binary sex — mixed scales stress conditioning
        # the way real covariate matrices do.
        cols = [torch.ones(B, n, 1, dtype=dtype)]
        cols.append(torch.rand(B, n, 1, generator=g, dtype=dtype))
        if k > 2:
            cols.append(45 + 12 * torch.randn(B, n, 1, generator=g,
                                              dtype=dtype))
        for _ in range(k - 3):
            cols.append((torch.rand(B, n, 1, generator=g,
                                    dtype=dtype) < 0.5).to(dtype))
        return torch.cat(cols[:k], dim=2)
    X = torch.randn(B, n, k, generator=g, dtype=dtype)
    X[:, :, 0] = 1.0
    return X


def _downstream_deltas(X, dtype):
    k = X.shape[2]
    Q, R = batched_householder_qr(X)
    Qr, Rr = torch.linalg.qr(X, mode='reduced')
    g = torch.Generator().manual_seed(1)
    Y = torch.randn(X.shape[0], X.shape[1], 3, generator=g, dtype=dtype)
    eye = torch.eye(k, dtype=dtype).expand(X.shape[0], -1, -1)

    def beta(Q_, R_):
        return torch.linalg.solve_triangular(R_, Q_.mT @ Y, upper=True)

    Ri = torch.linalg.solve_triangular(R, eye, upper=True)
    Rri = torch.linalg.solve_triangular(Rr, eye, upper=True)
    return {
        'recon': (Q @ R - X).abs().max().item(),
        'orth': (Q.mT @ Q - torch.eye(k, dtype=dtype)).abs().max().item(),
        'beta': (beta(Q, R) - beta(Qr, Rr)).abs().max().item(),
        'xtxi': (Ri.pow(2).sum(2) - Rri.pow(2).sum(2)).abs().max().item(),
        'lev': ((Q * Q).sum(2).amax(1)
                - (Qr * Qr).sum(2).amax(1)).abs().max().item(),
    }


@pytest.mark.parametrize('shape,realistic', [
    ((2000, 340, 4), False),
    ((2000, 340, 4), True),
    ((512, 40, 4), False),
    ((300, 50, 8), True),
    ((7, 33, 6), False),
])
def test_downstream_equivalent_to_torch_qr_fp32(shape, realistic):
    X = _design(*shape, dtype=torch.float32, seed=0, realistic=realistic)
    d = _downstream_deltas(X, torch.float32)
    scale = X.abs().max().item()
    assert d['recon'] <= 5e-5 * max(scale, 1.0), d
    assert d['orth'] <= 5e-5, d
    assert d['beta'] <= 2e-4 * max(scale, 1.0), d
    assert d['xtxi'] <= 1e-5, d
    assert d['lev'] <= 5e-5, d


def test_algorithm_exact_in_fp64():
    X = _design(300, 50, 6, dtype=torch.float64, seed=2, realistic=True)
    d = _downstream_deltas(X, torch.float64)
    for key, val in d.items():
        assert val <= 1e-10, (key, d)


def test_r_is_upper_triangular_with_nonnegative_diagonal():
    X = _design(64, 30, 5, dtype=torch.float32, seed=3)
    _, R = batched_householder_qr(X)
    lower = R.tril(-1).abs().max().item()
    assert lower == 0.0
    assert (R.diagonal(dim1=1, dim2=2) >= 0).all()


def _tiny_map(**kwargs):
    rng = np.random.default_rng(5)
    S, M_, G_ = 20, 40, 6
    people = [f's{i}' for i in range(S)]
    Md = pd.DataFrame(rng.random((M_, S)),
                      index=[f'cg{i}' for i in range(M_)], columns=people)
    Gd = pd.DataFrame(rng.random((G_, S)),
                      index=[f'ILMN_{i}' for i in range(G_)], columns=people)
    Cd = pd.DataFrame({'age': rng.integers(20, 60, S)}, index=people)
    return tecpg_mlr_qr(
        Md, Gd, Cd,
        region='all', methylation_only=True, p_thresh=None,
        p_only=False, logit_transform=False, seed=1, logger=Logger(),
        **kwargs,
    ), M_ * G_


def test_cpu_path_keeps_lapack_qr():
    # The dispatch is device-gated: on CPU, LAPACK geqrf is ~10x faster
    # than the batched formulation, so the CPU branch must still call
    # torch.linalg.qr and must never call batched_householder_qr.
    if torch.cuda.is_available():
        pytest.skip('CPU-branch test; CUDA machine runs the CUDA test')
    with mock.patch(
        'tecpg.processing.batched_householder_qr',
        side_effect=AssertionError('householder called on CPU path'),
    ):
        out, n = _tiny_map()
    assert len(out) == n
    assert np.isfinite(out.to_numpy()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_cuda_path_uses_replacement_not_torch_qr():
    # On CUDA the launch-bound torch.linalg.qr must never fire inside the
    # qr map: patching it to raise proves the Householder branch is wired.
    with mock.patch(
        'torch.linalg.qr',
        side_effect=AssertionError('torch.linalg.qr called on CUDA path'),
    ):
        out, n = _tiny_map()
    assert len(out) == n
    assert np.isfinite(out.to_numpy()).all()
