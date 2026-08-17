import os
import shutil
import time
import torch
import numpy as np
import pandas as pd
import pytest

from tecpg.processing import tecpg_mlr_qr
from tecpg.logger import Logger

_RNG = np.random.default_rng(0)
_S, _M, _G, _K = 30, 40, 20, 4   # samples, meth loci, genes, covariate columns + 2

_M_DATA = pd.DataFrame(
    _RNG.random((_M, _S)),
    index=[f'cg{i:06d}' for i in range(_M)],
    columns=[f's{i}' for i in range(_S)],
)
_G_DATA = pd.DataFrame(
    _RNG.random((_G, _S)),
    index=[f'ENSG{i:06d}' for i in range(_G)],
    columns=[f's{i}' for i in range(_S)],
)
_C_DATA = pd.DataFrame(
    _RNG.random((_S, _K - 2)),
    index=[f's{i}' for i in range(_S)],
    columns=[str(i) for i in range(_K - 2)],
)

def _run(tmp_path, seed, tag):
    out = tmp_path / tag
    out.mkdir()
    tecpg_mlr_qr(
        _M_DATA, _G_DATA, _C_DATA,
        region='all',
        methylation_only=True,
        p_thresh=None,
        p_only=False,
        logit_transform=False,
        output_dir=str(out),
        gene_loci_per_chunk=5,
        meth_loci_per_chunk=40,
        reservoir_count=100,
        seed=seed,
        logger=Logger(),
    )
    res_file_path = os.path.join(str(out), "sample_reservoir.csv")
    with open(res_file_path, "r") as f:
        return f.read()

def test_fixture_is_deterministic(tmp_path):
    run1 = _run(tmp_path, 42, 'run1')
    run2 = _run(tmp_path, 42, 'run2')
    assert run1 == run2

def test_reservoir_identical_across_runs_same_seed(tmp_path):
    run1 = _run(tmp_path, 42, 'run1_ident')
    run2 = _run(tmp_path, 42, 'run2_ident')
    assert run1 == run2

def test_reservoir_differs_across_seeds(tmp_path):
    run1 = _run(tmp_path, 42, 'run1_diff')
    run2 = _run(tmp_path, 1337, 'run2_diff')
    assert run1 != run2

def test_reservoir_ignores_global_torch_rng(tmp_path):
    torch.manual_seed(0)
    run1 = _run(tmp_path, 42, 'run1_ignore')
    torch.manual_seed(999)
    run2 = _run(tmp_path, 42, 'run2_ignore')
    assert run1 == run2

def test_reservoir_does_not_disturb_global_torch_rng(tmp_path):
    torch.manual_seed(7)
    _ = torch.rand(5)
    baseline_second = torch.rand(4)

    torch.manual_seed(7)
    _ = torch.rand(5)
    _run(tmp_path, 42, 'interleaved')
    after_run_second = torch.rand(4)

    assert torch.equal(baseline_second, after_run_second)

def test_reservoir_row_count(tmp_path):
    run_output = _run(tmp_path, 42, 'run_row_count')
    # Count rows excluding header
    res_file_path = os.path.join(str(tmp_path / 'run_row_count'), "sample_reservoir.csv")
    df = pd.read_csv(res_file_path)
    assert len(df) == 100

def test_reservoir_replacement_path_exercised(tmp_path):
    out = tmp_path / 'run_path_exercised'
    out.mkdir()

    # Capture stdout instead
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    logger_obj = Logger()
    tecpg_mlr_qr(
        _M_DATA, _G_DATA, _C_DATA,
        region='all',
        methylation_only=True,
        p_thresh=None,
        p_only=False,
        logit_transform=False,
        output_dir=str(out),
        gene_loci_per_chunk=5,
        meth_loci_per_chunk=40,
        reservoir_count=100,
        seed=42,
        logger=logger_obj,
    )

    found = False
    sys.stdout.seek(0)
    output = sys.stdout.read()
    sys.stdout = old_stdout
    for msg in output.split("\n"):
        if "Reservoir sampling completed. Processed 800 total items." in msg:
            found = True
            break
    assert found
