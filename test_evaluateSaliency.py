import subprocess
import os
import pandas as pd
import numpy as np
import pytest

def setup_module(module):
    # Create a synthetic parquet file
    data = {
        'mt_id': [f'cg{i}' for i in range(1, 201)],
        'gt_id': [f'gene{i}' for i in range(1, 201)],
        'region': ['CIS5'] * 50 + ['DISTAL'] * 50 + ['TRANS'] * 100,
        'mt_ig': np.linspace(100, 1, 200),
        'cov1_ig': np.linspace(5, 50, 200),
        'cov2_ig': np.random.uniform(1, 10, 200),
        'mt_est': np.random.uniform(-1, 1, 200),
        'p_boot': np.linspace(0.0001, 0.05, 200)
    }
    df = pd.DataFrame(data)
    df.to_parquet('test_saliency.parquet')

def teardown_module(module):
    if os.path.exists('test_saliency.parquet'):
        pass

def test_evaluate_saliency_basic():
    cmd = [
        "python", "tools/evaluateSaliency.py",
        "-i", "test_saliency.parquet",
        "-o", "test_out",
        "--rank-windows", "1-50", "150-200"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed with output: {res.stderr}"
    assert "SALIENCY EVALUATION REPORT" in res.stdout
    assert "Rank 1-50:" in res.stdout
    assert "Rank 150-200:" in res.stdout
    assert "DISTAL:" in res.stdout

def test_evaluate_saliency_no_covariates():
    # Test fallback when only mt_ig is present
    df = pd.read_parquet('test_saliency.parquet')
    df = df.drop(columns=['cov1_ig', 'cov2_ig'])
    df.to_parquet('test_saliency_no_cov.parquet')

    cmd = [
        "python", "tools/evaluateSaliency.py",
        "-i", "test_saliency_no_cov.parquet",
        "-o", "test_out_no_cov"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed: {res.stderr}"
    assert "Only scalar 'mt_ig' is available" in res.stderr
    assert "Saliency Magnitude Distribution (|mt_ig|)" in res.stdout

    os.remove('test_saliency_no_cov.parquet')

def test_evaluate_saliency_kneed_chord():
    cmd = [
        "python", "tools/evaluateSaliency.py",
        "-i", "test_saliency.parquet",
        "-o", "test_out_chord",
        "--inflection-method", "chord"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed: {res.stderr}"
    assert "(method: chord)" in res.stdout

def test_evaluate_saliency_kneed():
    cmd = [
        "python", "tools/evaluateSaliency.py",
        "-i", "test_saliency.parquet",
        "-o", "test_out_kneed",
        "--inflection-method", "kneed"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed: {res.stderr}"
    assert "(method: kneed)" in res.stdout

if __name__ == '__main__':
    setup_module(None)
    pytest.main(['-v', 'test_evaluateSaliency.py'])
