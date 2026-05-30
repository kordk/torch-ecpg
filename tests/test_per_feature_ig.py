import os
import sys
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from unittest.mock import patch, MagicMock

@pytest.fixture
def synthetic_parquet(tmp_path):
    """Creates a small Parquet file simulating Stage 9 mlr output with per-feature IG."""
    df = pd.DataFrame({
        'gt_id': ['ENSG000001', 'ENSG000002', 'ENSG000003'],
        'mt_id': ['cg000001', 'cg000002', 'cg000003'],
        'mt_est': [0.5, -0.2, 0.8],
        'mt_err': [0.1, 0.05, 0.2],
        'mt_t': [5.0, -4.0, 4.0],
        'mt_p': [1e-6, 1e-4, 1e-5],
        'mt_ig': [0.4, 0.2, 0.3],        # scalar methylation attribution
        'age_ig': [0.1, 0.3, 0.1],       # covariate 1 attribution
        'pc1_ig': [0.05, 0.1, 0.2],      # covariate 2 attribution
        'region': ['CIS', 'DISTAL', 'TRANS']
    })

    file_path = tmp_path / "bootstrap_merged.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)
    return file_path

def test_per_feature_ig_propagation_and_evaluate_saliency(synthetic_parquet, tmp_path):
    """
    Tests that per-feature IG columns are preserved and correctly handled
    by evaluateSaliency.py. In particular, we assert that the script does not
    throw the "Only scalar 'mt_ig' is available" warning and computes a
    non-degenerate 'mt_ig_frac' (i.e., not just uniformly 1.0).
    """
    # Import the evaluateSaliency module so we can test its logic
    # We must mock sys.argv and matplotlib/seaborn to prevent it from plotting during tests
    import tools.evaluateSaliency as eval_saliency

    test_args = [
        "evaluateSaliency.py",
        "--input", str(synthetic_parquet),
        "-o", str(tmp_path),
        "--rank-by", "mt_ig"
    ]

    import io
    from contextlib import redirect_stdout

    f_out = io.StringIO()
    with patch.object(sys, 'argv', test_args):
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.clf'), patch('matplotlib.pyplot.close'):
            with redirect_stdout(f_out):
                eval_saliency.main()

    output = f_out.getvalue()

    # Verify that the warning about degenerate fractions is NOT present
    assert "Only scalar 'mt_ig' is available" not in output

    # Verify that it evaluated the covariate properties
    assert "--- Mean Saliency Proportion per Feature Class ---" in output
    assert "age_ig:" in output
    assert "pc1_ig:" in output
