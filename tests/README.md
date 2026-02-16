# Tests for tecpg

This directory contains validation and regression tests for `tecpg`.

## Files

- **`test_accuracy.py`**: Validates the accuracy of `tecpg`'s regression results against `statsmodels` (OLS). It generates synthetic data, runs `tecpg` regression, and then randomly selects pairs of methylation/gene expression to compare with `statsmodels.OLS`.
- **`test_mlr_comparison.py`**: Compares the two implementation backends of `tecpg`:
  - `regression_full` (Manual implementation of Normal Equations)
  - `tecpg_mlr_lstsq` (Using PyTorch's `linalg.lstsq`)
  This test ensures both methods produce consistent results across different chunking and region filtration scenarios.
- **`validation_utils.py`**: Contains helper functions used by `test_accuracy.py`, such as running OLS with `statsmodels` and comparing results.

## Usage

To run the accuracy validation test:
```bash
python test_accuracy.py
```

To run the MLR backend comparison test:
```bash
python test_mlr_comparison.py
```

## Troubleshooting

### `TypeError: C function scipy.spatial._qhull._barycentric_coordinates has wrong signature`

If you encounter this error when running tests, it indicates a version mismatch between `scipy` and `numpy` (or other compiled extensions) in your environment.

**Fix:**
Upgrade `scipy` to a version compatible with your `numpy` installation (typically `scipy>=1.12.0`). You can try reinstalling dependencies:

```bash
pip install --upgrade scipy numpy
# or re-install all requirements
pip install -r ../requirements.txt
```
