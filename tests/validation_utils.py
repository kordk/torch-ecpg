
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict, Any, List

def run_statsmodels_ols(
    m_data: pd.Series,
    g_data: pd.Series,
    c_data: pd.DataFrame
) -> Dict[str, float]:
    """
    Runs OLS regression using statsmodels: G ~ Intercept + M + C

    Args:
        m_data: Series containing methylation data for one site across samples.
        g_data: Series containing gene expression data for one gene across samples.
        c_data: DataFrame containing covariates across samples.

    Returns:
        Dictionary containing:
            - mt_est: Coefficient for methylation
            - mt_err: Standard error for methylation coefficient
            - mt_t: T-statistic for methylation coefficient
            - mt_p: P-value for methylation coefficient (Student's t)
    """
    # Align data (though they should be aligned by index already if passed correctly)
    # Construct design matrix: Intercept, M, C

    # Ensure all are aligned by index
    common_index = m_data.index.intersection(g_data.index).intersection(c_data.index)
    if len(common_index) != len(m_data):
        raise ValueError("Indices of M, G, and C do not match.")

    y = g_data.loc[common_index]

    # Create design matrix X
    # tecpg order: Intercept, Methylation, Covariates
    X = c_data.loc[common_index].copy()
    X.insert(0, 'mt', m_data.loc[common_index])
    X = sm.add_constant(X, prepend=True) # Adds 'const' column at the beginning

    # Check column order to match tecpg expectations if needed, but for OLS results extraction it matters
    # that we extract the right coefficient.
    # tecpg output columns: const, mt, c1, c2...
    # statsmodels add_constant adds 'const'.

    model = sm.OLS(y, X)
    results = model.fit()

    # Extract results for 'mt' (methylation)
    return {
        'mt_est': results.params['mt'],
        'mt_err': results.bse['mt'],
        'mt_t': results.tvalues['mt'],
        'mt_p': results.pvalues['mt']
    }

def compare_results(
    tecpg_res: Dict[str, float],
    sm_res: Dict[str, float]
) -> Dict[str, float]:
    """
    Compares tecpg results with statsmodels results.
    """
    return {
        'diff_est': abs(tecpg_res['mt_est'] - sm_res['mt_est']),
        'diff_err': abs(tecpg_res['mt_err'] - sm_res['mt_err']),
        'diff_t': abs(tecpg_res['mt_t'] - sm_res['mt_t']),
        'diff_p': abs(tecpg_res['mt_p'] - sm_res['mt_p']),
        'rel_diff_est': abs(tecpg_res['mt_est'] - sm_res['mt_est']) / (abs(sm_res['mt_est']) + 1e-9),
    }

def save_scatter_plot(
    x: Any,
    y: Any,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str
):
    """
    Generates a scatter plot with a y=x reference line and saves it.
    """
    output_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, alpha=0.5)

    # Add y=x line
    # Calculate limits based on data, handle cases where data might be constant or empty
    if len(x) > 0 and len(y) > 0:
        min_val = min(np.min(x), np.min(y))
        max_val = max(np.max(x), np.max(y))
        padding = (max_val - min_val) * 0.05
        if padding == 0:
            padding = 1.0 # Default padding if all values are same

        lims = [min_val - padding, max_val + padding]
        plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        plt.xlim(lims)
        plt.ylim(lims)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot to {filepath}")
