from typing import Any, Callable
import pandas
from pearson_single import pearson_corr_tensor
from structure import DataMG


def compute_dataframe(
    M: pandas.DataFrame,
    G: pandas.DataFrame,
    out: Any,
    calculate_func: Callable[[Any, Any], Any] = pearson_corr_tensor,
) -> None:
    '''
    Takes in two pandas.DataFrame in M and G. Returns a dictionary
    mapping a tuple of the gene id and methylation site id to the value
    returned by calculate_func.

    Iterates over each gene and methylation site in the input. Two lists of
    gene expression values and methylation values for each person are
    inputted into calculate_func, which returns a value that is stored in
    the output.
    '''
    for m_label, m_row in M.T.iteritems():
        for g_label, g_row in G.T.iteritems():
            m_values, g_values = m_row.values, g_row.values
            res = calculate_func(m_values, g_values)
            out[m_label, g_label] = res


def compute_datamg(
    data: DataMG,
    out: Any,
    calculate_func: Callable[[Any, Any], Any] = pearson_corr_tensor,
) -> None:
    '''
    Takes in a DataMG instance of the input data. Returns ComputeResult
    mapping gene id and methylation site id to the value returned by
    calculate_func.

    Iterates over each gene and methylation site in the input. Two lists of
    gene expression values and methylation values for each person are
    inputted into calculate_func, which returns a value that is stored in
    the output.
    '''
    for label, values in data.iterate():
        out[label] = calculate_func(*values)
