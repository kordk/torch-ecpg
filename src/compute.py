from typing import Any, Callable, Dict, Tuple
import pandas
from pearson import pearson_corr_tensor
from structure import DataMG


def compute_cpu_df(
    M: pandas.DataFrame,
    G: pandas.DataFrame,
    calculate_func: Callable[[Any, Any], Any] = pearson_corr_tensor,
) -> Dict[Tuple[str, str], Any]:
    M = M.reindex(sorted(M.columns), axis=1)
    G = G.reindex(sorted(G.columns), axis=1)

    out = {}
    for m_label, m_row in M.iteritems():
        for g_label, g_row in G.iteritems():
            m_values, g_values = m_row.values, g_row.values
            res = calculate_func(m_values, g_values)
            out[m_label, g_label] = res
    return out


def compute_cpu_datamg(
    data: DataMG,
    calculate_func: Callable[[Any, Any], Any] = pearson_corr_tensor,
) -> Dict[Tuple[str, str], Any]:
    out = {}
    for label, values in data.iterate():
        out[label] = calculate_func(*values)
    return out
