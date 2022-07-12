from typing import Any, Callable, Dict, Tuple
import pandas
from scipy.stats import pearsonr
from samle_data import download_dataframes
from config import DATA_DIR


def compute_cpu(
    M: pandas.DataFrame,
    G: pandas.DataFrame,
    calculate_func: Callable[[Any, Any], Any] = pearsonr,
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


def main() -> None:
    dataframes = download_dataframes(DATA_DIR)
    M = dataframes['M.csv']
    G = dataframes['G.csv']
    print(compute_cpu(M, G))


if __name__ == '__main__':
    main()
