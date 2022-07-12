from typing import List, Tuple
import pandas
from scipy.stats import pearsonr
from samle_data import download_dataframes
from config import DATA_DIR


def calc_pearsonr(
    M: pandas.DataFrame, G: pandas.DataFrame
) -> List[Tuple[str, str, float, float]]:
    M = M.reindex(sorted(M.columns), axis=1)
    G = G.reindex(sorted(G.columns), axis=1)

    out = []
    for m_label, m_row in M.iteritems():
        for g_label, g_row in G.iteritems():
            corr, p_val = pearsonr(m_row.values, g_row.values)
            out.append((m_label, g_label, corr, p_val))
    return out


def main() -> None:
    dataframes = download_dataframes(DATA_DIR)
    M = dataframes['M.csv']
    G = dataframes['G.csv']
    print(calc_pearsonr(M, G))


if __name__ == '__main__':
    main()
