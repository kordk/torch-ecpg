from compute import compute_datamg
from data import download_dataframes
from config import DATA_DIR
from structure import DataMG


def main() -> None:
    dataframes = download_dataframes(DATA_DIR)
    M = dataframes['M.csv']
    G = dataframes['G.csv']
    data = DataMG(M, G)
    res = compute_datamg(data)
    res.visualize()


if __name__ == '__main__':
    main()
