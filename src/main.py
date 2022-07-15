from compute import datamg_cr
from import_data import download_dataframes
from config import DATA_DIR
from structure import DataMG


def main() -> None:
    dataframes = download_dataframes(DATA_DIR)
    M = dataframes['M.csv']
    G = dataframes['G.csv']
    data = DataMG(M, G)
    res = datamg_cr(data)
    res.visualize_image()


if __name__ == '__main__':
    main()
