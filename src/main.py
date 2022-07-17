from time import time
import torch
from import_data import download_dataframes
from config import WORKING_DATA_DIR
from pearson_full import pearson_chunk_tensor


def main() -> None:
    dataframes = download_dataframes(WORKING_DATA_DIR)
    M = dataframes['M.csv']
    G = dataframes['G.csv']

    torch.cuda.empty_cache()
    start = time()
    pearson_chunk_tensor(M, G, 10000, verbose=True)
    t1 = time() - start
    print(f'Chunk tensor corr time: {t1} seconds')


if __name__ == '__main__':
    main()
