import torch
from import_data import read_dataframes
from config import WORKING_DATA_DIR
from logger import Logger
from config import OUTPUT_DATA_DIR
from helper import initialize_dir
from pearson_full import pearson_chunk_save_tensor


def main() -> None:
    logger = Logger()

    dataframes = read_dataframes(WORKING_DATA_DIR, **logger)
    M = dataframes['M.csv']
    G = dataframes['G.csv']

    torch.cuda.empty_cache()
    initialize_dir(OUTPUT_DATA_DIR, **logger)
    pearson_chunk_save_tensor(M, G, 50, 11, OUTPUT_DATA_DIR, **logger)


if __name__ == '__main__':
    main()
