import torch
from import_data import read_dataframes
from config import WORKING_DATA_DIR
from methylation_gpu.gpu_methylation.src.logger import Logger
from pearson_full import pearson_chunk_tensor


def main() -> None:
    logger = Logger()

    dataframes = read_dataframes(WORKING_DATA_DIR, **logger)
    M = dataframes['M.csv']
    G = dataframes['G.csv']

    torch.cuda.empty_cache()
    pearson_chunk_tensor(M, G, 10000, **logger)


if __name__ == '__main__':
    main()
