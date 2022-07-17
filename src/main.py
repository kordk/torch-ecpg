from time import time
from import_data import download_dataframes
from config import WORKING_DATA_DIR
from compute import compute_dataframe
from structure import FlatComputeResult
from pearson_full import pearson_full_tensor


def main() -> None:
    dataframes = download_dataframes(WORKING_DATA_DIR)
    M = dataframes['M.csv']
    G = dataframes['G.csv']

    print(
        'Comparing iterative tensor corr to full tensor corr for'
        f' {len(M.columns)}x{len(M.index)}x{len(G.index)}. Total of'
        f' {len(M.columns)*len(M.index)*len(G.index)} values.'
    )

    res_single = FlatComputeResult()
    start = time()
    res_full = pearson_full_tensor(M, G)
    t1 = time() - start
    print(f'Full tensor corr time: {t1} seconds')
    start = time()
    compute_dataframe(M, G, res_single)
    t2 = time() - start
    print(f'Single tensor corr time: {t2}')

    print(f'Full tensor is {t2/t1} times faster than single tensor.')

    print('Max error:')
    print(max(res_full.sub(res_single.dataframe()).values.flat))


if __name__ == '__main__':
    main()
