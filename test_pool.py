import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def save_dummy(df, path):
    df.to_csv(path)
    return True

def main():
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
        list(pool.map(int, range(2)))
        df = pd.DataFrame({"a": [1, 2]})
        f = pool.submit(save_dummy, df, "test_out.csv")
        print(f.result())

if __name__ == '__main__':
    main()
