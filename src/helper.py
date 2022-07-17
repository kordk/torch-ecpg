import os
import shutil
from typing import List, Tuple
import numpy as np
import pandas
import requests


def random_list(length: int, minimum: float, maximum: float) -> List[float]:
    '''
    Returns a list of length, with random float values ranging from
    minimum to maximum. Returns a list of floats.
    '''
    return list(np.random.rand(length) * (maximum - minimum) + minimum)


def download_files(
    output_dir: str,
    files: List[Tuple[str, str]],
    verbose: bool = True,
) -> None:
    '''
    Downloads files from files, a list of tuples with file names and
    their corresponding urls. Saves files in output_dir. The function is
    very fast for large files. If verbose is true, it will print out the
    currently downloading file as the function runs.
    '''
    n = len(files)
    if verbose:
        print(f'Downloading {n} files:')

    for index, (file_name, url) in enumerate(files, 1):
        with requests.get(url, stream=True) as stream:
            with open(output_dir + file_name, 'wb') as file:
                if verbose:
                    print(f'Downloading {index}/{n}: {file_name}...')
                shutil.copyfileobj(stream.raw, file)

    if verbose:
        print(f'Successfully downloaded {n} files.')


def initialize_dir(directory: str) -> None:
    '''Clears and creates provided directory'''
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)


def read_csv(file_name: str, sep: str = ',') -> pandas.DataFrame:
    '''
    Reads file_name as a csv with separater sep and returns
    pandas.DataFrame. Reads pandas-style csv, where indices and columns
    are automatically generated.
    '''
    return pandas.read_csv(file_name, sep=sep, index_col=[0])
