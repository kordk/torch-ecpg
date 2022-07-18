import os
import shutil
from typing import List, Tuple
import numpy as np
import pandas
import requests
from logger import Logger


def random_list(length: int, minimum: float, maximum: float) -> List[float]:
    '''
    Returns a list of length, with random float values ranging from
    minimum to maximum. Returns a list of floats.
    '''
    return list(np.random.rand(length) * (maximum - minimum) + minimum)


def download_files(
    output_dir: str,
    files: List[Tuple[str, str]],
    *,
    logger: Logger = Logger(),
) -> None:
    '''
    Downloads files from files, a list of tuples with file names and
    their corresponding urls. Saves files in output_dir. The function is
    very fast for large files. If verbose is true, it will print out the
    currently downloading file as the function runs.
    '''
    n = len(files)
    logger.start_timer('info', 'Downloading {0} files...', n)

    for file_name, url in files:
        with requests.get(url, stream=True) as stream:
            with open(output_dir + file_name, 'wb') as file:
                logger.time('Downloading {i}/{0}: {1}...', n, file_name)
                shutil.copyfileobj(stream.raw, file)
                logger.time_check('Downloaded in {l} seconds', n)

    logger.time_check(
        'Finished downloading {0} files in {t} seconds.',
        n,
    )


def initialize_dir(directory: str, *, logger: Logger = Logger()) -> None:
    '''Clears and creates provided directory'''
    if os.path.isdir(directory):
        logger.info('Removing directory {0}...', directory)
        shutil.rmtree(directory)
    logger.info('Creating directory {0}...', directory)
    os.mkdir(directory)


def read_csv(
    file_name: str, sep: str = ',', logger: Logger = Logger()
) -> pandas.DataFrame:
    '''
    Reads file_name as a csv with separator sep and returns
    pandas.DataFrame. Reads pandas-style csv, where indices and columns
    are automatically generated.
    '''
    logger.info(
        'Reading csv file {0} with separator {1}',
        '[tab]' if sep == '\t' else sep,
    )
    return pandas.read_csv(file_name, sep=sep, index_col=[0])
