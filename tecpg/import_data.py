import itertools
import os
from typing import Callable, Dict, List, Optional
import pandas
from .config import data
from .helper import initialize_dir, read_csv
from .logger import Logger

data_path = os.path.join(data['root_path'], data['input_dir'])


def save_dataframes(
    dataframes: List[pandas.DataFrame],
    output_dir: str = data_path,
    file_names: List[str] = itertools.chain(
        ('M.csv', 'G.csv', 'P.csv'), itertools.count(1)
    ),
    save_func: Callable = pandas.DataFrame.to_csv,
    *,
    logger: Logger = Logger(),
) -> None:
    '''
    Saves any number of dataframes to an output_dir, with file_names for
    each file. Default file names count up from one for as many files
    that are given. The save_func function is called with the panads
    dataframe and the output path, which defaults to saving as a csv.
    '''
    initialize_dir(output_dir, **logger)

    logger.start_timer('info', 'Saving {0} dataframes...', len(dataframes))
    for df, file_name in zip(dataframes, file_names):
        logger.time('Saving {i}/{0}: {1}', len(dataframes), file_name)
        file_path = os.path.join(output_dir, file_name)
        save_func(df, file_path)
        logger.time_check(
            'Saved {i}/{0} in {l} seconds',
            len(dataframes),
        )

    logger.time_check(
        'Finished saving {0} dataframes in {t} seconds.',
        len(dataframes),
    )


def read_dataframes(
    input_dir: str, get_func: Callable = read_csv, *, logger: Logger = Logger()
) -> Dict[str, pandas.DataFrame]:
    '''
    Gets all available csv files from input_dir and gets them using
    get_func, which, by default, reads files as csvs. The get_func
    function is called with the path to the file and returns a pandas
    dataframe. The entire function returns a dictionary of file names
    and their corresponding dataframes.
    '''
    if not os.path.isdir(input_dir):
        raise ValueError(f'{input_dir=} is not a valid directory')

    file_names = os.listdir(input_dir)
    n = len(file_names)
    if n < 1:
        raise ValueError(f'Could not find any files in {input_dir}')

    logger.start_timer('info', 'Reading {0} dataframes...', n)
    out = {}
    for file_name in file_names:
        logger.time('Reading {i}/{0}: {1}', n, file_name)
        file_path = os.path.join(input_dir, file_name)
        out[file_name] = get_func(file_path, **logger)
        logger.time_check('Read {i}/{0} in {l} seconds', n)

    logger.time_check('Finished reading {0} dataframes in {t} seconds.', n)
    return out


def save_dataframe_part(
    dataframe: pandas.DataFrame,
    file_path: str,
    first: Optional[bool] = None,
    *,
    logger: Logger = Logger(),
) -> None:
    if not os.path.isfile(file_path):
        with open(file_path, 'w') as _:
            pass
        first = True

    if first is None:
        first = os.stat(file_path).st_size == 0

    mode = 'w' if first else 'a'
    dataframe.to_csv(file_path, mode=mode, header=first)

    logger.count_check('Done saving part {i}')
