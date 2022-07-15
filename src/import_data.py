import itertools
import os
import shutil
from typing import Callable, Dict, List
import pandas


def save_dataframes(
    dataframes: List[pandas.DataFrame],
    output_dir: str,
    file_names: List[str] = itertools.count(1),
    save_func: Callable = pandas.DataFrame.to_csv,
) -> None:
    '''
    Saves any number of dataframes to an output_dir, with file_names for
    each file. Default file names count up from one for as many files
    that are given. The save_func function is called with the panads
    dataframe and the output path, which defaults to saving as a csv.
    '''
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    for df, file_name in zip(dataframes, file_names):
        save_func(df, output_dir + str(file_name))


def download_dataframes(
    input_dir: str,
    get_func: Callable = lambda file_name: pandas.read_csv(
        file_name, index_col=[0]
    ),
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

    out = {}
    for file_name in os.listdir(input_dir):
        out[file_name] = get_func(input_dir + file_name)
    return out
