import itertools
import os
import time
from typing import Callable, Dict, List, Optional

import pandas

from .config import DEFAULT_FLOAT_FORMAT, data
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
    *args,
    logger: Logger = Logger(),
    clear_dir: bool = True,
    **kwargs,
) -> None:
    """
    Saves any number of dataframes to an output_dir, with file_names for
    each file. Default file names count up from one for as many files
    that are given. The save_func function is called with the pandas
    dataframe and the output path, which defaults to saving as a csv.
    Extra args and kwargs passed to save_func.
    """
    if clear_dir:
        initialize_dir(output_dir, **logger)
    else:
        os.makedirs(output_dir, exist_ok=True)

    logger.start_timer('info', 'Saving {0} dataframes...', len(dataframes))
    for df, file_name in zip(dataframes, file_names):
        logger.time('Saving {i}/{0}: {1}', len(dataframes), file_name)
        file_path = os.path.join(output_dir, file_name)
        save_func(
            df,
            file_path,
            *args,
            **kwargs,
        )
        logger.time_check(
            'Saved {i}/{0} in {l} seconds',
            len(dataframes),
        )

    logger.time_check(
        'Finished saving {0} dataframes in {t} seconds.',
        len(dataframes),
    )


def read_dataframes(
    input_dir: str,
    get_func: Callable = read_csv,
    file_names: Optional[List[str]] = None,
    *,
    logger: Logger = Logger(),
) -> Dict[str, pandas.DataFrame]:
    """
    Gets all available csv files from input_dir and gets them using
    get_func, which, by default, reads files as csvs. The get_func
    function is called with the path to the file and returns a pandas
    dataframe. The entire function returns a dictionary of file names
    and their corresponding dataframes.
    """
    if file_names:
        n = len(file_names)
        logger.start_timer('info', 'Reading {0} dataframes...', n)
        out = {}
        for file_name in file_names:
            logger.time('Reading {i}/{0}: {1}', n, file_name)
            if os.path.isfile(file_name):
                file_path = file_name
            elif os.path.isfile(os.path.join(input_dir, file_name)):
                file_path = os.path.join(input_dir, file_name)
            else:
                raise ValueError(
                    f'File {file_name} not found in {os.getcwd()} or'
                    f' {input_dir}'
                )

            out[file_name] = get_func(file_path, **logger)
            logger.time_check('Read {i}/{0} in {l} seconds', n)

            if file_name == 'M.csv':
                logger.info('Read in {0} methylation loci from M.csv', len(out[file_name]))
            elif file_name == 'G.csv':
                logger.info('Read in {0} genes from G.csv', len(out[file_name]))

        logger.time_check('Finished reading {0} dataframes in {t} seconds.', n)
        return out

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

        if file_name == 'M.csv':
            logger.info('Read in {0} methylation loci from M.csv', len(out[file_name]))
        elif file_name == 'G.csv':
            logger.info('Read in {0} genes from G.csv', len(out[file_name]))

    logger.time_check('Finished reading {0} dataframes in {t} seconds.', n)
    return out


def save_dataframe_part(
    dataframe: pandas.DataFrame,
    file_path: str,
    chunk_number: Optional[int] = None,
    first: Optional[bool] = None,
    output_format: str = 'csv',
    *,
    logger: Logger = Logger(),
) -> None:
    output_format = (output_format or 'csv').lower()

    if output_format == 'parquet':
        # Parquet is one file per call: rewrite the extension if the caller
        # passed a CSV-style file_format string. This keeps callers (and the
        # existing --file-format default '{meth_chunk}-{gene_chunk}.csv')
        # agnostic to the chosen output format.
        root, ext = os.path.splitext(file_path)
        if ext.lower() != '.parquet':
            file_path = root + '.parquet'

        logger.debug(
            "WRITE_START chunk={0} file={1} rows={2} mode=parquet",
            chunk_number,
            file_path,
            len(dataframe),
        )
        start_t = time.perf_counter()
        # pyarrow handles MultiIndex on rows correctly; snappy is the standard
        # fast/lightly-compressed codec and is the pyarrow default.
        dataframe.to_parquet(
            file_path,
            engine='pyarrow',
            compression='snappy',
        )
        end_t = time.perf_counter()
        ms_elapsed = (end_t - start_t) * 1000.0
        file_size = os.path.getsize(file_path)
        logger.debug(
            "WRITE_END chunk={0} file={1} rows={2} bytes={3} ms={4:.2f} mode=parquet",
            chunk_number,
            file_path,
            len(dataframe),
            file_size,
            ms_elapsed,
        )
        return

    if not os.path.isfile(file_path):
        with open(file_path, 'w') as _:
            pass
        first = True

    if first is None:
        first = os.stat(file_path).st_size == 0

    mode = 'w' if first else 'a'

    logger.debug(
        "WRITE_START chunk={0} file={1} rows={2} mode={3}",
        chunk_number,
        file_path,
        len(dataframe),
        mode,
    )

    start_t = time.perf_counter()
    dataframe.to_csv(
        file_path,
        float_format=logger.carry_data.get(
            'float_format', DEFAULT_FLOAT_FORMAT
        ),
        mode=mode,
        header=first,
        chunksize=logger.carry_data.get('csv_chunksize', 100_000),
    )
    end_t = time.perf_counter()

    ms_elapsed = (end_t - start_t) * 1000.0
    file_size = os.path.getsize(file_path)

    logger.debug(
        "WRITE_END chunk={0} file={1} rows={2} bytes={3} ms={4:.2f} mode={5}",
        chunk_number,
        file_path,
        len(dataframe),
        file_size,
        ms_elapsed,
        mode,
    )
