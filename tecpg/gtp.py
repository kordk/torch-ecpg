import os
import shutil
from typing import Dict, List, Optional, Tuple

import pandas

from .geo import geo_dict, geo_samples
from .helper import download_files, initialize_dir, read_csv
from .import_data import save_dataframes
from .logger import Logger

GTP_FILE_URLS = [
    (
        'CovariateMatrix.txt.gz',
        'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE72nnn/GSE72680/matrix/GSE7'
        '2680_series_matrix.txt.gz',
    ),
    (
        'MethylationBetaValues.tsv.gz',
        'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE72nnn/GSE72680/suppl/GSE72'
        '680_beta_values.txt.gz',
    ),
    (
        'GeneExpressionValues_1.tsv.gz',
        'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE58nnn/GSE58137/suppl/GSE58'
        '137_Raw_119_samplesremoved.csv.gz',
    ),
    (
        'GeneExpressionValues_2.tsv.gz',
        'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE58nnn/GSE58137/suppl/GSE58'
        '137_Raw_279_samplesremoved.csv.gz',
    ),
]


def download_gtp_raw(
    gtp_path: str, logger: Logger = Logger(), **kwargs
) -> None:
    """
    Downloads the raw data from the Grady Trauma Project study and
    stores it in RAW_DATA_DIR/GTP/....

    The data is stored online under the NCBI Gene Expression Omnibus.
    The Covariate Matrix and DNA Methylation are stored in GSE72680,
    while Gene Expression is stored in GSE58137.

    Gene expression and methylation are stored in tab separated value
    files (.tsv). The covariate matrix is stored as a GEO array (.txt).

    Gene expression data are downloaded in two parts. They get stitched
    together in the dataframe conversion section.
    """
    initialize_dir(gtp_path, **logger)
    logger.info('Downloading GTP raw data (this could take a very long time)')
    download_files(gtp_path, GTP_FILE_URLS, **kwargs, **logger)


def get_gtp_dataframes(
    gtp_path: str, *, logger: Logger = Logger()
) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """
    Reads the raw GTP files (.txt.gz) and returns a tuple of three
    pandas.DataFrame for the methylation beta values, gene expression
    part one, and gene expression part two.

    Verbose prints the progress of reading each tsv file.
    """
    dfs = []

    logger.start_timer('info', 'Reading 3 csv files...')
    for file_name, _ in GTP_FILE_URLS[1:]:
        logger.time('Reading {i}/3: {0}', file_name)
        file_path = os.path.join(gtp_path, file_name)
        df = read_csv(file_path, '\t', **logger)
        dfs.append(df)
        logger.time_check('Read {i}/3 in {l} seconds')

    logger.time_check(
        'Finished reading GTP csv files in {t} seconds.',
    )
    return tuple(dfs)


def gtp_raw_clean(gtp_path: str, *, logger: Logger = Logger()) -> bool:
    """
    Cleans GTP directory of files other than GTP raw files. If all four
    GTP raw files remain, returns true. Otherwise, returns false. The
    return boolean is useful for determining whether it is necessary to
    download the raw data before proceeding.
    """
    if not os.path.exists(gtp_path):
        initialize_dir(gtp_path, **logger)
        return False

    files = os.listdir(gtp_path)
    target_files = [file for file, _ in GTP_FILE_URLS]
    for file in files:
        if file not in target_files:
            logger.warning(f'{file} is being removed from {gtp_path}')
            file_path = os.path.join(gtp_path, file)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            elif os.path.isfile(file_path):
                os.remove(file_path)

    remaining = len(os.listdir(gtp_path))
    return remaining == 4


def process_gtp(
    M: pandas.DataFrame,
    G: pandas.DataFrame,
    C: pandas.DataFrame,
    geo_descs: List[str],
    geo_titles: List[str],
    drop_na: bool = True,
    *,
    logger: Logger = Logger(),
) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """
    Processes the gtp dataframes (Methylation Beta Values, Gene
    Expression Values, and Covariate Matrix). Drops unneeded columns (of
    p-values), renames columns, unionizes samples, and sorts indices.
    """
    logger.info('Dropping unneeded columns (p-values)')
    M.drop(M.iloc[:, 1::2], axis=1, inplace=True)
    G.drop(G.iloc[:, 1::2], axis=1, inplace=True)
    G.index.name = None

    logger.info('Normalizing column names')
    M_map = dict(zip(geo_descs, geo_titles))
    G_map = {f'Average signal_{title}': title for title in geo_titles}
    M.rename(columns=M_map, inplace=True)
    G.rename(columns=G_map, inplace=True)

    logger.info('Removing nonoverlapping columns')
    M_drop = set(M.columns) - set(G.columns)
    G_drop = set(G.columns) - set(M.columns)
    M.drop(M_drop, axis=1, inplace=True)
    G.drop(G_drop, axis=1, inplace=True)

    if drop_na:
        G_start = len(G)
        G.dropna(axis=0, inplace=True)
        G_end = len(G)
        logger.info(
            'Dropped {0} rows of G with missing values ({1}% remaining)',
            G_start - G_end,
            round(G_end / G_start * 100, 4),
        )
        M_start = len(M)
        M.dropna(axis=0, inplace=True)
        M_end = len(M)
        logger.info(
            'Dropped {0} rows of M with missing values ({1}% remaining)',
            M_start - M_end,
            round(M_end / M_start * 100, 4),
        )

    C_drop = set(C.index) - set(M.columns)
    C.drop(C_drop, axis=0, inplace=True)
    C.drop('tissue', axis=1, inplace=True, errors='ignore')
    C.drop('race/ethnicity', axis=1, inplace=True, errors='ignore')

    C['Sex'].replace('Male', 0, inplace=True)
    C['Sex'].replace('Female', 1, inplace=True)

    logger.info('Sorting columns')
    M = M.reindex(sorted(M.columns, key=int), axis=1)
    G = G.reindex(sorted(G.columns, key=int), axis=1)
    C = C.reindex(sorted(C.index, key=int), axis=0)

    return M, G, C


def get_covariates(
    chars: Dict[str, List[str]],
    geo_titles: List[str],
    *,
    logger: Logger = Logger(),
) -> pandas.DataFrame:
    """
    Gets a dataframe of covariates given the characteristics (chars),
    mapping characteristic names with values for each sample and a list
    of sample names given by geo_titles. Filters characteristics that
    do not have the same number of values as samples to avoid missing
    data.
    """
    n = len(geo_titles)
    logger.info('Removing covariates without enough data for all samples')
    full_chars = {char: vals for char, vals in chars.items() if len(vals) == n}
    C = pandas.DataFrame(full_chars, index=geo_titles)
    return C


def generate_data(
    gtp_path: str,
    simplify_covar: bool = False,
    drop_na: bool = True,
    *,
    logger: Logger = Logger(),
) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """
    Generates methylation beta values, gene expression values, and
    covariates pandas.DataFrames. Returns a tuple of these three
    dataframes.
    """
    if not gtp_raw_clean(gtp_path, **logger):
        download_gtp_raw(gtp_path, **logger)
    M, G_1, G_2 = get_gtp_dataframes(gtp_path, **logger)
    logger.info('Concatenating gene expression parts')
    G = pandas.concat([G_1, G_2], axis=1)
    covar_file = GTP_FILE_URLS[0][0]
    data, chars = geo_dict(
        os.path.join(gtp_path, covar_file),
        simplify_covar=simplify_covar,
        **logger,
    )
    geo_descs, _, geo_titles = geo_samples(data)
    C = get_covariates(chars, geo_titles, **logger)
    return process_gtp(M, G, C, geo_descs, geo_titles, drop_na, **logger)


def save_gtp_data(
    gtp_path: str,
    data_path: str,
    file_names: Optional[List[str]] = None,
    simplify_covar: bool = False,
    drop_na: bool = True,
    *,
    logger: Logger = Logger(),
) -> None:
    """
    Downloads data from www.ncbi.nlm.nih.gov. Saves GTP data in
    dataframes in the working data directory.
    """
    data = generate_data(gtp_path, simplify_covar, drop_na, **logger)
    logger.info('Saving into {0}', data_path)
    if file_names is None:
        save_dataframes(data, data_path, **logger)
    else:
        save_dataframes(data, data_path, file_names, **logger)
    logger.warning(
        'GTP methylation, gene expression, and covariates downloaded. If you'
        ' would like to use region filtration, please manually copy the'
        ' associated files from the tecpg/demo directory or produce them'
        ' yourself.'
    )
