import os
import shutil
from typing import Dict, List, Optional, Tuple

import pandas

from .geo import geo_dict, geo_samples
from .helper import download_files, initialize_dir, read_csv
from .import_data import save_dataframes
from .logger import Logger


MESA_KENNEDY_URL = (
    'eCpGs_Kennedy2018_MESA.txt',
    'https://static-content.springer.com/esm/art%3A10.1186%2Fs12864-018-4842-3/MediaObjects/12864_2018_4842_MOESM2_ESM.txt'
)

MESA_FILE_URLS = [
    (
        'CovariateMatrix.txt.gz',
        'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE56nnn/GSE56045/matrix/GSE56045_series_matrix.txt.gz',
    ),
    (
        'MethylationBetaValues.tsv.gz',
        'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE56nnn/GSE56046/suppl/GSE56046%5Fmethylome%5Fnormalized.txt.gz',
    ),
    (
        'GeneExpressionValues.tsv.gz',
        'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE56nnn/GSE56045/suppl/GSE56045%5Fnon%5Fnormalized.txt.gz',
    ),
]


def download_mesa_raw(
    mesa_path: str, logger: Logger = Logger(), **kwargs
) -> None:
    """
    Downloads the raw data from the MESA study and
    stores it in RAW_DATA_DIR/MESA/....
    """
    initialize_dir(mesa_path, **logger)
    logger.info('Downloading MESA raw data (this could take a very long time)')
    download_files(mesa_path, MESA_FILE_URLS, **kwargs, **logger)

    logger.info('Downloading eCpG-transcript pairs from Kennedy et al. 2018 study (PubMed ID 29914364; DOI: 10.1186/s12864-018-4842-3)')
    download_files(mesa_path, [MESA_KENNEDY_URL], **kwargs, **logger)


def get_mesa_dataframes(
    mesa_path: str, *, logger: Logger = Logger()
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Reads the raw MESA files (.txt.gz) and returns a tuple of two
    pandas.DataFrame for the methylation beta values and gene expression.
    """
    dfs = []

    logger.start_timer('info', 'Reading 2 csv files...')
    for i, (file_name, _) in enumerate(MESA_FILE_URLS[1:], 1):
        logger.time('Reading {0}/2: {1}', i, file_name)
        file_path = os.path.join(mesa_path, file_name)
        df = read_csv(file_path, '\t', **logger)
        dfs.append(df)
        logger.time_check('Read {i}/2 in {l} seconds')

    logger.time_check(
        'Finished reading MESA csv files in {t} seconds.',
    )
    return tuple(dfs)


def mesa_raw_clean(mesa_path: str, *, logger: Logger = Logger()) -> bool:
    """
    Cleans MESA directory of files other than MESA raw files. If all three
    MESA raw files remain, returns true. Otherwise, returns false. The
    return boolean is useful for determining whether it is necessary to
    download the raw data before proceeding.
    """
    if not os.path.exists(mesa_path):
        initialize_dir(mesa_path, **logger)
        return False

    files = os.listdir(mesa_path)
    target_files = [file for file, _ in MESA_FILE_URLS] + [MESA_KENNEDY_URL[0]]
    for file in files:
        if file not in target_files:
            logger.warning(f'{file} is being removed from {mesa_path}')
            file_path = os.path.join(mesa_path, file)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            elif os.path.isfile(file_path):
                os.remove(file_path)

    remaining = len(os.listdir(mesa_path))
    return remaining == len(target_files)


def process_mesa(
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
    Processes the mesa dataframes (Methylation Beta Values, Gene
    Expression Values, and Covariate Matrix). Drops unneeded columns (of
    p-values), renames columns, unionizes samples, and sorts indices.
    """
    logger.info('Dropping unneeded columns (p-values)')
    # In MESA, M has columns like '100001.Mvalue' and '100001.detectionPval'
    # G has columns like '100001.intensity' and '100001.detectionPval'
    M.drop(columns=M.columns[1::2], inplace=True)
    G.drop(columns=G.columns[1::2], inplace=True)
    G.index.name = None
    M.index.name = None

    logger.info('Normalizing column names')
    # MESA methylation has column names as '100001.Mvalue', G as '100001.intensity'
    # geo_titles look like '100001_peripheral_CD14'. We need to extract the ID '100001'
    # and map the columns in M and G to just '100001'.

    sample_ids = [title.split('_')[0] for title in geo_titles]

    # Check what columns look like after dropping detection p-vals
    # M.columns should be ['100001.Mvalue', '100002.Mvalue', ...]
    M_cols = M.columns
    G_cols = G.columns

    # Try mapping from column name to ID prefix
    M_map = {col: col.split('.')[0] for col in M_cols}
    G_map = {col: col.split('.')[0] for col in G_cols}

    M.rename(columns=M_map, inplace=True)
    G.rename(columns=G_map, inplace=True)

    # Now covariate has index '100001_peripheral_CD14' (geo_titles)
    # We rename index to '100001'
    C_map = dict(zip(geo_titles, sample_ids))
    C.rename(index=C_map, inplace=True)

    logger.info('Removing nonoverlapping columns')
    M_drop = set(M.columns) - set(G.columns)
    G_drop = set(G.columns) - set(M.columns)
    M.drop(M_drop, axis=1, inplace=True)
    G.drop(G_drop, axis=1, inplace=True)

    logger.info('Data diagnostics for Gene Expression (G):')
    logger.info(f'  Min: {G.min().min():.4f}, Max: {G.max().max():.4f}')
    logger.info('Data diagnostics for Methylation (M):')
    logger.info(f'  Min: {M.min().min():.4f}, Max: {M.max().max():.4f}')

    if drop_na:
        G_start = len(G)
        G_na = G.isna().any(axis=1)
        G.drop(G[G_na].index, inplace=True)
        G_end = len(G)
        dropped_G = G_start - G_end
        if dropped_G == 0:
            logger.info('No NaNs found in G.')
        else:
            logger.info(
                f'Excluded {dropped_G} gene expression loci due to missing data '
                f'({round(G_end / G_start * 100, 4)}% remaining)'
            )

        M_start = len(M)
        M_na = M.isna().any(axis=1)
        M.drop(M[M_na].index, inplace=True)
        M_end = len(M)
        dropped_M = M_start - M_end
        if dropped_M == 0:
            logger.info('No NaNs found in M.')
        else:
            logger.info(
                f'Excluded {dropped_M} methylation loci due to missing data '
                f'({round(M_end / M_start * 100, 4)}% remaining)'
            )

    C_drop = set(C.index) - set(M.columns)
    C.drop(C_drop, axis=0, inplace=True)

    # Also drop any columns in M and G that are not in C
    M_drop_from_C = set(M.columns) - set(C.index)
    G_drop_from_C = set(G.columns) - set(C.index)
    M.drop(columns=list(M_drop_from_C), inplace=True)
    G.drop(columns=list(G_drop_from_C), inplace=True)

    logger.info('Applying floor to zero and log2(x + 1) transformation to Gene Expression (G)')
    import numpy as np
    G = G.clip(lower=0)
    G = np.log2(G + 1)

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
    mesa_path: str,
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
    if not mesa_raw_clean(mesa_path, **logger):
        download_mesa_raw(mesa_path, **logger)
    M, G = get_mesa_dataframes(mesa_path, **logger)
    covar_file = MESA_FILE_URLS[0][0]
    data, chars = geo_dict(
        os.path.join(mesa_path, covar_file),
        simplify_covar=False,
        **logger,
    )
    geo_descs, _, geo_titles = geo_samples(data)

    if simplify_covar:
        # Keep only 'age' and 'racegendersite' keys in chars
        chars = {k: v for k, v in chars.items() if k in ['age', 'racegendersite']}

    C = get_covariates(chars, geo_titles, **logger)
    return process_mesa(M, G, C, geo_descs, geo_titles, drop_na, **logger)


def save_mesa_data(
    mesa_path: str,
    data_path: str,
    file_names: Optional[List[str]] = None,
    simplify_covar: bool = False,
    drop_na: bool = True,
    *,
    logger: Logger = Logger(),
) -> None:
    """
    Downloads data from www.ncbi.nlm.nih.gov. Saves MESA data in
    dataframes in the working data directory.
    """
    data = generate_data(mesa_path, simplify_covar, drop_na, **logger)
    logger.info('Saving into {0}', data_path)
    if file_names is None:
        save_dataframes(data, data_path, **logger)
    else:
        save_dataframes(data, data_path, file_names, **logger)

    kennedy_filename = MESA_KENNEDY_URL[0]
    kennedy_src = os.path.join(mesa_path, kennedy_filename)
    kennedy_dst = os.path.join(data_path, kennedy_filename)
    if os.path.exists(kennedy_src):
        shutil.copyfile(kennedy_src, kennedy_dst)
        logger.info('Copied {0} to {1}', kennedy_filename, data_path)

    logger.warning(
        'MESA methylation, gene expression, and covariates downloaded. If you'
        ' would like to use region filtration, please manually copy the'
        ' associated files from the tecpg/demo directory or produce them'
        ' yourself.'
    )
