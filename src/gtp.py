import os
import shutil
from typing import Dict, List, Tuple
import pandas
from config import RAW_DATA_DIR
from geo import geo_dict, geo_samples
from import_data import save_dataframes
from helper import initialize_dir, download_files, read_csv

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
GTP_DIR = RAW_DATA_DIR + 'GTP/'
GTP_COVARIATES = [('age', '')]


def download_gtp_raw(**kwargs) -> None:
    '''
    Downloads the raw data from the Grady Trauma Project study and
    stores it in RAW_DATA_DIR/GTP/....

    The data is stored online under the NCBI Gene Expression Omnibus.
    The Covariate Matrix and DNA Methylation are stored in GSE72680,
    while Gene Expression is stored in GSE58137.

    Gene expression and methylation are stored in tab separated value
    files (.tsv). The covariate matrix is stored as a GEO array (.txt).

    Gene expression data are downloaded in two parts. They get stitched
    together in the dataframe conversion section.
    '''
    initialize_dir(GTP_DIR)
    download_files(GTP_DIR, GTP_FILE_URLS, **kwargs)


def get_gtp_dataframes(
    verbose: bool = True,
) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    '''
    Reads the raw GTP files (.txt.gz) and returns a tuple of three
    pandas.DataFrame for the methylation beta values, gene expression
    part one, and gene expression part two.

    Verbose prints the progress of reading each tsv file.
    '''
    dfs = []
    if verbose:
        print('Reading csv files:')

    for index, (file_name, _) in enumerate(GTP_FILE_URLS[1:], 1):
        if verbose:
            print(f'Reading {index}/3: {file_name}')

        dfs.append(read_csv(GTP_DIR + file_name, '\t'))

    if verbose:
        print('Done reading csv files.')
    return tuple(dfs)


def gtp_raw_clean() -> bool:
    '''
    Cleans GTP directory of files other than GTP raw files. If all four
    GTW raw files remain, returns true. Otherwise, returns false. The
    return boolean is useful for determining whether it is necessary to
    download the raw data before proceeding.
    '''
    if not os.path.exists(GTP_DIR):
        initialize_dir(RAW_DATA_DIR)
        return False
    files = os.listdir(GTP_DIR)
    target_files = [file for file, _ in GTP_FILE_URLS]
    for file in files:
        if file not in target_files:
            print(f'{file} is being removed from {GTP_DIR}')
            if os.path.isdir(GTP_DIR + file):
                shutil.rmtree(GTP_DIR + file)
            elif os.path.isfile(GTP_DIR + file):
                os.remove(GTP_DIR + file)
    return len(os.listdir(GTP_DIR)) == 4


def process_gtp(
    M: pandas.DataFrame,
    G: pandas.DataFrame,
    P: pandas.DataFrame,
    geo_descs: List[str],
    geo_titles: List[str],
) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    '''
    Processes the gtp dataframes (Methylation Beta Values and Gene
    Expression Values). Drops unneeded columns (of p-values), renames
    columns, unionizes samples, and sorts indices.
    '''
    M.drop(M.iloc[:, 1::2], axis=1, inplace=True)
    G.drop(G.iloc[:, 1::2], axis=1, inplace=True)
    G.index.name = None

    M_map = dict(zip(geo_descs, geo_titles))
    G_map = {f'Average signal_{title}': title for title in geo_titles}
    M.rename(columns=M_map, inplace=True)
    G.rename(columns=G_map, inplace=True)

    M_drop = set(M.columns) - set(G.columns)
    G_drop = set(G.columns) - set(M.columns)
    M.drop(M_drop, axis=1, inplace=True)
    G.drop(G_drop, axis=1, inplace=True)

    P_drop = set(P.index) - set(M.columns)
    P.drop(P_drop, axis=0, inplace=True)

    M = M.reindex(sorted(M.columns, key=int), axis=1)
    G = G.reindex(sorted(G.columns, key=int), axis=1)
    P = P.reindex(sorted(P.index, key=int), axis=0)

    return M, G, P


def get_phenotypes(
    chars: Dict[str, List[str]], geo_titles: List[str]
) -> pandas.DataFrame:
    '''
    Gets a dataframe of covariates given the characteristics (chars),
    mapping characteristic names with values for each sample and a list
    of sample names given by geo_titles. Filters characteristics that
    do not have the same number of values as samples to avoid missing
    data.
    '''
    n = len(geo_titles)
    full_chars = {char: vals for char, vals in chars.items() if len(vals) == n}
    P = pandas.DataFrame(full_chars, index=geo_titles)
    return P


def generate_data() -> Tuple[
    pandas.DataFrame, pandas.DataFrame, pandas.DataFrame
]:
    '''
    Generates methylation beta values, gene expression values, and
    covariates pandas.DataFrames. Returns a tuple of these three
    dataframes.
    '''
    if not gtp_raw_clean():
        download_gtp_raw()
    M, G_1, G_2 = get_gtp_dataframes()
    G = pandas.concat([G_1, G_2], axis=1)
    data, chars = geo_dict(GTP_DIR + GTP_FILE_URLS[0][0])
    geo_descs, _, geo_titles = geo_samples(data)
    P = get_phenotypes(chars, geo_titles)
    return process_gtp(M, G, P, geo_descs, geo_titles)


def save_gtp_data() -> None:
    '''
    Downloads data from www.ncbi.nlm.nih.gov. Saves GTP data in
    dataframes in the working data directory.
    '''
    data = generate_data()
    print('Saving...')
    save_dataframes(data, verbose=True)


if __name__ == '__main__':
    save_gtp_data()
