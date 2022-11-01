import os
from typing import List
import click
import torch
from .helper import initialize_dir
from .config import data, using_gpu
from .logger import Logger
from .gtp import save_gtp_data
from .import_data import read_dataframes, save_dataframes
from .test_data import generate_data
from .pearson_full import (
    pearson_full_tensor,
    pearson_chunk_tensor,
    pearson_chunk_save_tensor,
)
from .regression_full import regression_full


@click.group()
@click.option(
    '-r',
    '--root-path',
    show_default=True,
    default=data['root_path'],
    type=click.Path(file_okay=False, readable=True, resolve_path=True),
)
@click.option(
    '-i',
    '--input-dir',
    show_default=True,
    default=data['input_dir'],
    type=click.Path(file_okay=False),
)
@click.option(
    '-o',
    '--output-dir',
    show_default=True,
    default=data['output_dir'],
    type=click.Path(file_okay=False),
)
@click.option(
    '-m',
    '--meth-file',
    show_default=True,
    default=data['meth_file'],
    type=click.Path(dir_okay=False),
)
@click.option(
    '-g',
    '--gene-file',
    show_default=True,
    default=data['gene_file'],
    type=click.Path(dir_okay=False),
)
@click.option(
    '-c',
    '--covar-file',
    show_default=True,
    default=data['covar_file'],
    type=click.Path(dir_okay=False),
)
@click.option(
    '-f',
    '--output-file',
    show_default=True,
    default=data['output'],
    type=click.Path(dir_okay=False),
)
@click.option(
    '-t',
    '--cpu-threads',
    show_default=True,
    default=0,
    type=int,
    help='If 0, runs on the GPU if available',
)
@click.option(
    '-v', '--verbosity', show_default=True, default=1, type=int, count=True
)
@click.option(
    '-d', '--debug', is_flag=True, show_default=True, default=False, type=bool
)
@click.option(
    '-l',
    '--log-dir',
    show_default=True,
    default=data['log_dir'],
    type=click.Path(file_okay=False),
)
@click.option(
    '-n',
    '--no-log-file',
    is_flag=True,
    show_default=True,
    default=False,
    type=bool,
)
@click.pass_context
def cli(
    ctx: click.Context,
    root_path: str,
    input_dir: str,
    output_dir: str,
    meth_file: str,
    gene_file: str,
    covar_file: str,
    output_file: str,
    cpu_threads: int,
    verbosity: int,
    debug: bool,
    log_dir: str,
    no_log_file: bool,
) -> None:
    '''The root cli group'''
    ctx.ensure_object(dict)

    data['root_path'] = click.format_filename(root_path)
    data['input_dir'] = click.format_filename(input_dir)
    data['output_dir'] = click.format_filename(output_dir)
    data['meth_file'] = click.format_filename(meth_file)
    data['gene_file'] = click.format_filename(gene_file)
    data['covar_file'] = click.format_filename(covar_file)
    data['output_file'] = click.format_filename(output_file)
    data['log_dir'] = click.format_filename(log_dir)

    log_path = None if no_log_file else os.path.join(root_path, log_dir)
    logger = Logger(verbosity, debug, log_path)
    using_gpu(**logger)
    if cpu_threads:
        torch.set_num_threads(cpu_threads)
        logger.carry_data['use_cpu'] = True
    ctx.obj['logger'] = logger


@cli.group()
def run() -> None:
    '''Base group for running algorithms.'''


@run.command()
@click.option('-c', '--chunks', show_default=True, default=0, type=int)
@click.option('-s', '--save-chunks', show_default=True, default=0, type=int)
@click.option(
    '-f',
    '--flatten',
    is_flag=True,
    show_default=True,
    default=True,
    type=bool,
)
@click.pass_context
def corr(
    ctx: click.Context, chunks: int, save_chunks: int, flatten: bool
) -> None:
    '''
    Calculate the pearson correlation coefficient.

    Calculate the pearson correlation coefficient with methylation and
    gene expression matrices. Optional compute and save chunking to
    avoid GPU and CPU memory limits.
    '''
    logger: Logger = ctx.obj['logger']

    data_path = os.path.join(data['root_path'], data['input_dir'])
    dataframes = read_dataframes(data_path, **logger)
    M = dataframes[data['meth_file']]
    G = dataframes[data['gene_file']]

    output_path = os.path.join(data['root_path'], data['output_dir'])
    output = None
    print(chunks, save_chunks)
    if chunks == 0:
        output = pearson_full_tensor(M, G, flatten=flatten, **logger)
    elif save_chunks == 0:
        output = pearson_chunk_tensor(M, G, chunks, flatten=flatten, **logger)
    else:
        pearson_chunk_save_tensor(
            M, G, chunks, save_chunks, output_path, flatten=flatten, **logger
        )
    if output is not None:
        save_dataframes([output], output_path, [data['output_file']], **logger)

    logger.save()


@run.command()
@click.option(
    '--exclude-est', is_flag=True, show_default=True, default=False, type=bool
)
@click.option(
    '--exclude-err', is_flag=True, show_default=True, default=False, type=bool
)
@click.option(
    '--exclude-t', is_flag=True, show_default=True, default=False, type=bool
)
@click.option(
    '--exclude-p', is_flag=True, show_default=True, default=False, type=bool
)
@click.pass_context
def mlr(
    ctx: click.Context,
    exclude_est: bool,
    exclude_err: bool,
    exclude_t: bool,
    exclude_p: bool,
) -> None:
    '''
    Calculates the multiple linear regression.

    Calculate the multiple linear regression with methylation and gene
    expression matrices.
    '''
    logger: Logger = ctx.obj['logger']

    data_path = os.path.join(data['root_path'], data['input_dir'])
    output_path = os.path.join(data['root_path'], data['output_dir'])
    dataframes = read_dataframes(data_path, **logger)
    M = dataframes[data['meth_file']]
    G = dataframes[data['gene_file']]
    C = dataframes[data['covar_file']]
    include = (not exclude_est, not exclude_err, not exclude_t, not exclude_p)

    output = regression_full(M, G, C, include, **logger)
    save_dataframes([output], output_path, [data['output_file']], **logger)


@cli.group(name='data')
def _data() -> None:
    '''Base group for data management.'''


@_data.command()
@click.option('-s', '--samples', type=int, prompt=True)
@click.option('-m', '--meth-rows', type=int, prompt=True)
@click.option('-g', '--gene-rows', type=int, prompt=True)
@click.pass_context
def dummy(
    ctx: click.Context,
    samples: int,
    meth_rows: int,
    gene_rows: int,
) -> None:
    '''
    Generates dummy data.

    Generates dummy data in the output directory with a given size with
    file names M.csv, G.csv, and C.csv.
    '''
    logger: Logger = ctx.obj['logger']

    dataframes = generate_data(samples, meth_rows, gene_rows)
    file_names = [data['meth_file'], data['gene_file'], data['covar_file']]
    data_path = os.path.join(data['root_path'], data['input_dir'])
    save_dataframes(dataframes, data_path, file_names, **logger)

    logger.save()


def abort_if_false(ctx: click.Context, _, value):
    if not value:
        ctx.abort()


@_data.command()
@click.option(
    '-g',
    '--gtp-dir',
    show_default=True,
    default='GTP',
    type=click.Path(file_okay=False),
)
@click.option(
    '-y',
    '--yes',
    is_flag=True,
    callback=abort_if_false,
    expose_value=False,
    prompt='Are you sure you want to overwrite the data directory?',
)
@click.pass_context
def gtp(ctx: click.Context, gtp_dir) -> None:
    '''
    Downloads and extracts GTP data.

    Downloads the methylation, gene expression, and covariate data from
    Grady Trauma Project study. Stores the raw data in gtp-dir. The raw
    data is extracted and processes before being saved in the data
    directory.
    '''
    logger: Logger = ctx.obj['logger']

    gtp_path = os.path.join(data['root_path'], gtp_dir)
    data_path = os.path.join(data['root_path'], data['input_dir'])
    file_names = [data['meth_file'], data['gene_file'], data['covar_file']]
    save_gtp_data(gtp_path, data_path, file_names, **logger)

    logger.save()


@cli.command()
@click.argument(
    'root-dirs',
    nargs=-1,
    type=click.Path(file_okay=False),
)
@click.option(
    '-y',
    '--yes',
    is_flag=True,
    callback=abort_if_false,
    expose_value=False,
    prompt=(
        'Are you sure you want to reset and initialize'
        f' {data["root_path"]}/[root_dir]?'
    ),
)
@click.pass_context
def init(ctx: click.Context, root_dirs: List[str]) -> None:
    '''
    Creates and initializes directory.

    Creates root_dir in the root_path. Creates input_dir and output_dir
    in this new directory. Changes directory too this new directory.
    '''
    logger: Logger = ctx.obj['logger']

    if not root_dirs:
        root_dir = 'tecpg_testing'
    else:
        root_dir = root_dirs[0]

    path = os.path.join(data['root_path'], root_dir)
    initialize_dir(path, **logger)
    os.mkdir(os.path.join(path, data['input_dir']))
    os.mkdir(os.path.join(path, data['output_dir']))
    os.mkdir(os.path.join(path, data['log_dir']))
    logger.info('Enter the {0} directory to start working.', path)

    log_dir = os.path.join(path, data['log_dir'])
    logger.save(log_dir=log_dir)


def start() -> None:
    cli(obj={})
