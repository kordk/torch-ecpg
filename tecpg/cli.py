import click
from .logger import Logger
from .import_data import read_dataframes
from .config import OUTPUT_DATA_DIR, WORKING_DATA_DIR
from .import_data import save_dataframes
from .test_data import generate_data
from .pearson_full import (
    pearson_full_tensor,
    pearson_chunk_tensor,
    pearson_chunk_save_tensor,
)


@click.group()
@click.option(
    '-v', '--verbosity', show_default=True, default=1, type=int, count=True
)
@click.option(
    '-d', '--debug', is_flag=True, show_default=True, default=False, type=bool
)
@click.option('-l', '--log-dir', show_default=True, default=None, type=str)
@click.pass_context
def cli(ctx: click.Context, verbosity: int, debug: bool, log_dir: str) -> None:
    '''The root cli group'''
    ctx.ensure_object(dict)

    logger = Logger(verbosity, debug, log_dir)
    ctx.obj['logger'] = logger


@cli.group()
@click.option(
    '-o', '--output-dir', show_default=True, default=OUTPUT_DATA_DIR, type=str
)
@click.option('-f', '--output', show_default=True, default='out.csv', type=str)
@click.option(
    '-i', '--input-dir', show_default=True, default=WORKING_DATA_DIR, type=str
)
@click.option('-m', '--meth', show_default=True, default='M.csv', type=str)
@click.option('-g', '--gene', show_default=True, default='G.csv', type=str)
@click.option('-c', '--covar', show_default=True, default='C.csv', type=str)
@click.pass_context
def run(
    ctx: click.Context,
    output_dir: str,
    output: str,
    input_dir: str,
    meth: str,
    gene: str,
    covar: str,
) -> None:
    '''
    Base group for running algorithms.

    Sets up the running environment given the input and output
    directories, methylation, gene expression, and covariate file names,
    and output file name. Choose an algorithm and add arguments.
    '''
    ctx.obj['output_dir'] = output_dir
    ctx.obj['output'] = output
    ctx.obj['input_dir'] = input_dir
    ctx.obj['meth'] = meth
    ctx.obj['gene'] = gene
    ctx.obj['covar'] = covar


@cli.command()
@click.option(
    '-o', '--output-dir', show_default=True, default=WORKING_DATA_DIR, type=str
)
@click.option('-s', '--samples', required=True, type=int)
@click.option('-m', '--meth-rows', required=True, type=int)
@click.option('-g', '--gene-rows', required=True, type=int)
@click.pass_context
def dummy(
    ctx: click.Context,
    output_dir: str,
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
    save_dataframes(dataframes, output_dir, **logger)
    logger.save()


@run.command()
@click.pass_context
@click.option('-c', '--chunks', show_default=True, default=0, type=int)
@click.option('-s', '--save-chunks', show_default=True, default=0, type=int)
def corr(ctx: click.Context, chunks: int, save_chunks: int) -> None:
    '''
    Calculate the pearson correlation coefficient.

    Calculate the pearson correlation coefficient with methylation and
    gene expression matrices. Optional compute and save chunking to
    avoid GPU and CPU memory limits.
    '''
    logger: Logger = ctx.obj['logger']

    dataframes = read_dataframes(ctx.obj['input_dir'], **logger)
    M = dataframes[ctx.obj['meth']]
    G = dataframes[ctx.obj['gene']]

    output = None
    if chunks == 0:
        output = pearson_full_tensor(M, G, **logger)
    elif save_chunks == 0:
        output = pearson_chunk_tensor(M, G, chunks, **logger)
    else:
        pearson_chunk_save_tensor(
            M, G, chunks, save_chunks, ctx.obj['output_dir'], **logger
        )
    if output is not None:
        save_dataframes(
            [output], ctx.obj['output_dir'], [ctx.obj['output']], **logger
        )

    logger.save()


def start() -> None:
    cli(obj={})


if __name__ == '__main__':
    start()
