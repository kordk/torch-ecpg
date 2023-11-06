import itertools
import math
import os
from typing import Any, List, Optional

import click
import pandas
import psutil
import torch

from .config import (
    DEFAULT_CIS_DOWNSTREAM,
    DEFAULT_CIS_UPSTREAM,
    DEFAULT_CIS_WINDOW_BASE,
    DEFAULT_DISTAL_DOWNSTREAM,
    DEFAULT_DISTAL_UPSTREAM,
    DEFAULT_DISTAL_WINDOW_BASE,
    DEFAULT_FLOAT_FORMAT,
    DTYPE,
    data,
    using_gpu,
)
from .gtp import save_gtp_data
from .helper import default_region_parameter, initialize_dir
from .import_data import read_dataframes, save_dataframes
from .logger import Logger
from .pearson_full import (
    pearson_chunk_save_tensor,
    pearson_chunk_tensor,
    pearson_full_tensor,
)
from .regression_full import regression_full
from .regression_single import regression_single
from .test_data import generate_data
from .tool import (
    estimate_constants_bytes,
    estimate_loci_per_chunk_e_peak,
    estimate_loci_per_chunk_results_peak,
)


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
    '-a',
    '--annot-dir',
    show_default=True,
    default=data['annot_dir'],
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
    '-M',
    '--meth-annot',
    show_default=True,
    default=data['meth_annot'],
    type=click.Path(dir_okay=False),
)
@click.option(
    '-G',
    '--gene-annot',
    show_default=True,
    default=data['gene_annot'],
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
@click.option(
    '-F',
    '--float-format',
    show_default=True,
    default=DEFAULT_FLOAT_FORMAT,
    type=str,
)
@click.pass_context
def cli(
    ctx: Optional[click.Context] = None,
    root_path: Optional[str] = None,
    input_dir: Optional[str] = None,
    annot_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    meth_file: Optional[str] = None,
    gene_file: Optional[str] = None,
    covar_file: Optional[str] = None,
    meth_annot: Optional[str] = None,
    gene_annot: Optional[str] = None,
    output_file: Optional[str] = None,
    cpu_threads: Optional[int] = None,
    verbosity: Optional[int] = None,
    debug: Optional[bool] = None,
    log_dir: Optional[str] = None,
    no_log_file: Optional[bool] = None,
    float_format: Optional[str] = None,
    obj: Optional[dict] = None,
) -> None:
    """The root cli group"""
    assert obj is None
    ctx.ensure_object(dict)

    data['root_path'] = click.format_filename(root_path)
    data['input_dir'] = click.format_filename(input_dir)
    data['annot_dir'] = click.format_filename(annot_dir)
    data['output_dir'] = click.format_filename(output_dir)
    data['meth_file'] = click.format_filename(meth_file)
    data['gene_file'] = click.format_filename(gene_file)
    data['covar_file'] = click.format_filename(covar_file)
    data['meth_annot'] = click.format_filename(meth_annot)
    data['gene_annot'] = click.format_filename(gene_annot)
    data['output_file'] = click.format_filename(output_file)
    data['log_dir'] = click.format_filename(log_dir)

    log_path = None if no_log_file else os.path.join(root_path, log_dir)
    logger = Logger(verbosity, debug, log_path)
    using_gpu(**logger)
    if cpu_threads:
        torch.set_num_threads(cpu_threads)
        logger.carry_data['use_cpu'] = True
    logger.carry_data['float_format'] = (
        DEFAULT_FLOAT_FORMAT if float_format is None else float_format
    )
    ctx.obj['logger'] = logger


@cli.group()
def run() -> None:
    """Base group for running algorithms."""


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
    """
    Calculate the pearson correlation coefficient.

    Calculate the pearson correlation coefficient with methylation and
    gene expression matrices. Optional compute and save chunking to
    avoid GPU and CPU memory limits.
    """
    logger: Logger = ctx.obj['logger']

    data_path = os.path.join(data['root_path'], data['input_dir'])
    dataframes = read_dataframes(data_path, **logger)
    M = dataframes[data['meth_file']]
    G = dataframes[data['gene_file']]

    output_path = os.path.join(data['root_path'], data['output_dir'])
    output = None
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
@click.option('-g', '--gene-loci-per-chunk', show_default=True, type=int)
@click.option('-m', '--meth-loci-per-chunk', show_default=True, type=int)
@click.option('-p', '--p-thresh', show_default=True, type=float)
@click.option(
    '--all', 'region', show_default=True, flag_value='all', default=True
)
@click.option('--cis', 'region', show_default=True, flag_value='cis')
@click.option('--distal', 'region', show_default=True, flag_value='distal')
@click.option('--trans', 'region', show_default=True, flag_value='trans')
@click.option('-w', '--window-base', show_default=True, type=int)
@click.option('-d', '--downstream', show_default=True, type=int)
@click.option('-u', '--upstream', show_default=True, type=int)
@click.option(
    '--full-output',
    '-f',
    is_flag=True,
    show_default=True,
    default=False,
    type=bool,
)
@click.option(
    '--p-only', '-P', is_flag=True, show_default=True, default=False, type=bool
)
@click.pass_context
def mlr(
    ctx: click.Context,
    gene_loci_per_chunk: Optional[int],
    meth_loci_per_chunk: Optional[int],
    p_thresh: Optional[float],
    region: str,
    window_base: Optional[int],
    downstream: Optional[int],
    upstream: Optional[int],
    full_output: bool,
    p_only: bool,
) -> None:
    logger: Logger = ctx.obj['logger']

    chunking = (
        gene_loci_per_chunk is not None or meth_loci_per_chunk is not None
    )
    data_path = os.path.join(data['root_path'], data['input_dir'])
    output_path = os.path.join(data['root_path'], data['output_dir'])

    dataframes = read_dataframes(data_path, **logger)
    M = dataframes[data['meth_file']]
    G = dataframes[data['gene_file']]
    C = dataframes[data['covar_file']]

    if region != 'all':
        annot_path = os.path.join(data['root_path'], data['annot_dir'])
        M_annot = pandas.read_csv(
            os.path.join(annot_path, data['meth_annot']), sep=None
        ).set_index('name')
        G_annot = pandas.read_csv(
            os.path.join(annot_path, data['gene_annot']), sep=None
        ).set_index('name')

    window_base = default_region_parameter(
        'window_base',
        window_base,
        region,
        {'cis': DEFAULT_CIS_WINDOW_BASE, 'distal': DEFAULT_DISTAL_WINDOW_BASE},
    )
    downstream = default_region_parameter(
        'downstream',
        downstream,
        region,
        {'cis': DEFAULT_CIS_DOWNSTREAM, 'distal': DEFAULT_DISTAL_DOWNSTREAM},
    )
    upstream = default_region_parameter(
        'upstream',
        upstream,
        region,
        {'cis': DEFAULT_CIS_UPSTREAM, 'distal': DEFAULT_DISTAL_UPSTREAM},
    )

    methylation_only = not full_output

    args = [M, G, C]
    args.extend((None, None) if region == 'all' else (M_annot, G_annot))
    args.extend(
        [
            region,
            window_base,
            downstream,
            upstream,
            gene_loci_per_chunk,
            meth_loci_per_chunk,
            p_thresh,
        ]
    )
    args.append(None if not chunking else output_path)  # output_dir
    args.extend([methylation_only, p_only])

    output = regression_full(*args, **logger)
    if not chunking:
        save_dataframes([output], output_path, [data['output_file']], **logger)


@run.command()
@click.option(
    '-r', '--regressions-per-chunk', show_default=True, default=0, type=int
)
@click.option('-p', '--p-thresh', show_default=True, type=float)
@click.option(
    '--all', 'region', show_default=True, flag_value='all', default=True
)
@click.option('--cis', 'region', show_default=True, flag_value='cis')
@click.option('--distal', 'region', show_default=True, flag_value='distal')
@click.option('--trans', 'region', show_default=True, flag_value='trans')
@click.option('-w', '--window', show_default=True, type=int)
@click.option(
    '--full-output',
    '-f',
    is_flag=True,
    show_default=True,
    default=False,
    type=bool,
)
@click.option(
    '--no-est', is_flag=True, show_default=True, default=False, type=bool
)
@click.option(
    '--no-err', is_flag=True, show_default=True, default=False, type=bool
)
@click.option(
    '--no-t', is_flag=True, show_default=True, default=False, type=bool
)
@click.option(
    '--no-p', is_flag=True, show_default=True, default=False, type=bool
)
@click.pass_context
def mlr_single(
    ctx: click.Context,
    regressions_per_chunk: int,
    p_thresh: Optional[float],
    region: str,
    window: Optional[int],
    full_output: bool,
    no_est: bool,
    no_err: bool,
    no_t: bool,
    no_p: bool,
) -> None:
    """
    Calculates the multiple linear regression.

    Calculate the multiple linear regression with methylation, gene
    expression, and covariate matrices. Optional chunking to avoid
    memory limits.
    """
    logger: Logger = ctx.obj['logger']

    data_path = os.path.join(data['root_path'], data['input_dir'])
    output_path = os.path.join(data['root_path'], data['output_dir'])

    dataframes = read_dataframes(data_path, **logger)
    M = dataframes[data['meth_file']]
    G = dataframes[data['gene_file']]
    C = dataframes[data['covar_file']]
    include = (not no_est, not no_err, not no_t, not no_p)

    if region != 'all':
        annot_path = os.path.join(data['root_path'], data['annot_dir'])
        M_annot = (
            pandas.read_csv(
                os.path.join(annot_path, data['meth_annot']), sep=None
            )
            .set_index('name')
            .drop(['chromEnd', 'score', 'strand'])
        )
        G_annot = (
            pandas.read_csv(
                os.path.join(annot_path, data['gene_annot']), sep=None
            )
            .set_index('name')
            .drop(['chromEnd', 'score', 'strand'])
        )

    methylation_only = not full_output
    if region in ['cis', 'distal'] and window is None:
        logger.info('No region window provided. Resorting to default.')
        if region == 'cis':
            logger.info(
                'Using default window for cis of {0} bases',
                DEFAULT_CIS_UPSTREAM,
            )
            window = DEFAULT_CIS_UPSTREAM
        if region == 'distal':
            logger.info(
                'Using default window for distal of {0} bases',
                DEFAULT_DISTAL_UPSTREAM,
            )
            window = DEFAULT_DISTAL_UPSTREAM

    args = [M, G, C, include, regressions_per_chunk, p_thresh, region, window]
    args.extend((None, None) if region == 'all' else (M_annot, G_annot))
    args.extend((methylation_only, 1))
    if regressions_per_chunk:
        args.append(output_path)
    output = regression_single(*args, **logger)
    if not regressions_per_chunk:
        save_dataframes([output], output_path, [data['output_file']], **logger)


@cli.group(name='data')
def _data() -> None:
    """Base group for data management."""


@_data.command()
@click.option('-s', '--samples', type=int, prompt=True)
@click.option('-m', '--meth-rows', type=int, prompt=True)
@click.option('-g', '--gene-rows', type=int, prompt=True)
@click.option(
    '-n',
    '--no-annotation',
    is_flag=True,
    show_default=True,
    default=False,
    type=bool,
)
@click.pass_context
def dummy(
    ctx: click.Context,
    samples: int,
    meth_rows: int,
    gene_rows: int,
    no_annotation: bool,
) -> None:
    """
    Generates dummy data.

    Generates dummy data in the output directory with a given size with
    file names M.csv, G.csv, and C.csv.
    """
    logger: Logger = ctx.obj['logger']
    annotation = not no_annotation

    dataframes = generate_data(
        samples, meth_rows, gene_rows, annotation=annotation
    )
    file_names = [data['meth_file'], data['gene_file'], data['covar_file']]
    data_path = os.path.join(data['root_path'], data['input_dir'])
    save_dataframes(dataframes[:3], data_path, file_names, **logger)
    if annotation:
        file_names = [data['meth_annot'], data['gene_annot']]
        data_path = os.path.join(data['root_path'], data['annot_dir'])
        save_dataframes(
            dataframes[3:],
            data_path,
            file_names,
            sep='\t',
            index=False,
            **logger,
        )

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
@click.option(
    '--full-covar',
    is_flag=True,
    show_default=True,
    default=False,
    type=bool,
)
@click.pass_context
def gtp(ctx: click.Context, gtp_dir: Any, full_covar: bool) -> None:
    """
    Downloads and extracts GTP data.

    Downloads the methylation, gene expression, and covariate data from
    Grady Trauma Project study. Stores the raw data in gtp-dir. The raw
    data is extracted and processes before being saved in the data
    directory.
    """
    logger: Logger = ctx.obj['logger']

    gtp_path = os.path.join(data['root_path'], gtp_dir)
    data_path = os.path.join(data['root_path'], data['input_dir'])
    file_names = [data['meth_file'], data['gene_file'], data['covar_file']]
    simplify_covar = not full_covar
    save_gtp_data(
        gtp_path,
        data_path,
        file_names,
        simplify_covar=simplify_covar,
        **logger,
    )

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
    """
    Creates and initializes directory.

    Creates root_dir in the root_path. Creates input_dir and output_dir
    in this new directory. Changes directory too this new directory.
    """
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


@cli.command()
@click.option('-t', '--target-bytes', type=int)
@click.option('-s', '--samples', type=int)
@click.option('-m', '--mt-count', type=int)
@click.option('-g', '--gt-count', type=int)
@click.option('-c', '--covar-count', type=int)
@click.option('-M', '--meth-loci-per-chunk', type=int)
@click.option('-F', '--filtration', type=float)
@click.option('-f', '--full-output', type=bool)
@click.option('-P', '--p-only', type=bool)
@click.option('-p', '--p-filtration', type=bool)
@click.option('-r', '--region-filtration', type=bool)
@click.option('-C', '--cpu', show_default=True, default=False, type=bool)
@click.pass_context
def chunks(
    ctx: click.Context,
    target_bytes: Optional[int],
    samples: Optional[int],
    mt_count: Optional[int],
    gt_count: Optional[int],
    covar_count: Optional[int],
    meth_loci_per_chunk: Optional[int],
    filtration: Optional[float],
    full_output: Optional[bool],
    p_only: Optional[bool],
    p_filtration: Optional[bool],
    region_filtration: Optional[bool],
    cpu: bool,
) -> None:
    '''
    Estimates --gene-loci-per-chunk.

    Estimate optimal --gene-loci-per-chunk to maximize parallelization within
    memory limits given certain variables about the input and the
    system.
    '''
    logger: Logger = ctx.obj['logger']

    if filtration is None and (
        None in (p_filtration, region_filtration)
        or True in (p_filtration, region_filtration)
    ):
        error = (
            'Define --filtration, a float from 0 to 1 for the proportion of'
            ' rows left after region or p-value filtration, if'
            ' region_filtration or p_filtration is included'
        )
        logger.error(error)
        raise ValueError(error)
    if filtration is None:
        filtration = 1
    datum_bytes = torch.ones(1, dtype=DTYPE).element_size()

    if target_bytes is None:
        if cpu or not torch.cuda.is_available():
            target_bytes = psutil.virtual_memory().total * 0.8
            logger.info(
                (
                    'Target memory not supplied. Inferred target of {0} MB of'
                    ' CPU memory (80% of detected)'
                ),
                target_bytes / 1_000_000,
            )
        else:
            target_bytes = torch.cuda.mem_get_info()[0] * 0.8
            logger.info(
                (
                    'Target memory not supplied. Inferred target of {0} MB of'
                    ' CUDA memory (80% of detected)'
                ),
                target_bytes / 1_000_000,
            )
    if None in (samples, mt_count, gt_count, covar_count):
        logger.info(
            'Data size not complete. Inferring from data in working directory.'
        )
        data_path = os.path.join(data['root_path'], data['input_dir'])
        dataframes = read_dataframes(data_path, **logger)
        M = dataframes[data['meth_file']]
        G = dataframes[data['gene_file']]
        C = dataframes[data['covar_file']]
        if samples is None:
            samples = len(C)
            logger.info('Samples not provided. Inferred {0}.', samples)
        if mt_count is None:
            mt_count = len(M)
            logger.info(
                'Methylation loci count not provided. Inferred {0}.', mt_count
            )
        if gt_count is None:
            gt_count = len(G)
            logger.info(
                'Gene expression loci count not provided. Inferred {0}.',
                gt_count,
            )
        if covar_count is None:
            covar_count = len(C.columns)
            logger.info(
                'Covariate count not provided. Inferred {0}.', covar_count
            )

    if meth_loci_per_chunk:
        mt_count = meth_loci_per_chunk

    logger.info(
        'Estimated loci per chunk for target peak memory usage of {0} bytes:',
        target_bytes,
    )
    if region_filtration is not True:
        constants_bytes = estimate_constants_bytes(
            samples, mt_count, gt_count, covar_count, datum_bytes, False
        )
        logger.info(
            '{0} bytes for constants (without region filtration)',
            constants_bytes,
        )
    if region_filtration is not False:
        constants_bytes = estimate_constants_bytes(
            samples, mt_count, gt_count, covar_count, datum_bytes, True
        )
        logger.info(
            '{0} bytes for constants (with region filtration)', constants_bytes
        )
    logger.info('Full output, p only, p filtration, region filtration')
    for (
        full_output_,
        p_only_,
        p_filtration_,
        region_filtration_,
    ) in itertools.product((False, True), repeat=4):
        if (
            (full_output is not None and full_output != full_output_)
            or (p_only is not None and p_only != p_only_)
            or (p_filtration is not None and p_filtration != p_filtration_)
            or (
                region_filtration is not None
                and region_filtration != region_filtration_
            )
        ):
            continue
        estimate_e = estimate_loci_per_chunk_e_peak(
            target_bytes,
            samples,
            mt_count,
            gt_count,
            covar_count,
            datum_bytes,
            filtration,
            full_output_,
            p_only_,
            p_filtration_,
            region_filtration_,
        )
        estimate_results = estimate_loci_per_chunk_results_peak(
            target_bytes,
            samples,
            mt_count,
            gt_count,
            covar_count,
            datum_bytes,
            filtration,
            full_output_,
            p_only_,
            region_filtration_,
        )

        if estimate_e < estimate_results:
            estimate = estimate_e
            peak = 'Peak memory after scalars and E'
        else:
            estimate = estimate_results
            peak = 'Peak memory after results concatenation'

        if estimate >= gt_count:
            estimate = 'No chunking needed'
        elif estimate < 1:
            estimate = 'Not possible'
        else:
            estimate = f'{math.floor(estimate)} loci per chunk'

        logger.info(
            '{0}, {1}, {2}, {3}: {4}, {5}',
            full_output_,
            p_only_,
            p_filtration_,
            region_filtration_,
            estimate,
            peak,
        )


def start() -> None:
    cli(obj={})
