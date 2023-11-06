import torch

from .logger import Logger

data = {
    'root_path': '.',
    'input_dir': 'data',
    'annot_dir': 'annot',
    'output_dir': 'output',
    'meth_file': 'M.csv',
    'gene_file': 'G.csv',
    'covar_file': 'C.csv',
    'meth_annot': 'M.bed6',
    'gene_annot': 'G.bed6',
    'output': 'out.csv',
    'log_dir': 'logs',
}

DEFAULT_CIS_DOWNSTREAM = 3_000  # 3 Kb
DEFAULT_CIS_UPSTREAM = 50_000  # 50 Kb
DEFAULT_CIS_WINDOW_BASE = 0  # No window offset
DEFAULT_DISTAL_DOWNSTREAM = 0  # No downstream search
DEFAULT_DISTAL_UPSTREAM = 500_000_000  # To the end of the chromosome
DEFAULT_DISTAL_WINDOW_BASE = 50_000  # 50 Kb

DTYPE = torch.float32
DEFAULT_FLOAT_FORMAT = '%.16f'


def get_device(*, logger: Logger = Logger()) -> torch.device:
    use_cpu = logger.carry_data.get('use_cpu', None)
    if use_cpu is None:
        logger.info('Use CPU not supplied. Checking if CUDA is available.')
        use_cpu = not torch.cuda.is_available()
    else:
        logger.info('Value for use_cpu supplied.')
    if use_cpu:
        logger.info(f'Using CPU with {torch.get_num_threads()} threads')
        device = torch.device('cpu')
    else:
        logger.info('Using CUDA')
        device = torch.device('cuda')
    return device


def using_gpu(*, logger: Logger = Logger()) -> None:
    if torch.cuda.is_available():
        logger.info('CUDA GPU detected. This device supports CUDA.')
    else:
        logger.info('CUDA GPU detected. This device does not support CUDA.')
