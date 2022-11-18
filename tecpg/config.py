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

CIS_WINDOW = 1_000_000  # 1 Mb
DISTAL_WINDOW = 50_000  # 50 Kb


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
