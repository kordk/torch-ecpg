import torch
from .logger import Logger

data = {
    'root_path': '.',
    'input_dir': 'data',
    'output_dir': 'output',
    'meth_file': 'M.csv',
    'gene_file': 'G.csv',
    'covar_file': 'C.csv',
    'output': 'out.csv',
    'log_dir': 'logs',
}

USING_GPU = torch.cuda.is_available()
device = torch.device('cuda' if USING_GPU else 'cpu')


def using_gpu(*, logger: Logger = Logger()) -> None:
    if USING_GPU:
        logger.info('This device supports CUDA. Torch will run on the GPU.')
    else:
        logger.info(
            'This device does not support CUDA. Torch will run on the CPU.'
        )
