from pathlib import Path
import torch
from logger import Logger

PATH = str(Path(__file__).parent.parent.resolve()) + '/'
WORKING_DATA_DIR = PATH + 'working_data/'
RAW_DATA_DIR = PATH + 'raw_data/'

USING_GPU = torch.cuda.is_available()
device = torch.device('cuda' if USING_GPU else 'cpu')


def using_gpu(*, logger: Logger = Logger()) -> None:
    if USING_GPU:
        logger.info('This device supports CUDA. Torch will run on the GPU.')
    else:
        logger.info(
            'This device does not support CUDA. Torch will run on the CPU.'
        )


if __name__ == '__main__':
    using_gpu()
