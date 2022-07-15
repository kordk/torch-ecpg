from typing import List
import numpy as np


def random_list(length: int, minimum: float, maximum: float) -> List[float]:
    '''
    Returns a list of length, with random float values ranging from
    minimum to maximum. Returns a list of floats.
    '''
    return list(np.random.rand(length) * (maximum - minimum) + minimum)
