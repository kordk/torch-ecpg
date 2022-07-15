from typing import List
import numpy as np


def random_list(length, minimum, maximum) -> List[float]:
    return list(np.random.rand(length) * (maximum - minimum) + minimum)
