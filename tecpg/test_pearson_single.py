import math
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

from .helper import random_list
from .logger import Logger
from .pearson_single import (
    pearson_corr_basic,
    pearson_corr_tensor,
    scipy_pearsonr_corr,
)

DEFAULT_FUNCTIONS = [
    pearson_corr_basic,
    pearson_corr_tensor,
    scipy_pearsonr_corr,
]


def test(
    base: Optional[float] = None,
    samples: Optional[int] = 50,
    max_size: Optional[int] = 1000000,
    functions: Optional[
        List[Callable[[List[float], List[float]], float]]
    ] = None,
    std_cutoff: float = float('1e-10'),
    time_cutoff: float = 5,
    show_results: bool = False,
    show_time: bool = True,
    input_range: Tuple[float, float] = (-1, 1),
    *,
    logger: Logger = Logger(),
) -> None:
    """
    Tests the speed and accuracy of the input functions over different
    inputs. Each function in functions is given two input lists, with
    random float values in input_range.

    Base, samples, and max_size are used to compute the input list
    sizes. The functions are tested using samples input lists. Their
    sizes grow exponentially with the provided base. Two of these three
    variables must be provided to have enough information to generate
    the input lists.

    The test passes if the maximum standard deviation of the runs is
    below std_cutoff. This indicates that the difference between the
    functions is low enough to pass all of the functions as accurate
    relative to the other functions in functions.

    The show_results boolean determines whether to show the results of
    each function for every pass. Defaults to False

    The show_time boolean determines whether to show the time plot.
    Defaults to true. The time plot is a matplotlib standard plot of the
    time it took each function to compute each input, given the input
    size. Both the x- and y-axes are log scale.
    """
    error = ValueError(
        'Not enough arguments provided to determine the test sizes. Include at'
        ' least two of base, samples, or max_size.'
    )

    if functions is None:
        functions = DEFAULT_FUNCTIONS

    missing = sum([base is None, samples is None, max_size is None])
    if missing > 1:
        raise error
    if missing == 0 or base is None:
        base = max_size ** (1 / samples)
    elif samples is None:
        samples = math.ceil(math.log(max_size, base))

    test_sizes = [math.ceil(base ** n) for n in range(2, samples + 1)]
    test_data = (
        (random_list(length, *input_range), random_list(length, *input_range))
        for length in test_sizes
    )

    logger.start_counter('info', 'Testing with functions = {0}', functions)
    stds = []
    times = []
    for x, y in test_data:
        results = []
        pass_times = []
        for f in functions:
            start = time.perf_counter()
            result = f(x, y, **logger)
            end = time.perf_counter()
            pass_times.append(end - start)
            results.append(result)

        times.append(pass_times)
        std = np.std(results)
        stds.append(std)
        logger.count('Pass {i}/{0}, std = {0}', samples, std)
        if show_results:
            logger.info('results = {0}', results)
        if sum(pass_times) > time_cutoff:
            logger.warning(
                'Time limit of {0} seconds reached. Last pass took {1}'
                ' seconds. Aborting after {2} passes.',
                time_cutoff,
                sum(pass_times),
                logger.current_count,
            )
            test_sizes = test_sizes[: logger.current_count]
            break

    max_std = max(stds)
    if max_std < std_cutoff:
        logger.info(f'Test passed! {max_std=} < {std_cutoff=}')
    else:
        logger.error(f'Test failed! {max_std=} > {std_cutoff=}')

    if not show_time:
        return

    labels = [f.__name__ for f in functions]

    logger.info('At max input size of {0}:', test_sizes[-1])
    for label, time in zip(labels, times[-1]):
        logger.info('    {0} took {1} seconds', label, time)

    _, ax = plt.subplots()
    for index, time in enumerate(zip(*times)):
        label = functions[index].__name__
        ax.plot(test_sizes, time, label=label)
    ax.set_xlabel('Input Size')
    ax.set_ylabel('Time')
    ax.set_title('Time of functions given input size')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.legend()

    logger.info('Press enter to show time graph... ')

    plt.show()
