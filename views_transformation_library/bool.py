""" Boolean transforms

"""

import numpy as np
from .utilities import dne_wrapper


@dne_wrapper
def greater_or_equal(tensor_container, value: float):
    """
    greater_or_equal

    Detects where input series is greater than or equal to a threshold value

    Returns 1 if s >= value, else 0

    Arguments:
        tensor_container: ViewsTensor object
        value: float specifying threshold

    """

    tensor_container.tensor = np.where(tensor_container.tensor >= value, 1, 0)

    return tensor_container


@dne_wrapper
def smaller_or_equal(tensor_container, value: float):
    """
    smaller_or_equal

    Detects where input series is smaller than or equal to a threshold value

    Returns 1 if s <= value, else 0

    Arguments:
        tensor_container: ViewsTensor object
        value: float specifying threshold

    """

    tensor_container.tensor = np.where(tensor_container.tensor <= value, 1, 0)

    return tensor_container


@dne_wrapper
def in_range(tensor_container, low: float, high: float):
    """
    greater_or_equal

    Detects where input series is greater than or equal to a threshold value

    Returns 1 if s >= value, else 0

    Arguments:
        tensor_container: ViewsTensor object
        low: float specifying lower threshold
        high: float specifying higher threshold

    """

    tensor_container.tensor = np.where(np.logical_and(tensor_container.tensor >= low,
                                                      tensor_container.tensor <= high), 1, 0)

    return tensor_container
