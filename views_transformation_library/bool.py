""" Boolean transforms

"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import warnings

def greater_or_equal(tensor, value: float):
    """
    greater_or_equal

    Detects where input series is greater than or equal to a threshold value

    Returns 1 if s >= value, else 0

    Arguments:
         value: float specifying threshold

    """

    return np.where(tensor>=value,1,0)

def smaller_or_equal(tensor, value: float):
    """
    smaller_or_equal

    Detects where input series is smaller than or equal to a threshold value

    Returns 1 if s <= value, else 0

    Arguments:
         value: float specifying threshold

    """

    return np.where(tensor<=value,1,0)

def in_range(tensor, low: float, high: float):
    """
    greater_or_equal

    Detects where input series is greater than or equal to a threshold value

    Returns 1 if s >= value, else 0

    Arguments:
         value: float specifying threshold

    """

    return np.where(np.logical_and(tensor>=low, tensor<=high),1,0)
