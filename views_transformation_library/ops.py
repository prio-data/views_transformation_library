""" mathemetical operations

"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import warnings

def ln(tensor):
    """
    ln

    Returns natural log of s+1

    Arguments:
        None

    """

    return np.log1p(tensor)
