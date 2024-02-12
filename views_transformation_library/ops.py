""" mathemetical operations

"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import warnings
from utilities import dne_wrapper

@dne_wrapper
def ln(tensor_container):
    """
    ln

    Returns natural log of s+1

    Arguments:
        None

    """

    tensor_container.tensor = np.log1p(tensor_container.tensor)

    return tensor_container
