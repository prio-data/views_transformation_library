""" unit transforms for rescaling input

"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import warnings

def mean(tensor):
    """
    mean

    Computes the arithmetic mean over time for each spatial unit

    Arguments:
        None

    """

    with warnings.catch_warnings(action="ignore"):

        for ifeature in range(tensor.shape[2]):
            time_mean = np.nanmean(tensor[:,:,ifeature],axis=0)

            for ispace in range(tensor.shape[1]):
                tensor[:,ispace,ifeature] = time_mean[ispace]

    return tensor

def demean(tensor):
    """
    demean

    Computes difference between value and mean of input, grouped by spatial unit

    Returns s - mean_group(s)

    Arguments:
        None

    """
    with warnings.catch_warnings(action="ignore"):

        for ifeature in range(tensor.shape[2]):
            time_mean = np.nanmean(tensor[:, :, ifeature], axis=0)

            for ispace in range(tensor.shape[1]):
                tensor[:, ispace, ifeature] -= time_mean[ispace]

    return tensor

def rollmax(tensor, window: int):
    """
    rollmax

    Computes rolling maximum over a specified time window

    Arguments:
         window: integer size of moving time window over which to compute maximum

    """

    if window < 1:
        raise RuntimeError(f"Time below 1 passed to moving sum: {window} \n")

    with warnings.catch_warnings(action="ignore"):

        tensor[window - 1:, :, :] = np.nanmax(sliding_window_view(tensor, window, 0), axis=3)

        stub = np.zeros_like(tensor[:window - 1, :, :])

        for itime in range(window - 1):
            stub[itime, :, :] = np.nanmax(tensor[:itime + 1, :, :], axis=0)

    tensor[:window - 1, :, :] = stub

    return tensor
