""" unit transforms for rescaling input

"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import warnings
from .utilities import dne_wrapper


@dne_wrapper
def mean(tensor_container):
    """
    mean

    Computes the arithmetic mean over time for each spatial unit

    Arguments:
        None

    """

    dne = tensor_container.dne
    missing = tensor_container.missing

    tensor_container.tensor[tensor_container.tensor == dne] = missing

    with warnings.catch_warnings(action="ignore"):

        for ifeature in range(tensor_container.tensor.shape[2]):
            time_mean = np.nanmean(tensor_container.tensor[:, :, ifeature].astype(np.float64), axis=0)

            for ispace in range(tensor_container.tensor.shape[1]):
                tensor_container.tensor[:, ispace, ifeature] = time_mean[ispace]

    return tensor_container


@dne_wrapper
def demean(tensor_container):
    """
    demean

    Computes difference between value and mean of input, grouped by spatial unit

    Returns s - mean_group(s)

    Arguments:
        None

    """

    dne = tensor_container.dne
    missing = tensor_container.missing

    tensor_container.tensor[tensor_container.tensor == dne] = missing

    with warnings.catch_warnings(action="ignore"):

        for ifeature in range(tensor_container.tensor.shape[2]):
            time_mean = np.nanmean(tensor_container.tensor[:, :, ifeature].astype(np.float64), axis=0)

            for ispace in range(tensor_container.tensor.shape[1]):
                tensor_container.tensor[:, ispace, ifeature] = (tensor_container.tensor[:, ispace, ifeature]
                                                                .astype(np.float64) - time_mean[ispace])

    return tensor_container


@dne_wrapper
def rollmax(tensor_container, window: int):
    """
    rollmax

    Computes rolling maximum over a specified time window

    Arguments:
         window: integer size of moving time window over which to compute maximum

    """

    dne = tensor_container.dne
    missing = tensor_container.missing

    tensor_container.tensor[tensor_container.tensor == dne] = missing

    if window < 1:
        raise RuntimeError(f"Time below 1 passed to moving sum: {window} \n")

    with warnings.catch_warnings(action="ignore"):

        tensor_container.tensor[window - 1:, :, :] = np.nanmax(sliding_window_view(
                                                     tensor_container.tensor, window, 0), axis=3)

        stub = np.zeros_like(tensor_container.tensor[:window - 1, :, :])

        for itime in range(window - 1):
            stub[itime, :, :] = np.nanmax(tensor_container.tensor[:itime + 1, :, :], axis=0)

    tensor_container.tensor[:window - 1, :, :] = stub

    return tensor_container
