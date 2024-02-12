""" Missing data filling functionality 

Transformations for handling missing values, such as simple replacements, and
more advanced extrapolations.

"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import warnings
from views_transformation_library import utilities
from utilities import dne_wrapper

@dne_wrapper
def replace_na(tensor_container, replacement = 0.):
    """
    replace_na

    Replaces NaNs in the input with the specified value (which defaults to zero)

    Arguments:
        replacement: quantity which will replace Nan (defaults to zero)

    """

    missing = tensor_container.missing

    tensor_container.tensor = np.where(tensor_container.tensor==missing,
                                       replacement,tensor_container.tensor)

    return tensor_container


@dne_wrapper
def fill(tensor_container, limit_direction='both', limit_area=None):
    """
    fill

    Perform forward and/or backward filling by spatial unit

    Args:
        limit_direction: 'forward', 'backward', 'both': Direction in which to fill. 'forward' propagates most recent
        valid value forward. 'backward' propagates oldest valid value backwards. 'both' performs a forward propagation,
        followed by a backward propagation
        limit_area: None, 'inside', 'outside': if 'inside', NaNs will only be filled if bracketed by valid values.
        If 'outside', NaNs are only filled outside valid values. If None, no restrictions are applied.

    """
    def mask_outside_forward(tensor, missing):
#        mask = np.isnan(tensor)
        mask = np.where(tensor == missing, True, False)

        shape1 = tensor.shape[1]

        mask_min = shape1 - 1 - np.flip(mask, axis=1).argmin(axis=1)

        mask_indices = np.tile(mask_min, (shape1, 1)).T - np.indices(mask.shape)[1]
#        nan_mask = np.where(mask_indices < 0, np.nan, 0)
        nan_mask = np.where(mask_indices < 0, missing, 0)
#        mask = np.isnan(nan_mask)
        mask = np.where(nan_mask == missing, True, False)

        return mask

    def outside_forward(tensor, missing):
        tensor = tensor.T
        shape1 = tensor.shape[1]

        mask = mask_outside_forward(tensor, missing)

        idx = np.where(~mask, np.arange(shape1), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        tensor = tensor[np.arange(idx.shape[0])[:, None], idx]

        return tensor.T

    def outside_backward(tensor, missing):
        tensor = tensor.T
        tensor = np.flip(tensor,axis=1)
        shape1 = tensor.shape[1]

        mask = mask_outside_forward(tensor, missing)

        idx = np.where(~mask, np.arange(shape1), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        tensor = tensor[np.arange(idx.shape[0])[:, None], idx]

        tensor = np.flip(tensor,axis=1)

        return tensor.T

    def outside_both(tensor, missing):
        return outside_backward(outside_forward(tensor, missing), missing)

    def inside_forward(tensor, missing):
        tensor = tensor.T
        shape1 = tensor.shape[1]

#        mask = np.isnan(tensor)
        mask = np.where(tensor == missing, True, False)

        outside_forward_mask = mask_outside_forward(tensor, missing)
        outside_backward_mask = np.flip(mask_outside_forward(np.flip(tensor), missing))

        outside_mask = np.logical_or(outside_forward_mask, outside_backward_mask)

        mask = np.logical_xor(mask,outside_mask)

        idx = np.where(~mask, np.arange(shape1), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        tensor = tensor[np.arange(idx.shape[0])[:, None], idx]

        return tensor.T

    def inside_backward(tensor, missing):
        tensor = tensor.T
        tensor = np.flip(tensor,axis=1)
        shape1 = tensor.shape[1]

        outside_forward_mask = mask_outside_forward(tensor, missing)
        outside_backward_mask = np.flip(mask_outside_forward(np.flip(tensor), missing))

        mask = np.logical_or(outside_forward_mask, outside_backward_mask)

        idx = np.where(~mask, np.arange(shape1), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        tensor = tensor[np.arange(idx.shape[0])[:, None], idx]

        tensor = np.flip(tensor,axis=1)

        return tensor.T

    def both_both(tensor, missing):
        return inside_forward(outside_both(tensor, missing), missing)

    def both_forward(tensor, missing):
        return inside_forward(outside_forward(tensor, missing), missing)

    def both_backward(tensor, missing):
        return inside_forward(outside_backward(tensor, missing), missing)


    def select_filler(limit_direction, limit_area):
        match (limit_direction, limit_area):
            case ('both', None):
                return both_both
            case ('both', 'inside'):
                return inside_forward
            case ('both', 'outside'):
                return outside_both
            case ('forward', None):
                return both_forward
            case ('forward', 'inside'):
                return inside_forward
            case ('forward', 'outside'):
                return outside_forward
            case ('backward', None):
                return both_backward
            case ('backward', 'inside'):
                return inside_backward
            case ('backward', 'outside'):
                return outside_backward
            case _:
                raise RuntimeError(f'Unrecognised limit dir/area: {limit_direction, limit_area}')

    tensor_fill = tensor_container.tensor.copy()
    missing = tensor_container.missing

    filler=select_filler(limit_direction, limit_area)

    for ifeature in range(tensor_fill.shape[2]):
        tensor_fill[:,:,ifeature] = filler(tensor_fill[:,:,ifeature], missing)

    tensor_container.tensor = tensor_fill

    return tensor_container
