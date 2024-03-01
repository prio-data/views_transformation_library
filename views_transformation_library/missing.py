""" Missing data filling functionality 

Transformations for handling missing values, such as simple replacements, and
more advanced extrapolations.

"""

import numpy as np
from .utilities import dne_wrapper


@dne_wrapper
def replace_na(tensor_container, replacement=0.):
    """
    replace_na

    Replaces NaNs in the input with the specified value (which defaults to zero)

    Arguments:
        replacement: quantity which will replace Nan (defaults to zero)

    """

    tensor_container.tensor = np.where(np.isnan(tensor_container.tensor), replacement, tensor_container.tensor)

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
    def mask_outside_forward(tensor, missing, dne):

        missing_filled = np.where(tensor == dne, missing, tensor)

        mask = np.isnan(missing_filled)

        shape1 = tensor.shape[1]

        mask_min = shape1 - 1 - np.flip(mask, axis=1).argmin(axis=1)

        mask_indices = np.tile(mask_min, (shape1, 1)).T - np.indices(mask.shape)[1]

        nan_mask = np.where(mask_indices < 0, missing, 0)
        mask = np.isnan(nan_mask)

        return mask

    def outside_forward(tensor, missing, dne):
        tensor = tensor.T
        shape1 = tensor.shape[1]

        mask = mask_outside_forward(tensor, missing, dne)

        idx = np.where(~mask, np.arange(shape1), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        tensor = tensor[np.arange(idx.shape[0])[:, None], idx]

        return tensor.T

    def outside_backward(tensor, missing, dne):
        tensor = tensor.T
        tensor = np.flip(tensor, axis=1)
        shape1 = tensor.shape[1]

        mask = mask_outside_forward(tensor, missing, dne)

        idx = np.where(~mask, np.arange(shape1), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        tensor = tensor[np.arange(idx.shape[0])[:, None], idx]

        tensor = np.flip(tensor, axis=1)

        return tensor.T

    def outside_both(tensor, missing, dne):
        return outside_backward(outside_forward(tensor, missing, dne), missing, dne)

    def inside_forward(tensor, missing, dne):
        tensor = tensor.T
        shape1 = tensor.shape[1]

        mask = np.isnan(tensor)

        outside_forward_mask = mask_outside_forward(tensor, missing, dne)
        outside_backward_mask = np.flip(mask_outside_forward(np.flip(tensor), missing, dne))

        outside_mask = np.logical_or(outside_forward_mask, outside_backward_mask)

        mask = np.logical_xor(mask,outside_mask)

        idx = np.where(~mask, np.arange(shape1), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        tensor = tensor[np.arange(idx.shape[0])[:, None], idx]

        return tensor.T

    def inside_backward(tensor, missing, dne):
        tensor = tensor.T
        tensor = np.flip(tensor, axis=1)
        shape1 = tensor.shape[1]

        outside_forward_mask = mask_outside_forward(tensor, missing, dne)
        outside_backward_mask = np.flip(mask_outside_forward(np.flip(tensor), missing, dne))

        mask = np.logical_or(outside_forward_mask, outside_backward_mask)

        idx = np.where(~mask, np.arange(shape1), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        tensor = tensor[np.arange(idx.shape[0])[:, None], idx]

        tensor = np.flip(tensor, axis=1)

        return tensor.T

    def both_both(tensor, missing, dne):
        return inside_forward(outside_both(tensor, missing, dne), missing, dne)

    def both_forward(tensor, missing, dne):
        return inside_forward(outside_forward(tensor, missing, dne), missing, dne)

    def both_backward(tensor, missing, dne):
        return inside_forward(outside_backward(tensor, missing, dne), missing, dne)

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
    dne = tensor_container.dne

    filler = select_filler(limit_direction, limit_area)

    for ifeature in range(tensor_fill.shape[-1]):
        tensor_fill[:, :, ifeature] = filler(tensor_fill[:, :, ifeature], missing, dne)

    tensor_container.tensor = tensor_fill

    return tensor_container
