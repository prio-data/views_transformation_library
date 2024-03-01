import numpy as np
from utilities import dne_wrapper


@dne_wrapper
def add_column(tensor_container, dummy):

    if len(tensor_container.tensor.shape[-1]) != 2:
        raise RuntimeError(f'Tensor with {len(tensor_container.tensor.shape[2])} passed to multicolumn add')

    left_feature = tensor_container.tensor[:, :, 0]
    right_feature = tensor_container.tensor[:, :, 1]

    missing = tensor_container.missing
    dne = tensor_container.dne

    tensor_container.tensor[tensor_container.tensor == dne] = missing

    tensor_container.tensor[:, :, 0] = left_feature + right_feature

    tensor_container.tensor = tensor_container.tensor[:, :, 0].reshape(tensor_container.tensor.shape[0],
                                                                       tensor_container.tensor.shape[1],
                                                                       1)

    return tensor_container


@dne_wrapper
def subtract_column(tensor_container, dummy):

    if len(tensor_container.tensor.shape[-1]) != 2:
        raise RuntimeError(f'Tensor with {len(tensor_container.tensor.shape[2])} passed to multicolumn subtract')

    left_feature = tensor_container.tensor[:, :, 0]
    right_feature = tensor_container.tensor[:, :, 1]

    missing = tensor_container.missing
    dne = tensor_container.dne

    tensor_container.tensor[tensor_container.tensor == dne] = missing

    tensor_container.tensor[:, :, 0] = left_feature - right_feature

    tensor_container.tensor = tensor_container.tensor[:, :, 0].reshape(tensor_container.tensor.shape[0],
                                                                       tensor_container.tensor.shape[1],
                                                                       1)

    return tensor_container


@dne_wrapper
def multiply_column(tensor_container, dummy):

    if len(tensor_container.tensor.shape[-1]) != 2:
        raise RuntimeError(f'Tensor with {len(tensor_container.tensor.shape[2])} passed to multicolumn multiply')

    left_feature = tensor_container.tensor[:, :, 0]
    right_feature = tensor_container.tensor[:, :, 1]

    missing = tensor_container.missing
    dne = tensor_container.dne

    tensor_container.tensor[tensor_container.tensor == dne] = missing

    tensor_container.tensor[:, :, 0] = left_feature * right_feature

    tensor_container.tensor = tensor_container.tensor[:, :, 0].reshape(tensor_container.tensor.shape[0],
                                                                       tensor_container.tensor.shape[1],
                                                                       1)

    return tensor_container


@dne_wrapper
def divide_column(tensor_container, dummy):

    if len(tensor_container.tensor.shape[-1]) != 2:
        raise RuntimeError(f'Tensor with {len(tensor_container.tensor.shape[2])} passed to multicolumn divide')

    left_feature = tensor_container.tensor[:, :, 0]
    right_feature = tensor_container.tensor[:, :, 1]

    missing = tensor_container.missing
    dne = tensor_container.dne

    tensor_container.tensor[tensor_container.tensor == dne] = missing

    tensor_container.tensor[:, :, 0] = left_feature / right_feature

    tensor_container.tensor = tensor_container.tensor[:, :, 0].reshape(tensor_container.tensor.shape[0],
                                                                       tensor_container.tensor.shape[1],
                                                                       1)

    return tensor_container
