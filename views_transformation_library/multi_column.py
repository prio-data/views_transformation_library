import numpy as np

def add_column(tensor):

    if len(tensor.shape[2])!=2:
        raise RuntimeError(f'Tensor with {len(tensor.shape[2])} passed to multicolumn add')

    return tensor[:,:,0] + tensor[:,:,1]

def subtract_column(tensor):

    if len(tensor.shape[2]) != 2:
        raise RuntimeError(f'Tensor with {len(tensor.shape[2])} passed to multicolumn subtract')


    return tensor[:,:,0] - tensor[:,:,1]

def multiply_column(tensor):

    if len(tensor.shape[2]) != 2:
        raise RuntimeError(f'Tensor with {len(tensor.shape[2])} passed to multicolumn multiply')

    return tensor[:,:,0] * tensor[:,:,1]

def divide_column(tensor):

    if len(tensor.shape[2]) != 2:
        raise RuntimeError(f'Tensor with {len(tensor.shape[2])} passed to multicolumn divide')

    return tensor[:,:,0] / tensor[:,:,1]