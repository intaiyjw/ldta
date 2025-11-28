#!/usr/bin/python3
# -*- coding:utf-8 -*-

# import numpy as np
import torch
from typing import Optional, Union, Tuple, Literal

def is_iter(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

"""
Imagine the matrix multiplication in 3-D way:

step_1: tense the left and right matrix in contrary direction;
step_2: perform the Hadamard element-wise product;
step_3: collapse the tensed dimension by sum up
"""

"""
1. tensor contraction, inner and outer product (einsum)
2. tensor tense (unsqueeze and repeat)
3. split, concatenate
4. index, slice
"""

# inverse function of torch.sum() in shape, but repeat in each new dimension
def tense_array_torch(a: torch.Tensor, direction: int, K: Union[int, tuple, torch.Size]) -> torch.Tensor:
    """
    Expansion of a tensor
    """
    if not is_iter(K):
        # return a.unsqueeze(direction).repeat_interleave(repeats=K, dim=direction)
        # torch.repeat_interleave(), and a_tensor.repeat(), are two kinds of tensor repeat in pytorch
        # torch.squeeze(), and torch.unsqueeze(), manipulate axes
        return torch.repeat_interleave(torch.unsqueeze(a, dim=direction),
                                       repeats=K,
                                       dim=direction)
    else:
        if direction >= 0:
            for repeat_time in K:
                a = a.unsqueeze(direction).repeat_interleave(repeats=repeat_time, dim=direction)
                direction += 1
        else:
            for repeat_time in K[::-1]:
                a = a.unsqueeze(direction).repeat_interleave(repeats=repeat_time, dim=direction)
                direction -= 1
        return a

# use instead: torch.meshgrid() and advanced indexing
def indice_tensor_torch(shape: tuple) -> torch.Tensor:
    """
    Given a shape, create a tensor with shape (shape, len(shape)),
    the deepest dimension represents the index of the position of 
    this element
    """
    if not is_iter(shape):
        shape = (shape, )
    indices = torch.cartesian_prod(*[torch.arange(dim) for dim in shape])
    tensor_indices = indices.reshape(*shape, len(shape))
    
    return tensor_indices

# deprecated!!! Use instead: vectorized_identity_torch
def generalized_identity_torch(shape: Union[int, tuple, torch.Size], num=1) -> torch.Tensor:  
    """
    Given a shape, e.g. (M, K, V), return a tensor A with shape
    (M, K, V, M, K, V), where A[m, k, v, m, k, v] = num

    shape must be a tuple or an integer or a torch.Size
    torch.Size's many behaviours are like tuple
    """
    if not is_iter(shape):
        shape = (shape, )

    diag = torch.zeros(shape + shape)

    # create an iterable containing indices of a tensor:
    # return a 2-dimensional tensor (length of shape > 1), or
    # a 1-dimensional tensor (length of shape = 1)
    indices = torch.cartesian_prod(*[torch.arange(dim) for dim in shape])
    for index in indices:
        # for 1-dimensional tensor, its elements are tensor with shape (),
        # and they are not iterable
        if not is_iter(index):
            index = (index, )
        index = tuple(index)
        diag[index][index] = num

    return diag

def vectorized_identity_torch(shape: Union[int, tuple, torch.Size], num=1) -> torch.Tensor:
    """
    Given a shape (e.g., (M, K, V)), return a tensor A with shape
    (M, K, V, M, K, V), where A[m, k, v, m, k, v] = num.

    Efficiently implemented using advanced indexing.
    """
    if isinstance(shape, int):
        shape = (shape, )

    # Create a zero tensor of shape (M, K, V, M, K, V)
    # diag = torch.zeros(shape + shape, dtype=torch.float32)
    diag = torch.zeros(shape + shape)  # notice datatype

    # Generate indices for the diagonal
    indices = torch.meshgrid(*[torch.arange(s) for s in shape], indexing='ij')
    
    # Use advanced indexing to set diagonal elements to num
    # a special way of slicing?
    diag[indices + indices] = num

    return diag

'''

M, N = 3, 4  # Example dimensions

# Initialize a zero tensor of shape (M, N, M, N)
tensor = torch.zeros(M, N, M, N)

# Set the elements where (m, n, m, n) = 1
for m in range(M):
    for n in range(N):
        tensor[m, n, m, n] = 1



def nested_loops_assign(tensor, number, current=[], first_call=True):
    """
    Given a tensor with shape like (K,K) or (M,K,M,K) etc.,
    assign (m,k,m,k) a certain number
    """
    if first_call:
        full_shape = tensor.shape
        shape = full_shape[:len(full_shape)//2]
    
    if len(current) == len(shape):
        pass
    else:
        for i in range(shape[len(current)]):
            nested_loops_assign(tensor, number, current + [i], first_call=False)

def generalized_identity(shape):
    if not is_iter(shape):
        shape = (shape,)

    tensor_diag = torch.zeros(shape*2)
    current = []
    if len(current) == len(shape):
        tensor_diag[shape*2] = 1
    else:
        for i in range(shape[len(current)]):
            pass
'''

# deprecated!!! use instead vectorized_diag_torch
def generalized_diag_torch(a: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor, e.g. A with shape (M, K, V), return a tenosr B 
    with shape (M, K, V, M, K, V) where B[m, k, v, m, k, v] = A[m, k, v]
    and 0 otherwise
    """
    diag = torch.zeros(a.shape * 2)
    indices = torch.cartesian_prod(*[torch.arange(dim) for dim in a.shape])

    if len(indices.shape) == 1:
        indices = tense_array_torch(indices, 1, 1)

    for index in indices:
        index = tuple(index)
        diag[index][index] = a[index]

    return diag

# deprecated!!! use instead: batch_tensor_diag()
def vectorized_diag_torch(a: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor A with shape (M, K, V), return a tensor B 
    with shape (M, K, V, M, K, V) where B[m, k, v, m, k, v] = A[m, k, v]
    and 0 otherwise.
    """
    # Get the shape
    shape = a.shape
    extended_shape = shape + shape  # (M, K, V, M, K, V)
    
    # Create an empty tensor of zeros
    diag = torch.zeros(extended_shape)

    # Generate indices for diagonal positions
    indices = torch.meshgrid(*[torch.arange(s) for s in shape], indexing='ij')
    diag[indices + indices] = a  # Indexing works because indices are tuples

    return diag

def batch_tensor_diag(a: torch.Tensor, nonBatchOrder: int = None) -> torch.Tensor:
    """
    Please notice the difference between:
     - advanced indexing (pair elements in each index tensor)
     - torch.meshgrid()  (cartesian product of index tensors)

    It's important to understand meshgrid + advancedIndexing routine
    """
    if nonBatchOrder is None:
        nonBatchOrder = len(a.shape)
    batchOrder = len(a.shape) - nonBatchOrder
    batchShape = a.shape[:batchOrder]
    nonBatchShape = a.shape[batchOrder:]

    indices = torch.meshgrid(*[torch.arange(dim) for dim in a.shape], indexing='ij')
    nonBatch_indices = torch.meshgrid(*[torch.arange(nonBatch_dim) for nonBatch_dim in nonBatchShape], indexing='ij')
    
    diag = torch.zeros(batchShape + nonBatchShape * 2)
    
    diag[indices + nonBatch_indices] = a  # advanced indexing, batch indexing

    return diag

def generalized_permute_torch(a: torch.Tensor, shape_partition: Tuple[int], partition_permute: Tuple[int]) -> torch.Tensor:
    """
    Generalized transpose of higher-ranked tensor, which
    can be seen as permute of its shape

    shape_partition: a partition of the shape of tensor
        e.g. a tenosr has shape (3, 4, 5, 6), shape_partition (2, 1, 1)
             the tensor is partitioned to (3, 4), (5,), (6,)
            
    partition_permute: the new permute of the partitioned axes
        e.g. partition_permute (2, 1, 0)
             the tensor is permute to a new one with shape (6, 5, 3, 4)

    Notice:
    - len(shape_partition) == len(partition_permute)
    - sum(shape_partition) == len(a.shape)
    """
    list_of_indice = []
    start_index = 0
    for len_part in shape_partition:
        end_index = start_index + len_part
        indice = tuple(range(start_index, end_index))
        list_of_indice.append(indice)
        start_index = end_index
    permuted_list_of_indice = [list_of_indice[i] for i in partition_permute]
    new_indice = tuple(index for indice in permuted_list_of_indice for index in indice)

    permuted = a.permute(*new_indice)

    return permuted
