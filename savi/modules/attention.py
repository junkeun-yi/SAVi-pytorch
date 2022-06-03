"""Attention module library."""

# TODO: 

import functools
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from savi.modules import misc

Shape = Tuple[int]

DType = Any
Array = torch.Tensor # np.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet  # TODO: what is this ?
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]


class SlotAttention(nn.Module):
    """Slot Attention module.
    
    Note: This module uses pre-normalization by default.
    """
    def __init__(self,
                 num_iterations: int = 1,
                 qkv_size: Optional[int] = None,
                 mlp_size: Optional[int] = None,
                 epsilon: float = 1e-8,
                 num_heads: int = 1
                ):
        super().__init__()

        self.num_iterations = num_iterations
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.epsilon = epsilon
        self.num_heads = num_heads

        self.dense_q = nn.Linear()
        self.

    def forward(self, slots: Array, inputs: Array,
                padding_mask: Optional[Array] = None,
                train: bool = False) -> Array:
        """Slot Attention module forward pass."""
        del padding_mask, train # Unused.

        qkv_size = self.qkv_size or slots.shape[-1]
        head_dim = qkv_size // self.num_heads
        