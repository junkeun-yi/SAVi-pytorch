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
                 input_size: int, # size of encoded inputs. # FIXME: added for submodules.
                 qkv_size: int, # fixed size, or slot size. # Optional[int] = None,
                 slot_size: int, # fixed size. or same as qkv_size.
                 num_slots: int, # fixed size.
                 num_iterations: int = 1,
                 mlp_size: Optional[int] = None,
                 epsilon: float = 1e-8,
                 num_heads: int = 1
                ):
        super().__init__()

        self.input_size = input_size
        self.qkv_size = qkv_size
        self.slot_size = slot_size
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.epsilon = epsilon
        self.num_heads = num_heads

        # shared modules
        self.w_q = nn.Parameter(torch.Tensor(num_heads, slot_size, qkv_size))
        self.w_k = nn.Parameter(torch.Tensor(num_heads, input_size, qkv_size))
        self.w_v = nn.Parameter(torch.Tensor(num_heads, input_size, qkv_size))

        self.layernorm_input = nn.LayerNorm(input_size)
        self.layernorm_q = nn.LayerNorm(qkv_size)

        self.inverted_attention = InvertedDotProductAttention(
            input_size=qkv_size, output_size=slot_size,
            num_heads=self.num_heads, norm_type="mean")

        self.gru = nn.GRUCell(slot_size, slot_size)

        if self.mlp_size is not None:
            self.mlp = misc.MLP(
                input_size=slot_size, hidden_size=self.mlp_size,
                output_size=slot_size, layernorm="pre", residual=True)

    def forward(self, slots: Array, inputs: Array,
                padding_mask: Optional[Array] = None,
                train: bool = False) -> Array:
        """Slot Attention module forward pass."""
        del padding_mask, train # Unused.

        # inputs.shape = (b, n_inputs, input_size).
        inputs = self.layernorm_input(inputs)
        # k.shape = (b, n_inputs, num_heads, qkv_size).
        k = torch.einsum("bkm,hmd->bkhd", inputs, self.w_k)
        # v.shape = (b, n_inputs, num_heads, qkv_size).
        v = torch.einsum("bkm,hmd->bkhd", inputs, self.w_v)

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):

            # Inverted dot-product attention.
            slots_n = self.layernorm_q(slots)
            ## q.shape = (b, num_objects, num_heads, qkv_size).
            q = torch.einsum("bqs,hsd->bqhd", slots_n, self.w_q)
            updates, attn = self.inverted_attention(query=q, key=k, value=v)

            # Recurrent update.
            slots = self.gru(updates, slots)

            # Feedforward block with pre-normalization.
            if self.mlp_size is not None:
                slots = self.mlp(slots)

        return slots, attn


class InvertedDotProductAttention(nn.Module):
    """Inverted version of dot-product attention (softmax over query axis)."""

    def __init__(self,
                 input_size: int, # qkv_size # FIXME: added for submodules
                 output_size: int, # FIXME: added for submodules
                 num_heads: Optional[int] = None, # FIXME: added for submodules
                 norm_type: Optional[str] = "mean", # mean, layernorm, or None
                 # multi_head: bool = False, # FIXME: can infer from num_heads.
                 epsilon: float = 1e-8,
                 dtype: DType = torch.float32,
                 # precision # not used
                ):

        self.norm_type = norm_type
        self.multi_head = True if num_heads is not None else False
        self.epsilon = epsilon
        self.dtype = dtype

        # submodules
        self.attn_fn = GeneralizedDotProductAttention(
            inverted_attn=True,
            renormalize_keys=True if self.norm_type == "mean" else False,
            epsilon=self.epsilon,
            dtype=self.dtype)
        if self.multi_head:
            self.w_o = nn.Parameter(torch.Tensor(num_heads, input_size, output_size))
        if self.norm_type == "layernorm":
            self.layernorm = nn.LayerNorm(output_size)

    def forward(self, query: Array, key: Array, value: Array,
                train: bool = False) -> Array:
        """Computes inverted dot-product attention.

        Args:
            query: Queries with shape of `[batch, q_num, qk_features]`.
            key: Keys with shape of `[batch, kv_num, qk_features]`.
            value: Values with shape of `[batch, kv_num, v_features]`.
            train: Indicating whether we're training or evaluating.

        Returns:
            Output of shape `[batch, n_queries, v_features]`
        """
        del train # Unused.

        # Apply attention mechanism
        output, attn = self.attn_fn(query=query, key=key, value=value)

        if self.multi_head:
            # Multi-head aggregation. Equivalent to concat + dense layer.
            output = torch.einsum("bqhd,hds->bqs", output, self.w_o)
        else:
            # Remove head dimension.
            output = output.squeeze(-2)

        if self.norm_type == "layernorm":
            output = self.layernorm(output)

        return output, attn


class GeneralizedDotProductAttention(nn.Module):
    """Multi-head dot-product attention with customizable normalization axis.

    This module supports logging of attention weights in a variable collection.
    """

    def __init__(self,
                 dtype: DType = torch.float32,
                 # precision: Optional[] # not used
                 epsilon: float = 1e-8,
                 inverted_attn: bool = False,
                 renormalize_keys: bool = False,
                 attn_weights_only: bool = False
                ):
        super().__init__()

        self.dtype = dtype
        self.epsilon = epsilon
        self.inverted_attn = inverted_attn
        self.renormalize_keys = renormalize_keys
        self.attn_weights_only = attn_weights_only

    def forward(self, query: Array, key: Array, value: Array,
                train: bool = False, **kwargs) -> Array:
        """Computes multi-head dot-product attention given query, key, and value.

        Args:
            query: Queries with shape of `[batch..., q_num, num_heads, qk_features]`.
            key: Keys with shape of `[batch..., kv_num, num_heads, qk_features]`.
            value: Values with shape of `[batch..., kv_num, num_heads, v_features]`.
            train: Indicating whether we're training or evaluating.
            **kwargs: Additional keyword arguments are required when used as attention
                function in nn.MultiHeadDotPRoductAttention, but they will be ignored here.

        Returns:
            Output of shape `[batch..., q_num, num_heads, v_features]`.
        """
        del train # Unused.

        assert query.ndim == key.ndim == value.ndim, (
            "Queries, keys, and values must have the same rank.")
        assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
            "Query, key, and value batch dimensions must match.")
        assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
            "Query, key, and value num_heads dimensions must match.")
        assert key.shape[-3] == value.shape[-3], (
            "Key and value cardinality dimensions must match.")
        assert query.shape[-1] == key.shape[-1], (
            "Query and key feature dimensions must match.")

        if kwargs.get("bias") is not None:
            raise NotImplementedError(
                "Support for masked attention is not yet implemented.")

        if "dropout_rate" in kwargs:
            if kwargs["dropout_rate"] > 0.:
                raise NotImplementedError("Support for dropout is not yet implemented.")

        # Temperature normalization.
        qk_features = query.shape[-1]
        query = query / torch.sqrt(qk_features)

        # attn.shape = (batch..., num_heads, q_num, kv_num)
        attn = torch.einsum("bqhd,bkhd->bhqk", query, key)

        if self.inverted_attn:
            attention_dim = -2 # Query dim
        else:
            attention_dim = -1 # Key dim

        # Softmax normalization (by default over key dim)
        attn = torch.softmax(attn, dim=attention_dim, dtype=self.dtype)

        if self.renormalize_keys:
            # Corresponds to value aggregation via weighted mean (as opposed to sum).
            normalizer = torch.sum(attn, axis=-1, keepdim=True) + self.epsilon
            attn_normalized = attn / normalizer

        if self.attn_weights_only:
            return attn_normalized

        # Aggregate values using a weighted sum with weights provided by `attn`
        updates = torch.einsum("bhqk,bkhd->bqhd", attn_normalized, value)

        return updates, attn # FIXME: return attention too, as no option for intermediate storing in module in torch.
        