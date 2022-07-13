"""Attention module library."""

import functools
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from savi.modules import misc
from savi.lib.utils import init_param, init_fn

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
                 slot_size: int, # fixed size. or same as qkv_size.
                 qkv_size: int = None, # fixed size, or slot size. # Optional[int] = None,
                 num_iterations: int = 1,
                 mlp_size: Optional[int] = None,
                 epsilon: float = 1e-8,
                 num_heads: int = 1,
                 weight_init: str = 'xavier_uniform'
                ):
        super().__init__()

        self.input_size = input_size
        self.slot_size = slot_size
        self.qkv_size = qkv_size if qkv_size is not None else slot_size
        self.num_iterations = num_iterations
        self.mlp_size = mlp_size
        self.epsilon = epsilon
        self.num_heads = num_heads
        self.weight_init = weight_init
        # other definitions
        self.head_dim = qkv_size // self.num_heads

        # shared modules
        ## gru
        self.gru = misc.myGRUCell(slot_size, slot_size, weight_init=weight_init)

        ## weights
        self.dense_q = nn.Linear(slot_size, qkv_size, bias=False)
        self.dense_k = nn.Linear(input_size, qkv_size, bias=False)
        self.dense_v = nn.Linear(input_size, qkv_size, bias=False)
        init_fn[weight_init['linear_w']](self.dense_q.weight)
        init_fn[weight_init['linear_w']](self.dense_k.weight)
        init_fn[weight_init['linear_w']](self.dense_v.weight)

        ## layernorms
        self.layernorm_q = nn.LayerNorm(qkv_size, eps=1e-6)
        self.layernorm_input = nn.LayerNorm(input_size, eps=1e-6)

        ## attention
        self.inverted_attention = InvertedDotProductAttention(
            input_size=qkv_size, output_size=slot_size,
            num_heads=self.num_heads, norm_type="mean",
            epsilon=epsilon, weight_init=weight_init)

        ## output transform
        if self.mlp_size is not None:
            self.mlp = misc.MLP(
                input_size=slot_size, hidden_size=self.mlp_size,
                output_size=slot_size, layernorm="pre", residual=True,
                weight_init=weight_init)

    def forward(self, slots: Array, inputs: Array,
                padding_mask: Optional[Array] = None) -> Array:
        """Slot Attention module forward pass."""
        del padding_mask # Unused.

        B, O, D = slots.shape
        _, L, M = inputs.shape

        # inputs.shape = (b, n_inputs, input_size).
        inputs = self.layernorm_input(inputs)
        # k.shape = (b, n_inputs, num_heads, head_dim).
        k = self.dense_k(inputs).view(B, L, self.num_heads, self.head_dim)
        # v.shape = (b, n_inputs, num_heads, head_dim).
        v = self.dense_v(inputs).view(B, L, self.num_heads, self.head_dim)

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):

            # Inverted dot-product attention.
            slots_n = self.layernorm_q(slots)
            ## q.shape = (b, num_objects, num_heads, qkv_size).
            q = self.dense_q(slots_n).view(B, O, self.num_heads, self.head_dim)
            updates, attn = self.inverted_attention(query=q, key=k, value=v)

            # Recurrent update.
            slots = self.gru(
                updates.reshape(-1, D), 
                slots.reshape(-1, D))
            slots = slots.reshape(B, -1, D)

            # Feedforward block with pre-normalization.
            if self.mlp_size is not None:
                slots = self.mlp(slots)

        return slots, attn

    def compute_attention(self, slots, inputs):
        """Slot Attention without GRU and iteration."""
                # inputs.shape = (b, n_inputs, input_size).
        B, O, D = slots.shape
        _, L, M = inputs.shape
        inputs = self.layernorm_input(inputs)
        slots = self.layernorm_q(slots)
        q = self.dense_q(slots).view(B, O, self.num_heads, self.head_dim)
        k = self.dense_k(inputs).view(B, L, self.num_heads, self.head_dim)
        v = self.dense_v(inputs).view(B, L, self.num_heads, self.head_dim)
        updated_slots, attn = self.inverted_attention(query=q, key=k, value=v)

        # updated_slots [B Q S], attn TODO: shape
        return updated_slots, attn

class InvertedDotProductAttention(nn.Module):
    """Inverted version of dot-product attention (softmax over query axis)."""

    def __init__(self,
                 input_size: int, # qkv_size # FIXME: added for submodules
                 output_size: int, # FIXME: added for submodules
                 num_heads: Optional[int] = 1, # FIXME: added for submodules
                 norm_type: Optional[str] = "mean", # mean, layernorm, or None
                 # multi_head: bool = False, # FIXME: can infer from num_heads.
                 epsilon: float = 1e-8,
                 dtype: DType = torch.float32,
                 weight_init = None
                 # precision # not used
                ):
        super().__init__()

        assert num_heads >= 1 and isinstance(num_heads, int)

        self.input_size = input_size
        self.output_size = output_size
        self.norm_type = norm_type
        self.num_heads = num_heads
        self.multi_head = True if num_heads > 1 else False
        self.epsilon = epsilon
        self.dtype = dtype
        self.weight_init = weight_init
        # other definitions
        self.head_dim = input_size // self.num_heads

        # submodules
        self.attn_fn = GeneralizedDotProductAttention(
            inverted_attn=True,
            renormalize_keys=True if self.norm_type == "mean" else False,
            epsilon=self.epsilon,
            dtype=self.dtype)
        if self.multi_head:
            self.dense_o = nn.Linear(input_size, output_size, bias=False)
            init_fn[weight_init['linear_w']](self.dense_o.weight)
        if self.norm_type == "layernorm":
            self.layernorm = nn.LayerNorm(output_size, eps=1e-6)

    def forward(self, query: Array, key: Array, value: Array) -> Array:
        """Computes inverted dot-product attention.

        Args:
            qk_features = [num_heads, head_dim] = qkv_dim
            query: Queries with shape of `[batch, q_num, qk_features]`.
            key: Keys with shape of `[batch, kv_num, qk_features]`.
            value: Values with shape of `[batch, kv_num, v_features]`.
            train: Indicating whether we're training or evaluating.

        Returns:
            Output of shape `[batch, n_queries, v_features]`
        """
        B, Q = query.shape[:2]

        # Apply attention mechanism
        output, attn = self.attn_fn(query=query, key=key, value=value)

        if self.multi_head:
            # Multi-head aggregation. Equivalent to concat + dense layer.
            output = self.dense_o(output.view(B, Q, self.input_size)).view(B, Q, self.output_size)
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
        query = query / (qk_features ** 0.5) # torch.sqrt(qk_features)

        # attn.shape = (batch..., num_heads, q_num, kv_num)
        attn = torch.matmul(query.permute(0, 2, 1, 3), key.permute(0, 2, 3, 1)) # bhqd @ bhdk -> bhqk

        if self.inverted_attn:
            attention_dim = -2 # Query dim
        else:
            attention_dim = -1 # Key dim

        # Softmax normalization (by default over key dim)
        attn = torch.softmax(attn, dim=attention_dim, dtype=self.dtype)

        if self.renormalize_keys:
            # Corresponds to value aggregation via weighted mean (as opposed to sum).
            normalizer = torch.sum(attn, axis=-1, keepdim=True) + self.epsilon
            attn_n = attn / normalizer
        else:
            attn_n = attn

        if self.attn_weights_only:
            return attn_n

        # Aggregate values using a weighted sum with weights provided by `attn`
        updates = torch.einsum("bhqk,bkhd->bqhd", attn_n, value)

        return updates, attn # FIXME: return attention too, as no option for intermediate storing in module in torch.


class Transformer(nn.Module):
    """Transformer with multiple blocks."""

    def __init__(self,
                 embed_dim: int, # FIXME: added for submodules
                 num_heads: int,
                 qkv_size: int,
                 mlp_size: int,
                 num_layers: int,
                 pre_norm: bool = False
                ):
        super().__init__()

        self.num_heads = num_heads
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.num_layes = num_layers
        self.pre_norm = pre_norm

        # submodules
        self.model = nn.ModuleList()
        for lyr in range(self.num_layers):
            self.model.add_module(
                name=f"TransformerBlock_{lyr}",
                module=TransformerBlock(
                    embed_dim=embed_dim, num_heads=num_heads,
                    qkv_size=qkv_size, mlp_size=mlp_size,
                    pre_norm=pre_norm)
            )

    def forward(self, queries: Array, inputs: Optional[Array] = None,
                padding_mask: Optional[Array] = None,
                train: bool = False) -> Array:
        x = queries
        for layer in self.model:
            x = layer(x, inputs, padding_mask, train)
        return x


class TransformerBlockOld(nn.Module):
    """Tranformer decoder block."""

    def __init__(self,
                 embed_dim: int, # FIXME: added for submodules
                 num_heads: int,
                 qkv_size: int,
                 mlp_size: int,
                 pre_norm: bool = False,
                 cross_attn: bool = False
                ):
        super().__init__()

        self.num_heads = num_heads
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.pre_norm = pre_norm

        # submodules
        ## MHA
        self.attn_self = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.attn_cross = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True) if cross_attn else None
        ## mlps
        self.mlp = misc.MLP(
            input_size=embed_dim, hidden_size=mlp_size, 
            output_size=embed_dim)
        ## layernorms
        self.layernorm_query = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_inputs = nn.LayerNorm(embed_dim, eps=1e-6) if cross_attn else None
        self.layernorm_mlp = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, queries: Array, inputs: Optional[Array] = None,
                padding_mask: Optional[Array] = None,
                train: bool = False) -> Array:
        del padding_mask, train # Unused.
        assert queries.ndim == 3

        if self.pre_norm:
            # Self-attention on queries.
            x = self.layernorm_query(queries)
            x, _ = self.attn_self(query=x, key=x, value=x)
            x = x + queries

            # Cross-attention on inputs.
            if inputs is not None:
                assert inputs.ndim == 3
                y = self.layernorm_inputs(x)
                y, _ = self.attn_cross(q=y, k=inputs, v=inputs)
                y = y + x
            else:
                y = x

            # MLP
            z = self.layernorm_mlp(y)
            z = self.mlp(z)
            z = z + y
        else:
            # Self-attention on queries.
            x = queries
            x, _ = self.attn_self(query=x, key=x, value=x)
            x = x + queries
            x = self.layernorm_query(x)

            # Cross-attention on inputs.
            if inputs is not None:
                assert inputs.ndim == 3
                y, _ = self.attn_cross(query=x, key=inputs, value=inputs)
                y = y + x
                y = self.layernorm_inputs(y)
            else:
                y = x

            # MLP
            z = self.mlp(y)
            z = z + y
            z = self.layernorm_mlp(z)
        return z


class TransformerBlock(nn.Module):
    """Tranformer decoder block."""

    def __init__(self,
                 embed_dim: int, # FIXME: added for submodules
                 num_heads: int,
                 qkv_size: int,
                 mlp_size: int,
                 pre_norm: bool = False,
                 weight_init = None
                ):
        super().__init__()

        self.embed_dim = embed_dim
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        self.weight_init = weight_init

        assert num_heads >= 1
        assert qkv_size % num_heads == 0, "embed dim must be divisible by num_heads"
        self.head_dim = qkv_size // num_heads

        # submodules
        ## MHA #
        self.attn = GeneralizedDotProductAttention()
        ## mlps
        self.mlp = misc.MLP(
            input_size=embed_dim, hidden_size=mlp_size, 
            output_size=embed_dim, weight_init=weight_init)
        ## layernorms
        self.layernorm_query = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_mlp = nn.LayerNorm(embed_dim, eps=1e-6)
        ## weights
        self.dense_q = nn.Linear(embed_dim, qkv_size)
        self.dense_k = nn.Linear(embed_dim, qkv_size)
        self.dense_v = nn.Linear(embed_dim, qkv_size)
        init_fn[weight_init['linear_w']](self.dense_q.weight)
        init_fn[weight_init['linear_b']](self.dense_q.bias)
        init_fn[weight_init['linear_w']](self.dense_k.weight)
        init_fn[weight_init['linear_b']](self.dense_k.bias)
        init_fn[weight_init['linear_w']](self.dense_v.weight)
        init_fn[weight_init['linear_b']](self.dense_v.bias)
        if self.num_heads > 1:
            self.dense_o = nn.Linear(qkv_size, embed_dim)
            # nn.init.xavier_uniform_(self.w_o.weight)
            init_fn[weight_init['linear_w']](self.dense_o.weight)
            init_fn[weight_init['linear_b']](self.dense_o.bias)
            self.multi_head = True
        else:
            self.multi_head = False

    def forward(self, inputs: Array) -> Array: # TODO: add general attention for q, k, v, not just for x = qkv
        assert inputs.ndim == 3

        B, L, _ = inputs.shape
        head_dim = self.embed_dim // self.num_heads

        if self.pre_norm:
            # Self-attention.
            x = self.layernorm_query(inputs)
            q = self.dense_q(x).view(B, L, self.num_heads, head_dim)
            k = self.dense_k(x).view(B, L, self.num_heads, head_dim)
            v = self.dense_v(x).view(B, L, self.num_heads, head_dim)
            x, _ = self.attn(query=q, key=k, value=v)
            if self.multi_head:
                x = self.dense_o(x.reshape(B, L, self.qkv_size)).view(B, L, self.embed_dim)
            else:
                x = x.squeeze(-2)
            x = x + inputs

            y = x

            # MLP
            z = self.layernorm_mlp(y)
            z = self.mlp(z)
            z = z + y
        else:
            # Self-attention on queries.
            x = inputs
            q = self.dense_q(x).view(B, L, self.num_heads, head_dim)
            k = self.dense_k(x).view(B, L, self.num_heads, head_dim)
            v = self.dense_v(x).view(B, L, self.num_heads, head_dim)
            x, _ = self.attn(query=q, key=k, value=v)
            if self.multi_head:
                x = self.dense_o(x.reshape(B, L, self.qkv_size)).view(B, L, self.embed_dim)
            else:
                x = x.squeeze(-2)
            x = x + inputs
            x = self.layernorm_query(x)

            y = x

            # MLP
            z = self.mlp(y)
            z = z + y
            z = self.layernorm_mlp(z)
        return z