"""Decoder module library."""

# FIXME

import functools
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union
from pyparsing import alphas

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from savi.lib import utils

Shape = Tuple[int]

DType = Any
Array = torch.Tensor
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]



class SpatialBroadcastMaskDecoder(nn.Module):
	"""Spatial broadcast decoder for as set of slots (per frame)."""

	def __init__(self,
				 resolution: Sequence[int],
				 backbone: nn.Module,
				 pos_emb: nn.Module
				):
		super().__init__()

		self.resolution = resolution
		self.backbone = backbone
		self.pos_emb = pos_emb

		# submodules
		self.mask_pred = nn.Linear(self.backbone.features[-1], 1)

	def forward(self, slots: Array) -> Array:

		batch_size, n_slots, n_features = slots.shape

		# Fold slot dim into batch dim.
		x = slots.reshape(shape=(batch_size * n_slots, n_features))

		# Spatial broadcast with position embedding.
		x = utils.spatial_broadcast(x, self.resolution)
		x = self.pos_emb(x)

		# bb_features.shape = (batch_size * n_slots, h, w, c)
		bb_features = self.backbone(x, channels_last=True)
		spatial_dims = bb_features.shape[-3:-1]

		alpha_logits = self.mask_pred( # take each feature separately
			bb_features.reshape(shape=(-1, bb_features.shape[-1])))
		alpha_logits = alpha_logits.reshape(
			shape=(batch_size, n_slots, *spatial_dims, -1)) # (B O H W 1)

		return bb_features, alpha_logits