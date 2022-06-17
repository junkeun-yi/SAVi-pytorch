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



class SpatialBroadcastDecoder(nn.Module):
	"""Spatial broadcast decoder for as set of slots (per frame)."""

	def __init__(self,
				 resolution: Sequence[int],
				 backbone: nn.Module,
				 pos_emb: nn.Module,
				 target_readout: nn.Module = None
				):
		super().__init__()

		self.resolution = resolution
		self.backbone = backbone
		self.pos_emb = pos_emb
		self.target_readout = target_readout

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

		alpha_mask = alpha_logits.softmax(dim=1)

		# TODO: figure out what to do with readout.
		targets_dict = self.target_readout(bb_features)

		preds_dict = dict()
		for target_key, channels in targets_dict.items():

			# channels.shape = (batch_size, n_slots, h, w, c)
			channels = channels.reshape(shape=(batch_size, n_slots, *spatial_dims, -1))

			# masked_channels.shape = (batch_size, n_slots, h, w, c)
			masked_channels = channels * alpha_mask

			# decoded_target.shape = (batch_size, h, w, c)
			decoded_target = torch.sum(masked_channels, dim=1) # Combine target
			preds_dict[target_key] = decoded_target

			if not self.training: # intermediates for logging.
				preds_dict[f"eval/{target_key}_slots"] = channels
				preds_dict[f"eval/{target_key}_masked"] = masked_channels
				preds_dict[f"eval/{target_key}_combined"] = decoded_target

		# if not self.training: # intermediates for logging.
		#     preds_dict["eval/alpha_mask"] = alpha_mask
		preds_dict["alpha_mask"] = alpha_mask

		if not self.training: # only return for evaluation
			preds_dict["segmentations"] = alpha_logits.argmax(dim=1)

		return preds_dict