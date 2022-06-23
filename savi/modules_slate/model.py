"""Video Slot Attention model based on STEVE / SLATE."""

from typing import Any, Callable, Dict, Iterable, Mapping, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from savi.lib import utils
from savi.modules import misc

Array = torch.Tensor
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]


class STEVE(nn.Module):
	"""Video model consisting of encoder, slot attention module, and mask decoder."""

	def __init__(self,
				 encoder: nn.Module,
				 decoder: nn.Module,
				 dvae: nn.Module,
				 corrector: nn.Module,
				 predictor: nn.Module,
				 initializer: nn.Module,
				 decode_corrected: bool = True,
				 decode_predicted: bool = True
				):
		super().__init__()

		self.encoder = encoder
		self.decoder = decoder
		self.dvae = dvae
		self.corrector = corrector
		self.predictor = predictor
		self.initializer = initializer
		self.decode_corrected = decode_corrected
		self.decode_predicted = decode_predicted

		# submodules

	def forward(self, video: Array, conditioning: Optional[Array] = None,
				padding_mask: Optional[Array] = None, **kwargs) -> ArrayTree:
		"""Performs a forward pass on a video.

		Args:
			video: Video of shape `[batch_size, n_frames, height, width, n_channels]`.
			conditioning: Optional tensor used for conditioning the initial state
				of the object attention module.

		Returns:
			pred_frames: (B T H W C). predicted video frames.
			pred_seg: (B T H W 1). predicted segmentation.
			pred_flow: (B (T-1) H W 2). predicted flow.
			slots_t: (B T N S). framewise slots.
			att_t: (B T (h* w*) N). attention maps.
		"""
		del padding_mask # Unused.
		return 1