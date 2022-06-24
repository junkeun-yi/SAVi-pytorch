"""dVAE.

From https://github.com/singhgautam/slate/blob/master/dvae.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Any, Callable, Dict, Iterable, Mapping, NamedTuple, Optional, Tuple, Union

Array = torch.Tensor
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]

class dVAE(nn.Module):
	"""dVAE model for discrete tokenization of video frames."""

	def __init__(self,
				 encoder: nn.Module,
				 decoder: nn.Module,
				 vocab_size: int = 4096,
				 img_channels: int = 3
				):
		super().__init__()

		self.encoder = encoder
		self.decoder = decoder
		self.vocab_size = vocab_size
		self.img_channels = img_channels

	def switch_to_channels_last(self, inputs: Array):
		"""[B C H W] -> [B H W C]"""
		assert inputs.ndim == 4
		return inputs.permute(0, 2, 3, 1)

	def switch_to_channels_first(self, inputs: Array):
		"""[B H W C] -> [B C H W]"""
		assert inputs.ndim == 4
		return inputs.permute(0, 3, 1, 2)

	def forward(self, inputs: Array, channels_last=False):
		"""forward pass of the autoencoder"""
		if channels_last:
			inputs = self.switch_to_channels_first(inputs)

		x = inputs
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)

		if channels_last:
			encoded = self.switch_to_channels_last(encoded)
			decoded = self.switch_to_channels_last(decoded)

		return encoded, decoded

	def encode(self, inputs: Array, channels_last=False):
		if channels_last:
			inputs = self.switch_to_channels_first(inputs)
		encoded = self.encoder(inputs)
		if channels_last:
			encoded =  self.switch_to_channels_last(encoded)
		return encoded

	def decode(self, inputs: Array, channels_last=False):
		if channels_last:
			inputs = self.switch_to_channels_first(inputs)
		decoded = self.decoder(inputs)
		if channels_last:
			decoded =  self.switch_to_channels_last(decoded)
		return decoded