"""dVAE.

From https://github.com/singhgautam/slate/blob/master/dvae.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import savi.modules_steve.utils as stutils

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

	def forward(self, inputs: Array, tau, hard, channels_last=False):
		"""forward pass of the autoencoder"""
		if channels_last:
			inputs = self.switch_to_channels_first(inputs)

		x = inputs

		# dvae encode
		encoded = self.encoder(x)
		z_logits = F.log_softmax(encoded, dim=1)
		_, _, H_enc, W_enc = z_logits.shape
		z = stutils.gumbel_softmax(z_logits, tau, hard, dim=1)

		# dvae recon
		decoded = self.decoder(z)

		return z, decoded

	def encode(self, inputs: Array, tau, hard, channels_last=False):
		if channels_last:
			inputs = self.switch_to_channels_first(inputs)
		encoded = self.encoder(inputs)
		z_logits = F.log_softmax(encoded, dim=1)
		_, _, H_enc, W_enc = z_logits.shape
		z = stutils.gumbel_softmax(z_logits, tau, hard, dim=1)
		return encoded, z_logits, z

	def decode(self, z: Array, channels_last=False):
		decoded = self.decoder(z)
		if channels_last:
			decoded =  self.switch_to_channels_last(decoded)
		return decoded


"""
Example:

	dvae = dVAE(
		encoder=nn.Sequential(
			Conv2dBlock(3, 64, 4, 4, 0),
			Conv2dBlock(64, 64, 1, 1, 0),
			Conv2dBlock(64, 64, 1, 1, 0),
			Conv2dBlock(64, 64, 1, 1, 0),
			Conv2dBlock(64, 64, 1, 1, 0),
			Conv2dBlock(64, 64, 1, 1, 0),
			Conv2dBlock(64, 64, 1, 1, 0),
			conv2d(64, vocab_size, 1, 1, 0)),
		decoder=nn.Sequential(
			Conv2dBlock(vocab_size, 64, 1, 1, 0),
			Conv2dBlock(64, 64, 3, 1, 1),
			Conv2dBlock(64, 64, 1, 1, 0),
			Conv2dBlock(64, 64, 1, 1, 0),
			Conv2dBlock(64, 64 * 2 * 2, 1, 1, 0),
			nn.PixelShuffle(2),
			Conv2dBlock(64, 64, 3, 1, 1),
			Conv2dBlock(64, 64, 1, 1, 0),
			Conv2dBlock(64, 64, 1, 1, 0),
			Conv2dBlock(64, 64 * 2 * 2, 1, 1, 0),
			nn.PixelShuffle(2),
			conv2d(64, 3, 1, 1, 0)))
"""