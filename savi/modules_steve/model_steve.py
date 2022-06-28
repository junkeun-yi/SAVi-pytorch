"""Video Slot Attention model based on STEVE."""

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


class Processor(nn.Module):
	"""Recurrent processor module.

	This module is scanned (applied recurrently) over the sequence dimension of
	the input and applies a corrector and a predictor module. The corrector is
	only applied if new inputs (such as new image/frame) are received and uses
	the new input to correct its internal state.

	The predictor is equivalent to a latent transition model and produces a
	prediction for the state at the next time step, given teh current (corrected)
	state.
	"""

	def __init__(self,
				 corrector: nn.Module,
				 predictor: nn.Module
				):
		super().__init__()

		self.corrector = corrector
		self.predictor = predictor

	def forward(self, slots: ProcessorState, inputs: Optional[Array],
				padding_mask: Optional[Array]) -> Tuple[Array, Array]:
		
		# Only apply corrector if we receive new inputs.
		if inputs is not None:
			# flatten spatial dims
			inputs = inputs.flatten(1, 2)
			corrected_slots, attn = self.corrector(slots, inputs, padding_mask)
		# Otherwise simply use previous state as input for predictor
		else:
			corrected_slots = slots
		
		# Always apply predictor (i.e. transition model).
		predicted_slots = self.predictor(corrected_slots)

		# Prepare outputs
		return corrected_slots, predicted_slots, attn


class STEVE(nn.Module):
	"""Video model consisting of encoder, recurrent processor, and decoder."""

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
		self.processor = Processor(corrector, predictor)

	def forward(self, video: Array, conditioning: Optional[Array] = None,
				continue_from_previous_state: bool = False,
				padding_mask: Optional[Array] = None, **kwargs) -> ArrayTree:
		"""Performs a forward pass on a video.

		Args:
			video: Video of shape `[batch_size, n_frames, height, width, n_channels]`.
			conditioning: Optional tensor used for conditioning the initial state
				of the recurrent processor.
			continue_from_previous_state: Boolean, whether to continue from a previous
				state or not. If True, the conditioning variable is used directly as
				initial state.
			padding_mask: Binary mask for padding video inputs (e.g. for videos of
				different sizes/lengths). Zero corresponds to padding.

		Returns:
			A dictionary of model predictions.
		"""
		del kwargs # Unused.

		if padding_mask is None:
			padding_mask = torch.ones(video.shape[:-1], dtype=torch.int32)
		
		# video.shape = (batch_size, n_frames, height, width, n_channels)
		B, T, H, W, C = video.shape
		# encoded_inputs = self.encoder(video, padding_mask)
		# flatten over B * Time and unflatten after to get [B, T, h*, w*, F]
		encoded_inputs = self.encoder(video.flatten(0, 1))
		encoded_inputs = encoded_inputs.reshape(shape=(B, T, *encoded_inputs.shape[-3:]))

		if continue_from_previous_state:
			assert conditioning is not None, (
				"When continuing from a previous state, the state has to be passed "
				"via the `conditioning` variable, which cannot be `None`."
			)
			init_slots = conditioning[:, -1] # currently, only use last state.
			# init_slots = conditioning # given [B, N, D], the slots of the last state
		else:
			# same as above but without encoded inputs.
			init_slots = self.initializer(
				conditioning, batch_size=video.shape[0])

		# Scan recurrent processor over encoded inputs along sequence dimension.
		outputs, outputs_pred, attn = None, None, None
		slots_corrected_list, slots_predicted_list, attn_list = [], [], []
		predicted_slots = init_slots
		for t in range(T):
			slots = predicted_slots
			encoded_frame = encoded_inputs[:, t]
			corrected_slots, predicted_slots, attn_t = self.processor(slots, encoded_frame, padding_mask)

			slots_corrected_list.append(corrected_slots.unsqueeze(1))
			slots_predicted_list.append(predicted_slots.unsqueeze(1))
			attn_list.append(attn_t.unsqueeze(1))

		corrected_slots = torch.cat(slots_corrected_list, dim=1)
		predicted_slots = torch.cat(slots_predicted_list, dim=1)
		attn = torch.cat(attn_list, dim=1)

		# Decode latent states
		outputs = self.decoder(corrected_slots.flatten(0,1)) if self.decode_corrected else None
		outputs_pred = self.decoder(predicted_slots.flatten(0,1)) if self.decode_predicted else None

		if outputs is not None:
			for key, value in outputs.items():
				outputs[key] = value.reshape(B, T, *value.shape[1:])
		if outputs_pred is not None:
			for key, value in outputs_pred.items():
				outputs_pred[key] = value.reshape(B, T, *value.shape[1:])

		return {
			"states": corrected_slots,
			"states_pred": predicted_slots,
			"outputs": outputs,
			"outputs_pred": outputs_pred,
			"attention": attn
		}