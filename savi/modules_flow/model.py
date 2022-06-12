"""Video Slot Attention model based on Flow prediction."""

from typing import Any, Callable, Dict, Iterable, Mapping, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from savi.lib import utils
from savi.modules import misc


Array = torch.Tensor
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]

class FlowPrediction(nn.Module):
	"""Vide model consisting of encoder, slot attention module, and mask decoder."""

	def __init__(self,
				 encoder: nn.Module,
				 decoder: nn.Module,
				 obj_slot_attn: nn.Module,
				 frame_pred: nn.Module,
				 initializer: nn.Module
				):
		super().__init__()

		self.encoder = encoder
		self.decoder = decoder
		self.obj_slot_attn = obj_slot_attn
		self.frame_pred = frame_pred
		self.initializer = initializer

		# submodules

	def forward(self, video: Array, conditioning: Optional[Array] = None,
				padding_mask: Optional[Array] = None) -> ArrayTree:
		"""Performs a forward pass on a video.

		Args:
			video: Video of shape `[batch_size, n_frames, height, width, n_channels]`.
			conditioning: Optional tensor used for conditioning the initial state
				of the object attention module.

		Returns:
			pred_frames: (B T H W C). predicted video frames.
			masks_t: (B T H W 1). segmentation masks.
			slot_flow_pred: (B T H W 2). predicted flow.
			slots_t: (B T N S). framewise slots.
			att_t: (B T (h* w*) N). attention maps.
		"""
		del padding_mask # Unused. only here for consistency and easier training code writing.

		B, T, H, W, C = video.shape
		# encode and position embed.
		# flatten over batch * time and encode to get [B, T, h* w*, F]
		enc = self.encoder(video.flatten(0, 1))
		_, h, w, F = enc.shape
		enc = enc.reshape(shape=(B, T, h, w, F))
		
		# initialize slots. [B, N, S] where N = num_slots, S = slot_dim
		slots = self.initializer(
			conditioning, batch_size=video.shape[0])
		_, N, S = slots.shape

		# get attn b/w slots and spatio-temporal features
		# attn with inputs as [B (T h* w*) F]
		slots, _ = self.obj_slot_attn(slots, enc.flatten(1, 3))

		# slots = ((B T) N S)
		slots = slots.repeat_interleave(T, 0)

		# get attn b/w slots and spatial features
		# attn with inputs as [(B T) (h* w*) F]
		slots_t, att_t = self.obj_slot_attn(slots, enc.flatten(2, 3).flatten(0, 1))

		# slots_t = [B T N S], att_t = (B T (h* w*) N)
		slots_t = slots_t.reshape(shape=(B, T, N, S))
		att_t = att_t.reshape(shape=(B, T, N, (h*w))).permute(0, 1, 3, 2)

		# find object-wise masks and object-wise forward flows per frame
		# TODO: originally, we concatenated adjacent slots, 
		#   but here we try a sum to preserve slot dimension.
		# adjacent_slots = torch.cat([slots_t[:, :-1], slots_t[:, 1:]], dim=2)
		adjacent_slots = slots_t[:, :-1] + slots_t[:, 1:] # (B (T-1) N S)
		# add the first slot to get a mask for it too ...
		# adding the first slot to itself will model no movement.
		adjacent_slots = torch.cat([slots_t[:, :1]*2, adjacent_slots], dim=1)
		# inputs are adjacent slots = ((B T) N S)
		outputs = self.decoder(adjacent_slots.flatten(0, 1))

		masks_t = outputs["segmentations"].reshape(shape=(B, T, H, W, 1)) # (B T N H W 1)
		slot_flow_pred = outputs["flow"].reshape(shape=(B, T, H, W, 2)) # (B T N H W 2)

		# predict next frames (first frame copied twice. don't predict unknown future.)
		# pred_frames = (B T H W C)
		vid_input = torch.cat([video[:, :1], video[:, :-1]], dim=1).flatten(0,1)
		pred_frames = self.frame_pred(vid_input, slot_flow_pred.flatten(0, 1), channels_last=True)
		pred_frames = pred_frames.reshape(shape=(B, T, H, W, C))

		return pred_frames, masks_t, slot_flow_pred, slots_t, att_t


class FlowWarp(nn.Module):
	"""Warp an image with its forward optical flow."""

	def get_grid(self, batchsize, rows, cols, gpu_id=0):
		hor = torch.linspace(-1.0, 1.0, cols)
		hor.requires_grad = False
		hor = hor.view(1, 1, 1, cols)
		hor = hor.expand(batchsize, 1, rows, cols)
		ver = torch.linspace(-1.0, 1.0, rows)
		ver.requires_grad = False
		ver = ver.view(1, 1, rows, 1)
		ver = ver.expand(batchsize, 1, rows, cols)

		t_grid = torch.cat([hor, ver], 1)
		t_grid.requires_grad = False

		return t_grid.cuda(gpu_id)

	def forward(self, image, flow, channels_last=False):
		if channels_last:
			# inputs.shape = (batch_size, height, width, n_channels)
			image = image.permute((0, 3, 1, 2))
			flow = flow.permute((0, 3, 1, 2))
			# inputs.shape = (batch_size, n_channels, height, width)

		# image shape B C H W
		# flow shape B 2 H W
		b, c, h, w = image.size()
		grid = self.get_grid(b, h, w, gpu_id=flow.get_device())
		flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)
		# final_grid shape B H W 2
		final_grid = (grid + flow).permute(0, 2, 3, 1).cuda(image.get_device())
		# warp the image using grid sample on image and flow field (final_grid)
		warped_image = torch.nn.functional.grid_sample(image, final_grid, mode='bilinear', padding_mode='border', align_corners=True)
		
		if channels_last:
			warped_image = warped_image.permute((0, 2, 3, 1))

		return warped_image