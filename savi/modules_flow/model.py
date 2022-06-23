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

def create_mlp(input_dim, output_dim):
    obj_vecs_net = nn.Sequential(
        nn.Linear(input_dim, input_dim, bias=True),
        nn.BatchNorm1d(input_dim),
        nn.ReLU(inplace=True),
        nn.Linear(input_dim, input_dim, bias=True),
        nn.BatchNorm1d(input_dim),
        nn.ReLU(inplace=True),
        nn.Linear(input_dim, output_dim, bias=True),
        nn.Tanh()
    )
    return obj_vecs_net

class FramePrediction(nn.Module):
	"""Video model consisting of encoder, decoder, and slot attention module."""

	def __init__(self,
				 encoder: nn.Module,
				 decoder: nn.Module,
				 pos_embed: nn.Module,
				 obj_slot_attn: nn.Module,
				 obj_frame_pred: nn.Module,
				 initializer: nn.Module,
				):
		super().__init__()

		self.encoder = encoder
		self.decoder = decoder
		self.pos_embed = pos_embed
		self.obj_slot_attn = obj_slot_attn
		self.obj_frame_pred = obj_frame_pred
		self.initializer = initializer

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
			pred_flow: (B (T-1) H W 2). predicted flow. TODO: get rid of.
			slots_t: (B T N S). framewise slots.
			att_t: (B T (h* w*) N). attention maps.
		"""
		del padding_mask # Unused. only here for consistency and easier training code writing.

		B, T, H, W, C = video.shape
		# encode and position embed.
		# flatten over batch * time and encode to get [B, T, h, w, F]
		enc = self.encoder(video.reshape(shape=(B*T, H, W, C)), channels_last=True)
		_, h, w, F = enc.shape
		# add position embedding
		enc = self.pos_embed(enc)
		# change to [B, T, h, w, F]
		enc = enc.reshape(shape=(B, T, H, W, F))

		# initialize slots. [B, N, S] where N = num_slots, S = slot_dim
		slots = self.initializer(
			conditioning, batch_size=B)
		_, N, S = slots.shape

		# apply attn b/w slots and spatio-temporal features
		# attn with inputs as [B (T h w) F]
		slots, _ = self.obj_slot_attn(slots, enc.reshape(shape=(B, T*h*w, F)))

		# slots = (B*T N S)
		slots = slots.repeat_interleave(repeats=T, dim=0)

		# apply attn b/w slots and spatial features
		# attn with inputs as [(B T) (h* w*) F]
		slots_t, att_t = self.obj_slot_attn.compute_attention(
			slots, enc.reshape(shape=(B*T, h*w, F)))

		slots_t = slots_t.reshape(shape=(B, T, N, S))
		att_t = att_t.reshape(shape=(B, T, N, H, W))

		# # get one-step forward slots
		# adjacent_slots = torch.cat([slots_t[:, :-1], slots_t[:, 1:]], dim=-1)
		# # add first slot twice to complete whole sequence.
		# first_slot = torch.cat([slots_t[:, :1], slots_t[:, :1]], dim=-1)
		# adjacent_slots = torch.cat([first_slot, adjacent_slots], dim=1)

		# get object-wise predicted frames and mask
		bb_features, alpha_logits = self.decoder(
			slots_t.reshape(shape=(B*T, N, S)))
		slot_frame_pred = self.obj_frame_pred(
			bb_features.reshape(shape=(B*T*N*H*W, -1)))
		slot_frame_pred = slot_frame_pred.reshape(shape=(B, T, N, H, W, 3))
		alpha_logits = alpha_logits.reshape(shape=(B, T, N, H, W, 1))
		masks = alpha_logits.softmax(dim=2)

		# construct frame by combining masked object-wise predicted frames
		pred_images = (slot_frame_pred * masks).sum(2)

		# make segmentation masks if in eval mode
		if not self.training:
			pred_seg = alpha_logits.argmax(2)
		else:
			pred_seg = None

		# TODO: get rid of flow
		images_flow = alpha_logits[:, :-1, :, :, :, :].sum(2).repeat_interleave(repeats=2, dim=-1)

		return pred_images, pred_seg, images_flow, slots_t, att_t.flatten(3,4).permute(0,1,3,2)


class FlowPrediction(nn.Module):
	"""Video model consisting of encoder, slot attention module, and mask decoder."""

	def __init__(self,
				 encoder: nn.Module,
				 pos_embed: nn.Module,
				 obj_slot_attn: nn.Module,
				 obj_flow_pred: nn.Module,
				 next_frame_pred: nn.Module,
				 initializer: nn.Module,
				 mask_decoder: Optional[nn.Module] = None
				):
		super().__init__()

		self.encoder = encoder
		self.pos_embed = pos_embed
		self.obj_slot_attn = obj_slot_attn
		self.obj_flow_pred = obj_flow_pred
		self.next_frame_pred = next_frame_pred
		self.initializer = initializer
		self.mask_decoder = mask_decoder

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
		del padding_mask # Unused. only here for consistency and easier training code writing.

		B, T, H, W, C = video.shape
		# encode and position embed.
		# flatten over batch * time and encode to get [B, T, h, w, F]
		enc = self.encoder(video.reshape(shape=(B*T, H, W, C)), channels_last=True)
		_, h, w, F = enc.shape
		# add position embedding
		enc = self.pos_embed(enc)
		# change to [B, T, h, w, F]
		enc = enc.reshape(shape=(B, T, H, W, F))

		# initialize slots. [B, N, S] where N = num_slots, S = slot_dim
		slots = self.initializer(
			conditioning, batch_size=B)
		_, N, S = slots.shape

		# apply attn b/w slots and spatio-temporal features
		# attn with inputs as [B (T h w) F]
		slots, _ = self.obj_slot_attn(slots, enc.reshape(shape=(B, T*h*w, F)))

		# slots = (B*T N S)
		slots = slots.repeat_interleave(repeats=T, dim=0)

		# apply attn b/w slots and spatial features
		# attn with inputs as [(B T) (h* w*) F]
		slots_t, att_t = self.obj_slot_attn.compute_attention(
			slots, enc.reshape(shape=(B*T, h*w, F)))

		slots_t = slots_t.reshape(shape=(B, T, N, S))
		att_t = att_t.reshape(shape=(B, T, N, H, W))

		# compute one-step forward flows
		adjacent_slots = torch.cat([slots_t[:, :-1], slots_t[:, 1:]], -1)
		slot_flow_pred = self.obj_flow_pred(
			adjacent_slots.reshape(shape=(B*(T-1)*N, S*2))) * 20
		slot_flow_pred = slot_flow_pred.reshape(shape=(B, T-1, N, 2))

		# find image flow
		if self.mask_decoder is not None:
			# get flow masks from decoder applied on slots
			alpha_logits = self.mask_decoder(slots_t.reshape(shape=(B*T, N, S)))[0]
			alpha_logits = alpha_logits.reshape(shape=(B, T, N, H, W, 1))
			masks = alpha_logits.softmax(dim=2)
		else:
			# use attention masks as flow masks
			masks = att_t.unsqueeze(-1)

		# find image flow by combining masked object-wise flows
		images_flow = (slot_flow_pred[:, :, :, None, None, :]*masks[:, :-1]).sum(2)

		# predict next frames with original frame and predcited flows
		pred_images = self.next_frame_pred(
			video[:, :-1].reshape(shape=(B*(T-1), H, W, C)),
			images_flow.reshape(shape=(B*(T-1), H, W, 2)), 
			channels_last=True)
		pred_images = pred_images.reshape(shape=(B, T-1, H, W, C))
		pred_images = torch.cat([video[:, :1], pred_images], dim=1) # add first frame to complete sequence

		# make segmentation masks if in eval mode
		if not self.training:
			pred_seg = alpha_logits.argmax(2) if self.mask_decoder is not None else masks.argmax(2)
		else:
			pred_seg = None

		return pred_images, pred_seg, images_flow, slots_t, att_t.flatten(3,4).permute(0,1,3,2)

class FlowPredictionOld(nn.Module):
	"""Video model consisting of encoder, slot attention module, and mask decoder."""

	def __init__(self,
				 encoder: nn.Module,
				 decoder: nn.Module,
				 obj_slot_attn: nn.Module,
				 flow_pred: nn.Module,
				 frame_pred: nn.Module,
				 initializer: nn.Module
				):
		super().__init__()

		self.encoder = encoder
		self.decoder = decoder
		self.obj_slot_attn = obj_slot_attn
		self.flow_pred = flow_pred
		self.frame_pred = frame_pred
		self.initializer = initializer

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
		del padding_mask # Unused. only here for consistency and easier training code writing.

		B, T, H, W, C = video.shape
		# encode and position embed.
		# flatten over batch * time and encode to get [B, T, h* w*, F]
		enc = self.encoder(video.reshape(shape=(B*T, H, W, C)))
		_, h, w, F = enc.shape
		enc = enc.reshape(shape=(B, T, h, w, F))
		
		# initialize slots. [B, N, S] where N = num_slots, S = slot_dim
		slots = self.initializer(
			conditioning, batch_size=B)
		_, N, S = slots.shape

		# get attn b/w slots and spatio-temporal features
		# attn with inputs as [B (T h* w*) F]
		slots, _ = self.obj_slot_attn(slots, enc.reshape(shape=(B, T*h*w, F)))

		# slots = ((B T) N S)
		slots = slots.repeat_interleave(T, 0)

		# get attn b/w slots and spatial features
		# attn with inputs as [(B T) (h* w*) F]
		# slots_t, att_t = self.obj_slot_attn(slots, enc.flatten(2, 3).flatten(0, 1))
		slots_t, att_t = self.obj_slot_attn.compute_attention(slots, enc.reshape(shape=(B*T, h*w, F)))

		# slots_t = [B T N S], att_t = (B T (h* w*) N)
		slots_t = slots_t.reshape(shape=(B, T, N, S))
		att_t = att_t.reshape(shape=(B, T, N, (h*w))).permute(0, 1, 3, 2)

		# get objet-wise masks. alpha mask = (B T N H W 1)
		# pred_seg = None
		# if kwargs.get('slice_decode_inputs'):
		# 	# decode over slices to bypass memory constraints
		# 	# just do every timestep separately. (naive)
		# 	alpha_mask = []
		# 	pred_seg = []
		# 	for t in range(T):
		# 		decoded = self.decoder(slots_t[:, t:t+1].reshape(shape=(B, N, S)))
		# 		alpha_mask.append(decoded["alpha_mask"].unsqueeze(1))
		# 		if "segmentations" in decoded:
		# 			pred_seg.append(decoded["segmentations"].unsqueeze(1))
		# 	alpha_mask = torch.cat(alpha_mask, dim=1)
		# 	if len(pred_seg) > 0:
		# 		pred_seg = torch.cat(pred_seg, dim=1)
		# else:
		# 	decoded = self.decoder(slots_t.reshape(shape=(B*T, N, S)))
		# 	alpha_mask = decoded["alpha_mask"].reshape(shape=(B, T, N, H, W, 1))
		# 	if "segmentations" in decoded:
		# 		pred_seg = decoded["segmentations"].reshape(shape=(B, T, H, W, 1))
		# TODO: alpha masks are really bad for some reason.
		# Somehow, all the slots_t are basically the same for each slot.
		# maybe use attention mask instead of alpha masks to debug, like in original model.
		# also, break up model to look more like original model ?
		alpha_mask = att_t.permute(0, 1, 3, 2).reshape(shape=(B,T,N,H,W,1))
		pred_seg = alpha_mask.argmax(2)

		# get predicted flow per object
		adjacent_slots = torch.cat([slots_t[:, :-1], slots_t[:, 1:]], dim=-1)
		# inputs are adjacent slots = ((B (T-1) N) S*2)
		slot_flow_pred = self.flow_pred(adjacent_slots.reshape(-1, S*2)) * 20
		slot_flow_pred = slot_flow_pred.reshape(shape=(B, T-1, N, 2))
		# broadcast and mask flow, combine object flows. (B (T-1) H W 2)
		pred_flow = (slot_flow_pred[:, :, :, None, None, :]*alpha_mask[:, :-1]).sum(2)

		# predict next frames, add first frame because it's not predicted.
		# pred_frames = (B T H W C)
		pred_frames = self.frame_pred(video[:, :-1].flatten(0,1), pred_flow.flatten(0,1), channels_last=True)
		pred_frames = pred_frames.reshape(shape=(B, T-1, H, W, C))
		pred_frames = torch.cat([video[:, :1], pred_frames], dim=1)

		return pred_frames, pred_seg, pred_flow, slots_t, att_t


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