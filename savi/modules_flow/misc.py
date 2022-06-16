"""Losses and Eval Metrics for flow-pred video SA model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.transforms as transforms

import savi.lib.metrics as metrics

#####################################################
# Losses

class L2Loss(nn.Module):
	"""L2 loss."""
	
	def __init__(self, l2_weight=1, reduction="sum"):
		super().__init__()

		self.l2 = nn.MSELoss(reduction=reduction)
		self.l2_weight = l2_weight

	def forward(self, model_outputs, batch):
		# pred_frames, _, _, _, _, = model_outputs
		# pred_frames, _, _, _, _, _= model_outputs
		pred_frames, pred_seg, pred_flow, slots_t, att_t = model_outputs
		gt_frames, boxes, segmentations, flow, padding_mask, mask = batch

		# l2 loss between images and predicted images
		loss = self.l2_weight * self.l2(pred_frames, gt_frames)

		return loss


#######################################################
# Eval Metrics

class ARI(nn.Module):
	"""ARI."""

	def forward(self, model_outputs, batch, args):
		# pred_frames, masks_t, slot_flow_pred, slots_t, att_t = model_outputs
		pred_frames, pred_seg, pred_flow, slots_t, att_t = model_outputs
		video, boxes, segmentations, flow, padding_mask, mask = batch

		pr_seg = pred_seg.squeeze(-1).int().cpu().numpy()
		gt_seg = segmentations.int().cpu().numpy()
		input_pad = padding_mask.cpu().numpy()
		mask = mask.cpu().numpy()

		ari_bg = metrics.Ari.from_model_output(
		# ari_bg = metrics_jax.Ari.from_model_output(
			predicted_segmentations=pr_seg, ground_truth_segmentations=gt_seg,
			padding_mask=input_pad,
			ground_truth_max_num_instances=args.max_instances + 1,
			predicted_max_num_instances=args.num_slots,
			ignore_background=False, mask=mask)
		ari_nobg = metrics.Ari.from_model_output(
		# ari_nobg = metrics_jax.Ari.from_model_output(
			predicted_segmentations=pr_seg, ground_truth_segmentations=gt_seg,
			padding_mask=input_pad,
			ground_truth_max_num_instances=args.max_instances + 1,
			predicted_max_num_instances=args.num_slots,
			ignore_background=True, mask=mask)
		
		return ari_bg, ari_nobg
