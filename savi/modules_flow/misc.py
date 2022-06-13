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
		predicted_frames, _, _, _, _ = model_outputs
		gt_frames, _, _, _, _ = batch

		# l2 loss between images and predicted images
		loss = self.l2_weight * self.l2(predicted_frames, gt_frames)

		return loss


#######################################################
# Eval Metrics

class ARI(nn.Module):
	"""ARI."""

	def forward(self, model_outputs, batch, args):
		pred_frames, masks_t, slot_flow_pred, slots_t, att_t = model_outputs
		video, boxes, flow, padding_mask, segmentations = batch

		pr_seg = masks_t.squeeze(-1).int().cpu().numpy()
		gt_seg = segmentations.int().cpu().numpy()
		input_pad = padding_mask.cpu().numpy()

		ari_bg = metrics.Ari.from_model_output(
			predicted_segmentations=pr_seg, ground_truth_segmentations=gt_seg,
			predicted_max_num_instances=args.num_slots,
			ground_truth_max_num_instances=args.max_instances + 1,
			padding_mask=input_pad, ignore_background=False)
		ari_nobg = metrics.Ari.from_model_output(
			predicted_segmentations=pr_seg, ground_truth_segmentations=gt_seg,
			predicted_max_num_instances=args.num_slots,
			ground_truth_max_num_instances=args.max_instances + 1,
			padding_mask=input_pad, ignore_background=True)
		
		return ari_bg, ari_nobg
