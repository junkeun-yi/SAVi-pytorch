"""Model evaluation."""

# TODO: rename file

from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

Array = torch.Tensor
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]


def get_eval_metrics(
	preds: Dict[str, ArrayTree],
	batch: Dict[str, Array],
	loss_fn,
	eval_metrics_processor,
	predicted_max_num_instances: int,
	ground_truth_max_num_instances: int,
) -> Union[None, Dict]:
	"""Compute the metrics for the model predictions in inference mode.

	The metrics are averaged across *all* devices (of all hosts).

	Args:
		preds: Model predictions.
		batch: Inputs that should be evaluated.
		loss_fn: Loss function that takes model predictions and a batch of data.
		eval_metrics_cls: Dictionary of evaluation metrics.
		predicted_max_num_instances: Maximum number of instances (objects) in prediction.
		ground_truth_max_num_instances: Maximum number of instances in ground truth,
			including background (which counts as a separate instance).

	Returns:
		The evaluation metrics.
	"""
	loss = loss_fn(preds, batch)
	metrics_update = eval_metrics_processor.from_model_output(
		
	)
	# TODO
	return metrics_update

def eval_first_step(
	model: nn.Module,
	batch: Tuple[Array],
	# conditioning_key: Optional[str] = None
) -> Dict[str, ArrayTree]:
	"""Get the model predictions with a freshly initialized recurrent state.

	Args:
		model: Model used in eval step.
		state_variables: State variables for the model.
		params: Params for the model.
		batch: Inputs that should be evaluated.
		conditioning_key: Optional key for conditioning slots.
	Returns:
		The model's predictions.
	"""

	video, boxes, segmentations, gt_flow, padding_mask, mask = batch
	# TODO: delete hardcode
	conditioning = boxes

	preds = model(
		video=video, conditioning=conditioning, 
		padding_mask=padding_mask
	)

	return preds


def eval_continued_step(
	model: nn.Module,
	batch: Tuple[Array],
	recurrent_states: Array
	) -> Dict[str, ArrayTree]:
	"""Get the model predictions, continuing from a provided recurrent state.
	
	Args:
		model: Model used in eval step.
		batch: Inputs that should be evaluated.
		recurrent_states: Recurrent internal model state from which to continue.
			i.e. slots
	Returns:
		The model's predictions.
	"""

	video, boxes, segmentations, gt_flow, padding_mask, mask = batch

	preds = model(
		video=video, conditioning=recurrent_states, 
		continue_from_previous_state=True, padding_mask=padding_mask
	)

	return preds

def batch_slicer(
	batch: Tuple[Array],
	start_idx: int,
	end_idx: int,
	pad_value: int = 0) -> Tuple[Array]:
	"""Slicing the batch along axis 1. (hardcoded)

	Pads when sequence ends before `end`.
	hardcoded parameters included, don't use as a general slicing fn  
	"""
	assert start_idx <= end_idx
	video, boxes, segmentations, gt_flow, padding_mask, mask = batch

	seq_len = video.shape[1]
	# Infer end index if not provided.
	if end_idx == -1:
		end_idx = seq_len
	# Set padding size if end index > sequence length
	pad_size = 0
	if end_idx > seq_len:
		pad_size = end_idx - start_idx
		end_idx = seq_len

	sliced_batch = []
	for array in (video, boxes, segmentations, gt_flow, padding_mask):
		if pad_size > 0:
			# array shape: (B, T, ...)
			pad_shape = list(array.shape[:1]) + [pad_size] + list(array.shape[2:]) # (B, pad, ...)
			padding = torch.full(pad_shape, pad_value)
			item = torch.cat([array[:, start_idx:end_idx], padding], dim=1)
			sliced_batch.append(item)
		else:
			sliced_batch.append(array[:, start_idx:end_idx])
	sliced_batch.append(mask) # hardcoded. only array with shape (B,)
	
	return sliced_batch

def preds_reader(model_outputs):
	"""Hardcoded helper function for eval_step readability"""
	recurrent_states = model_outputs["states_pred"] # [B, T, N, S]
	pred_seg = model_outputs["outputs"]["segmentations"] # [B, T, H, W, 1]
	pred_flow = model_outputs["outputs"]["flow"] # [B, T, H, W, 3]
	att_t = model_outputs["attention"] # [B, T, ?] # TODO: figure this out

	return recurrent_states, pred_seg, pred_flow, att_t


def eval_step(
	model: nn.Module,
	batch: Tuple[Array],
	slice_size: Optional[int] = None
	) -> Tuple[Array]:
	"""Compute the metrics for the given model in inference mode.

	The metrics are averaged across all devices.

	Args:
		model: Model used in eval step
		batch: inputs
		eval_first_step_fn: eval first step fn
		eval_continued_step_fn: eval continued step fn
		slice_size: Optional int, if provided, evaluate model on temporal
			slices of this size instead of full sequence length at once.
	Returns:
		Model predictions (hardcoded)
			pred_seg, pred_flow, att_t
	"""

	video, boxes, segmentations, flow, padding_mask, mask = batch
	temporal_axis = axis = 1

	seq_len = video.shape[axis]
	# Sliced evaluation (i.e. onsmaller temporal slices of the video).
	if slice_size is not None and slice_size < seq_len:
		num_slices = int(np.ceil(seq_len / slice_size))

		# Get predictions for first slice (with fresh recurrrent state (i.e. slots)).
		batch_slice = batch_slicer(batch, 0, slice_size)
		preds_slice = eval_first_step(
			model=model, batch=batch_slice)
		recurrent_states, pred_seg, pred_flow, att_t = preds_reader(preds_slice)
		# make predictions array
		preds = [[item] for item in (pred_seg, pred_flow, att_t)]

		# Iterate over remaining slices (re-using the previous recurrent state).
		for slice_idx in range(1, num_slices):
			batch_slice = batch_slicer(batch,
				start_idx=slice_idx * slice_size,
				end_idx=(slice_idx+1) * slice_size)
			preds_slice = eval_continued_step(
				model, batch_slice, recurrent_states)
			recurrent_states, pred_seg, pred_flow, att_t = preds_reader(preds_slice)
			for i in range(len(preds)):
				preds[i].append((pred_seg, pred_flow, att_t)[i])
		
		# join the predictions
		for i in range(len(preds)):
			preds[i] = torch.cat(preds[i], dim=axis)

	else:
		preds = eval_first_step(model, batch)
		preds = preds_reader(preds)[1:]

	return preds
