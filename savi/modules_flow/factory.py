"""Return model, loss, and eval metrics in 1 go 
for the Flow-based Frame Prediction Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import savi.modules_flow as modules_flow
import savi.modules as modules

def build_model(args):
	slot_size = 128
	num_slots = args.num_slots
	# Encoder
	encoder = modules.FrameEncoder(
		backbone=modules.CNN(
			features=[3, 32, 32, 32, 32],
			kernel_size=[(5, 5), (5, 5), (5, 5), (5, 5)],
			strides=[(1, 1), (1, 1), (1, 1), (1, 1)],
			padding="same",
			layer_transpose=[False, False, False, False]),
		pos_emb=modules.PositionEmbedding(
			input_shape=(args.batch_size, 64, 64, 32),
			embedding_type="linear",
			update_type="project_add",
			output_transform=modules.MLP(
				input_size=32,
				hidden_size=64,
				output_size=32,
				layernorm="pre")))
	# Object Slot Attention
	obj_slot_attn = modules.SlotAttention(
		input_size=32, # TODO: validate, should be backbone output size
		qkv_size=128,
		slot_size=slot_size,
		num_iterations=1)
	# Mask and Flwo Decoder
	decoder = modules.SpatialBroadcastDecoder(
		resolution=(8,8), # Update if data resolution or strides change.
		backbone=modules.CNN(
			features=[slot_size, 64, 64, 64, 64],
			kernel_size=[(5, 5), (5, 5), (5, 5), (5, 5)],
			strides=[(2, 2), (2, 2), (2, 2), (1, 1)],
			padding=[2, 2, 2, "same"],
			transpose_double=True,
			layer_transpose=[True, True, True, False]),
		pos_emb=modules.PositionEmbedding(
			input_shape=(args.batch_size, 8, 8, 128),
			embedding_type="linear",
			update_type="project_add"),
		target_readout=modules.Readout(
			keys=["flow"],
			readout_modules=nn.ModuleList([
				nn.Linear(64, 2)])))
	# Frame Predictor
	frame_pred = modules_flow.FlowWarp()
	# Initializer
	initializer = modules.CoordinateEncoderStateInit(
		embedding_transform=modules.MLP(
			input_size=4, # bounding boxes have feature size 4
			hidden_size=256,
			output_size=slot_size,
			layernorm=None),
		prepend_background=True,
		center_of_mass=False)
	# Flow Prediction Model
	model = modules_flow.FlowPrediction(
		encoder=encoder,
		decoder=decoder,
		obj_slot_attn=obj_slot_attn,
		frame_pred=frame_pred,
		initializer=initializer
	)
	return model


def build_modules(args):
	"""Return the model and loss/eval processors."""
	model = build_model(args)	
	loss = modules_flow.L2Loss()
	metrics = modules_flow.ARI()

	return model, loss, metrics