"""Return model, loss, and eval metrics in 1 go 
for the Flow-based Frame Prediction Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import savi.modules_flow as modules_flow
import savi.modules as modules

def build_model_old(args):
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
			input_shape=(-1, 64, 64, 32),
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
	# Mask Decoder
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
			input_shape=(-1, 8, 8, slot_size),
			embedding_type="linear",
			update_type="project_add"),
		target_readout=modules.misc.DummyReadout())
	# Flow Predictor
	flow_pred = modules_flow.model.create_mlp(
		input_dim=slot_size*2,
		output_dim=2)
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
		flow_pred=flow_pred,
		obj_slot_attn=obj_slot_attn,
		frame_pred=frame_pred,
		initializer=initializer
	)
	return model

def build_model(args):
	slot_size = 128
	num_slots = args.num_slots
	# Encoder
	encoder = modules.CNN(
		features=[3, 32, 32, 32, 32],
		kernel_size=[(5, 5), (5, 5), (5, 5), (5, 5)],
		strides=[(1, 1), (1, 1), (1, 1), (1, 1)],
		padding="same",
		layer_transpose=[False, False, False, False])
	# Positional Embedding
	pos_embed=modules.PositionEmbedding(
		input_shape=(-1, 64, 64, 32),
		embedding_type="linear",
		update_type="project_add",
		output_transform=modules.MLP(
			input_size=32,
			hidden_size=64,
			output_size=32,
			layernorm="pre"))
	# Object Slot Attention
	obj_slot_attn = modules.SlotAttention(
		input_size=32, # TODO: validate, should be backbone output size
		qkv_size=128,
		slot_size=slot_size,
		num_iterations=3)
	# Object Flow Predictor
	obj_flow_pred = modules_flow.model.create_mlp(
		input_dim=slot_size*2,
		output_dim=2)
	# Next Frame Predictor
	next_frame_pred = modules_flow.FlowWarp()
	# Initializer
	initializer = modules.CoordinateEncoderStateInit(
		embedding_transform=modules.MLP(
			input_size=4, # bounding boxes have feature size 4
			hidden_size=slot_size*2,
			output_size=slot_size,
			layernorm=None),
		prepend_background=True,
		center_of_mass=False)
	# Mask Decoder
	decoder = modules_flow.SpatialBroadcastMaskDecoder(
		resolution=(8,8),
		backbone=modules.CNN(
				features=[slot_size, 64, 64, 64, 64],
				kernel_size=[(5, 5), (5, 5), (5, 5), (5, 5)],
				strides=[(2, 2), (2, 2), (2, 2), (1, 1)],
				padding=[2, 2, 2, "same"],
				transpose_double=True,
				layer_transpose=[True, True, True, True]),
		pos_emb=modules.PositionEmbedding(
			input_shape=(-1, 8, 8, slot_size),
			embedding_type="linear",
			update_type="project_add"))
	mask_decoder = decoder # None
	# Flow Prediction Model
	model = modules_flow.FlowPrediction(
		encoder=encoder,
		pos_embed=pos_embed,
		obj_slot_attn=obj_slot_attn,
		obj_flow_pred=obj_flow_pred,
		next_frame_pred=next_frame_pred,
		initializer=initializer,
		mask_decoder=mask_decoder
	)
	return model


def build_model_frame_pred(args):
	slot_size = 128
	num_slots = args.num_slots
	# Encoder
	encoder = modules.CNN(
		features=[3, 32, 32, 32, 32],
		kernel_size=[(5, 5), (5, 5), (5, 5), (5, 5)],
		strides=[(1, 1), (1, 1), (1, 1), (1, 1)],
		padding="same",
		layer_transpose=[False, False, False, False])
	# Decoder
	decoder = modules_flow.SpatialBroadcastMaskDecoder(
		resolution=(8,8),
		backbone=modules.CNN(
				features=[slot_size, 64, 64, 64, 64],
				kernel_size=[(5, 5), (5, 5), (5, 5), (5, 5)],
				strides=[(2, 2), (2, 2), (2, 2), (1, 1)],
				padding=[2, 2, 2, "same"],
				transpose_double=True,
				layer_transpose=[True, True, True, False]),
		pos_emb=modules.PositionEmbedding(
			input_shape=(-1, 8, 8, slot_size),
			embedding_type="linear",
			update_type="project_add"))
	# Positional Embedding
	pos_embed=modules.PositionEmbedding(
		input_shape=(-1, 64, 64, 32),
		embedding_type="linear",
		update_type="project_add",
		output_transform=modules.MLP(
			input_size=32,
			hidden_size=64,
			output_size=32,
			layernorm="pre"))
	# Object Slot Attention
	obj_slot_attn = modules.SlotAttention(
		input_size=32, # TODO: validate, should be backbone output size
		qkv_size=128,
		slot_size=slot_size,
		num_iterations=3)
	# Object Frame Predictor
	obj_frame_pred = modules_flow.model.create_mlp(
		input_dim=decoder.backbone.features[-1],
		output_dim=3)
	# Initializer
	initializer = modules.CoordinateEncoderStateInit(
		embedding_transform=modules.MLP(
			input_size=4, # bounding boxes have feature size 4
			hidden_size=slot_size*2,
			output_size=slot_size,
			layernorm=None),
		prepend_background=True,
		center_of_mass=False)
	# Flow Prediction Model
	model = modules_flow.FramePrediction(
		encoder=encoder,
		decoder=decoder,
		pos_embed=pos_embed,
		obj_slot_attn=obj_slot_attn,
		obj_frame_pred=obj_frame_pred,
		initializer=initializer
	)
	return model


def build_modules(args):
	"""Return the model and loss/eval processors."""
	# model = build_model(args)
	model = build_model_frame_pred(args)
	loss = modules_flow.L2Loss()
	metrics = modules_flow.ARI()

	return model, loss, metrics