"""Return model, loss, and eval metrics in 1 go 
for the SAVi model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import savi.modules as modules
import savi.modules_steve as modules_steve
from savi.modules_steve import (conv2d, Conv2dBlock)


def build_model(args):
	if args.model_size == "small":
		slot_size = 128
		num_slots = args.num_slots
		vocab_size = 4096
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
		# Corrector
		corrector = modules.SlotAttention(
			input_size=32, # TODO: validate, should be backbone output size
			qkv_size=128,
			slot_size=slot_size,
			num_iterations=1)
		# Predictor
		predictor = modules.TransformerBlock(
			embed_dim=slot_size,
			num_heads=4,
			qkv_size=128,
			mlp_size=256)
		# Initializer
		initializer = modules.CoordinateEncoderStateInit(
			embedding_transform=modules.MLP(
				input_size=4, # bounding boxes have feature size 4
				hidden_size=256,
				output_size=slot_size,
				layernorm=None),
			prepend_background=True,
			center_of_mass=False)
		# Decoder
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
			target_readout=modules.Readout(
				keys=list(args.targets),
				readout_modules=nn.ModuleList([
					nn.Linear(64, out_features) for out_features in args.targets.values()])))
		
		
		
		# dVAE
		# for architecture, see appendix p.17 of https://arxiv.org/pdf/2205.14065.pdf
		dvae = modules_steve.dVAE(
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
		
		
		
		
		
		
		# SAVi Model
		model = modules.SAVi(
			encoder=encoder,
			decoder=decoder,
			corrector=corrector,
			predictor=predictor,
			initializer=initializer,
			decode_corrected=True,
			decode_predicted=False)
	else:
		raise NotImplementedError
	# for name, param in model.named_parameters():
	# 	if 'bias' in name:
	# 		nn.init.zeros_(param)
	return model


def build_modules(args):
	"""Return the model and loss/eval processors."""
	model = build_model(args)	
	# loss = modules_flow.L2Loss()
	# metrics = modules_flow.ARI()

	return model#, loss, metrics