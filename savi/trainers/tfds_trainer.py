# Reference: MAE github https://github.com/facebookresearch/mae

# TODO

import jax
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from typing import Iterable, Optional

import random
import math
import os
import sys
import argparse
import datetime
import time
import json
from pathlib import Path

from savi.datasets.tfds import tfds_input_pipeline
from savi.datasets.tfds.tfds_dataset_wrapper import MOViData, MOViDataByRank
import savi.modules as modules

import savi.lib.losses as losses
import savi.lib.metrics as metrics

import savi.trainers.utils.misc as misc
import savi.trainers.utils.lr_sched as lr_sched
import savi.trainers.utils.lr_decay as lr_decay
from savi.trainers.utils.misc import NativeScalerWithGradNormCount as NativeScaler

def get_args():
	parser = argparse.ArgumentParser('TFDS dataset training for SAVi.')
	def adrg(name, default, type=str, help=None):
		"""ADd aRGuments to parser."""
		if help:
			parser.add_argument(name, default=default, type=type, help=help)
		else:
			parser.add_argument(name, default=default, type=type)
	
	# Training config
	adrg('--seed', 42, int)
	adrg('--batch_size', 8, int, help='Batch size per GPU \
		(effective batch size = batch_size * accum_iter * # gpus')
	# Try to use 8 gpus to get batch size 64, as it is the batch size used in the SAVi code.
	adrg('--epochs', 50, int)
	adrg('--accum_iter', 1, int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
	adrg('--num_train_steps', 100000, int)
	adrg('--device', 'cuda', help='device to use for training / testing')
	adrg('--num_workers', 10, int)

	# Resuming
	parser.add_argument('--resume', default='',
					help='resume from checkpoint')
	
	# distributed training parameters
	parser.add_argument('--world_size', default=1, type=int,
						help='number of distributed processes')
	parser.add_argument('--local_rank', default=-1, type=int)
	parser.add_argument('--dist_on_itp', action='store_true')
	parser.add_argument('--dist_url', default='env://',
						help='url used to set up distributed training')
	
	# Adam optimizer config
	adrg('--lr', 2e-4, float)
	adrg('--warmup_steps', 2500, int)
	adrg('--max_grad_norm', 0.05, float)

	# Logging and Saving config
	adrg('--log_loss_every_step', 50, int)
	adrg('--eval_every_steps', 1000, int)
	adrg('--checkpoint_every_steps', 5000)
	adrg('--output_dir', './output_dir', help="path where to save, empty for no saving.")
	adrg('--log_dir', './output_dir', help="path where to log tensorboard log")

	# Misc
	parser.add_argument('--pin_mem', action='store_true',
					help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')


	# Metrics Spec
	adrg('--metrics', 'loss,ari,ari_nobg')

	# Dataset
	adrg('--tfds_name', "movi_a/128x128:1.0.0", help="Dataset for training/eval")
	adrg('--data_dir', "/home/junkeun-yi/current/datasets/kubric/")
	adrg('--shuffle_buffer_size', 8*8, help="should be batch_size * 8")

	# Model
	adrg('--max_instances', 10, int, help="Number of slots") # For Movi-A,B,C, only up to 10. for MOVi-D,E, up to 23.
	adrg('--model_size', 'small', help="How to prepare data and model architecture.")

	# Evaluation
	adrg('--eval_slice_size', 6, int)
	adrg('--eval_slice_keys', 'video,segmentations,flow,boxes')
	parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
	parser.add_argument('--dist_eval', action='store_true', default=False,
		help='Enabling distributed evaluation (recommended during training for faster monitor')


	args = parser.parse_args()
	# Metrics
	args.train_metrics_spec = {
		v: v for v in args.metrics.split(',')}
	args.eval_metrics_spec = {
		f"eval_{v}": v for v in args.metrics.split(',')}
	# Misc
	args.num_slots = args.max_instances + 1 # only used for metrics
	args.logging_min_n_colors = args.max_instances
	args.eval_slice_keys = [v for v in args.eval_slice_keys.split(',')]

	# HARDCODED
	args.targets = {"flow": 3}
	args.losses = {f"recon_{target}": {"loss_type": "recon", "key": target}
		for target in args.targets}

	# Preprocessing
	if args.model_size =="small":
		args.preproc_train = [
			"video_from_tfds",
			f"sparse_to_dense_annotation(max_instances={args.max_instances})",
			"temporal_random_strided_window(length=6)",
			"resize_small(64)",
			"flow_to_rgb()"  # NOTE: This only uses the first two flow dimensions.
		]
		args.preproc_eval = [
			"video_from_tfds",
			f"sparse_to_dense_annotation(max_instances={args.max_instances})",
			"temporal_crop_or_pad(length=24)",
			"resize_small(64)",
			"flow_to_rgb()"  # NOTE: This only uses the first two flow dimensions.
		]
	
	return args

def build_model(args):
	if args.model_size == "small":
		slot_size = 128
		num_slots = args.num_slots
		# Encoder
		encoder = modules.FrameEncoder(
			backbone=modules.CNN(
				features=[3, 32, 32, 32, 32],
				kernel_size=[(5, 5), (5, 5), (5, 5), (5, 5)],
				strides=[(1, 1), (1, 1), (1, 1), (1, 1)],
				layer_transpose=[False, False, False, False]),
			pos_emb=modules.PositionEmbedding(
				input_shape=(args.batch_size, 4, 4, 32), # TODO: validate, should be backbone output size
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
				layer_transpose=[True, True, True, False]),
			pos_emb=modules.PositionEmbedding(
				input_shape=(args.batch_size, 8, 8, 128),
				embedding_type="linear",
				update_type="project_add"),
			target_readout=modules.Readout(
				keys=list(args.targets),
				readout_modules=nn.ModuleList([
					nn.Linear(64, out_features) for out_features in args.targets.values()])))
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
	return model

def build_datasets(args):
	rng = jax.random.PRNGKey(args.seed)
	train_ds, eval_ds = tfds_input_pipeline.create_datasets(args, rng)

	num_tasks = misc.get_world_size()
	global_rank = misc.get_rank()

	traindata = MOViDataByRank(train_ds, global_rank, num_tasks)
	evaldata = MOViDataByRank(eval_ds, global_rank, num_tasks)

	return traindata, evaldata

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
					data_loader: Iterable, optimizer: torch.optim.Optimizer,
					device: torch.device, epoch: int, loss_scaler, global_step, max_norm: float = 0,
					log_writer=None, args=None):
	model.train(True)
	metric_logger = misc.MetricLogger(delimiter="  ")
	metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f'))
	header = 'Epoch: [{}]'.format(epoch)
	print_freq = args.log_loss_every_step

	accum_iter = args.accum_iter

	optimizer.zero_grad()

	if log_writer is not None:
		print('log_dir: {}'.format(log_writer.log_dir))

	# TODO: only first epoch has scheduler, and does step-wise scheduling
	if epoch == 0:
		scheduler = lr_sched.get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.num_train_steps, num_cycles=1, last_epoch=-1)
	else:
		scheduler = None

	for data_iter_step, (video, boxes, flow, padding_mask, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
		
		if global_step % args.eval_every_steps:
			# TODO: evaluate
			pass

		if global_step % args.checkpoint_every_steps:
			# TODO: checkpoint
			pass

		# SAVi doesn't train on epochs, just on steps.
		if global_step > args.num_train_steps:
			break

		# need to squeeze because of weird dataset wrapping ...
		video = video.squeeze(0).to(device, non_blocking=True)
		boxes = boxes.squeeze(0).to(device, non_blocking=True)
		flow = flow.squeeze(0).to(device, non_blocking=True)
		padding_mask = padding_mask.squeeze(0).to(device, non_blocking=True)
		# segmentations = segmentations.squeeze(0).to(device, non_blocking=True)

		print('video', video.shape, end='\r')

		conditioning = boxes # TODO: make this not hardcoded

		with torch.cuda.amp.autocast():
			outputs = model(video=video, conditioning=conditioning, 
				padding_mask=padding_mask)
			loss = criterion(outputs["outputs"]["flow"], flow)

		loss_value = loss.item()

		if not math.isfininte(loss_value):
			print("Loss is {}, stopping training".format(loss_value))
			sys.exit(1)
		
		loss /= accum_iter
		loss_scaler(loss, optimizer, clip_grad=max_norm,
					parameters=model.parameters(), create_graph=False,
					update_grad=(data_iter_step + 1) % accum_iter == 0)
		if (data_iter_step + 1) % accum_iter == 0:
			optimizer.zero_grad()
			if scheduler is not None:
				scheduler.step()
		
		torch.cuda.synchronize()

		metric_logger.update(loss=loss_value)
		
		lr = optimizer.param_groups[0]["lr"]
		metric_logger.update(lr=lr)

		loss_value_reduce = misc.all_reduce_mean(loss_value)
		if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
			""" We use epoch_1000x as the x-axis in tensorboard.
			This calibrates different curves when batch size changes.
			"""
			epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
			log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
			log_writer.add_scalar('lr', lr, epoch_1000x)

		global_step += 1
	
	# gather the stats from all processes
	metric_logger.synchronize_between_processes()
	print("Averaged stats:", metric_logger)
	return global_step, {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
	loss = losses.recon_loss
	ari = metrics.adjusted_rand_index


	metric_logger = misc.MetricLogger(delimiter="  ")
	header = 'Test:'

	# switch to evaluation mode
	model.eval()

	for (video, boxes, flow, padding_mask, segmentations) in metric_logger.log_every(data_loader, 10, header):
		video = video.squeeze(0).to(device, non_blocking=True)
		boxes = boxes.squeeze(0).to(device, non_blocking=True)
		flow = flow.squeeze(0).to(device, non_blocking=True)
		padding_mask = padding_mask.squeeze(0).to(device, non_blocking=True)
		segmentations = segmentations.squeeze(0).to(device, non_blocking=True)

		conditioning = boxes # TODO: don't hardcode

		# compute output
		with torch.cuda.amp.autocast():
			outputs = model(video=video, conditioning=conditioning, 
				padding_mask=padding_mask)
			loss = loss(outputs["outputs"]["flow"], flow)
			ari_bg = ari(pred_ids=outputs["outputs"]["segmentations"],
						 true_ids=segmentations, num_instances_pred=args.num_slots,
						 num_instances_true=args.max_instances + 1, # add bg,
						 padding_mask=padding_mask, ignore_background=False)
			ari_nobg = ari(pred_ids=outputs["outputs"]["segmentations"],
						 true_ids=segmentations, num_instances_pred=args.num_slots,
						 num_instances_true=args.max_instances + 1, # add bg,
						 padding_mask=padding_mask, ignore_background=True)

		# TODO: change tensors to numpy before doing calculations.
		# TODO: update meters with number of items according to given by metrics fn

		batch_size = video.shape[0]
		metric_logger.update(loss=loss.item())
		metric_logger.meters['ari'].update(ari_bg, n=batch_size)
		metric_logger.meters['ari_nobg'].update(ari_nobg, n=batch_size)
	
	# gather the stats from all processes
	metric_logger.synchronize_between_processes()
	print('* ARI {ari_bg.global_avg:.3f} ARI_NoBg {ari_nobg.global_avg:.3f} loss {losses.global_avg:.3f}'
		  .format(ari_bg=metric_logger.ari, ari_nobg=metric_logger.ari_nobg, losses=metric_logger.loss))

	# switch back to training
	model.train()

	return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def run(args):
	misc.init_distributed_mode(args)

	print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
	print("{}".format(args).replace(', ', ',\n'))

	device = torch.device(args.device)

	# fix the seed for reproducibility
	seed = args.seed + misc.get_rank()
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	cudnn.benchmark = True

	dataset_train, dataset_val = build_datasets(args)

	if True: # args.distributed:
		num_tasks = misc.get_world_size()
		global_rank = misc.get_rank()
		# sampler_train = torch.utils.data.DistributedSampler(
		# 	dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False)
		sampler_train = torch.utils.data.SequentialSampler(dataset_train)
		# print("Sampler_train")
		if args.dist_eval:
			if len(dataset_val) % num_tasks != 0:
				print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
					  'This will slightly alter validation results as extra duplicate entries are added to achieve '
					  'equal num of samples per-process.')
			# sampler_val = torch.utils.data.DistributedSampler(
			# 	dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
			sampler_val = torch.utils.data.SequentialSampler(dataset_val)
		else:
			sampler_val = torch.utils.data.SequentialSampler(dataset_val)
	else:
		sampler_train = torch.utils.data.RandomSampler(dataset_train)
		sampler_val = troch.utils.data.SequentialSampler(dataset_val)

	if global_rank == 0 and args.log_dir is not None:
		os.makedirs(args.log_dir, exist_ok=True)
		log_writer = SummaryWriter(log_dir=args.log_dir)
	else:
		log_writer = None
	
	data_loader_train = torch.utils.data.DataLoader(
		dataset_train, sampler=sampler_train,
		batch_size=1, # HARDCODED because doing something weird with this.
		num_workers=args.num_workers,
		pin_memory=args.pin_mem,
		drop_last=True
	)

	data_loader_val = torch.utils.data.DataLoader(
		dataset_val, sampler=sampler_val,
		batch_size=1, # HARDCODED because doing something weird with this.
		num_workers=args.num_workers,
		pin_memory=args.pin_mem,
		drop_last=True
	)

	# Model setup
	model = build_model(args)

	# TODO: make checkpoint loading

	model.to(device)

	model_without_ddp = model
	n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

	# print("Model = %s" % str(model_without_ddp))
	print('number of params (M): %.2f' % (n_parameters / 1.e6))

	eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

	print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
	print("actual lr: %.2e" % args.lr)

	print("accumulate grad iterations: %d" % args.accum_iter)
	print("effective batch size: %d" % eff_batch_size)

	if args.distributed:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
		model_without_ddp = model.module

	# build optimizer
	optimizer = torch.optim.Adam(model_without_ddp.parameters(), lr=args.lr)
	loss_scaler = NativeScaler()

	# Loss
	criterion = losses.recon_loss
	print("criterion = %s" % str(criterion))

	misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

	if args.eval:
		test_stats = evaluate(data_loader_val, model, device)
		print(test_stats)
		exit(0)

	print(f"Start training for {args.num_train_steps} steps.")
	start_time = time.time()
	max_accuracy = 0.0
	global_step = 0
	for epoch in range(0, args.epochs):
		# if args.distributed:
		# 	data_loader_train.sampler.set_epoch(epoch)
		step_add, train_stats = train_one_epoch(
			model, criterion, data_loader_train,
			optimizer, device, epoch, loss_scaler,
			global_step, args.max_grad_norm,
			log_writer, args
		)
		global_step += step_add
		if args.output_dir:
			misc.save_model(
				args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
				loss_scaler=loss_scaler, epoch=epoch)

		test_stats = evaluate(data_loader_val, model, device, args)
		print(test_stats)
		
		# log writer stuff.

def main():
	args = get_args()
	
	if args.output_dir:
		Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	
	run(args)


def test():
	# args = get_args()
	# model = build_model(args)
	# print(model)

	main()


if __name__ == "__main__":
	test()


"""

PYTHONPATH=$PYTHONPATH:./ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 savi/main.py

PYTHONPATH=$PYTHONPATH:./ CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 savi/main.py

PYTHONPATH=$PYTHONPATH:./ CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 savi/main.py

"""