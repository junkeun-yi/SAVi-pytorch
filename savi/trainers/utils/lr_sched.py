# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from torch.optim.lr_scheduler import LambdaLR

import math

def adjust_learning_rate(optimizer, lr, step, warmup_steps):
	"""Decay the learning rate with half-cycle cosine after warmup"""
	if step < warmup_steps:
		lr = args.lr * epoch / args.warmup_epochs 
	else:
		lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
			(1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
	for param_group in optimizer.param_groups:
		if "lr_scale" in param_group:
			param_group["lr"] = lr * param_group["lr_scale"]
		else:
			param_group["lr"] = lr
	return lr

	# TODO:
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
	""" Create a schedule with a learning rate that decreases following the
		values of the cosine function between 0 and `pi * cycles` after a warmup
		period during which it increases linearly between 0 and 1.
	"""
	def lr_lambda(current_step):
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
		return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

	return LambdaLR(optimizer, lr_lambda, last_epoch)