"""Input pipeline for TFDS datasets."""

# FIXME

import functools
from typing import Dict, List, Tuple

from clu import deterministic_data
from clu import preprocess_spec

import ml_collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from savi.datasets.tfds import tfds_preprocessing as preprocessing

import tensorflow as tf
import tensorflow_datasets as tfds

Array = torch.Tensor
PRNGKey = Array

def preprocess_example(features: Dict[str, tf.Tensor],
					   preprocess_strs: List[str]) -> Dict[str, tf.Tensor]:
	"""Process a single data example.

	Args:
		features: A dictionary containing the tensors of a single data example.
		preprocess_strs: List of strings, describing one preprocessing operation
			each, in clu.preprocess_spec format.

	Returns:
		Dictionary containing the preprocessed tensors of a single data example.
	"""
	all_ops = preprocessing.all_ops()
	preprocess_fn = preprocess_spec.parse("|".join(preprocess_strs), all_ops)
	return preprocess_fn(features)


# def get_batch_dims(gloabl_batch_size: int) -> List[int]:
#     """Gets the first two axis sizes for data batches.
	
	
#     """

def create_datasets(
	args,
	data_rng: PRNGKey) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
	"""Create datasets for training and evaluation

	For the same data_rng and config this will return the same datasets.
	The datasets only contain stateless operations.

	Args:
		args: Configuration to use.
		data_rng: JAX PRNGKey for dataset pipeline.

	Returns:
		A tuple with the training dataset and the evaluation dataset.
	"""
	dataset_builder = tfds.builder(
		args.tfds_name, data_dir=args.data_dir)
	
	batch_dims = (args.batch_size,)

	train_preprocess_fn = functools.partial(
		preprocess_example, preprocess_strs=args.preproc_train)
	eval_preprocess_fn = functools.partial(
		preprocess_example, preprocess_strs=args.preproc_eval)

	train_split_name = "train" # args.get("train_split", "train")
	eval_split_name = "validation" # args.get("validation_split", "validation")

	# TODO: may need to do something to only run on one host
	train_split = deterministic_data.get_read_instruction_for_host(
		train_split_name, dataset_info=dataset_builder.info)
	train_ds = deterministic_data.create_dataset(
		dataset_builder,
		split=train_split,
		rng=data_rng,
		preprocess_fn=train_preprocess_fn,
		cache=False,
		shuffle_buffer_size=args.shuffle_buffer_size,
		batch_dims=batch_dims,
		num_epochs=None,
		shuffle=True)

	eval_split = deterministic_data.get_read_instruction_for_host(
		eval_split_name, dataset_info=dataset_builder.info, drop_remainder=False)
	eval_ds = deterministic_data.create_dataset(
		dataset_builder,
		split=eval_split,
		rng=None,
		preprocess_fn=eval_preprocess_fn,
		cache=False,
		batch_dims=batch_dims,
		num_epochs=1,
		shuffle=False,
		pad_up_to_batches="auto")

	return train_ds, eval_ds