"""Model evaluation."""

# TODO: might not have to implement for torch

import functools
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from savi.lib import losses
from savi.lib import utils

Array = torch.Tensor
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]


def get_eval_metrics(
    preds: Dict[str, ArrayTree],
    batch: Dict[str, Array],
    loss_fn: losses.LossFn,
    eval_metrics_cls: Dict,
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
    loss, loss_aux = loss_fn(preds, batch)
    metrics_update = 1
    # TODO
    return 1

def eval_first_step(
    model: nn.Module,
    state_variables: Dict,
    params: Dict[str, ArrayTree],
    batch: Dict[str, Array],
    conditioning_key: Optional[str] = None
) -> Dict[str, ArrayTree]:
    """Get the model predictions with a freshly initialized recurrent state.

    The model is applied to 
    """