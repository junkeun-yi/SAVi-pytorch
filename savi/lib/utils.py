"""Common utils."""

# TODO:

import functools
import importlib
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt
import skimage.transform

from savi.lib import metrics

Array = Union[np.ndarray, torch.Tensor] # FIXME:
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
DictTree = Dict[str, Union[Array, "DictTree"]]  # pytype: disable=not-supported-yet
PRNGKey = Array
ConfigAttr = Any
MetricSpec = Dict[str, str]

class TrainState:
    """Data structure for checkpointing the model."""
    step: int
    optimizer: torch.optim.Optimizer
    variables: torch.nn.parameter.Parameter
    rng: int # FIXME: seed ?


# TODO: not sure what to do with this
METRIC_TYPE_TO_CLS = {
    "loss": Any,
    "ari": metrics.Ari,
    "ari_nobg": metrics.AriNoBg
}

# FIXME: make metrics collection just a dictionary
def make_metrics_collection(metrics_spec: Optional[MetricSpec]) -> Dict[str, Any]:
    metrics_dict = {}
    if metrics_spec:
        for m_name, m_type in metrics_spec.items():
            metrics_dict[m_name] = METRIC_TYPE_TO_CLS[m_type]
    
    return metrics_dict


def _flatten_dict(xs, is_leaf=None, sep=None):
  assert isinstance(xs, dict), 'expected (frozen)dict'

  def _key(path):
    if sep is None:
      return path
    return sep.join(path)

  def _flatten(xs, prefix):
    if not isinstance(xs, dict) or (
        is_leaf and is_leaf(prefix, xs)):
      return {_key(prefix): xs}
    result = {}
    is_empty = True
    for key, value in xs.items():
      is_empty = False
      path = prefix + (key,)
      result.update(_flatten(value, path))
    return result
  return _flatten(xs, ())
        
def flatten_named_dicttree(metrics_res: DictTree, sep: str = "/"):
    """Flatten dictionary."""
    metrics_res_flat = {}
    for k, v in _flatten_dict(metrics_res).items():
        metrics_res_flat[(sep.join(k)).strip(sep)] = v
    return metrics_res_flat


# def clip_grads(grad_tree: ArrayTree, max_norm: float, epsilon: float = 1e-6):
#     """Gradient clipping with epsilon.
    
#     """

def lecun_uniform_(tensor, gain=1.):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(1.0 / float(fan_in))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return nn.init._no_grad_uniform_(tensor, -a, a)


def lecun_normal_(tensor, gain=1.):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * (math.sqrt(1.0 / float(fan_in)) / .87962566103423978)
    # return nn.init._no_grad_normal_(tensor, 0., std)
    return torch.nn.init._no_grad_trunc_normal_(tensor, 0, std, -2, 2)

init_fn = {
    'xavier_uniform': nn.init.xavier_uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'kaiming_normal': nn.init.kaiming_normal_,
    'lecun_uniform': lecun_uniform_,
    'lecun_normal': lecun_normal_,
    'ones': nn.init.ones_,
    'zeros': nn.init.zeros_,
    'default': lambda x: x}
def init_param(name, gain=1.):
    assert name in init_fn.keys(), "not a valid init method"
    # return init_fn[name](tensor, gain)
    return functools.partial(init_fn[name], gain=gain)

def spatial_broadcast(x: torch.Tensor, resolution: Sequence[int]) -> Array:
    """Broadcast flat inputs to a 2D grid of a given resolution."""
    x = x[:, None, None, :]
    # return np.tile(x, [1, resolution[0], resolution[1], 1])
    return torch.tile(x, [1, resolution[0], resolution[1], 1])

def broadcast_across_batch(inputs: Array, batch_size: int) -> Array:
  """Broadcasts inputs across a batch of examples (creates new axis)."""
  return torch.broadcast_to(
      torch.unsqueeze(0),
      size=(batch_size,) + inputs.shape)

# def time_distributed(cls, in_axes=1, axis=1):

def create_gradient_grid(
    samples_per_dim: Sequence[int], value_range: Sequence[float] = (-1.0, 1.0)
    ) -> Array:
    """Creates a tensor with equidistant entries from -1 to +1 in each dim
    
    Args:
        samples_per_dim: Number of points to have along each dimension.
        value_range: In each dimension, points will go from range[0] to range[1]
    
    Returns:
        A tensor of shape [samples_per_dim] + [len(samples_per_dim)].
    """

    s = [np.linspace(value_range[0], value_range[1], n) for n in samples_per_dim]
    pe = np.stack(np.meshgrid(*s, sparse=False, indexing="ij"), axis=-1)
    return np.array(pe)

def convert_to_fourier_features(inputs: Array, basis_degree: int) -> Array:
    """Convert inputs to Fourier features, e.g. for positional encoding."""

    # inputs.shape = (..., n_dims).
    # inputs should be in range [-pi, pi] or [0, 2pi].
    n_dims = inputs.shape[-1]

    # Generate frequency basis
    freq_basis = np.concatenate( # shape = (n_dims, n_dims * basis_degree)
        [2**i * np.eye(n_dims) for i in range(basis_degree)], 1)
    
    # x.shape = (..., n_dims * basis_degree)
    x = inputs @ freq_basis # Project inputs onto frequency basis.

    # Obtain Fourier feaures as [sin(x), cos(x)] = [sin(x), sin(x + 0.5 * pi)].
    return np.sin(np.concatenate([x, x + 0.5 * np.pi], axis=-1))