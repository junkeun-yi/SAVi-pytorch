"""Convolutional module library."""

# FIXME

import functools
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

Shape = Tuple[int]

DType = Any
Array = torch.Tensor
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]

class CNN(nn.Module):
    """Flexible CNN model with conv. and normalization layers."""

    # TODO: add padding ?
    def __init__(self,
                 features: Sequence[int], # FIXME: [in_channels, *out_channels]
                 kernel_size: Sequence[Tuple[int, int]],
                 strides: Sequence[Tuple[int, int]],
                 layer_transpose: Sequence[bool],
                 activation_fn: Callable[[Array], Array] = nn.ReLU,
                 norm_type: Optional[str] = None,
                 axis_name: Optional[str] = None, # Over which axis to aggregate batch stats.
                 output_size: Optional[int] = None
                ):
        super().__init__()

        self.features = features
        self.kernel_size = kernel_size
        self.strides = strides
        self.layer_transpose = layer_transpose
        self.activation_fn = activation_fn
        self.norm_type = norm_type
        self.axis_name = axis_name
        self.output_size = output_size

        # submodules
        num_layers = len(features) - 1 # account for input features (channels)

        assert num_layers >= 1, "Need to have at least one layer."
        assert len(kernel_size) == num_layers, (
            f"len(kernel_size): {len(kernel_size)} and len(features): {len(features)} must match.")
        assert len(strides) == num_layers, (
            f"len(strides): {len(strides)} and len(features): {len(features)} must match.")
        assert len(layer_transpose) == num_layers, (
            f"len(layer_transpose): {len(layer_transpose)} and len(features): {len(features)} must match.")

        if self.norm_type:
            assert self.norm_type in {"batch", "group", "instance", "layer"}, (
                f"({self.norm_type}) is not a valid normalization type")

        # Whether transpose conv or regular conv
        conv_module = {False: nn.Conv2d, True: nn.ConvTranspose2d}

        if self.norm_type == "batch":
            norm_module = functools.partial(nn.BatchNorm2d, momentum=0.9)
        elif self.norm_type == "group":
            norm_module = lambda x: nn.GroupNorm(num_groups=32, num_channels=x)
        elif self.norm_type == "layer":
            norm_module = functools.partial(nn.LayerNorm)
        elif self.norm_type == "instance":
            norm_module = functools.partial(nn.InstanceNorm2d)

        # model
        ## Convnet Architecture.
        self.cnn_layers = nn.ModuleList()
        for i in range(num_layers):

            ### Convolution Layer.
            convname = "convtranspose" if layer_transpose[i] else "conv"
            self.cnn_layers.add_module(
                f"{convname}_{i}",
                conv_module[self.layer_transpose[i]](
                    in_channels=features[i], out_channels=features[i+1],
                    kernel_size=kernel_size[i], stride=strides[i],
                    bias=False if norm_type else True))

            ### Normalization Layer.
            if self.norm_type:
                self.cnn_layers.add_module(
                    f"{self.norm_type}_norm_{i}",
                    norm_module(features[i+1]))

            ### Activation Layer
            self.cnn_layers.add_module(
                f"activ_{i}",
                activation_fn())

        ## Final Dense Layer
        if self.output_size:
            self.project_to_output = nn.Linear(features[-1], self.output_size, bias=True)

    def forward(self, inputs: Array, channels_last=False) -> Tuple[Dict[str, Array]]:
        if channels_last:
            # inputs.shape = (batch_size, height, width, n_channels)
            inputs = inputs.permute((0, 3, 1, 2))
            # inputs.shape = (batch_size, n_channels, height, width)

        x = inputs
        for layer in self.cnn_layers:
            x = layer(x)
        if self.output_size:
            x = self.project_to_output(x)

        if channels_last:
            # x.shape = (batch_size, n_features, h*, w*)
            x = x.permute((0, 3, 1, 2))
            # x.shape = (batch_size, h*, w*, n_features)

        return x