"""Initializers module library."""

# FIXME

import functools
from turtle import forward
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from savi.lib import utils
from savi.modules import misc
from savi.modules import video

Shape = Tuple[int]

DType = Any
Array = torch.Tensor
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]


class ParamStateInit(nn.Module):
    """Fixed, learnable state initialization.

    Note: This module ignores any conditional input (by design).
    """

    def __init__(self,
                 shape: Sequence[int],
                 init_fn: str = "normal"
                ):
        super().__init__()

        if init_fn == "normal":
            self.init_fn = functools.partial(nn.init.normal_, std=1.)
        elif init_fn == "zeros":
            self.init_fn = nn.init.zeros_()
        else:
            raise ValueError(f"Unknown init_fn: {init_fn}")

        self.param = nn.Parameter(torch.empty(size=(shape)))

    def forward(self, inputs: Optional[Array], batch_size: int) -> Array:
        del inputs # Unused.
        self.param = self.init_fn(self.param)
        return utils.broadcast_across_batch(self.param, batch_size=batch_size)


class GaussianStateInit(nn.Module):
    """Random state initialization with zero-mean, unit-variance Gaussian

    Note: This module does not contain any trainable parameters.
        This module also ignores any conditional input (by design).
    """

    def __init__(self,
                 shape: Sequence[int],
                ):
        super().__init__()

        self.shape = shape
    
    def forward(self, inputs: Optional[Array], batch_size: int) -> Array:
        del inputs # Unused.
        # TODO: Use torch generator ?
        return torch.normal(mean=torch.zeros([batch_size] + list(self.shape)))


class SegmentationEncoderStateInit(nn.Module):
    """State init that encodes segmentation masks as conditional input."""
    
    def __init__(self,
                 max_num_slots: int,
                 backbone: nn.Module,
                 pos_emb: nn.Module = nn.Identity(),
                 reduction: Optional[str] = "all_flatten", # Reduce spatial dim by default.
                 output_transform: nn.Module = nn.Identity(),
                 zero_background: bool = False
                ):
        super().__init__()
        
        self.max_num_slots = max_num_slots
        self.backbone = backbone
        self.pos_emb = pos_emb
        self.reduction = reduction
        self.output_transform = output_transform
        self.zero_background = zero_background

        # submodules
        self.encoder = video.FrameEncoder(
            backbone=backbone, pos_emb=pos_emb,
            reduction=reduction, output_transform=output_transform)

    def forward(self, inputs: Array, batch_size: Optional[int]) -> Array:
        del batch_size # Unused.

        # inputs.shape = (batch_size, seq_len, height, width)
        inputs = inputs[:, 0] # Only condition on first time step.

        # Convert mask index to one-hot.
        inputs_oh = F.one_hot(inputs, self.max_num_slots)
        # inputs_oh.shape = (batch_size, height, width, n_slots)
        # NOTE: 0th entry inputs_oh[... 0] will typically correspond to background.

        # Set background slot to all-zeros.
        if self.zero_background:
            inputs_oh[:, :, :, 0] = 0
        
        # Switch one-hot axis into 1st position (i.e. sequence axis).
        inputs_oh = inputs_oh.permute((0, 3, 1, 2))
        # inputs_oh.shape = (batch_size, max_num_slots, height, width)

        # Append dummy feature axis.
        inputs_oh = torch.unsqueeze(-1)

        # encode slots
        # slots.shape = (batch_size, n_slots, n_features)
        slots = self.encoder(inputs_oh, None)

        return slots


class CoordinateEncoderStateInit(nn.Module):
    """State init that encodes bounding box corrdinates as conditional input.

    Attributes:
        embedding_transform: A nn.Module that is applied on inputs (bounding boxes).
        prepend_background: Boolean flag' whether to prepend a special, zero-valued
            background bounding box to the input. Default: False.
        center_of_mass: Boolean flag; whether to convert bounding boxes to center
            of mass coordinates. Default: False.
        background_value: Default value to fill in the background.
    """

    def __init__(self,
                embedding_transform: nn.Module,
                prepend_background: bool = False,
                center_of_mass: bool = False,
                background_value: float = 0.
                ):
        super().__init__()

        self.embedding_transform = embedding_transform
        self.prepend_background = prepend_background
        self.center_of_mass = center_of_mass
        self.background_value = background_value
    
    def forward(self, inputs: Array, batch_size: Optional[int]) -> Array:
        del batch_size # Unused.

        # inputs.shape = (batch_size, seq_len, bboxes, 4)
        inputs = inputs[:, 0] # Only condition on first time step.
        # inputs.shape = (batch_size, bboxes, 4)

        if self.prepend_background:
            # Adds a fake background box [0, 0, 0, 0] at the beginning.
            batch_size = inputs.shape[0]

            # Encode the background as specified by the background_value.
            background = torch.full(
                (batch_size, 1, 4), self.background_value, dtype=inputs.dtype,
                device = inputs.get_device())

            inputs = torch.cat([background, inputs], dim=1)

        if self.center_of_mass:
            y_pos = (inputs[:, :, 0] + inputs[:, :, 2]) / 2
            x_pos = (inputs[:, :, 1] + inputs[:, :, 3]) / 2
            inputs = torch.stack([y_pos, x_pos], dim=-1)

        slots = self.embedding_transform(inputs)

        return slots