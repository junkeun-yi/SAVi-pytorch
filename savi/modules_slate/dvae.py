"""dVAE.

From https://github.com/singhgautam/slate/blob/master/dvae.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Any, Callable, Dict, Iterable, Mapping, NamedTuple, Optional, Tuple, Union

Array = torch.Tensor
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]

class dVAE(nn.Module):
    """dVAE model for discrete tokenization of video frames."""

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 vocab_size: int = 4096,
                 img_channels: int = 3
                ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.img_channels = img_channels

    def forward(self, inputs: Array):
        """forward pass of the autoencoder"""

        x = inputs
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded

    def encode(self, inputs: Array):
        return self.encoder(inputs)

    def decode(self, inputs: Array):
        return self.decoder(inputs)