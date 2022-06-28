"""Video Slot Attention model based on SLATE.

From https://github.com/singhgautam/slate/blob/master/slate.py
"""


from typing import Any, Callable, Dict, Iterable, Mapping, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from savi.lib import utils
from savi.modules import misc
import savi.modules_steve.utils as stutils

Array = torch.Tensor
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]


class SLATE(nn.Module):
    def __init__(self,
                 num_slots: int,
                 vocab_size: int,
                 d_model: int,
                 dvae: nn.Module,
                 pos_emb: nn.Module,
                 slot_attn: nn.Module,
                 encoder: nn.Module,
                 decoder: nn.Module
                ):
        super().__init__()

        self.num_slots = num_slots
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dvae = dvae
        self.pos_emb = pos_emb
        self.slot_attn = slot_attn
        self.encoder = encoder
        self.decoder = decoder

        # submodules
    
    def forward(self, image, tau, hard):
        """
        Forward pass on a batch of images.

        Args:
            image: [B, H, W, C]
            Tau:
            Hard:

        Returns:
            recons:
            cross_entropy:
            mse:
            attns:
        """

        B, H, W, C = image.shape

        # dvae encode
        _, z_logits, z = self.dvae.encode(image, channels_last=True)

        # dvae recon
        recon = self.dvae.decoder(z, channels_last=True)
        mse = ((image - recon) ** 2).sum() / B

        # hard z
        z_hard = stutils.gumbel_softmax(z_logits, tau, True, dim=-1).detach()

        # target tokens for transformer
        z_transformer_target = z_hard.flatten(1, 2)

        # add BOS token
        z_transformer_input = torch.cat