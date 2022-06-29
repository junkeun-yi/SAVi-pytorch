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
                 slot_size: int,
                 vocab_size: int,
                 d_model: int,
                 dvae: nn.Module,
                 pos_emb: nn.Module,
                 slot_attn: nn.Module,
                 dictionary: nn.Module,
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
        self.dictionary = dictionary
        self.encoder = encoder
        self.decoder = decoder

        # submodules
        self.slot_proj = nn.Linear(slot_size, d_model, bias=False)
        self.out = nn.Linear(d_model, vocab_size, bias=False)

    
    def forward(self, image, tau, hard):
        """
        Forward pass on a batch of images.

        Args:
            image: [B, H, W, C]
            Tau: gumbel softmax parameter
            Hard: gumbel softmax -> hardmax

        Returns:
            recons:
            cross_entropy:
            mse:
            attns:
        """
        # TODO: disambiguate shapes.

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
        z_transformer_input = torch.cat(
            [torch.zeros_like(z_transformer_target[..., :1]), z_transformer_target], dim=-1)
        z_transformer_input = torch.cat(
            [torch.zeros_like(z_transformer_input[..., :1, :]), z_transformer_input], dim=-2)
        z_transformer_input[:, 0, 0] = 1.0

        # tokens to 


class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        x: B, N, vocab_size
        """

        tokens = torch.argmax(x, dim=-1) # [B, N]
        token_embs = self.dictionary(tokens) # [B, N, emb_size]
        return token_embs