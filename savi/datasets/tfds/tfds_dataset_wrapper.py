"""Try to wrap TFDS dataset."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import jax

from torch.utils.data import Dataset

from savi.datasets.tfds import tfds_input_pipeline

# MoVi dataset
class MOViData(Dataset):
    def __init__(self, tfds_dataset):
        self.dataset = tfds_dataset
        self.itr = iter(self.dataset)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        batch = jax.tree_map(np.asarray, next(self.itr))

        video = torch.from_numpy(batch['video']) # (B T H W 3)
        boxes = torch.from_numpy(batch['boxes']) # (B T maxN 4)
        flow = torch.from_numpy(batch['flow']) # (B T H W 3)
        padding_mask = torch.from_numpy(batch['padding_mask'])
        segmentations = torch.from_numpy(batch['segmentations'])

        return video, boxes, flow, padding_mask, segmentations
    
    def reset_itr(self):
        self.itr = iter(self.dataset)