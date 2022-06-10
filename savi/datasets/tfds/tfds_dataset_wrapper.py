"""Try to wrap TFDS dataset."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import jax

from torch.utils.data import Dataset

import os

from savi.datasets.tfds import tfds_input_pipeline

# MoVi dataset
class MOViData(Dataset):
    def __init__(self, tfds_dataset):
        self.dataset = tfds_dataset
        self.itr = iter(self.dataset)
        # TODO: check if running iter(self.dataset) always returns the same data
    
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

class MOViDataByRank(Dataset):
    def __init__(self, tfds_dataset, rank, world_size):
        self.dataset = tfds_dataset
        self.rank = rank
        self.world_size = world_size
        
        self.reset_itr()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        print('hello', self.rank, self.world_size)
        for _ in range(self.world_size):
            # move by stride
            next(self.itr)
        
        print('retrieving')
        print(next(self.itr))
        batch = jax.tree_map(np.asarray, next(self.itr))

        video = torch.from_numpy(batch['video']) # (B T H W 3)
        boxes = torch.from_numpy(batch['boxes']) # (B T maxN 4)
        flow = torch.from_numpy(batch['flow']) # (B T H W 3)
        padding_mask = torch.from_numpy(batch['padding_mask'])
        segmentations = torch.from_numpy(batch['segmentations'])

        print('video', video.shape)

        return video, boxes, flow, padding_mask, segmentations

    def reset_itr(self):
        # move itr by rank steps to return strided data
        self.itr = iter(self.dataset)
        for _ in range(self.rank):
            next(self.itr)