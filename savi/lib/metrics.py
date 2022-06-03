"""Clustering metrics."""

# TODO:

from typing import Optional, Sequence, Union, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def check_shape(x, expected_shape: Sequence[Optional[int]], name: str):
    """Check whether shape x is as expected.
    
    Args:
        x: Any data type with `shape` attribute. if `shape` sttribute is not present
            it is assumed to be a scalar with shape ().
        expected shape: The shape that is expected of x. For example,
            [None, None, 3] can be the `expected shape` for a color image,
            [4, None, None, 3] if we know that batch size is 4.
        name: Name of `x` to provide informative error messages.
    Raises: ValueError if x's shape does not match expected_shape. Also raises
        ValueError if expected_shape is not a list or tuple.
    """
    if not isinstance(expected_shape, (list, tuple)):
        raise ValueError(
            "expected_shape should be a list or tuple of ints but got "
            f"{expected_shape}.")
    
    # Scalars have shape () by definition
    shape = getattr(x, "shape", ())

    if (len(shape) != len(expected_shape) or
        any(j is not None and i != j for i, j in zip(shape, expected_shape))):
        raise ValueError(
            f"Input {name} had shape {shape} but {expected_shape} was expected"
        )
    

def _validate_inputs(predicted_segmentations: np.ndarray,
                     ground_truth_segmentations: np.ndarray,
                     padding_mask: np.ndarray,
                     mask: Optional[np.ndarray] = None) -> None:
    """Checks that all inputs have the expected shapes.
    
    Args:
        predicted_segmentations: An array of integers of shape [bs, seq_len, H, W]
            containing model segmentation predictions.
        ground_truth_segmentations: An array of integers of shape [bs, seq_len, H, W]
            containing ground truth segmentations.
        padding_mask: An array of integers of shape [bs, seq_len, H, W] defining
            regions where the ground truth is meaningless, for example because this
            corresponds to regions which were padded during data augmentation.
            Value 0 corresponds to padded regions, 1 corresponds to valid regions to
            be used for metric calculation.
        mask: An optional array of boolean mask values of shape [bs]. `True`
            corresponds to actual batch examples whereas `False` corresponds to padding.
            TODO: what exactly is this ?
    
    Raises:
        ValueError if the inputs are not valid.
    """

    check_shape(
        predicted_segmentations, [None, None, None, None],
        "predicted_segmentations[bs, seq_len, h, w]")
    check_shape(
        ground_truth_segmentations, [None, None, None, None],
        "ground_truth_segmentations [bs, seq_len, h, w]")
    check_shape(
        predicted_segmentations, ground_truth_segmentations.shape,
        "predicted_segmentations [should match ground_truth_segmentations]")
    check_shape(
        padding_mask, ground_truth_segmentations.shape,
        "padding_mask [should match ground_truth_segmentations]")
    
    if not np.issubdtype(predicted_segmentations.dtype, np.integer):
        raise ValueError("predicted_segmentations has to be integer-valued. "
                         "Got {}".format(predicted_segmentations.dtype))
    
    if not np.issubdtype(ground_truth_segmentations.dtype, np.integer):
        raise ValueError("ground_truth_segmentations has to be integer-valued. "
                         "Got {}".format(ground_truth_segmentations.dtype))
    
    if not np.issubdtype(padding_mask.dtype, np.integer):
        raise ValueError("padding_mask has to be integer_valued. "
                         "Got {}".format(padding_mask.dtype))
    
    if mask is not None:
        check_shape(mask, [None], "mask [bs]")
        if not np.issubdtype(mask.dtype, np.bool_):
            raise ValueError("mask has to be boolean. Got {}".format(mask.dtype))


def adjusted_rand_index(true_ids: np.ndarray, pred_ids: np.ndarray,
                        num_instances_true: int, num_instances_pred: int,
                        padding_mask: Optional[np.ndarray] = None,
                        ignore_background: bool = False) -> np.ndarray:
    """Computes the adjusted Rand Index (ARI), a clustering similarity score.
    
    Args:
        true_ids: An integer-valued array of shape
            [bs, seq_len, H, W]. The true cluster assignment encoded as integer ids.
        pred_ids: An integer-valued array of shape
            [bs, seq_len, H, W]. The predicted cluster assignment encoder as integer ids.
        num_instances_true: An integer, the number of instances in true_ids
            (i.e. max(true_ids) + 1).
        num_instances_pred: An integer, the number of instances in true_ids
            (i.e. max(pred_ids) + 1).
        padding_mask: An array of integers of shape [bs, seq_len, H, W] defining regions
            where the ground truth is meaningless, for example because this corresponds to
            regions which were padded during data augmentation. Value 0 corresponds to
            padded regions, 1 corresponds to valid regions to be used for metric calculation.
        ignore_background: Boolean, if True, then ignore all pixels where true_ids == 0 (default: False).
        
    Returns:
        ARI scores as a float32 array of shape [bs].
    """
    
    true_oh = F.one_hot(torch.from_numpy(true_ids).long(), num_instances_true)
    pred_oh = F.one_hot(torch.from_numpy(pred_ids).long(), num_instances_pred)
    if padding_mask is not None:
        true_oh = true_oh * padding_mask[..., None]
    
    if ignore_background:
        true_oh = true_oh[..., 1:] # remove the background row

    N = torch.einsum("bthwc,bthwk->bck", true_oh, pred_oh)
    A = torch.sum(N, dim=-1) # row-sum (bs, c)
    B = torch.sum(N, dim=-2) # col-sum (bs, k)
    num_points = torch.sum(A, dim=1)

    rindex = torch.sum(N * (N - 1), dim=1).sum(dim=1)
    aindex = torch.sum(A * (A - 1), dim=1)
    bindex = torch.sum(B * (B - 1), dim=1)
    expected_rindex = aindex * bindex / torch.clip(num_points * (num_points-1), 1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator

    # There are two cases for which the denominator can be zero:
    # 1. If both label_pred and label_true assign all pixels to a single cluster.
    #    (max_rindex == expected_rindex == rindex == num_points * (num_points-1))
    # 2. If both label_pred and label_true assign max 1 point to each cluster.
    #    (max_rindex == expected_rindex == rindex == 0)
    # In both cases, we want the ARI score to be 1.0:
    # return torch.where(denominator, ari, 1.0)
    return torch.where(denominator > 0, ari.double(), 1.0)

class Ari():
    """Adjusted Rand Index (ARI) computed from predictions and labels.
    
    ARI is a similarity score to compare two clusterings. ARI returns values in
    the range [-1, 1], where 1 corresponds to two identical clusterings (up to
    permutation), i.e. a perfect match between the predicted clustering and the 
    ground-truth clustering. A value of (close to) 0 corresponds to chance.
    Negative values corresponds to cases where the agreement between the
    clusterings is less than expected from a random assignment.
    In this implementations, we use ARI to compare predicted instance segmentation
    masks (including background prediction) with ground-trueht segmentation
    annotations.
    """

    @staticmethod
    def from_model_output(predicted_segmentations: np.ndarray,
                          ground_truth_segmentations: np.ndarray,
                          padding_mask: np.ndarray,
                          ground_truth_max_num_instances: int,
                          predicted_max_num_instances: int,
                          ignore_background: bool= False,
                          mask: Optional[np.ndarray] = None,
                          **_):
        """Computation of the ARI clustering metric.
        
        NOTE: This implementation does not currently support padding masks.
        Args:
            predicted_segmentations: An array of integers of shape
                [bs, seq_len, H, W] containing model segmentation predictions.
            ground_truth_segmentations: An array of integers of shape
                [bs, seq_len, H, W] containing ground truth segmentations.
            padding_mask: An array of integers of shape [bs, seq_len, H, W]
                defining regions where the ground truth is meaningless, for example
                because this corresponds to regions which were padded during data
                augmentation. Value 0 corresponds to padded regions, 1 corresponds to
                valid regions to be used for metric calculation.
            ground_truth_max_num_instances: Maximum number of instances (incl.
                background, which counts as the 0-th instance) possible in the dataset.
            predicted_max_num_instances: Maximum number of predicted instances (incl.
                background).
            ignore_background: If True, then ignore all pixels where
                ground_truth_segmentations == 0 (default: False).
            mask: An optional array of boolean mask values of shape [bs]. `True`
                corresponds to actual batch examples whereas `False` corresponds to
                padding.
        
        Returns:
            Object of Ari with computed intermediate values.
        """
        _validate_inputs(predicted_segmentations=predicted_segmentations,
            ground_truth_segmentations=ground_truth_segmentations,
            padding_mask=padding_mask,
            mask=mask)
        
        batch_size = predicted_segmentations.shape[0]

        if mask is None:
            mask = np.ones(batch_size, dtype=padding_mask.dtype)
        else:
            mask = np.asarray(mask, dtype=padding_mask.dtype)
        
        ari_batch = adjusted_rand_index(
            pred_ids=predicted_segmentations,
            true_ids=ground_truth_segmentations,
            num_instances_true=ground_truth_max_num_instances,
            num_instances_pred=predicted_max_num_instances,
            padding_mask=padding_mask,
            ignore_background=ignore_background)
        
        # return cls(total=torch.sum(ari_batch * mask), count=torch.sum(mask))
        return {'total': torch.sum(ari_batch * mask), 'count': np.sum(mask)}

class AriNoBg(Ari):
    """Adjusted Rand Index (ARI), ignoring the ground-truth background label."""

    @classmethod
    def from_model_output(cls, **kwargs):
        """See `Ari` dostring for allowed keyword arguments."""
        return super().from_model_output(**kwargs, ignore_background=True)