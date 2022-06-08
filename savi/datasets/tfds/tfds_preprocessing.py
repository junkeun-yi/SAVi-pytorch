# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Video preprocessing ops for TFDS datasets."""

# FIXME

import abc
import dataclasses
from typing import Optional, Sequence, Tuple, Union

from absl import logging
from clu import preprocess_spec

import numpy as np
import tensorflow as tf

Features = preprocess_spec.Features
all_ops = lambda: preprocess_spec.get_all_ops(__name__)
SEED_KEY = preprocess_spec.SEED_KEY
NOTRACK_BOX = (0., 0., 0., 0.)  # No-track bounding box for padding.
NOTRACK_LABEL = -1

IMAGE = "image"
VIDEO = "video"
SEGMENTATIONS = "segmentations"
RAGGED_SEGMENTATIONS = "ragged_segmentations"
SPARSE_SEGMENTATIONS = "sparse_segmentations"
SHAPE = "shape"
PADDING_MASK = "padding_mask"
RAGGED_BOXES = "ragged_boxes"
BOXES = "boxes"
FRAMES = "frames"
FLOW = "flow"
DEPTH = "depth"
ORIGINAL_SIZE = "original_size"
INSTANCE_LABELS = "instance_labels"
INSTANCE_MULTI_LABELS = "instance_multi_labels"


def convert_uint16_to_float(array, min_val, max_val):
  return tf.cast(array, tf.float32) / 65535. * (max_val - min_val) + min_val


def get_resize_small_shape(original_size: Tuple[tf.Tensor, tf.Tensor],
                           small_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
  h, w = original_size
  ratio = (
      tf.cast(small_size, tf.float32) / tf.cast(tf.minimum(h, w), tf.float32))
  h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
  w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)
  return h, w


def adjust_small_size(original_size: Tuple[tf.Tensor, tf.Tensor],
                      small_size: int, max_size: int) -> int:
  """Computes the adjusted small size to ensure large side < max_size."""
  h, w = original_size
  min_original_size = tf.cast(tf.minimum(w, h), tf.float32)
  max_original_size = tf.cast(tf.maximum(w, h), tf.float32)
  if max_original_size / min_original_size * small_size > max_size:
    small_size = tf.cast(tf.floor(
        max_size * min_original_size / max_original_size), tf.int32)
  return small_size


def crop_or_pad_boxes(boxes: tf.Tensor, top: int, left: int, height: int,
                      width: int, h_orig: tf.Tensor, w_orig: tf.Tensor):
  """Transforms the relative box coordinates according to the frame crop.

  Note that, if height/width are larger than h_orig/w_orig, this function
  implements the equivalent of padding.

  Args:
    boxes: Tensor of bounding boxes with shape (..., 4).
    top: Top of crop box in absolute pixel coordinates.
    left: Left of crop box in absolute pixel coordinates.
    height: Height of crop box in absolute pixel coordinates.
    width: Width of crop box in absolute pixel coordinates.
    h_orig: Original image height in absolute pixel coordinates.
    w_orig: Original image width in absolute pixel coordinates.
  Returns:
    Boxes tensor with same shape as input boxes but updated values.
  """
  # Video track bound boxes: [num_instances, num_tracks, 4]
  # Image bounding boxes: [num_instances, 4]
  assert boxes.shape[-1] == 4
  seq_len = tf.shape(boxes)[0]
  has_tracks = len(boxes.shape) == 3
  if has_tracks:
    num_tracks = boxes.shape[1]
  else:
    assert len(boxes.shape) == 2
    num_tracks = 1

  # Transform the box coordinates.
  a = tf.cast(tf.stack([h_orig, w_orig]), tf.float32)
  b = tf.cast(tf.stack([top, left]), tf.float32)
  c = tf.cast(tf.stack([height, width]), tf.float32)
  boxes = tf.reshape(
      (tf.reshape(boxes, (seq_len, num_tracks, 2, 2)) * a - b) / c,
      (seq_len, num_tracks, len(NOTRACK_BOX)))

  # Filter the valid boxes.
  boxes = tf.minimum(tf.maximum(boxes, 0.0), 1.0)
  if has_tracks:
    cond = tf.reduce_all((boxes[:, :, 2:] - boxes[:, :, :2]) > 0.0, axis=-1)
    boxes = tf.where(cond[:, :, tf.newaxis], boxes, NOTRACK_BOX)
  else:
    boxes = tf.reshape(boxes, (seq_len, 4))

  return boxes


def flow_tensor_to_rgb_tensor(motion_image, flow_scaling_factor=50.):
  """Visualizes flow motion image as an RGB image.

  Similar as the flow_to_rgb function, but with tensors.

  Args:
    motion_image: A tensor either of shape [batch_sz, height, width, 2] or of
      shape [height, width, 2]. motion_image[..., 0] is flow in x and
      motion_image[..., 1] is flow in y.
    flow_scaling_factor: How much to scale flow for visualization.

  Returns:
    A visualization tensor with same shape as motion_image, except with three
    channels. The dtype of the output is tf.uint8.
  """

  hypot = lambda a, b: (a ** 2.0 + b ** 2.0) ** 0.5  # sqrt(a^2 + b^2)

  height, width = motion_image.get_shape().as_list()[-3:-1]  # pytype: disable=attribute-error  # allow-recursive-types
  scaling = flow_scaling_factor / hypot(height, width)
  x, y = motion_image[..., 0], motion_image[..., 1]
  motion_angle = tf.atan2(y, x)
  motion_angle = (motion_angle / np.math.pi + 1.0) / 2.0
  motion_magnitude = hypot(y, x)
  motion_magnitude = tf.clip_by_value(motion_magnitude * scaling, 0.0, 1.0)
  value_channel = tf.ones_like(motion_angle)
  flow_hsv = tf.stack([motion_angle, motion_magnitude, value_channel], axis=-1)
  flow_rgb = tf.image.convert_image_dtype(
      tf.image.hsv_to_rgb(flow_hsv), tf.uint8)
  return flow_rgb


def get_paddings(image_shape: tf.TensorShape,
                 size: Union[int, Tuple[int, int], Sequence[int]],
                 pre_spatial_dim: Optional[int] = None,
                 allow_crop: bool = True) -> tf.Tensor:
  """Returns paddings tensors for tf.pad operation.

  Args:
    image_shape: The shape of the Tensor to be padded. The shape can be
      [..., N, H, W, C] or [..., H, W, C]. The paddings are computed for H, W
      and optionally N dimensions.
    size: The total size for the H and W dimensions to pad to.
    pre_spatial_dim: Optional, additional padding dimension before the spatial
      dimensions. It is only used if given and if len(shape) > 3.
    allow_crop: If size is bigger than requested max size, padding will be
      negative. If allow_crop is true, negative padding values will be set to 0.

  Returns:
    Paddings the given tensor shape.
  """
  assert image_shape.shape.rank == 1
  if isinstance(size, int):
    size = (size, size)
  h, w = image_shape[-3], image_shape[-2]
  # Spatial padding.
  paddings = [
      tf.stack([0, size[0] - h]),
      tf.stack([0, size[1] - w]),
      tf.stack([0, 0])
  ]
  ndims = len(image_shape)  # pytype: disable=wrong-arg-types
  # Prepend padding for temporal dimension or number of instances.
  if pre_spatial_dim is not None and ndims > 3:
    paddings = [[0, pre_spatial_dim - image_shape[-4]]] + paddings
  # Prepend with non-padded dimensions if available.
  if ndims > len(paddings):
    paddings = [[0, 0]] * (ndims - len(paddings)) + paddings
  if allow_crop:
    paddings = tf.maximum(paddings, 0)
  return tf.stack(paddings)


@dataclasses.dataclass
class VideoFromTfds:
  """Standardize features coming from TFDS video datasets."""

  video_key: str = VIDEO
  segmentations_key: str = SEGMENTATIONS
  ragged_segmentations_key: str = RAGGED_SEGMENTATIONS
  shape_key: str = SHAPE
  padding_mask_key: str = PADDING_MASK
  ragged_boxes_key: str = RAGGED_BOXES
  boxes_key: str = BOXES
  frames_key: str = FRAMES
  instance_multi_labels_key: str = INSTANCE_MULTI_LABELS
  flow_key: str = FLOW
  depth_key: str = DEPTH

  def __call__(self, features: Features) -> Features:

    features_new = {}

    if "rng" in features:
      features_new[SEED_KEY] = features.pop("rng")

    if "instances" in features:
      features_new[self.ragged_boxes_key] = features["instances"]["bboxes"]
      features_new[self.frames_key] = features["instances"]["bbox_frames"]
      if "segmentations" in features["instances"]:
        features_new[self.ragged_segmentations_key] = tf.cast(
            features["instances"]["segmentations"][..., 0], tf.int32)

      # Special handling of CLEVR (https://arxiv.org/abs/1612.06890) objects.
      if ("color" in features["instances"] and
          "shape" in features["instances"] and
          "material" in features["instances"]):
        color = tf.cast(features["instances"]["color"], tf.int32)
        shape = tf.cast(features["instances"]["shape"], tf.int32)
        material = tf.cast(features["instances"]["material"], tf.int32)
        features_new[self.instance_multi_labels_key] = tf.stack(
            (color, shape, material), axis=-1)

    if "segmentations" in features:
      features_new[self.segmentations_key] = tf.cast(
          features["segmentations"][..., 0], tf.int32)

    if "depth" in features:
      # Undo float to uint16 scaling
      depth_range = features["metadata"]["depth_range"]
      features_new[self.depth_key] = convert_uint16_to_float(
          features["depth"], depth_range[0], depth_range[1])

    if "flows" in features:
      # Some datasets use "flows" instead of "flow" for optical flow.
      features["flow"] = features["flows"]
    if "backward_flow" in features:
      # By default, use "backward_flow" if available.
      features["flow"] = features["backward_flow"]
      features["metadata"]["flow_range"] = features["metadata"][
          "backward_flow_range"]
    if "flow" in features:
      # Undo float to uint16 scaling
      flow_range = features["metadata"].get("flow_range", (-255, 255))
      features_new[self.flow_key] = convert_uint16_to_float(
          features["flow"], flow_range[0], flow_range[1])

    # Convert video to float and normalize.
    video = features["video"]
    assert video.dtype == tf.uint8  # pytype: disable=attribute-error  # allow-recursive-types
    video = tf.image.convert_image_dtype(video, tf.float32)
    features_new[self.video_key] = video

    # Store original video shape (e.g. for correct evaluation metrics).
    features_new[self.shape_key] = tf.shape(video)

    # Store padding mask
    features_new[self.padding_mask_key] = tf.cast(
        tf.ones_like(video)[..., 0], tf.uint8)

    return features_new


@dataclasses.dataclass
class AddTemporalAxis:
  """Lift images to videos by adding a temporal axis at the beginning.

  We need to distinguish two cases because `image_ops.py` uses
  ORIGINAL_SIZE = [H,W] and `video_ops.py` uses SHAPE = [T,H,W,C]:
  a) The features are fed from image ops: ORIGINAL_SIZE is converted
    to SHAPE ([H,W] -> [1,H,W,C]) and removed from the features.
    Typical use case: Evaluation of GV image tasks in a video setting. This op
    is added after the image preprocessing in order not to change the standard
    image preprocessing.
  b) The features are fed from video ops: The image SHAPE is lifted to a video
    SHAPE ([H,W,C] -> [1,H,W,C]).
    Typical use case: Training using images in a video setting. This op is added
    before the video preprocessing in order not to change the standard video
    preprocessing.
  """

  image_key: str = IMAGE
  video_key: str = VIDEO
  boxes_key: str = BOXES
  padding_mask_key: str = PADDING_MASK
  segmentations_key: str = SEGMENTATIONS
  sparse_segmentations_key: str = SPARSE_SEGMENTATIONS
  shape_key: str = SHAPE
  original_size_key: str = ORIGINAL_SIZE

  def __call__(self, features: Features) -> Features:
    assert self.image_key in features

    features_new = {}
    for k, v in features.items():
      if k == self.image_key:
        features_new[self.video_key] = v[tf.newaxis]
      elif k in (self.padding_mask_key, self.boxes_key, self.segmentations_key,
                 self.sparse_segmentations_key):
        features_new[k] = v[tf.newaxis]
      elif k == self.original_size_key:
        pass  # See comment in the docstring of the class.
      else:
        features_new[k] = v

    if self.original_size_key in features:
      # The features come from an image preprocessing pipeline.
      shape = tf.concat([[1], features[self.original_size_key],
                         [features[self.image_key].shape[-1]]],  # pytype: disable=attribute-error  # allow-recursive-types
                        axis=0)
    elif self.shape_key in features:
      # The features come from a video preprocessing pipeline.
      shape = tf.concat([[1], features[self.shape_key]], axis=0)
    else:
      shape = tf.shape(features_new[self.video_key])
    features_new[self.shape_key] = shape

    if self.padding_mask_key not in features_new:
      features_new[self.padding_mask_key] = tf.cast(
          tf.ones_like(features_new[self.video_key])[..., 0], tf.uint8)

    return features_new


@dataclasses.dataclass
class SparseToDenseAnnotation:
  """Converts the sparse to a dense representation.

  Returns the following fields:
    - `video`: A dense tensor of shape (number of frames, height, width, 3).
    - `boxes`: Converts the tracks to a dense tensor of shape
      (number of annotated frames, `max_instances` tracks, 4).
    - `segmentations`: If sparse segmentations are available, they are converted
      to a dense segmentation tensor of shape (#frames, height, width, 1) with
      integers reaching from 0 (background) to `max_instances`.
  """

  max_instances: int = 10

  video_key: str = VIDEO
  ragged_boxes_key: str = RAGGED_BOXES
  boxes_key: str = BOXES
  frames_key: str = FRAMES
  ragged_segmentations_key: str = RAGGED_SEGMENTATIONS
  segmentations_key: str = SEGMENTATIONS
  padding_mask_key: str = PADDING_MASK
  instance_labels_key: str = INSTANCE_LABELS
  instance_multi_labels_key: str = INSTANCE_MULTI_LABELS

  def __call__(self, features: Features) -> Features:

    def crop_or_pad(t, size, value, allow_crop=True):
      pad_size = tf.maximum(size - tf.shape(t)[0], 0)
      t = tf.pad(
          t, ((0, pad_size),) + ((0, 0),) * (len(t.shape) - 1),  # pytype: disable=attribute-error  # allow-recursive-types
          constant_values=value)
      if allow_crop:
        t = t[:size]
      return t

    updated_keys = {
        self.video_key, self.frames_key, self.ragged_boxes_key,
        self.ragged_segmentations_key, self.segmentations_key,
        self.instance_labels_key, self.instance_multi_labels_key
    }
    features_new = {k: v for k, v in features.items() if k not in updated_keys}

    frames = features[self.frames_key]
    frames_dense = frames.to_tensor(default_value=0)  # pytype: disable=attribute-error  # allow-recursive-types
    video = features[self.video_key]
    features_new[self.video_key] = video
    num_frames = tf.shape(video)[0]
    num_tracks = tf.shape(frames_dense)[0]

    # Densify segmentations.
    if self.ragged_segmentations_key in features:
      segmentations = features[self.ragged_segmentations_key]
      dense_segmentations = tf.zeros_like(features[self.padding_mask_key],
                                          tf.int32)

      def densify_segmentations(dense_segmentations, vals):
        """Densify non-overlapping segmentations."""
        frames, segmentations, idx = vals
        return tf.tensor_scatter_nd_add(dense_segmentations, frames[:,
                                                                    tf.newaxis],
                                        segmentations * idx)

      # We can safely convert the RaggedTensors to dense as all zero values are
      # ignored due to the aggregation via scatter_nd_add. We also crop to the
      # maximum number of tracks.
      scan_tuple = (crop_or_pad(frames_dense, self.max_instances, 0),
                    crop_or_pad(
                        segmentations.to_tensor(default_value=0),  # pytype: disable=attribute-error  # allow-recursive-types
                        self.max_instances,
                        0), tf.range(1, self.max_instances + 1))

      features_new[self.segmentations_key] = tf.scan(densify_segmentations,
                                                     scan_tuple,
                                                     dense_segmentations)[-1]
    elif self.segmentations_key in features:
      # Dense segmentations are available for this dataset. It may be that
      # max_instances < max(features_new[self.segmentations_key]). We prune out
      # extra objects here.
      segmentations = features[self.segmentations_key]
      segmentations = tf.where(
          tf.less_equal(segmentations, self.max_instances), segmentations, 0)
      features_new[self.segmentations_key] = segmentations

    # Densify boxes.
    bboxes = features[self.ragged_boxes_key]

    def densify_boxes(n):
      boxes_n = tf.tensor_scatter_nd_update(
          tf.tile(tf.constant(NOTRACK_BOX)[tf.newaxis], (num_frames, 1)),
          frames[n][:, tf.newaxis], bboxes[n])
      return boxes_n

    boxes = tf.map_fn(
        densify_boxes,
        tf.range(tf.minimum(num_tracks, self.max_instances)),
        fn_output_signature=tf.float32)
    boxes = tf.transpose(
        crop_or_pad(boxes, self.max_instances, NOTRACK_BOX[0]), (1, 0, 2))
    features_new[self.boxes_key] = tf.ensure_shape(
        boxes, (None, self.max_instances, len(NOTRACK_BOX)))

    # Labels.
    if self.instance_labels_key in features:
      labels = crop_or_pad(features[self.instance_labels_key],
                           self.max_instances, NOTRACK_LABEL)
      features_new[self.instance_labels_key] = tf.ensure_shape(
          labels, (self.max_instances,))

    # Multi-labels.
    if self.instance_multi_labels_key in features:
      multi_labels = crop_or_pad(features[self.instance_multi_labels_key],
                                 self.max_instances, NOTRACK_LABEL)
      features_new[self.instance_multi_labels_key] = tf.ensure_shape(
          multi_labels, (self.max_instances, multi_labels.get_shape()[1]))

    # Frames.
    features_new[self.frames_key] = frames

    return features_new


class VideoPreprocessOp(abc.ABC):
  """Base class for all video preprocess ops."""

  video_key: str = VIDEO
  segmentations_key: str = SEGMENTATIONS
  padding_mask_key: str = PADDING_MASK
  boxes_key: str = BOXES
  flow_key: str = FLOW
  depth_key: str = DEPTH
  sparse_segmentations_key: str = SPARSE_SEGMENTATIONS

  def __call__(self, features: Features) -> Features:
    # Get current video shape.
    video_shape = tf.shape(features[self.video_key])
    # Assemble all feature keys that the op should be applied on.
    all_keys = [
        self.video_key, self.segmentations_key, self.padding_mask_key,
        self.flow_key, self.depth_key, self.sparse_segmentations_key,
        self.boxes_key
    ]
    # Apply the op to all features.
    for key in all_keys:
      if key in features:
        features[key] = self.apply(features[key], key, video_shape)
    return features

  @abc.abstractmethod
  def apply(self, tensor: tf.Tensor, key: str,
            video_shape: tf.TensorShape) -> tf.Tensor:
    """Returns the transformed tensor.

    Args:
      tensor: Any of a set of different video modalites, e.g video, flow,
        bounding boxes, etc.
      key: a string that indicates what feature the tensor represents so that
        the apply function can take that into account.
      video_shape: The shape of the video (which is necessary for some
        transformations).
    """
    pass


class RandomVideoPreprocessOp(VideoPreprocessOp):
  """Base class for all random video preprocess ops."""

  def __call__(self, features: Features) -> Features:
    if features.get(SEED_KEY) is None:
      logging.warning(
          "Using random operation without seed. To avoid this "
          "please provide a seed in feature %s.", SEED_KEY)
      op_seed = tf.random.uniform(shape=(2,), maxval=2**32, dtype=tf.int64)
    else:
      features[SEED_KEY], op_seed = tf.unstack(
          tf.random.experimental.stateless_split(features[SEED_KEY]))
    # Get current video shape.
    video_shape = tf.shape(features[self.video_key])
    # Assemble all feature keys that the op should be applied on.
    all_keys = [
        self.video_key, self.segmentations_key, self.padding_mask_key,
        self.flow_key, self.depth_key, self.sparse_segmentations_key,
        self.boxes_key
    ]
    # Apply the op to all features.
    for key in all_keys:
      if key in features:
        features[key] = self.apply(features[key], op_seed, key, video_shape)
    return features

  @abc.abstractmethod
  def apply(self, tensor: tf.Tensor, seed: tf.Tensor, key: str,
            video_shape: tf.TensorShape) -> tf.Tensor:
    """Returns the transformed tensor.

    Args:
      tensor: Any of a set of different video modalites, e.g video, flow,
        bounding boxes, etc.
      seed: A random seed.
      key: a string that indicates what feature the tensor represents so that
        the apply function can take that into account.
      video_shape: The shape of the video (which is necessary for some
        transformations).
    """
    pass


@dataclasses.dataclass
class ResizeSmall(VideoPreprocessOp):
  """Resizes the smaller (spatial) side to `size` keeping aspect ratio.

  Attr:
    size: An integer representing the new size of the smaller side of the input.
    max_size: If set, an integer representing the maximum size in terms of the
      largest side of the input.
  """

  size: int
  max_size: Optional[int] = None

  def apply(self, tensor, key=None, video_shape=None):
    """See base class."""

    # Boxes are defined in normalized image coordinates and are not affected.
    if key == self.boxes_key:
      return tensor

    if key in (self.padding_mask_key, self.segmentations_key):
      tensor = tensor[..., tf.newaxis]
    elif key == self.sparse_segmentations_key:
      tensor = tf.reshape(tensor,
                          (-1, tf.shape(tensor)[2], tf.shape(tensor)[3], 1))

    h, w = tf.shape(tensor)[1], tf.shape(tensor)[2]

    # Determine resize method based on dtype (e.g. segmentations are int).
    if tensor.dtype.is_integer:
      resize_method = "nearest"
    else:
      resize_method = "bilinear"

    # Clip size to max_size if needed.
    small_size = self.size
    if self.max_size is not None:
      small_size = adjust_small_size(
          original_size=(h, w), small_size=small_size, max_size=self.max_size)
    new_h, new_w = get_resize_small_shape(
        original_size=(h, w), small_size=small_size)
    tensor = tf.image.resize(tensor, [new_h, new_w], method=resize_method)

    # Flow needs to be rescaled according to the new size to stay valid.
    if key == self.flow_key:
      scale_h = tf.cast(new_h, tf.float32) / tf.cast(h, tf.float32)
      scale_w = tf.cast(new_w, tf.float32) / tf.cast(w, tf.float32)
      scale = tf.reshape(tf.stack([scale_h, scale_w], axis=0), (1, 2))
      # Optionally repeat scale in case both forward and backward flow are
      # stacked in the last dimension.
      scale = tf.repeat(scale, tf.shape(tensor)[-1] // 2, axis=0)
      scale = tf.reshape(scale, (1, 1, 1, tf.shape(tensor)[-1]))
      tensor *= scale

    if key in (self.padding_mask_key, self.segmentations_key):
      tensor = tensor[..., 0]
    elif key == self.sparse_segmentations_key:
      tensor = tf.reshape(tensor, (video_shape[0], -1, new_h, new_w))

    return tensor


@dataclasses.dataclass
class CentralCrop(VideoPreprocessOp):
  """Makes central (spatial) crop of a given size.

  Attr:
    height: An integer representing the height of the crop.
    width: An (optional) integer representing the width of the crop. Make square
      crop if width is not provided.
  """

  height: int
  width: Optional[int] = None

  def apply(self, tensor, key=None, video_shape=None):
    """See base class."""
    if key == self.boxes_key:
      width = self.width or self.height
      h_orig, w_orig = video_shape[1], video_shape[2]
      top = (h_orig - self.height) // 2
      left = (w_orig - width) // 2
      tensor = crop_or_pad_boxes(tensor, top, left, self.height,
                                 width, h_orig, w_orig)
      return tensor
    else:
      if key in (self.padding_mask_key, self.segmentations_key):
        tensor = tensor[..., tf.newaxis]
      seq_len, n_channels = tensor.get_shape()[0], tensor.get_shape()[3]
      h_orig, w_orig = tf.shape(tensor)[1], tf.shape(tensor)[2]
      width = self.width or self.height
      crop_size = (seq_len, self.height, width, n_channels)
      top = (h_orig - self.height) // 2
      left = (w_orig - width) // 2
      tensor = tf.image.crop_to_bounding_box(tensor, top, left, self.height,
                                             width)
      tensor = tf.ensure_shape(tensor, crop_size)
      if key in (self.padding_mask_key, self.segmentations_key):
        tensor = tensor[..., 0]
      return tensor


@dataclasses.dataclass
class CropOrPad(VideoPreprocessOp):
  """Spatially crops or pads a video to a specified size.

  Attr:
    height: An integer representing the new height of the video.
    width: An integer representing the new width of the video.
    allow_crop: A boolean indicating if cropping is allowed.
  """

  height: int
  width: int
  allow_crop: bool = True

  def apply(self, tensor, key=None, video_shape=None):
    """See base class."""
    if key == self.boxes_key:
      # Pad and crop the spatial dimensions.
      h_orig, w_orig = video_shape[1], video_shape[2]
      if self.allow_crop:
        # After cropping, the frame shape is always [self.height, self.width].
        height, width = self.height, self.width
      else:
        # If only padding is performed, the frame size is at least
        # [self.height, self.width].
        height = tf.maximum(h_orig, self.height)
        width = tf.maximum(w_orig, self.width)
      tensor = crop_or_pad_boxes(
          tensor,
          top=0,
          left=0,
          height=height,
          width=width,
          h_orig=h_orig,
          w_orig=w_orig)
      return tensor
    elif key == self.sparse_segmentations_key:
      seq_len = tensor.get_shape()[0]
      paddings = get_paddings(
          tf.shape(tensor[..., tf.newaxis]), (self.height, self.width),
          allow_crop=self.allow_crop)[:-1]
      tensor = tf.pad(tensor, paddings, constant_values=0)
      if self.allow_crop:
        tensor = tensor[..., :self.height, :self.width]
      tensor = tf.ensure_shape(
          tensor, (seq_len, None, self.height, self.width))
      return tensor
    else:
      if key in (self.padding_mask_key, self.segmentations_key):
        tensor = tensor[..., tf.newaxis]
      seq_len, n_channels = tensor.get_shape()[0], tensor.get_shape()[3]
      paddings = get_paddings(
          tf.shape(tensor), (self.height, self.width),
          allow_crop=self.allow_crop)
      tensor = tf.pad(tensor, paddings, constant_values=0)
      if self.allow_crop:
        tensor = tensor[:, :self.height, :self.width, :]
      tensor = tf.ensure_shape(tensor,
                               (seq_len, self.height, self.width, n_channels))
      if key in (self.padding_mask_key, self.segmentations_key):
        tensor = tensor[..., 0]
      return tensor


@dataclasses.dataclass
class RandomCrop(RandomVideoPreprocessOp):
  """Gets a random (width, height) crop of input video.

  Assumption: Height and width are the same for all video-like modalities.

  Attr:
    height: An integer representing the height of the crop.
    width: An integer representing the width of the crop.
  """

  height: int
  width: int

  def apply(self, tensor, seed, key=None, video_shape=None):
    """See base class."""
    if key == self.boxes_key:
      # We copy the random generation part from tf.image.stateless_random_crop
      # to generate exactly the same offset as for the video.
      crop_size = (video_shape[0], self.height, self.width, video_shape[-1])
      size = tf.convert_to_tensor(crop_size, tf.int32)
      limit = video_shape - size + 1
      offset = tf.random.stateless_uniform(
          tf.shape(video_shape), dtype=tf.int32, maxval=tf.int32.max,
          seed=seed) % limit
      tensor = crop_or_pad_boxes(tensor, offset[1], offset[2], self.height,
                                 self.width, video_shape[1], video_shape[2])
      return tensor
    elif key == self.sparse_segmentations_key:
      raise NotImplementedError("Sparse segmentations aren't supported yet")
    else:
      if key in (self.padding_mask_key, self.segmentations_key):
        tensor = tensor[..., tf.newaxis]
      seq_len, n_channels = tensor.get_shape()[0], tensor.get_shape()[3]
      crop_size = (seq_len, self.height, self.width, n_channels)
      tensor = tf.image.stateless_random_crop(tensor, size=crop_size, seed=seed)
      tensor = tf.ensure_shape(tensor, crop_size)
      if key in (self.padding_mask_key, self.segmentations_key):
        tensor = tensor[..., 0]
      return tensor


@dataclasses.dataclass
class DropFrames(VideoPreprocessOp):
  """Subsamples a video by skipping frames.

  Attr:
    frame_skip: An integer representing the subsampling frequency of the video,
      where 1 means no frames are skipped, 2 means every other frame is skipped,
      and so forth.
  """

  frame_skip: int

  def apply(self, tensor, key=None, video_shape=None):
    """See base class."""
    del key
    del video_shape
    tensor = tensor[::self.frame_skip]
    new_length = tensor.get_shape()[0]
    tensor = tf.ensure_shape(tensor, [new_length] + tensor.get_shape()[1:])
    return tensor


@dataclasses.dataclass
class TemporalCropOrPad(VideoPreprocessOp):
  """Crops or pads a video in time to a specified length.

  Attr:
    length: An integer representing the new length of the video.
    allow_crop: A boolean, specifying whether temporal cropping is allowed. If
      False, will throw an error if length of the video is more than "length"
  """

  length: int
  allow_crop: bool = True

  def _apply(self, tensor, constant_values):
    frames_to_pad = self.length - tf.shape(tensor)[0]
    if self.allow_crop:
      frames_to_pad = tf.maximum(frames_to_pad, 0)
    tensor = tf.pad(
        tensor, ((0, frames_to_pad),) + ((0, 0),) * (len(tensor.shape) - 1),
        constant_values=constant_values)
    tensor = tensor[:self.length]
    tensor = tf.ensure_shape(tensor, [self.length] + tensor.get_shape()[1:])
    return tensor

  def apply(self, tensor, key=None, video_shape=None):
    """See base class."""
    del video_shape
    if key == self.boxes_key:
      constant_values = NOTRACK_BOX[0]
    else:
      constant_values = 0
    return self._apply(tensor, constant_values=constant_values)


@dataclasses.dataclass
class TemporalRandomWindow(RandomVideoPreprocessOp):
  """Gets a random slice (window) along 0-th axis of input tensor.

  Pads the video if the video length is shorter than the provided length.

  Assumption: The number of frames is the same for all video-like modalities.

  Attr:
    length: An integer representing the new length of the video.
  """

  length: int

  def _apply(self, tensor, seed, constant_values):
    length = tf.minimum(self.length, tf.shape(tensor)[0])
    frames_to_pad = tf.maximum(self.length - tf.shape(tensor)[0], 0)
    window_size = tf.concat(([length], tf.shape(tensor)[1:]), axis=0)
    tensor = tf.image.stateless_random_crop(tensor, size=window_size, seed=seed)
    tensor = tf.pad(
        tensor, ((0, frames_to_pad),) + ((0, 0),) * (len(tensor.shape) - 1),
        constant_values=constant_values)
    tensor = tf.ensure_shape(tensor, [self.length] + tensor.get_shape()[1:])
    return tensor

  def apply(self, tensor, seed, key=None, video_shape=None):
    """See base class."""
    del video_shape
    if key == self.boxes_key:
      constant_values = NOTRACK_BOX[0]
    else:
      constant_values = 0
    return self._apply(tensor, seed, constant_values=constant_values)


@dataclasses.dataclass
class TemporalRandomStridedWindow(RandomVideoPreprocessOp):
  """Gets a random strided slice (window) along 0-th axis of input tensor.

  This op is like TemporalRandomWindow but it samples from one of a set of
  strides of the video, whereas TemporalRandomWindow will densely sample from
  all possible slices of `length` frames from the video.

  For the following video and `length=3`: [1, 2, 3, 4, 5, 6, 7, 8, 9]

  This op will return one of [1, 2, 3], [4, 5, 6], or [7, 8, 9]

  This pads the video if the video length is shorter than the provided length.

  Assumption: The number of frames is the same for all video-like modalities.

  Attr:
    length: An integer representing the new length of the video and the sampling
      stride width.
  """

  length: int

  def _apply(self, tensor: tf.Tensor, seed: Sequence[int],
             constant_values: Union[int, float]) -> tf.Tensor:
    """Applies the strided crop operation to the video tensor."""
    num_frames = tf.shape(tensor)[0]
    num_crop_points = tf.cast(tf.math.ceil(num_frames / self.length), tf.int32)
    crop_point = tf.random.stateless_uniform(
        shape=(), minval=0, maxval=num_crop_points, dtype=tf.int32, seed=seed)
    crop_point *= self.length
    frames_sample = tensor[crop_point:crop_point + self.length]
    frames_to_pad = tf.maximum(self.length - tf.shape(frames_sample)[0], 0)
    frames_sample = tf.pad(
        frames_sample,
        ((0, frames_to_pad),) + ((0, 0),) * (len(frames_sample.shape) - 1),
        constant_values=constant_values)
    frames_sample = tf.ensure_shape(frames_sample, [self.length] +
                                    frames_sample.get_shape()[1:])
    return frames_sample

  def apply(self, tensor, seed, key=None, video_shape=None):
    """See base class."""
    del video_shape
    if key == self.boxes_key:
      constant_values = NOTRACK_BOX[0]
    else:
      constant_values = 0
    return self._apply(tensor, seed, constant_values=constant_values)


@dataclasses.dataclass
class FlowToRgb:
  """Converts flow to an RGB image.

  NOTE: This operation requires a statically known shape for the input flow,
    i.e. it is best to place it as final operation into the preprocessing
    pipeline after all shapes are statically known (e.g. after cropping /
    padding).
  """
  flow_key: str = FLOW

  def __call__(self, features: Features) -> Features:
    if self.flow_key in features:
      flow_rgb = flow_tensor_to_rgb_tensor(features[self.flow_key])
      assert flow_rgb.dtype == tf.uint8
      features[self.flow_key] = tf.image.convert_image_dtype(
          flow_rgb, tf.float32)
    return features
