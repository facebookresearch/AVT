from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import random
import numpy as np
import cv2
import numbers
import collections
import torch


# default list of interpolations
_DEFAULT_INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]
_DEFAULT_BLUR_PARAMS = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0),
                        (3, 0.4), (3, 0.6), (3, 0.8), (5, 1.0), (7, 1.2), (7, 1.4)]

# helper function for resizing a video clip
def resize_clip(clip, target_size, interpolation):
  num_frames = clip.shape[0]
  num_channels = clip.shape[3]
  tw, th = target_size
  resized_clip = np.zeros([num_frames, th, tw, num_channels], dtype=clip.dtype)
  for idx in range(num_frames):
    resized_clip[idx, :, :, :] = cv2.resize(
      clip[idx, :, :, :], (tw, th), interpolation=interpolation)
  return resized_clip

class Compose(object):
  """Composes several video transforms together.

  Args:
      transforms (list of ``Transform`` objects): list of transforms to compose.

  Example:
      >>> Compose([
      >>>     VideoScale(256),
      >>>     VideoRandomSizedCrop(224),
      >>> ])
  """
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, clip):
    for t in self.transforms:
      clip = t(clip)
    return clip

  def __repr__(self):
    repr_str = ""
    for t in self.transforms:
      repr_str += t.__repr__() + '\n'
    return repr_str

class VideoRandomHorizontalFlip(object):
  """Horizontally flip the given numpy array randomly
     (with a probability of 0.5).
  """
  def __call__(self, clip):
    """
    Args:
        clip (numpy array): Clip to be flipped.

    Returns:
        numpy array: Randomly flipped image
    """
    if random.random() < 0.5:
      clip = clip[:, :, ::-1, :]
    return clip

  def __repr__(self):
    return "Video Random Horizontal Flip"

class VideoRandomBlur(object):
  """
  Add random blue to a numpy.ndarray =clip
  """
  def __init__(self, blur_params=_DEFAULT_BLUR_PARAMS):
    # create blur by sample from a pre-specified (ksize, sigma)
    self.blur_params = blur_params

  def __call__(self, clip):
    blur_param = random.sample(self.blur_params, 1)[0]
    # no blur, just return as it is
    if blur_param[0] == 1:
      return clip
    num_frames = clip.shape[0]
    blurred_clip = np.zeros_like(clip)

    for idx in range(num_frames):
      blurred_clip[idx, :, :, :] = cv2.GaussianBlur(
        clip[idx, :, :, :], (blur_param[0], blur_param[0]), blur_param[1])
    return blurred_clip

  def __repr__(self):
    return "Video Random Blur"

class VideoScale(object):
  """Rescale the input numpy array to the given size.

  Args:
      size (sequence or int): Desired output size. If size is a sequence like
          (w, h), output size will be matched to this. If size is an int,
          smaller edge of the image will be matched to this number.
          i.e, if height > width, then image will be rescaled to
          (size, size * height / width)

      interpolations (list of int, optional): Desired interpolation.
      Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
      Pass None during testing: always use CV2.INTER_LINEAR
  """
  def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS):
    assert (isinstance(size, int)
            or (isinstance(size, collections.Iterable)
                and len(size) == 2)
           )
    self.size = size
    # use a proper antialiasing filter if interpolation is not specified
    if interpolations is None:
      interpolations = [cv2.INTER_LANCZOS4]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations

  def __call__(self, clip):
    """
    Args:
        clip (numpy array): Clip to be scaled.

    Returns:
        numpy array: Rescaled clip
    """
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]

    # scale the video clip
    if isinstance(self.size, int):
      h, w = clip.shape[1], clip.shape[2]
      if (w <= h and w == self.size) or (h <= w and h == self.size):
        return clip
      if w < h:
        ow, oh = int(self.size), int(round(self.size * h / w))
      else:
        oh, ow = int(self.size), int(round(self.size * w / h))
      clip = resize_clip(clip, (ow, oh), interpolation=interpolation)
      return clip
    else:
      h, w = clip.shape[1], clip.shape[2]
      if w == self.size[0] and h == self.size[1]:
        return clip
      else:
        clip = resize_clip(clip, self.size, interpolation=interpolation)
        return clip

  def __repr__(self):
    if isinstance(self.size, int):
      return "Video Scale [Shortest side {:d}]".format(self.size)
    else:
      target_size = self.size
      return "Video Scale [Exact Size ({:d}, {:d})]".format(
        target_size[0], target_size[1])

class VideoRandomSizedCrop(object):
  """Crop the given numpy array to random area and aspect ratio.

  A crop of random area of the original size and a random aspect ratio
  of the original aspect ratio is made. This crop is finally resized to given size.
  This is widely used as data augmentation for training image classification models

  Args:
      size (sequence or int): size of target image. If size is a sequence like
          (w, h), output size will be matched to this. If size is an int,
          output size will be (size, size).
      interpolations (list of int, optional): Desired interpolation.
      Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
      area_range (list of int): range of the areas to sample from
      ratio_range (list of int): range of aspect ratio to sample from
      num_trials (int): number of sampling trials
  """

  def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS,
               area_range=(0.16, 1.0), ratio_range=(0.75, 1.33), num_trials=10):
    self.size = size
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations
    self.num_trials = int(num_trials)
    self.area_range = area_range
    self.ratio_range = ratio_range

  def __call__(self, clip):
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]

    for attempt in range(self.num_trials):

      # sample target area / aspect ratio from area range and ratio range
      area = clip.shape[1] * clip.shape[2]
      target_area = random.uniform(self.area_range[0], self.area_range[1]) * area
      aspect_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])

      # compute the width and height
      # note that there are two possibilities
      # crop the image and resize to output size
      w = int(round(np.sqrt(target_area * aspect_ratio)))
      h = int(round(np.sqrt(target_area / aspect_ratio)))
      if random.random() < 0.5:
        w, h = h, w

      # crop the image
      if w <= clip.shape[2] and h <= clip.shape[1]:
        x1 = random.randint(0, clip.shape[2] - w)
        y1 = random.randint(0, clip.shape[1] - h)

        clip = clip[:, y1 : y1 + h, x1 : x1 + w, :]
        if isinstance(self.size, int):
          clip = resize_clip(clip, (self.size, self.size),
                             interpolation=interpolation)
        else:
          clip = resize_clip(clip, self.size, interpolation=interpolation)
        return clip

    # Fall back
    vid_scale = VideoScale(self.size, interpolations=self.interpolations)
    clip = vid_scale(clip)
    if isinstance(self.size, int):
      # with a square sized output, the default is to crop the patch in the center
      # (after all trials fail)
      h, w = clip.shape[1], clip.shape[2]
      tw, th = self.size, self.size
      x1 = int(round((w - tw) / 2.))
      y1 = int(round((h - th) / 2.))
      clip = clip[:, y1 : y1 + th, x1 : x1 + tw, :]
      return clip
    else:
      # with a pre-specified output size, the default crop is the image itself
      return clip

  def __repr__(self):
    if isinstance(self.size, int):
      target_size = (self.size, self.size)
    else:
      target_size = self.size
    return "Video Random Size Crop " + \
           "[Size ({:d}, {:d}); Area {:.2f} - {:.2f}; Ratio {:.2f} - {:.2f}]".format(
            target_size[0], target_size[1],
            self.area_range[0], self.area_range[1],
            self.ratio_range[0], self.ratio_range[1])

class VideoRandomColor(object):
  """Perturb color channels of a given image
  Sample alpha in the range of (-r, r) and multiply 1 + alpha to a color channel.
  The sampling is done independently for each channel.

  This implementation uses lookup time for efficiency (~15% cut of total time)

  Args:
      color_range (float): range of color jitter ratio (-r ~ +r) max r = 1.0
  """
  def __init__(self, color_range):
    self.color_range = color_range
    self.color_lut = np.arange(256, dtype=np.float32)

  def __call__(self, clip):
    orig_clip = clip
    if isinstance(orig_clip, torch.Tensor):
      clip = clip.numpy()
    num_frames = clip.shape[0]
    color_lut = np.zeros([1, 256, 3], dtype=np.float32)
    for c in range(3):
      target_ratio = random.uniform(-self.color_range, self.color_range) + 1.0
      color_lut[0, :, c] = self.color_lut * target_ratio
    color_lut = np.clip(color_lut, 0, 255).astype(np.uint8)
    for idx in range(num_frames):
      clip[idx, :, :, :] = cv2.LUT(clip[idx, :, :, :], color_lut)
    if isinstance(orig_clip, torch.Tensor):
      clip = torch.as_tensor(clip, dtype=orig_clip.dtype)
    return clip

  def __repr__(self):
    return "Video Random Color [Range {:.2f} - {:.2f}]".format(
            1-self.color_range, 1+self.color_range)

class VideoRandomRotate(object):
  """Rotate the given numpy array (around the image center) by a random degree.
  Expensive for videos (avoid this transform!)

  Args:
      degree_range (float): range of degree (-d ~ +d)
  """
  def __init__(self, degree_range, interpolations=_DEFAULT_INTERPOLATIONS):
    self.degree_range = degree_range
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations

  def __call__(self, clip):
    orig_clip = clip
    if isinstance(orig_clip, torch.Tensor):
      clip = clip.numpy()
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]
    # sample rotation
    degree = random.uniform(-self.degree_range, self.degree_range)
    # ignore small rotations
    if np.abs(degree) <= 1.0:
      return clip

    # get the max area rectangular within the rotated image
    # ref: stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    h, w = clip.shape[1], clip.shape[2]
    side_long = float(max([h, w]))
    side_short = float(min([h, w]))

    # since the solutions for angle, -angle and pi-angle are all the same,
    # it suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a = np.abs(np.sin(np.pi * degree / 180))
    cos_a = np.abs(np.cos(np.pi * degree / 180))

    if (side_short <= 2.0 * sin_a * cos_a * side_long):
      # half constrained case: two crop corners touch the longer side,
      # the other two corners are on the mid-line parallel to the longer line
      x = 0.5 * side_short
      if w >= h:
        wr, hr = x / sin_a, x / cos_a
      else:
        wr, hr = x / cos_a, x / sin_a
    else:
      # fully constrained case: crop touches all 4 sides
      cos_2a = cos_a * cos_a - sin_a * sin_a
      wr = (w * cos_a - h * sin_a) / cos_2a
      hr = (h * cos_a - w * sin_a) / cos_2a

    rot_mat = cv2.getRotationMatrix2D((w/2.0, h/2.0), degree, 1.0)
    rot_mat[0,2] += (wr - w)/2.0
    rot_mat[1,2] += (hr - h)/2.0

    # rotate each frame within the clip
    wr, hr = int(round(wr)), int(round(hr))
    rotated_clip = np.zeros([clip.shape[0], hr, wr, clip.shape[3]],
                            dtype=clip.dtype)
    for idx in range(clip.shape[0]):
      rotated_clip[idx, :, :, :] = cv2.warpAffine(clip[idx, :, :, :],
                                                  rot_mat,
                                                  (wr, hr),
                                                  flags=interpolation)
    if isinstance(orig_clip, torch.Tensor):
      rotated_clip = torch.as_tensor(rotated_clip, dtype=orig_clip.dtype)
    return rotated_clip

  def __repr__(self):
    return "Video Random Rotation [Range {:.2f} - {:.2f} Degree]".format(
            -self.degree_range, self.degree_range)

class VideoCenterCrop(object):
  """Crops the given numpy array at the center.

  Args:
      size (sequence or int): Desired output size of the crop. If size is an
      int instead of sequence like (w, h), a square crop (size, size) is
      made.
  """

  def __init__(self, size):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size

  def __call__(self, clip):
    """
    Args:
      clip (numpy array): Image to be cropped.

    Returns:
      numpy array: Cropped image
    """
    h, w = clip.shape[1], clip.shape[2]
    tw, th = self.size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    clip = clip[:, y1 : y1 + th, x1 : x1 + tw, :]
    return clip

  def __repr__(self):
    target_size = self.size
    return "Center Crop " + \
           "[Size ({:d}, {:d})]".format(target_size[0], target_size[1])

class VideoToTensor(object):
  """Convert a ``numpy.ndarray`` image to tensor.

  Converts a numpy.ndarray (T x H x W x C) image in the range
  [0, 255] to a torch.FloatTensor of shape (C x T x H x W) in the range [0.0, 1.0].
  """
  def __init__(self, normalize=True):
    self.normalize = normalize

  def __call__(self, clip):
    assert isinstance(clip, np.ndarray)
    assert (clip.ndim == 4) or (clip.ndim == 5)
    # convert to pytorch tensor
    if clip.ndim == 4:
      tensor_clip = torch.from_numpy(
        np.ascontiguousarray(
          clip.transpose((3, 0, 1, 2))
          ))
    else:
      tensor_clip = torch.from_numpy(
        np.ascontiguousarray(
          clip.transpose((0, 4, 1, 2, 3))
          ))
    # backward compatibility
    if isinstance(tensor_clip, torch.ByteTensor):
      tensor_clip = tensor_clip.float()
      if self.normalize:
        return tensor_clip.div_(255.0)
      else:
        return tensor_clip
    else:
      return tensor_clip

  def __repr__(self):
    if self.normalize:
      return "Video To Tensor() with normalization"
    else:
      return "Video To Tensor() without normalization"

class VideoRandomCrop(object):
  """Crop the given numpy array at a random location.

  Args:
      size (sequence or int): Desired output size of the crop. If size is an
      int instead of sequence like (w, h), a square crop (size, size) is
      made.
  """

  def __init__(self, size, padding=0):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size
    self.padding = int(padding)

  def __call__(self, clip):
    """
    Args:
      clip (numpy array): Clip to be cropped.

    Returns:
      numpy array: Cropped image
    """
    # zero pad the inputs
    if self.padding > 0:
      clip = np.pad(clip, ((0, 0),
                         (self.padding, self.padding),
                         (self.padding, self.padding),
                         (0, 0)), "constant")

    # get input / output size
    h, w = clip.shape[1], clip.shape[2]
    tw, th = self.size

    # return the image
    if w == tw and h == th:
      return clip

    # a corner case where the croping size is larger than image size
    # zero padding instead of resize (otherwise this will cancel the resizing)
    # we randomize how we put the patch back
    if w < tw or h < th:
      new_clip = np.zeros((clip.shape[0], th, tw, 3), dtype=clip.dtype)
      # all cases
      if w >= tw and h < th:
        # sample x
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, th - h)
        new_clip[:, y1 : y1+h, :, :] = clip[:, :, x1 : x1+tw, :]
      elif w < tw and h >= th:
        # sample y
        x1 = random.randint(0, tw - w)
        y1 = random.randint(0, h - th)
        new_clip[:, :, x1 : x1+w, :] = clip[:, y1 : y1+th, :, :]
      else:
        # w<tw, h<th
        x1 = random.randint(0, tw - w)
        y1 = random.randint(0, th - h)
        new_clip[:, y1 : y1+h, x1 : x1+w, :] = clip
      return new_clip

    # crop a smaller patch within the image
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    clip = clip[:, y1 : y1 + th, x1 : x1 + tw, :]
    return clip

  def __repr__(self):
    target_size = self.size
    return "Video Random Crop " + \
           "[Size ({:d}, {:d})] with padding {:d}".format(
             target_size[0], target_size[1], self.padding)

class VideoRandomMultiScaleCrop(object):
  """Crop the given numpy array by a random choice of fixed scales / crops
     (used in TSN). Similar to RandomSizedCrop yet slightly more efficient.

  Args:
      size (sequence or int): Desired output size of the crop. If size is an
      int instead of sequence like (w, h), a square crop (size, size) is
      made.
  """
  def __init__(self, size,
               scales=None, max_distort=1, fix_crop=False,
               interpolations=_DEFAULT_INTERPOLATIONS):
    self.input_size = size if not isinstance(size, int) else [size, size]
    self.scales = scales if scales is not None else [1, 0.875, 0.75, 0.66]
    self.max_distort = max_distort
    self.fix_crop = fix_crop
    self.interpolations = interpolations

  def __call__(self, clip):
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]
    # sample crop size
    num_frames, im_h, im_w = clip.shape[0], clip.shape[1], clip.shape[2]
    crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_w, im_h)
    # crop and resize
    cropped_clip = clip[:, offset_h:offset_h+crop_h, offset_w:offset_w+crop_w, :]
    clip = resize_clip(cropped_clip, self.input_size, interpolation=interpolation)
    return clip

  def _sample_crop_size(self, im_w, im_h):
    base_size = min(im_w, im_h)
    crop_sizes = [int(base_size * x) for x in self.scales]
    crop_h = [self.input_size[1] if abs(x-self.input_size[1]) < 3 else x \
                for x in crop_sizes]
    crop_w = [self.input_size[0] if abs(x-self.input_size[0]) < 3 else x \
                for x in crop_sizes]

    pairs = []
    for i, h in enumerate(crop_h):
      for j, w in enumerate(crop_w):
        if abs(i - j) <= self.max_distort:
          # crop size is capped by image size
          pairs.append((min(w, im_w), min(h, im_h)))

    crop_pair = random.choice(pairs)
    if not self.fix_crop:
      w_offset = random.randint(0, im_w - crop_pair[0])
      h_offset = random.randint(0, im_h - crop_pair[1])
    else:
      w_offset, h_offset = self._sample_fix_offset(
        im_w, im_h, crop_pair[0], crop_pair[1])

    return crop_pair[0], crop_pair[1], w_offset, h_offset

  def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
    offsets = self._fill_fix_offset(image_w, image_h, crop_w, crop_h)
    return random.choice(offsets)

  @staticmethod
  def _fill_fix_offset(image_w, image_h, crop_w, crop_h):
    w_step = (image_w - crop_w) // 4
    h_step = (image_h - crop_h) // 4
    # standard crops
    ret = []
    ret.append((0, 0))  # upper left
    ret.append((4 * w_step, 0))  # upper right
    ret.append((0, 4 * h_step))  # lower left
    ret.append((4 * w_step, 4 * h_step))  # lower right
    ret.append((2 * w_step, 2 * h_step))  # center
    # additional crops
    ret.append((0, 2 * h_step))  # center left
    ret.append((4 * w_step, 2 * h_step))  # center right
    ret.append((2 * w_step, 4 * h_step))  # lower center
    ret.append((2 * w_step, 0 * h_step))  # upper center
    ret.append((1 * w_step, 1 * h_step))  # upper left quarter
    ret.append((3 * w_step, 1 * h_step))  # upper right quarter
    ret.append((1 * w_step, 3 * h_step))  # lower left quarter
    ret.append((3 * w_step, 3 * h_step))  # lower righ quarter
    return ret

  def __repr__(self):
    scales_str = []
    for scale in self.scales:
      scales_str.append("{:0.3f}".format(scale))
    scales_str = ','.join(scales_str)
    return "Multi-scale crop [size ({:d}, {:d}), scale {:s}]".format(
      self.input_size[0], self.input_size[1], scales_str)

class VideoFixedCrop(object):
  """Crop an tensor clip with multiple fixed crop

  This is used as test time augmentation. The transform will create center crops
  along the longest side of the image. Importantly, we always assume the shortest
  side of the image is larger than or equal to the target crop size.

  Args:
    size: crop size
    num_crops: number of crops

  """
  def __init__(self, size, num_crops):
    self.size = size if not isinstance(size, int) else [size, size]
    self.num_crops = num_crops

  def __call__(self, clip):
    # clip of size T * H * W * 3
    h, w = clip.shape[1], clip.shape[2]
    tw, th = self.size
    clips = []
    # crop along width
    if h <= w:
      y1 = int(round((h - th) / 2.))
      if self.num_crops == 1:
        # corner case: single center crop
        all_x1 = [(w - tw) / 2.]
      elif (w - tw - 1) > 0:
        # multipe crops along the longest side
        all_x1 = np.linspace(0, w - tw - 1, self.num_crops)
      else:
        # corner case
        all_x1 = [0] * self.num_crops

      for x1 in all_x1:
        x1 = int(round(x1))
        clip_crop = clip[:, y1 : y1 + th, x1 : x1 + tw, :]
        clips.append(clip_crop[np.newaxis, :, :, :, :])

    # crop along height
    else:
      x1 = int(round((w - tw) / 2.))
      if self.num_crops == 1:
        # corner case: single center crop
        all_y1 = [(h - th) / 2.]
      elif (h - th - 1) > 0:
        # multipe crops along the longest side
        all_y1 = np.linspace(0, h - th - 1, self.num_crops)
      else:
        all_y1 = [0] * self.num_crops

      for y1 in all_y1:
        y1 = int(round(y1))
        clip_crop = clip[:, y1 : y1 + th, x1 : x1 + tw, :]
        clips.append(clip_crop[np.newaxis, :, :, :, :])

    # cat the crops, output size -> #crops * T * H * W * 3
    clips = np.concatenate(clips, axis=0)
    return clips

  def __repr__(self):
    return "Video Fixed Crop [size ({:d}, {:d}), #crops {:d}]".format(
      self.size[0], self.size[1], self.num_crops)

#################################################################################
# Obsolete video transforms
#################################################################################
# class VideoNormalize(object):
#   """Normalize an tensor image with mean and standard deviation.
#
#   Given mean: (R, G, B) and std: (R, G, B),
#   will normalize each channel of the torch.*Tensor, i.e.
#   channel = (channel - mean) / std
#
#   Args:
#       mean (sequence): Sequence of means for R, G, B channels respecitvely.
#       std (sequence): Sequence of standard deviations for R, G, B channels
#         respecitvely.
#   """
#   def __init__(self, mean, std):
#     self.mean = mean
#     self.std = std
#
#   def __call__(self, tensor_clip):
#     # subtract mean (per channel) -> divide by std (per channel)
#     mean = torch.as_tensor(self.mean, dtype=torch.float32,
#                            device=tensor_clip.device)
#     std = torch.as_tensor(self.std, dtype=torch.float32,
#                           device=tensor_clip.device)
#     if tensor_clip.dim() == 4:
#       tensor_clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
#     else:
#       tensor_clip.sub_(mean[None, :, None, None, None]
#         ).div_(std[None, :, None, None, None])
#     return tensor_clip
#
#   def __repr__(self):
#     return "Video Normalize " + '(mean={0}, std={1})'.format(self.mean, self.std)
#
# class VideoDenormalize(object):
#   """De-normalize an tensor image with mean and standard deviation.
#
#   Given mean: (R, G, B) and std: (R, G, B),
#   will normalize each channel of the torch.*Tensor, i.e.
#   channel = channel * std + mean
#
#   Args:
#       mean (sequence): Sequence of means for R, G, B channels respecitvely.
#       std (sequence): Sequence of standard deviations for R, G, B channels
#         respecitvely.
#   """
#   def __init__(self, mean, std):
#     self.mean = mean
#     self.std = std
#
#   def __call__(self, tensor_clip):
#     # multiply by std (per channel) -> add mean (per channel)
#     mean = torch.as_tensor(self.mean, dtype=torch.float32,
#                            device=tensor_clip.device)
#     std = torch.as_tensor(self.std, dtype=torch.float32,
#                           device=tensor_clip.device)
#     tensor_clip.mul_(std[:, None, None, None]).add_(mean[:, None, None, None])
#     return tensor_clip
#
#   def __repr__(self):
#     return "Video De-normalize " + '(mean={0}, std={1})'.format(
#       self.mean, self.std)

# class VideoPadToLength(object):
#   """Padding (repeat the last frame) of a video to a fixed number of frames
#
#     Args:
#       num_frames (int): max number of frames
#   """
#   def __init__(self, num_frames):
#     self.num_frames = num_frames
#
#   def __call__(self, clip):
#     # assuming input of T * H * W * C
#     assert clip.shape[0] <= self.num_frames
#     # no padding needed
#     if clip.shape[0] == self.num_frames:
#       return clip
#
#     # prep for padding
#     new_clip = np.zeros(
#       [self.num_frames, clip.shape[1], clip.shape[2], clip.shape[3]],
#       dtype=clip.dtype)
#     clip_num_frames = clip.shape[0]
#     padded_num_frames = self.num_frames - clip_num_frames
#
#     # fill in new_clip (repeating the last frame)
#     new_clip[0:clip_num_frames, :, :, :] = clip[:, :, :, :]
#     new_clip[clip_num_frames:, :, :, :] = np.tile(
#       clip[-1, :, :, :], (padded_num_frames, 1, 1, 1))
#     return new_clip
#
#   def __repr__(self):
#     return "Video Padding to max {:d} frames".format(self.num_frames)

################################################################################
def create_video_transforms(input_config):
  """Create transforms from configuration
  Parameters
  ----------
  Args : dict
    Dictionary containing the configuration options for input pre-processing
    Example:
      # params for training
      "rotation": 10,  # if <=0 do not rotate
      "flip": True,  # random flip during training
      "color_jitter": 0.05, # color pertubation (if <=0 disabled)
      "padding": 0,    # padding before crop
      "crop_resize": False, # resize the regions before crop
      "crop_resize_area_range": [0.16, 1.0],   # default imagenet
      "crop_resize_ratio_range": [0.75, 1.33], # default imagenet
      "scale_train": 36,  # If -1 do not scale
      "crop_train": 32,  # crop size (must > 0)
      # param for val
      "scale_val": 36,  # If -1 do not scale
      "crop_val": -1,   # if -1 do not crop

  Returns
  -------
  train_transforms : composed list
    List of transforms to be applied to the input during training
  val_transforms : composed list
    List of transforms to be applied to the input during validation / testing
  """
  # scale the clip by shortest side
  train_transforms = []
  if input_config["scale_train"] != -1:
    train_transforms.append(VideoScale(
      input_config["scale_train"], interpolations=None))
  # random rotation
  if input_config["rotation"] > 0:
    train_transforms.append(VideoRandomRotate(input_config["rotation"]))
  # random scale and crop
  if input_config['crop_train'] != -1:
    if input_config['crop_resize']:
      train_transforms.append(
        VideoRandomSizedCrop(input_config["crop_train"],
          area_range=input_config["crop_resize_area_range"],
          ratio_range=input_config["crop_resize_ratio_range"]))
    elif input_config['crop_scale']:
      train_transforms.append(
        VideoRandomMultiScaleCrop(input_config["crop_train"]))
    else:
      train_transforms.append(
        VideoRandomCrop(input_config["crop_train"],
                        padding=input_config["padding"]))
  # random flip
  if input_config['flip']:
    train_transforms.append(VideoRandomHorizontalFlip())
  if input_config["blur"]:
    train_transforms.append(VideoRandomBlur())
  if input_config["color_jitter"] >= 0:
    train_transforms.append(VideoRandomColor(input_config["color_jitter"]))
  # convert to tensor & normalize is delayed to data prefetcher (on GPU)
  train_transforms = Compose(train_transforms)

  # transforms for val (scale and crop)
  val_transforms = []
  if input_config["scale_val"] != -1:
    val_transforms.append(VideoScale(
      input_config["scale_val"], interpolations=None))
  if input_config["crop_val"] != -1:
    val_transforms.append(VideoCenterCrop(input_config["crop_val"]))
  # convert to tensor & normalize is delayed to data prefetcher (on GPU)
  val_transforms = Compose(val_transforms)

  # transforms for test (with data augmentations)
  test_transforms = []
  if input_config["scale_test"] != -1:
    test_transforms.append(VideoScale(
      input_config["scale_test"], interpolations=None))
  if input_config["crop_test"] != -1:
    test_transforms.append(VideoFixedCrop(
      input_config["crop_test"], input_config["num_test_crops"]))
  # convert to tensor & normalize is delayed to data prefetcher (on GPU)
  test_transforms = Compose(test_transforms)

  return train_transforms, val_transforms, test_transforms
