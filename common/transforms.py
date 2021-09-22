# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import numbers
import random

from torchvision.transforms import (
    RandomCrop,
    RandomResizedCrop,
    ColorJitter,
    ToPILImage,
    ToTensor,
)

__all__ = [
    "RandomCropVideo",
    "RandomResizedCropVideo",
    "CenterCropVideo",
    "NormalizeVideo",
    "ToTensorVideo",
    "RandomHorizontalFlipVideo",
    "Resize",
    "TemporalCenterCrop",
    "ColorJitterVideo",
]


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True


def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
    """
    assert len(clip.size()) == 4, "clip should be a 4D tensor"
    return clip[..., i:i + h, j:j + w]


def temporal_center_crop(clip, clip_len):
    """
    Args:
        clip (torch.tensor): Video clip to be
        cropped along the temporal axis. Size is (C, T, H, W)
    """
    assert len(clip.size()) == 4, "clip should be a 4D tensor"
    assert clip.size(1) >= clip_len, "clip is shorter than the proposed lenght"
    middle = int(clip.size(1) // 2)
    start = middle - clip_len // 2
    return clip[:, start:start + clip_len, ...]


def resize(clip, target_size, interpolation_mode):
    """
    Args:
        target_size can be a
            integer: Which is the length of the smaller side
            string: with format <min>-<max>: will randomly pick a size from
                min and max (included) to be the smaller side
            or tuple of either integers and/or string
    """
    def _convert_size_to_integer(size_str):
        if isinstance(size_str, int):
            return size_str
        size_min, size_max = [int(el) for el in size_str.split('-')]
        return random.randint(size_min, size_max)

    if isinstance(target_size, (list, tuple)):
        target_size = (_convert_size_to_integer(target_size[0]),
                       _convert_size_to_integer(target_size[1]))
    else:
        target_size = _convert_size_to_integer(target_size)

    if isinstance(target_size, int):
        clip_h, clip_w = clip.shape[-2], clip.shape[-1]
        scale_factor = target_size * 1.0 / min(clip_h, clip_w)
        # Make sure the new sizes def follow the target_size, sometimes the
        # ratio leads to a couple pixel below which can lead to errors
        new_h = max(int(clip_h * scale_factor), target_size)
        new_w = max(int(clip_w * scale_factor), target_size)
        target_size = (new_h, new_w)
    return torch.nn.functional.interpolate(clip,
                                           size=target_size,
                                           mode=interpolation_mode)


def resized_crop(clip, i, j, h, w, size, interpolation_mode="bilinear"):
    """
    Do spatial cropping and resizing to the video clip
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped region.
        w (int): Width of the cropped region.
        size (tuple(int, int)): height and width of resized clip
    Returns:
        clip (torch.tensor): Resized and cropped clip. Size is (C, T, H, W)
    """
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    clip = crop(clip, i, j, h, w)
    clip = resize(clip, size, interpolation_mode)
    return clip


def center_crop(clip, crop_size):
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    h, w = clip.size(-2), clip.size(-1)
    th, tw = crop_size
    assert h >= th and w >= tw, "height and width must be >= than crop_size"

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
    """
    try:
        _is_tensor_video_clip(clip)
    except (TypeError, ValueError):
        # Needed to add this since this happens when using Miao's transforms
        clip = torch.as_tensor(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" %
                        str(clip.dtype))
    return clip.float().permute(3, 0, 1, 2) / 255.0


def normalize(clip, mean, std, inplace=False):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (C, T, H, W)
    """
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip


def hflip(clip):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
    Returns:
        flipped clip (torch.tensor): Size is (C, T, H, W)
    """
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    return clip.flip((-1))


class RandomCropVideo(RandomCrop):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized video clip.
                size is (C, T, OH, OW)
        """
        i, j, h, w = self.get_params(clip, self.size)
        return crop(clip, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomResizedCropVideo(RandomResizedCrop):
    def __init__(
            self,
            size,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode
        self.scale = scale
        self.ratio = ratio

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized video clip.
                size is (C, T, H, W)
        """
        i, j, h, w = self.get_params(clip, self.scale, self.ratio)
        return resized_crop(clip, i, j, h, w, self.size,
                            self.interpolation_mode)

    def __repr__(self):
        return self.__class__.__name__ + \
            '(size={0}, interpolation_mode={1}, scale={2}, ratio={3})'.format(
                self.size, self.interpolation_mode, self.scale, self.ratio
            )


class CenterCropVideo(object):
    def __init__(self, crop_size):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        return center_crop(clip, self.crop_size)

    def __repr__(self):
        r = self.__class__.__name__ + '(crop_size={0})'.format(self.crop_size)
        return r


def multi_crop(video, crop_size, num_crops, flips):
    """
    Returns a list of video crops of crop_size, num_crops * 2 in length
        (including flipped versions)
    """
    assert _is_tensor_video_clip(video), "clip should be a 4D torch.tensor"
    h, w = video.size(-2), video.size(-1)
    th, tw = crop_size
    assert h >= th and w >= tw, "height and width must be >= than crop_size"
    if num_crops == 1:
        # Center crop, as used in the CenterCrop function
        pos = [(int(round((h - th) / 2.0)), int(round((w - tw) / 2.0)))]
    elif num_crops == 3:
        # top left, center, and bottom right
        pos = [(0, 0), (int(round((h - th) / 2.0)), int(round(
            (w - tw) / 2.0))), (h - th, w - tw)]
    else:
        raise NotImplementedError('Not supported')
    cropped = [crop(video, i, j, th, tw) for i, j in pos]
    if flips:
        cropped += [hflip(el) for el in cropped]
    return cropped


class MultiCropVideo(object):
    def __init__(self, crop_size, num_crops, flips=False):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        self.num_crops = num_crops
        self.flips = flips

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (num_crops, C, T, crop_size, crop_size)
        """
        return torch.stack(
            multi_crop(clip, self.crop_size, self.num_crops, self.flips), 0)

    def __repr__(self):
        return (self.__class__.__name__ +
                f'(crop_size={self.crop_size},num_crops={self.num_crops})')


class TemporalCenterCrop(object):
    def __init__(self, clip_len):
        self.clip_len = clip_len

    def __call__(self, clip):
        return temporal_center_crop(clip, self.clip_len)


class UnfoldClips(object):
    def __init__(self, clip_len, overlap):
        self.clip_len = clip_len
        assert overlap > 0 and overlap <= 1
        self.step = round(clip_len * overlap)

    def __call__(self, clip):
        if clip.size(1) < self.clip_len:
            return clip.unfold(1, clip.size(1),
                               clip.size(1)).permute(1, 0, 4, 2, 3)

        results = clip.unfold(1, self.clip_len,
                              self.clip_len).permute(1, 0, 4, 2, 3)
        return results


class NormalizeVideo(object):
    """
    Normalize the video clip by mean subtraction
    and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be
                                normalized. Size is (C, T, H, W)
        """
        return normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, inplace={2})'.format(
            self.mean, self.std, self.inplace)


class ToTensorVideo(object):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of clip tensor
    """
    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
        """
        return to_tensor(clip)

    def __repr__(self):
        return self.__class__.__name__


class RandomHorizontalFlipVideo(object):
    """
    Flip the video clip along the horizonal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        """
        if random.random() < self.p:
            clip = hflip(clip)
        return clip

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)


class ColorJitterVideo():
    """
    Randomly add color jitter to video
    Args:
        Same as original ColorJitter
    """
    def __init__(self, *args, **kwargs):
        self.frame_color_jitter = ColorJitter(*args, **kwargs)

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        """
        assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
        # Stack the frames on height dimension
        stacked_frames = clip.view((clip.size(0), -1, clip.size(-1)))
        stacked_frames_pil = ToPILImage()(stacked_frames)
        output_stacked_frames = ToTensor()(
            self.frame_color_jitter(stacked_frames_pil))
        return output_stacked_frames.view(clip.shape)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size, interpolation_mode="bilinear")
