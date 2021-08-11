import math
import torch
from torch.utils.data import Sampler
import torch.distributed as dist
import torchvision.datasets.video_utils


class DistributedSampler(Sampler):
    """
    Extension of DistributedSampler, as discussed in
    https://github.com/pytorch/pytorch/issues/23430
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        if isinstance(self.dataset, Sampler):
            orig_indices = list(iter(self.dataset))
            indices = [orig_indices[i] for i in indices]

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class UniformClipSampler(torch.utils.data.Sampler):
    """
    Samples at most `max_video_clips_per_video` clips for each video, equally spaced
    Arguments:
        video_clips (VideoClips): video clips to sample from
        max_clips_per_video (int): maximum number of clips to be sampled per video
    """
    def __init__(self, video_clips, max_clips_per_video):
        if not isinstance(video_clips, torchvision.datasets.video_utils.VideoClips):
            raise TypeError("Expected video_clips to be an instance of VideoClips, "
                            "got {}".format(type(video_clips)))
        self.video_clips = video_clips
        self.max_clips_per_video = max_clips_per_video

    def __iter__(self):
        idxs = []
        s = 0
        # select at most max_clips_per_video for each video, uniformly spaced
        for c in self.video_clips.clips:
            length = len(c)
            step = max(length // self.max_clips_per_video, 1)
            sampled = torch.arange(length)[::step] + s
            s += length
            idxs.append(sampled)
        idxs = torch.cat(idxs).tolist()
        return iter(idxs)

    def __len__(self):
        return sum(min(len(c), self.max_clips_per_video) for c in self.video_clips.clips)


class RandomClipSampler(torch.utils.data.Sampler):
    """
    Samples at most `max_video_clips_per_video` clips for each video randomly

    Arguments:
        video_clips (VideoClips): video clips to sample from
        max_clips_per_video (int): maximum number of clips to be sampled per video
    """
    def __init__(self, video_clips, max_clips_per_video):
        if not isinstance(video_clips, torchvision.datasets.video_utils.VideoClips):
            raise TypeError("Expected video_clips to be an instance of VideoClips, "
                            "got {}".format(type(video_clips)))
        self.video_clips = video_clips
        self.max_clips_per_video = max_clips_per_video

    def __iter__(self):
        idxs = []
        s = 0
        # select at most max_clips_per_video for each video, randomly
        for c in self.video_clips.clips:
            length = len(c)
            size = min(length, self.max_clips_per_video)
            sampled = torch.randperm(length)[:size] + s
            s += length
            idxs.append(sampled)
        idxs = torch.cat(idxs)
        # shuffle all clips randomly
        perm = torch.randperm(len(idxs))
        idxs = idxs[perm].tolist()
        return iter(idxs)

    def __len__(self):
        return sum(min(len(c), self.max_clips_per_video) for c in self.video_clips.clips)
