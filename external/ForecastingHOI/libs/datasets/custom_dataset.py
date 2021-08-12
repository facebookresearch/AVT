from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import numpy as np
from functools import partial

from torch.utils.data import Dataset

# https://github.com/pytorch/pytorch/issues/973
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

# for video IO
import lintel

class CustomVideoDataset(Dataset):
  """
  A custom json dataset
  video_list: each video item should be a dictionary of
  {
   "filename": "ApplyEyeMakeup/v_ApplyEyeMakeup_g15_c01.avi",
   "label": 0,
   "video_info": {"width": 320, "height": 240, "fps": 25.0, "num_frames": 169},
   "meta_label": []
  }

  label_dict: a dictionary that maps label id to label text
  NOTE: label id in label_dict is a string (instead of integer)
  """
  def __init__(self, root_folder, ant_file, num_classes, sample_pattern,
               is_training=True, transforms=None, protect_frames=2, k_clips=1):
    # set up params
    self.root_folder = root_folder
    self.transforms = transforms
    self.is_training = is_training
    self.protect_frames = protect_frames
    self.num_classes = num_classes
    self.k_clips = k_clips
    self.ant_file = ant_file

    # sample pattern: (N, M, K)
    # will draw N frames in the chunk of N*M with random offsets of M+-K
    # with K = 0, frames are sampled at equal interval
    assert len(sample_pattern) == 3
    assert (sample_pattern[2]>=0) and (sample_pattern[2]<=sample_pattern[1]//2)
    self.sample_pattern = sample_pattern

    # remove randomness for sample pattern
    if not self.is_training:
      self.sample_pattern[2] = 0
    return

  def reset_num_clips(self, num_clips):
    self.k_clips = num_clips
    return

  def get_num_clips(self):
    return self.k_clips

  def get_label_dict(self):
    # return a deep copy of label_dict (prevent any modification)
    return copy.deepcopy(self.label_dict)

  def _load_frames(self, video_file, video_info):
    # load video frames using lintel
    frame_idx = self._sample_frames(
      [0, video_info['num_frames'] - 1 - self.protect_frames])
    # TO-DO: This part is problematic for large videos (e.g., action detection)
    #        It might be possible to read a chunck of the video
    with open(video_file, 'rb') as f:
      encoded_video = f.read()

    decoded_frames = lintel.loadvid_frame_nums(encoded_video,
                                               frame_nums=frame_idx,
                                               width=video_info['width'],
                                               height=video_info['height'])
    decoded_frames = np.frombuffer(decoded_frames, dtype=np.uint8)
    clip = np.reshape(decoded_frames,
                      newshape=(len(frame_idx),
                      video_info['height'],
                      video_info['width'],
                      3))
    # pad to fixed length
    clip = self._pad_to_length(clip)
    return clip

  def _load_clips(self, video_file, video_info):
    # load a set of clips from the video
    all_frame_idx = self._select_k_clips(
      [0, video_info['num_frames'] - 1 - self.protect_frames])

    with open(video_file, 'rb') as f:
      encoded_video = f.read()

    # Reading all frames at once
    frame_idx = [int(item) for sublist in all_frame_idx for item in sublist]
    frame_idx = sorted(list(set(frame_idx)))
    decoded_frames = lintel.loadvid_frame_nums(encoded_video,
                                               frame_nums=frame_idx,
                                               width=video_info['width'],
                                               height=video_info['height'])
    decoded_frames = np.frombuffer(decoded_frames, dtype=np.uint8)
    frame_chunk = np.reshape(decoded_frames,
                             newshape=(len(frame_idx),
                             video_info['height'],
                             video_info['width'],
                             3))

    # tricky! We need to re-map the frames back into clips
    all_clips = []
    for clip_idx in range(self.k_clips):
      clip_frame_idx = all_frame_idx[clip_idx]
      clip_frame_mapping = []
      for cur_frame_idx in clip_frame_idx:
        clip_frame_mapping.append(frame_idx.index(cur_frame_idx))
      clip = frame_chunk[clip_frame_mapping, :, :, :]
      # pad -> expand dim -> add to list
      clip = self._pad_to_length(clip)
      all_clips.append(clip)

    return all_clips

  def _pad_to_length(self, clip):
    num_frames = self.sample_pattern[0]
    assert clip.shape[0] <= num_frames
    # no padding needed
    if clip.shape[0] == num_frames:
      return clip

    # prep for padding
    new_clip = np.zeros(
      [num_frames, clip.shape[1], clip.shape[2], clip.shape[3]],
      dtype=clip.dtype)
    clip_num_frames = clip.shape[0]
    padded_num_frames = num_frames - clip_num_frames

    # fill in new_clip (repeating the last frame)
    new_clip[0:clip_num_frames, :, :, :] = clip[:, :, :, :]
    new_clip[clip_num_frames:, :, :, :] = np.tile(
      clip[-1, :, :, :], (padded_num_frames, 1, 1, 1))
    return new_clip

  def _select_k_clips(self, frame_range):
    # select k clips of N*M frames from the frame_range
    # get video info
    starting_frame, ending_frame = min(frame_range), max(frame_range)
    num_video_frames = ending_frame - starting_frame + 1

    # compute clip interval
    all_frame_idx = []
    N, M, _ = self.sample_pattern
    clip_interval = (num_video_frames - N * M + 1) // self.k_clips

    # corner case: k_clips = 1
    if self.k_clips == 1:
      # position the clip at roughly the mid point
      clip_interval = clip_interval // 2

    # easy case k + N*M <= num_frames (where we can position all k_clips)
    if clip_interval > 0:
      # caution: arange exclude the end point
      clip_starting_frames = np.round(np.linspace(
        starting_frame, ending_frame - N*M + 1, self.k_clips))
    # no enough clips
    else:
      # no sufficient frames: equal interal between the head & tail
      # also make sure the last clip has some frames for the model
      clip_starting_frames = np.round(np.linspace(
        starting_frame, ending_frame - N*M // 2  + 1, self.k_clips))

    for clip_starting_frame  in clip_starting_frames:
      clip_ending_frame = clip_starting_frame + N * M
      clip_frame_idx = np.arange(clip_starting_frame, clip_ending_frame, M)
      all_frame_idx.append(clip_frame_idx)

    # sort and filter
    for clip_idx in range(self.k_clips):
      clip_frame_idx = all_frame_idx[clip_idx]
      clip_frame_idx = clip_frame_idx[(clip_frame_idx>=starting_frame)
                                      & (clip_frame_idx<=ending_frame)]
      clip_frame_idx = np.sort(clip_frame_idx).tolist()
      all_frame_idx[clip_idx] = clip_frame_idx

    return all_frame_idx

  def _sample_frames(self, frame_range):
    # sample N frames in frame_range using sample_pattern
    # get video info
    starting_frame, ending_frame = min(frame_range), max(frame_range)
    num_video_frames = ending_frame - starting_frame + 1

    # pick N*M chunk from frame_range
    N, M, K = self.sample_pattern
    if N * M < num_video_frames:
      # make sure samples are positioned at midpoints
      clip_starting_frame = np.random.randint(
        starting_frame, ending_frame - N * M + 1) + M // 2
    else:
      clip_starting_frame = starting_frame
    clip_ending_frame = clip_starting_frame + N * M

    # draw N frames from the chunk (with pertubations)
    frame_idx = np.arange(clip_starting_frame, clip_ending_frame, M)
    if K != 0:
      frame_offset = np.random.randint(-K, K+1, len(frame_idx))
      frame_idx = frame_idx + frame_offset

    # remove out of range frames
    # also make sure the frames idx in increasing order
    frame_idx = \
      frame_idx[(frame_idx>=starting_frame) & (frame_idx<=ending_frame)]
    frame_idx = np.sort(frame_idx).tolist()

    return frame_idx

  def __len__(self):
    return len(self.video_list)

  def __getitem__(self, index):
    # get video item
    video_item = self.video_list[index]

    # training / testing
    if self.is_training:
      clip, label, clip_id = self.prep_train_data(video_item)
      if self.transforms != None:
        clip = self.transforms(clip)
      return clip, label, clip_id
    else:
      clips, label, clip_id = self.prep_test_data(video_item)
      if self.transforms != None:
        assert isinstance(clips, list)
        for idx, clip in enumerate(clips):
          clips[idx] = self.transforms(clip)

      # numpy -> torch tensor is delayed to prefetcher
      clips = np.concatenate(clips, axis=0)

      return clips, label, clip_id

  """
  These functions should be overridden in a child class e.g., for a new dataset
  """
  def load(self):
    # load annotations (this function must be called first)
    video_list, label_dict = self.load_ants(self.ant_file)
    assert len(label_dict) == self.num_classes, "# classes does not match"
    self.video_list = video_list
    self.label_dict = label_dict
    return

  def load_ants(self, ant_file):
    # read annotation file
    with open(ant_file, 'r') as f:
      ant_data = json.load(f)
    video_list = ant_data['video_list']
    label_dict = ant_data['label_dict']
    return video_list, label_dict

  def prep_train_data(self, video_item):
    # get video info & load frames / labels
    video_file = os.path.join(self.root_folder, video_item['filename'])
    video_info = video_item['video_info']
    clip = self._load_frames(video_file, video_info)
    clip_id = video_item['filename']
    label = video_item['label']
    return clip, label, clip_id

  def prep_test_data(self, video_item):
    # get video info & load frames
    video_file = os.path.join(self.root_folder, video_item['filename'])
    video_info = video_item['video_info']
    clips = self._load_clips(video_file, video_info)
    clip_id = video_item['filename']
    # ignore empty list
    if isinstance(video_item['label'], list) and (len(video_item['label']) == 0):
      # fill in dummy label if we don't have one
      label = -1
    else:
      label = video_item['label']

    return clips, label, clip_id

  def get_num_samples_per_cls(self):
    # return number of samples per class
    assert self.is_training
    num_samples_per_cls = [0] * self.num_classes
    for video_item in self.video_list:
      label = video_item['label']
      num_samples_per_cls[label] += 1
    return num_samples_per_cls


class EpicKitchens(CustomVideoDataset):
  def __init__(self, root_folder, ant_file, num_classes, sample_pattern, action_type,
               is_training=True, transforms=None, protect_frames=2, k_clips=1):
    super(EpicKitchens, self).__init__(
      root_folder, ant_file, num_classes, sample_pattern,
      is_training=is_training,
      transforms=transforms,
      protect_frames=protect_frames,
      k_clips=k_clips)
    assert action_type in ["verb", "noun"]
    self.action_type = action_type
    return

  def load(self):
    # load annotations (this function must be called first)
    video_list, label_dict = self.load_ants(self.ant_file)
    assert len(label_dict) == self.num_classes, "# classes does not match"
    self.video_list = video_list
    self.label_dict = label_dict
    return

  def load_ants(self, ant_file):
    # read annotation file
    with open(ant_file, 'r') as f:
      ant_data = json.load(f)
    video_list = ant_data['video_list']
    label_dict = ant_data['{:s}_dict'.format(self.action_type)]
    return video_list, label_dict

  def prep_train_data(self, video_item):
    # get video info & load frames / labels
    video_file = os.path.join(self.root_folder, video_item['filename'])
    video_info = video_item['video_info']
    clip = self._load_frames(video_file, video_info)
    clip_id = video_item['filename']
    idx = ['verb', 'noun'].index(self.action_type)
    label = video_item['label'][idx]
    return clip, label, clip_id

  def prep_test_data(self, video_item):
    # get video info & load frames
    video_file = os.path.join(self.root_folder, video_item['filename'])
    video_info = video_item['video_info']
    clips = self._load_clips(video_file, video_info)
    clip_id = video_item['filename']
    # ignore empty list
    if isinstance(video_item['label'], list) and (len(video_item['label']) == 0):
      # fill in dummy label if we don't have one
      label = -1
    else:
      idx = ['verb', 'noun'].index(self.action_type)
      label = video_item['label'][idx]

    return clips, label, clip_id

  def get_num_samples_per_cls(self):
    # return number of samples per class
    assert self.is_training
    num_samples_per_cls = [0] * self.num_classes
    for video_item in self.video_list:
      idx = ['verb', 'noun'].index(self.action_type)
      label = video_item['label'][idx]
      num_samples_per_cls[label] += 1
    return num_samples_per_cls


################################################################################
def create_video_dataset(dataset_config, train_transforms, val_transforms):
  """Get a dataset from dataset config
  return a supported dataset

  Example dataset config:
  {
    "name": "ucf101",
    "root_folder": "./data/",
    "ant_file": {"train" : None, "val" : None},
    "split": ["train", "val"],
    "num_classes": 101,
    "sample_pattern": [8, 8, 0],
    "drop_last_frames": 0,
  }
  """
  train_dataset, val_dataset = None, None
  dataset_name = dataset_config['name']

  # this seems redundant for now (as current datasets share the same loader)
  # but it is highly flexible ...
  for split in dataset_config['split']:
    if split == "train":
      train_dataset = {
        'ucf101' : partial(CustomVideoDataset,
                           root_folder=dataset_config['root_folder'],
                           ant_file=dataset_config['ant_file'][split],
                           num_classes=dataset_config['num_classes'],
                           sample_pattern=dataset_config['sample_pattern'],
                           is_training=True,
                           transforms=train_transforms,
                           protect_frames=dataset_config['drop_last_frames']),
        'hmdb51' : partial(CustomVideoDataset,
                           root_folder=dataset_config['root_folder'],
                           ant_file=dataset_config['ant_file'][split],
                           num_classes=dataset_config['num_classes'],
                           sample_pattern=dataset_config['sample_pattern'],
                           is_training=True,
                           transforms=train_transforms,
                           protect_frames=dataset_config['drop_last_frames']),
        '20bn-v2' : partial(CustomVideoDataset,
                            root_folder=dataset_config['root_folder'],
                            ant_file=dataset_config['ant_file'][split],
                            num_classes=dataset_config['num_classes'],
                            sample_pattern=dataset_config['sample_pattern'],
                            is_training=True,
                            transforms=train_transforms,
                            protect_frames=dataset_config['drop_last_frames']),
        'egtea' : partial(CustomVideoDataset,
                          root_folder=dataset_config['root_folder'],
                          ant_file=dataset_config['ant_file'][split],
                          num_classes=dataset_config['num_classes'],
                          sample_pattern=dataset_config['sample_pattern'],
                          is_training=True,
                          transforms=train_transforms,
                          protect_frames=dataset_config['drop_last_frames']),
        'epic-kitchens' : partial(EpicKitchens,
                                  root_folder=dataset_config['root_folder'],
                                  ant_file=dataset_config['ant_file'][split],
                                  num_classes=dataset_config['num_classes'],
                                  sample_pattern=dataset_config['sample_pattern'],
                                  is_training=True,
                                  transforms=train_transforms,
                                  action_type=dataset_config['action_type'],
                                  protect_frames=dataset_config['drop_last_frames']),
      }[dataset_name]

    elif split in ["val", "test"]:
      val_dataset = {
        'ucf101' : partial(CustomVideoDataset,
                           root_folder=dataset_config['root_folder'],
                           ant_file=dataset_config['ant_file'][split],
                           num_classes=dataset_config['num_classes'],
                           sample_pattern=dataset_config['sample_pattern'],
                           is_training=False,
                           transforms=val_transforms,
                           protect_frames=dataset_config['drop_last_frames'],
                           k_clips=dataset_config['val_clips']),
        'hmdb51' : partial(CustomVideoDataset,
                           root_folder=dataset_config['root_folder'],
                           ant_file=dataset_config['ant_file'][split],
                           num_classes=dataset_config['num_classes'],
                           sample_pattern=dataset_config['sample_pattern'],
                           is_training=False,
                           transforms=val_transforms,
                           protect_frames=dataset_config['drop_last_frames'],
                           k_clips=dataset_config['val_clips']),
        '20bn-v2' : partial(CustomVideoDataset,
                            root_folder=dataset_config['root_folder'],
                            ant_file=dataset_config['ant_file'][split],
                            num_classes=dataset_config['num_classes'],
                            sample_pattern=dataset_config['sample_pattern'],
                            is_training=False,
                            transforms=val_transforms,
                            protect_frames=dataset_config['drop_last_frames'],
                            k_clips=dataset_config['val_clips']),
        'egtea' : partial(CustomVideoDataset,
                          root_folder=dataset_config['root_folder'],
                          ant_file=dataset_config['ant_file'][split],
                          num_classes=dataset_config['num_classes'],
                          sample_pattern=dataset_config['sample_pattern'],
                          is_training=False,
                          transforms=val_transforms,
                          protect_frames=dataset_config['drop_last_frames'],
                          k_clips=dataset_config['val_clips']),
        'epic-kitchens' : partial(EpicKitchens,
                                  root_folder=dataset_config['root_folder'],
                                  ant_file=dataset_config['ant_file'][split],
                                  num_classes=dataset_config['num_classes'],
                                  sample_pattern=dataset_config['sample_pattern'],
                                  is_training=False,
                                  transforms=val_transforms,
                                  action_type=dataset_config['action_type'],
                                  protect_frames=dataset_config['drop_last_frames']),
      }[dataset_name]

    else:
      raise TypeError("Unsupported split")

  return train_dataset, val_dataset
