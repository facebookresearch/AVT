# Copyright (c) Facebook, Inc. and its affiliates.

"""The Breakfast/50Salads dataset loader.
"""

from pathlib import Path
import logging
import pandas as pd
from tqdm import tqdm
import gzip
import numpy as np

import torch
import torch.nn as nn
import hydra
from hydra.types import TargetConf

from common.utils import get_video_info
from .base_video_dataset import BaseVideoDataset
from .reader_fns import Reader, DefaultReader


def load_mappings_file(fpath: Path) -> list:
    """
    Read the mappings file shared by Abu farha (CVPR'18) to read the class
    names.
    """
    res = []
    with open(fpath, 'r') as fin:
        for line in fin:
            res.append(line.rpartition(' ')[-1].strip())
    # convert to dict
    return dict(zip(res, range(len(res))))


def bundle_entry_to_video_fname_50salads(bundle_entry, root):
    del root  # not needed here
    # remove the "rgb-"
    video_id = bundle_entry.strip()[len('rgb-'):-(len('.txt'))]
    video_fname = f'rgb-{video_id}.avi'
    annot_fname = f'{video_id}-activityAnnotation.txt'
    return video_fname, annot_fname


def read_orig_50salads_annotations(videos: list, root: Path,
                                   action_classes: dict, annots_dir: Path,
                                   timestamps_dir: Path):
    all_segments = []
    for video in videos:
        video_fname, annot_fname = bundle_entry_to_video_fname_50salads(
            video.strip(), root)
        video_id = video.strip()[len('rgb-'):-(len('.txt'))]
        ts_fpath = f'timestamps-{video_id}.txt'
        frame_rate = get_video_info(Path(root) / video_fname, ['fps'])['fps']
        frame_ts = []
        # Read the "timestamp" of each frame
        with open(Path(timestamps_dir) / ts_fpath, 'r') as fin:
            for line in fin:
                frame_ts.append(int(line.partition(' ')[0]))
        first_start = len(frame_ts)
        last_end = 0
        with open(Path(annots_dir) / annot_fname, 'r') as fin:
            for line in fin:
                start_ts, end_ts, activity = line.split(' ')
                act_pre, _, act_post = activity.strip().rpartition('_')
                if not act_post in ['prep', 'core', 'post']:
                    # This is a coarse grained label, so ignore it
                    continue
                label = action_classes[act_pre]
                start = frame_ts.index(int(start_ts)) / frame_rate
                first_start = min(first_start, start)
                end = frame_ts.index(int(end_ts) + 1) / frame_rate
                last_end = max(last_end, end)
                all_segments.append((video_fname, start, end, label))
    return all_segments


def bundle_entry_to_video_fname_breakfast(bundle_entry, root):
    # remove the "rgb-"
    person, camera, _, topic = bundle_entry.strip()[:-len('.txt')].split('_')
    channels = ['']
    if camera.startswith('stereo'):
        channels = ['_ch0', '_ch1']  # ch0 is not always available
        camera = 'stereo'
    video_fname = f'{person}/{camera}/{person}_{topic}{{channel}}.avi'
    annot_fname = f'{video_fname}.labels'
    # Try both, if defined
    for channel in channels:
        if (Path(root) / annot_fname.format(channel=channel)).exists():
            video_fname = video_fname.format(channel=channel)
            annot_fname = annot_fname.format(channel=channel)
            break
    return video_fname, annot_fname


def read_orig_breakfast_annotations(videos: list, root: Path,
                                    action_classes: dict):
    all_segments = []
    for video in videos:
        video_fname, annot_fname = bundle_entry_to_video_fname_breakfast(
            video.strip(), root)
        # All videos are 15fps as says here:
        # https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/
        # though can read out from the video if needed..
        video_fps = 15
        with open(Path(root) / annot_fname, 'r') as fin:
            lines = [el.strip() for el in fin.readlines()]
            # No longer removing SIL -- based on email conversation
            # with Fadime, they keep everything, and drop any action segments
            # they don't have any context for prediction; so the SIL from the
            # beginning will be removed anyway.
            # # Ignore the lead and end SIL (silence) segments as per
            # # Sec 5.4 https://serre-lab.clps.brown.edu/wp-content/uploads/2014/05/paper_cameraReady-2.pdf (Unit recognition)
            # if lines[0].endswith('SIL'):
            #     lines = lines[1:]
            # if lines[-1].endswith('SIL'):
            #     lines = lines[:-1]
            for line in lines:
                start_end, activity = line.split(' ')
                start, end = start_end.split('-')
                if activity in action_classes:
                    label = action_classes[activity]
                else:
                    logging.warning(
                        'Didnt find %s. Ignoring... Ideally '
                        'should merge with the next action, or '
                        'just use abu_farha annotations which '
                        'already does that.', activity)
                    continue
                start = int(start) / video_fps
                end = int(end) / video_fps
                all_segments.append((video_fname, start, end, label))
    return all_segments


def read_abu_farha_annotations(videos: list,
                               root: Path,
                               action_classes: dict,
                               annots_dir: Path,
                               bundle_entry_to_vname_fn: TargetConf,
                               frame_rate: int = None):
    all_segments = []
    for video in tqdm(videos, desc='Loading Abu Farha annots'):
        video_fname, _ = hydra.utils.call(bundle_entry_to_vname_fn,
                                          video.strip(), root)
        if frame_rate is None:
            frame_rate = get_video_info(Path(root) / video_fname,
                                        ['fps'])['fps']
        with open(Path(annots_dir) / video.strip(), 'r') as fin:
            cur_action = ''  # dummy, will fire to insert action first
            for lno, line in enumerate(fin):
                if line == cur_action:
                    # Update the end time
                    # Using lno + 1 to avoid any gaps between the clips, which
                    # would lead to the -1 clips, making it harder for the
                    # model to learn
                    # Update the last added segment's end time point to this
                    # frame
                    all_segments[-1][-2] = (lno + 1) / frame_rate
                    continue
                # Else a new action is starting, add to the segments
                cur_action = line
                label = action_classes[cur_action.strip()]
                all_segments.append([
                    video,
                    video_fname,
                    lno / frame_rate,  # start
                    (lno + 1) / frame_rate,  # end
                    label,
                ])
    return all_segments


def init_df(bundle_fpath: Path, annot_reader_fn: TargetConf, root: Path,
            action_classes: dict):
    with open(bundle_fpath, 'r') as fin:
        videos = fin.readlines()
        # Remove the "#bundle.txt" line from top
        assert videos[0].startswith('#')
        videos = videos[1:]
    all_segments = hydra.utils.call(annot_reader_fn,
                                    videos,
                                    root,
                                    action_classes,
                                    _recursive_=False)
    dataframe = pd.DataFrame(all_segments,
                             columns=[
                                 'video_bundle_name', 'video_path', 'start',
                                 'end', 'action_class'
                             ])
    dataframe = dataframe.astype(dtype={
        'start': 'float16',
        'end': 'float16',
        'video_path': 'object',
    })
    return dataframe


class Breakfast50Salads(BaseVideoDataset):
    """Wrapper for Univ of Dundee 50Salads, or Bonn Breakfast dataset."""
    def __init__(
            self,
            which: str,  # specify which of BF or 50S
            root: Path,
            splits_dir: Path,
            classes_fpath: Path,
            is_train: bool = True,
            fold: int = 1,
            annot_reader_fn: TargetConf = None,
            **kwargs):
        bundle_fpath = (
            Path(splits_dir) /
            f'{"train" if is_train else "test"}.split{fold}.bundle')
        self.which = which
        if self.which == '50Salads':
            assert 1 <= fold <= 5
        elif self.which == 'Breakfast':
            assert 1 <= fold <= 4
        else:
            raise NotImplementedError(f'Unknown type {which}')
        action_classes = load_mappings_file(classes_fpath)
        dataframe = init_df(bundle_fpath, annot_reader_fn, root,
                            action_classes)
        kwargs['action_classes'] = action_classes
        kwargs['label_type'] = 'action'
        super().__init__(dataframe, root=root, **kwargs)


class FormatReader(nn.Module):
    pass


class GZFormatReader(FormatReader):
    def forward(self, path, start_frame, end_frame):
        feats = []
        with gzip.open(str(path).replace('.txt', '.gz'), 'r') as fin:
            for lno, line in enumerate(fin):
                if lno >= start_frame and lno <= end_frame:
                    feats.append(
                        [float(el) for el in line.strip().split(b' ')])
        feats = torch.FloatTensor(feats)
        return feats


class NPYFormatReader(FormatReader):
    def forward(self, path, start_frame, end_frame):
        feats = np.load(str(path).replace('.txt', '.npy'))
        start_frame = max(start_frame, 0)
        end_frame = min(end_frame, feats.shape[1])
        feats_subset = feats[:, start_frame:(end_frame + 1)]
        return torch.from_numpy(feats_subset.transpose())


class SenerFeatsReader(Reader):
    def __init__(self, feat_dir: Path, format_reader: FormatReader):
        super().__init__()
        self.feat_dir = Path(feat_dir)
        # No need to init the reader again, will be done recursively
        self.format_reader = format_reader

    def get_frame_rate(self, *args, **kwargs) -> float:
        # Use the actual frame rate, since I guess that's what is used in the
        # Abu Farha annotations, which is what the features here correspond
        # to as well.
        return DefaultReader.get_frame_rate(*args, **kwargs)

    def forward(self,
                video_path: Path,
                start_sec: float,
                end_sec: float,
                fps: float,
                df_row: pd.DataFrame,
                pts_unit='sec'):
        """
        Returns:
            feats: (T, 1, 1, C)  -- features shaped like a video
        """
        del pts_unit, video_path  # Not supported here
        vidname = df_row['video_bundle_name'].strip()
        start_frame = int(round(start_sec * fps - 1))
        end_frame = int(round(end_sec * fps - 1))
        feats = self.format_reader(self.feat_dir / vidname, start_frame,
                                   end_frame)
        return feats.unsqueeze(1).unsqueeze(1), {}, {}
