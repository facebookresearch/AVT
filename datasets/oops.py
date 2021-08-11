"""The Epic Kitchens dataset loaders."""

from typing import List, Dict
import logging
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

from .base_video_dataset import BaseVideoDataset
from common.utils import get_video_info

LBL_INTENTIONAL = 0
LBL_TRANSITIONAL = 1
LBL_UNINTENTIONAL = 2


class Oops(BaseVideoDataset):
    """OOPS dataloader."""
    def __init__(
            self,
            root: Path,
            splits_file: Path,
            transition_times_json: Path,
            keep_unlabeled: bool = True,
            **other_kwargs,
    ):
        """
        Args:
            splits_file: The train/test txt files
            transition_times_json: The JSON with all the transition times
        """
        with open(splits_file, 'r') as fin:
            clip_fpaths = fin.read().splitlines()
        with open(transition_times_json, 'r') as fin:
            transition_json = json.load(fin)
        df, classes = self._init_df(root, clip_fpaths, transition_json,
                                    keep_unlabeled)
        other_kwargs['action_classes'] = classes
        other_kwargs['label_type'] = 'action'
        super().__init__(df, root, **other_kwargs)

    @staticmethod
    def _select_trans_time(trans_times):
        return np.median(trans_times)

    @classmethod
    def _check_valid_video(cls, vid_len, trans_times, rel_trans_times):
        """
        Implements the logic from
        https://github.com/cvlab-columbia/oops/blob/ef2d93ff301b33c77617d16f7150c13dec99f82d/dataloader.py#L140
        to check if this clip is supposed to be in the set or not.
        """
        sel_rel_trans_times = cls._select_trans_time(rel_trans_times)
        sel_trans_times = cls._select_trans_time(trans_times)
        if (sel_trans_times < 0 or sel_rel_trans_times > 0.99
                or sel_rel_trans_times < 0.01 or vid_len < 3.2
                or vid_len > 30):
            return False
        return True

    def _init_df(self, root: Path, clips_fpath: List[str],
                 transition_json: Dict,
                 keep_unlabeled: bool) -> (pd.DataFrame, List[str]):
        per_clip_df = []
        unlabeled_videos = 0
        df_dtypes = {
            'video_path': 'category',
            'start': 'float16',
            'end': 'float16',
            'action_class': 'category',
        }
        for clip_fpath in tqdm(clips_fpath, desc='Loading annots'):
            clip_fpath_full = clip_fpath + '.mp4'
            if clip_fpath in transition_json:
                vid_len = transition_json[clip_fpath]['len']
                trans_times = transition_json[clip_fpath]['t']
                rel_trans_times = transition_json[clip_fpath]['rel_t']
                if not keep_unlabeled and not self._check_valid_video(
                        vid_len, trans_times, rel_trans_times):
                    continue
            elif keep_unlabeled:
                # Unlabeled segment, happens during training
                vid_len = get_video_info(
                    Path(root) / clip_fpath_full, ['len'])['len']
                trans_times = []
                unlabeled_videos += 1
            else:
                continue  # Skip this video
            segments = self.split_into_segments(vid_len, trans_times)
            rows = [
                sum(el, [])
                for el in zip([[clip_fpath_full]] * len(segments), segments)
            ]
            dataframe = pd.DataFrame(
                rows, columns=['video_path', 'start', 'end', 'action_class'])
            dataframe = dataframe.astype(dtype=df_dtypes)
            per_clip_df.append(dataframe)
        logging.warning('Found %d/%d unlabeled vids', unlabeled_videos,
                        len(per_clip_df))
        final_df = pd.concat(per_clip_df)
        final_df.reset_index(drop=True, inplace=True)
        # For some reason the following doesn't work, so have to loop
        # final_df.astype(dtype=df_dtypes)
        for colname, dtype in df_dtypes.items():
            final_df.loc[:, colname] = final_df.loc[:, colname].astype(dtype)
        classes = {
            'intentional': LBL_INTENTIONAL,
            'transitional': LBL_TRANSITIONAL,
            'unintentional': LBL_UNINTENTIONAL
        }
        return final_df, classes

    @classmethod
    def split_into_segments(cls,
                            vid_len: float,
                            transition_times: List[float],
                            clip_len: float = 1,
                            clip_stride: float = 0.25) -> List[List[float]]:
        """
        Args:
            vid_len (float): The video duration in seconds
            transition_times (List[float]): The transition times labeled by the
                3 labelers.
            clip_len: The length of clips to break into, in seconds
            clip_stride: The stride to use when selecting clips
        Returns:
            A list of segments, where each segment is (start, end, class_lbl)
        """
        label_space = [LBL_INTENTIONAL, LBL_TRANSITIONAL, LBL_UNINTENTIONAL]
        if len(transition_times) == 0:
            # This clip was not labeled, so the label space should be -1s
            label_space = [-1, -1, -1]
        # Unclear what the original author did, following what MemDPC (ECCV'20)
        # author Tengda Han mentioned over email.
        # Take the median if > 1 labeled. If none labeled, assume no
        # transitions.
        transition_times = [el for el in transition_times if el >= 0]
        if len(transition_times) > 0:
            transition_time = cls._select_trans_time(transition_times)
        else:
            transition_time = float('inf')  # i.e. all intentional
        segments = []
        for start in np.arange(0, vid_len - clip_len, clip_stride):
            end = start + clip_len
            if start <= transition_time and end >= transition_time:
                cls_lbl = label_space[1]  # Transition
            elif end < transition_time:
                cls_lbl = label_space[0]  # Intentional
            elif start > transition_time:
                cls_lbl = label_space[2]  # Unintentional
            else:
                raise ValueError('This should not happen')
            segments.append([start, end, cls_lbl])
        return segments
