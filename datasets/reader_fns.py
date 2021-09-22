# Copyright (c) Facebook, Inc. and its affiliates.

"""Implementation of reader functions."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

from common.utils import get_video_info


# An abstract class to keep track of all reader type classes
class Reader(nn.Module):
    pass


class DefaultReader(Reader):
    def forward(self, video_path, start, end, fps, df_row, **kwargs):
        del df_row, fps  # Not needed here
        video_info = torchvision.io.read_video(video_path, start, end,
                                               **kwargs)
        # DEBUG see what is breaking
        logging.debug('Read %s from %s', video_info[0].shape, video_path)
        return video_info

    @staticmethod
    def get_frame_rate(video_path: Path) -> float:
        return get_video_info(video_path, ['fps'])['fps']


class VideoAsLabelOnehotReader(Reader):
    @staticmethod
    def get_frame_rate(video_path: Path) -> float:
        raise NotImplementedError('Not sure what it is here... TODO')

    def forward(self,
                video_path,
                start,
                end,
                fps,
                df_row,
                pts_unit='sec',
                num_classes=1000):
        """
        Return the video as a 1-hot representation of the actual labels.
        Args:
            video_path
            start: start time in sec
            end: end time in sec
            fps: frame rate of this video
            df_row: The data frame row corresponding to this video. Includes
                labels
            num_classes: Total number of classes for the 1-hot representation.
                Could just be a large number, should work too.
        Returns:
            video_feature of shape T x 1 x 1 x num_classes
        """
        del pts_unit, video_path, start, fps
        assert abs(end -
                   df_row['end']) < 0.1, 'For now just supporting last_clip'
        labels = df_row['obs_action_class'][:, 1]
        # Convert to 1-hot, TxC shape
        feats = nn.functional.one_hot(torch.LongTensor(labels), num_classes)
        return feats.unsqueeze(1).unsqueeze(1).float(), {}, {}
