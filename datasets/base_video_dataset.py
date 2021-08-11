"""The Epic Kitchens dataset loaders."""

from typing import Tuple, Union, Sequence, Dict
import logging
from pathlib import Path
from collections import OrderedDict
import operator
from multiprocessing import Manager
import math
import h5py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision

from omegaconf import OmegaConf
import hydra
from hydra.types import TargetConf

from common.utils import get_video_info, get_world_size, get_rank

SAMPLE_STRAT_CNTR = 'center_clip'
SAMPLE_STRAT_RAND = 'random_clip'
SAMPLE_STRAT_LAST = 'last_clip'
SAMPLE_STRAT_FIRST = 'first_clip'
FUTURE_PREFIX = 'future'  # to specify future videos

# This is specific to EPIC kitchens
RULSTM_TSN_FPS = 30.0  # The frame rate the feats were stored by RULSTM ppl

# This is important for some datasets, like Breakfast, where reading using the
# pyAV reader leads to jerky videos for some reason. This requires torchvision
# to be compiled from source, instructions in the top level README
torchvision.set_video_backend('video_reader')


def convert_to_anticipation(df: pd.DataFrame,
                            root_dir: Sequence[Path],
                            tau_a: float,
                            tau_o: float,
                            future_clip_ratios: Sequence[float] = (1.0, ),
                            drop_style='correct'
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Based on the definition in the original paper
        https://arxiv.org/pdf/1804.02748.pdf, convert the start and end
        video times to as used in anticipation.
    tau_a (float): Anticipation time in seconds. By default -1, since
        we train the model to do action recognition, in which case
        the model sees a clip that finishes tau_a seconds before
        the action to be anticipated starts. This is as per defn
        in https://arxiv.org/pdf/1804.02748.pdf (pg 15)
    tau_o (float): The amount of video to see before doing the
        anticipation. In the original paper they used 1s
        (https://arxiv.org/pdf/1804.02748.pdf), but in further ones
        they use 3.5 (https://arxiv.org/pdf/1905.09035.pdf).
    future_clip_ratios: A list of ratios (< 1.0) of tau_a, to define what clips
        to set as the future clips. These will be used when returning future
        clips. Ideally the labels should be adjusted to match this too, but
        not doing that for now.
    """
    del root_dir
    if tau_a == -999:
        # No anticipation, just simple recognition
        # still add the orig_start and orig_end, future etc
        # so the future prediction baseline can do the case where not future
        # is predicted.
        # This will ensure the future clip ends up being the same as current
        tau_a = df.loc[:, 'start'] - df.loc[:, 'end']
        tau_o = df.loc[:, 'end'] - df.loc[:, 'start']
    logging.debug(
        'Converting data to anticipation with tau_a=%s and '
        'tau_o=%s.', tau_a, tau_o)
    # Copy over the current start and end times
    df.loc[:, 'orig_start'] = df.start
    df.loc[:, 'orig_end'] = df.end
    # Convert using tau_o and tau_a
    df.loc[:, 'end'] = df.loc[:, 'start'] - tau_a
    df.loc[:, 'start'] = df.loc[:, 'end'] - tau_o
    # Add the future clips
    for i, future_clip_ratio in enumerate(future_clip_ratios):
        if future_clip_ratio == -999:
            # A spl number to use the exact current clip as the future
            df.loc[:, f'{FUTURE_PREFIX}_{i}_start'] = df.loc[:, 'start']
            df.loc[:, f'{FUTURE_PREFIX}_{i}_end'] = df.loc[:, 'end']
        elif future_clip_ratio > -10 and future_clip_ratio < 10:
            eff_tau_a = tau_a * future_clip_ratio
            df.loc[:, f'{FUTURE_PREFIX}_{i}_start'] = (df.loc[:, 'end'] +
                                                       eff_tau_a)
            df.loc[:, f'{FUTURE_PREFIX}_{i}_end'] = (
                df.loc[:, f'future_{i}_start'] + tau_o)
        else:
            raise ValueError(f'Seems out of bound {future_clip_ratio}')
    # RULSTM (in EpicKitchens) drops certain clips that do not have enough
    # stuff before to set up anticipation (8 data points). So drop them too.
    # However, I think there is a bug in RULSTM, here:
    # https://github.com/fpv-iplab/rulstm/blob/96e38666fad7feafebbeeae94952dba24771e512/RULSTM/dataset.py#L164
    # They are copying over the last visible frame, but that frame might be
    # within the 1s to the actual action -- since they consider predicting
    # even from just 0.25s before. So the equivalent setup here would be to
    # allow anything that ends before 0.25s to the action, and let them see the
    # frame. However, the correct way would be to only drop things that have no
    # frames visible.
    # first frame seconds
    f1_sec = 1 / RULSTM_TSN_FPS
    old_df = df
    if drop_style == 'correct':
        # at least 1 frame
        df = df[df.end >= f1_sec]
    elif drop_style == 'full_context_in':
        # All frames should be in
        df = df[df.start >= f1_sec]
    elif drop_style == 'rulstm':
        # Anything that would be able to see at least 1 frame 0.25s before the
        # clip. Keep those.
        # A point before orig_start,
        df = df[df.orig_start >= 0.25 + f1_sec]
        for i, row in df.iterrows():
            if row.end >= f1_sec:
                continue
            earliest_avail_frame_sec = min(
                np.arange(row.orig_start - 0.25, f1_sec - 0.0001, -0.25))
            df.at[i, 'start'] = 0
            df.at[i, 'end'] = earliest_avail_frame_sec
    elif drop_style == 'action_banks':
        # Based on their dataset_anticipation:__get_snippet_features()
        df = df[df.end >= 2]
    else:
        raise NotImplementedError(f'Unknown style {drop_style}')
    discarded_df = pd.concat([old_df, df]).drop_duplicates(subset=['uid'],
                                                           keep=False)
    df.reset_index(inplace=True, drop=True)
    return df, discarded_df


def convert_to_dense_anticipation_runtime(
        df_row: pd.DataFrame, root_dir: Sequence[Path], df: pd.DataFrame,
        addl_df_proc_for_dense_fn: callable) -> pd.DataFrame:
    """
    Collect all clips per video into one data frame, along with video length.
    The actual splitting into video and predictions will be done in the reader
    thread.
    """
    del root_dir
    df_relevant_subset = df[df.video_path == df_row.video_path]
    # Assuming there is an action_class column in the data frame (and not
    # verb, noun). Could be easily extended but for now just assuming this
    # since unlikely I'll need verb/noun for dense anticipation.
    # Do this properly
    df_relevant_subset = addl_df_proc_for_dense_fn(df_relevant_subset)
    annotations = df_relevant_subset.loc[:,
                                         ('start', 'end',
                                          'action_class')].sort_values('start')
    new_df_row = pd.DataFrame([[df_row.video_path, annotations.values]],
                              columns=('video_path', 'dense_labels')).loc[0]
    return new_df_row


# TODO reuse the above function, lots of functionlity can be shared
def convert_to_dense_anticipation(df: pd.DataFrame,
                                  root_dir: Sequence[Path]) -> pd.DataFrame:
    """
    Collect all clips per video into one data frame, along with video length.
    The actual splitting into video and predictions will be done in the reader
    thread.
    """
    del root_dir
    new_rows = []
    for video_path, rows in df.groupby('video_path'):
        # Assuming there is an action_class column in the data frame (and not
        # verb, noun). Could be easily extended but for now just assuming this
        # since unlikely I'll need verb/noun for dense anticipation.
        annotations = rows.loc[:, ('start', 'end',
                                   'action_class')].sort_values('start')
        new_rows.append((video_path, annotations.values))
    new_df = pd.DataFrame(new_rows, columns=('video_path', 'dense_labels'))
    return new_df, None


def break_segments_by_duration(duration, label, segment_len):
    """
    Return a list of [(duration, label1, label2, ...), ...] such that each
        duration is == segment_len if set.
        Note label can be a scalar or vector (in case of multi-label cls)
    """
    if not isinstance(label, list):
        label = [label]
    if segment_len is None:
        return [[duration] + label], duration
    nseg = int(round(duration / segment_len))
    return [[segment_len] + label for _ in range(nseg)], nseg * segment_len


def dense_labels_to_segments(
        dense_labels,
        segment_start_time,
        segment_end_time,
        # -1 => get as many as possible
        pred_steps=-1,
        fixed_duration=None,
        dummy_label=-1):
    segments = []
    for start, end, label in dense_labels:
        if end < segment_start_time:
            # Then this action is past, not relevant here
            # should only happen for the pos-1 action being added
            continue
        if start > segment_end_time:
            # This action starts after the segment, so leave this
            continue
        # should not look at anything beyond the segment end time
        end = min(end, segment_end_time)
        if start > segment_start_time:
            # Add an empty slot of action, for the time where we don't know
            # what happened. Setting the action itself to be -1, so the
            # model can predict whatever and it won't be penalized
            new_segments, duration_used = break_segments_by_duration(
                start - segment_start_time, dummy_label, fixed_duration)
            segments += new_segments
            segment_start_time += duration_used
        new_segments, duration_used = break_segments_by_duration(
            end - segment_start_time, label, fixed_duration)
        segments += new_segments
        segment_start_time += duration_used
        if fixed_duration is None:
            assert segment_start_time == end
        if pred_steps > 0 and len(segments) >= pred_steps:
            break
    if pred_steps > 0:
        segments = segments[:pred_steps]
        # Pad it with dummy intervals for batching, if lower
        if not isinstance(dummy_label, list):
            dummy_label = [dummy_label]
        segments += [[-1] + dummy_label] * (pred_steps - len(segments))
    return segments


def process_df_for_dense_anticpation(
        df_row: pd.DataFrame,
        root_dir: Sequence[Path],
        rng: np.random.Generator,
        label_type: Sequence[str],
        frames_per_clip: int,
        frame_rate: float,
        sample_strategy: str,
        dummy_label: Union[list, int],
        # KWARGS
        obs_ratio: Union[float, Tuple[float]] = None,
        pred_steps: int = None,
        fixed_duration: float = None,
        fixed_duration_obs: float = None) -> pd.DataFrame:
    """
    Given a row from the df, converts into an actual clip and corr label.
    Args:
        dummy_label: To be used when adding segments that are not labeled.
        obs_ratio: If a scalar ratio, then that much of the clip will be
            observed. If a list, it will randomly pick from the
            range.
        pred_steps: Specify the future to be predicted in terms
            of number of future steps. Not doing ratio as that will add
            unnecessary complexity -- might as well just roll out multiple
            multiple steps until it is done
        fixed_duration: If set, will only make segments of that duration (sec).
            If an action extends into a fractional segment, it wil be rounded
            (hence removed if it occupies < half of duration). So the duration
            prediction problem here would become trivial. Do not do this for
            evaluation data since it will lead to inaccurate results due to
            segments being dropped due to quantization.
            It should typically correspond to the temporal resolution of each
            video feature. Eg for 32frame, 15fps and temporal resolution 4 =>
            Each corresponds to ~0.5s, so that should this be as well.
        fixed_duration_obs: Same defn as fixed_duration, but for the observed
            part of the video.
    """
    if fixed_duration_obs is None and fixed_duration is not None:
        fixed_duration_obs = fixed_duration
    assert sample_strategy == SAMPLE_STRAT_LAST, (
        'Dense anticipation should be able to see the last clip before '
        'it has to start anticipating.')
    video_path = df_row['video_path']
    # This much amount of video context has to be provided, so the obs_ratio
    # better match that.
    abs_video_path = get_abs_path(root_dir, video_path)
    vid_info = get_video_info(abs_video_path, ['len', 'fps'])
    vid_len = vid_info['len']
    vid_fps = vid_info['fps'] if frame_rate is None else frame_rate
    if vid_fps == 0 or vid_len == 0:
        logging.error('%s video was not readable. Ignoring', df_row)
        return None  # Don't want to deal with this
    video_len_reqd = frames_per_clip / vid_fps
    obs_ratio_reqd = video_len_reqd / vid_len
    if not OmegaConf.get_type(obs_ratio) == list:
        # So a range with only one possible output
        obs_ratio = [obs_ratio, obs_ratio]
    obs_ratio = list(obs_ratio)  # In case it was ListConfig
    if obs_ratio_reqd > obs_ratio[1]:
        logging.warning(
            'For %s, the video length (%f) was too short '
            'to get the %d frames at %d fps at %f obs ratio. '
            'Will end up sampling fewer frames + padding etc.', abs_video_path,
            vid_len, frames_per_clip, vid_fps, obs_ratio[1])
        obs_ratio_reqd = obs_ratio[1]
    obs_ratio[0] = max(obs_ratio_reqd, obs_ratio[0])
    # Now select
    obs_ratio = rng.uniform(*obs_ratio)
    assert 0.0 < obs_ratio <= 1.0
    vid_end = vid_len * obs_ratio
    vid_start = max(vid_end - video_len_reqd, 0)
    # the rows are sorted by the start time, so find the first action after the
    # video clip being considered
    all_dense_labels = df_row['dense_labels']
    pos = all_dense_labels[:, 0].searchsorted(vid_end)
    # starting from the potentially overlapping action, output the
    # action class, and duration, from vid_end
    segments_future = dense_labels_to_segments(
        all_dense_labels[max(pos - 1, 0):], vid_end, float('inf'), pred_steps,
        fixed_duration, dummy_label)
    segments_obs = dense_labels_to_segments(all_dense_labels[:max(pos, 0)], 0,
                                            vid_end, -1, fixed_duration_obs,
                                            dummy_label)
    data_lst = [[
        video_path,
        vid_len,
        vid_start,
        vid_end,
        np.array(segments_future, dtype=np.float32),
        np.array(segments_obs, dtype=np.float32),
    ]]
    assert len(label_type) == 1, 'Larger not implemented yet'
    col_names = ('video_path', 'video_len', 'start', 'end',
                 f'{label_type[0]}_class', f'obs_{label_type[0]}_class')
    return pd.DataFrame(data_lst, columns=col_names).loc[0]


def get_abs_path(root_dirs: Sequence[Path], fpath: Path):
    """
    Combine the fpath with the first root_dir it exists in.
    """
    res_fpath = None
    for root_dir in root_dirs:
        res_fpath = root_dir / fpath
        if res_fpath.exists():
            return res_fpath
    logging.warning('Did not find any directory for %s [from %s]', fpath,
                    root_dirs)
    return res_fpath  # return the last one for now


def read_saved_results_uids(resfpath: Path):
    if not resfpath.exists():
        return set([])
    with h5py.File(resfpath, 'r') as fin:
        res = fin['uid'][()].tolist()
    # For fast lookup when filtering (makes big difference)
    return set([el.decode() for el in res])


def dense_clip_sampler(df: pd.DataFrame,
                       root_dir: Sequence[Path],
                       clip_len: Union[float, str] = 'mean_action_len',
                       stride: float = 1.0,
                       shard_per_worker: bool = False,
                       keep_orig_clips: bool = True,
                       featext_skip_done: bool = False):
    """
    Add clips to the data frame sampling the videos densely from the video.
        This function is also compatible with the convert_to_anticipation_fn
        to extract features etc. The class label for those clips
        is -1, it's mostly just used for SSL/feat ext.
    Args:
        stride (float): stride in seconds on how the clips are sampled.
        shard_per_worker (bool): If true, create subset DF for this process
        featext_skip_done (bool): Set this to true only when extracting
            features. This will go through saved results files and check
            what features have been stored and skip those from populating
            into the dataset to the computed, hence continuing from what
            has already been done.
    """
    uniq_videos = sorted(list(df.video_path.unique()))
    if shard_per_worker:
        world_size = get_world_size()
        rank = get_rank()
        vids_per_shard = int(math.ceil(len(uniq_videos) / world_size))
        uniq_videos = uniq_videos[(vids_per_shard * rank):min((
            (rank + 1) * vids_per_shard), len(uniq_videos))]
    skip_uids = []
    if featext_skip_done:
        # TODO replace with RESULTS_SAVE_DIR
        skip_uids = read_saved_results_uids(Path(f'./results/{get_rank()}.h5'))
        logging.info('Found %d done UIDs, skipping those', len(skip_uids))
    if clip_len == 'mean_action_len':
        clip_len = np.mean(df.end - df.start)
    new_rows = []
    total_possible_clips = 0
    for vid_path in uniq_videos:
        end_s = get_video_info(get_abs_path(root_dir, vid_path),
                               ['len'])['len']
        new_ends = np.arange(0, end_s, stride)
        for new_end in new_ends:
            total_possible_clips += 1
            uid = f'{vid_path.stem}_{new_end}'
            if uid in skip_uids:
                continue
            new_rows.append({
                'participant_id': vid_path.stem.split('_')[0],
                'narration': '',
                'video_id': vid_path.stem,
                'start': new_end - clip_len,
                'end': new_end,
                'verb_class': -1,
                'noun_class': -1,
                'action_class': -1,
                'video_path': vid_path,
                'uid': uid,
            })
    logging.info('Out of %d total potential clips, kept %d',
                 total_possible_clips, len(new_rows))
    new_df = pd.DataFrame(new_rows)
    if keep_orig_clips:
        # Convert the uid to str since the new UIDs being added to the new DF
        # are all strings
        df.uid = df.uid.astype('str')
        new_df = pd.concat([df, new_df])
        new_df.reset_index(drop=True, inplace=True)
    return new_df, pd.DataFrame([])


class BaseVideoDataset(torch.utils.data.Dataset):
    """Basic video dataset."""
    def __init__(
            self,
            df,
            root: Union[Sequence[Path], Path] = Path(''),
            frames_per_clip: int = 32,
            frame_rate: float = None,
            subclips_options: Dict[str, float] = None,
            load_seg_labels: bool = False,
            load_long_term_future_labels: int = 0,
            reader_fn: TargetConf = {
                '_target_': 'datasets.reader_fns.DefaultReader'
            },
            transform: torchvision.transforms.Compose = None,
            # verb, noun, action
            label_type: Union[str, Sequence[str]] = 'verb',
            return_future_clips_too: bool = False,
            sample_strategy: str = SAMPLE_STRAT_RAND,
            sample_strategy_future: str = SAMPLE_STRAT_FIRST,
            conv_to_anticipate_fn: TargetConf = None,
            conv_to_anticipate_fn_runtime: TargetConf = None,
            process_df_before_read_fn: TargetConf = None,
            sample_clips_densely: bool = False,
            sample_clips_densely_fn: TargetConf = None,
            random_seed: int = 42,
            verb_classes: dict = {},
            noun_classes: dict = {},
            action_classes: dict = {},
            repeat_data_times: float = 1.0,
            dummy_label: Union[list, int] = -1,
            class_balanced_sampling: bool = False,
            return_unsampled_video: bool = False,
            uid_subset: list = None):
        """
        Args:
            df: DataFrame of all the data (see a subclass for example/fmt).
                Must be passed in through super() when init-ing the subclass
            root: The path where all the videos are stored, will be
                prepended to video path.
            load_seg_labels: Set to true to load frame level segmentation
                labels that can be jointly used to finetune the model for
                classification as well.
            load_long_term_future_labels: Set to the number of future labels
                to also return, from where load_seg_labels stops. This is
                used for long-term rollout visualization and getting GT for
                those.
            transform: The video transform function
            return_future_clips_too: Set to true to also return future, actual
                action clips along with the tau_o clips. This is used for SSL.
            sample_strategy_future: Samplnig strategy used to return future
                clips, if return_future_clips_too is set.
            conv_to_anticipate_fn: The function that converts to anticipation.
            conv_to_anticipate_fn_runtime: A similar fn as ^, but is applied
                in the getitem function. Useful if don't want to do upfront,
                for large datasets like HowTo.
            sample_clips_densely: Add clips to the data frame sampling the
                videos densely between the first and the last labeled clip.
                The class label for those clips is -1, it's mostly just
                used for SSL.
            sample_clips_densely_fn: If this function is set, then no need
                to set the sample_clip_densely to true. It will use this fn
                to densify.
            process_df_before_read_fn: A function that is applied to the
                data frame[idx] before it's used for reading the video etc.
            repeat_data: Set to number of times to repeat the data in the
                DF. This is used if the epoch is too small, so can roll
                through the data more than once during a single epoch. Also
                helps if the preprocessing at read time effectively means
                each data item corresponds to > 1 data items really through
                random cropping etc.
            class_balanced_sampling: If true, it will sample from the data
                such that each class appears approximately equally -- so using
                the distribution of labels, it will try to enforce unformity.
                This is independent of adding loss weights based on how
                often a class appears, which is done in train_eval_ops.
            return_unsampled_video (bool): If true, return the video clip
                before it was sub-sampled to match the FPS requirements.
                So if experimenting at 1FPS, this will also return the
                original frame rate clip that could be used for visualization.
                MUST use batch size = 1 if using this, since it will return
                different length videos which won't be batch-able.
            uid_subset: Make a dataset keeping only those UIDs. This is useful
                for visualization code when I just want to visualize on
                specific clips.
        """
        super().__init__()
        # Based on https://github.com/pytorch/pytorch/issues/13246#issuecomment-612396143,
        # trying to avoid mem leaks by wrapping lists and dicts in this
        # manager class objects
        manager = Manager()
        self.root = root
        # Convert to list if not already
        if OmegaConf.get_type(self.root) != list:
            self.root = [self.root]
        self.root = [Path(el) for el in self.root]
        self.subclips_options = subclips_options
        self.load_seg_labels = load_seg_labels
        self.load_long_term_future_labels = load_long_term_future_labels
        # TODO: Move away from DataFrames... based on
        # https://github.com/pytorch/pytorch/issues/5902#issuecomment-374611523
        # it seems data frames are not ideal and cause memory leaks...
        self.df = df  # Data frame that will contain all info
        # To be consistent with EPIC, add a uid column if not already present
        if 'uid' not in self.df.columns:
            self.df.loc[:, 'uid'] = range(1, len(self.df) + 1)
        if sample_clips_densely or sample_clips_densely_fn:
            if sample_clips_densely_fn is None:
                # Use the default parameters. Keeping this sample_clips_densely
                # param to be backward compatible.
                sample_clips_densely_fn = {
                    '_target_':
                    'datasets.base_video_dataset.dense_clip_sampler',
                }
            self.df, _ = hydra.utils.call(sample_clips_densely_fn, self.df,
                                          self.root)
        assert not (conv_to_anticipate_fn and conv_to_anticipate_fn_runtime), (
            'At max only one of these should be set.')
        self.conv_to_anticipate_fn = conv_to_anticipate_fn
        self.discarded_df = None
        if conv_to_anticipate_fn is not None:
            self.df, self.discarded_df = hydra.utils.call(
                conv_to_anticipate_fn, self.df, self.root)
            logging.info('Discarded %d elements in anticipate conversion',
                         len(self.discarded_df))
        # this is an alternate implementation of ^, run in getitem,
        # useful for large datasets like HowTo, but won't work for
        # any dataset where you want to run testing
        self.conv_to_anticipate_fn_runtime = conv_to_anticipate_fn_runtime
        # This is used in the output files for EPIC submissions
        self.challenge_type = 'action_recognition'
        if conv_to_anticipate_fn or conv_to_anticipate_fn_runtime:
            # If either of these are set, this must be an anticipation setup
            self.challenge_type = 'action_anticipation'
        self.repeat_data_times = repeat_data_times
        self.process_df_before_read_fn = process_df_before_read_fn
        self.frames_per_clip = frames_per_clip
        self.frame_rate = frame_rate
        self.reader_fn = hydra.utils.instantiate(reader_fn)
        self.transform = transform
        self.label_type = label_type
        if OmegaConf.get_type(self.label_type) != list:
            # Will use the first one for the balancing etc
            self.label_type = [self.label_type]
        self.verb_classes = manager.dict(verb_classes)
        self.noun_classes = manager.dict(noun_classes)
        self.action_classes = manager.dict(action_classes)
        self.return_future_clips_too = return_future_clips_too
        self.sample_strategy = sample_strategy
        self.sample_strategy_future = sample_strategy_future
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)
        self.dummy_label = dummy_label
        if isinstance(self.dummy_label, list):
            self.dummy_label = manager.list(self.dummy_label)
        # Precompute some commonly useful stats
        self.classes_counts = manager.dict(self._compute_stats_cls_counts())
        self.class_balanced_sampling = class_balanced_sampling
        if self.class_balanced_sampling:
            # sort the data frame by labels, to allow for the runtime
            # remapping of idx
            assert len(self.label_type) == 1, 'Not supported more yet'
            self.df.sort_values(by=self.label_type[0] + '_class', inplace=True)
        self.return_unsampled_video = return_unsampled_video
        if self.return_unsampled_video:
            logging.warning('Make sure using batch size = 1 since '
                            'return_unsampled_videos is set to True.')
        # store the full DF so far in df_before_subset, since I will now keep a
        # subset that may be used for testing etc. df_before_subset will be
        # used to get intermediate labels for L_cls etc still (even during
        # visualizations sometimes I want to show that)
        self.df_before_subset = self.df
        if uid_subset is not None:
            # Select a subset in the order of the list
            self.df = self.df.iloc[pd.Index(
                self.df.uid).get_indexer(uid_subset)].reset_index(drop=True)

    def _compute_stats_cls_counts(self):
        """
        Compute some stats that are useful, like ratio of classes etc.
        """
        all_classes_counts = {}
        for tname, tclasses in self.classes.items():
            col_name = tname + '_class'
            if col_name not in self.df:
                logging.warning('Didnt find %s column in %s', col_name,
                                self.df)
                continue
            lbls = np.array(self.df.loc[:, col_name].values)
            # not removing the -1 labels, it's a dict so keep all of them.
            classes_counts = {
                cls_id: np.sum(lbls == cls_id)
                for _, cls_id in [('', -1)] + tclasses.items()
            }
            assert sum(classes_counts.values()) == len(self.df)
            all_classes_counts[tname] = classes_counts
        logging.debug('Found %s classes counts', all_classes_counts)
        return all_classes_counts

    @property
    def classes(self) -> OrderedDict:
        return OrderedDict([(tname,
                             operator.attrgetter(tname + '_classes')(self))
                            for tname in self.label_type])

    @property
    def classes_manyshot(self) -> OrderedDict:
        """This is subset of classes that are labeled as "many shot".
        These were used in EPIC-55 for computing recall numbers. By default
        using all the classes.
        """
        return self.classes

    @property
    def class_mappings(self) -> Dict[Tuple[str, str], torch.FloatTensor]:
        return {}

    @property
    def primary_metric(self) -> str:
        """
        The primary metric for this dataset. Datasets should override this
        if top1 is not the metric to be used. This is the key to the dictionary
        in the func/train.py when accuracies are computed. Some of these come
        from the notebook utils.
        """
        return 'final_acc/action/top1'

    def _get_text(self, df_row, df_key='narration'):
        if df_key in df_row:
            text = df_row[df_key]
        else:
            text = ''
        return text

    def _get_label_from_df_row(self, df_row, tname):
        col_name = tname + '_class'
        if col_name not in df_row:
            lbl = self.dummy_label
        else:
            lbl = df_row[col_name]
        return lbl

    def _get_labels(self, df_row) -> OrderedDict:
        labels = OrderedDict()
        for tname in self.label_type:
            labels[tname] = self._get_label_from_df_row(df_row, tname)
        return labels

    @classmethod
    def _sample(cls, video_path: Path, fps: float, start: float, end: float,
                df_row: pd.DataFrame, frames_per_clip: int, frame_rate: float,
                sample_strategy: str, reader_fn: nn.Module,
                rng: np.random.Generator):
        """
        Need this since VideoClip/RandomSampler etc are not quite compatible
            with this dataset. So recreating that here. Gets the full clip and
            crops out a fixed size region.
        Args:
            video_path: The path to read the video from
            fps: What this video's natural FPS is.
            start, end: floats of the start and end point in seconds
        Returns:
            video between start', end'; info of the video
        """
        start = max(start, 0)  # No way can read negative time anyway
        end = max(end, 0)  # No way can read negative time anyway
        if fps <= 0:
            logging.error('Found %f FPS video => likely empty [%s].', fps,
                          video_path)
            fps = frame_rate  # So code works, will anyway return black frames
        req_fps = frame_rate
        if req_fps is None:
            req_fps = fps
        nframes = int(fps * (end - start))
        frames_to_ext = int(round(frames_per_clip * (fps / req_fps)))
        # Find a point in the video and crop out
        if sample_strategy == SAMPLE_STRAT_RAND:
            start_frame = max(nframes - frames_to_ext, 0)
            if start_frame > 0:
                start_frame = rng.integers(start_frame)
        elif sample_strategy == SAMPLE_STRAT_CNTR:
            start_frame = max((nframes - frames_to_ext) // 2, 0)
        elif sample_strategy == SAMPLE_STRAT_LAST:
            start_frame = max(nframes - frames_to_ext, 0)
        elif sample_strategy == SAMPLE_STRAT_FIRST:
            start_frame = 0
        else:
            raise NotImplementedError(f'Unknown {sample_strategy}')
        new_start = start + max(start_frame / fps, 0)
        new_end = start + max((start_frame + frames_to_ext) / fps, 0)
        # Do not bleed out.. since this function could be used for anticipation
        # as well
        new_end = max(min(end, new_end), 0)
        # Start from the beginning of the video in case anticipation made it
        # go even further back
        new_start = min(max(new_start, 0), new_end)
        args = [str(video_path), new_start, new_end, fps, df_row]
        kwargs = dict(pts_unit='sec')
        outputs = reader_fn(*args, **kwargs)
        video, _, info = outputs
        # Debug code to eval the fwd model works -- cat image
        # from PIL import Image
        # video = torch.from_numpy(
        #     np.tile(
        #         np.array(
        #             Image.open('/private/home/rgirdhar/cat.png'))[np.newaxis],
        #         [10, 1, 1, 1]))
        # Keep track of where each frame came from, for video segmentation,
        # and then process exactly how the video gets processed
        if new_start >= new_end:
            video_frame_sec = new_start * torch.ones((video.size(0), ))
        else:
            video_frame_sec = torch.linspace(new_start, new_end, video.size(0))
        assert video_frame_sec.size(0) == video.size(0)
        # Subsample the video to the req_fps
        if sample_strategy == SAMPLE_STRAT_LAST:
            # From the back
            frames_to_keep = range(
                len(video))[::-max(int(round(fps / req_fps)), 1)][::-1]
        else:
            # Otherwise this is fine
            frames_to_keep = range(len(video))[::max(int(round(fps /
                                                               req_fps)), 1)]
        # Convert video to the required fps
        video_without_fps_subsample = video
        video = video[frames_to_keep]
        video_frame_sec = video_frame_sec[frames_to_keep]
        sampled_frames = torch.LongTensor(frames_to_keep)
        info['video_fps'] = req_fps
        # Ideally could have done the following operations only on the
        # frames_to_keep and done the above slice after, but to avoid bugs
        # and ensuring reproducibility (since earlier it was done separately),
        # just doing on all separately

        # Pad the video with the last frame, or crop out the extra frames
        # so that it is consistent with the frames_per_clip
        vid_t = video.size(0)
        if video.ndim != 4 or (video.size(0) * video.size(1) * video.size(2) *
                               video.size(3)) == 0:
            # Empty clip if any of the dims are 0, corrupted file likely
            logging.warning('Generating empty clip...')
            video = torch.zeros((frames_per_clip, 100, 100, 3),
                                dtype=torch.uint8)
            video_frame_sec = -torch.ones((frames_per_clip, ))
            sampled_frames = torch.range(0, frames_per_clip, dtype=torch.int64)
        elif vid_t < frames_per_clip:
            # # Repeat the video
            # video_reqfps = torch.cat([video_reqfps] *
            #                          int(math.ceil(frames_per_clip / vid_t)),
            #                          dim=0)
            # Pad the last frame..
            if sample_strategy == SAMPLE_STRAT_LAST:
                # Repeat the first frame
                def padding_fn(T, npad):
                    return torch.cat([T[:1]] * npad + [T], dim=0)
            else:
                # Repeat the last frame
                def padding_fn(T, npad):
                    return torch.cat([T] + [T[-1:]] * npad, dim=0)

            npad = frames_per_clip - vid_t
            logging.debug('Too few frames read, padding with %d frames', npad)
            video = padding_fn(video, npad)
            video_frame_sec = padding_fn(video_frame_sec, npad)
            sampled_frames = padding_fn(sampled_frames, npad)
        if sample_strategy == SAMPLE_STRAT_LAST:
            video = video[-frames_per_clip:]
            video_frame_sec = video_frame_sec[-frames_per_clip:]
            sampled_frames = sampled_frames[-frames_per_clip:]
        else:
            video = video[:frames_per_clip]
            video_frame_sec = video_frame_sec[:frames_per_clip]
            sampled_frames = sampled_frames[:frames_per_clip]

        # TODO(rgirdhar): Resample the audio in the same way too..
        return (video, video_frame_sec, video_without_fps_subsample,
                sampled_frames, info)

    def _get_video(self, df_row):
        # While we only need the absolute path for certain reader_fns, worth
        # doing it for all since some might still need it to read fps etc.
        video_path = get_abs_path(self.root, df_row['video_path'])
        fps = self.reader_fn.get_frame_rate(video_path)
        video_dict = {}
        (video, video_frame_sec, video_without_fps_subsample,
         frames_subsampled,
         info) = self._sample(video_path, fps, df_row['start'], df_row['end'],
                              df_row, self.frames_per_clip, self.frame_rate,
                              self.sample_strategy, self.reader_fn, self.rng)
        if 'audio_fps' not in info:
            # somehow this is missing is some elts.. it causes issues with
            # batching... anyway not using it so this is fine
            info['audio_fps'] = 0
        # Assuming no temporal transformation is done here (except moving the
        # dimension around), so no need to change the video_frame_sec
        video = self._apply_vid_transform(video)
        video_dict['video'] = video
        if self.return_unsampled_video:
            video_without_fps_subsample = self._apply_vid_transform(
                video_without_fps_subsample)
            video_dict[
                'video_without_fps_subsample'] = video_without_fps_subsample
            video_dict['video_frames_subsampled'] = frames_subsampled
        # Using video.size(-3) since at test there is a #crops dimension too
        # in the front, so from back it will always work
        assert video_frame_sec.size(0) == video.size(-3), (
            'nothing should have changed temporally')
        video_dict['video_frame_sec'] = video_frame_sec
        video_dict['video_info'] = info
        if self.return_future_clips_too:
            assert 'orig_start' in df_row, 'Has to be anticipation data'
            nfutures = len([
                el for el in df_row.keys() if el.startswith(FUTURE_PREFIX)
            ]) // 2  # Since start and end for each
            for future_id in range(nfutures):
                video_future, _, _, _, _ = self._sample(
                    video_path, fps,
                    df_row[f'{FUTURE_PREFIX}_{future_id}_start'],
                    df_row[f'{FUTURE_PREFIX}_{future_id}_end'], df_row,
                    self.frames_per_clip, self.frame_rate,
                    self.sample_strategy_future, self.reader_fn, self.rng)
                video_future = self._apply_vid_transform(video_future)
                video_dict[f'{FUTURE_PREFIX}_{future_id}_video'] = video_future
        video_dict['start'] = df_row['start']
        video_dict['end'] = df_row['end']
        return video_dict

    def _get_subclips(self, video: torch.Tensor, num_frames: int, stride: int):
        """
        Args:
            video (C, T, *): The original read video
            num_frames: Number of frames in each clip
            stride: stride to use when getting clips
        Returns:
            video (num_subclips, C, num_frames, *)
        """
        total_time = video.size(1)
        subclips = []
        for i in range(0, total_time, stride):
            subclips.append(video[:, i:i + num_frames, ...])
        return torch.stack(subclips)

    def _get_vidseg_labels(self, df_row, video_frame_sec: torch.Tensor):
        """
        Args:
            video_frame_sec (#clips, T): The time point each frame in the video
                comes from.
        """
        this_video_df = self.df_before_subset[self.df_before_subset.video_path
                                              == df_row.video_path]
        assert video_frame_sec.ndim == 2
        labels = OrderedDict()
        for tname in self.label_type:
            labels[tname] = -torch.ones_like(video_frame_sec, dtype=torch.long)
        for clip_id in range(video_frame_sec.size(0)):
            for t in range(video_frame_sec[clip_id].size(0)):
                cur_t = video_frame_sec[clip_id][t].tolist()
                matching_rows = this_video_df[
                    (this_video_df.orig_start <= cur_t)
                    & (this_video_df.orig_end >= cur_t)]
                if len(matching_rows) == 0:
                    continue  # Nothing labeled at this point
                elif len(matching_rows) > 1:
                    # logging.warning(
                    #     'Found multiple labels for a given time. '
                    #     'Should not happen.. overlapping labels. '
                    #     '%f %s %s', t, df_row, matching_rows)
                    # Apparently ^ happens often in epic100, so lets take the
                    # label closest to the center
                    closest_row = np.argmin(
                        np.abs(cur_t - np.array((
                            (matching_rows.orig_end -
                             matching_rows.orig_start) / 2.0).tolist())))
                    matching_row = matching_rows.iloc[closest_row]
                else:
                    matching_row = matching_rows.iloc[0]
                for tname in self.label_type:
                    labels[tname][clip_id][t] = self._get_label_from_df_row(
                        matching_row, tname)
        return labels

    def _apply_vid_transform(self, video):
        # Only apply the transform to normal videos, not if features are
        # being read
        if video.nelement() == 0:  # Placeholder
            return video
        if self.transform:
            assert video.ndim == 4
            if video.size(1) > 1 and video.size(2) > 1:
                # Normal video with spatial dimension
                video = self.transform(video)
            else:
                # Make sure the video is in the right permutation as expected
                # Esp important when video is the RULSTM features
                # TxHxWxC -> CxTxHxW
                # No other transformation to be applied in this case
                video = video.permute(3, 0, 1, 2)
        return video

    def addl_df_proc_for_dense(self, df_row):
        """
        This function allows processing the DF row after it is passed through
        the `process_df_before_read_fn` function, so it's like 2 layers of
        processing. This is a function that a specific dataset can override.
        Used by HowTo100M to convert narrations to classes
        """
        return df_row

    def __getitem__(self, idx):
        idx = self._class_balance_data_idx(idx)  # Must be run before repeat
        idx = self._repeat_process_idx(idx)
        df_row = self.df.loc[idx, :]
        if self.conv_to_anticipate_fn_runtime is not None:
            df_row = hydra.utils.call(self.conv_to_anticipate_fn_runtime,
                                      df_row, self.df, self.root,
                                      self.addl_df_proc_for_dense)
        if df_row is None:
            return None
        if self.process_df_before_read_fn is not None:
            df_row = hydra.utils.call(self.process_df_before_read_fn, df_row,
                                      self.root, self.rng, self.label_type,
                                      self.frames_per_clip, self.frame_rate,
                                      self.sample_strategy, self.dummy_label)
        if df_row is None:
            return None
        video_dict = self._get_video(df_row)
        video = video_dict['video']
        orig_video_shape = video.shape
        if len(orig_video_shape) == 5:
            # #ncrops, C, T, H, W -- flatten first 2 dims for subclips
            video = video.flatten(0, 1)
        # #ncrops * C, T, H, W -> #clips, #ncrops * C, T', H, W
        video = self._get_subclips(video, **self.subclips_options)
        if len(orig_video_shape) == 5:
            # unflatten back
            video = video.reshape((video.size(0), ) + orig_video_shape[:2] +
                                  video.shape[-3:])
        video_dict['video'] = video
        video_dict['video_frame_sec'] = self._get_subclips(
            video_dict['video_frame_sec'].unsqueeze(0),
            # squeeze(1) because the 0th dim now will be the clips
            **self.subclips_options).squeeze(1)
        sentence = self._get_text(df_row)  # Not used at the moment
        label_idx = self._get_labels(df_row)
        video_dict.update({
            'idx':
            idx,
            'text':
            sentence,
            'target':
            label_idx,
            'audio': [],  # TODO?
            'orig_vid_len':
            df_row.video_len if 'video_len' in df_row else -1,
            'uid':
            df_row.uid,
        })
        if self.load_seg_labels:
            video_dict.update({
                'target_subclips':
                self._get_vidseg_labels(df_row, video_dict['video_frame_sec'])
            })
        if self.load_long_term_future_labels > 0:
            # This is only really used for visualization for now
            last_frame = video_dict['video_frame_sec'][-1].item()
            gap_in_frames = (video_dict['video_frame_sec'][-1].item() -
                             video_dict['video_frame_sec'][-2].item())
            video_dict.update({
                'future_subclips':
                self._get_vidseg_labels(
                    df_row,
                    torch.FloatTensor([
                        last_frame + gap_in_frames * i
                        for i in range(1, self.load_long_term_future_labels +
                                       1)
                    ]).reshape(-1, 1))
            })
        return video_dict

    def _repeat_process_idx(self, idx):
        """
        Depending on repeat_data_times, convert to the idx to actual idx.
        """
        total_len = len(self.df)
        scaled_idx = idx / self.repeat_data_times
        if self.repeat_data_times < 1:
            # Add some jitter, since it is being mapped to a bigger space
            scaled_idx += self.rng.integers(int(1 / self.repeat_data_times))
        scaled_idx = int(scaled_idx)
        scaled_idx %= total_len
        return scaled_idx

    def _class_balance_data_idx(self, idx):
        """
        If asked for balanced sampling based on labels, remap the idx to try to
        follow a uniform distribution over the dataset, based on classes.
        This must be run before repeating the df etc, since it assumes values
        based on self.df (not repeated versions) (can be done, but this is
        how it's currently implememented).
        """
        if not self.class_balanced_sampling:
            return idx
        classes_counts = OrderedDict(self.classes_counts)
        # if there is > 0 elements with -1, then keep it, else remove it
        if classes_counts[-1] == 0:
            del classes_counts[-1]
        # By equal distribution, the idx should land in this class
        # Counts sorted in the same way class IDs are sorted in the DF
        cls_counts = [classes_counts[i] for i in sorted(classes_counts.keys())]
        cls_cumsum = np.cumsum(cls_counts).tolist()
        cls_firstelt = [0] + cls_cumsum[:-1]
        share_per_class = max(cls_counts)
        # effective idx, given that we would have replicated each class to have
        # same number of elements
        new_total_len = len(cls_counts) * share_per_class
        old_total_len = sum(cls_counts)
        # inflation_per_idx = (new_total_len - old_total_len) // len(old_total_len)
        # Any random position in the scaled up indexing space
        # eff_idx = (int(idx * (new_total_len / old_total_len)) +
        #            self.rng.integers(inflation_per_idx))
        eff_idx = int(round(idx * ((new_total_len - 1) / (old_total_len - 1))))
        assert eff_idx <= new_total_len
        cls_idx = eff_idx // share_per_class
        # Ideally do something like this, to get the nearest data point
        # balanced by the classes
        # offset_cls_idx_ratio = (eff_idx % share_per_class) / share_per_class
        # new_idx = cls_firstelt[cls_idx] + int(
        #     round(offset_cls_idx_ratio * cls_counts[cls_idx]))
        # For now just simply pick any random data point for that class
        # This is not ideal ofc.. in expectation it will pick each clip
        # but no guarantee.. ideally should use something that will map to the
        # actual provided idx and makes sure each clips gets seen.
        # This could have been done simply by randomly sampling a class and
        # then sampling a video from it.
        new_idx = self.rng.integers(cls_firstelt[cls_idx], cls_cumsum[cls_idx])

        # Make sure it doesn't go over
        new_idx = new_idx % len(self.df)
        return new_idx

    def __len__(self):
        return int(len(self.df) * self.repeat_data_times)
