# Copyright (c) Facebook, Inc. and its affiliates.

"""The Epic Kitchens dataset loaders."""

from typing import List, Dict, Sequence, Tuple, Union
from datetime import datetime, date
from collections import OrderedDict
import pickle as pkl
import csv
import logging
from pathlib import Path
import lmdb
import pandas as pd

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn as nn

from .base_video_dataset import BaseVideoDataset, RULSTM_TSN_FPS
from .reader_fns import Reader

EGTEA_VERSION = -1  # This class also supports EGTEA Gaze+
EPIC55_VERSION = 0.1
EPIC100_VERSION = 0.2


class EPICKitchens(BaseVideoDataset):
    """EPICKitchens dataloader."""
    def __init__(
            self,
            annotation_path: Sequence[Path],
            only_keep_persons: str = None,
            only_keep_videos: Path = None,
            action_labels_fpath: Path = None,
            annotation_dir: Path = None,
            rulstm_annotation_dir: Path = None,
            _precomputed_metadata: Path = None,
            version: float = EPIC55_VERSION,
            **other_kwargs,
    ):
        """
        Args:
            label_type (str): The type of label to return
            only_keep_persons (str): If None, ignore. Else, will only keep
                videos of persons P<start> to P<end> (both included), where this
                string is "<start>-<end>". This is used to create
                the train_minus_val and val sets, as per
                https://arxiv.org/abs/1806.06157
            only_keep_videos (Path): Path to a file with list of videos to keep.
                This was used to define the val set as used in anticipation
                in https://arxiv.org/abs/1905.09035
            action_labels_fpath (Path): Path to map the verb and noun labels to
                actions. It was used in the anticipation paper, that defines
                a set of actions and train for action prediction, as opposed
                to verb and noun prediction.
            annotation_dir: Where all the other annotations are typically stored
        """
        self.version = version
        df = pd.concat([self._load_df(el) for el in annotation_path])
        df.reset_index(inplace=True, drop=True)  # to combine all of them
        df = self._subselect_df_by_videos(
            self._subselect_df_by_person(df, only_keep_persons),
            only_keep_videos)
        # If no specific annotation_dir specified, use the parent dir of
        # the first annot path
        if annotation_dir is None:
            self.annotation_dir = Path(annotation_path[0]).parent
        else:
            self.annotation_dir = Path(annotation_dir)
        self.rulstm_annotation_dir = rulstm_annotation_dir
        epic_postfix = ''
        if self.version == EPIC100_VERSION:
            epic_postfix = '_100'
        if self.version != EGTEA_VERSION:
            verb_classes = self._load_class_names(
                self.annotation_dir / f'EPIC{epic_postfix}_verb_classes.csv')
            noun_classes = self._load_class_names(
                self.annotation_dir / f'EPIC{epic_postfix}_noun_classes.csv')
        else:
            verb_classes, noun_classes = [], []
        # Create action classes
        if action_labels_fpath is not None:
            load_action_fn = self._load_action_classes
            if self.version == EGTEA_VERSION:
                load_action_fn = self._load_action_classes_egtea
            action_classes, verb_noun_to_action = (
                load_action_fn(action_labels_fpath))
        else:
            action_classes, verb_noun_to_action = self._gen_all_actions(
                verb_classes, noun_classes)
        # Add the action classes to the data frame
        if ('action_class' not in df.columns
                and {'noun_class', 'verb_class'}.issubset(df.columns)):
            df.loc[:, 'action_class'] = df.loc[:, (
                'verb_class', 'noun_class')].apply(
                    lambda row: (verb_noun_to_action[
                        (row.at['verb_class'], row.at['noun_class'])]
                                 if (row.at['verb_class'], row.at['noun_class']
                                     ) in verb_noun_to_action else -1),
                    axis=1)
        elif 'action_class' not in df.columns:
            df.loc[:, 'action_class'] = -1
            df.loc[:, 'verb_class'] = -1
            df.loc[:, 'noun_class'] = -1
        num_undefined_actions = len(df[df['action_class'] == -1].index)
        if num_undefined_actions > 0:
            logging.error(
                'Did not found valid action label for %d/%d samples!',
                num_undefined_actions, len(df))
        assert _precomputed_metadata is None, 'Not supported yet'
        other_kwargs['verb_classes'] = verb_classes
        other_kwargs['noun_classes'] = noun_classes
        other_kwargs['action_classes'] = action_classes
        super().__init__(df, **other_kwargs)
        # following is used in the notebooks for marginalization, so save it
        self.verb_noun_to_action = verb_noun_to_action
        logging.info('Created EPIC %s dataset with %d samples', self.version,
                     len(self))

    @property
    def primary_metric(self) -> str:
        if self.version == EPIC100_VERSION:
            # For EK100, we want to optimize for AR5
            return 'final_acc/action/AR5'
        return super().primary_metric

    @property
    def class_mappings(self) -> Dict[Tuple[str, str], torch.FloatTensor]:
        num_verbs = len(self.verb_classes)
        if num_verbs == 0:
            num_verbs = len(
                set([el[0] for el, _ in self.verb_noun_to_action.items()]))
        num_nouns = len(self.noun_classes)
        if num_nouns == 0:
            num_nouns = len(
                set([el[1] for el, _ in self.verb_noun_to_action.items()]))
        num_actions = len(self.action_classes)
        if num_actions == 0:
            num_actions = len(
                set([el for _, el in self.verb_noun_to_action.items()]))
        verb_in_action = torch.zeros((num_actions, num_verbs),
                                     dtype=torch.float)
        noun_in_action = torch.zeros((num_actions, num_nouns),
                                     dtype=torch.float)
        for (verb, noun), action in self.verb_noun_to_action.items():
            verb_in_action[action, verb] = 1.0
            noun_in_action[action, noun] = 1.0
        return {
            ('verb', 'action'): verb_in_action,
            ('noun', 'action'): noun_in_action
        }

    @property
    def classes_manyshot(self) -> OrderedDict:
        """
        In EPIC-55, the recall computation was done for "many shot" classes,
        and not for all classes. So, for that version read the class names as
        provided by RULSTM.
        Function adapted from
        https://github.com/fpv-iplab/rulstm/blob/57842b27d6264318be2cb0beb9e2f8c2819ad9bc/RULSTM/main.py#L386
        """
        if self.version != EPIC55_VERSION:
            return super().classes_manyshot
        # read the list of many shot verbs
        many_shot_verbs = {
            el['verb']: el['verb_class']
            for el in pd.read_csv(self.annotation_dir /
                                  'EPIC_many_shot_verbs.csv').to_dict(
                                      'records')
        }
        # read the list of many shot nouns
        many_shot_nouns = {
            el['noun']: el['noun_class']
            for el in pd.read_csv(self.annotation_dir /
                                  'EPIC_many_shot_nouns.csv').to_dict(
                                      'records')
        }
        # create the list of many shot actions
        # an action is "many shot" if at least one
        # between the related verb and noun are many shot
        many_shot_actions = {}
        action_names = {val: key for key, val in self.action_classes.items()}
        for (verb_id, noun_id), action_id in self.verb_noun_to_action.items():
            if (verb_id in many_shot_verbs.values()) or (
                    noun_id in many_shot_nouns.values()):
                many_shot_actions[action_names[action_id]] = action_id
        return {
            'verb': many_shot_verbs,
            'noun': many_shot_nouns,
            'action': many_shot_actions,
        }

    @staticmethod
    def _load_action_classes(
            action_labels_fpath: Path
    ) -> Tuple[Dict[str, int], Dict[Tuple[int, int], int]]:
        """
        Given a CSV file with the actions (as from RULSTM paper), construct
        the set of actions and mapping from verb/noun to action
        Args:
            action_labels_fpath: path to the file
        Returns:
            class_names: Dict of action class names
            verb_noun_to_action: Mapping from verb/noun to action IDs
        """
        class_names = {}
        verb_noun_to_action = {}
        with open(action_labels_fpath, 'r') as fin:
            reader = csv.DictReader(fin, delimiter=',')
            for lno, line in enumerate(reader):
                class_names[line['action']] = lno
                verb_noun_to_action[(int(line['verb']),
                                     int(line['noun']))] = int(line['id'])
        return class_names, verb_noun_to_action

    @staticmethod
    def _load_action_classes_egtea(
            action_labels_fpath: Path
    ) -> Tuple[Dict[str, int], Dict[Tuple[int, int], int]]:
        """
        Given a CSV file with the actions (as from RULSTM paper), construct
        the set of actions and mapping from verb/noun to action
        Args:
            action_labels_fpath: path to the file
        Returns:
            class_names: Dict of action class names
            verb_noun_to_action: Mapping from verb/noun to action IDs
        """
        class_names = {}
        verb_noun_to_action = {}
        with open(action_labels_fpath, 'r') as fin:
            reader = csv.DictReader(
                fin,
                delimiter=',',
                # Assuming the order is verb/noun
                # TODO check if that is correct
                fieldnames=['id', 'verb_noun', 'action'])
            for lno, line in enumerate(reader):
                class_names[line['action']] = lno
                verb, noun = [int(el) for el in line['verb_noun'].split('_')]
                verb_noun_to_action[(verb, noun)] = int(line['id'])
        return class_names, verb_noun_to_action

    @staticmethod
    def _gen_all_actions(
            verb_classes: List[str], noun_classes: List[str]
    ) -> Tuple[Dict[str, int], Dict[Tuple[int, int], int]]:
        """
        Given all possible verbs and nouns, construct all possible actions
        Args:
            verb_classes: All verbs
            noun_classes: All nouns
        Returns:
            class_names: list of action class names
            verb_noun_to_action: Mapping from verb/noun to action IDs
        """
        class_names = {}
        verb_noun_to_action = {}
        action_id = 0
        for verb_id, verb_cls in enumerate(verb_classes):
            for noun_id, noun_cls in enumerate(noun_classes):
                class_names[f'{verb_cls}:{noun_cls}'] = action_id
                verb_noun_to_action[(verb_id, noun_id)] = action_id
                action_id += 1
        return class_names, verb_noun_to_action

    def _load_class_names(self, annot_path: Path):
        res = {}
        with open(annot_path, 'r') as fin:
            reader = csv.DictReader(fin, delimiter=',')
            for lno, line in enumerate(reader):
                res[line['class_key' if self.version ==
                         EPIC55_VERSION else 'key']] = lno
        return res

    def _load_df(self, annotation_path):
        if annotation_path.endswith('.pkl'):
            return self._init_df_orig(annotation_path)
        elif annotation_path.endswith('.csv'):
            # Else, it must be the RULSTM annotations (which are a
            # little different, perhaps due to quantization into frames)
            return self._init_df_rulstm(annotation_path)
        else:
            raise NotImplementedError(annotation_path)

    def _init_df_gen_vidpath(self, df):
        # generate video_path
        if self.version == EGTEA_VERSION:
            df.loc[:, 'video_path'] = df.apply(
                lambda x: Path(x.video_id + '.mp4'),
                axis=1,
            )
        else:  # For the EPIC datasets
            df.loc[:, 'video_path'] = df.apply(
                lambda x: (Path(x.participant_id) / Path(x.video_id + '.MP4')),
                axis=1,
            )
        return df

    def _init_df_rulstm(self, annotation_path):
        logging.info('Loading RULSTM EPIC csv annotations %s', annotation_path)
        df = pd.read_csv(
            annotation_path,
            names=[
                'uid',
                'video_id',
                'start_frame_30fps',
                'end_frame_30fps',
                'verb_class',
                'noun_class',
                'action_class',
            ],
            index_col=0,
            skipinitialspace=True,
            dtype={
                'uid': str,  # In epic-100, this is a str
                'video_id': str,
                'start_frame_30fps': int,
                'end_frame_30fps': int,
                'verb_class': int,
                'noun_class': int,
                'action_class': int,
            })
        # Make a copy of the UID column, since that will be needed to gen
        # output files
        df.reset_index(drop=False, inplace=True)
        # Convert the frame number to start and end
        df.loc[:, 'start'] = df.loc[:, 'start_frame_30fps'].apply(
            lambda x: x / RULSTM_TSN_FPS)
        df.loc[:, 'end'] = df.loc[:, 'end_frame_30fps'].apply(
            lambda x: x / RULSTM_TSN_FPS)
        # Participant ID from video_id
        df.loc[:, 'participant_id'] = df.loc[:, 'video_id'].apply(
            lambda x: x.split('_')[0])
        df = self._init_df_gen_vidpath(df)
        df.reset_index(inplace=True, drop=True)
        return df

    def _init_df_orig(self, annotation_path):
        """
        Loading the original EPIC Kitchens annotations
        """
        def timestr_to_sec(s, fmt='%H:%M:%S.%f'):
            timeobj = datetime.strptime(s, fmt).time()
            td = datetime.combine(date.min, timeobj) - datetime.min
            return td.total_seconds()

        # Load the DF from annot path
        logging.info('Loading original EPIC pkl annotations %s',
                     annotation_path)
        with open(annotation_path, 'rb') as fin:
            df = pkl.load(fin)
        # Make a copy of the UID column, since that will be needed to gen
        # output files
        df.reset_index(drop=False, inplace=True)

        # parse timestamps from the video
        df.loc[:, 'start'] = df.start_timestamp.apply(timestr_to_sec)
        df.loc[:, 'end'] = df.stop_timestamp.apply(timestr_to_sec)

        # original annotations have text in weird format - fix that
        if 'noun' in df.columns:
            df.loc[:, 'noun'] = df.loc[:, 'noun'].apply(
                lambda s: ' '.join(s.replace(':', ' ').split(sep=' ')[::-1]))
        if 'verb' in df.columns:
            df.loc[:, 'verb'] = df.loc[:, 'verb'].apply(
                lambda s: ' '.join(s.replace('-', ' ').split(sep=' ')))
        df = self._init_df_gen_vidpath(df)
        df.reset_index(inplace=True, drop=True)
        return df

    @staticmethod
    def _subselect_df_by_person(df, only_keep_persons):
        if only_keep_persons is None:
            return df
        start, end = [int(el) for el in only_keep_persons.split('-')]
        df = df.loc[df['participant_id'].isin(
            ['P{:02d}'.format(el) for el in range(start, end + 1)]), :]
        df.reset_index(inplace=True, drop=True)
        return df

    @staticmethod
    def _subselect_df_by_videos(df, videos_fpath):
        if videos_fpath is None:
            return df
        with open(videos_fpath, 'r') as fin:
            videos_to_keep = [el.strip() for el in fin.read().splitlines()]
        df = df.loc[df['video_id'].isin(videos_to_keep), :]
        df.reset_index(inplace=True, drop=True)
        return df


class EpicRULSTMFeatsReader(Reader):
    def __init__(self,
                 lmdb_path: Union[Path, List[Path]] = None,
                 read_type: str = 'exact_rulstm',
                 warn_if_using_closeby_frame: bool = True):
        """
        Args:
            feats_lmdb_path: LMDB path for RULSTM features. Must be
                specified if using rulstm_tsn_feat input_type. Could be a
                list, in which case it will concat all those features together.
            read_type: [rulstm_exact/normal] This specifies what style of
                feature reading for RULSTM features. Until Oct 22, I have been
                exactly reading 11 frames at 0.25s, but that is not scalable to
                learn language models, so making it more generic to read all
                frames and let the base_video_dataset code figure how to
                re-sample to get the relevant frames. Not making it default
                to be able to repro older results.
        """
        super().__init__()
        if OmegaConf.get_type(lmdb_path) != list:
            lmdb_path = [lmdb_path]
        self.lmdb_envs = [
            lmdb.open(el, readonly=True, lock=False) for el in lmdb_path
        ]
        self.read_type = read_type
        self.warn_if_using_closeby_frame = warn_if_using_closeby_frame

    def forward(self, *args, **kwargs):
        return self._read_rulstm_features(*args, **kwargs)

    @staticmethod
    def get_frame_rate(video_path: Path) -> float:
        del video_path
        return RULSTM_TSN_FPS

    def read_representations(self, frames, env, frame_format):
        """Reads a set of representations, given their frame names and an LMDB
            environment.
            From https://github.com/fpv-iplab/rulstm/blob/96e38666fad7feafebbeeae94952dba24771e512/RULSTM/dataset.py#L10
        """
        features = []
        # for each frame
        for frame_id in frames:
            # read the current frame
            with env.begin() as e:
                # Need to search for a frame that has features stored,
                # the exact frame may not have.
                # To avoid looking at the future when training/testing,
                # (important for anticipation), look only for previous to
                # current position.
                dd = None
                search_radius = 0
                for search_radius in range(10):
                    dd = e.get(
                        frame_format.format(
                            frame_id - search_radius).strip().encode('utf-8'))
                    if dd is not None:
                        break
                if dd is not None and search_radius > 0:
                    if self.warn_if_using_closeby_frame:
                        logging.warning('Missing %s, but used %d instead',
                                        frame_format.format(frame_id),
                                        frame_id - search_radius)
            if dd is None:
                logging.error(
                    'Missing %s, Only specific frames are stored in lmdb :(',
                    frame_format.format(frame_id))
                features.append(None)
            else:
                # convert to numpy array
                data = np.frombuffer(dd, 'float32')
                # append to list
                features.append(data)
        # For any frames we didn't find a feature, use a series of 0s
        features_not_none = [el for el in features if el is not None]
        assert len(features_not_none) > 0, (
            f'No features found in {frame_format} - {frames}')
        feature_not_none = features_not_none[0]  # any
        features = [
            np.zeros_like(feature_not_none) if el is None else el
            for el in features
        ]
        # convert list to numpy array
        features = np.array(features)
        # Add singleton dimensions to make it look like a video, so
        # rest of the code just works
        features = features[:, np.newaxis, np.newaxis, :]
        # Make it torch Tensor to be consistent
        features = torch.as_tensor(features)
        return features

    def _read_rulstm_features(self,
                              video_path: Path,
                              start_sec: float,
                              end_sec: float,
                              fps: float,
                              df_row: pd.DataFrame,
                              pts_unit='sec'):
        del pts_unit  # Not supported here
        if self.read_type == 'exact_rulstm':
            # get frames every 0.25s between start and end frames
            # 0.25 comes from their code, and they typically do 2.5s total
            # observation time. 14 is the sequence length they use.
            time_stamps = end_sec - np.arange(0.0, 0.25 * 11, 0.25)[::-1]
            frames = np.floor(time_stamps * fps).astype(int)
        elif self.read_type == 'normal':
            # Read every single frame between the start and end, the
            # base_video_dataset code will deal with how to sample into 4fps
            # (i.e. 0.25s steps)
            # Rather than first computing the timestamps, just compute the
            # frame ID of the start and end, and do a arange .. that avoids
            # any repeated frames due to quantization/floor
            time_stamps = None
            start_frame = np.floor(start_sec * fps)
            end_frame = np.floor(end_sec * fps)
            frames = np.arange(end_frame, start_frame, -1).astype(int)[::-1]
        else:
            raise NotImplementedError(f'Unknown {self.read_type}')
        # If the frames go below 1, replace them with the lowest time pt
        assert frames.max() >= 1, (
            f'The dataset shouldnt have cases otherwise. {video_path} '
            f'{start_sec} {end_sec} {df_row} {frames} {time_stamps} ')
        frames[frames < 1] = frames[frames >= 1].min()
        # Get the features
        all_feats = []
        for lmdb_env in self.lmdb_envs:
            all_feats.append(
                self.read_representations(
                    frames, lmdb_env,
                    Path(video_path).stem + '_frame_{:010d}.jpg'))
        final_feat = torch.cat(all_feats, dim=-1)
        # Must return rgb, audio, info; so padding with empty dicts for those
        return final_feat, {}, {}
