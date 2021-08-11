"""The HowTo100M dataset loader."""

import os
import logging
from pathlib import Path
import json

import pandas as pd
from hydra.types import TargetConf

from common.utils import get_video_info
from .base_video_dataset import BaseVideoDataset

# Obtained from //s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
# The link above obtained from
# https://huggingface.co/transformers/_modules/transformers/tokenization_gpt2.html
GPT2_VOCAB_FPATH = '/checkpoint/rgirdhar/Work/FB/2020/001_VideoSSL/VidCls/Data/HowTo100M/gpt2-vocab.json'


def load_word_vocab():
    # using the full GPT2 vocabulary as the classes for now
    with open(GPT2_VOCAB_FPATH, encoding='utf-8') as vocab_handle:
        classes_dict = json.load(vocab_handle)
    return classes_dict


class HowTo100M(BaseVideoDataset):
    def __init__(self,
                 videos_csv_fpath: Path,
                 captions_fpath: Path,
                 tasks_csv_fpath: Path,
                 keep_ratio: float = 1.0,
                 keep_cat1: tuple = (),
                 keep_cat2: tuple = (),
                 max_label_length: int = 10,
                 **kwargs):
        # Max label length relevant to multi-label output
        self.max_label_length = max_label_length
        df, action_classes = self._init_df(captions_fpath, videos_csv_fpath,
                                           tasks_csv_fpath, keep_ratio,
                                           keep_cat1, keep_cat2)
        kwargs['action_classes'] = action_classes
        kwargs['dummy_label'] = [-1] * self.max_label_length
        kwargs['label_type'] = 'action'
        super().__init__(df, **kwargs)

    def _init_df(self, captions_fpath, videos_csv_fpath, tasks_csv_fpath,
                 keep_ratio, keep_cat1, keep_cat2):
        """
        Args:
            keep_cat1: List of category 1s to keep. By default, keep all.
            keep_cat2: Similar, for category 2.
        """
        # This videos list, tasks were not used, so leaving it out
        logging.info('Reading and joining videos and action tasks CSV...')
        videos = pd.read_csv(videos_csv_fpath)
        action_tasks = pd.read_csv(tasks_csv_fpath,
                                   sep='\t',
                                   names=['task_id', 't_names'])
        videos = videos.merge(action_tasks, on='task_id')
        if keep_cat1:
            videos = videos[videos.category_1.isin(keep_cat1)]
            logging.info('Keeping category 1 from %s. Left %d videos.',
                         keep_cat1, len(videos))
        if keep_cat2:
            videos = videos[videos.category_2.isin(keep_cat2)]
            logging.info('Keeping category 2 from %s. Left %d videos.',
                         keep_cat2, len(videos))
        logging.info('done.')
        if keep_ratio >= 0 and keep_ratio < 1.0:
            # Subselect videos to keep, randomly
            # TODO: Make it use the random_seed from parent class somehow
            videos = videos.sample(frac=keep_ratio, random_state=42)
        # # Delete some un-needed cols
        # del (videos['category_1'], videos['category_2'], videos['rank'],
        #      videos['task_id'])

        logging.info('Total videos %d', len(videos))

        logging.info('Loading captions from %s', captions_fpath)
        captions_df = pd.read_feather(captions_fpath)
        captions_df = captions_df[captions_df.video_path.apply(
            lambda x: os.path.splitext(x)[0]).isin(videos.video_id)]
        captions_df.reset_index(inplace=True)
        logging.info('Done')

        logging.info('Loading vocab as classes')
        action_classes = load_word_vocab()
        logging.info('Done')

        # # Not merging since no videos being computed separately
        # logging.info('Joining all captions DF with videos...')
        # final_df = captions_df.merge(videos, on='video_id')
        # logging.info('Done.')
        final_df = captions_df
        return final_df, action_classes

    def addl_df_proc_for_dense(self, df_subset):
        """
        Process the df to add action class based on the vocab.
        """
        labels_col = []
        for _, df_row in df_subset.iterrows():
            multi_labels = []
            words = df_row['narration'].split(' ') if isinstance(
                df_row['narration'], str) else []
            for word in words:
                if word in self.action_classes:
                    multi_labels.append(self.action_classes[word])
            if len(multi_labels) >= self.max_label_length:
                multi_labels = multi_labels[:self.max_label_length]
            else:
                multi_labels += [-1] * (self.max_label_length -
                                        len(multi_labels))
            labels_col.append(multi_labels)
        # TODO: fix the slice warning
        # A value is trying to be set on a copy of a slice from a DataFrame.
        df_subset['action_class'] = labels_col
        return df_subset
