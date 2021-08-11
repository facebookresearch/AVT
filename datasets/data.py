import os
import torch
from importlib import import_module
from tqdm import tqdm

import omegaconf
import hydra

from common import utils

__all__ = [
    "get_dataset",
]


def get_dataset(dataset_cfg, data_cfg, transform, logger):
    # If there is _precomputed_metadata file passed in, load that in
    kwargs = {}
    precomp_metadata_fpath = None
    if '_precomputed_metadata_file' in dataset_cfg:
        precomp_metadata_fpath = dataset_cfg._precomputed_metadata_file
        # Remove from the config since otherwise can't init the obj
        with omegaconf.open_dict(dataset_cfg):
            del dataset_cfg['_precomputed_metadata_file']
        if os.path.exists(precomp_metadata_fpath):
            _precomputed_metadata = torch.load(precomp_metadata_fpath)
            kwargs['_precomputed_metadata'] = _precomputed_metadata

    kwargs['transform'] = transform
    kwargs['frame_rate'] = data_cfg.frame_rate
    kwargs['frames_per_clip'] = data_cfg.num_frames
    # Have to call dict() here since relative interpolation somehow doesn't
    # work once I get the subclips object
    kwargs['subclips_options'] = dict(data_cfg.subclips)
    kwargs['load_seg_labels'] = data_cfg.load_seg_labels
    logger.info('Creating the dataset object...')
    # Not recursive since many of the sub-instantiations would need positional
    # arguments
    _dataset = hydra.utils.instantiate(dataset_cfg,
                                       _recursive_=False,
                                       **kwargs)
    try:
        logger.info('Computing clips...')
        _dataset.video_clips.compute_clips(data_cfg.num_frames,
                                           1,
                                           frame_rate=data_cfg.frame_rate)
        logger.info('Done')
    except AttributeError:  # if video_clips not in _dataset
        logger.warning('No video_clips present')
    logger.info(f'Created dataset with {len(_dataset)} elts')

    if precomp_metadata_fpath and not os.path.exists(precomp_metadata_fpath):
        utils.save_on_master(_dataset.metadata, precomp_metadata_fpath)
    return _dataset
