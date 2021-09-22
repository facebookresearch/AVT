# Copyright (c) Facebook, Inc. and its affiliates.

"""Main training entry."""

import os
import logging
import random
import subprocess

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

import func


OmegaConf.register_new_resolver('minus', lambda x, y: x - y)
# Multiply and cast to integer
OmegaConf.register_new_resolver('times_int', lambda x, y: int(x * y))


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    # Since future runs might corrupt the stored hydra config, copy it over
    # for backup.
    if not os.path.exists('.hydra.orig'):
        subprocess.call('cp -r .hydra .hydra.orig', shell=True)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    try:
        print(subprocess.check_output('nvidia-smi'))
    except subprocess.CalledProcessError:
        print('Could not run nvidia-smi..')
    # cudnn.deterministic = True  # Makes it slow..
    getattr(func, cfg.train.fn).main(cfg)


if __name__ == "__main__":
    logging.basicConfig(format=('%(asctime)s %(levelname)-8s'
                                ' {%(module)s:%(lineno)d} %(message)s'),
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    torch.multiprocessing.set_start_method('spawn')
    main()  # pylint: disable=no-value-for-parameter  # Uses hydra
