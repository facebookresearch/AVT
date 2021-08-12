from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import json

DEFAULTS = {
  # dataset loader, specify the dataset here
  "dataset": {
    # name of the dataset
    "name": "ucf101",
    # where all the videos are stored
    "root_folder": "./data/",
    # annotation file in JSON format
    "ant_file": {"train": None, "val": None},
    # different split
    "split": ["train", "val"],
    # num of classes
    "num_classes": 101,
    # frame sampling pattern
    # (M frames with interval of N and random offset of K)
    "sample_pattern": [8, 8, 0],
    # drop the last few frames of a video (to prevent decoder issues)
    "drop_last_frames": 0,
    # action type (for epic kitchens dataset)
    "action_type": None,
  },
  # input pipeline, specify data augmentations here
  "input": {
    # param for training
    "rotation": 15,  # if <=0 do not rotate
    "flip": True,  # random flip during training
    "blur": False,
    "color_jitter": 0.1, # color pertubation (if <=0 disabled)
    "padding": 0,    # padding before crop
    # priority: crop_resize > crop_scale > random_crop
    "crop_resize": False, # resize the regions before crop
    "crop_resize_area_range": [0.16, 1.0],   # default imagenet
    "crop_resize_ratio_range": [0.75, 1.33], # default imagenet
    "crop_scale": True, # mutli-scale crop
    "scale_train": 256,  # If -1 do not scale
    "crop_train": 224,  # crop size (must > 0)
    # param for val
    "scale_val": 256,  # If -1 do not scale
    "crop_val": 256,   # if -1 do not crop
    # normalization params (from ImageNet)
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    # num of works for data loader
    "num_workers": 6,
    # use mixup for training?
    "mixup": False,
    # batch_size
    "batch_size": 16, # must be N x #GPUs
    # number of clips for val / test
    "val_clips": 1,
    "test_clips": 1,
    # for test augmentation
    "scale_test": 256,
    "crop_test": 256,
    "num_test_crops": 3,
  },
  # network architecture
  "network": {
    # multi gpu support
    "devices": [0],  # default: single gpu
    # backbone
    "backbone": "i3d-inception",
    "modality": "rgb",
    # use pre-trained model when specify a file
    "pretrained": None,
    # freeze the network from the specify stage (-1 for training all stages)
    "frozen_stages": -1,
    # if use gradient checkpointing
    "gradient_cp": False,
    # output dim from the backbone
    "feat_dim": 1024,
    # decoder params
    "decoder": "avgfc",
    "decoder_num_nodes": 4,
    "decoder_stride": (1, 2, 2),
    "decoder_fc_std": 0.01, # how to init the fc layer
    "dropout_prob": 0.5,
    # use label smoothing for loss
    "label_smoothing": 0.1, # -1 to disable
    # class balanced loss
    "balanced_beta": -1,
    "cls_weights": None,
  },
  # optimizer (for training)
  "optimizer": {
    # solver
    "type": "SGD",      # SGD or ADAM
    # solver params
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "learning_rate": 0.0025,
    "decoder_lr_ratio": 10.0,
    "nesterov": True,
    "epochs": 90,
    # lr scheduler: cosine / multistep
    "schedule": {
      "type": "cosine",
      "steps": [], # in #epochs
      "gamma": 0.1
    },
  }
}

def _merge(src, dst):
  for k, v in src.items():
    if k in dst:
      if isinstance(v, dict):
        _merge(src[k], dst[k])
    else:
      dst[k] = v

def load_default_config():
  config = DEFAULTS
  # copy config params
  config["dataset"]["val_clips"] = config["input"]["val_clips"]
  config["network"]["num_classes"] = config["dataset"]["num_classes"]
  return config

def load_config(config_file, defaults=DEFAULTS):
  with open(config_file, "r") as fd:
    config = json.load(fd)
  _merge(defaults, config)
  # copy config params
  config["dataset"]["val_clips"] = config["input"]["val_clips"]
  config["network"]["num_classes"] = config["dataset"]["num_classes"]
  if "inception" in config["network"]["backbone"]:
    config["network"]["decoder_stride"] = (2, 2, 2)
  elif "152" in config["network"]["backbone"]:
    config["network"]["decoder_stride"] = (2, 2, 2)
  return config
