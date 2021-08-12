"""
Compute flops / params of a deep network
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import os
sys.path.insert(0, os.getcwd())

# python imports
import argparse
import time
import math
from pprint import pprint

# torch imports
import torch
import torch.nn as nn

# thop
from thop import profile

# our code
from libs.core import load_config
from libs.models import EncoderDecoder as ModelBuilder

# the arg parser
parser = argparse.ArgumentParser(
  description='Profile 3D ConvNet model')
parser.add_argument('config', metavar='DIR',
                    help='path to a config file')

# main function for testing
def main(args):
  # parse args
  if os.path.exists(args.config):
    config = load_config(args.config)
  else:
    raise ValueError("Config file does not exist.")
  print("Current configurations:")
  pprint(config)

  # create model on CPU
  config["network"]["pretrained"] = None
  model = ModelBuilder(config['network'])

  # evaluation
  model_arch = "{:s}-{:s}".format(
    config['network']['backbone'], config['network']['decoder'])
  print("Profiling model {:s}...".format(model_arch))

  # create dummy input
  num_frames = config["dataset"]["sample_pattern"][0]
  if isinstance(config["input"]["crop_train"], int):
    frame_width = config["input"]["crop_train"]
    frame_height = config["input"]["crop_train"]
  else:
    frame_width, frame_height = config["input"]["crop_train"]
  dummy_input = torch.randn(1, 3, num_frames, frame_height, frame_width)

  # profile
  flops, params = profile(model, inputs=(dummy_input, ))
  print("Model {:s} [Params (M): {:0.2f}, FlOPs (G): {:0.2f}]".format(
    model_arch, params / 1e6, flops / 1e9))
  return


################################################################################
if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
