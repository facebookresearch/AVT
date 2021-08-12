"""
Taken from pytorch image classfication code
This part will probably need refactoring
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# python imports
import argparse
import os
import time
import math
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.multiprocessing as mp

# our code
from libs.datasets import create_video_dataset_joint, create_video_transforms_joint
from libs.core import load_config
from libs.models import EncoderDecoder as ModelBuilder
from libs.utils import (AverageMeter, save_checkpoint, accuracy,
                        create_optim, create_scheduler, ClipPrefetcherJoint,
                        fast_clip_collate_joint, reduce_tensor, sync_processes,
                        get_cls_weights, mean_class_accuracy, confusion_matrix)

# the arg parser
parser = argparse.ArgumentParser(
  description='Evaluate 3D ConvNet model on given dataset')
parser.add_argument('config', metavar='DIR',
                    help='path to a config file')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    help='print frequency (default: 10 iterations)')
parser.add_argument('--slice', action='store_true',
                    help='Slice clips from a single video (used for large models)')

# main function for testing
def main(args):
  # parse args
  if os.path.exists(args.config):
    config = load_config(args.config)
  else:
    raise ValueError("Config file does not exist.")
  print("Current configurations:")
  pprint(config)

  # use spawn for mp, this will fix a deadlock by OpenCV
  if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')

  # set up transforms and dataset
  _, _, test_transforms = create_video_transforms_joint(config['input'])
  _, test_dataset = create_video_dataset_joint(config['dataset'], None, test_transforms)
  print("Testing time data augmentations:")
  pprint(test_transforms)

  # skip weight loading if resume from a checkpoint
  if args.resume:
    config['network']['pretrained'] = None
  else:
    print("No model specified. Existing ... ")
    return

  # create model, optimizer and loss function (on GPUs)
  model = ModelBuilder(config['network'])

  # freeze the model
  model.eval()
  for param in model.parameters():
    param.requires_grad = False

  # model -> gpu
  model = model.cuda()
  model = nn.DataParallel(model, device_ids=config['network']['devices'])

  # must resume from a previous checkpoint
  if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    best_acc1 = checkpoint['best_acc1']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("=> loaded checkpoint '{}' (epoch {}, acc1 {})"
        .format(args.resume, checkpoint['epoch'], best_acc1))
  else:
    print("=> no checkpoint found at '{}'".format(args.resume))
    return

  # only instantiate the dataset if necessary
  test_dataset = test_dataset()
  test_dataset.load()
  # quick hack: reset the number of lips
  test_dataset.reset_num_clips(config['input']['test_clips'])

  # test batch_size = 1
  test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=len(config['network']['devices']),
    num_workers=config['input']['num_workers'],
    collate_fn=fast_clip_collate_joint,
    shuffle=False, pin_memory=True, sampler=None, drop_last=False)

  # evaluation
  model_arch = "{:s}-{:s}".format(
    config['network']['backbone'], config['network']['decoder'])
  print("Testing model {:s}...".format(model_arch))

  # testing: make sure cudnn runs in deterministic mode
  cudnn.enabled = True
  cudnn.benchmark = False
  cudnn.deterministic = True

  # evaluate the model
  validate(test_loader, model, args, config)

  # exit
  print("All done!")


def validate(test_loader, model, args, config):
  """Test the model on the validation set
  We follow "fully convolutional" testing:
    * Scale the video with shortest side =256
    * Uniformly sample 10 clips within a video
    * For each clip, crop K=3 regions of 256*256 along the longest side
    * This is equivalent to 30-crop testing
  """
  # set up meters
  batch_time = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()
  cm_meter = AverageMeter()
  model.eval()
  # data prefetcher with noramlization
  test_loader = ClipPrefetcherJoint(test_loader,
                               config['input']['mean'],
                               config['input']['std'])

  # loop over validation set
  end = time.time()
  input, target = test_loader.next()
  i = 0

  # for large models
  if args.slice:
    batch_size = input.size(1)
    max_split_size = 1
    for split_size in range(2, batch_size):
      if (batch_size % split_size) == 0 and split_size > max_split_size:
        max_split_size = split_size
    num_batch_splits = batch_size // max_split_size
    print("Split the input by size: {:d}x{:d}".format(
      max_split_size, num_batch_splits))

  while input is not None:
    i += 1
    # disable/enable gradients
    with torch.no_grad():
      if args.slice:
        # slice the inputs for testing
        splited_inputs = torch.split(input, max_split_size, dim=1)
        splited_outputs = []
        for idx in range(num_batch_splits):
          split_output = model(splited_inputs[idx])
          # test time augmentation (minor performance boost)
          flipped_split_input = torch.flip(splited_inputs[idx], (-1,))
          flipped_split_output = model(flipped_split_input)
          split_output = 0.5 * (split_output + flipped_split_output)
          splited_outputs.append(split_output)
        output = torch.mean(torch.stack(splited_outputs), dim=0)
      else:
        # forward all inputs
        output,_,_ = model(input)
        # print(output.size())
        # test time augmentation (minor performance boost)
        # always flip the last dim (width)
        flipped_input = torch.flip(input, (-1,))
        flipped_output,_,_ = model(flipped_input)
        output = 0.5 * (output + flipped_output)
    # print(target[1].size())
    # print(target[2].size())
    # measure accuracy and record loss
    acc1, acc5 = accuracy(output.data, target[0], topk=(1, 5))
    top1.update(acc1.item(), input.size(0))
    top5.update(acc5.item(), input.size(0))
    batch_cm = confusion_matrix(output.data, target[0])
    cm_meter.update(batch_cm.data.cpu().double())

    # prefetch next batch
    input, target = test_loader.next()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    # printing
    if i % (args.print_freq * 2) == 0:
      print('Test: [{0}/{1}]\t'
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'
        'Acc@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
         i, len(test_loader), batch_time=batch_time,
         top1=top1, top5=top5))

  cls_acc = mean_class_accuracy(cm_meter.sum)
  print('***Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Mean Cls Acc {cls_acc:.3f}'
        .format(top1=top1, top5=top5, cls_acc=100*cls_acc))

  return top1.avg, top5.avg


################################################################################
if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
