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
from pprint import pprint

# control the number of threads (to prevent blocking ...)
# should work for both numpy / opencv
# it really depends on the CPU / GPU ratio ...
# set to a small number (e.g., 1) for distributed training
TARGET_NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = TARGET_NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = TARGET_NUM_THREADS
os.environ["MKL_NUM_THREADS"] = TARGET_NUM_THREADS

# numpy imports
import numpy as np
import random

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.multiprocessing as mp

# for visualization
from torch.utils.tensorboard import SummaryWriter
# from tensorboard import SummaryWriter
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
  description='Video Classification using 3D ConvNets')
parser.add_argument('config', metavar='DIR',
                    help='path to a config file')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    help='print frequency (default: 10 iterations)')
parser.add_argument('-v', '--valid-freq', default=5, type=int,
                    help='validation frequency (default: every 5 epochs)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--local_rank", default=0, type=int,
                    help="For distributed training. No manual specification!")
parser.add_argument('--sync_bn', action='store_true',
                    help='enabling sync BN (only for distributed training).')
parser.add_argument('--fix_res', action='store_true',
                    help='Fix the inconsistency of training/testing resolution')
parser.add_argument('--prec_bn', action='store_true',
                    help='Use precise BN every few epochs')


# main function for training and testing
def main(args):
  ##############################################################################
  """Setup parameters"""
  # parse args
  best_acc1 = 0.0
  args.start_epoch = 0
  if os.path.exists(args.config):
    config = load_config(args.config)
  else:
    raise ValueError("Config file does not exist.")
  if args.local_rank == 0:
    print("Current configurations:")
    pprint(config)

  # prep for output folder
  config_filename = os.path.basename(args.config).replace('.json', '')
  ckpt_folder = os.path.join('./ckpt', config_filename)
  if (args.local_rank == 0) and not os.path.exists(ckpt_folder):
    os.mkdir(ckpt_folder)
  # tensorboard writer
  global writer
  if args.local_rank == 0:
    writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

  # use spawn for mp, this will fix a deadlock by OpenCV
  if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')

  # for distributed training
  args.distributed = False
  if 'WORLD_SIZE' in os.environ:
    args.distributed = (int(os.environ['WORLD_SIZE']) > 1)
  args.world_size = 1
  if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    print("Distributed training (local rank {:d} / world size {:d})".format(
            args.local_rank, args.world_size))

  # fix the random seeds (the best we can)
  fixed_random_seed = 2019 + int(args.distributed) * args.local_rank
  torch.manual_seed(fixed_random_seed)
  np.random.seed(fixed_random_seed)
  random.seed(fixed_random_seed)

  # skip weight loading if resume from a checkpoint
  if args.resume:
    config['network']['pretrained'] = None

  # re-scale learning rate based on world size
  if args.distributed:
    config['optimizer']["learning_rate"] *= args.world_size
  else:
    # also need to re-scale the worker number if using data parallel
    config['optimizer']["learning_rate"] *= len(config['network']['devices'])
    config['input']['num_workers'] *= len(config['network']['devices'])

  ##############################################################################
  """Create datasets"""
  # set up transforms and dataset
  train_transforms, val_transforms, _ = \
    create_video_transforms_joint(config['input'])
  train_dataset, val_dataset = create_video_dataset_joint(
    config['dataset'], train_transforms, val_transforms)
  is_train, is_test = (train_dataset is not None), (val_dataset is not None)

  # print the data augs
  if args.local_rank == 0:
    print("Training time data augmentations:")
    pprint(train_transforms)
    print("Testing time data augmentations:")
    pprint(val_transforms)

  if is_train:
    # only instantiate the dataset if necessary
    train_dataset = train_dataset()
    train_dataset.load()

  if is_test:
    # only instantiate the dataset if necessary
    val_dataset = val_dataset()
    val_dataset.load()

  # reset loss params
  if config['network']['balanced_beta'] > 0:
    num_samples_per_cls = train_dataset.get_num_samples_per_cls()
    config['network']['cls_weights'] = get_cls_weights(
      num_samples_per_cls, config['network']['balanced_beta'])
    if args.local_rank == 0:
      print("Using class balanced loss with beta = {:0.4f}".format(
        config['network']['balanced_beta']))

  ##############################################################################
  """Create model (w. loss) & optimizer"""
  # create model -> GPU 0
  model = ModelBuilder(config['network'])
  if args.sync_bn and args.distributed:
    # discrepancy in docs (this now works as default for pytorch 1.2)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # this will force the model to re-freeze the params (bug fixed in 1.2)
    model.train()
  model = model.cuda()

  # create optimizer
  optimizer = create_optim(model, config['optimizer'])

  # create the model
  if args.distributed:
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[args.local_rank],
                                                output_device=args.local_rank,
                                                find_unused_parameters=True)
  else:
    model = nn.DataParallel(model, device_ids=config['network']['devices'])

  ##############################################################################
  """Create data loaders / scheduler"""
  if is_train:
    train_sampler = None
    if args.distributed:
      train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=config['input']['batch_size'],
      num_workers=config['input']['num_workers'],
      collate_fn=fast_clip_collate_joint,
      shuffle=(train_sampler is None), pin_memory=True,
      sampler=train_sampler, drop_last=True)

  if is_test:
    val_sampler = None
    val_batch_size = max(
      1, config['input']['batch_size'] // val_dataset.get_num_clips())
    if args.distributed:
      val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset)
    # validation here is not going to be accurate any way ...
    val_loader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=val_batch_size,
      num_workers=config['input']['num_workers'],
      collate_fn=fast_clip_collate_joint,
      shuffle=False, pin_memory=True, sampler=val_sampler, drop_last=True)

  # set up learning rate scheduler
  if is_train:
    num_iters_per_epoch = len(train_loader)
    scheduler = create_scheduler(optimizer,
                                 config['optimizer']['schedule'],
                                 config['optimizer']['epochs'],
                                 num_iters_per_epoch)

  ##############################################################################
  """Resume from model / Misc"""
  # resume from a checkpoint?
  if args.resume:
    if os.path.isfile(args.resume):
      if args.local_rank == 0:
        print("=> loading checkpoint '{}'".format(args.resume))
      checkpoint = torch.load(args.resume,
        map_location = lambda storage, loc: storage.cuda(args.local_rank))
      if not args.fix_res:
        args.start_epoch = checkpoint['epoch']
      best_acc1 = checkpoint['best_acc1']
      model.load_state_dict(checkpoint['state_dict'])
      # only load the optimizer if necessary
      if is_train and (not args.fix_res):
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
      if args.local_rank == 0:
        print("=> loaded checkpoint '{}' (epoch {}, acc1 {})"
          .format(args.resume, checkpoint['epoch'], best_acc1))
    else:
      print("=> no checkpoint found at '{}'".format(args.resume))
      return

  # training: enable cudnn benchmark
  cudnn.enabled = True
  cudnn.benchmark = True
  # model architecture
  model_arch = "{:s}-{:s}".format(
    config['network']['backbone'], config['network']['decoder'])

  ##############################################################################
  """Training / Validation"""
  # start the training
  if is_train:
    # start the training
    if args.local_rank == 0:
      # save the current config
      with open(os.path.join(ckpt_folder, 'config.text'), 'w') as fid:
        pprint(config, stream=fid)
      print("Training model {:s} ...".format(model_arch))
      pprint(model)

    for epoch in range(args.start_epoch, config['optimizer']['epochs']):
      if args.distributed:
        train_sampler.set_epoch(epoch)
      # acc1, acc5 = validate(val_loader, model, epoch, args, config)
      # train for one epoch
      train(train_loader, model, optimizer, scheduler, epoch, args, config)

      # evaluate on validation set once in a while
      # test on every epoch at the end of training
      # Note this will also run after first epoch (make sure training is on track)
      # print(epoch)
      if (epoch % args.valid_freq == 0) \
          or (epoch > 0.6 * config['optimizer']['epochs']):
        # use prec bn to aggregate stats before validation
        if args.prec_bn:
          prec_bn(train_loader, model, epoch, args, config)
        acc1, acc5 = validate(val_loader, model, epoch, args, config)

        if args.local_rank == 0:
          # remember best acc@1 and save checkpoint
          is_best = acc1 > best_acc1
          best_acc1 = max(acc1, best_acc1)
          save_checkpoint({
            'epoch': epoch + 1,
            'model_arch': model_arch,
            'state_dict': model.state_dict(),
            'best_acc1': acc1,
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
          }, is_best, file_folder=ckpt_folder)
        
      # sync all processes manually
      if args.distributed:
        sync_processes()
      else:
        torch.cuda.empty_cache()

  if args.local_rank == 0:
    writer.close()
    print("All done!")

def train(train_loader, model, optimizer, scheduler, epoch, args, config):
  """Training the model"""
  # set up meters
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()
  cm_meter = AverageMeter()
  # number of iterations per epoch
  num_iters = len(train_loader)
  # switch to train mode
  model.train()

  # data prefetcher with noramlization
  train_loader = ClipPrefetcherJoint(train_loader,
    config['input']['mean'], config['input']['std'])

  # main loop
  end = time.time()
  input, target = train_loader.next()
  
  i = 0
  while input is not None:
    # input & target are pre-fetched
    i += 1
    # print(target)
    # compute output
    # print(input.size())
    # print(target[0].size())
    # print(target[1].size())
    # print(target[2].size())
    output, loss = model(input, targets=target)

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # printing (on the first GPU)
    # print(i)
    # print(args.print_freq)
    if (i % args.print_freq) == 0:
      # only check the stats when necessary
      # avoid additional cost at each iter

      acc1, acc5 = accuracy(output.data, target[0], topk=(1, 5))
      batch_cm = confusion_matrix(output.data, target[0])

      # measure accuracy and record loss
      if args.distributed:
        reduced_loss = reduce_tensor(loss.data, args.world_size)
        reduced_acc1 = reduce_tensor(acc1, args.world_size)
        reduced_acc5 = reduce_tensor(acc5, args.world_size)
        reduced_cm = reduce_tensor(batch_cm.data, args.world_size, avg=False)
      else:
        reduced_loss = loss.mean().data
        reduced_acc1 = acc1
        reduced_acc5 = acc5
        reduced_cm = batch_cm.data
      losses.update(reduced_loss.item(), input.size(0))
      top1.update(reduced_acc1.item(), input.size(0))
      top5.update(reduced_acc5.item(), input.size(0))
      cm_meter.update(reduced_cm.cpu().clone())

      # measure elapsed time
      torch.cuda.synchronize()
      batch_time.update((time.time() - end) / args.print_freq)
      end = time.time()

      if args.local_rank == 0:
        lr = scheduler.get_lr()[0]
        print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
          'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'
          'Acc@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
           epoch + 1, i, num_iters, batch_time=batch_time,
           loss=losses, top1=top1, top5=top5))
        # log loss / lr
        writer.add_scalar('data/training_loss',
          losses.val, epoch * num_iters + i)
        writer.add_scalar('data/learning_rate',
          lr, epoch * num_iters + i)

    # step the lr scheduler after each iteration
    scheduler.step()
    # prefetch next batch
    input, target = train_loader.next()

  # finish up
  if args.local_rank == 0:
    # print & step the learning rate
    lr = scheduler.get_lr()[0]
    cls_acc = 100 * mean_class_accuracy(cm_meter.sum)
    print("[Train]: Epoch {:d} finished with lr={:f}".format(epoch + 1, lr))
    # log top-1/5 acc
    writer.add_scalars('data/top1_accuracy',
      {"train" : top1.avg}, epoch + 1)
    writer.add_scalars('data/top5_accuracy',
      {"train" : top5.avg}, epoch + 1)
    writer.add_scalars('data/mean_cls_acc',
      {"train" : cls_acc}, epoch + 1)
  return

def validate(val_loader, model, epoch, args, config):
  """Test the model on the validation set"""
  # set up meters
  batch_time = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()
  cm_meter = AverageMeter()
  # switch to evaluate mode
  model.eval()

  # data prefetcher with noramlization
  val_loader = ClipPrefetcherJoint(val_loader,
    config['input']['mean'], config['input']['std'])

  # loop over validation set
  end = time.time()
  input, target = val_loader.next()
  i = 0
  while input is not None:
    i += 1
    with torch.no_grad():
      # forward the model (without gradients)
      output = model(input)
    # print(target[0])
    # measure accuracy and record loss
    acc1, acc5 = accuracy(output[0].data, target[0], topk=(1, 5))
    batch_cm = confusion_matrix(output[0].data, target[0])
    if args.distributed:
      reduced_acc1 = reduce_tensor(acc1, args.world_size)
      reduced_acc5 = reduce_tensor(acc5, args.world_size)
      reduced_cm = reduce_tensor(batch_cm.data, args.world_size, avg=False)
    else:
      reduced_acc1 = acc1
      reduced_acc5 = acc5
      reduced_cm = batch_cm.data
    top1.update(reduced_acc1.item(), input.size(0))
    top5.update(reduced_acc5.item(), input.size(0))
    cm_meter.update(reduced_cm.cpu().clone())

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    # printing
    if i % (args.print_freq * 2) == 0 and (args.local_rank == 0):
      print('Test: [{0}/{1}]\t'
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'
        'Acc@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
         i, len(val_loader), batch_time=batch_time,
         top1=top1, top5=top5))

    # prefetch next batch
    input, target = val_loader.next()

  # finish up
  if args.local_rank == 0:
    cls_acc = 100 * mean_class_accuracy(cm_meter.sum)
    print('******Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Cls Acc {cls_acc:.3f}'
          .format(top1=top1, top5=top5, cls_acc=cls_acc))
    # log top-1/5 acc
    writer.add_scalars('data/top1_accuracy',
      {"val" : top1.avg}, epoch + 1)
    writer.add_scalars('data/top5_accuracy',
      {"val" : top5.avg}, epoch + 1)
    writer.add_scalars('data/mean_cls_acc',
      {"val" : cls_acc}, epoch + 1)

  return top1.avg, top5.avg


def prec_bn(train_loader, model, epoch, args, config):
  """precise batch norm"""
  # set up meters
  batch_time = AverageMeter()
  # number of iterations per epoch
  num_iters = len(train_loader)
  # switch to train mode & reduce batch norm momentum
  model.train()
  for m in model.modules():
    if isinstance(m, nn.BatchNorm3d):
      m.momentum = 0.05

  # data prefetcher with noramlization
  train_loader = ClipPrefetcherJoint(train_loader,
    config['input']['mean'], config['input']['std'])

  # main loop
  end = time.time()
  input, _ = train_loader.next()
  i = 0
  while input is not None:
    # input & target are pre-fetched
    i += 1

    # no gradient to params
    with torch.no_grad():
      _ = model(input)

    # printing (on the first GPU)
    if (i % (args.print_freq * 2)) == 0 and (args.local_rank == 0):
      # measure elapsed time
      torch.cuda.synchronize()
      batch_time.update((time.time() - end) / (args.print_freq * 2))
      end = time.time()
      print('Prec BN: [{0}][{1}/{2}]\t'
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
         epoch + 1, i, num_iters, batch_time=batch_time))

    # prefetch next batch
    input, _ = train_loader.next()

  # reset bn params back
  for m in model.modules():
    if isinstance(m, nn.BatchNorm3d):
      m.momentum = 0.1
  return

################################################################################
if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
