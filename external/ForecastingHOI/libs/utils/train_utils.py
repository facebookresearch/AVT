from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
################################################################################
"""
Class-Balanced Loss Based on Effective Number of Samples
"""
def get_cls_weights(num_samples_per_cls, beta):
  effective_num = 1.0 - np.power(beta, num_samples_per_cls)
  weights = (1.0 - beta) / effective_num
  weights = weights / np.sum(weights) * float(len(num_samples_per_cls))
  cls_weights = torch.from_numpy(weights.astype(np.float32))
  return cls_weights

################################################################################
"""Helper functions for optimizer / scheduler"""
def filter_bias_bn(model, weight_decay, lr, decoder_ratio=10):
  encoder_decay, encoder_no_decay = [], []
  decoder_decay, decoder_no_decay = [], []

  for name, param in model.encoder.named_parameters():
    # ignore the param without grads
    if not param.requires_grad:
      continue
    if ("bias" in name) or ("bn" in name):
      encoder_no_decay.append(param)
    else:
      encoder_decay.append(param)

  for name, param in model.decoder.named_parameters():
    # ignore the param without grads
    if not param.requires_grad:
      continue
    if ("bias" in name) or ("bn" in name):
      decoder_no_decay.append(param)
    else:
      decoder_decay.append(param)

  return [{'params': encoder_no_decay,
           'weight_decay': 0.,
           'lr': lr},
          {'params': encoder_decay,
           'weight_decay': weight_decay,
           'lr': lr},
          {'params': decoder_no_decay,
           'weight_decay': 0.,
           'lr': decoder_ratio * lr},
          {'params': decoder_decay,
           'weight_decay': weight_decay,
           'lr': decoder_ratio * lr}]

def save_checkpoint(state, is_best, file_folder,
                    filename='checkpoint.pth.tar'):
  """save checkpoint to file"""
  if not os.path.exists(file_folder):
    os.mkdir(file_folder)
  torch.save(state, os.path.join(file_folder, filename))
  if is_best:
    # skip the optimization / scheduler state
    state.pop('optimizer', None)
    state.pop('scheduler', None)
    torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))

def create_optim(model, optimizer_config):
  """get optimizer
  return a supported optimizer
  """
  params = filter_bias_bn(model,
                          optimizer_config["weight_decay"],
                          optimizer_config["learning_rate"],
                          decoder_ratio=optimizer_config["decoder_lr_ratio"])
  if optimizer_config["type"] == "SGD":
    optimizer = optim.SGD(params,
                          lr=optimizer_config["learning_rate"],
                          momentum=optimizer_config["momentum"],
                          nesterov=optimizer_config["nesterov"])
  elif optimizer_config["type"] == "Adam":
    optimizer = optim.Adam(params,
                           lr=optimizer_config["learning_rate"],
                           momentum=optimizer_config["momentum"])
  else:
    raise TypeError("Unsupported solver")

  return optimizer

def create_scheduler(optimizer, schedule_config, max_epochs, num_iters,
                     last_epoch=-1):
  if schedule_config["type"] == "cosine":
    # step per iteration
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
      optimizer, num_iters * max_epochs,
      last_epoch=last_epoch)
  elif schedule_config["type"] == "multistep":
    # step every some epochs
    steps = [num_iters * step for step in schedule_config["steps"]]
    scheduler = optim.lr_scheduler.MultiStepLR(
      optimizer, steps,
      gamma=schedule_config["gamma"], last_epoch=last_epoch)
  else:
    raise TypeError("Unsupported scheduler")

  return scheduler

################################################################################
"""
Mixup data augmentation
"""
def mixup_data(x, y, alpha=1.0):
  '''Returns mixed inputs, pairs of targets, and lambda

  Ref: "Mixup: Beyond Empirical Risk Minimization" ICLR 2018
  '''
  if alpha > 0:
    lam = np.random.beta(alpha, alpha)
  else:
    lam = 1.0

  batch_size = x.size()[0]
  index = torch.randperm(batch_size, device=x.get_device())

  mixed_x = lam * x + (1 - lam) * x[index, :]
  y_a, y_b = y, y[index]
  return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
  return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

################################################################################
"""
When you do care about performance ... Helper functions from Nvidia
"""
# def fast_clip_collate(batch):
#   # note: we drop the clip id here ...
#   clips = [img[0] for img in batch]
#   # print(batch[1])
#   # print(len(batch))
#   # for target in batch:
#   #   print(target[1])
#   #   print(torch.tensor(target[1], dtype=torch.int64))
#     # print(target[2])
#   targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
#   # train
#   if clips[0].ndim == 4:
#     t, h, w, c = clips[0].shape
#     tensor = torch.zeros((len(clips), c, t, h, w), dtype=torch.uint8)
#   # val
#   elif clips[0].ndim == 5:
#     k, t, h, w, c = clips[0].shape
#     tensor = torch.zeros((len(clips), k, c, t, h, w), dtype=torch.uint8)
#   else:
#     raise TypeError("Clip dimension mis-match")

#   for idx, clip in enumerate(clips):
#     if clip.ndim == 4:
#       numpy_array = np.ascontiguousarray(clip.transpose((3, 0, 1, 2)))
#     else:
#       numpy_array = np.ascontiguousarray(clip.transpose((0, 4, 1, 2, 3)))
#     tensor[idx] += torch.from_numpy(numpy_array)
#   return tensor, targets

# def fast_clip_collate_test(batch):
#   # note: we drop the clip id here ...
#   clips = [img[0] for img in batch]
#   clip_ids = [img[2] for img in batch]
#   # print(batch[1])
#   # print(len(batch))
#   # for target in batch:
#   #   print(target[1])
#   #   print(torch.tensor(target[1], dtype=torch.int64))
#     # print(target[2])
#   targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
#   # train
#   if clips[0].ndim == 4:
#     t, h, w, c = clips[0].shape
#     tensor = torch.zeros((len(clips), c, t, h, w), dtype=torch.uint8)
#   # val
#   elif clips[0].ndim == 5:
#     k, t, h, w, c = clips[0].shape
#     tensor = torch.zeros((len(clips), k, c, t, h, w), dtype=torch.uint8)
#   else:
#     raise TypeError("Clip dimension mis-match")

#   for idx, clip in enumerate(clips):
#     if clip.ndim == 4:
#       numpy_array = np.ascontiguousarray(clip.transpose((3, 0, 1, 2)))
#     else:
#       numpy_array = np.ascontiguousarray(clip.transpose((0, 4, 1, 2, 3)))
#     tensor[idx] += torch.from_numpy(numpy_array)
#   return tensor, targets, clip_ids

def fast_clip_collate_joint(batch):
  # note: we drop the clip id here ...
  clips = [img[0][0] for img in batch]
  hands = [img[0][1] for img in batch]
  hotspots = [img[0][2] for img in batch]
  
  # print(batch[1])
  # print(len(batch))
  # for target in batch:
  #   print(target[1])
  #   print(torch.tensor(target[1], dtype=torch.int64))
    # print(target[2])
  # print(len(clips))
  targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
  # train
  if clips[0].ndim == 4:
    t, h, w, c = clips[0].shape
    tensor = torch.zeros((len(clips), c, t, h, w), dtype=torch.uint8)

    t, h, w= hands[0].shape
    hand_tensor = torch.zeros((len(clips),1, t, h, w), dtype=torch.double)

    h, w= hotspots[0].shape
    hotspot_tensor = torch.zeros((len(clips),1, 1, h, w), dtype=torch.double)
  # val
  elif clips[0].ndim == 5:
    k, t, h, w, c = clips[0].shape
    tensor = torch.zeros((len(clips), k, c, t, h, w), dtype=torch.uint8)

    k,t, h, w= hands[0].shape
    hand_tensor = torch.zeros((len(clips),k, t, h, w), dtype=torch.double)

    k,h, w= hotspots[0].shape
    hotspot_tensor = torch.zeros((len(clips),k,1, h, w), dtype=torch.double)
  else:
    raise TypeError("Clip dimension mis-match")
  # print(len(clips))
  for idx, clip in enumerate(clips):
    if clip.ndim == 4:
      numpy_array = np.ascontiguousarray(clip.transpose((3, 0, 1, 2)))
      hand_array = np.ascontiguousarray(hands[idx])
      hotspot_array = np.ascontiguousarray(hotspots[idx])
    else:
      numpy_array = np.ascontiguousarray(clip.transpose((0, 4, 1, 2, 3)))
      hand_array = np.ascontiguousarray(hands[idx])
      hotspot_array = np.ascontiguousarray(hotspots[idx])

    tensor[idx] += torch.from_numpy(numpy_array)
    hand_tensor[idx,:,:,:,:] += torch.from_numpy(hand_array)
    hotspot_tensor[idx,:,0,:,:] += torch.from_numpy(hotspot_array)

  return tensor, (targets,hand_tensor,hotspot_tensor)


def fast_clip_collate_joint_test(batch):
  # note: we drop the clip id here ...
  clips = [img[0][0] for img in batch]
  hands = [img[0][1] for img in batch]
  hotspots = [img[0][2] for img in batch]
  clip_ids = [img[2] for img in batch]
  # print(batch[1])
  # print(len(batch))
  # for target in batch:
  #   print(target[1])
  #   print(torch.tensor(target[1], dtype=torch.int64))
    # print(target[2])
  # print(len(clips))
  targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
  # train
  if clips[0].ndim == 4:
    t, h, w, c = clips[0].shape
    tensor = torch.zeros((len(clips), c, t, h, w), dtype=torch.uint8)

    t, h, w= hands[0].shape
    hand_tensor = torch.zeros((len(clips),1, t, h, w), dtype=torch.double)

    h, w= hotspots[0].shape
    hotspot_tensor = torch.zeros((len(clips),1, 1, h, w), dtype=torch.double)
  # val
  elif clips[0].ndim == 5:
    k, t, h, w, c = clips[0].shape
    tensor = torch.zeros((len(clips), k, c, t, h, w), dtype=torch.uint8)

    k,t, h, w= hands[0].shape
    hand_tensor = torch.zeros((len(clips),k, t, h, w), dtype=torch.double)

    k,h, w= hotspots[0].shape
    hotspot_tensor = torch.zeros((len(clips),k,1, h, w), dtype=torch.double)
  else:
    raise TypeError("Clip dimension mis-match")
  # print(len(clips))
  for idx, clip in enumerate(clips):
    if clip.ndim == 4:

      numpy_array = np.ascontiguousarray(clip.transpose((3, 0, 1, 2)))
      hand_array = np.ascontiguousarray(hands[idx])
      hotspot_array = np.ascontiguousarray(hotspots[idx])
    else:
      numpy_array = np.ascontiguousarray(clip.transpose((0, 4, 1, 2, 3)))
      hand_array = np.ascontiguousarray(hands[idx])
      hotspot_array = np.ascontiguousarray(hotspots[idx])

    tensor[idx] += torch.from_numpy(numpy_array)
    hand_tensor[idx,:,:,:,:] += torch.from_numpy(hand_array)
    hotspot_tensor[idx,:,0,:,:] += torch.from_numpy(hotspot_array)
    # print(torch.max(hotspot_tensor))
  return tensor, (targets,hand_tensor,hotspot_tensor), clip_ids


class ClipPrefetcherJoint():
  """Efficient data prefetching from nvidia"""
  def __init__(self, loader, mean, std):
    self.num_samples = len(loader)
    self.loader = iter(loader)
    self.stream = torch.cuda.Stream()
    self.mean = torch.tensor([255.0*val for val in mean]).cuda().view(1,3,1,1,1)
    self.std = torch.tensor([255.0*val for val in std]).cuda().view(1,3,1,1,1)
    self.preload()
    self.maxpool1 = nn.MaxPool3d(kernel_size=(3, 8, 8),
                                   stride=(3, 8, 8),
                                   padding=(0, 1, 1))
    self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 16, 16),
                                   stride=(1, 16, 16),
                                   padding=(0, 1, 1))
  def __len__(self):
    return self.num_samples

  def preload(self):
    try:
      self.next_input, self.next_target = next(self.loader)
    except StopIteration:
      self.next_input = None
      self.next_target = None
      return

    with torch.cuda.stream(self.stream):
      self.next_input = self.next_input.cuda(non_blocking=True)

      self.next_target = (self.next_target[0].cuda(non_blocking=True),
                          self.next_target[1].cuda(non_blocking=True),
                          self.next_target[2].cuda(non_blocking=True))

      self.next_input = self.next_input.float()
      self.next_input = self.next_input.sub_(self.mean).div_(self.std)

  def next(self):
    torch.cuda.current_stream().wait_stream(self.stream)
    input = self.next_input
    target = self.next_target
    if input is not None:
      input.record_stream(torch.cuda.current_stream())
    if target is not None:
      target[0].record_stream(torch.cuda.current_stream())
      target[1].record_stream(torch.cuda.current_stream())

      downsample_hand = self.maxpool1(target[1])

      batch_size,k, T,H,W = downsample_hand.shape
      downsample_hand = downsample_hand.view(batch_size, k, T, -1)
      normalizer = torch.sum(downsample_hand,dim=-1)
      normalizer = normalizer.view(batch_size, k, T,1)

      downsample_hand = downsample_hand/normalizer
      downsample_hand = downsample_hand.view(batch_size, k, T, H,W)
      target[2].record_stream(torch.cuda.current_stream())

      downsample_hotspot = self.maxpool2(target[2])
      batch_size,k, T,H,W = downsample_hotspot.shape
      downsample_hotspot = downsample_hotspot.view(batch_size, k, T, -1)
      normalizer = torch.sum(downsample_hotspot,dim=-1)
      normalizer = normalizer.view(batch_size, k, T,1)

      downsample_hotspot = downsample_hotspot/normalizer
      downsample_hotspot = downsample_hotspot.view(batch_size, k, T, H,W)
      self.preload()
      return input, (target[0],downsample_hand.type(torch.cuda.FloatTensor),downsample_hotspot.type(torch.cuda.FloatTensor))
    else:
      return input, target

      # print(torch.sum(downsample_hotspot))

class ClipPrefetcherJointTest():
  """Efficient data prefetching from nvidia"""
  def __init__(self, loader, mean, std):
    self.num_samples = len(loader)
    self.loader = iter(loader)
    self.stream = torch.cuda.Stream()
    self.mean = torch.tensor([255.0*val for val in mean]).cuda().view(1,3,1,1,1)
    self.std = torch.tensor([255.0*val for val in std]).cuda().view(1,3,1,1,1)
    self.preload()
    self.maxpool1 = nn.MaxPool3d(kernel_size=(3, 8, 8),
                                   stride=(3, 8, 8),
                                   padding=(0, 1, 1))
    self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 16, 16),
                                   stride=(1, 16, 16),
                                   padding=(0, 1, 1))
  def __len__(self):
    return self.num_samples

  def preload(self):
    try:
      self.next_input, self.next_target, self.next_clipID = next(self.loader)
    except StopIteration:
      self.next_input = None
      self.next_target = None
      self.next_clipID = None
      return
    with torch.cuda.stream(self.stream):
      self.next_input = self.next_input.cuda(non_blocking=True)

      self.next_target = (self.next_target[0].cuda(non_blocking=True),
                          self.next_target[1].cuda(non_blocking=True),
                          self.next_target[2].cuda(non_blocking=True))
      self.next_clipID = self.next_clipID

      # normalization (on GPU)
      self.next_input = self.next_input.float()
      self.next_input = self.next_input.sub_(self.mean).div_(self.std)

  def next(self):
    torch.cuda.current_stream().wait_stream(self.stream)
    input = self.next_input
    target = self.next_target
    clipId = self.next_clipID
    if input is not None:
      input.record_stream(torch.cuda.current_stream())
    if target is not None:
      target[0].record_stream(torch.cuda.current_stream())
      target[1].record_stream(torch.cuda.current_stream())

      downsample_hand = self.maxpool1(target[1])
      batch_size,k, T,H,W = downsample_hand.shape
      downsample_hand = downsample_hand.view(batch_size, k, T, -1)
      normalizer = torch.sum(downsample_hand,dim=-1)
      normalizer = normalizer.view(batch_size, k, T,1)
      downsample_hand = downsample_hand/normalizer
      downsample_hand = downsample_hand.view(batch_size, k, T, H,W)

      # print(torch.sum(downsample_hand))
      target[2].record_stream(torch.cuda.current_stream())

      downsample_hotspot = self.maxpool2(target[2])

      batch_size,k, T,H,W = downsample_hotspot.shape
      downsample_hotspot = downsample_hotspot.view(batch_size, k, T, -1)
      normalizer = torch.sum(downsample_hotspot,dim=-1)
      normalizer = normalizer.view(batch_size, k, T,1)
      downsample_hotspot = downsample_hotspot/normalizer
      downsample_hotspot = downsample_hotspot.view(batch_size, k, T, H,W)
      self.preload()
      return input, (target[0],downsample_hand.type(torch.cuda.FloatTensor),downsample_hotspot.type(torch.cuda.FloatTensor)), clipId 
    else:
      return input, target, clipId


################################################################################
def sync_processes():
  torch.distributed.barrier()
  # try to address a weird memory leak
  torch.cuda.empty_cache()

def reduce_tensor(tensor, world_size, avg=True):
  rt = tensor.clone()
  torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
  if avg:
    rt /= world_size
  return rt
