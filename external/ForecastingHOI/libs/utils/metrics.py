from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import torch


def to_onehot(indices, num_classes):
  """Convert a tensor of indices of any shape `(N, ...)` to a
  tensor of one-hot indicators of shape `(N, num_classes, ...)`.
  """
  onehot = torch.zeros(indices.shape[0],
                       num_classes,
                       *indices.shape[1:],
                       device=indices.device)
  # rgirdhar: When test on test set, there will be some data points where
  # we don't have the labels
  return onehot.scatter_(1, indices[indices >= 0].unsqueeze(1), 1)


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.initialized = False
    self.val = None
    self.avg = None
    self.sum = None
    self.count = 0.0

  def initialize(self, val, n):
    self.val = val
    self.avg = val
    self.sum = val * n
    self.count = n
    self.initialized = True

  def update(self, val, n=1):
    if not self.initialized:
      self.initialize(val, n)
    else:
      self.add(val, n)

  def add(self, val, n):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def confusion_matrix(pred, target):
  num_classes = pred.shape[1]
  assert pred.shape[0] == target.shape[0]
  with torch.no_grad():
    target_ohe = to_onehot(target, num_classes)
    target_ohe_t = target_ohe.transpose(0, 1).float()

    pred_idx = torch.argmax(pred, dim=1)
    pred_ohe = to_onehot(pred_idx.reshape(-1), num_classes)
    pred_ohe = pred_ohe.float()

    confusion_matrix = torch.matmul(target_ohe_t, pred_ohe)
  return confusion_matrix

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
  return res

def mean_class_accuracy(cm):
  """Compute mean class accuracy based on the input confusion matrix"""
  # Increase floating point precision
  cm = cm.type(torch.float64)
  cls_cnt = cm.sum(dim=1) + 1e-15
  cls_hit = cm.diag()
  cls_acc = (cls_hit/cls_cnt).mean().item()
  return cls_acc
