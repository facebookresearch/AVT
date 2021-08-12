"""
loss functions for the models
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class CustomLoss(nn.Module):
  """
  A thin wrapper for loss functions
  L = sum_i (lam_i * loss_i)
  """
  def __init__(self, criterion_list, lam_list):
    super(CustomLoss, self).__init__()
    assert len(criterion_list) > 0
    # compile the list of loss functions into a module list
    # we assume same order of the loss w.r.t. the output vars
    assert len(criterion_list) == len(lam_list)
    self.criterion = nn.ModuleList(criterion_list)
    self.register_buffer('lam', torch.FloatTensor(lam_list))

  def forward(self, preds, targets):
    assert len(self.criterion) == len(preds)
    assert len(targets) <= len(self.criterion)
    # attach loss to each of the outputs
    # print(preds[0])
    # print(targets[0])
    loss = self.lam[0] * self.criterion[0](preds[0], targets[0])
    # print(loss)
    for idx in range(1, len(self.criterion)):
      target = targets[idx] if idx < len(targets) else None
      loss = loss + self.lam[idx] * self.criterion[idx](preds[idx], target)
    return loss

class SmoothedCrossEntropy(nn.Module):
  """Labled smoothing with cross entropy loss
  Note that this version is slightly different from our previous code
  Specifically, the smooth facotor is scaled by K / (K-1). This new
  implementation follows Hinton's orginal paper, and is slight faster and memory
  efficient (also supports ignore_index & weight)
  Weight: if specified, must be a pytorch tensor of size C
  """
  def __init__(self, label_smoothing=0.1, weight=None, ignore_index=None):
    super(SmoothedCrossEntropy, self).__init__()
    assert (label_smoothing < 1) and (label_smoothing > 0)
    self.label_smoothing = label_smoothing
    self.ignore_index = ignore_index
    self.register_buffer('weight', weight)

  def forward(self, pred, target):
    """ pred must be N * C """
    # create ignore mask
    if self.ignore_index is not None:
      ignore_mask = (target == self.ignore_index)
      valid_num_samples = float(pred.size(0) - sum(ignore_mask).item())
    else:
      valid_num_samples = float(pred.size(0))

    # divide the loss into two terms:
    # (\alpha - 1) \sum w_k y_k p_k - \alpha \sum w_k p_k /K
    log_prb = nn.functional.log_softmax(pred, dim=1)
    if self.weight is not None:
      log_prb = log_prb * self.weight
    ll_loss = log_prb.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
    smooth_loss = log_prb.mean(dim=1)
    losses = (self.label_smoothing - 1.0) * ll_loss \
             - self.label_smoothing * smooth_loss

    # mask out invalid samples & avg
    if self.ignore_index is not None:
      losses.masked_fill_(ignore_mask, 0.0)
    loss = losses.sum(0).div_(valid_num_samples)
    return loss

class KLDiv(nn.Module):
  """
    KL divergence for 3D attention maps
  """
  def __init__(self):
    super(KLDiv, self).__init__()
    self.register_buffer('norm_scalar', torch.tensor(1, dtype=torch.float32))

  def forward(self, pred, target=None):
    # get output shape
    batch_size, T = pred.shape[0], pred.shape[2]
    H, W = pred.shape[3], pred.shape[4]
    # N T HW
    atten_map = pred.view(batch_size, T, -1)
    log_atten_map = torch.log(atten_map)
    # print(torch.sum(target))
    # print(torch.max(target))
    # print(torch.min(target))
    if target is None:
      # uniform prior: this is really just neg entropy
      # we keep the loss scale the same here
      log_q = torch.log(self.norm_scalar / float(H*W))
      # \sum p logp - log(1/hw) -> N T
      kl_losses = (atten_map * log_atten_map).sum(dim=-1) - log_q
    else:
      log_q = torch.log(target.view(batch_size, T, -1))
      # \sum p logp - \sum p logq -> N T
      kl_losses = (atten_map * log_atten_map).sum(dim=-1) \
                  - (atten_map * log_q).sum(dim=-1)
    # N T -> N
    norm_scalar = T * torch.log(self.norm_scalar * H * W)
    kl_losses = kl_losses.sum(dim=-1) / norm_scalar
    kl_loss = kl_losses.mean()
    return kl_loss
