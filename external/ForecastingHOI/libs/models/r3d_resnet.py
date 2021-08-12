"""
Model ported from https://github.com/facebookresearch/VMZ
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import torch.utils.checkpoint as cp

from .depthwise_conv3d import DepthwiseConv3d

# def get_mask(in_channels, channels, t, h, w):
#   mask = np.zeros((in_channels, channels, t, h, w), dtype=np.float32)
#   for idx in range(in_channels):
#     mask[idx, idx % channels, :, :, :] = 1.
#   return mask
#
# class DepthwiseConv3d(nn.Module):
#   """
#     Efficient Implementation of Depthwise Conv 3D by using diagnoal refactorization
#   """
#   def __init__(self, in_channels, kernel_size, stride=1, padding=0, bias=False):
#     super(DepthwiseConv3d, self).__init__()
#     out_channels = in_channels
#     groups = max(in_channels // 32, 1)
#     channels = in_channels // groups
#     self.in_channels = in_channels
#     self.groups = groups
#     self.stride = stride
#     self.padding = padding
#     self.register_buffer('mask', torch.Tensor(
#       get_mask(in_channels, channels, kernel_size[0], kernel_size[1], kernel_size[2]))
#     )
#     self.weight = nn.Parameter(torch.Tensor(
#       in_channels, channels, kernel_size[0], kernel_size[1], kernel_size[2]),
#       requires_grad=True)
#
#     if bias:
#       self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1, 1))
#     else:
#       self.register_parameter('bias', None)
#
#     self.reset_params()
#
#   def reset_params(self):
#     nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
#     self.weight.data.mul_(self.mask.data)
#     if self.bias is not None:
#       nn.init.constant_(self.bias, 0)
#
#   def forward(self, x):
#     weight = torch.mul(self.weight, self.mask)
#     x = torch.nn.functional.conv3d(x, weight, bias=self.bias,
#       stride=self.stride, padding=self.padding, groups=self.groups)
#     return x


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride, downsample, gradient_cp=False):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv3d(inplanes, planes,
                           kernel_size=(1, 1, 1),
                           stride=(1, 1, 1),
                           padding=(0, 0, 0),
                           bias=False)
    self.bn1 = nn.BatchNorm3d(planes, eps=1e-3)
    self.conv2 = DepthwiseConv3d(planes,
                           kernel_size=(3, 3, 3),
                           stride=stride,
                           padding=(1, 1, 1),
                           bias=False)
    self.bn2 = nn.BatchNorm3d(planes, eps=1e-3)
    self.conv3 = nn.Conv3d(planes, planes * self.expansion,
                           kernel_size=1,
                           stride=(1, 1, 1),
                           padding=(0, 0, 0),
                           bias=False)
    self.bn3 = nn.BatchNorm3d(planes * self.expansion, eps=1e-3)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride
    self.gradient_cp = gradient_cp

  def forward(self, x):

    def _inner_forward(x):
      residual = x

      out = self.conv1(x)
      out = self.bn1(out)
      out = self.relu(out)

      out = self.conv2(out)
      out = self.bn2(out)
      out = self.relu(out)

      out = self.conv3(out)
      out = self.bn3(out)

      if self.downsample is not None:
        residual = self.downsample(x)

      out += residual
      return out

    if self.gradient_cp and x.requires_grad:
      out = cp.checkpoint(_inner_forward, x)
    else:
      out = _inner_forward(x)
    out = self.relu(out)
    return out

#-----------------------------------------------------------------------------------------------#

class irCSN152(nn.Module):
  """
  R3D Net (architecture is hand-coded)
  This is the resnet_video in FB repo (modified from caffe2)
  Note: this model used a non-standard BN eps
  """
  def __init__(self,
               block=Bottleneck,
               layers=[3, 8, 36, 3],
               frozen_stages=-1,
               modality='rgb',
               gradient_cp=False,
               pretrained=None):
    super(irCSN152, self).__init__()

    # sanity check
    assert modality == 'rgb'
    self.inplanes = 64
    self.modality = modality
    self.frozen_stages = frozen_stages

    # network ops
    # conv block 1: L * 224 * 224 -> L * 56 * 56
    self.conv1 = nn.Conv3d(3, 64,
                           kernel_size=(3, 7, 7),
                           stride=(1, 2, 2),
                           padding=(1, 3, 3),
                           bias=False)
    self.bn1 = nn.BatchNorm3d(64, eps=1e-3)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                 stride=(1, 2, 2),
                                 padding=(0, 1, 1))
    # res block 1: L * 56 * 56 -> L * 56 * 56
    self.layer1 = self._make_layer(block, 64, layers[0],
                                   stride=1, gradient_cp=gradient_cp)
    # res block 2: L * 56 * 56 -> L/2 * 28 * 28
    self.layer2 = self._make_layer(block, 128, layers[1],
                                   stride=2, gradient_cp=gradient_cp)
    # res block 3: L/2 * 28 * 28 -> L/4 * 14 * 14
    self.layer3 = self._make_layer(block, 256, layers[2],
                                   stride=2, gradient_cp=gradient_cp)
    # res block 4: L/4 * 14 * 14 -> L/8 * 7 * 7
    self.layer4 = self._make_layer(block, 512, layers[3],
                                   stride=2, gradient_cp=gradient_cp)

    # load pre-trained model or re-init weights
    if pretrained is not None:
      print("Loading pre-trained weights from {:s}".format(pretrained))
      missing_keys, unexpected_keys = \
        self.load_state_dict(torch.load(pretrained), strict=False)
      if missing_keys:
        print("Missing keys:")
        for key in missing_keys:
          print(key)
      if unexpected_keys:
        print("Unexpected keys:")
        for key in unexpected_keys:
          print(key)
    else:
      self._init_weights()

    # freeze part of the model
    self._freeze_stages()

  def _make_layer(self, block, planes, blocks, stride=1, gradient_cp=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv3d(self.inplanes, planes * block.expansion,
                  kernel_size=(1, 1, 1),
                  stride=(stride, stride, stride),
                  padding=(0, 0, 0),
                  bias=False),
        nn.BatchNorm3d(planes * block.expansion, eps=1e-3))

    layers = []
    layers.append(block(
      self.inplanes, planes, stride, downsample, gradient_cp=gradient_cp))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(
        self.inplanes, planes, 1, None, gradient_cp=gradient_cp))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool1(x)
    x1 = self.layer1(x)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)
    # Modified by rgirdhar: To be usable with the larger codebase, just return
    # the final feature map
    # # return a tuple of feature maps
    # out = (x1, x2, x3, x4)
    out = x4
    return out

  def _freeze_stages(self):
    # train the full network with frozen_stages < 0
    if self.frozen_stages < 0:
      return
    # a bit ugly, stages from [0, 4] - the five conv blocks
    stage_mapping = [
      [self.conv1, self.bn1],
      [self.layer1],
      [self.layer2],
      [self.layer3],
      [self.layer4]
    ]
    # freeze the params (but still allow bn to aggregate the stats)
    for idx in range(self.frozen_stages+1):
      for m in stage_mapping[idx]:
        for param in m.parameters():
          if param.requires_grad:
            param.requires_grad = False

  def _init_weights(self):
    # simple param init
    for m in self.modules():
      if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def train(self, mode=True):
    super(irCSN152, self).train(mode)
    self._freeze_stages()
