"""
Modified code from resnet3d repo https://github.com/Tushar-N/pytorch-resnet3d
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride, downsample, temp_conv, temp_stride):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv3d(inplanes, planes,
                           kernel_size=(1 + temp_conv * 2, 1, 1),
                           stride=(temp_stride, 1, 1),
                           padding=(temp_conv, 0, 0),
                           bias=False)
    self.bn1 = nn.BatchNorm3d(planes)
    self.conv2 = nn.Conv3d(planes, planes,
                           kernel_size=(1, 3, 3),
                           stride=(1, stride, stride),
                           padding=(0, 1, 1),
                           bias=False)
    self.bn2 = nn.BatchNorm3d(planes)
    self.conv3 = nn.Conv3d(planes, planes * self.expansion,
                           kernel_size=1,
                           stride=(1, 1, 1),
                           padding=(0, 0, 0),
                           bias=False)
    self.bn3 = nn.BatchNorm3d(planes * self.expansion)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
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
    out = self.relu(out)

    return out

#-----------------------------------------------------------------------------------------------#

class I3DRes50(nn.Module):
  """
  I3D ResNet50 (architecture is hand-coded)
  This is the resnet_video in FB repo (modified arch)
  """
  def __init__(self,
               block=Bottleneck,
               layers=[3, 4, 6, 3],
               frozen_stages=-1,
               modality='rgb',
               pretrained=None,
               version='v2'):
    super(I3DRes50, self).__init__()

    # sanity check
    assert modality == 'rgb'
    self.inplanes = 64
    self.modality = modality
    self.frozen_stages = frozen_stages

    # network ops
    if version == 'v2':
      self.conv1 = nn.Conv3d(3, 64,
                             kernel_size=(5, 7, 7),
                             stride=(1, 2, 2),
                             padding=(2, 3, 3),
                             bias=False)
    elif version == 'v1':
      self.conv1 = nn.Conv3d(3, 64,
                             kernel_size=(5, 7, 7),
                             stride=(2, 2, 2),
                             padding=(2, 3, 3),
                             bias=False)
    else:
      raise TypeError("3D ResNet version not supported!")

    self.bn1 = nn.BatchNorm3d(64)
    self.relu = nn.ReLU(inplace=True)
    # caffe 2 has a different way of computing the output size for pooling
    # Our code is not technically right (should use ceil_mode=True)
    # but it makes the signal center aligned
    if version == 'v2':
      self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                   stride=(1, 2, 2),
                                   padding=(0, 1, 1))
    elif version == 'v1':
      self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3),
                                   stride=(2, 2, 2),
                                   padding=(0, 1, 1))
    else:
      raise TypeError("3D ResNet version not supported!")

    self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1),
                                 stride=(2, 1, 1),
                                 padding=(0, 0, 0))
    self.layer1 = self._make_layer(block, 64, layers[0],
                                   stride=1,
                                   temp_conv=[1, 1, 1],
                                   temp_stride=[1, 1, 1])
    self.layer2 = self._make_layer(block, 128, layers[1],
                                   stride=2,
                                   temp_conv=[1, 0, 1, 0],
                                   temp_stride=[1, 1, 1, 1])
    self.layer3 = self._make_layer(block, 256, layers[2],
                                   stride=2,
                                   temp_conv=[1, 0, 1, 0, 1, 0],
                                   temp_stride=[1, 1, 1, 1, 1, 1])
    self.layer4 = self._make_layer(block, 512, layers[3],
                                   stride=2,
                                   temp_conv=[0, 1, 0],
                                   temp_stride=[1, 1, 1])
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

  def _make_layer(self, block, planes, blocks, stride, temp_conv, temp_stride):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion \
       or temp_stride[0] != 1:
      downsample = nn.Sequential(
        nn.Conv3d(self.inplanes, planes * block.expansion,
                  kernel_size=(1, 1, 1),
                  stride=(temp_stride[0], stride, stride),
                  padding=(0, 0, 0),
                  bias=False),
        nn.BatchNorm3d(planes * block.expansion))

    layers = []
    layers.append(block(
      self.inplanes, planes, stride, downsample, temp_conv[0], temp_stride[0]))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(
        self.inplanes, planes, 1, None, temp_conv[i], temp_stride[i]))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool1(x)

    x1 = self.maxpool2(self.layer1(x))
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)

    # return a tuple of feature maps
    out = (x1, x2, x3, x4)
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
    super(I3DRes50, self).train(mode)
    self._freeze_stages()

# test
if __name__=='__main__':
  net = I3DRes50()
