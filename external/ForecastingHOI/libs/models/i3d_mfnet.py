"""
Modified code from MFNet repo:
https://github.com/cypw/PyTorch-MFNet
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import torch
import torch.nn as nn

"""
Till someone patch the group conv code in pytorch,
this arch is even slower than res50...
"""

class BN_AC_CONV3D(nn.Module):
  def __init__(self, num_in, num_filter,
               kernel=(1,1,1), pad=(0,0,0), stride=(1,1,1), g=1, bias=False):
    super(BN_AC_CONV3D, self).__init__()
    self.bn = nn.BatchNorm3d(num_in)
    self.relu = nn.ReLU(inplace=True)
    self.conv = nn.Conv3d(num_in, num_filter, kernel_size=kernel, padding=pad,
                          stride=stride, groups=g, bias=bias)

  def forward(self, x):
    h = self.relu(self.bn(x))
    h = self.conv(h)
    return h

class MF_UNIT(nn.Module):
  def __init__(self, num_in, num_mid, num_out,
               g=1, stride=(1,1,1), first_block=False, use_3d=True):
    super(MF_UNIT, self).__init__()
    num_ix = int(num_mid/4)
    kt, pt = (3,1) if use_3d else (1,0)
    self.first_block = first_block
    # prepare input
    self.conv_i1 = BN_AC_CONV3D(num_in=num_in,
                                num_filter=num_ix,
                                kernel=(1,1,1),
                                pad=(0,0,0))
    self.conv_i2 = BN_AC_CONV3D(num_in=num_ix,
                                num_filter=num_in,
                                kernel=(1,1,1),
                                pad=(0,0,0))
    # main part
    self.conv_m1 = BN_AC_CONV3D(num_in=num_in,
                                num_filter=num_mid,
                                kernel=(kt,3,3),
                                pad=(pt,1,1),
                                stride=stride,
                                g=g)
    if first_block:
      self.conv_m2 = BN_AC_CONV3D(num_in=num_mid,
                                  num_filter=num_out,
                                  kernel=(1,1,1),
                                  pad=(0,0,0))
      self.conv_w1 = BN_AC_CONV3D(num_in=num_in,
                                  num_filter=num_out,
                                  kernel=(1,1,1),
                                  pad=(0,0,0),
                                  stride=stride)
    else:
      self.conv_m2 = BN_AC_CONV3D(num_in=num_mid,
                                  num_filter=num_out,
                                  kernel=(1,3,3),
                                  pad=(0,1,1),
                                  g=g)

  def forward(self, x):
    h = self.conv_i1(x)
    x_in = x + self.conv_i2(h)
    h = self.conv_m1(x_in)
    h = self.conv_m2(h)
    if self.first_block:
      x = self.conv_w1(x)
    return h + x


class I3DMFNet(nn.Module):
  def __init__(self,
               frozen_stages=-1,
               modality='rgb',
               pretrained=None):
    super(I3DMFNet, self).__init__()

    # sanity check
    assert modality == 'rgb'
    self.modality = modality
    self.frozen_stages = frozen_stages

    # hand coded params
    groups = 16
    k_sec  = {  2: 3, \
                3: 4, \
                4: 6, \
                5: 3  }

    # conv1 - x224 (x16)
    conv1_num_out = 16
    self.conv1 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv3d(3, conv1_num_out,
                                   kernel_size=(3,5,5),
                                   padding=(1,2,2),
                                   stride=(1,2,2),
                                   bias=False)),
                ('bn', nn.BatchNorm3d(conv1_num_out)),
                ('relu', nn.ReLU(inplace=True))
                ]))
    self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3),
                                stride=(1,2,2),
                                padding=(0,1,1))

    # conv2 - x56 (x8)
    num_mid = 96
    conv2_num_out = 96
    self.conv2 = nn.Sequential(OrderedDict([
      ("B%02d"%i, MF_UNIT(num_in=conv1_num_out if i==1 else conv2_num_out,
                          num_mid=num_mid,
                          num_out=conv2_num_out,
                          stride=(2,1,1) if i==1 else (1,1,1),
                          g=groups,
                          first_block=(i==1))) for i in range(1,k_sec[2]+1)
      ]))

    # conv3 - x28 (x8)
    num_mid *= 2
    conv3_num_out = 2 * conv2_num_out
    self.conv3 = nn.Sequential(OrderedDict([
      ("B%02d"%i, MF_UNIT(num_in=conv2_num_out if i==1 else conv3_num_out,
                          num_mid=num_mid,
                          num_out=conv3_num_out,
                          stride=(1,2,2) if i==1 else (1,1,1),
                          g=groups,
                          first_block=(i==1))) for i in range(1,k_sec[3]+1)
      ]))

    # conv4 - x14 (x8)
    num_mid *= 2
    conv4_num_out = 2 * conv3_num_out
    self.conv4 = nn.Sequential(OrderedDict([
      ("B%02d"%i, MF_UNIT(num_in=conv3_num_out if i==1 else conv4_num_out,
                          num_mid=num_mid,
                          num_out=conv4_num_out,
                          stride=(1,2,2) if i==1 else (1,1,1),
                          g=groups,
                          first_block=(i==1))) for i in range(1,k_sec[4]+1)
      ]))

    # conv5 - x7 (x8)
    num_mid *= 2
    conv5_num_out = 2 * conv4_num_out
    self.conv5 = nn.Sequential(OrderedDict([
      ("B%02d"%i, MF_UNIT(num_in=conv4_num_out if i==1 else conv5_num_out,
                          num_mid=num_mid,
                          num_out=conv5_num_out,
                          stride=(1,2,2) if i==1 else (1,1,1),
                          g=groups,
                          first_block=(i==1))) for i in range(1,k_sec[5]+1)
      ]))

    # final
    self.tail = nn.Sequential(OrderedDict([
                ('bn', nn.BatchNorm3d(conv5_num_out)),
                ('relu', nn.ReLU(inplace=True))
                ]))

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

  def forward(self, x):
    x = self.conv1(x)    # x224 ->x112
    x = self.maxpool(x)  # x112 -> x56
    x1 = self.conv2(x)   # x56 -> x56
    x2 = self.conv3(x1)   # x56 -> x28
    x3 = self.conv4(x2)   # x28 -> x14
    x4 = self.tail(self.conv5(x3))   # x14 -> x7 / final relu + bn

    # return a tuple of feature maps
    out = (x1, x2, x3, x4)
    return out

  def _freeze_stages(self):
    # train the full network with frozen_stages < 0
    if self.frozen_stages < 0:
      return
    # a bit ugly, stages from [0, 4] - the five conv blocks
    stage_mapping = [
      [self.conv1],
      [self.conv2],
      [self.conv3],
      [self.conv4],
      [self.conv5, self.tail]
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
    super(I3DMFNet, self).train(mode)
    self._freeze_stages()

if __name__ == "__main__":
  net = I3DMFNet()
