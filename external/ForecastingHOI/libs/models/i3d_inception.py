"""
Modified code from I3D repo https://github.com/hassony2/kinetics_i3d_pytorch
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from .model_utils import get_padding_shape, simplify_padding

class Unit3Dpy(nn.Module):
  """
  Conv3D + BN3D + Relu
  """
  def __init__(self,
               in_channels,
               out_channels,
               kernel_size=(1, 1, 1),
               stride=(1, 1, 1),
               activation='relu',
               padding='SAME',
               use_bias=False,
               use_bn=True):
    super(Unit3Dpy, self).__init__()

    # setup params
    self.padding = padding
    self.activation = activation
    self.use_bn = use_bn
    self.stride = stride

    # follow the padding of tensorflow (somewhat complicated logic here)
    if padding == 'SAME':
      padding_shape = get_padding_shape(kernel_size, stride)
      simplify_pad, pad_size = simplify_padding(padding_shape)
      self.simplify_pad = simplify_pad
      if stride[0] > 1:
        padding_shapes = [get_padding_shape(kernel_size, stride, mod) for
                          mod in range(stride[0])]
      else:
        padding_shapes = [padding_shape]
    elif padding == 'VALID':
      padding_shape = 0
    else:
      raise ValueError(
        'padding should be in [VALID|SAME] but got {}'.format(padding))

    if padding == 'SAME':
      if not simplify_pad:
        # pad - conv
        self.pads = [nn.ConstantPad3d(x, 0) for x in padding_shapes]
        self.conv3d = nn.Conv3d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                bias=use_bias)
      else:
        self.conv3d = nn.Conv3d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=pad_size,
                                bias=use_bias)
    elif padding == 'VALID':
      self.conv3d = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size,
                              padding=padding_shape,
                              stride=stride,
                              bias=use_bias)

    if self.use_bn:
      # This is not strictly the correct map between epsilons in keras and
      # pytorch (which have slightly different definitions of the batch norm
      # forward pass), but it seems to be good enough. The PyTorch formula
      # is described here:
      # https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html
      tf_style_eps = 1E-3
      self.batch3d = nn.BatchNorm3d(out_channels, eps=tf_style_eps)

    if activation == 'relu':
      self.activation = nn.ReLU(inplace=True)

  def forward(self, inp):
    # pad -> conv3d -> bn -> relu
    if self.padding == 'SAME' and self.simplify_pad is False:
      # Determine the padding to be applied by examining the input shape
      pad_idx = inp.shape[2] % self.stride[0]
      pad_op = self.pads[pad_idx]
      inp = pad_op(inp)
    out = self.conv3d(inp)
    if self.use_bn:
      out = self.batch3d(out)
    if self.activation is not None:
      out = self.activation(out)
    return out


class MaxPool3dTFPadding(nn.Module):
  """
  3D max pooling with TF style padding
  """
  def __init__(self, kernel_size, stride=None, padding='SAME'):
    super(MaxPool3dTFPadding, self).__init__()
    if padding == 'SAME':
      padding_shape = get_padding_shape(kernel_size, stride)
      self.padding_shape = padding_shape
      self.stride = stride
      if stride[0] > 1:
        padding_shapes = [get_padding_shape(kernel_size, stride, mod) for
                          mod in range(stride[0])]
      else:
        padding_shapes = [padding_shape]
      self.pads = [nn.ConstantPad3d(x, 0) for x in padding_shapes]
    self.pool = nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

  def forward(self, inp):
    pad_idx = inp.shape[2] % self.stride[0]
    pad_op = self.pads[pad_idx]
    inp = pad_op(inp)
    out = self.pool(inp)
    return out

class Mixed(nn.Module):
  """
  Inception block
  """
  def __init__(self, in_channels, out_channels):
    super(Mixed, self).__init__()
    # Branch 0
    self.branch_0 = Unit3Dpy(
      in_channels, out_channels[0], kernel_size=(1, 1, 1))

    # Branch 1
    branch_1_conv1 = Unit3Dpy(
      in_channels, out_channels[1], kernel_size=(1, 1, 1))
    branch_1_conv2 = Unit3Dpy(
      out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
    self.branch_1 = nn.Sequential(branch_1_conv1, branch_1_conv2)

    # Branch 2
    branch_2_conv1 = Unit3Dpy(
      in_channels, out_channels[3], kernel_size=(1, 1, 1))
    branch_2_conv2 = Unit3Dpy(
      out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
    self.branch_2 = nn.Sequential(branch_2_conv1, branch_2_conv2)

    # Branch3
    branch_3_pool = MaxPool3dTFPadding(
      kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
    branch_3_conv2 = Unit3Dpy(
      in_channels, out_channels[5], kernel_size=(1, 1, 1))
    self.branch_3 = nn.Sequential(branch_3_pool, branch_3_conv2)

  def forward(self, inp):
    out_0 = self.branch_0(inp)
    out_1 = self.branch_1(inp)
    out_2 = self.branch_2(inp)
    out_3 = self.branch_3(inp)
    out = torch.cat((out_0, out_1, out_2, out_3), 1)
    return out

class I3DInception(nn.Module):
  """
  I3D inception network
  """
  def __init__(self,
               frozen_stages=-1,
               modality='rgb',
               pretrained=None):
    super(I3DInception, self).__init__()

    # modality
    if modality == 'rgb':
      in_channels = 3
    elif modality == 'flow':
      in_channels = 2
    else:
      raise ValueError(
        '{} not among known modalities [rgb|flow]'.format(modality))
    self.modality = modality
    self.frozen_stages = frozen_stages

    # 1st conv-pool
    self.conv3d_1a_7x7 = Unit3Dpy(out_channels=64,
                                  in_channels=in_channels,
                                  kernel_size=(7, 7, 7),
                                  stride=(2, 2, 2),
                                  padding='SAME')

    # 2nd conv
    self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(
      kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
    self.conv3d_2b_1x1 = Unit3Dpy(out_channels=64,
                                  in_channels=64,
                                  kernel_size=(1, 1, 1),
                                  padding='SAME')
    self.conv3d_2c_3x3 = Unit3Dpy(out_channels=192,
                                  in_channels=64,
                                  kernel_size=(3, 3, 3),
                                  padding='SAME')

    # Mixed 3
    self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(
      kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
    self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
    self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])

    # Mixed 4
    self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(
      kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')
    self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
    self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
    self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
    self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
    self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])

    # Mixed 5
    self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
      kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')
    self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
    self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

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
    # conv1
    x = self.conv3d_1a_7x7(x)
    # conv2
    h = self.maxPool3d_2a_3x3(x)
    h = self.conv3d_2b_1x1(h)
    x1 = self.conv3d_2c_3x3(h)
    # conv3
    h = self.maxPool3d_3a_3x3(x1)
    h = self.mixed_3b(h)
    x2 = self.mixed_3c(h)
    # conv4
    h = self.maxPool3d_4a_3x3(x2)
    h = self.mixed_4b(h)
    h = self.mixed_4c(h)
    h = self.mixed_4d(h)
    h = self.mixed_4e(h)
    x3 = self.mixed_4f(h)
    # conv5
    h = self.maxPool3d_5a_2x2(x3)
    h = self.mixed_5b(h)
    x4 = self.mixed_5c(h)

    # return a tuple of feature maps
    out = (x1, x2, x3, x4)
    return out

  def _freeze_stages(self):
    # train the full network with frozen_stages < 0
    if self.frozen_stages < 0:
      return
    # a bit ugly, stages from [0, 4] - the five conv blocks
    stage_mapping = [
      [self.conv3d_1a_7x7],
      [self.conv3d_2b_1x1, self.conv3d_2c_3x3],
      [self.mixed_3b, self.mixed_3c],
      [self.mixed_4b, self.mixed_4c, self.mixed_4d, self.mixed_4e, self.mixed_4f],
      [self.mixed_5b, self.mixed_5c]
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
    super(I3DInception, self).train(mode)
    self._freeze_stages()

if __name__=='__main__':
  net = I3DInception()
