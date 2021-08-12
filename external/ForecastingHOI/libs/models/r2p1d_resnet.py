"""
Modified code from r2p1d repo https://github.com/pytorch/vision
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import torch.utils.model_zoo as model_zoo

__all__ = ['r2plus1d_18']

model_urls = {
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}


class Conv2Plus1D(nn.Sequential):
  """
  2 + 1 D Convolution
  """
  def __init__(self,
               in_planes,
               out_planes,
               midplanes,
               stride=1,
               padding=1):
    super(Conv2Plus1D, self).__init__(
      nn.Conv3d(in_planes, midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False),
      nn.BatchNorm3d(midplanes),
      nn.ReLU(inplace=True),
      nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                stride=(stride, 1, 1), padding=(padding, 0, 0),
                bias=False))

  @staticmethod
  def get_downsample_stride(stride):
      return (stride, stride, stride)

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, conv_builder,
               stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    midplanes = (inplanes * planes * 3 * 3 * 3) \
                // (inplanes * 3 * 3 + 3 * planes)
    self.conv1 = nn.Sequential(
      conv_builder(inplanes, planes, midplanes, stride),
      nn.BatchNorm3d(planes),
      nn.ReLU(inplace=True)
    )
    self.conv2 = nn.Sequential(
      conv_builder(planes, planes, midplanes),
      nn.BatchNorm3d(planes)
    )
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.conv2(out)
    if self.downsample is not None:
      residual = self.downsample(x)
    out += residual
    out = self.relu(out)
    return out


class Bottleneck(nn.Module):
  expansion = 4
  def __init__(self, inplanes, planes, conv_builder,
               stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    midplanes = (inplanes * planes * 3 * 3 * 3) \
                // (inplanes * 3 * 3 + 3 * planes)

    # 1x1x1
    self.conv1 = nn.Sequential(
      nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
      nn.BatchNorm3d(planes),
      nn.ReLU(inplace=True)
    )
    # Second kernel
    self.conv2 = nn.Sequential(
      conv_builder(planes, planes, midplanes, stride),
      nn.BatchNorm3d(planes),
      nn.ReLU(inplace=True)
    )

    # 1x1x1
    self.conv3 = nn.Sequential(
      nn.Conv3d(planes, planes * self.expansion,
                kernel_size=1, bias=False),
      nn.BatchNorm3d(planes * self.expansion)
    )
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class BasicStem(nn.Sequential):
  """The default conv-batchnorm-relu stem
  """
  def __init__(self):
    super(BasicStem, self).__init__(
      nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                padding=(1, 3, 3), bias=False),
      nn.BatchNorm3d(64),
      nn.ReLU(inplace=True))


class R2Plus1dStem(nn.Sequential):
  """R(2+1)D stem is different than the default one
     as it uses separated 3D convolution
  """
  def __init__(self):
    super(R2Plus1dStem, self).__init__(
      nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                stride=(1, 2, 2), padding=(0, 3, 3),
                bias=False),
      nn.BatchNorm3d(45),
      nn.ReLU(inplace=True),
      nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                stride=(1, 1, 1), padding=(1, 0, 0),
                bias=False),
      nn.BatchNorm3d(64),
      nn.ReLU(inplace=True))


class VideoResNet(nn.Module):
  def __init__(self, block, conv_makers, layers,
               stem, frozen_stages=-1,
               zero_init_residual=False):
    """Generic resnet video generator.

    Args:
        block (nn.Module): resnet building block
        conv_makers (list(functions)): generator function for each layer
        layers (List[int]): number of blocks per layer
        stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
        num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
        zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
    """
    super(VideoResNet, self).__init__()
    self.inplanes = 64
    self.frozen_stages = frozen_stages
    self.stem = stem()

    self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
    self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

    # init weights
    self._init_weights()

    # zero init
    if zero_init_residual:
      for m in self.modules():
        if isinstance(m, Bottleneck):
          nn.init.constant_(m.bn3.weight, 0)

    # freeze part of the model
    self._freeze_stages()

  def forward(self, x):
    x = self.stem(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    return x

  def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      ds_stride = conv_builder.get_downsample_stride(stride)
      downsample = nn.Sequential(
          nn.Conv3d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=ds_stride, bias=False),
          nn.BatchNorm3d(planes * block.expansion))

    layers = []
    layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, conv_builder))

    return nn.Sequential(*layers)

  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _freeze_stages(self):
    # train the full network with frozen_stages < 0
    if self.frozen_stages < 0:
      return
    # a bit ugly, stages from [0, 4] - the five conv blocks
    stage_mapping = [
      [self.stem],
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

  def train(self, mode=True):
    super(VideoResNet, self).train(mode)
    self._freeze_stages()

def load_pretrained_model(model, model_id):
  "Make the loading verbose"
  # load the model from model zoo and print the missing/unexpected keys
  missing_keys, unexpected_keys = model.load_state_dict(
    model_zoo.load_url(model_urls[model_id]), strict=False)
  if missing_keys:
    print("Missing keys:")
    for key in missing_keys:
      print(key)
  if unexpected_keys:
    print("Unexpected keys:")
    for key in unexpected_keys:
      print(key)

def _video_resnet(arch, pretrained=False, **kwargs):
  model = VideoResNet(**kwargs)
  if pretrained:
    load_pretrained_model(model, arch)
  return model


def r2plus1d_18(pretrained=False, modality='rgb', **kwargs):
  """Constructor for the 18 layer deep R(2+1)D network as in
  https://arxiv.org/abs/1711.11248

  Args:
      pretrained (bool): If True, returns a model pre-trained on Kinetics-400

  Returns:
      nn.Module: R(2+1)D-18 network
  """
  return _video_resnet('r2plus1d_18',
                       pretrained,
                       block=BasicBlock,
                       conv_makers=[Conv2Plus1D] * 4,
                       layers=[2, 2, 2, 2],
                       stem=R2Plus1dStem, **kwargs)

def r2plus1d_34(pretrained=False, modality='rgb', **kwargs):
  """Constructor for the 34 layer deep R(2+1)D network as in
  https://arxiv.org/abs/1711.11248

  Args:
      pretrained (bool): If True, returns a model pre-trained on Kinetics-400

  Returns:
      nn.Module: R(2+1)D-18 network
  """
  return _video_resnet('r2plus1d_34',
                       pretrained,
                       block=BasicBlock,
                       conv_makers=[Conv2Plus1D] * 4,
                       layers=[3, 4, 6, 3],
                       stem=R2Plus1dStem, **kwargs)
