"""
Decoders
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from .gcu import GraphUnit3D


class AvgLinear(nn.Module):
  """
  Avg (w. dropout) + linear classifier
  """
  def __init__(self, in_channels, num_classes,
               dropout_prob=0.0, fc_std=0.01):
    super(AvgLinear, self).__init__()
    # set up params
    self.in_channels = in_channels
    self.num_classes = num_classes
    self.dropout_prob = dropout_prob
    self.fc_std = fc_std

    # pooling, dropout and fc
    self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
    self.dropout = nn.Dropout(self.dropout_prob)
    self.fc = nn.Linear(self.in_channels, self.num_classes, bias=True)

    self.reset_params()

  def reset_params(self):
    # manuall init fc params
    nn.init.normal_(self.fc.weight, 0.0, self.fc_std)
    nn.init.constant_(self.fc.bias, 0.0)

  def forward(self, x):
    # only take the last conv block
    x_in = x[-1]
    out = self.avgpool(x_in)
    out = out.view(out.shape[0], -1)
    out = self.dropout(out)
    out = self.fc(out)

    return (out, )

class AvgLinearJoint(nn.Module):
  """
  Avg (w. dropout) + linear classifier
  """
  def __init__(self, in_channels, num_classes,
               dropout_prob=0.0, temperature = 2.0, fc_std=0.01):
    super(AvgLinearJoint, self).__init__()
    # set up params
    self.in_channels = in_channels
    self.num_classes = num_classes
    self.dropout_prob = dropout_prob
    self.fc_std = fc_std
    self.conv_motor = nn.Conv3d(512, 1,
                                kernel_size=(1, 3, 3),
                                stride=(1,1,1),
                                padding=(0, 1, 1),
                                bias=False)
    self.conv_hotspot = nn.Conv3d(1024, 1,
                                kernel_size=(3, 3, 3),
                                stride=(2,1,1),
                                padding=(0, 1, 1),
                                bias=False)
    self.temperature = temperature
    # pooling, dropout and fc

    self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
    self.dropout = nn.Dropout(self.dropout_prob)
    self.fc = nn.Linear(self.in_channels, self.num_classes, bias=True)

    self.reset_params()

  def reset_params(self):
    # manuall init fc params
    nn.init.normal_(self.fc.weight, 0.0, self.fc_std)
    nn.init.constant_(self.fc.bias, 0.0)

  def softmax_3d(self, logits):
    batch_size, T = logits.shape[0], logits.shape[2]
    H, W = logits.shape[3], logits.shape[4]

    # reshape -> softmax (dim=-1) -> reshape back
    logits = logits.view(batch_size, -1, T, H*W)
    atten_map = nn.functional.softmax(logits / self.temperature, dim=-1)
    atten_map = atten_map.view(batch_size, -1, T, H, W)
    return atten_map

  def forward(self, x):
    motor_logits  = self.conv_motor(x[1])
    motor_atten_map = self.softmax_3d(motor_logits)

    hotspot_logits  = self.conv_hotspot(x[2])
    hotspot_atten_map = self.softmax_3d(hotspot_logits)

    x_in = x[-1]
    out = self.avgpool(x_in)
    out = out.view(out.shape[0], -1)
    out = self.dropout(out)
    out = self.fc(out)

    return (out, motor_atten_map, hotspot_atten_map, )



class JointSoftAtten(nn.Module):
  """
  Avg (w. dropout) + linear classifier
  """
  def __init__(self, in_channels, num_classes,stride,
               dropout_prob=0.0, temperature=2.0, fc_std=0.01):
    super(JointSoftAtten, self).__init__()
    # set up params
    self.in_channels = in_channels
    self.num_classes = num_classes
    self.dropout_prob = dropout_prob
    self.fc_std = fc_std
    self.stride1 = stride
    self.temperature = temperature
    self.stride2 = (stride[0],stride[1]*2,stride[2]*2)
    print(self.stride1)
    print(self.stride2)
    self.conv_motor = nn.Conv3d(512, 1,
                                kernel_size=(1, 3, 3),
                                stride=(1,1,1),
                                padding=(0, 1, 1),
                                bias=False)

    self.conv_hotspot = nn.Conv3d(1024, 1,
                                kernel_size=(3, 3, 3),
                                stride=(2,1,1),
                                padding=(0, 1, 1),
                                bias=False)


    self.maxpool1 = nn.AvgPool3d(kernel_size=self.stride1,
                                 stride=self.stride1,
                                 padding=(0, 0, 0))
    self.maxpool2 = nn.AvgPool3d(kernel_size=self.stride2,
                                 stride=self.stride2,
                                 padding=(0, 0, 0))
    # pooling, dropout and fc
    self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
    self.dropout = nn.Dropout(self.dropout_prob)
    self.fc = nn.Linear(self.in_channels, self.num_classes, bias=True)

    self.reset_params()

  def reset_params(self):
    # manuall init fc params
    nn.init.normal_(self.fc.weight, 0.0, self.fc_std)
    nn.init.constant_(self.fc.bias, 0.0)

  def softmax_3d(self, logits):
    batch_size, T = logits.shape[0], logits.shape[2]
    H, W = logits.shape[3], logits.shape[4]
    # reshape -> softmax (dim=-1) -> reshape back
    logits = logits.view(batch_size, -1, T, H*W)
    atten_map = nn.functional.softmax(logits / self.temperature, dim=-1)
    atten_map = atten_map.view(batch_size, -1, T, H, W)
    return atten_map

  def forward(self, x):
    motor_logits  = self.conv_motor(x[1])
    pooled_motor_logits_1= self.maxpool1(motor_logits)
    pooled_motor_logits_2 = self.maxpool2(motor_logits)

    motor_atten_map = self.softmax_3d(motor_logits)
    pooled_motor_atten_map_1 = self.softmax_3d(pooled_motor_logits_1)
    pooled_motor_atten_map_2 = self.softmax_3d(pooled_motor_logits_2)

    H, W = pooled_motor_atten_map_1.shape[3], pooled_motor_atten_map_1.shape[4]
    hotspot_x_in = x[2]*pooled_motor_atten_map_1*H*W
    hotspot_logits  = self.conv_hotspot(hotspot_x_in)
    hotspot_atten_map = self.softmax_3d(hotspot_logits)


    classifier_x_in = x[-1]*pooled_motor_atten_map_2

    batch_size, T = classifier_x_in.shape[0], classifier_x_in.shape[2]
    # N C T H W -> N C T HW -> (sum) N C T
    out = classifier_x_in.view(batch_size, self.in_channels, T, -1).sum(dim=-1)
    out = out.mean(dim=-1)
    # dropout + fc
    out = self.dropout(out)
    out = self.fc(out)


    return (out, motor_atten_map, hotspot_atten_map, )

# class JointProbAtten(nn.Module):
#   """
#   Avg (w. dropout) + linear classifier
#   """
#   def __init__(self, in_channels, num_classes,stride,
#                dropout_prob=0.0, temperature=2.0, fc_std=0.01, eps=1e-6):
#     super(JointProbAtten, self).__init__()
#     # set up params
#     self.in_channels = in_channels
#     self.num_classes = num_classes
#     self.dropout_prob = dropout_prob
#     self.fc_std = fc_std
#     self.stride1 = stride
#     self.stride2 = (stride[0],stride[1]*2,stride[2]*2)
#     if stride[0] == 2:
#       self.stride2 = (stride[0]*2,stride[1]*2,stride[2]*2)
#     self.temperature = temperature
#     self.eps = eps
#     print(self.stride1)
#     print(self.stride2)
#     print(self.temperature)
#     self.conv_motor = nn.Conv3d(512, 1,
#                                 kernel_size=(1, 3, 3),
#                                 stride=(1,1,1),
#                                 padding=(0, 1, 1),
#                                 bias=False)

#     self.conv_hotspot = nn.Conv3d(1024, 1,
#                                   kernel_size=(3, 3, 3),
#                                   stride=(2,1,1),
#                                   padding=(0, 1, 1),
#                                   bias=False)


#     self.maxpool1 = nn.MaxPool3d(kernel_size=self.stride1,
#                                  stride=self.stride1,
#                                  padding=(0, 0, 0))
#     self.maxpool2 = nn.MaxPool3d(kernel_size=self.stride2,
#                                  stride=self.stride2,
#                                  padding=(0, 0, 0))
#     # pooling, dropout and fc
#     self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
#     self.dropout = nn.Dropout(self.dropout_prob)

#     # addition pooling for CSN152 Model
#     self.add_pool1 = nn.AvgPool3d(kernel_size=(4,1,1),stride=(4,1,1))
#     self.add_pool2 = nn.AvgPool3d(kernel_size=(3,1,1),stride=(3,1,1))
#     self.fc = nn.Linear(self.in_channels, self.num_classes, bias=True)

#     self.reset_params()

#   def reset_params(self):
#     # manuall init fc params
#     nn.init.normal_(self.fc.weight, 0.0, self.fc_std)
#     nn.init.constant_(self.fc.bias, 0.0)

#   def gumbel_softmax_3d(self, logits, x):
#     batch_size, T = logits.shape[0], logits.shape[2]
#     H, W = logits.shape[3], logits.shape[4]

#     if self.training:
#       # gumbel softmax sampling (different mask for each feature channel)
#       U = torch.zeros_like(x).uniform_()
#       gumbel_noise = torch.log(-torch.log(U + self.eps) + self.eps)
#       logits = logits - gumbel_noise.detach_()

#     # reshape -> softmax (dim=-1) -> reshape back
#     logits = logits.view(batch_size, -1, T, H*W)
#     atten_map = nn.functional.softmax(logits / self.temperature, dim=-1)
#     atten_map = atten_map.view(batch_size, -1, T, H, W)
#     return atten_map
#   def softmax_3d(self, logits):
#     batch_size, T = logits.shape[0], logits.shape[2]
#     H, W = logits.shape[3], logits.shape[4]
#     # reshape -> softmax (dim=-1) -> reshape back
#     logits = logits.view(batch_size, -1, T, H*W)
#     atten_map = nn.functional.softmax(logits / self.temperature, dim=-1)
#     atten_map = atten_map.view(batch_size, -1, T, H, W)
#     return atten_map
#   def forward(self, x):
#     motor_logits  = self.conv_motor(x[1])
#     pooled_motor_logits_1= self.maxpool1(motor_logits)
#     pooled_motor_logits_2 = self.maxpool2(motor_logits)
#     sample_motor_atten_map_1 = self.gumbel_softmax_3d(pooled_motor_logits_1, x[2])
#     sample_motor_atten_map_2 = self.gumbel_softmax_3d(pooled_motor_logits_2, x[-1])

#     # downsample the motor attention for CSN152 model so that it could match with gorund truth dimension
#     if self.stride1[0] == 2:
#       print(motor_logits.size())
#       motor_logits = self.add_pool1(motor_logits)
#     motor_atten_map = self.softmax_3d(motor_logits)

#     scale_factor = sample_motor_atten_map_1.shape[3]*sample_motor_atten_map_1.shape[4]/self.temperature
#     hotspot_x_in = x[2]*sample_motor_atten_map_1*scale_factor
#     hotspot_logits  = self.conv_hotspot(hotspot_x_in)
#     # downsample the hotspot attention for CSN152 model so that it could match with gorund truth dimension
#     if self.stride1[0] == 2:
#       print(hotspot_logits.size())
#       hotspot_logits = self.add_pool2(hotspot_logits)
#     hotspot_atten_map = self.softmax_3d(hotspot_logits)

#     classifier_x_in = x[-1]*sample_motor_atten_map_2
#     # N C T H W -> N C T HW -> (sum) N C T
#     out = classifier_x_in.view(classifier_x_in.shape[0], self.in_channels, classifier_x_in.shape[2], -1).sum(dim=-1)
#     # N C T (mean) -> N C
#     out = out.mean(dim=-1)
#     # dropout + fc
#     out = self.dropout(out)
#     out = self.fc(out)

#     return (out, motor_atten_map, hotspot_atten_map, )

class JointProbAtten(nn.Module):
  """
  Avg (w. dropout) + linear classifier
  """
  def __init__(self, in_channels, num_classes,stride,
               dropout_prob=0.0, temperature=2.0, fc_std=0.01, eps=1e-6):
    super(JointProbAtten, self).__init__()
    # set up params
    self.in_channels = in_channels
    self.num_classes = num_classes
    self.dropout_prob = dropout_prob
    self.fc_std = fc_std
    self.stride1 = stride
    self.stride2 = (stride[0],stride[1]*2,stride[2]*2)
    if stride[0] == 2:
      self.stride2 = (stride[0]*2,stride[1]*2,stride[2]*2)
    self.temperature = temperature
    self.eps = eps
    print(self.stride1)
    print(self.stride2)
    print(self.temperature)
    self.conv_motor = nn.Conv3d(512, 1,
                                kernel_size=(1, 3, 3),
                                stride=(1,1,1),
                                padding=(0, 1, 1),
                                bias=False)

    self.conv_hotspot = nn.Conv3d(1024, 1,
                                  kernel_size=(1, 3, 3),
                                  stride=(1,1,1),
                                  padding=(0, 1, 1),
                                  bias=False)


    self.avgpool_motor_1 = nn.AvgPool3d(kernel_size=self.stride1,
                                 stride=self.stride1,
                                 padding=(0, 0, 0))
    self.avgpool_motor_2 = nn.AvgPool3d(kernel_size=self.stride2,
                                 stride=self.stride2,
                                 padding=(0, 0, 0))
    self.avgpool_hotspot_1 = nn.AvgPool3d(kernel_size=(4,1,1),
                                 stride=(4,1,1),
                                 padding=(0, 0, 0))
    self.avgpool_hotspot_2 = nn.AvgPool3d(kernel_size=(1,2,2),
                                 stride=(1,2,2),
                                 padding=(0, 0, 0))
    # pooling, dropout and fc
    self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
    self.dropout = nn.Dropout(self.dropout_prob)

    # addition pooling for CSN152 Model
    self.add_pool1 = nn.AvgPool3d(kernel_size=(4,1,1),stride=(4,1,1))
    self.add_pool2 = nn.AvgPool3d(kernel_size=(2,1,1),stride=(2,1,1))
    self.fc = nn.Linear(self.in_channels, self.num_classes, bias=True)

    self.reset_params()

  def reset_params(self):
    # manuall init fc params
    nn.init.normal_(self.fc.weight, 0.0, self.fc_std)
    nn.init.constant_(self.fc.bias, 0.0)

  def gumbel_softmax_3d(self, logits, x):
    batch_size, T = logits.shape[0], logits.shape[2]
    H, W = logits.shape[3], logits.shape[4]

    if self.training:
      # gumbel softmax sampling (different mask for each feature channel)
      U = torch.zeros_like(x).uniform_()
      gumbel_noise = torch.log(-torch.log(U + self.eps) + self.eps)
      logits = logits - gumbel_noise.detach_()

    # reshape -> softmax (dim=-1) -> reshape back
    logits = logits.view(batch_size, -1, T, H*W)
    atten_map = nn.functional.softmax(logits / self.temperature, dim=-1)
    atten_map = atten_map.view(batch_size, -1, T, H, W)
    return atten_map
  def softmax_3d(self, logits):
    batch_size, T = logits.shape[0], logits.shape[2]
    H, W = logits.shape[3], logits.shape[4]
    # reshape -> softmax (dim=-1) -> reshape back
    logits = logits.view(batch_size, -1, T, H*W)
    atten_map = nn.functional.softmax(logits / self.temperature, dim=-1)
    atten_map = atten_map.view(batch_size, -1, T, H, W)
    return atten_map
  def forward(self, x):
    motor_logits  = self.conv_motor(x[1])
    pooled_motor_logits_1= self.avgpool_motor_1(motor_logits)
    pooled_motor_logits_2 = self.avgpool_motor_2(motor_logits)
    sample_motor_atten_map_1 = self.gumbel_softmax_3d(pooled_motor_logits_1, x[2])
    sample_motor_atten_map_2 = self.gumbel_softmax_3d(pooled_motor_logits_2, x[-1])

    # downsample the motor attention for CSN152 model so that it could match with gorund truth dimension
    if self.stride1[0] == 2:
      # print(motor_logits.size())
      motor_logits = self.add_pool1(motor_logits)
    motor_atten_map = self.softmax_3d(motor_logits)

    scale_factor = sample_motor_atten_map_1.shape[3]*sample_motor_atten_map_1.shape[4]
    hotspot_x_in = x[2]*sample_motor_atten_map_1*scale_factor
    hotspot_logits  = self.conv_hotspot(hotspot_x_in)
    # downsample the hotspot attention for CSN152 model so that it could match with gorund truth dimension
    if self.stride1[0] == 2:
      # print(hotspot_logits.size())
      hotspot_logits = self.add_pool2(hotspot_logits)

    pooled_hotspot_logits_1 = self.avgpool_hotspot_1(hotspot_logits)
    pooled_hotspot_logits_2 = self.avgpool_hotspot_2(hotspot_logits)
    hotspot_atten_map = self.softmax_3d(pooled_hotspot_logits_1)
    sample_hotspot_atten_map = self.gumbel_softmax_3d(pooled_hotspot_logits_2, x[-1])

    scale_factor = sample_hotspot_atten_map.shape[3]*sample_hotspot_atten_map.shape[4]
    # classifier_x_in = x[-1]*sample_motor_atten_map_2
    classifier_x_in = x[-1]*(sample_motor_atten_map_2+sample_hotspot_atten_map)/2
    # N C T H W -> N C T HW -> (sum) N C T
    out = classifier_x_in.view(classifier_x_in.shape[0], self.in_channels, classifier_x_in.shape[2], -1).sum(dim=-1)
    # N C T (mean) -> N C
    out = out.mean(dim=-1)
    # dropout + fc
    out = self.dropout(out)
    out = self.fc(out)

    return (out, motor_atten_map, hotspot_atten_map, )

class MaxLinear(nn.Module):
  """
  Avg (w. dropout) + linear classifier
  """
  def __init__(self, in_channels, num_classes,
               dropout_prob=0.0, fc_std=0.01):
    super(MaxLinear, self).__init__()
    # set up params
    self.in_channels = in_channels
    self.num_classes = num_classes
    self.dropout_prob = dropout_prob
    self.fc_std = fc_std

    # pooling, dropout and fc
    self.maxpool = nn.AdaptiveMaxPool3d((1, 1, 1))
    self.dropout = nn.Dropout(self.dropout_prob)
    self.fc = nn.Linear(self.in_channels, self.num_classes, bias=True)

    self.reset_params()

  def reset_params(self):
    # manuall init fc params
    nn.init.normal_(self.fc.weight, 0.0, self.fc_std)
    nn.init.constant_(self.fc.bias, 0)

  def forward(self, x):
    # only take the last conv block
    x_in = x[-1]
    out = self.maxpool(x_in)
    out = out.view(out.shape[0], -1)
    out = self.dropout(out)
    out = self.fc(out)
    return (out, )

class GCULinear(nn.Module):
  """
  Avg (w. dropout) + linear classifier
  """
  def __init__(self, in_channels, num_nodes, num_classes,
               dropout_prob=0.0, fc_std=0.01):
    super(GCULinear, self).__init__()
    # set up params
    self.in_channels = in_channels
    self.num_nodes = num_nodes
    self.num_classes = num_classes
    self.dropout_prob = dropout_prob
    self.fc_std = fc_std

    # pooling, dropout and fc
    self.gcu = GraphUnit3D(in_channels, num_nodes)
    # note: DO NOT use inplace droput
    # (will trigger a weird bug in distributed training)
    self.dropout = nn.Dropout(self.dropout_prob)
    self.fc = nn.Linear(self.in_channels, self.num_classes, bias=True)

    self.reset_params()

  def reset_params(self):
    # manuall init fc params
    nn.init.normal_(self.fc.weight, 0.0, self.fc_std)
    nn.init.constant_(self.fc.bias, 0)

  def forward(self, x):
    # only take the last conv block

    x_in = x[-1]
    out, map = self.gcu(x_in)
    out = self.dropout(out)
    out = self.fc(out)
    return (out, )

class AttenLinear(nn.Module):
  """
  Re-implementation of our previous paper
    "In the Eye of Beholder:
     Joint Learning of Gaze and Actions in First Person Vision"
  """
  def __init__(self, in_channels, num_classes, stride=(1, 2, 2),
               dropout_prob=0.0, fc_std=0.01, temperature=1.0, eps=1e-6):
    super(AttenLinear, self).__init__()
    self.in_channels = in_channels
    self.num_classes = num_classes
    self.dropout_prob = dropout_prob
    self.fc_std = fc_std

    # attention params
    self.eps = eps
    self.stride = stride
    self.temperature = temperature

    # attention branch (1x1 conv without bias/bn)
    self.conv_atten = nn.Conv3d(self.in_channels // 2, 1,
                                kernel_size=(1, 3, 3),
                                stride=self.stride,
                                padding=(0, 1, 1),
                                bias=False)

    # classification branch
    self.dropout = nn.Dropout(self.dropout_prob)
    self.fc = nn.Linear(self.in_channels, self.num_classes, bias=True)

    self.reset_params()

  def reset_params(self):
    # manuall init atten / fc params
    nn.init.normal_(self.conv_atten.weight, 0.0, self.fc_std)
    nn.init.normal_(self.fc.weight, 0.0, self.fc_std)
    nn.init.constant_(self.fc.bias, 0)

  def gumbel_softmax_3d(self, logits, x):
    batch_size, T = logits.shape[0], logits.shape[2]
    H, W = logits.shape[3], logits.shape[4]
    if self.training:
      # gumbel softmax sampling (different mask for each feature channel)
      U = torch.zeros_like(x).uniform_()
      gumbel_noise = torch.log(-torch.log(U + self.eps) + self.eps)
      logits = logits - gumbel_noise.detach_()

    # reshape -> softmax (dim=-1) -> reshape back
    logits = logits.view(batch_size, -1, T, H*W)
    atten_map = nn.functional.softmax(logits / self.temperature, dim=-1)
    atten_map = atten_map.view(batch_size, -1, T, H, W)
    return atten_map

  def forward(self, x):
    x_in, atten_in = x[-1], x[-2]
    # print(x_in.size())
    # print(atten_in.size())
    batch_size, T = x_in.shape[0], x_in.shape[2]
    # generate attention map
    atten_logits = self.conv_atten(atten_in)
    # print(atten_logits.size())
    atten_map = self.gumbel_softmax_3d(atten_logits, x_in)
    # print(atten_map.size())
    # mutliply attention map with the feature map
    atten_x = atten_map * x_in
    # print(atten_x.size())
    # N C T H W -> N C T HW -> (sum) N C T
    out = atten_x.view(atten_x.shape[0], self.in_channels, T, -1).sum(dim=-1)
    # print(out.size())
    # N C T (mean) -> N C
    out = out.mean(dim=-1)
    # dropout + fc
    out = self.dropout(out)
    out = self.fc(out)
    # also output attention distribution
    if self.training:
      # reshape -> softmax (dim=-1) -> reshape back
      atten_logits = atten_logits.view(batch_size, 1, T, -1)
      out_atten_map = nn.functional.softmax(
        atten_logits / self.temperature, dim=-1)
      out_atten_map = out_atten_map.view(
        batch_size, 1, T, x_in.shape[3], x_in.shape[4])
    else:
      # for testing, it is the same
      out_atten_map = atten_map
    return (out, out_atten_map)


# test
if __name__=='__main__':
  net = AvgLinear(1024, 200)
  net = MaxLinear(1024, 200)
  net = GCULinear(1024, 4, 200)
  net = AttenLinear(1024, 200)
