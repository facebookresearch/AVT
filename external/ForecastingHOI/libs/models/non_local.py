# Non-local block from Xiaolong's paper
# Modified code from
# https://github.com/AlexHex7/Non-local_pytorch/blob/master/Non-Local_pytorch_0.3.1/lib/non_local_embedded_gaussian.py
import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
  def __init__(self,
               in_channels,
               inter_channels=None,
               dimension=3,
               sub_sample=True):
    super(_NonLocalBlockND, self).__init__()

    assert dimension in [1, 2, 3]

    self.dimension = dimension
    self.sub_sample = sub_sample

    self.in_channels = in_channels
    self.inter_channels = inter_channels

    if self.inter_channels is None:
      self.inter_channels = max(1, in_channels // 2)

    # conv1d/2d/3d
    if dimension == 3:
      conv_nd = nn.Conv3d
      max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2),
                                    stride=(1, 2, 2))
      bn = nn.BatchNorm3d
    elif dimension == 2:
      conv_nd = nn.Conv2d
      max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2),
                                    stride=(2, 2))
      bn = nn.BatchNorm2d
    else:
      conv_nd = nn.Conv1d
      max_pool_layer = nn.MaxPool1d(kernel_size=(2),
                                    stride=(2))
      bn = nn.BatchNorm1d

    # projection: C_i -> C
    self.g = conv_nd(in_channels=self.in_channels,
                     out_channels=self.inter_channels,
                     kernel_size=1,
                     stride=1,
                     padding=0)

    # projection: C -> C_i with BN (zero init)
    self.W = nn.Sequential(
      conv_nd(in_channels=self.inter_channels,
              out_channels=self.in_channels,
              kernel_size=1,
              stride=1,
              padding=0,
              bias=False),  # bias absorbed in bn
      bn(self.in_channels)
    )

    self.theta = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)
    self.phi = conv_nd(in_channels=self.in_channels,
                       out_channels=self.inter_channels,
                       kernel_size=1,
                       stride=1,
                       padding=0)

    if sub_sample:
      self.g = nn.Sequential(self.g, max_pool_layer)
      self.phi = nn.Sequential(self.phi, max_pool_layer)

    self.reset_params()

  def reset_params(self):
    # zero init bn
    nn.init.constant_(self.W[1].weight, 0)
    nn.init.constant_(self.W[1].bias, 0)

    # zero init bias in conv_nd
    nn.init.constant_(self.theta.bias, 0)
    if self.sub_sample:
      nn.init.constant_(self.g[0].bias, 0)
      nn.init.constant_(self.phi[0].bias, 0)
    else:
      nn.init.constant_(self.g.bias, 0)
      nn.init.constant_(self.phi.bias, 0)

  def forward(self, x):
    '''
    :param x: (b, c, t, h, w)
    :return: (b, c, t, h, w)
    '''
    batch_size = x.size(0)

    # (b, c, t, h, w) -> (b, c_1, t, h, w)
    g_x = self.g(x)
    # (b, c_1, t, h, w) -> (b, c_1, thw) -> (b, thw, c_1)
    g_x = g_x.view(batch_size, self.inter_channels, -1)
    g_x = g_x.permute(0, 2, 1)

    # save for theta -> (b, thw, c_1)
    theta_x = self.theta(x)
    theta_x = theta_x.view(batch_size, self.inter_channels, -1)
    theta_x = theta_x.permute(0, 2, 1)

    # and phi -> (b, c_1, thw)
    phi_x = self.phi(x)
    phi_x = phi_x.view(batch_size, self.inter_channels, -1)

    # (b, thw, c_1) * (b, c_1, thw) -> (b, thw, thw)
    f = torch.bmm(theta_x, phi_x)
    # (b, thw, thw)
    f_div_C = F.softmax(f, dim=-1)
    # (b, thw, thw) * (b, thw, c_1) -> (b, thw, c_1)
    y = torch.bmm(f_div_C, g_x)

    # (b, thw, c_1) -> (b, c1, thw) -> (b, c1, t, h, w)
    y = y.permute(0, 2, 1).contiguous()
    y = y.view(batch_size, self.inter_channels, *x.size()[2:])

    # Wy + x
    W_y = self.W(y)
    z = W_y + x
    return z

class NonLocalBlock1D(_NonLocalBlockND):
  def __init__(self, in_channels, inter_channels=None, sub_sample=True):
    super(NonLocalBlock1D, self).__init__(in_channels,
                                          inter_channels=inter_channels,
                                          dimension=1, sub_sample=sub_sample)


class NonLocalBlock2D(_NonLocalBlockND):
  def __init__(self, in_channels, inter_channels=None, sub_sample=True):
    super(NonLocalBlock2D, self).__init__(in_channels,
                                          inter_channels=inter_channels,
                                          dimension=2, sub_sample=sub_sample)


class NonLocalBlock3D(_NonLocalBlockND):
  def __init__(self, in_channels, inter_channels=None, sub_sample=True):
    super(NonLocalBlock3D, self).__init__(in_channels,
                                          inter_channels=inter_channels,
                                          dimension=3, sub_sample=sub_sample)


if __name__ == '__main__':
  """A quick test"""
  sub_sample = True
  input = torch.zeros(2, 3, 20)
  net = NonLocalBlock1D(3, sub_sample=sub_sample)
  out = net(input)
  print("1D input/output size")
  print(input.size(), out.size())

  input = torch.zeros(2, 3, 20, 20)
  net = NonLocalBlock2D(3, sub_sample=sub_sample)
  out = net(input)
  print("2D input/output size")
  print(input.size(), out.size())

  input = torch.randn(2, 3, 10, 20, 20)
  net = NonLocalBlock3D(3, sub_sample=sub_sample)
  out = net(input)
  print("3D input/output size")
  print(input.size(), out.size())
