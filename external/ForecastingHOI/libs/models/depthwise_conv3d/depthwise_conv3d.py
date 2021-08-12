from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from . import depthwise_conv3d_cuda

class DepthwiseConv3dFunction(Function):
  """
    Functional interface for depthwise conv 3d (calls to cuda kernels)
  """
  @staticmethod
  def forward(ctx, input, weight,
              bias=None, stride=(1, 1, 1), padding=(0, 0, 0)):
    assert (input.dim() == 5) and (weight.dim() == 5)
    # only gpu kernels are implemented
    if not input.is_cuda:
      raise NotImplementedError
    ctx.stride = stride
    ctx.padding = padding

    # compute output / input dims
    n, c, t, h, w = input.shape
    _, _, kt, kh, kw = weight.shape
    st, sh, sw = stride
    pt, ph, pw = padding

    out_t = (t - kt + 2 * pt) // st + 1
    out_h = (h - kh + 2 * ph) // sh + 1
    out_w = (w - kw + 2 * pw) // sw + 1

    # prep for output
    output = input.new_empty([n, c, out_t, out_h, out_w])
    # call the cuda function
    depthwise_conv3d_cuda.depthwise_conv3d_forward_cuda(
      input, weight, output, st, sh, sw, pt, ph, pw)

    # adding bias
    if bias is not None:
      output += bias.view([1, c, 1, 1, 1])

    # save tensor for backward
    ctx.save_for_backward(input, weight, bias)

    return output

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_output):
    # only gpu kernels are implemented
    if not grad_output.is_cuda:
      raise NotImplementedError
    grad_input = grad_weight = grad_bias = None

    # recover from ctx
    input, weight, bias = ctx.saved_tensors
    st, sh, sw = ctx.stride
    pt, ph, pw = ctx.padding

    if ctx.needs_input_grad[0]:
      # grad w.r.t to input
      grad_input = torch.zeros_like(input)
      depthwise_conv3d_cuda.depthwise_conv3d_backward_input_cuda(
        grad_output, weight, grad_input, st, sh, sw, pt, ph, pw)
    if ctx.needs_input_grad[1]:
      # grad w.r.t to weight
      grad_weight = torch.zeros_like(weight)
      depthwise_conv3d_cuda.depthwise_conv3d_backward_weight_cuda(
        grad_output, input, grad_weight, st, sh, sw, pt, ph, pw)
    if bias is not None and ctx.needs_input_grad[2]:
      # grad w.r.t. bias (if any)
      grad_bias = grad_output.sum((0,2,3,4))

    return (grad_input, grad_weight, grad_bias, None, None)

depthwise_conv3d = DepthwiseConv3dFunction.apply

class DepthwiseConv3d(nn.Module):
  """
    A wrapper for depthwise conv 3d. We only support the following condition
    * in_channels = out_channels
    * no dilation
  """
  def __init__(self, in_channels, kernel_size=3, stride=1, padding=0, bias=False):
    super(DepthwiseConv3d, self).__init__()
    self.in_channels = in_channels
    if isinstance(stride, int):
      self.stride = (stride, stride, stride)
    else:
      self.stride = stride
    if isinstance(kernel_size, int):
      self.kernel_size = (kernel_size, kernel_size, kernel_size)
    else:
      self.kernel_size = kernel_size
    if isinstance(padding, int):
      self.padding = (padding, padding, padding)
    else:
      self.padding = padding

    self.weight = nn.Parameter(
      torch.Tensor(in_channels, 1, kernel_size[0], kernel_size[1], kernel_size[2]),
      requires_grad=True)
    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_channels))
    else:
      self.register_parameter('bias', None)

    self.reset_params()

  def reset_params(self):
    nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
    if self.bias is not None:
      nn.init.constant_(self.bias, 0)

  def forward(self, x):
    out = depthwise_conv3d(x, self.weight, self.bias, self.stride, self.padding)
    return out

  def extra_repr(self):
    s = ('{in_channels}, {in_channels}, kernel_size={kernel_size}'
         ', stride={stride}, padding={padding}')
    if self.bias is None:
      s += ', bias=False'
    return s.format(**self.__dict__)
