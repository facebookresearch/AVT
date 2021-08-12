"""
Helper functions for our models
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_padding_shape(filter_shape, stride, mod=0):
  """Fetch a tuple describing the input padding shape.

  NOTES: To replicate "TF SAME" style padding, the padding shape needs to be
  determined at runtime to handle cases when the input dimension is not divisible
  by the stride.
  See https://stackoverflow.com/a/49842071 for explanation of TF padding logic
  """
  def _pad_top_bottom(filter_dim, stride_val, mod):
    if mod:
      pad_along = max(filter_dim - mod, 0)
    else:
      pad_along = max(filter_dim - stride_val, 0)
    pad_top = pad_along // 2
    pad_bottom = pad_along - pad_top
    return pad_top, pad_bottom

  padding_shape = []
  for idx, (filter_dim, stride_val) in enumerate(zip(filter_shape, stride)):
    depth_mod = (idx == 0) and mod
    pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val, depth_mod)
    padding_shape.append(pad_top)
    padding_shape.append(pad_bottom)

  depth_top = padding_shape.pop(0)
  depth_bottom = padding_shape.pop(0)
  padding_shape.append(depth_top)
  padding_shape.append(depth_bottom)
  return tuple(padding_shape)

def simplify_padding(padding_shapes):
  all_same = True
  padding_init = padding_shapes[0]
  for pad in padding_shapes[1:]:
    if pad != padding_init:
      all_same = False
  return all_same, padding_init
