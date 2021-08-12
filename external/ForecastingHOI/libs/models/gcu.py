import torch
import torch.nn as nn
from torch.nn import functional as F

class GCN(nn.Module):
  """ Graph convolution unit (single layer)
  modified from: https://github.com/facebookresearch/GloRe
  """
  def __init__(self, num_state, num_node, bias=True):
    super(GCN, self).__init__()
    self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, bias=bias)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

  def forward(self, x):
    # (n, num_state, num_node) -> (n, num_node, num_state)
    #                          -> (n, num_state, num_node)
    h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
    h = h + x
    # (n, num_state, num_node) -> (n, num_state, num_node)
    h = self.conv2(self.relu(h))
    return h

class GraphUnit(nn.Module):
  """
  Graph-based Unit
  """
  def __init__(self, in_channels, num_nodes,
               inter_channels=None,
               ConvNd=nn.Conv3d):
    super(GraphUnit, self).__init__()

    if inter_channels == None:
      # our read out function will double num_s
      self.num_s = int(in_channels // 2)
    self.num_n = int(num_nodes)

    # reduce dim -> map features into new dims
    self.conv_state = ConvNd(in_channels, self.num_s, kernel_size=1)
    # projection map -> number of nodes (disable bias?)
    self.conv_proj = ConvNd(in_channels, self.num_n, kernel_size=1, bias=False)
    # graph convolutional neural network
    self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
    # read out
    self.avgpool = nn.AdaptiveAvgPool1d(1)
    self.maxpool = nn.AdaptiveMaxPool1d(1)

    self.reset_params()

  def reset_params(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0.0)

  def forward(self, x):
    '''
    :param x: (n, c, d, h, w)
    '''
    n = x.shape[0]
    # (n, num_in, h, w) --> (n, num_state, h*w)
    x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)

    # (n, num_in, h, w) --> (n, num_node, h*w)
    x_proj_reshaped = F.softmax(self.conv_proj(x).view(n, self.num_n, -1), 1)

    # graph projection
    # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
    gcu_feat = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
    # normalize the graph features (add eps to prevent overflow)
    gcu_feat = gcu_feat / (x_proj_reshaped.sum(dim=-1).unsqueeze(1) + 1e-6)

    # gcn: (n, num_state, num_node) -> (n, num_state, num_node)
    gcn_out = self.gcn(gcu_feat)

    # readout (n, num_state, num_node) -> (n, 2*num_state)
    out = torch.cat((self.avgpool(gcn_out), self.maxpool(gcn_out)), dim=1)
    out = out.view(n, -1)

    # prep for output
    x_proj = x_proj_reshaped.view(n, self.num_n, *x.size()[2:])
    return out, x_proj

class GraphUnit1D(GraphUnit):
  def __init__(self, in_channels, num_nodes, inter_channels=None):
    super(GraphUnit1D, self).__init__(in_channels, num_nodes,
                                      inter_channels=inter_channels,
                                      ConvNd=nn.Conv1d)

class GraphUnit2D(GraphUnit):
  def __init__(self, in_channels, num_nodes, inter_channels=None):
    super(GraphUnit2D, self).__init__(in_channels, num_nodes,
                                      inter_channels=inter_channels,
                                      ConvNd=nn.Conv2d)

class GraphUnit3D(GraphUnit):
  def __init__(self, in_channels, num_nodes, inter_channels=None):
    super(GraphUnit3D, self).__init__(in_channels, num_nodes,
                                      inter_channels=inter_channels,
                                      ConvNd=nn.Conv3d)


if __name__ == '__main__':
  """Quick test for graph units"""
  test_input = torch.randn(2, 32, 16)
  net = GraphUnit1D(32, 4)
  test_output, test_map = net(test_input)
  print(test_output.shape, test_map.shape)

  test_input = torch.randn(2, 32, 16, 16)
  net = GraphUnit2D(32, 4)
  test_output, test_map = net(test_input)
  print(test_output.shape, test_map.shape)

  test_input = torch.randn(2, 32, 8, 16, 16)
  net = GraphUnit3D(32, 4)
  test_output, test_map = net(test_input)
  print(test_output.shape, test_map.shape)
