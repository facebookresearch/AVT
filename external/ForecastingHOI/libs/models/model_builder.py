"""
Building models
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from functools import partial

import torch
import torch.nn as nn

from .i3d_inception import I3DInception
from .i3d_resnet import I3DRes50
from .i3d_mfnet import I3DMFNet
from .r2p1d_resnet import r2plus1d_18
from .r3d_resnet import irCSN152
from .decoders import AvgLinear, MaxLinear, GCULinear, AttenLinear, AvgLinearJoint, JointSoftAtten, JointProbAtten
from .losses import SmoothedCrossEntropy, KLDiv, CustomLoss


def build_backbone(network_config):
  """Get backbone network
  return a supported backbone
  """
  model_name = network_config['backbone']
  # by default, all backbone models will return a tuple of feature maps
  # from conv block 2 to conv block 5
  model = {
      'i3d-inception':  partial(I3DInception,
                                modality=network_config['modality'],
                                frozen_stages=network_config['frozen_stages'],
                                pretrained=network_config['pretrained']),
      'i3d-resnet50':  partial(I3DRes50,
                               modality=network_config['modality'],
                               frozen_stages=network_config['frozen_stages'],
                               pretrained=network_config['pretrained'],
                               version='v2'),
      'i3d-resnet50v1':  partial(I3DRes50,
                                 modality=network_config['modality'],
                                 frozen_stages=network_config['frozen_stages'],
                                 pretrained=network_config['pretrained'],
                                 version="v1"),
      'i3d-mfnet':  partial(I3DMFNet,
                            modality=network_config['modality'],
                            frozen_stages=network_config['frozen_stages'],
                            pretrained=network_config['pretrained']),
      'r2plus1d-resnet18': partial(r2plus1d_18,
                                   modality=network_config['modality'],
                                   frozen_stages=network_config['frozen_stages'],
                                   pretrained=network_config['pretrained']),
      # FB model (152 layers) pre-trained on 65 million videos
      # may need to enable gradient cp (to fit into memory)
      'r3d-csn152': partial(irCSN152,
                            modality=network_config['modality'],
                            frozen_stages=network_config['frozen_stages'],
                            pretrained=network_config['pretrained'],
                            gradient_cp=network_config['gradient_cp']),
      # new backbone models ....
  }[model_name]

  return model

def build_decoder(network_config):
  """Get the head of the network
  return a supported decoder
  """
  model_name = network_config['decoder']
  model = {
      'avgfc':  partial(AvgLinear,
                        in_channels=network_config['feat_dim'],
                        num_classes=network_config['num_classes'],
                        dropout_prob=network_config['dropout_prob'],
                        fc_std=network_config['decoder_fc_std']),
      'maxfc':  partial(MaxLinear,
                        in_channels=network_config['feat_dim'],
                        num_classes=network_config['num_classes'],
                        dropout_prob=network_config['dropout_prob'],
                        fc_std=network_config['decoder_fc_std']),
      'gcufc':  partial(GCULinear,
                        in_channels=network_config['feat_dim'],
                        num_nodes=network_config['decoder_num_nodes'],
                        num_classes=network_config['num_classes'],
                        dropout_prob=network_config['dropout_prob'],
                        fc_std=network_config['decoder_fc_std']),
      'attenfc': partial(AttenLinear,
                         in_channels=network_config['feat_dim'],
                         num_classes=network_config['num_classes'],
                         stride=network_config['decoder_stride'],
                         dropout_prob=network_config['dropout_prob'],
                         fc_std=network_config['decoder_fc_std']),
      'avgfcjoint': partial(AvgLinearJoint,
                         in_channels=network_config['feat_dim'],
                         num_classes=network_config['num_classes'],
                         dropout_prob=network_config['dropout_prob'],
                         fc_std=network_config['decoder_fc_std']),
      'softfcjoint': partial(JointSoftAtten,
                         in_channels=network_config['feat_dim'],
                         num_classes=network_config['num_classes'],
                         stride=network_config['decoder_stride'],
                         dropout_prob=network_config['dropout_prob'],
                         fc_std=network_config['decoder_fc_std']),
      'vaefcjoint': partial(JointProbAtten,
                         in_channels=network_config['feat_dim'],
                         num_classes=network_config['num_classes'],
                         stride=network_config['decoder_stride'],
                         dropout_prob=network_config['dropout_prob'],
                         fc_std=network_config['decoder_fc_std']),
      # new decoders ....
  }[model_name]

  return model

def build_loss(network_config):
  """Get the loss function
  """
  # list of loss and their coefficients
  criterion = []
  lam = []

  # classification loss
  if network_config['label_smoothing'] > 0:
    criterion.append(SmoothedCrossEntropy(
      label_smoothing=network_config['label_smoothing'],
      weight=network_config['cls_weights'], ignore_index=-1))
  else:
    criterion.append(nn.CrossEntropyLoss(ignore_index=-1,
      weight=network_config['cls_weights']))
  lam.append(0.0)

  # additional losses
  if "atten" in network_config['decoder']:
    criterion.append(KLDiv())
    lam.append(np.log(network_config['num_classes']))

  if "joint" in network_config['decoder']:
    criterion.append(KLDiv())
    lam.append(2*np.log(network_config['num_classes']))
    criterion.append(KLDiv())
    lam.append(0.0)
  # assemble all loss functions & return
  criterion = CustomLoss(criterion, lam)
  return criterion

class EncoderDecoder(nn.Module):
  """A thin wrapper that builds a full model from network config
  This model will include:
      encoder: backbone network
      decoder: segmentation network
  An example network config
  {
    # multi gpu support
    "devices": [0],  # default: single gpu
    "backbone": "i3d-inception",
    "modality": "rgb",
    "pretrained": False,
    "frozen_stages": -1,
    "feat_dim": 1024,
    "decoder": "avgfc",
    "dropout_prob": 0.5,
    "num_classes": 200, # auto inferred from dataset
    'decoder_fc_std': 0.01,
    'balanced_beta': -1,
    'cls_weights': None,
  }
  """
  def __init__(self, network_config):
    super(EncoderDecoder, self).__init__()
    # delayed instaniation
    encoder = build_backbone(network_config)
    self.encoder = encoder()
    decoder = build_decoder(network_config)
    self.decoder = decoder()
    # get loss function (as a module list)
    self.criterion = build_loss(network_config)
    # save the config
    self.network_config = network_config

  def extract_feats(self, clips):
    feats = self.encoder(clips)
    return feats

  def forward(self, clips, targets=None):
    """
      This entry forward function will dispatch all forward calls
      (1) during training, it will apply a set of loss functions to the outputs
      (2) during testing, it will aggregate scores from augmented samples
      * It will always return the prediction scores as the first input
        if additional outputs are needed, use a forward hook
    """
    if clips.dim() == 5:
      outputs = self.forward_single(clips)
      # apply loss function one by one to every output var
      if (targets is not None) and self.training:
        loss = self.criterion(outputs, targets)
        return outputs[0], loss
      else:
        return outputs
    elif (clips.dim()) == 6 and (targets is None):
      outputs = self.forward_multi(clips)
      return outputs
    else:
      raise TypeError("Input size mis-match!")

  def forward_single(self, clips):
    feats = self.encoder(clips)
    outputs = self.decoder(feats)
    return outputs

  def forward_multi(self, clips):
    # multi crop testing
    batch_size, num_clips, c, t, h, w = clips.size()
    # B, K, C, T, H, W -> B*K, C, T, H, W

    clips = clips.view(batch_size * num_clips, c, t, h, w)

    # B*K, #cls
    outputs = self.forward_single(clips)
    # softmax
    clip_scores = nn.functional.softmax(outputs[0], dim=1)
    # B*K, #cls -> B, K, #cls
    clip_scores = clip_scores.view(batch_size, num_clips, -1)
    # B, #cls
    clip_scores = clip_scores.mean(dim=1)

    hand_map = outputs[1]
    hotspots_map = outputs[2]
    # print(hand_map.size())
    # print(hotspots_map.size())

    return (clip_scores, hand_map, hotspots_map)


# test
if __name__=='__main__':
  # network_config = {"devices": [0],
  #                   "backbone": "i3d-resnet50",
  #                   "modality": "rgb",
  #                   "pretrained": None,
  #                   "frozen_stages": 1,
  #                   "feat_dim": 2048,
  #                   "decoder": "avgfc",
  #                   "gradient_cp": False,  # added by rgirdhar
  #                   "decoder_num_nodes": 1,  # Dunno? added by rgirdhar
  #                   "decoder_stride": 1,  # Dunno? added by rgirdhar
  #                   "label_smoothing": -1,  # added by rgirdhar
  #                   'cls_weights': None,  # added by rgirdhar
  #                   "dropout_prob": 0.5,
  #                   "num_classes": 200,
  #                   'decoder_fc_std': 0.01}
  # The network used in Miao's config
  network_config = {
    "devices": None,
    "backbone": "r3d-csn152",
    "pretrained": None,
    "frozen_stages": 2,
    "gradient_cp": True,
    "feat_dim": 2048,
    "decoder": "avgfc",
    "decoder_fc_std": 0.01,
    "dropout_prob": 0.8,
    "label_smoothing": -1,
    # added by rgirdhar
    "modality": "rgb",
    "num_classes": 200,
    "decoder_num_nodes": 1,
    "decoder_stride": 1,
    "label_smoothing": -1,
    "cls_weights": None,
  }
  model = EncoderDecoder(network_config)

  # test on gpu
  input = torch.randn(16, 3, 8, 128, 128).cuda()
  input_multi = torch.randn(2, 10, 3, 8, 128, 128).cuda()
  model = model.cuda()
  model.eval()
  with torch.no_grad():
    output = model(input)
    print(output.size())
    # multi_crop
    output = model(input_multi)
    print(output.size())
