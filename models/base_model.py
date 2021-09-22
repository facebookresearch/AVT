# Copyright (c) Facebook, Inc. and its affiliates.

"""
The overall base model.
"""
from typing import Dict, Tuple
import operator
import torch
import torch.nn as nn
import hydra
from omegaconf import OmegaConf

CLS_MAP_PREFIX = 'cls_map_'
PAST_LOGITS_PREFIX = 'past_'


class BaseModel(nn.Module):
    def __init__(self, model_cfg: OmegaConf, num_classes: Dict[str, int],
                 class_mappings: Dict[Tuple[str, str], torch.FloatTensor]):
        super().__init__()
        # Takes as input (B, T, H, W, C) -> (B, T', H', W', C')
        _backbone_full = hydra.utils.instantiate(
            model_cfg.backbone,
            # Add dummy value for num_cls
            # will be removed next anyway
            num_classes=1)
        if model_cfg.backbone_last_n_modules_to_drop > 0:
            self.backbone = nn.Sequential()
            for name, child in list(_backbone_full.named_children(
            ))[:-model_cfg.backbone_last_n_modules_to_drop]:
                self.backbone.add_module(name, child)
        else:
            self.backbone = _backbone_full
        # Map the (B, T', H', W', C') -> (B, T', H', W', C*)
        # to the intermediate feature dimensions
        # IMP: this is only used if C' != C*
        if (model_cfg.backbone_last_n_modules_to_drop == 0
                and 'output_dim' in dir(self.backbone)):
            backbone_dim = self.backbone.output_dim
        else:
            backbone_dim = model_cfg.backbone_dim  # TODO: Figure automatically
        self.mapper_to_inter = None
        if model_cfg.intermediate_featdim is None:
            model_cfg.intermediate_featdim = backbone_dim
        if backbone_dim != model_cfg.intermediate_featdim:
            self.mapper_to_inter = nn.Linear(backbone_dim,
                                             model_cfg.intermediate_featdim,
                                             bias=False)
        # Takes as input (B, T', H', W', C*) -> (B, C**)
        self.temporal_aggregator = hydra.utils.instantiate(
            model_cfg.temporal_aggregator,
            in_features=model_cfg.intermediate_featdim)
        self.reset_temp_agg_feat_dim = nn.Sequential()
        temp_agg_output_dim = self.temporal_aggregator.output_dim
        if model_cfg.same_temp_agg_dim and (temp_agg_output_dim !=
                                            model_cfg.intermediate_featdim):
            # Ideally want to maintain it so that the same project_mlp
            # can be used for the temporally aggregated features, or the
            # original features.
            self.reset_temp_agg_feat_dim = nn.Linear(
                temp_agg_output_dim, model_cfg.intermediate_featdim)
            temp_agg_output_dim = model_cfg.intermediate_featdim
        # Transforms the current features to future ones
        # (B, C**) -> (B, C**)
        self.future_predictor = hydra.utils.instantiate(
            model_cfg.future_predictor,
            in_features=temp_agg_output_dim,
            _recursive_=False)
        # Projection layer
        self.project_mlp = nn.Sequential()
        if model_cfg.project_dim_for_nce is not None:
            self.project_mlp = nn.Sequential(
                nn.Linear(temp_agg_output_dim, temp_agg_output_dim),
                nn.ReLU(inplace=True),
                nn.Linear(temp_agg_output_dim, model_cfg.project_dim_for_nce))
        # 2nd round of temporal aggregation, if needed
        self.temporal_aggregator_after_future_pred = hydra.utils.instantiate(
            model_cfg.temporal_aggregator_after_future_pred,
            self.future_predictor.output_dim)
        # Dropout
        self.dropout = nn.Dropout(model_cfg.dropout)
        # Takes as input (B, C**) -> (B, num_classes)
        cls_input_dim = self.temporal_aggregator_after_future_pred.output_dim
        # Make a separate classifier for each output
        self.classifiers = nn.ModuleDict()
        self.num_classes = num_classes
        for i, (cls_type, cls_dim) in enumerate(num_classes.items()):
            if model_cfg.use_cls_mappings and i > 0:
                # In this case, rely on the class mappings to generate the
                # other predictions, rather than creating a new linear layer
                break
            self.classifiers.update({
                cls_type:
                hydra.utils.instantiate(model_cfg.classifier,
                                        in_features=cls_input_dim,
                                        out_features=cls_dim)
            })
        # Store the class mappings as buffers
        for (src, dst), mapping in class_mappings.items():
            self.register_buffer(f'{CLS_MAP_PREFIX}{src}_{dst}', mapping)
        self.regression_head = None
        if model_cfg.add_regression_head:
            self.regression_head = nn.Linear(cls_input_dim, 1)
        # Init weights, as per the video resnets
        self._initialize_weights()
        # Set he BN momentum and eps here, Du uses a different value and its imp
        self._set_bn_params(model_cfg.bn.eps, model_cfg.bn.mom)
        self.cfg = model_cfg

    def _initialize_weights(self):
        # Copied over from
        # https://github.com/pytorch/vision/blob/75f5b57e680549d012b3fc01b356b2fb92658ea7/torchvision/models/video/resnet.py#L261
        # Making sure all layers get init to good video defaults
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _set_bn_params(self, bn_eps=1e-3, bn_mom=0.1):
        """
        Set the BN parameters to the defaults: Du's models were trained
            with 1e-3 and 0.9 for eps and momentum resp.
            Ref: https://github.com/facebookresearch/VMZ/blob/f4089e2164f67a98bc5bed4f97dc722bdbcd268e/lib/models/r3d_model.py#L208
        """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.eps = bn_eps
                module.momentum = bn_mom

    def forward_singlecrop(self, video, target_shape=None):
        """
        Args:
            video (torch.Tensor, Bx#clipsxCxTxHxW)
            target_shape: The shape of the target. Some of these layers might
                be able to use this information.
        """
        outputs = {}
        aux_losses = {}
        batch_size = video.size(0)
        num_clips = video.size(1)
        # Fold the clips dimension into the batch for feature extraction, upto
        # temporal aggregation
        video = video.flatten(0, 1)
        feats = self.backbone(video)
        outputs['backbone'] = feats
        # Spatial mean
        feats = torch.mean(feats, [-1, -2])
        # store temporal mean as well
        outputs['backbone_mean'] = torch.mean(feats, [-1])
        # If it's not sequential and can be applied here
        if len(self.project_mlp) > 0 and (outputs['backbone_mean'].size(-1) ==
                                          self.project_mlp[0].in_features):
            outputs['backbone_mean_projected'] = self.project_mlp(
                outputs['backbone_mean'])
        # Move the time dimension inside: B,C,T -> B,T,C
        feats = feats.permute((0, 2, 1))
        # Map the feats to intermediate dimension, that rest of the code
        # will operate on. Only if the original feature is not already
        if feats.shape[-1] != self.cfg.intermediate_featdim:
            assert self.mapper_to_inter is not None, (
                f'The backbone feat does not match intermediate {feats.shape} '
                f'and {self.cfg.intermediate_featdim}. Please set '
                f'model.backbone_dim correctly.')
            feats = self.mapper_to_inter(feats)
        feats_agg, agg_losses = self.temporal_aggregator(feats)
        aux_losses.update(agg_losses)
        feats_agg = self.reset_temp_agg_feat_dim(feats_agg)
        outputs['temp_agg'] = feats_agg
        # For the contrastive loss, I need a projected version of this feature
        outputs['temp_agg_projected'] = self.project_mlp(feats_agg)
        # Now before future prediction, need to unfold the clips back out,
        # and concat on the temporal dimension
        if num_clips > 1:
            assert (
                (feats_agg.ndim == 2)
                or (feats_agg.ndim == 3 and feats_agg.size(1) == 1)
            ), ('Should be using some temporal aggregation when using clips')
            feats_agg = feats_agg.reshape((batch_size, num_clips) +
                                          feats_agg.shape[1:])
            if feats_agg.ndim == 4:
                feats_agg = torch.flatten(feats_agg, 1, 2)
            # now feats_agg back to 3D (B, T, F)
        feats_past = feats_agg
        # Now the future prediction, also it might update the past features
        # like the GPT style models would
        (feats_past, feats_future, future_losses,
         endpoints) = self.future_predictor(feats_past, target_shape)
        aux_losses.update(future_losses)
        outputs.update(endpoints)
        outputs['future'] = feats_future
        outputs['past'] = feats_past
        # Apply a classifier on the past features, might be supervising that
        if self.cfg.classifier_on_past:
            feats_past_drop = self.dropout(feats_past)
            outputs.update(
                self._apply_classifier(feats_past_drop,
                                       outputs_prefix=PAST_LOGITS_PREFIX))
        # For the contrastive loss, I need a projected version of this feature
        outputs['future_projected'] = self.project_mlp(feats_agg)
        # Aggregate again, if asked for
        feats_future_agg, future_agg_losses = (
            self.temporal_aggregator_after_future_pred(feats_future))
        aux_losses.update(future_agg_losses)
        outputs['future_agg'] = feats_future_agg
        feats_future_agg_drop = self.dropout(feats_future_agg)
        outputs.update(self._apply_classifier(feats_future_agg_drop))
        if self.regression_head:
            outputs['logits_regression'] = self.regression_head(
                feats_future_agg_drop)
        return outputs, aux_losses

    def _apply_classifier(self, input_feat, outputs_prefix=''):
        outputs = {}
        for key in self.num_classes.keys():
            if key in self.classifiers:
                outputs[f'{outputs_prefix}logits/{key}'] = self.classifiers[
                    key](input_feat)
            else:
                # A mapping must exist, in order to compute this, and must
                # have been computed already (so ordering in the config
                # matters)
                src_key = next(iter(self.classifiers.keys()))
                src_tensor = outputs[f'{outputs_prefix}logits/{src_key}']
                mapper = operator.attrgetter(
                    f'{CLS_MAP_PREFIX}{key}_{src_key}')(self)
                outputs[f'{outputs_prefix}logits/{key}'] = torch.mm(
                    src_tensor, mapper)
        return outputs

    def forward(self, video, *args, **kwargs):
        """
            Args: video (torch.Tensor)
                Could be (B, #clips, C, T, H, W) or
                    (B, #clips, #crops, C, T, H, W)
            Returns:
                Final features
                And any auxiliarly losses produced by the model
        """
        if video.ndim == 6:
            video_crops = [video]
        elif video.ndim == 7 and video.size(2) == 1:
            video_crops = [video.squeeze(2)]
        elif video.ndim == 7:
            video_crops = torch.unbind(video, dim=2)
        else:
            raise NotImplementedError('Unsupported size %s' % video.shape)
        feats_losses = [
            self.forward_singlecrop(el, *args, **kwargs) for el in video_crops
        ]
        feats, losses = zip(*feats_losses)
        # Convert to dict of lists
        feats = {k: [dic[k] for dic in feats] for k in feats[0]}
        losses = {k: [dic[k] for dic in losses] for k in losses[0]}
        # Average over the crops
        feats = {
            k: torch.mean(torch.stack(el, dim=0), dim=0)
            for k, el in feats.items()
        }
        losses = {
            k: torch.mean(torch.stack(el, dim=0), dim=0)
            for k, el in losses.items()
        }
        return feats, losses
