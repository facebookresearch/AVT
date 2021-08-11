"""
Modular implementation of the basic train ops
"""
from typing import Dict, Union, Tuple
import torch
import torch.nn as nn
import hydra
from hydra.types import TargetConf

from common import utils

from datasets.base_video_dataset import FUTURE_PREFIX
from models.base_model import PAST_LOGITS_PREFIX
from loss_fn.multidim_xentropy import MultiDimCrossEntropy


class NoLossAccuracy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return {}, {}


class BasicLossAccuracy(nn.Module):
    def __init__(self, dataset, device, balance_classes=False):
        super().__init__()
        kwargs = {'ignore_index': -1}
        if balance_classes:
            assert dataset.class_balanced_sampling is False, (
                'Do not re-weight the losses, and do balanced sampling')
            weight = torch.zeros((len(dataset.classes, )),
                                 device=device,
                                 dtype=torch.float)
            for cls_id, count in dataset.classes_counts.items():
                weight[cls_id] = count
            weight = weight / torch.sum(weight)  # To get ratios for non -1 cls
            weight = 1 / (weight + 0.00001)
            kwargs['weight'] = weight
        kwargs['reduction'] = 'none'  # to get batch level output
        self.cls_criterion = MultiDimCrossEntropy(**kwargs)

    def forward(self, outputs, target, target_subclips):
        """
        Args:
            outputs['logits'] torch.Tensor (B, num_classes) or
                (B, T, num_classes)
                Latter in case of dense prediction
            target: {type: (B) or (B, T')}; latter in case of dense prediction
            target_subclips: {type: (B, #clips, T)}: The target for each input
                frame
        """
        losses = {}
        accuracies = {}
        for tgt_type, tgt_val in target.items():
            logits = outputs[f'logits/{tgt_type}']
            assert logits.ndim == tgt_val.ndim + 1
            loss = self.cls_criterion(logits, tgt_val)
            dataset_max_classes = logits.size(-1)
            acc1, acc5 = utils.accuracy(logits,
                                        tgt_val,
                                        topk=(1, min(5, dataset_max_classes)))
            # Don't use / in loss since I use the config to set weights, and
            # can't use / there.
            losses[f'cls_{tgt_type}'] = loss
            accuracies[f'acc1/{tgt_type}'] = acc1
            accuracies[f'acc5/{tgt_type}'] = acc5
            # Incur past losses
            past_logits_key = f'{PAST_LOGITS_PREFIX}logits/{tgt_type}'
            # If this key exists, means we asked for classifier on the last
            # layer, so the loss should be incurred.
            if past_logits_key in outputs and target_subclips is not None:
                past_logits = outputs[past_logits_key]
                # Take mode over the frames to get the subclip level loss
                past_target = torch.mode(target_subclips[tgt_type], -1)[0]
                assert past_logits.shape[:-1] == past_target.shape, (
                    f'The subclips should be set such that the past logits '
                    f'and past targets match in shape. Currently they are '
                    f'{past_logits.shape} and {past_target.shape}')
                losses[f'past_cls_{tgt_type}'] = self.cls_criterion(
                    past_logits, past_target)
            # Else likely not using subclips, so no way to do this loss
        return losses, accuracies


class DenseRegressionLossAccuracy(BasicLossAccuracy):
    """
    Note that this is the only function that will run the dense prediction
        evaluation in the main train function.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_criterion = nn.L1Loss(reduction='none')

    def forward(self, outputs, target, target_subclips):
        """
        Args:
            target: {type: (B, 1+N)} tensor. Here N is the number of active
                classes for the input. The N is 1 in simple classification, but
                > 1 when multi-label classification. For now, for simplicity,
                just incurring multiple losses for each output with the same
                logits. Ideally should use binary cross entropy but this
                should have a similar effect.
        """
        # Padding to match dims?
        # output_time_dim = outputs['logits'].size(1)
        # target_time_dim = target.size(1)
        # if target_time_dim < output_time_dim:
        #     target_pad = -torch.ones(
        #         (target.size(0), output_time_dim - target.size(1),
        #          target.size(2)),
        #         device=target.device,
        #         dtype=target.dtype)
        #     target = torch.cat((target_pad, target), dim=1)
        # Pad the targets since the outputs might have more stuff since GPT
        all_losses = []
        all_acc1 = []
        all_acc5 = []
        raise NotImplementedError('Not yet handled the dict format of target')
        for i in range(target.size(2) - 1):
            loss, acc1, acc5 = super().forward(
                outputs, target[:, :, i + 1].to(torch.long))
            all_losses.append(loss)
            all_acc1.append(acc1)
            all_acc5.append(acc5)
        # Mean over all the loss accuracy for each active class
        loss = {
            k: torch.mean(torch.stack([dic[k] for dic in all_losses]))
            for k in all_losses[0]
        }
        acc1 = torch.mean(torch.stack(all_acc1))
        acc5 = torch.mean(torch.stack(all_acc5))
        # Incur loss and only keep the ones where target > 0, i.e. not padded
        # inputs
        reg_loss = torch.mean(self.reg_criterion(
            outputs['logits_regression'].squeeze(-1), target[:, :, 0]) *
                              (target[:, :, 0] > 0),
                              dim=-1)
        loss['reg'] = reg_loss
        return loss, acc1, acc5


class Basic:
    def __init__(self,
                 model,
                 device,
                 dataset,
                 cls_loss_acc_fn: TargetConf,
                 reg_criterion: TargetConf = None):
        super().__init__()
        self.model = model
        self.device = device
        self.cls_loss_acc_fn = hydra.utils.instantiate(cls_loss_acc_fn,
                                                       dataset, device)
        del reg_criterion  # not used here

    def _basic_preproc(self, data, train_mode):
        if not isinstance(data, dict):
            video, target = data
            # Make a dict so that later code can use it
            data = {}
            data['video'] = video
            data['target'] = target
            data['idx'] = -torch.ones_like(target)

        if train_mode:
            self.model.train()
        else:
            self.model.eval()
        return data

    def __call__(
            self,
            data: Union[Dict[str, torch.Tensor],  # If dict
                        Tuple[torch.Tensor, torch.Tensor]],  # vid, target
            train_mode: bool = True):
        """
        Args:
            data (dict): Dictionary of all the data from the data loader
        """
        data = self._basic_preproc(data, train_mode)
        video = data['video'].to(self.device, non_blocking=True)
        target = {}
        target_subclips = {}
        for key in data['target'].keys():
            target[key] = data['target'][key].to(self.device,
                                                 non_blocking=True)
        outputs, aux_losses = self.model(video,
                                         target_shape=next(
                                             iter(target.values())).shape)
        if 'target_subclips' in data:
            for key in data['target_subclips'].keys():
                target_subclips[key] = data['target_subclips'][key].to(
                    self.device, non_blocking=True)
        else:
            target_subclips = None
        losses, accuracies = self.cls_loss_acc_fn(outputs, target,
                                                  target_subclips)
        losses.update(aux_losses)
        return data, outputs, losses, accuracies


class PredFutureFeat(Basic):
    """
    Earlier I was trying to run the DDP model multiple times for a current
        and future videos and then back-proping. That seems like it was failing
        because it would lead to some in-place updates of some parameters.
        This fixes it.
    Jul 14 2020: Figured what the above issue was due to. Doing multiple
        forward passes was leading to the BatchNorm running mean/var params
        being updated in place, and were not available for computing gradients
        for other parameters. Was also able to repro in a CIFAR distributed
        training setup, but not quite in a standalone setup. Perhaps this error
        is related to https://github.com/pytorch/pytorch/issues/13402
        Will need to dig deeper later, for now this common feature extraction
        should be fine.
    """
    def __init__(self,
                 *args,
                 reg_criterion: TargetConf = None,
                 future_target: str = 'temp_agg_projected',
                 incur_loss_style: str = 'separately',
                 combine_future_losses: TargetConf = {'_target_': 'torch.min'},
                 cumulative_future: bool = False,
                 **kwargs):
        '''
        Args:
            incur_loss_style (str): Defines how to incur losses for multiple
                futures. Could do 'separately', and then combine using
                `combine_future_losses`. Or 'together', such as for MIL-NCE.
        '''
        super().__init__(*args, **kwargs)
        self.reg_criterion = hydra.utils.instantiate(reg_criterion)
        self.future_target = future_target
        self.incur_loss_style = incur_loss_style
        self.combine_future_losses = combine_future_losses
        self.cumulative_future = cumulative_future

    def __call__(
            self,
            data: Union[Dict[str, torch.Tensor],  # If dict
                        Tuple[torch.Tensor, torch.Tensor]],  # vid, target
            train_mode: bool = True):
        data = self._basic_preproc(data, train_mode)
        video = data['video'].to(self.device, non_blocking=True)
        target = {
            key: val.to(self.device, non_blocking=True)
            for key, val in data['target'].items()
        }
        batch_size = video.size(0)
        if train_mode:
            # At test time, I don't sample the extra future video, since
            # that is only used during training
            all_videos = [video]
            nfutures = len(
                [key for key in data.keys() if key.startswith(FUTURE_PREFIX)])
            for i in range(nfutures):
                future_vid = data[f'{FUTURE_PREFIX}_{i}_video'].to(
                    self.device, non_blocking=True)
                all_videos.append(future_vid)
            video = torch.cat(all_videos, dim=0)  # Add to batch dim
        outputs_full, aux_losses = self.model(video)
        # Just the actual video for outputs
        outputs = {key: val[:batch_size] for key, val in outputs_full.items()}
        # if self.cls_loss_wt != 0:
        # Doing this makes some layers not have gradients and it gives errors,
        # so just leaving it here for now. The gradient should be 0 anyway
        losses, accuracies = self.cls_loss_acc_fn(outputs, target)
        losses.update(aux_losses)
        losses['cls'] = losses['cls']
        if train_mode:
            # Incur the regression losses, for each of the futures
            reg_losses = []
            if self.incur_loss_style == 'separately':
                for i in range(nfutures):
                    future_feats = outputs_full[self.future_target][
                        (i + 1) * batch_size:(i + 2) * batch_size]
                    if self.cumulative_future:
                        future_feats = torch.cumsum(future_feats, 0)
                        # Divide by the position to get mean of features until then
                        future_feats = future_feats / (torch.range(
                            1,
                            future_feats.size(0),
                            device=future_feats.device,
                            dtype=future_feats.dtype).unsqueeze(1))
                    loss = self.reg_criterion(outputs['future_projected'],
                                              future_feats)
                    reg_losses.append(loss)
                final_reg_loss = hydra.utils.call(self.combine_future_losses,
                                                  torch.stack(reg_losses))
            elif self.incur_loss_style == 'together':
                future_feats = outputs_full[self.future_target][batch_size:]
                future_feats = future_feats.reshape(
                    (-1, batch_size, future_feats.size(-1))).transpose(0, 1)
                final_reg_loss = self.reg_criterion(
                    outputs['future_projected'], future_feats)
            else:
                raise NotImplementedError(self.incur_loss_style)
            losses['reg'] = final_reg_loss
        return data, outputs, losses, accuracies
