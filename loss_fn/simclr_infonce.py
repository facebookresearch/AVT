# Copyright (c) Facebook, Inc. and its affiliates.

"""The SimCLR InfoNCE loss."""
import torch
import torch.nn as nn

from common import utils

LARGE_NUM = 1e9


class MILCrossEntropyLoss(nn.Module):
    def __init__(self, mil_type='sum', reduction='mean'):
        super().__init__()
        self.mil_type = mil_type
        self.reduction = reduction

    def forward(self, *args, **kwargs):
        if self.mil_type == 'sum':
            return self.forward_sum(*args, **kwargs)
        elif self.mil_type == 'max':
            return self.forward_max(*args, **kwargs)
        else:
            raise NotImplementedError(f'Unknown type {self.mil_type}')

    def forward_sum(self, pred, labels_onehot):
        """
        Args:
            pred: BxC is the output
            labels: BxC is 1s for positive, and 0s for negatives
        Based on https://github.com/antoine77340/MIL-NCE_HowTo100M/blob/master/loss.py
        Or the MIL-NCE paper Eq 1 (https://arxiv.org/pdf/1912.06430.pdf)
        """
        assert pred.shape == labels_onehot.shape
        # In the MILNCE code there is a sum, followed by logsumexp. I think
        # using the labels to select the positive samples and then doing
        # logsumexp will have the same effect.
        pos_pred = pred[labels_onehot.bool()].reshape((pred.size(0), -1))
        numerator = torch.logsumexp(pos_pred, dim=1)
        denominotor = torch.logsumexp(pred, dim=1)
        loss = denominotor - numerator
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'none':
            pass
        else:
            raise NotImplementedError(f'Unknown reduction {self.reduction}')
        return loss

    def forward_max(self, pred, labels_onehot):
        """
        Args:
            pred: BxC is the output
            labels: BxC is 1s for positive, and 0s for negatives
        Based on Appendix A (https://arxiv.org/pdf/1912.06430.pdf)
        """
        assert pred.shape == labels_onehot.shape
        # Do max before, and then logsumexp. Works since exp is monotonic fn
        # so the max with exp or without will be the same.
        pos_pred = pred[labels_onehot.bool()].reshape((pred.size(0), -1))
        pos_pred = torch.max(pos_pred, dim=1, keepdim=True)[0]
        neg_pred = pred[~labels_onehot.bool()].reshape((pred.size(0), -1))
        numerator = torch.logsumexp(pos_pred, dim=1)
        denominotor = torch.logsumexp(torch.cat([pos_pred, neg_pred], dim=1),
                                      dim=1)
        return torch.mean(denominotor - numerator)


class DistributedSimclrInfoNCELoss(nn.Module):
    def __init__(self,
                 temperature: float = 0.1,
                 target_to_output_loss=True,
                 mil_type='sum',
                 reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.criterion = MILCrossEntropyLoss(mil_type, reduction=reduction)
        # This defines whether the reverse part of the loss, from target to
        # the output features, is incurred.
        self.target_to_output_loss = target_to_output_loss

    def forward(self, output: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output: BxC
            target: BxC or BxKxC <-- In case of MIL NCE, K is the number of
                positives for each batch element.
            Following https://github.com/google-research/simclr/blob/master/objective.py
        """
        # Normalize first, before the gather -- so that all the features I get
        # are normalized
        output = nn.functional.normalize(output, dim=-1, p=2)
        target = nn.functional.normalize(target, dim=-1, p=2)
        # To be consistent with MIL-NCE input, convert K to batch dim,
        # and repeat the output to same value for each repeated target
        elt_for_back_loss = 0
        if target.ndim == 3:
            num_matching = target.size(1)
            target_flat = target.reshape((-1, target.size(-1)))
            # Keep the first one for the back loss
            target = target[:, elt_for_back_loss]
        else:
            num_matching = 1
            target_flat = target
        # Gather all the outputs and all the targets
        output_all = self.gather_embeddings(output)
        target_flat_all = self.gather_embeddings(target_flat)
        batch_size = output.size(0)
        replica_id = utils.get_rank()
        # -> (B, B_full * num_matching)
        labels_onehot = torch.zeros((batch_size, output_all.size(0)),
                                    dtype=output.dtype,
                                    device=output.device)
        extra_zeros = torch.zeros((batch_size, output_all.size(0)),
                                  dtype=output.dtype,
                                  device=output.device)
        ones_diag = torch.eye(batch_size,
                              batch_size,
                              dtype=output.dtype,
                              device=output.device)
        labels_onehot[:, replica_id * batch_size:(replica_id + 1) *
                      batch_size] = ones_diag
        labels_onehot_interleaved = labels_onehot.repeat_interleave(
            num_matching, dim=1)
        # (B, C) * (B_full, C) -> (B, B_full)
        logits_aa = torch.mm(output, output_all.t() / self.temperature)
        # (B, C) * (B_full * num_matching, C) -> (B, B_full * num_matching)
        logits_ab = torch.mm(output, target_flat_all.t() / self.temperature)
        logits_aa = logits_aa - labels_onehot * LARGE_NUM
        loss = self.criterion(
            torch.cat([logits_ab, logits_aa], 1),
            torch.cat([labels_onehot_interleaved, extra_zeros], 1))
        if self.target_to_output_loss:
            # Keep only the first prediction, since that is what I will incur
            # reverse loss with
            target_all = target_flat_all[elt_for_back_loss::num_matching]
            logits_bb = torch.mm(target, target_all.t() / self.temperature)
            logits_bb = logits_bb - labels_onehot * LARGE_NUM
            logits_ba = torch.mm(target, output_all.t() / self.temperature)
            loss = loss + self.criterion(
                torch.cat([logits_ba, logits_bb], 1),
                torch.cat([labels_onehot, extra_zeros], 1))
        return loss

    def gather_embeddings(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Do a gather over all embeddings, so we can compute the loss.
        Final shape is like: (batch_size * num_gpus) x embedding_dim
        """
        if torch.distributed.is_available(
        ) and torch.distributed.is_initialized():
            # gather all embeddings.
            embedding_gathered = utils.gather_from_all(embedding)
        else:
            embedding_gathered = embedding
        return embedding_gathered


class MultiDimDistributedSimclrInfoNCELoss(DistributedSimclrInfoNCELoss):
    """
    Fold in the initial dimensions and run simple NCE.
    """
    def forward(self, output: torch.Tensor, target: torch.Tensor, *args,
                **kwargs) -> torch.Tensor:
        return super().forward(output.flatten(0, -2), target.flatten(0, -2),
                               *args, **kwargs)
