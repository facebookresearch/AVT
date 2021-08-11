"""
Implementation of the temporal aggregation algorithms.
    Input: (B, C, T)
    Output: (B, C)
"""
from typing import Sequence
import math
import warnings

import torch
import torch.nn as nn

from models.non_local import NonLocalBlock
from external.rulstm.RULSTM.models import RULSTM


class Identity(nn.Identity):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs), {}

    @property
    def output_dim(self):
        return self.in_features


class Mean(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

    def forward(self, feats):
        """
            feats: B, T, C dimensional input
        """
        return torch.mean(feats, dim=1), {}

    @property
    def output_dim(self):
        return self.in_features


class PositionalEncoding(nn.Module):
    """For now, just using simple pos encoding from language.
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    """ Using a transformer encoder and simple decoder. """
    def __init__(self,
                 in_features,
                 inter_rep=512,
                 nheads=8,
                 nlayers=6,
                 agg_style='mean',
                 cloze_loss_ratio=0.0,
                 cloze_loss_wt=0.0):
        super().__init__()
        self.in_features = in_features
        self.inter_rep = inter_rep
        self.downproject = nn.Linear(in_features, inter_rep)
        layer = nn.TransformerEncoderLayer(d_model=inter_rep, nhead=nheads)
        # Don't think I'll ever consider longer than 1000 features?
        self.pos_encoder = PositionalEncoding(inter_rep, max_len=1000)
        self.transformer_encoder = nn.TransformerEncoder(
            layer, num_layers=nlayers, norm=nn.LayerNorm(inter_rep))
        self.agg_style = agg_style
        self.cloze_loss_ratio = cloze_loss_ratio
        self.cloze_loss_wt = cloze_loss_wt
        self.cloze_loss_fn = nn.MSELoss(reduction='none')
        # The embedding for the [MASK] token
        if self.cloze_loss_ratio > 0:
            self.extra_embeddings = nn.Embedding(1, in_features)

    def forward(self, feats):
        """
        Args:
            feats (B, T, C)
        Returns:
            aggregated features (B, C')
        """
        # Convert to the format used by transformer: T, B, C
        feats = feats.transpose(0, 1)
        kwargs = {}
        if self.training and self.cloze_loss_ratio > 0:
            # Mask out certain positions, so when doing attention these
            # positions will be ignored
            key_padding_mask = torch.rand((feats.size(0), feats.size(1)),
                                          device=feats.device)
            # Get close_ratio amount as True, so those will be ignored
            key_padding_mask = key_padding_mask <= self.cloze_loss_ratio
            # Set the features to MASK embedding, for the ones that are masked
            key_padding_mask_rep = key_padding_mask.unsqueeze(-1).expand(
                -1, -1, feats.size(2))
            # Set the masked elements to 0, and add the MASK embedding
            replaced_feats = (
                feats * (~key_padding_mask_rep) +
                key_padding_mask_rep * self.extra_embeddings(
                    torch.tensor([0], dtype=torch.long,
                                 device=feats.device)).unsqueeze(0))
            feats = replaced_feats
            # Transpose since the function takes in B, T
            kwargs['src_key_padding_mask'] = key_padding_mask.t()
        feats = self.pos_encoder(self.downproject(feats))
        feats_encoded = self.transformer_encoder(feats, **kwargs)
        aux_losses = {}
        if self.training and self.cloze_loss_ratio > 0:
            dist = self.cloze_loss_fn(feats_encoded, feats)
            dist_masked_elts = self.cloze_loss_wt * torch.mean(
                torch.mean(dist, dim=-1) * key_padding_mask)
            aux_losses['tx_mlm'] = dist_masked_elts
        if self.agg_style == 'mean':
            res = torch.mean(feats_encoded, dim=[0])
        elif self.agg_style == 'last':
            res = feats_encoded[-1]
        else:
            raise NotImplementedError(f'Unknown agg style {self.agg_style}')
        return res, aux_losses

    @property
    def output_dim(self):
        return self.inter_rep


class RULSTMAggregation(RULSTM):
    def __init__(self,
                 in_features: int,
                 intermediate_featdim: int = 1024,
                 dropout: float = 0.8,
                 num_pad_feats: int = 0):
        """
        Args:
            num_pad_feats (int): Pad the features with zero feats for this
                many times on the time axis. This is because the unrolling
                LSTM unrolls forward as many times as input, and since original
                models were trained for 14 steps unrolling (upto 0.25s
                before the action), and I usually test for 11 steps (1s before
                action), need to pad 3 times to get the same output when
                testing pre-trained models.
        """
        super().__init__(1, in_features, intermediate_featdim, dropout)
        # Remove the classifier, since the outside code will deal with that
        self.classifier = nn.Sequential()
        self.output_dim = intermediate_featdim
        self.num_pad_feats = num_pad_feats
        # Ignore warnings because it UserWarning: RNN module weights are not
        # part of single contiguous chunk of memory. This means they need to be
        # compacted at every call, possibly greatly increasing memory usage.
        # To compact weights again call flatten_parameters().
        # Not sure how to fix this, adding the flatten didn't really fix
        # Happens only with DataParallel, not DDP
        # Using https://github.com/pytorch/pytorch/issues/24155#issuecomment-604474511
        # Just ignoring the warning
        warnings.filterwarnings('ignore')

    def forward(self, feats):
        """
            Args:
                feats (B, T, C)
            Returns:
                aggregated (B, C)
        """
        if self.num_pad_feats > 0:
            empty_feats = torch.zeros(
                (feats.size(0), self.num_pad_feats, feats.size(-1)),
                dtype=feats.dtype,
                device=feats.device)
            feats = torch.cat([feats, empty_feats], dim=1)
        res = super().forward(feats)
        # Return output corresponding to the last input frame. Note that in
        # original RULSTM they do -4 since they predict 3 steps further into
        # the anticipation time, whereas I stop when the anticipation time
        # starts here.
        # Subtract num_pad_feat as that would mean it predicted further into
        # the future
        return res[:, -1 - self.num_pad_feats, :], {}
