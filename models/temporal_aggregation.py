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


class SenerCouplingBlock(nn.Module):
    def __init__(self, in_features_spanning, k_spanning, in_features_recent,
                 k_recent, intermediate_featdim):
        """Implements Coupling Block described in Sec. 3.2 on pg 47 of
        https://epic-kitchens.github.io/Reports/EPIC-KITCHENS-Challenges-2020-Report.pdf#page=47
        """
        super().__init__()
        self.nl_self = NonLocalBlock(in_features_spanning,
                                     in_features_spanning,
                                     intermediate_featdim)
        self.nl_cross = NonLocalBlock(in_features_spanning, in_features_recent,
                                      intermediate_featdim)
        self.combine_spanning = nn.Linear(
            intermediate_featdim * (k_spanning + k_recent),
            intermediate_featdim)
        self.combine_recent = nn.Linear(intermediate_featdim * k_recent * 2,
                                        intermediate_featdim)

    def forward(self, spanning_feat: torch.Tensor, recent_feat: torch.Tensor):
        """
        Args:
            spanning_feat: (B, T, C'): Note it's different from the default
                input for TAB (T and C are switched).
            recent_feat: (B, T, C'')
        Returns:
            Fixed length representations, as in Fig 2 (top right)
            spanning: (B, intermediate_dim)
            recent: (B, intermediate_dim)
        """
        spanning2 = self.nl_self(spanning_feat, spanning_feat)
        # Based on the fig 2 (top right), the smaller (recent) feature
        # has to be the "key"
        recent2 = torch.flatten(self.nl_cross(recent_feat, spanning2), 1, 2)
        spanning2 = torch.flatten(spanning2, 1, 2)
        spanning_final = self.combine_spanning(
            torch.cat((spanning2, recent2), dim=-1))
        recent_final = self.combine_recent(
            torch.cat((torch.flatten(recent_feat, 1, 2), recent2), dim=-1))
        return spanning_final, recent_final


class SenerTemporalAggregationBlock(nn.Module):
    """
    Implementation of the temporal aggregation block, as introduced in
    https://arxiv.org/abs/2006.00830
    Specifically, using their anticipation model described in
    https://epic-kitchens.github.io/Reports/EPIC-KITCHENS-Challenges-2020-Report.pdf#page=45
    """
    def __init__(self,
                 in_features: int,
                 recent_num_snippets: int = 2,
                 recent_start_pos: Sequence[float] = (0.27, 0.2, 0.13, 0.06),
                 spanning_num_snippets: Sequence[int] = (2, 3, 5),
                 intermediate_featdim=512):
        """
        Args:
            in_features: The dimensions of the input features (C)
            recent_num_snippets (K_r): The number of snippets for recent feats
            recent_start_pos (i): The ratio of the full clip, defining the
                distance from the end of the clip to get recent features from.
                EPIC defaults used 6s total clip and (1.6, 1.2, 0.8, 0.4), which
                translates to the default ratios used above.
        """
        super().__init__()
        self.recent_num_snippets = recent_num_snippets
        self.recent_start_pos = recent_start_pos
        self.spanning_num_snippets = spanning_num_snippets
        self.intermediate_featdim = intermediate_featdim
        for i, _ in enumerate(recent_start_pos):
            for j, K in enumerate(spanning_num_snippets):
                recent_feat_dim = in_features
                spanning_feat_dim = in_features
                self.add_module(
                    '%d_%d' % (i, j),
                    SenerCouplingBlock(spanning_feat_dim, K, recent_feat_dim,
                                       recent_num_snippets,
                                       intermediate_featdim))
        self.recent_combiner = nn.Linear(
            intermediate_featdim * len(spanning_num_snippets),
            intermediate_featdim)

    @property
    def output_dim(self):
        return self.intermediate_featdim * 2  # concat of recent and spanning

    def forward(self, feats):
        """
        Extract the temporal aggregated representation
        Args:
            feats: (B, T, C)
        Returns:
            aggregated (B, C)
        """
        # Converting from B,T,C -> B,C,T since this code was written with that
        # input in mind, and haven't had the chance to refactor to the new
        # input. TODO
        feats = feats.transpose(1, 2)
        recent_feats = self.compute_recent_features(feats,
                                                    self.recent_num_snippets,
                                                    self.recent_start_pos)
        spanning_feats = self.compute_spanning_features(
            feats, self.spanning_num_snippets)
        # Swap the feature dimension to the end, i.e. to (B, T, C)
        recent_feats = [el.permute(0, 2, 1) for el in recent_feats]
        spanning_feats = [el.permute(0, 2, 1) for el in spanning_feats]
        final_feats = []
        for i, _ in enumerate(self.recent_start_pos):
            all_attended_spanning_feat_i = []
            all_attended_recent_feat_i = []
            for j, _ in enumerate(self.spanning_num_snippets):
                attended_spanning_feat, attended_recent_feat = getattr(
                    self, '%d_%d' % (i, j))(spanning_feats[j], recent_feats[i])
                all_attended_spanning_feat_i.append(attended_spanning_feat)
                all_attended_recent_feat_i.append(attended_recent_feat)
            attended_spanning_feat_i = torch.stack(
                all_attended_spanning_feat_i, dim=0)
            attended_spanning_feat_i = torch.max(attended_spanning_feat_i,
                                                 dim=0)[0]  # S'''_i
            attended_recent_feat_i = self.recent_combiner(
                torch.cat(all_attended_recent_feat_i, dim=-1))  # R'''_i
            # Figured out concat from the fig 3 in challenge report,
            # where the black and red dots are concat before passing through
            # the classifier. And black -- spanning -- comes first (see fig 2
            # bottom, the max over is the black one -- spanning.
            final_feats.append(
                torch.cat([attended_spanning_feat_i, attended_recent_feat_i],
                          dim=-1))
        # Normally each of the i-th prediction will be used to make a cls
        # separately but to be consistent with the output format, lets combine
        # the representations by summing them.
        return torch.sum(torch.stack(final_feats, dim=0), dim=0), {}

    @classmethod
    def compute_spanning_features(cls, feats: torch.Tensor,
                                  num_snippets: Sequence[int]):
        return [cls.compute_snippet_feature(feats, ns) for ns in num_snippets]

    @classmethod
    def compute_recent_features(cls, feats: torch.Tensor, num_snippets: int,
                                start_pos: Sequence[float]):
        res = []
        for pos in start_pos:
            nfeat = int(feats.shape[-1] * pos)
            res.append(
                cls.compute_snippet_feature(feats[..., -nfeat:], num_snippets))
        return res

    @staticmethod
    def compute_snippet_feature(feats: torch.Tensor, num_snippets: int):
        """
        Args:
            feats (B, C, T)
        Returns:
            (B, C, num_snippets)
        """
        # Split the features into K groups
        total_feats = feats.shape[-1]
        snippet_length = int(math.ceil(total_feats / num_snippets))
        # Max pool over snippet length
        # Pad the past with 0s if needed to get the sizes to match up
        feats_pooled = nn.MaxPool1d(
            snippet_length,
            stride=snippet_length,
        )(nn.ConstantPad1d((num_snippets * snippet_length - total_feats, 0),
                           0)(feats))
        return feats_pooled


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
