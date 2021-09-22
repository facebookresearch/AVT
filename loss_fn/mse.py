# Copyright (c) Facebook, Inc. and its affiliates.

"""Variants of MSE loss."""
import torch.nn as nn


class NormedMSE(nn.MSELoss):
    def forward(self, inp, tgt, *args, **kwargs):
        """
        Args:
            inp: (*, C)
            tgt: (*, C)
            Will normalize the input before the loss
        """
        inp = nn.functional.normalize(inp, dim=-1, p=2)
        tgt = nn.functional.normalize(tgt, dim=-1, p=2)
        return super().forward(inp, tgt, *args, **kwargs)
