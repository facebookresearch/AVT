# Copyright (c) Facebook, Inc. and its affiliates.

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, out_features, nlayers, **kwargs):
        super().__init__()
        layers = [[nn.Linear(in_features, in_features, **kwargs),
                   nn.ReLU()] for _ in range(nlayers - 1)]
        # flatten out the pairs
        layers = [item for sublist in layers for item in sublist]
        layers.append(nn.Linear(in_features, out_features))
        self.cls = nn.Sequential(*layers)

    def forward(self, inp):
        return self.cls(inp)
