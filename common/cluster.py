# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn


class KmeansAssigner(nn.Module):
    def __init__(self, centroids_fpath, norm=False):
        super().__init__()
        # NxC dimension
        # Not converting this to linear layer as then the weights get
        # overwriten during random init, and these cluster centers are lost.
        self.register_buffer('centroids',
                             torch.load(centroids_fpath)['weight'])
        self.norm = norm

    @property
    def num_clusters(self):
        return self.centroids.size(0)

    @staticmethod
    def feat2cluster(feats, centroids, norm):
        """
        Compute index for the feats, w.r.t centroids.
        Args:
            feats *xC
            centroids KxC
        Returns:
            assignments *
        """
        feats_flat = feats.flatten(0, -2)
        if norm:
            feats_flat = nn.functional.normalize(feats_flat, dim=-1, p=2)
        dists = torch.cdist(feats_flat.unsqueeze(0), centroids.unsqueeze(0))
        assgns = torch.argmin(dists[0], dim=-1)
        assgns = assgns.reshape(feats.shape[:-1])
        return assgns

    @staticmethod
    def cluster2feat(idx, centroids):
        """
        Get features for cluster ids
        Args:
            idx *
            centroids KxC
        Returns:
            assignments *xC
        """
        idx_flat = idx.reshape((-1, ))
        feats = centroids[idx_flat, :]
        return feats.reshape(list(idx.shape) + [feats.size(-1)])

    def forward(self, inp):
        """
        If inp is torch.float, then find the nearest assignments.
        If torch.long, return the corresponding features.
        """
        if inp.dtype == torch.long:
            return self.cluster2feat(inp, self.centroids)
        return self.feat2cluster(inp, self.centroids, self.norm)
