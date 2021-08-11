"""Cross entropy loss, that works with multi-dim input."""
import torch
import torch.nn as nn
from common.cluster import KmeansAssigner


class MultiDimCrossEntropy(nn.CrossEntropyLoss):
    def forward(self, inp, tgt, *args, **kwargs):
        """
        Args:
            inp: (*, C)
            tgt: (*, )
            Will reshape the flatten initial dimensions and then incur loss
        """
        assert inp.ndim == tgt.ndim + 1
        assert inp.shape[:-1] == tgt.shape
        res = super().forward(inp.reshape(-1, inp.size(-1)), tgt.reshape(
            (-1, )), *args, **kwargs)
        if torch.numel(res) == torch.numel(tgt):
            # Reduction was not done, so reshape back to orig shape
            res = res.reshape(tgt.shape)
        return res


class QuantizeAndCrossEntropy(MultiDimCrossEntropy):
    """Given a set of cluster centers, project the features to that before
    incurring the loss."""
    def __init__(self, centroids_fpath, norm=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assigner = KmeansAssigner(centroids_fpath)
        self.norm = norm

    def forward(self, inp, tgt):
        """
        Args:
            inp: (*, C)
            tgt: (*, C)
            Will reshape the flatten initial dimensions and then incur loss
        """
        # Normalize L2 both target and input, since that's how I'm computing
        # centroids
        if self.norm:
            inp = nn.functional.normalize(inp, dim=-1, p=2)
            tgt = nn.functional.normalize(tgt, dim=-1, p=2)
        # assign the GT and predictions to the centroids
        inp_proj = torch.mm(inp.flatten(0, 1),
                            self.centroids.t()).view(inp.shape[:-1] +
                                                     self.centroids.shape[:1])
        # the weights of project layer are the centroids, so pick from there
        tgt_proj_q = self.assigner(tgt)
        return super().forward(inp_proj, tgt_proj_q)
