"""
Implementation of the Non-Local block (modified version) as in the
Long-term feature banks paper: https://arxiv.org/abs/1812.05038
"""
import math
import torch
import torch.nn as nn


class NonLocalBlock(nn.Module):
    """
    Implements Fig 4 from https://arxiv.org/pdf/1812.05038.pdf
    """
    def __init__(self, in_dim1, in_dim2, inter_dim=512):
        super().__init__()
        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim2
        self.inter_dim = inter_dim
        assert in_dim1 == inter_dim, (
            'Since the NL block produces a residual that is added')
        # Map the features to inter dim
        self.project_key = nn.Linear(in_dim1, inter_dim)
        self.project_mem_1 = nn.Linear(in_dim2, inter_dim)
        self.project_mem_2 = nn.Linear(in_dim2, inter_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(inter_dim)
        self.relu = nn.ReLU()
        self.final_project = nn.Linear(inter_dim, inter_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, key, memory):
        """
        Args:
            key (B, N1, in_dim1) - The "Key" feature
            memory (B, N2, in_dim2) - The "Memory/Context" feature
        """
        key_proj = self.project_key(key)
        memory_proj1 = self.project_mem_1(memory)
        memory_proj2 = self.project_mem_2(memory)
        # Will gen BxN1xN2
        inner_prod = torch.bmm(key_proj, memory_proj1.transpose(2, 1))
        inner_prod *= math.sqrt(1 / self.inter_dim)
        inner_prod_softmax = self.softmax(
            inner_prod.view(inner_prod.shape[0], -1)).view(inner_prod.shape)
        # Generates BxN1xC
        memory_proj2_attended = torch.bmm(inner_prod_softmax, memory_proj2)
        return (key + self.dropout(
            self.final_project(
                self.relu(self.layer_norm(memory_proj2_attended)))))
