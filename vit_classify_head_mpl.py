"""Classifier with MPL Head
Input: (N, 1, D)
Output: (N, num_class)
"""
from typing import *
from typing_extensions import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class VitHeadMpl(nn.Module):

    def __init__(self, embed_dim: int, num_class: int):
        super(VitHeadMpl, self).__init__()

        self.classify_mpl: nn.Sequential = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(embed_dim, num_class)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (N, D)
        # ret : (N, num_class)
        return self.classify_mpl(x)
    
