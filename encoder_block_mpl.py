"""EncoderBlock内で使うMPL
Input: 
Output: 

MPL: Linear -> GELU(activation) -> Dropout -> Linear -> Dropout
"""

from typing import *
from typing_extensions import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlockMpl(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_p: float):
        super(EncoderBlockMpl, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p

        self.mpl: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mpl(x)


def test_encoder_block_mpl():
    pass


if __name__ == '__main__':
    test_encoder_block_mpl()

