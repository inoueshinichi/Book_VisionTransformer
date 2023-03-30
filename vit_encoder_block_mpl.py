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


class VitEncoderBlockMpl(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_p: float):
        super(VitEncoderBlockMpl, self).__init__()

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
    N = 5
    C = 8
    D = 128

    input_dim = D
    hidden_dim = 512
    output_dim = D
    dropout = 0.2

    encoder_block_mpl = VitEncoderBlockMpl(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout_p=dropout
    )

    print('encoder_block_mpl', encoder_block_mpl)

    p_numel_list = [ p.numel() for p in encoder_block_mpl.parameters() ]
    print("sum(p_numel_list): {}, p_numel_list: {}".format(sum(p_numel_list), p_numel_list))

    input = torch.randn(N, C, D)
    print('input.shape', input.shape)

    output = encoder_block_mpl(input)

    print('output.shape', output.shape)


if __name__ == '__main__':
    test_encoder_block_mpl()

