"""Encoder Module of ViT
Input: (N, T, D); Class token + PatchPositional Embedding tokens
Output: (N, 1, D); Class token vector to `Classify Mpl Head`

Sequencial(EncoderBlock1, EncoderBlock2, ..., EncoderBlock#)
"""

from typing import *
from typing_extensions import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_encoder_block import VitEncoderBlock

class VitEncoder(nn.Module):

    def __init__(self, num_blocks: int, num_head: int, embed_dim: int, mpl_dim: int, dropout_p: float):
        super(VitEncoder, self).__init__()

        self.num_blocks: int = num_blocks

        self.block_list = [ 
            VitEncoderBlock(
                num_head=num_head, 
                embed_dim=embed_dim, 
                mpl_dim=mpl_dim, 
                dropout_p=dropout_p
            ) for _ in range(self.num_blocks)]
        self.encoder: nn.Sequential = nn.Sequential(*self.block_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    

def test_encoder():
    N = 2
    C = 3
    head_num = 3
    D = 48 * head_num

    embed_dim = D
    mpl_dim = 512
    dropout_p = 0.2

    blocks = 8
    encoder = VitEncoder(
        num_blocks=blocks,
        head_num=head_num,
        embed_dim=embed_dim,
        mpl_dim=mpl_dim,
        dropout_p=dropout_p
    )
    
    print('encoder', encoder)
    p_numel_list = [ p.numel() for p in encoder.parameters() ]
    print("sum(p_numel_list): {}, p_numel_list: {}".format(sum(p_numel_list), p_numel_list))

    input = torch.randn(N, C, D)

    output = encoder(input)

    print('output.shape', output.shape)

if __name__ == '__main__':
    test_encoder()
