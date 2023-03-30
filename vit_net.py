""" ViT(Vision Transformer) Network
Input: (N, C, H, W)
Output: (N, num_class)
"""

from typing import *
from typing_extensions import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_input_layer import VitInputLayer
from vit_encoder import VitEncoder
from vit_classify_head_mpl import VitHeadMpl

class VitNet(nn.Module):

    def __init__(self, 
                 num_class: int, 
                 in_channels: int, 
                 embed_dim: int, 
                 image_size: Tuple[int, int],
                 num_patch: Tuple[int, int],
                 num_encoder_block: int,
                 num_head: int,
                 mpl_dim: int,
                 dropout_p: float):
        
        super(VitNet, self).__init__()

        # Input Layer
        self.input_layer: VitInputLayer = VitInputLayer(
            in_channels=in_channels,
            embed_dim=embed_dim,
            image_size=image_size,
            num_patch=num_patch
        )

        # Encoder
        self.encoder: VitEncoder = VitEncoder(
            num_blocks=num_encoder_block,
            num_head=num_head,
            embed_dim=embed_dim,
            mpl_dim=mpl_dim,
            dropout_p=dropout_p
        )

        # MPL Head
        self.head_mpl: VitHeadMpl = VitHeadMpl(embed_dim=embed_dim, num_class=num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (N, C, H, W)
        # score : (N, num_class)

        input_encoder = self.input_layer(x) # (N, T, D)
        output_encoder = self.encoder(input_encoder) # (N, T, D)
        class_token = output_encoder[:, 0, :] # (N, T, D) -> (N, D)
        score = self.head_mpl(class_token)
        return score

def test_vit_net():

    num_class = 10
    N = 2
    C = 3
    H = 32
    W = 32

    input = torch.randn(N, C, H, W)

    vit_net = VitNet(
        num_class=num_class,
        in_channels=C,
        embed_dim=384,
        image_size=(H, W),
        num_patch=(4, 4),
        num_encoder_block=7,
        num_head=8,
        mpl_dim=2048,
        dropout_p=0.2)

    score = vit_net(input)
    
    print('score.shape', score.shape)
    print('score: \n', score)

if __name__ == '__main__':
    test_vit_net()
