"""EncoderBlock of Transformer
Input: (N, T, D)
Output: (N, T, H=`T`)

MHSA: MultiHeadSelfAttention
MPL: Linear -> GELU -> Dropout -> Linear -> Dropout

Input ---> LayerNorm ---> MHSA ---> Dropout ---> (+) ---> LayerNorm ---> MPL ---> (+) ---> Output
       |                                          |   |                            |
       ------------- Skip Connection --------------   -------- Skip Connection -----

"""

from typing import *
from typing_extensions import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_head_attention import MultiHeadAttention
from vit_encoder_block_mpl import VitEncoderBlockMpl


class VitEncoderBlock(nn.Module):

    def __init__(self, num_head: int, embed_dim: int, mpl_dim: int, dropout_p: float):
        super(VitEncoderBlock, self).__init__()

        self.num_head: int = num_head
        self.embed_dim: int = embed_dim
        self.mpl_dim: int = mpl_dim
        self.dropout_p: float = dropout_p

        # Layer Normalization 1
        self.layer_norm_1: nn.LayerNorm = nn.LayerNorm(normalized_shape=self.embed_dim)

        # MultiHeadSelfAttention
        self.multi_head_attention: MultiHeadAttention = MultiHeadAttention(
            num_head=self.num_head, 
            input_dim=self.embed_dim, 
            output_dim=self.embed_dim, 
            divide_input_dim=True
        )

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout_p)

        # Layer Normalization 2
        self.layer_norm_2: nn.LayerNorm = nn.LayerNorm(normalized_shape=self.embed_dim)

        # MPL
        self.mpl: VitEncoderBlockMpl = VitEncoderBlockMpl(
            input_dim=self.embed_dim,
            hidden_dim=self.mpl_dim,
            output_dim=self.embed_dim,
            dropout_p=self.dropout_p
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (N, T, D)

        # With Skip Connection
        z0 = self.layer_norm_1(x)
        z1 = self.multi_head_attention(z0)
        z2 = self.dropout_layer(z1)
        z3 = z2 + x

        # With Skip Connection
        out = self.mpl(self.layer_norm_2(z3)) + z3

        return out
    

def test_encoder_block():
    N = 2
    C = 3
    head_num = 3
    D = 48 * head_num

    embed_dim = D
    mpl_dim = 512
    dropout_p = 0.2

    encoder_block = VitEncoderBlock(
        head_num=head_num,
        embed_dim=embed_dim,
        mpl_dim=mpl_dim,
        dropout_p=dropout_p
    )

    print('encoder_block', encoder_block)
    p_numel_list = [ p.numel() for p in encoder_block.parameters() ]
    print("sum(p_numel_list): {}, p_numel_list: {}".format(sum(p_numel_list), p_numel_list))

    input = torch.randn(N, C, D)

    output = encoder_block(input)

    print('output.shape', output.shape)

    print('Attention Weights \n')
    for i, attn in enumerate(encoder_block.multi_head_attention.self_attention_list):
        print(f'{i}th attention weight shape', attn.attention_weight.shape) 
        print(f'{i}th attension weight: \n', attn.attention_weight)  


if __name__ == '__main__':
    test_encoder_block()
