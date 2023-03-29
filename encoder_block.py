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
from encoder_block_mpl import EncoderBlockMpl


class EncoderBlock(nn.Module):

    def __init__(self, head_num: int, embed_dim: int, mpl_dim: int):
        super(EncoderBlock, self).__init__()

        self.head_num: int = head_num
        self.embed_dim: int = embed_dim
        self.mpl_dim: int = mpl_dim

        # Layer Normalization 1
        self.layer_norm_1: nn.LayerNorm = nn.LayerNorm(self.embed_dim)

        # MultiHeadSelfAttention
        self.multi_head_attention: MultiHeadAttention = \
            MultiHeadAttention(
            self.head_num, 
            self.embed_dim, 
            self.embed_dim, 
            divide_input_dim=True
        )

        # Layer Normalization 2
        self.layer_norm_2: nn.LayerNorm = nn.LayerNorm(self.embed_dim)

        # MPL
        self.mpl: EncoderBlockMpl = EncoderBlockMpl(
            input_dim=self.embed_dim,
            hidden_dim=self.mpl_dim,
            output_dim=self.embed_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (N, T, D)

        z1: torch.Tensor = self.layer_norm_1(x)

