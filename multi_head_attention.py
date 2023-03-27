"""MultiHead-Attention with Self-Attention.

# MultiHead-Attention
--------------------------------------------------------------------------
Input: (B, P+1, D)
Output: (B, P+1, H)
Head数(k) x Self-Attention [Query_W(D,H), Key_W(D,H), Value_(D,H), Attention_Weight(B,P+1,P+1)]
OneHead_W(kH, H): 各Self-Attentionの結合(Concatenate)出力(B, P+1, kH)を(B, P+1, H)に変換する線形演算子.
※Head数(k)はハイパーパラメータ. 
"""

from typing import *
from typing_extensions import *

import torch
from torch import nn
from torch.nn.functional import F

from self_attention import SelfAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, head_num: int, input_dim: int, output_dim: int):
        super().__init__()

        self.head_num: int = head_num
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim

        assert (self.input_dim % self.head_num) == 0
        self.ebmed_dim: int = input_dim // self.head_num

        self.attentions: nn.ModuleList = [ SelfAttention(input_dim=self.embed_dim, embed_dim=self.embed_dim) for _ in range(self.head_num) ]

        self.aggregation_w: nn.Linear = nn.Linear(self.input_dim, self.output_dim, bias=False)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass