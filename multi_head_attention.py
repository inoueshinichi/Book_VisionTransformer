"""MultiHead-Attention with Self-Attention.

# MultiHead-Attention
--------------------------------------------------------------------------
Input: (N, T, D)
Output: (N, T, H)
Head数(k) x Self-Attention [Query_W(D,H), Key_W(D,H), Value_(D,H), Attention_Weight(B,T,T)]
OneHead_W(kH, H): 各Self-Attentionの結合(Concatenate)出力(N, T, kH)を(N, T, H)に変換する線形演算子.
※Head数(k)はハイパーパラメータ. 
"""

from typing import *
from typing_extensions import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from self_attention import SelfAttention

class MultiHeadAttention(nn.Module):

    def __init__(self, series_dim: int, input_dim: int, output_dim: int, 
                 head_num: int, divide_input_dim: bool=True):
        super(MultiHeadAttention, self).__init__()

        self.series_dim = series_dim
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.head_num: int = head_num
        self.divide_input_dim: bool = divide_input_dim
        self.embed_in_dim: int = 0
        self.embed_out_dim: int = 0 

        # input_dimをhead_numで分割する場合
        if self.divide_input_dim:
            fmod: float = self.input_dim % self.head_num
            assert (int)(fmod) == 0, \
                "No zero about mod(self.input_dim, self.head_num). Given is {%f}".format(fmod)
            self.embed_in_dim = self.input_dim // self.head_num
        else:
            self.embed_in_dim = self.input_dim
        
        # self-attentionの出力は入力と同じにする
        self.embed_out_dim = self.embed_in_dim

        # self-attentions
        self.self_attention_list: nn.ModuleList = nn.ModuleList([ 
            SelfAttention(series_dim=self.series_dim, 
                          input_dim=self.embed_in_dim, 
                          output_dim=self.embed_out_dim) for _ in range(self.head_num) 
            ])

        # aggregation linear weight
        self.aggregation_w: nn.Linear = nn.Linear(self.head_num * self.embed_out_dim, self.output_dim, bias=False)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # divide for last dimension
        px_list: Optional[List[torch.Tensor]] = None
        if self.divide_input_dim:
            px_list = torch.chunk(x, self.head_num, dim=-1) # 最終次元をヘッド数で分割
        else:
            px_list = [ x.clone() for _ in range(self.head_num) ]

        # multi self-attentions
        out_list = []
        for px, self_attention in zip(px_list, self.self_attention_list):
            pout = self_attention(px)
            out_list.append(pout)

        # concatenate
        out = torch.cat(out_list, dim=-1)

        # aggregate
        out = self.aggregation_w(out)

        return out


def test_multi_head_attention():

    batch_size = 4
    series_dim = 5
    input_dim = 18
    input = torch.randn([batch_size, series_dim, input_dim], dtype=torch.float32)
    print("input.shape: ", input.size())
    print("input.requires_grad: ", input.requires_grad)
    
    output_dim = 18
    multi_head_attention = MultiHeadAttention(series_dim=series_dim, 
                                              input_dim=input_dim, 
                                              output_dim=output_dim, 
                                              head_num=3,
                                              divide_input_dim=True)
    
    print("multi head attention: ", multi_head_attention)

    output = multi_head_attention(input)
    print("output.shape: ", output.shape)
    print("output: \n", output)

    # in_1 = torch.Tensor([[1,2],[3,4]]).unsqueeze(0)
    # print("in_1: \n", in_1)
    # in_2 = torch.Tensor([[5,6],[7,8]]).unsqueeze(0)
    # print("in_2: \n", in_2)
    # concat = torch.cat([in_1, in_2], dim=-1)
    # print("concat: \n", concat)

    # mhsa = MultiHeadAttention(series_dim=2,
    #                           input_dim=4,
    #                           output_dim=4*3,
    #                           head_num=3,
    #                           divide_input_dim=False)
    # mhsa_out = mhsa(concat)
    # print("mhsa_out.shape: ", mhsa_out.shape)
    # print("mhsa_out: \n", mhsa_out)

if __name__ == "__main__":
    test_multi_head_attention()