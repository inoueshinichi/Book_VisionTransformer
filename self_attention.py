"""Self-Attention
--------------------------------------------------------------------------
N: バッチサイズ
T: 系列長(トークン数)
D: トークンの次元数
H: 埋め込みベクトルの次元数

次元Dから次元Hへの埋め込み(Linear Embedding)
Query: (N, T, H) = Input(N*T, D) @ Query_W(D, H)
Key  : (N, T, H) = Input(N*T, D) @ Key_W(D, H)
Value: (N, T, H) = Inpu(N*T, D) @ Value_W(D, H)
※Query_W, Key_W, Value_Wの各重み行列のサイズは同じ

Attention_Weight: (N, T, T) = softmax_with_rows( Query(N*T, H) @ Key^t(H, N*T) ) / √H
※データ要素(H次元を持つベクトル)同士の類似度を計算しているだけ.
※√Hで割るので `Scale Dot Product Self-Attention`とも呼ぶ. 
※埋め込みベクトルの次元数Hが大きくなるとAttention Weightの1要素(類似度)の値が大きくなりすぎてしまうので, 
次元数Hに依存した数値で除算している.
(T,T)の相互相関行列, ただし, 行方向の和=1 (softmaxを適用しているため)
--------------------------------------------------------------------------
"""
from typing import *
from typing_extensions import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super(SelfAttention, self).__init__()

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.dim_sqrt: float = output_dim ** 0.5

        # Query, Key, Value
        self.query_w: nn.Linear = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.key_w: nn.Linear = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.value_w: nn.Linear = nn.Linear(self.input_dim, self.output_dim, bias=False)

        # Attention Weight
        self.attention_weight: Optional[torch.Tensor] = None
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query: torch.Tensor = self.query_w(x) # (N,T,H) T=(class_token + patch_token) トークン数
        key: torch.Tensor  = self.key_w(x) # (N,T,H)
        value: torch.Tensor = self.value_w(x) # (N,T,H)

        key_t: torch.Tensor = key.transpose(1,2) # (N,H,T)

        # Attension Weight (N,T,T) = (N,T,H) @ (N,H,T)
        self.attention_weight: torch.Tensor = F.softmax((query @ key_t).div_(self.dim_sqrt), dim=-1) # 最終次元方向にsoftmax
        # print("self.attention_weight.requres_grad", self.attention_weight.requires_grad)

        # 出力 (N,T,T) @ (N,T,H) = (N,T,H)
        out: torch.Tensor = self.attention_weight @ value # 加重和

        return out
    
def test_self_attention():

    batch_size = 3
    series_dim = 4
    input_dim = 5

    # Input (N=3, T=4, D=5)
    input = torch.randn([batch_size, series_dim, input_dim], dtype=torch.float32)
    print("input.shape: ", input.size())
    # print("input: ", input)

    # Self Attention
    output_dim = 8
    self_attention = SelfAttention(input_dim=input_dim, output_dim=output_dim)
    print("self-attention: ", self_attention)
    p_numel_list = [ p.numel() for p in self_attention.parameters() ]
    print("sum(p_numel_list): {}, p_numel_list: {}".format(sum(p_numel_list), p_numel_list))


    # Output
    output = self_attention(input)
    print("output.shape: ", output.size())
    # print("output: ", output)

    # Attention Weight
    print("attention weight.shape: ", self_attention.attention_weight.size())
    print("attention_weight.requres_grad", self_attention.attention_weight.requires_grad)
    print("attention weight: \n", self_attention.attention_weight)

    # Sum Attention Weight along last dimension
    print("Sum of attention weight along row: \n", self_attention.attention_weight.sum(dim=-1))

if __name__ == '__main__':
    test_self_attention()