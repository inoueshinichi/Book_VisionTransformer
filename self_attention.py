"""Self-Attention
--------------------------------------------------------------------------
次元Dから次元Hへの埋め込み(Linear Embedding)
Query: (B, P+1, H) = Input(N*P+1, D) @ Query_W(D, H)
Key  : (B, P+1, H) = Input(N*P+1, D) @ Key_W(D, H)
Value: (B, P+1, H) = Inpu(N*P+1, D) @ Value_W(D, H)
※Query_W, Key_W, Value_Wの各重み行列のサイズは同じ

Attention_Weight: (B, P+1, P+1) = softmax_with_rows( Query(B*P+1, H) @ Key^T(H, B*P+1) ) / √H
※データ要素(H次元を持つベクトル)同士の類似度を計算しているだけ.
※√Hで割るので `Scale Dot Product Self-Attention`とも呼ぶ. 
※埋め込みベクトルの次元数Hが大きくなるとAttention Weightの1要素(類似度)の値が大きくなりすぎてしまうので, 
次元数Hに依存した数値で除算している.
(P+1,P+1)の相互相関行列, ただし, 行方向の和=1 (softmaxを適用しているため)
--------------------------------------------------------------------------
"""
from typing import *
from typing_extensions import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()

        self.input_dim: int = input_dim
        self.embed_dim: int = embed_dim
        self.dim_sqrt: float = embed_dim ** 0.5

        self.query_w: nn.Linear = nn.Linear(self.input_dim, self.embed_dim, bias=False)
        self.key_w: nn.Linear = nn.Linear(self.input_dim, self.embed_dim, bias=False)
        self.value_w: nn.Linear = nn.Linear(self.input_dim, self.embed_dim, bias=False)
        self.attention_weight: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query_w(x) # (B,N,H) N=P+1 トークン数
        key = self.key_w(x) # (B,N,H)
        value = self.value_w(x) # (B,N,H)

        # 内積行列 (B,N,H) @ (B,H,N) = (B,N,N)
        key_t = key.transpose(1,2) # (B,H,N)
        ipm = query @ key_t

        # Attension Weight (B,N,N)
        self.attention_weight = F.softmax(ipm, dim=-1) # 最終次元方向にsoftmax
        self.attention_weight /= self.dim_sqrt

        # 出力 (B,N,N) @ (B,N,H) = (B,N,H)
        out = self.attention_weight @ value

        return out