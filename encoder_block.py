"""EncoderBlock of Transformer
Input: (N, P+1, D)
Output: (N, P+1, H)

step1. MultiHead-Attention.
step2. MPL
"""

from typing import *
from typing_extensions import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_head_attention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, head_num: int):
        super().__init__()

        self.embed_dim: int = embed_dim
        self.head_num: int = head_num

        self.multi_head_attention = MultiHeadAttention(self.head_num, self.embed_dim, self.embed_dim)