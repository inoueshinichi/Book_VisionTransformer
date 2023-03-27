"""Class Token Embedding Vector
Input: (B, C, H, W) = (B, C*H*W=F)
Output: (B, 1, D) = Input(B, F) @ Class_Token_W(F, D)
"""