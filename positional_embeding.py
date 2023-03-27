"""Positional Embedding
Input: (N, P, 2?); calculated from patch position in Image.
Output: (N, P, D); same dimension as patch embedding token
Output = Input(N, P, 2) @ Positional_Embedding_W(2, D)
"""

