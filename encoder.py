"""Encoder Module of Transformer
Input: (N, P+1, D); Class token + PatchPositional Embedding tokens
Output: (N, 1, H); Class token vector to `Classify Mpl Head`

Sequencial(EncoderBlock1, EncoderBlock2, ..., EncoderBlock#)
"""