"""TransformerのEncoderモジュール
Input: (N, P+1, D); Class token + PatchPositional Embedding tokens
Output: (N, 1, D); Class token vector to `Classify Mpl Head`
Encoderモジュールの構成 Sequencial(EncoderBlock1, EncoderBlock2, ..., EncoderBlock#)
"""