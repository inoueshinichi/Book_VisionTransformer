"""Encoderへの入力を整えるモジュール
Input_1 (Patch+Positional Embed Tokens): (N, P, D)
Input_2 (Class Token): (N, 1, D)
Output: (N, P+1, D)

Images ---> patch embedding tokens -------->+ ----->+----> (Encoder) 
        |                                   |       |
        --> positional embedding tokens -----       |
        |                                           |
        --> class token -----------------------------
"""