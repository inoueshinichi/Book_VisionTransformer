"""パッチベクトルを埋め込みベクトルに変換
Input: (N, P, L)
Output: (N, P, D)
Output = Input(N, P, L) @ Embedding_W(L, D)
"""