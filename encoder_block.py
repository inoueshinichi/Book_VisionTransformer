"""TransformerのEncoderBlock
Input: (N, P+1, D)
Output: (N, P+1, H)

step1. Attention: MultiHead-Attention with Self-Attention.

次元Dから次元Hへの埋め込み(Linear Embedding)
Query: (N, P+1, H) = Input(N*P+1, D) @ Query_W(D, H)
Key  : (N, P+1, H) = Input(N*P+1, D) @ Key_W(D, H)
Value: (N, P+1, H) = Inpu(N*P+1, D) @ Value_W(D, H)
※Query_W, Key_W, Value_Wの各重み行列のサイズは同じ
--------------------------------------------------------------------------
Attention Weight: (N, P+1, P+1) = softmax_with_rows( Query(N*P+1, H) @ Key^T(H, N*P+1) ) / √H
※データ要素(H次元を持つベクトル)同士の類似度を計算しているだけ.
※√Hで割るので `Scale Dot Product Self-Attention`とも呼ぶ. 
※埋め込みベクトルの次元数Hが大きくなるとAttention Weightの1要素(類似度)の値が大きくなりすぎてしまうので, 次元数Hに依存した数値で除算している.
(P+1,P+1)の相互相関行列, ただし, 行方向の和=1 (softmaxを適用しているため)
--------------------------------------------------------------------------
Output: (N)


step2. mpl
"""