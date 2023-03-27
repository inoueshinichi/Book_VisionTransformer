"""画像から[size,size]への複数パッチへの変換
Input: (N, C, H, W)
Output: (N, P, L) # Pはパッチ数
L = H * W / P # パッチベクトル
"""

